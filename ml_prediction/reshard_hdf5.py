"""
HDF5 数据重组脚本

将单个大HDF5文件重组为多个shuffle过的小文件
优化策略：
1. 顺序读取原文件，每128条换一个shard写入
2. 如果当前shard满了，跳到下一个未满的shard
3. 对每个小文件单独shuffle
4. 裁剪到512的整数倍
"""

import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class ShardInfo:
    """分片信息"""
    file_handle: h5py.File
    dataset: h5py.Dataset
    counter: int  # 当前已写入的样本数


def scan_input_file(
    input_path: Path,
    dataset_names: List[str] = None,
) -> Tuple[Dict, List[str], int, Tuple, np.dtype]:
    """
    扫描输入文件，获取基本信息

    Returns:
        metadata: 元数据字典
        dataset_names: dataset名称列表
        total_samples: 总样本数
        sample_shape: 单个样本的shape
        sample_dtype: 数据类型
    """
    print("\n[1/4] 扫描输入文件...")

    with h5py.File(input_path, "r") as f:
        # 获取dataset列表
        if dataset_names is None:
            dataset_names = list(f.keys())
        else:
            available = set(f.keys())
            missing = set(dataset_names) - available
            if missing:
                raise ValueError(f"以下datasets不存在: {missing}")

        # 保存metadata
        metadata = dict(f.attrs)

        # 获取sample shape和dtype
        first_dataset = f[dataset_names[0]]
        sample_shape = first_dataset[0].shape
        sample_dtype = first_dataset[0].dtype

        # 统计总样本数
        total_samples = 0
        for name in dataset_names:
            dataset_len = len(f[name])
            print(f"  Dataset '{name}': {dataset_len:,} 样本")
            total_samples += dataset_len

    print(f"  总样本数: {total_samples:,}")
    print(f"  样本shape: {sample_shape}, dtype: {sample_dtype}")

    return metadata, dataset_names, total_samples, sample_shape, sample_dtype


def create_temp_files(
    temp_dir: Path,
    num_shards: int,
    metadata: Dict,
    sample_shape: Tuple,
    sample_dtype: np.dtype,
) -> List[ShardInfo]:
    """
    创建临时分片文件（使用动态增长的dataset）

    Returns:
        shard_infos: ShardInfo列表
    """
    print("\n[2/4] 创建临时文件...")

    shard_infos = []

    for shard_id in range(num_shards):
        temp_file_path = temp_dir / f"temp_shard_{shard_id:04d}.h5"
        f_temp = h5py.File(temp_file_path, "w")

        # 复制metadata
        for key, value in metadata.items():
            f_temp.attrs[key] = value
        f_temp.attrs["shard_id"] = shard_id
        f_temp.attrs["num_shards"] = num_shards

        # 创建可变大小的dataset（maxshape=(None, ...)表示第一维可扩展）
        ds_temp = f_temp.create_dataset(
            "data_temp",
            shape=(0, *sample_shape),  # 初始大小为0
            maxshape=(None, *sample_shape),  # 第一维可扩展
            dtype=sample_dtype,
            chunks=(min(1000, 128), *sample_shape),
        )

        shard_info = ShardInfo(
            file_handle=f_temp,
            dataset=ds_temp,
            counter=0,
        )
        shard_infos.append(shard_info)

    print(f"  ✓ 创建{num_shards}个临时文件（动态增长模式）")

    return shard_infos


def distribute_data(
    input_path: Path,
    dataset_names: List[str],
    shard_infos: List[ShardInfo],
    chunk_size: int = 128,
) -> int:
    """
    顺序读取数据并按策略分配到各个shard

    策略：
    1. 每chunk_size条数据换一个shard
    2. 如果dataset末尾不足chunk_size条，丢弃这部分数据
    3. 下一个dataset从上一个dataset结束的shard继续
    4. 批量读取和写入，提升性能

    Args:
        input_path: 输入文件路径
        dataset_names: dataset名称列表
        shard_infos: 分片信息列表
        chunk_size: 多少条数据后换shard

    Returns:
        total_dropped: 总共丢弃的样本数
    """
    print("\n[3/4] 顺序读取并分配数据...")
    print(f"  策略: 每{chunk_size}条换shard，批量读写提升性能")

    num_shards = len(shard_infos)
    current_shard = 0
    total_dropped = 0

    with h5py.File(input_path, "r") as f_in:
        for dataset_idx, dataset_name in enumerate(dataset_names):
            dataset = f_in[dataset_name]
            dataset_len = len(dataset)

            # 计算这个dataset会处理多少数据，会丢弃多少
            num_complete_chunks = dataset_len // chunk_size
            num_to_drop = dataset_len - num_complete_chunks * chunk_size
            total_dropped += num_to_drop

            print(f"\n  [{dataset_idx+1}/{len(dataset_names)}] 处理 '{dataset_name}': {dataset_len:,} 样本")
            print(f"      完整chunks: {num_complete_chunks}, 丢弃尾部: {num_to_drop} 样本")
            print(f"      起始shard: {current_shard}")

            # 按chunk批量处理
            with tqdm(total=num_complete_chunks, desc=f"      进度", unit="chunks") as pbar:
                for chunk_idx in range(num_complete_chunks):
                    # 批量读取chunk_size条数据
                    start_idx = chunk_idx * chunk_size
                    end_idx = start_idx + chunk_size
                    chunk_data = dataset[start_idx:end_idx]

                    # 获取当前shard
                    shard_info = shard_infos[current_shard]

                    # 一次性扩展dataset
                    old_size = shard_info.counter
                    new_size = old_size + chunk_size
                    shard_info.dataset.resize((new_size, *shard_info.dataset.shape[1:]))

                    # 批量写入
                    shard_info.dataset[old_size:new_size] = chunk_data
                    shard_info.counter = new_size

                    # 换到下一个shard
                    current_shard = (current_shard + 1) % num_shards

                    # 更新进度条
                    pbar.update(1)

            # 下一个dataset从当前shard继续（保持连续性）

    # 关闭所有临时文件
    for shard_info in shard_infos:
        shard_info.file_handle.close()

    # 统计信息
    counters = [s.counter for s in shard_infos]
    print(f"\n  ✓ 完成")
    print(f"  每个shard样本数: 最小={min(counters):,}, 最大={max(counters):,}, 平均={sum(counters)/len(counters):.1f}")
    print(f"  总共丢弃: {total_dropped:,} 样本")

    return total_dropped


def shuffle_and_finalize(
    temp_dir: Path,
    output_dir: Path,
    num_shards: int,
    shard_counters: List[int],
    sample_shape: Tuple,
    align_to: int = 512,
    random_seed: int = 42,
) -> List[int]:
    """
    对每个临时文件进行shuffle并保存为最终文件
    同时裁剪到align_to的整数倍

    Args:
        temp_dir: 临时文件目录
        output_dir: 输出目录
        num_shards: 分片数量
        shard_counters: 每个shard的实际样本数
        sample_shape: 样本shape
        align_to: 对齐到多少的整数倍
        random_seed: 随机种子

    Returns:
        final_counters: 每个shard最终的样本数（裁剪后）
    """
    print("\n[4/4] Shuffle并裁剪到512的整数倍...")

    rng = np.random.default_rng(random_seed)
    final_counters = []

    for shard_id in tqdm(range(num_shards), desc="  进度"):
        temp_file_path = temp_dir / f"temp_shard_{shard_id:04d}.h5"
        final_file_path = output_dir / f"shard_{shard_id:04d}.h5"

        # 读取临时文件
        with h5py.File(temp_file_path, "r") as f_temp:
            # 计算裁剪后的大小
            actual_size = shard_counters[shard_id]
            aligned_size = (actual_size // align_to) * align_to

            if aligned_size == 0:
                print(f"  警告: shard_{shard_id:04d} 数据不足{align_to}条，跳过")
                continue

            # 读取数据（只读取对齐后的大小）
            data = f_temp["data_temp"][:aligned_size]

            # Shuffle
            indices = np.arange(aligned_size)
            rng.shuffle(indices)
            shuffled_data = data[indices]

            # 写入最终文件
            with h5py.File(final_file_path, "w") as f_out:
                # 复制metadata
                for key, value in f_temp.attrs.items():
                    f_out.attrs[key] = value
                f_out.attrs["total_samples_in_shard"] = aligned_size
                f_out.attrs["original_size_before_align"] = actual_size
                f_out.attrs["align_to"] = align_to

                # 写入数据
                f_out.create_dataset(
                    "data",
                    data=shuffled_data,
                    chunks=(min(1000, aligned_size), *sample_shape),
                    compression="gzip",
                    compression_opts=4,
                )

            final_counters.append(aligned_size)

        # 删除临时文件
        temp_file_path.unlink()

    print(f"  ✓ 完成")
    print(f"  裁剪统计: 对齐到{align_to}的整数倍")
    if final_counters:
        print(f"  最终样本数: 最小={min(final_counters):,}, 最大={max(final_counters):,}, 平均={sum(final_counters)/len(final_counters):.1f}")

    return final_counters


def reshard_hdf5(
    input_h5_path: str,
    output_dir: str,
    num_shards: int = 64,
    chunk_size: int = 128,
    align_to: int = 512,
    dataset_names: List[str] = None,
    random_seed: int = 42,
):
    """
    将单个HDF5文件重组为多个shuffle过的分片文件

    Args:
        input_h5_path: 输入HDF5文件路径
        output_dir: 输出目录
        num_shards: 分片数量
        chunk_size: 每多少条数据换一个shard
        align_to: 最终裁剪到多少的整数倍
        dataset_names: 要处理的dataset名称列表（None=全部）
        random_seed: 随机种子
    """
    input_path = Path(input_h5_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    temp_dir = output_path / "temp"
    temp_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"HDF5数据重组: {input_path.name} -> {num_shards}个分片")
    print(f"分配策略: 每{chunk_size}条换shard | 对齐: {align_to}的整数倍")
    print("=" * 70)

    # 第一步：扫描输入文件
    metadata, dataset_names, total_samples, sample_shape, sample_dtype = scan_input_file(
        input_path, dataset_names
    )

    # 第二步：创建临时文件
    shard_infos = create_temp_files(
        temp_dir, num_shards, metadata, sample_shape, sample_dtype
    )

    # 第三步：分配数据
    total_dropped = distribute_data(input_path, dataset_names, shard_infos, chunk_size)

    # 第四步：shuffle并最终化
    shard_counters = [s.counter for s in shard_infos]
    final_counters = shuffle_and_finalize(
        temp_dir, output_path, num_shards, shard_counters, sample_shape, align_to, random_seed
    )

    # 删除临时目录
    try:
        temp_dir.rmdir()
    except:
        print("  注意: 临时目录可能不为空，请手动检查")

    print("\n" + "=" * 70)
    print("✓ 重组完成！")
    print(f"  输出目录: {output_path}")
    print(f"  文件数量: {len(final_counters)}")
    print(f"  原始样本数: {total_samples:,}")
    print(f"  分配后样本数: {sum(shard_counters):,}")
    print(f"  最终样本数: {sum(final_counters):,} (对齐后)")
    print(f"  chunk_size不足丢弃: {total_dropped:,} ({total_dropped / total_samples * 100:.2f}%)")
    print(f"  对齐裁剪损失: {sum(shard_counters) - sum(final_counters):,} ({(sum(shard_counters) - sum(final_counters)) / sum(shard_counters) * 100:.2f}%)")
    print(f"  总损失: {total_samples - sum(final_counters):,} ({(total_samples - sum(final_counters)) / total_samples * 100:.2f}%)")
    print("=" * 70)


def verify_shards(output_dir: str, num_shards: int):
    """验证分片文件的完整性"""
    output_path = Path(output_dir)

    print("\n验证分片文件...")
    total_samples = 0
    align_check = []

    for shard_id in range(num_shards):
        shard_file = output_path / f"shard_{shard_id:04d}.h5"
        if not shard_file.exists():
            print(f"  警告: shard_{shard_id:04d}.h5 不存在")
            continue

        with h5py.File(shard_file, "r") as f:
            num_samples = len(f["data"])
            total_samples += num_samples
            align_to = f.attrs.get("align_to", 512)

            # 检查是否对齐
            is_aligned = (num_samples % align_to == 0)
            align_check.append(is_aligned)

            if shard_id == 0:
                print(f"  ✓ shard_0000.h5: {num_samples:,} 样本, shape={f['data'].shape}, 对齐={is_aligned}")
            elif shard_id == num_shards - 1:
                print(f"  ✓ shard_{shard_id:04d}.h5: {num_samples:,} 样本, 对齐={is_aligned}")

    print(f"\n  总计: {total_samples:,} 样本")
    print(f"  对齐检查: {sum(align_check)}/{len(align_check)} 个文件已对齐")
    print("  ✓ 验证完成")


if __name__ == "__main__":
    # 配置
    INPUT_H5 = "./data/all_data.h5"
    OUTPUT_DIR = "./data/sharded"
    NUM_SHARDS = 64
    CHUNK_SIZE = 2048  # 每128条换一个shard
    ALIGN_TO = 512    # 裁剪到512的整数倍

    # 执行重组
    reshard_hdf5(
        input_h5_path=INPUT_H5,
        output_dir=OUTPUT_DIR,
        num_shards=NUM_SHARDS,
        chunk_size=CHUNK_SIZE,
        align_to=ALIGN_TO,
        random_seed=42,
    )

    # 验证
    verify_shards(OUTPUT_DIR, NUM_SHARDS)