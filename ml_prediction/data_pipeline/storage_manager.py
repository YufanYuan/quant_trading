"""
分片HDF5存储管理器

负责在数据预处理时直接写入多个分片文件，避免先生成大文件再切分的浪费流程
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ShardInfo:
    """分片信息"""
    file_handle: h5py.File
    dataset: h5py.Dataset
    counter: int  # 当前已写入的样本数


class ShardedHDF5Writer:
    """
    分片HDF5存储管理器

    在数据预处理时直接将数据分配到多个小文件，避免创建巨大的单一文件
    """

    def __init__(
        self,
        output_dir: str,
        num_shards: int = 64,
        batch_size: int = 512,
        chunk_size: int = 128,
        strategy: str = "round_robin",
        window_size: int = 120,
        feature_names: str = "",
        metadata: Optional[Dict] = None,
    ):
        """
        Args:
            output_dir: 输出目录
            num_shards: 分片数量
            batch_size: 对齐的batch大小（最终会裁剪到此值的整数倍）
            chunk_size: 分片切换间隔（每N个样本切换到下一个shard）
            strategy: 分片策略 ('round_robin' 为轮询分配)
            window_size: 窗口大小
            feature_names: 特征名称（逗号分隔）
            metadata: 额外的元数据
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        self.num_shards = num_shards
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.strategy = strategy
        self.window_size = window_size
        self.feature_names = feature_names
        self.metadata = metadata or {}

        self.shard_infos: List[ShardInfo] = []
        self.current_shard_idx = 0
        self.current_chunk_counter = 0  # 当前chunk中已写入的样本数

        self.sample_shape: Optional[Tuple] = None
        self.sample_dtype: Optional[np.dtype] = None

        self._initialized = False

    def _initialize_shards(self, sample_shape: Tuple, sample_dtype: np.dtype):
        """
        初始化分片文件（首次写入数据时调用）

        Args:
            sample_shape: 单个样本的shape (例如 (120, 11))
            sample_dtype: 数据类型
        """
        if self._initialized:
            return

        self.sample_shape = sample_shape
        self.sample_dtype = sample_dtype

        print(f"\n初始化存储管理器:")
        print(f"  输出目录: {self.output_dir}")
        print(f"  分片数量: {self.num_shards}")
        print(f"  Chunk大小: {self.chunk_size} (每{self.chunk_size}个样本切换shard)")
        print(f"  Batch对齐: {self.batch_size}")
        print(f"  样本shape: {sample_shape}")
        print(f"  样本dtype: {sample_dtype}")

        # 创建所有分片的临时文件
        for shard_id in range(self.num_shards):
            temp_file_path = self.temp_dir / f"temp_shard_{shard_id:04d}.h5"
            f_temp = h5py.File(temp_file_path, "w")

            # 写入元数据
            f_temp.attrs["window_size"] = self.window_size
            f_temp.attrs["feature_names"] = self.feature_names
            f_temp.attrs["shard_id"] = shard_id
            f_temp.attrs["num_shards"] = self.num_shards

            # 写入额外的元数据
            for key, value in self.metadata.items():
                f_temp.attrs[key] = value

            # 创建可变大小的dataset (maxshape=(None, ...) 表示第一维可扩展)
            ds_temp = f_temp.create_dataset(
                "data_temp",
                shape=(0, *sample_shape),  # 初始大小为0
                maxshape=(None, *sample_shape),  # 第一维可扩展
                dtype=sample_dtype,
                chunks=(min(1000, self.chunk_size), *sample_shape),
            )

            shard_info = ShardInfo(
                file_handle=f_temp,
                dataset=ds_temp,
                counter=0,
            )
            self.shard_infos.append(shard_info)

        self._initialized = True
        print(f"  ✓ 已创建{self.num_shards}个临时文件（动态增长模式）\n")

    def add_dataset(self, symbol: str, year: int, data: np.ndarray):
        """
        添加一个数据集（如BTCUSDT_2022）

        数据会被按照round-robin策略分配到不同的shard

        Args:
            symbol: 交易对符号
            year: 年份
            data: 处理好的数据，shape为 (N, window_size, num_features+1)
        """
        if len(data) == 0:
            print(f"  警告: {symbol}_{year} 数据为空，跳过")
            return

        # 首次写入时初始化
        if not self._initialized:
            self._initialize_shards(data[0].shape, data.dtype)

        # 验证数据shape
        if data[0].shape != self.sample_shape:
            raise ValueError(
                f"数据shape不匹配: 期望{self.sample_shape}, 实际{data[0].shape}"
            )

        print(f"  写入 {symbol}_{year}: {len(data):,} 样本")

        # 计算会保留多少数据，会丢弃多少
        num_complete_chunks = len(data) // self.chunk_size
        num_to_keep = num_complete_chunks * self.chunk_size
        num_to_drop = len(data) - num_to_keep

        if num_to_drop > 0:
            print(f"    注意: 尾部{num_to_drop}个样本不足{self.chunk_size}，将被丢弃")
            data = data[:num_to_keep]

        # 按chunk分配到不同shard
        for chunk_idx in range(num_complete_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = start_idx + self.chunk_size
            chunk_data = data[start_idx:end_idx]

            # 获取当前shard
            shard_info = self.shard_infos[self.current_shard_idx]

            # 扩展dataset并写入
            old_size = shard_info.counter
            new_size = old_size + self.chunk_size
            shard_info.dataset.resize((new_size, *shard_info.dataset.shape[1:]))
            shard_info.dataset[old_size:new_size] = chunk_data
            shard_info.counter = new_size

            # 轮询切换到下一个shard
            self.current_shard_idx = (self.current_shard_idx + 1) % self.num_shards

        actual_written = num_complete_chunks * self.chunk_size
        print(f"    ✓ 已写入{actual_written:,}个样本到分片")

    def finalize(self, shuffle: bool = True, random_seed: int = 42):
        """
        完成写入，进行最终处理

        - 对每个shard进行shuffle（可选）
        - 裁剪到batch_size的整数倍
        - 写入metadata
        - 关闭所有文件

        Args:
            shuffle: 是否对每个shard进行独立shuffle
            random_seed: shuffle随机种子
        """
        if not self._initialized:
            print("警告: 没有数据被写入，跳过finalize")
            return

        print(f"\n最终处理分片文件...")
        print(f"  Shuffle: {'是' if shuffle else '否'}")
        print(f"  对齐到: {self.batch_size}的整数倍")

        # 先关闭所有临时文件
        for shard_info in self.shard_infos:
            shard_info.file_handle.close()

        rng = np.random.default_rng(random_seed) if shuffle else None
        final_counters = []
        total_original = sum(s.counter for s in self.shard_infos)
        total_dropped = 0

        # 处理每个临时文件
        for shard_id in tqdm(range(self.num_shards), desc="  处理进度"):
            temp_file_path = self.temp_dir / f"temp_shard_{shard_id:04d}.h5"
            final_file_path = self.output_dir / f"shard_{shard_id:04d}.h5"

            shard_counter = self.shard_infos[shard_id].counter

            if shard_counter == 0:
                print(f"    警告: shard_{shard_id:04d} 为空，跳过")
                temp_file_path.unlink()  # 删除空文件
                continue

            # 计算裁剪后的大小
            aligned_size = (shard_counter // self.batch_size) * self.batch_size

            if aligned_size == 0:
                print(f"    警告: shard_{shard_id:04d} 数据不足{self.batch_size}条，跳过")
                temp_file_path.unlink()
                continue

            dropped = shard_counter - aligned_size
            total_dropped += dropped

            # 读取、处理、写入
            with h5py.File(temp_file_path, "r") as f_temp:
                # 读取数据（只读取对齐后的大小）
                data = f_temp["data_temp"][:aligned_size]

                # Shuffle (可选)
                if shuffle:
                    indices = np.arange(aligned_size)
                    rng.shuffle(indices)
                    data = data[indices]

                # 写入最终文件
                with h5py.File(final_file_path, "w") as f_out:
                    # 复制metadata
                    for key, value in f_temp.attrs.items():
                        f_out.attrs[key] = value

                    # 添加额外的metadata
                    f_out.attrs["total_samples_in_shard"] = aligned_size
                    f_out.attrs["original_size_before_align"] = shard_counter
                    f_out.attrs["align_to"] = self.batch_size

                    # 写入数据
                    f_out.create_dataset(
                        "data",
                        data=data,
                        chunks=(min(1000, aligned_size), *self.sample_shape),
                        compression="gzip",
                        compression_opts=4,
                    )

            final_counters.append(aligned_size)

            # 删除临时文件
            temp_file_path.unlink()

        # 删除临时目录
        try:
            self.temp_dir.rmdir()
        except:
            print("  注意: 临时目录可能不为空，请手动检查")

        # 统计信息
        total_final = sum(final_counters)

        print(f"\n  ✓ 完成")
        print(f"  原始样本数: {total_original:,}")
        print(f"  最终样本数: {total_final:,}")
        print(f"  对齐裁剪损失: {total_dropped:,} ({total_dropped/total_original*100:.2f}%)")
        if final_counters:
            print(f"  每个shard样本数: 最小={min(final_counters):,}, 最大={max(final_counters):,}, 平均={total_final/len(final_counters):.1f}")
        print(f"  有效shard数: {len(final_counters)}/{self.num_shards}")

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """自动调用finalize"""
        if self._initialized:
            self.finalize()


def verify_shards(output_dir: str):
    """
    验证分片文件的完整性

    Args:
        output_dir: 分片文件目录
    """
    output_path = Path(output_dir)
    shard_files = sorted(output_path.glob("shard_*.h5"))

    if not shard_files:
        print(f"未找到分片文件: {output_dir}")
        return

    print(f"\n验证分片文件: {output_dir}")
    print(f"找到{len(shard_files)}个文件")

    total_samples = 0
    align_check = []

    for shard_file in shard_files:
        with h5py.File(shard_file, "r") as f:
            num_samples = len(f["data"])
            total_samples += num_samples
            align_to = f.attrs.get("align_to", 512)

            # 检查是否对齐
            is_aligned = (num_samples % align_to == 0)
            align_check.append(is_aligned)

            # 只打印首尾文件
            if len(align_check) <= 2 or len(align_check) == len(shard_files):
                print(f"  ✓ {shard_file.name}: {num_samples:,} 样本, 对齐={is_aligned}")

    print(f"\n  总计: {total_samples:,} 样本")
    print(f"  对齐检查: {sum(align_check)}/{len(align_check)} 个文件已对齐")
    print("  ✓ 验证完成")
