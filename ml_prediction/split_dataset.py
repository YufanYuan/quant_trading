"""
将大型数据集分割成多个小文件，便于使用memmap加载

这个脚本会将现有的train_features.npy和train_labels.npy分割成多个shard文件
"""

import numpy as np
from pathlib import Path
from typing import Tuple


def split_dataset(
    features_path: Path,
    labels_path: Path,
    output_dir: Path,
    samples_per_shard: int = 10000,
    shuffle: bool = True,
    random_seed: int = 42,
    convert_dtype: bool = False,
    features_dtype: np.dtype = np.float32,
    labels_dtype: np.dtype = np.int32
) -> int:
    """
    将数据集分割成多个shard文件

    Args:
        features_path: 原始features文件路径
        labels_path: 原始labels文件路径
        output_dir: 输出目录
        samples_per_shard: 每个shard包含的样本数
        shuffle: 是否在分片前shuffle数据（推荐True）
        random_seed: 随机种子
        convert_dtype: 是否在分片时转换数据类型
        features_dtype: features目标数据类型（当convert_dtype=True时）
        labels_dtype: labels目标数据类型（当convert_dtype=True时）

    Returns:
        shard数量
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用mmap读取原始数据（不加载到内存）
    print("正在读取数据集元信息...")
    features = np.load(features_path, mmap_mode='r')
    labels = np.load(labels_path, mmap_mode='r')

    total_samples = features.shape[0]
    num_shards = (total_samples + samples_per_shard - 1) // samples_per_shard

    print(f"总样本数: {total_samples:,}")
    print(f"每个shard样本数: {samples_per_shard:,}")
    print(f"将分割为 {num_shards} 个shard文件")
    print(f"Features shape: {features.shape}, dtype: {features.dtype}")
    print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"Shuffle: {shuffle}")

    if convert_dtype:
        print(f"数据类型转换:")
        print(f"  Features: {features.dtype} -> {features_dtype}")
        print(f"  Labels: {labels.dtype} -> {labels_dtype}")

        # 计算节省的空间
        old_size = total_samples * np.prod(features.shape[1:]) * features.dtype.itemsize
        new_size = total_samples * np.prod(features.shape[1:]) * np.dtype(features_dtype).itemsize
        saved_gb = (old_size - new_size) / (1024**3)
        print(f"  预计节省空间: {saved_gb:.2f} GB")

    # 创建索引数组（shuffle或顺序）
    if shuffle:
        print(f"\n创建shuffled索引数组（随机种子: {random_seed}）...")
        np.random.seed(random_seed)
        indices = np.random.permutation(total_samples)
        print(f"索引数组大小: {indices.nbytes / 1024 / 1024:.2f} MB")
    else:
        print("\n使用顺序索引...")
        indices = np.arange(total_samples)

    # 分割数据
    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = min((shard_idx + 1) * samples_per_shard, total_samples)
        shard_size = end_idx - start_idx

        print(f"\n处理 shard {shard_idx+1}/{num_shards} ({shard_size:,} 样本)...")

        # 获取当前shard的索引
        shard_indices = indices[start_idx:end_idx]

        # 按照shuffled索引读取数据
        # 为了提高效率，我们分批读取
        batch_size = 10000
        shard_features_list = []
        shard_labels_list = []

        for i in range(0, shard_size, batch_size):
            batch_end = min(i + batch_size, shard_size)
            batch_indices = shard_indices[i:batch_end]

            # 从memmap中读取对应的数据
            batch_features = features[batch_indices]
            batch_labels = labels[batch_indices]

            shard_features_list.append(batch_features)
            shard_labels_list.append(batch_labels)

            if (i // batch_size) % 10 == 0:
                print(f"  进度: {i:,}/{shard_size:,} ({i/shard_size*100:.1f}%)")

        # 合并batch
        shard_features = np.concatenate(shard_features_list, axis=0)
        shard_labels = np.concatenate(shard_labels_list, axis=0)

        # 类型转换（如果需要）
        if convert_dtype:
            if shard_features.dtype != features_dtype:
                shard_features = shard_features.astype(features_dtype)
            if shard_labels.dtype != labels_dtype:
                shard_labels = shard_labels.astype(labels_dtype)

        # 保存shard
        features_file = output_dir / f"features_shard_{shard_idx:03d}.npy"
        labels_file = output_dir / f"labels_shard_{shard_idx:03d}.npy"

        np.save(features_file, shard_features)
        np.save(labels_file, shard_labels)

        print(f"  已保存: {features_file.name} ({shard_features.shape})")
        print(f"  已保存: {labels_file.name} ({shard_labels.shape})")

    # 保存元数据
    final_features_dtype = features_dtype if convert_dtype else features.dtype
    final_labels_dtype = labels_dtype if convert_dtype else labels.dtype

    metadata = {
        'num_shards': num_shards,
        'total_samples': total_samples,
        'samples_per_shard': samples_per_shard,
        'feature_shape': features.shape[1:],  # (120, 10)
        'feature_dtype': str(final_features_dtype),
        'label_dtype': str(final_labels_dtype),
        'shuffled': shuffle,
        'random_seed': random_seed if shuffle else None,
        'converted': convert_dtype,
        'original_feature_dtype': str(features.dtype) if convert_dtype else None,
        'original_label_dtype': str(labels.dtype) if convert_dtype else None
    }

    metadata_file = output_dir / 'metadata.npy'
    np.save(metadata_file, metadata, allow_pickle=True)
    print(f"\n元数据已保存到: {metadata_file}")

    return num_shards


def convert_existing_dataset_to_float32(
    features_path: Path,
    labels_path: Path,
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    将现有的float64数据集转换为float32并分片

    Args:
        features_path: 原始features文件路径
        labels_path: 原始labels文件路径
        output_dir: 输出目录

    Returns:
        (新features路径, 新labels路径)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("正在转换数据类型...")
    print("警告: 这将创建新的文件，需要额外的磁盘空间")

    # 使用mmap读取
    features = np.load(features_path, mmap_mode='r')
    labels = np.load(labels_path, mmap_mode='r')

    print(f"原始 features dtype: {features.dtype}, shape: {features.shape}")
    print(f"原始 labels dtype: {labels.dtype}, shape: {labels.shape}")

    # 分批转换并保存，避免内存溢出
    batch_size = 100000
    total_samples = features.shape[0]

    new_features_path = output_dir / 'train_features_float32.npy'
    new_labels_path = output_dir / 'train_labels_int32.npy'

    # 创建新的memmap文件
    new_features = np.lib.format.open_memmap(
        new_features_path,
        mode='w+',
        dtype=np.float32,
        shape=features.shape
    )
    new_labels = np.lib.format.open_memmap(
        new_labels_path,
        mode='w+',
        dtype=np.int32,
        shape=labels.shape
    )

    # 分批复制并转换
    num_batches = (total_samples + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)

        print(f"处理批次 {batch_idx+1}/{num_batches} ({start_idx:,} - {end_idx:,})...")

        new_features[start_idx:end_idx] = features[start_idx:end_idx].astype(np.float32)
        new_labels[start_idx:end_idx] = labels[start_idx:end_idx].astype(np.int32)

    print(f"\n转换完成!")
    print(f"新 features: {new_features_path} ({new_features.dtype})")
    print(f"新 labels: {new_labels_path} ({new_labels.dtype})")

    # 计算文件大小
    old_size = features_path.stat().st_size / (1024**3)
    new_size = new_features_path.stat().st_size / (1024**3)
    print(f"Features 文件大小: {old_size:.2f}GB -> {new_size:.2f}GB (节省 {old_size-new_size:.2f}GB)")

    return new_features_path, new_labels_path


def main():
    """主函数"""
    data_dir = Path('./data')

    # 原始文件路径
    features_path = data_dir / 'train_features.npy'
    labels_path = data_dir / 'train_labels.npy'

    if not features_path.exists() or not labels_path.exists():
        print("错误: 找不到训练数据文件")
        return

    print("=" * 60)
    print("数据集分片工具")
    print("=" * 60)

    print("\n请选择分片模式:")
    print("选项1: 分片时转换为float32/int32（推荐，节省空间）")
    print("选项2: 先转换整个数据集再分片（需要额外磁盘空间）")
    print("选项3: 直接分片原始数据集（不转换类型）")
    choice = input("请选择 (1/2/3): ").strip()

    if choice == '1':
        # 新方式：分片时直接转换类型（节省磁盘空间）
        output_dir = data_dir / 'shards_float32'
        print(f"\n输出目录: {output_dir}")
        print("将在分片时同时转换数据类型...")

        split_dataset(
            features_path,
            labels_path,
            output_dir,
            samples_per_shard=10000,
            shuffle=True,
            random_seed=42,
            convert_dtype=True,
            features_dtype=np.float32,
            labels_dtype=np.int32
        )

    elif choice == '2':
        # 旧方式：先转换整个数据集，再分片
        output_dir = data_dir / 'shards_float32'
        new_features_path, new_labels_path = convert_existing_dataset_to_float32(
            features_path, labels_path, output_dir
        )

        # 分片新的float32数据
        print("\n" + "=" * 60)
        print("开始分片float32数据...")
        print("=" * 60)
        split_dataset(
            new_features_path,
            new_labels_path,
            output_dir,
            samples_per_shard=10000,
            shuffle=True,
            random_seed=42,
            convert_dtype=False
        )

    else:
        # 直接分片原始数据，不转换类型
        output_dir = data_dir / 'shards'
        split_dataset(
            features_path,
            labels_path,
            output_dir,
            samples_per_shard=10000,
            shuffle=True,
            random_seed=42,
            convert_dtype=False
        )

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
