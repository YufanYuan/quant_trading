"""
Sharded HDF5 DataLoader (Batch-Aligned版本)

优化策略：
- 数据以512为单位对齐
- 索引以batch为单位（而不是sample）
- 每个epoch后可以重新shuffle batch顺序
- 减少索引数量，提升效率
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings


class ShardedHDF5Dataset(Dataset):
    """
    支持多个HDF5分片文件的Dataset（batch对齐版本）

    特点：
    - 索引以512-batch为单位
    - 自动发现并加载所有分片文件
    - 独立的train/val dataset实例
    - 支持每个epoch重新shuffle
    """

    def __init__(
        self,
        shard_dir: str,
        shard_pattern: str = "shard_*.h5",
        batch_size: int = 512,
        mode: str = "train",
        batch_indices: Optional[np.ndarray] = None,
        batch_info: Optional[List[Tuple[str, int, int]]] = None,
        shuffle_on_epoch: bool = True,
        shuffle_seed: int = 42,
    ):
        """
        Args:
            shard_dir: 分片文件所在目录
            shard_pattern: 分片文件名模式（glob pattern）
            batch_size: batch大小（必须与数据对齐值一致）
            mode: 'train' 或 'val'
            batch_indices: batch索引数组（如果提供，直接使用）
            batch_info: batch信息列表（如果提供，直接使用）
            shuffle_on_epoch: 每个epoch后是否重新shuffle
            shuffle_seed: shuffle随机种子
        """
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle_on_epoch = shuffle_on_epoch
        self._epoch = 0
        self._shuffle_rng = np.random.default_rng(shuffle_seed)

        if not self.shard_dir.exists():
            raise FileNotFoundError(f"分片目录不存在: {self.shard_dir}")

        # 如果提供了batch_info和batch_indices，直接使用（子实例）
        if batch_info is not None and batch_indices is not None:
            self.batch_info = batch_info
            self.batch_indices = batch_indices
            self.total_batches = len(batch_indices)
            self.total_samples = len(batch_indices) * batch_size

            # 读取metadata（从第一个文件）
            first_shard_file = batch_info[0][0]
            with h5py.File(first_shard_file, "r") as f:
                self.window_size = f.attrs.get("window_size", 120)
                feature_names = f.attrs.get("feature_names", "")
                if feature_names:
                    self.num_features = len(feature_names.split(",")) - 1
                else:
                    self.num_features = f["data"].shape[-1] - 1

            # 初始shuffle
            if self.shuffle_on_epoch and self.mode == "train":
                self._shuffle_rng.shuffle(self.batch_indices)

            return

        # 否则，扫描文件并建立索引（主实例）
        shard_files = sorted(self.shard_dir.glob(shard_pattern))
        if not shard_files:
            raise FileNotFoundError(f"未找到分片文件: {self.shard_dir}/{shard_pattern}")

        # 扫描所有文件，建立batch索引
        # batch_info: [(shard_file_path, batch_start_idx_in_shard, num_samples_in_batch)]
        self.batch_info = []
        total_samples = 0

        for shard_file in shard_files:
            with h5py.File(shard_file, "r") as f:
                num_samples = len(f["data"])
                align_to = f.attrs.get("align_to", batch_size)

                # 验证对齐
                if num_samples % batch_size != 0:
                    raise ValueError(
                        f"文件 {shard_file.name} 的样本数 {num_samples} 不是 {batch_size} 的整数倍"
                    )

                # 计算这个shard有多少个batch
                num_batches = num_samples // batch_size

                # 记录每个batch的位置
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * batch_size
                    self.batch_info.append((str(shard_file), batch_start, batch_size))

                total_samples += num_samples

                # 读取metadata（从第一个文件）
                if len(self.batch_info) == num_batches:  # 第一个文件
                    self.window_size = f.attrs.get("window_size", 120)
                    feature_names = f.attrs.get("feature_names", "")
                    if feature_names:
                        self.num_features = len(feature_names.split(",")) - 1
                    else:
                        # 从数据shape推断
                        self.num_features = f["data"].shape[-1] - 1

        self.total_batches = len(self.batch_info)
        self.total_samples = total_samples

        # 注意：这是主实例，batch_indices包含所有batch
        # 稍后会用create_train_val_split创建独立的train/val实例
        self.batch_indices = np.arange(self.total_batches)

    def on_epoch_end(self):
        """
        Epoch结束时调用，重新shuffle batch顺序（仅用于train）

        使用示例：
            for epoch in range(num_epochs):
                for batch in train_loader:
                    train_step(batch)
                train_dataset.on_epoch_end()  # 重新shuffle
        """
        self._epoch += 1
        if self.shuffle_on_epoch and self.mode == "train":
            self._shuffle_rng.shuffle(self.batch_indices)

    @property
    def epoch(self):
        return self._epoch

    def create_train_val_split(self, val_ratio: float = 0.2) -> Tuple['ShardedHDF5Dataset', 'ShardedHDF5Dataset']:
        """
        创建train/val的独立dataset实例

        Args:
            val_ratio: 验证集比例

        Returns:
            (train_dataset, val_dataset)
        """
        # Shuffle并划分
        all_indices = np.arange(self.total_batches)
        rng = np.random.default_rng(42)  # 固定种子
        rng.shuffle(all_indices)

        num_val = int(self.total_batches * val_ratio)
        val_indices = all_indices[:num_val]
        train_indices = all_indices[num_val:]

        # 创建train dataset
        train_dataset = ShardedHDF5Dataset(
            shard_dir=str(self.shard_dir),
            batch_size=self.batch_size,
            mode="train",
            batch_indices=train_indices,
            batch_info=self.batch_info,
            shuffle_on_epoch=self.shuffle_on_epoch,
            shuffle_seed=42,
        )

        # 创建val dataset
        val_dataset = ShardedHDF5Dataset(
            shard_dir=str(self.shard_dir),
            batch_size=self.batch_size,
            mode="val",
            batch_indices=val_indices,
            batch_info=self.batch_info,
            shuffle_on_epoch=False,  # 验证集不shuffle
            shuffle_seed=43,
        )

        return train_dataset, val_dataset

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        通过索引获取一个batch

        Args:
            idx: batch索引（在当前dataset中的索引）

        Returns:
            (features, labels)
            features: (batch_size, window_size, num_features)
            labels: (batch_size,)
        """
        # 获取全局batch索引
        global_batch_idx = self.batch_indices[idx]

        # 获取batch信息
        shard_file, batch_start, batch_size = self.batch_info[global_batch_idx]

        # 读取整个batch
        with h5py.File(shard_file, "r") as f:
            data = f["data"][batch_start : batch_start + batch_size]

        # 分离特征和标签
        features = torch.from_numpy(data[:, :, :-1]).float()  # (batch_size, window_size, num_features)
        labels = torch.from_numpy(data[:, 0, -1]).long()  # (batch_size,)

        return features, labels

    def __len__(self):
        """返回当前dataset的batch数量"""
        return len(self.batch_indices)

    def get_shard_stats(self):
        """获取分片统计信息"""
        # 统计每个shard有多少batch
        shard_batch_counts = {}
        for shard_file, _, _ in self.batch_info:
            shard_batch_counts[shard_file] = shard_batch_counts.get(shard_file, 0) + 1

        stats = {
            "num_shards": len(shard_batch_counts),
            "total_samples": self.total_samples,
            "total_batches": self.total_batches,
            "batch_size": self.batch_size,
            "batches_per_shard": list(shard_batch_counts.values()),
            "current_epoch": self._epoch,
        }
        return stats


def create_sharded_dataloader(
    shard_dir: str,
    batch_size: int = 512,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    shuffle_on_epoch: bool = True,
) -> Tuple[DataLoader, DataLoader, ShardedHDF5Dataset, ShardedHDF5Dataset]:
    """
    创建ShardedDataLoader（batch对齐版本）

    Args:
        shard_dir: 分片文件目录
        batch_size: 批次大小（必须与数据对齐）
        val_ratio: 验证集比例
        num_workers: 并行worker数量
        shuffle_on_epoch: 每个epoch后是否重新shuffle（仅用于训练集）

    Returns:
        (train_loader, val_loader, train_dataset, val_dataset)

    使用示例：
        train_loader, val_loader, train_ds, val_ds = create_sharded_dataloader(
            './data/sharded',
            batch_size=512,
            num_workers=4
        )

        for epoch in range(10):
            # 训练
            for features, labels in train_loader:
                train(features, labels)
            train_ds.on_epoch_end()  # 重新shuffle

            # 验证
            for features, labels in val_loader:
                validate(features, labels)
    """
    # 创建主dataset并扫描文件
    print(f"ShardedHDF5Dataset初始化:")
    print(f"  目录: {shard_dir}")
    print(f"  Batch大小: {batch_size}")

    main_dataset = ShardedHDF5Dataset(
        shard_dir=shard_dir,
        batch_size=batch_size,
        shuffle_on_epoch=shuffle_on_epoch,
    )

    print(f"  找到 {len(set(info[0] for info in main_dataset.batch_info))} 个分片文件")
    print(f"  总样本数: {main_dataset.total_samples:,}")
    print(f"  总batch数: {main_dataset.total_batches:,}")

    # 创建train/val划分
    train_dataset, val_dataset = main_dataset.create_train_val_split(val_ratio)

    print(f"  训练batches: {len(train_dataset):,} ({len(train_dataset) * batch_size:,} 样本)")
    print(f"  验证batches: {len(val_dataset):,} ({len(val_dataset) * batch_size:,} 样本)")
    print(f"  Epoch shuffle: {'开启' if shuffle_on_epoch else '关闭'} (仅训练集)")

    # 创建DataLoader
    # 注意：DataLoader的batch_size设为None，因为Dataset已经返回batch了
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,  # 重要！Dataset已经返回batch
        shuffle=False,  # 由Dataset内部控制shuffle
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, train_dataset, val_dataset


def test_sharded_dataloader():
    """测试ShardedHDF5Dataset"""
    print("=" * 70)
    print("测试ShardedHDF5Dataset (Batch-Aligned)")
    print("=" * 70)

    shard_dir = "./data/sharded"
    if not Path(shard_dir).exists():
        print(f"错误: {shard_dir} 不存在，请先运行 reshard_hdf5.py")
        return

    # 创建DataLoader
    loader, dataset = create_sharded_dataloader(
        shard_dir=shard_dir,
        batch_size=512,
        num_workers=0,
        shuffle_on_epoch=True,
    )

    # 打印分片统计
    stats = dataset.get_shard_stats()
    print(f"\n分片统计:")
    print(f"  分片数量: {stats['num_shards']}")
    print(f"  总样本数: {stats['total_samples']:,}")
    print(f"  总batch数: {stats['total_batches']:,}")
    print(f"  Batch大小: {stats['batch_size']}")
    print(f"  每个分片batch数: 最小={min(stats['batches_per_shard'])}, 最大={max(stats['batches_per_shard'])}")

    # 训练模式 - 测试2个epoch
    print(f"\n训练模式 - 测试2个epoch:")
    dataset.set_mode("train")
    print(f"  训练集batch数: {len(dataset):,}")

    for epoch in range(2):
        print(f"\n  Epoch {epoch+1}:")
        count = 0
        for features, labels in loader:
            count += 1
            if count == 1:
                print(f"    Batch 1: features={features.shape}, labels={labels.shape}")
            if count >= 5:  # 只测试前5个batch
                break
        print(f"    测试了 {count} batches")

        # Epoch结束，重新shuffle
        dataset.on_epoch_end()
        print(f"    -> Epoch结束，已重新shuffle (当前epoch={dataset.epoch})")

    # 验证模式
    print(f"\n验证模式:")
    dataset.set_mode("val")
    print(f"  验证集batch数: {len(dataset):,}")

    count = 0
    for features, labels in loader:
        count += 1
        if count == 1:
            print(f"  Batch 1: features={features.shape}, labels={labels.shape}")
        if count >= 3:
            break

    print(f"  测试了 {count} batches")

    # 测试索引访问
    print(f"\n测试直接索引访问:")
    dataset.set_mode("train")
    features, labels = dataset[0]
    print(f"  Batch 0: features={features.shape}, labels={labels.shape}")
    print(f"  标签分布: {np.bincount(labels.numpy())}")

    print("\n" + "=" * 70)
    print("✓ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_sharded_dataloader()