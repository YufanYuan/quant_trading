"""
分片数据集加载器

用于加载预先分片并shuffle的数据集
支持PyTorch DataLoader
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Iterator
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, IterableDataset


class ShardDataset(Dataset):
    """
    单个shard的数据集

    这是一个简单的Dataset，加载单个shard文件
    """

    def __init__(self, features_path: Path, labels_path: Path):
        """
        Args:
            features_path: features文件路径
            labels_path: labels文件路径
        """
        # 直接加载到内存（单个shard应该不大，约2-4GB）
        self.features = np.load(features_path)
        self.labels = np.load(labels_path)

        assert len(self.features) == len(self.labels), "Features和labels长度不匹配"

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (features, label)元组
        """
        feature = torch.from_numpy(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return feature, label


class MultiShardDataset:
    """
    多分片数据集管理器

    负责加载和管理多个shard文件
    支持两种模式：
    1. 一次性加载所有shard（内存足够时）
    2. 逐个加载shard训练（内存不足时）
    """

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 包含shard文件的目录
        """
        self.data_dir = Path(data_dir)

        # 加载元数据
        metadata_path = self.data_dir / 'metadata.npy'
        if not metadata_path.exists():
            raise FileNotFoundError(f"找不到元数据文件: {metadata_path}")

        self.metadata = np.load(metadata_path, allow_pickle=True).item()

        self.num_shards = self.metadata['num_shards']
        self.total_samples = self.metadata['total_samples']
        self.feature_shape = tuple(self.metadata['feature_shape'])

        print(f"多分片数据集已初始化")
        print(f"  数据目录: {self.data_dir}")
        print(f"  分片数量: {self.num_shards}")
        print(f"  总样本数: {self.total_samples:,}")
        print(f"  特征维度: {self.feature_shape}")
        print(f"  数据已预先shuffle: {self.metadata.get('shuffled', False)}")

    def get_all_shards_dataset(self) -> ConcatDataset:
        """
        一次性加载所有shard，返回合并的Dataset

        注意：这会将所有数据加载到内存，需要足够的RAM
        适合内存充足的情况

        Returns:
            ConcatDataset
        """
        print("加载所有shard到内存...")
        datasets = []

        for shard_idx in range(self.num_shards):
            features_file = self.data_dir / f"features_shard_{shard_idx:03d}.npy"
            labels_file = self.data_dir / f"labels_shard_{shard_idx:03d}.npy"

            print(f"  加载 shard {shard_idx+1}/{self.num_shards}...")
            dataset = ShardDataset(features_file, labels_file)
            datasets.append(dataset)

        concat_dataset = ConcatDataset(datasets)
        print(f"所有shard已加载，总样本数: {len(concat_dataset):,}")

        return concat_dataset

    def get_shard_paths(self, shard_idx: int) -> Tuple[Path, Path]:
        """
        获取指定shard的文件路径

        Args:
            shard_idx: shard索引

        Returns:
            (features_path, labels_path)
        """
        features_file = self.data_dir / f"features_shard_{shard_idx:03d}.npy"
        labels_file = self.data_dir / f"labels_shard_{shard_idx:03d}.npy"

        if not features_file.exists():
            raise FileNotFoundError(f"找不到shard文件: {features_file}")
        if not labels_file.exists():
            raise FileNotFoundError(f"找不到shard文件: {labels_file}")

        return features_file, labels_file

    def get_single_shard_dataset(self, shard_idx: int) -> ShardDataset:
        """
        加载单个shard的数据集

        Args:
            shard_idx: shard索引

        Returns:
            ShardDataset
        """
        features_file, labels_file = self.get_shard_paths(shard_idx)
        return ShardDataset(features_file, labels_file)

    def iter_shards(self, shuffle_shards: bool = False) -> List[int]:
        """
        返回shard索引的迭代顺序

        Args:
            shuffle_shards: 是否随机打乱shard顺序

        Returns:
            shard索引列表
        """
        shard_indices = list(range(self.num_shards))

        if shuffle_shards:
            np.random.shuffle(shard_indices)

        return shard_indices


class RotatingShardDataset(IterableDataset):
    """
    支持分片轮转的 IterableDataset

    这个类继承自 torch.utils.data.IterableDataset
    可以直接用标准的 PyTorch DataLoader 来加载
    每次迭代时逐个加载 shard，训练完后自动释放内存

    适合内存不足的情况
    """

    def __init__(
        self,
        multi_shard: MultiShardDataset,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True
    ):
        """
        Args:
            multi_shard: MultiShardDataset 实例
            shuffle_shards: 是否在每个 epoch 开始时随机打乱 shard 顺序
            shuffle_within_shard: 是否在每个 shard 内部 shuffle
        """
        super().__init__()
        self.multi_shard = multi_shard
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """迭代所有 shard 的所有样本"""
        # 获取 shard 顺序
        shard_order = self.multi_shard.iter_shards(shuffle_shards=self.shuffle_shards)

        for shard_idx in shard_order:
            # 加载当前 shard
            dataset = self.multi_shard.get_single_shard_dataset(shard_idx)

            # 创建索引列表
            indices = list(range(len(dataset)))
            if self.shuffle_within_shard:
                np.random.shuffle(indices)

            # 迭代当前 shard 的所有样本
            for idx in indices:
                yield dataset[idx]

            # 释放内存
            del dataset
            import gc
            gc.collect()