"""
HDF5 DataLoader

从HDF5文件中顺序读取已shuffle的数据
支持通过set_mode方法在训练/验证模式之间切换
"""

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Iterator


class HDF5Dataset(IterableDataset):
    """
    基于HDF5的数据集

    在初始化时随机划分训练/验证集（使用固定种子确保一致性）
    迭代时顺序遍历（数据已经shuffle过）
    支持多次迭代，每次都是完整的一个epoch
    """

    def __init__(
        self,
        h5_path: str,
        dataset_names: Optional[List[str]] = None,
        batch_size: int = 256,
        val_ratio: float = 0.2,
    ):
        """
        Args:
            h5_path: HDF5文件路径
            dataset_names: 要使用的dataset名称列表，如果为None则使用全部
            batch_size: 批次大小
            val_ratio: 验证集比例
        """
        super().__init__()
        self.h5_path = Path(h5_path)
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self._mode = 'train'

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5文件不存在: {self.h5_path}")

        # 读取并划分数据
        with h5py.File(self.h5_path, "r") as f:
            # 获取所有可用的datasets
            if dataset_names is None:
                self.dataset_names = list(f.keys())
            else:
                available = set(f.keys())
                specified = set(dataset_names)
                missing = specified - available
                if missing:
                    raise ValueError(f"以下datasets不存在: {missing}")
                self.dataset_names = dataset_names

            # 收集所有样本的全局索引：(dataset_name, local_index)
            all_samples = []
            for name in self.dataset_names:
                total_len = len(f[name])
                # 使用列表推导式，比循环append快得多
                all_samples.extend([(name, idx) for idx in range(total_len)])

            # 使用固定种子shuffle并划分
            rng = np.random.default_rng(42)
            rng.shuffle(all_samples)

            num_val = int(len(all_samples) * val_ratio)
            self.val_samples = all_samples[:num_val]
            self.train_samples = all_samples[num_val:]

            # 读取metadata
            self.window_size = f.attrs.get("window_size", 120)
            self.num_features = len(f.attrs.get("feature_names", "").split(",")) - 1  # 最后一个是label

        print(f"HDF5Dataset初始化:")
        print(f"  文件: {self.h5_path}")
        print(f"  Datasets: {self.dataset_names}")
        print(f"  训练集: {len(self.train_samples):,} 样本")
        print(f"  验证集: {len(self.val_samples):,} 样本")
        print(f"  Batch size: {batch_size}")

    def set_mode(self, mode: str):
        """切换训练/验证模式"""
        if mode not in ['train', 'val']:
            raise ValueError(f"mode必须是'train'或'val'，当前为: {mode}")
        self._mode = mode

    @property
    def mode(self):
        return self._mode

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        迭代生成batch

        Yields:
            (features, labels)
            features: (batch_size, window_size, num_features)
            labels: (batch_size,)
        """
        # 选择样本列表
        samples = self.train_samples if self._mode == 'train' else self.val_samples

        # 处理多worker情况
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = len(samples) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(samples)
            samples = samples[start:end]

        # 打开HDF5并顺序遍历
        with h5py.File(self.h5_path, "r") as f:
            for batch_start in range(0, len(samples), self.batch_size):
                batch_samples = samples[batch_start:batch_start + self.batch_size]

                # 读取batch
                batch_data = []
                for dataset_name, idx in batch_samples:
                    batch_data.append(f[dataset_name][idx])

                batch_data = np.array(batch_data)  # (batch_size, window_size, num_features+1)

                features = torch.from_numpy(batch_data[:, :, :-1])  # (batch_size, window_size, num_features)
                labels = torch.from_numpy(batch_data[:, 0, -1]).long()  # (batch_size,)

                yield features, labels

    def __len__(self):
        """返回一个epoch的batch数量"""
        n_samples = len(self.train_samples) if self._mode == 'train' else len(self.val_samples)
        return (n_samples + self.batch_size - 1) // self.batch_size


def create_dataloader(
    h5_path: str,
    dataset_names: Optional[List[str]] = None,
    batch_size: int = 256,
    val_ratio: float = 0.2,
    num_workers: int = 0,
) -> Tuple[DataLoader, HDF5Dataset]:
    """
    创建DataLoader

    Args:
        h5_path: HDF5文件路径
        dataset_names: 要使用的dataset名称列表（None=全部）
        batch_size: 批次大小
        val_ratio: 验证集比例
        num_workers: 并行worker数量

    Returns:
        (dataloader, dataset) 元组

    使用示例：
        loader, dataset = create_dataloader('data.h5', batch_size=256)

        for epoch in range(10):
            # 训练
            dataset.set_mode('train')
            for features, labels in loader:
                train(features, labels)

            # 验证
            dataset.set_mode('val')
            for features, labels in loader:
                validate(features, labels)
    """
    dataset = HDF5Dataset(h5_path, dataset_names, batch_size, val_ratio)
    dataset.set_mode('train')

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, dataset


def test_dataloader():
    """测试"""
    print("="*60)
    print("测试HDF5Dataset")
    print("="*60)

    h5_path = './data/test_data.h5'
    if not Path(h5_path).exists():
        print(f"错误: {h5_path} 不存在")
        return

    loader, dataset = create_dataloader(h5_path, batch_size=64)

    # 测试多个epoch
    print("\n训练模式 - 2个epoch:")
    dataset.set_mode('train')
    for epoch in range(2):
        count = 0
        for features, labels in loader:
            count += 1
            if count == 1:
                print(f"  Epoch {epoch+1}, Batch 1: {features.shape}, {labels.shape}")
        print(f"  Epoch {epoch+1} 完成: {count} batches")

    # 测试验证
    print("\n验证模式:")
    dataset.set_mode('val')
    count = 0
    for features, labels in loader:
        count += 1
        if count == 1:
            print(f"  Batch 1: {features.shape}")
    print(f"  完成: {count} batches")

    print("\n" + "="*60)


if __name__ == '__main__':
    test_dataloader()