"""
HDF5 DataLoader

Map-style数据集，支持随机访问和自动batching
在初始化时完成训练/验证集划分（使用固定种子确保一致性）
支持通过set_mode方法在训练/验证模式之间切换
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional


class HDF5Dataset(Dataset):
    """
    基于HDF5的Map-style数据集

    在初始化时随机划分训练/验证集（使用固定种子确保一致性）
    支持通过索引随机访问样本
    """

    def __init__(
        self,
        h5_path: str,
        dataset_names: Optional[List[str]] = None,
        val_ratio: float = 0.2,
    ):
        """
        Args:
            h5_path: HDF5文件路径
            dataset_names: 要使用的dataset名称列表，如果为None则使用全部
            val_ratio: 验证集比例
        """
        super().__init__()
        self.h5_path = Path(h5_path)
        self.val_ratio = val_ratio
        self._mode = "train"

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

            # 收集所有样本的全局索引：(dataset_idx, local_index)
            # 使用整数索引代替字符串，节省内存
            all_samples = []
            for dataset_idx, name in enumerate(self.dataset_names):
                total_len = len(f[name])
                # 使用列表推导式，比循环append快得多
                all_samples.extend([(dataset_idx, idx) for idx in range(total_len)])

            # 使用固定种子shuffle并划分
            rng = np.random.default_rng(42)
            rng.shuffle(all_samples)

            num_val = int(len(all_samples) * val_ratio)
            self.val_samples = all_samples[:num_val]
            self.train_samples = all_samples[num_val:]

            # 读取metadata
            self.window_size = f.attrs.get("window_size", 120)
            self.num_features = (
                len(f.attrs.get("feature_names", "").split(",")) - 1
            )  # 最后一个是label

        print(f"HDF5Dataset初始化:")
        print(f"  文件: {self.h5_path}")
        print(f"  Datasets: {self.dataset_names}")
        print(f"  训练集: {len(self.train_samples):,} 样本")
        print(f"  验证集: {len(self.val_samples):,} 样本")

    def set_mode(self, mode: str):
        """切换训练/验证模式"""
        if mode not in ["train", "val"]:
            raise ValueError(f"mode必须是'train'或'val'，当前为: {mode}")
        self._mode = mode

    @property
    def mode(self):
        return self._mode

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        通过索引获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (features, label)
            features: (window_size, num_features)
            label: 标量
        """
        # 选择样本列表
        samples = self.train_samples if self._mode == "train" else self.val_samples
        dataset_idx, local_idx = samples[idx]
        dataset_name = self.dataset_names[dataset_idx]

        # 读取单个样本
        with h5py.File(self.h5_path, "r") as f:
            data = f[dataset_name][local_idx]  # (window_size, num_features+1)

        # 分离特征和标签
        features = torch.from_numpy(data[:, :-1])  # (window_size, num_features)
        label = torch.tensor(data[:, -1], dtype=torch.long)  # 标量

        return features, label

    def __len__(self):
        """返回当前模式下的样本数量"""
        return (
            len(self.train_samples) if self._mode == "train" else len(self.val_samples)
        )


def create_dataloader(
    h5_path: str,
    dataset_names: Optional[List[str]] = None,
    batch_size: int = 256,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    shuffle: bool = True,
) -> Tuple[DataLoader, HDF5Dataset]:
    """
    创建DataLoader

    Args:
        h5_path: HDF5文件路径
        dataset_names: 要使用的dataset名称列表（None=全部）
        batch_size: 批次大小
        val_ratio: 验证集比例
        num_workers: 并行worker数量
        shuffle: 是否在每个epoch打乱数据（默认True）

    Returns:
        (dataloader, dataset) 元组

    使用示例：
        loader, dataset = create_dataloader('data.h5', batch_size=256)

        for epoch in range(10):
            # 训练（自动shuffle）
            dataset.set_mode('train')
            for features, labels in loader:
                train(features, labels)

            # 验证（不shuffle）
            dataset.set_mode('val')
            for features, labels in loader:
                validate(features, labels)
    """
    dataset = HDF5Dataset(h5_path, dataset_names, val_ratio)
    dataset.set_mode("train")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, dataset


def test_dataloader():
    """测试"""
    print("=" * 60)
    print("测试HDF5Dataset")
    print("=" * 60)

    h5_path = "./data/all_data.h5"
    if not Path(h5_path).exists():
        print(f"错误: {h5_path} 不存在")
        return

    loader, dataset = create_dataloader(h5_path, batch_size=64, shuffle=True)

    # 测试训练模式
    print("\n训练模式 - 测试2个epoch:")
    dataset.set_mode("train")
    print(f"  训练集总样本数: {len(dataset)}")

    for epoch in range(2):
        count = 0
        total_samples = 0
        for features, labels in loader:
            count += 1
            total_samples += features.shape[0]
            if count == 1:
                print(f"  Epoch {epoch+1}, Batch 1: features={features.shape}, labels={labels.shape}")
        print(f"  Epoch {epoch+1} 完成: {count} batches, {total_samples} 样本")

    # 测试验证模式
    print("\n验证模式:")
    dataset.set_mode("val")
    print(f"  验证集总样本数: {len(dataset)}")

    count = 0
    total_samples = 0
    for features, labels in loader:
        count += 1
        total_samples += features.shape[0]
        if count == 1:
            print(f"  Batch 1: features={features.shape}, labels={labels.shape}")
    print(f"  完成: {count} batches, {total_samples} 样本")

    # 测试索引访问
    print("\n测试索引访问:")
    dataset.set_mode("train")
    features, label = dataset[0]
    print(f"  样本 0: features={features.shape}, label={label}")
    features, label = dataset[100]
    print(f"  样本 100: features={features.shape}, label={label}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_dataloader()
