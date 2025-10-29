"""
使用分片数据集的模型训练脚本

相比train.py，这个版本支持：
1. 加载预先分片并shuffle的数据集
2. 可选择一次性加载所有shard或逐个shard训练
3. 大幅降低内存占用
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import json
from datetime import datetime
from tqdm import tqdm

from model import TransformerClassifier
from sharded_dataset import MultiShardDataset, RotatingShardDataset


class Trainer:
    """模型训练器（支持分片数据集）"""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        tensorboard_dir: Optional[str] = None
    ):
        """
        Args:
            model: 模型
            device: 设备
            learning_rate: 学习率
            weight_decay: L2正则化系数
            tensorboard_dir: TensorBoard 日志目录
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=1000,  # 按step更新，所以patience提高到1000
        )

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        # 创建 TensorBoard writer
        self.writer = None
        if tensorboard_dir is not None:
            self.writer = SummaryWriter(tensorboard_dir)
            print(f"TensorBoard 日志目录: {tensorboard_dir}")

    def train_epoch(self, train_loader: DataLoader, epoch: int = 0, num_epochs: int = 0, global_step: int = 0) -> Tuple[float, int]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # 创建进度条
        desc = f"Epoch {epoch}/{num_epochs}" if num_epochs > 0 else "Training"
        pbar = tqdm(train_loader, desc=desc, leave=False)

        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 每个step更新scheduler
            self.scheduler.step(loss.item())

            # 记录到 TensorBoard（每个step）
            if self.writer is not None:
                self.writer.add_scalar('Loss/train_step', loss.item(), global_step)
                # 记录当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('LearningRate_step', current_lr, global_step)

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            # 更新进度条显示当前loss和学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, global_step

    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        # 用于计算混淆矩阵
        all_preds = []
        all_labels = []

        # 创建验证进度条
        pbar = tqdm(val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for features, labels in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                num_batches += 1

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 更新进度条显示当前准确率
                current_acc = correct / total if total > 0 else 0.0
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        # 计算每个类别的准确率
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        class_metrics = {}
        for label in [0, 1, 2]:  # 三分类：横盘、上涨、下跌
            mask = all_labels == label
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_labels[mask]).mean()
                class_metrics[f'class_{label}_accuracy'] = class_acc
                class_metrics[f'class_{label}_count'] = mask.sum()

        return avg_loss, accuracy, class_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = './checkpoints'
    ):
        """训练模型"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0
        global_step = 0  # 初始化全局步数

        print(f"开始训练，设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)

        for epoch in range(num_epochs):
            # 训练
            train_loss, global_step = self.train_epoch(train_loader, epoch + 1, num_epochs, global_step)
            self.train_losses.append(train_loss)

            # 验证
            val_loss, val_accuracy, class_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            # 记录到 TensorBoard（按epoch）
            if self.writer is not None:
                self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)
                self.writer.add_scalar('Loss/val_epoch', val_loss, epoch)
                self.writer.add_scalar('Accuracy/val_epoch', val_accuracy, epoch)

                # 记录每个类别的准确率
                for key, value in class_metrics.items():
                    if 'accuracy' in key:
                        class_label = key.split('_')[1]  # 提取类别标签
                        self.writer.add_scalar(f'ClassAccuracy/class_{class_label}_epoch', value, epoch)

            # 打印信息
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            for key, value in class_metrics.items():
                if 'accuracy' in key:
                    print(f"  {key}: {value:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_checkpoint(checkpoint_dir / 'best_model.pth')
                print(f"  -> 保存最佳模型 (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            print("-" * 60)

        # 关闭 TensorBoard writer
        if self.writer is not None:
            self.writer.close()

        print("\n训练完成！")
        print(f"最佳验证损失: {best_val_loss:.4f}")

    def save_checkpoint(self, filepath: Path):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: Path):
        """加载检查点"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']


def main():
    """主训练函数（使用分片数据集）"""
    # 配置
    config = {
        'model_type': 'transformer',  # 使用Transformer模型（基于Attention）
        'batch_size': 512,
        'num_epochs': 100,
        'learning_rate': 0.0001,
        'weight_decay': 1e-5,
        'val_split': 0.2,
        'early_stopping_patience': 15,
        'data_dir': './data/shards_float32',  # 分片数据目录
        'transformer_config': {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.3
        }
    }

    print("=" * 60)
    print("使用分片数据集训练模型（逐个shard加载）")
    print("=" * 60)

    # 加载分片数据集
    print(f"\n加载数据集: {config['data_dir']}")
    multi_shard = MultiShardDataset(config['data_dir'])

    # 获取特征维度（从第一个shard取样本）
    sample_dataset = multi_shard.get_single_shard_dataset(0)
    sample_features, _ = sample_dataset[0]
    input_size = sample_features.shape[-1]
    seq_len = sample_features.shape[0]
    del sample_dataset

    print(f"\n特征信息:")
    print(f"  输入特征数: {input_size}")
    print(f"  序列长度: {seq_len}")

    # 划分训练集和验证集的 shard
    # 使用最后 20% 的 shard 作为验证集
    num_val_shards = max(1, int(multi_shard.num_shards * config['val_split']))
    num_train_shards = multi_shard.num_shards - num_val_shards

    print(f"\nShard 划分:")
    print(f"  训练集 shards: 0-{num_train_shards-1} ({num_train_shards} 个)")
    print(f"  验证集 shards: {num_train_shards}-{multi_shard.num_shards-1} ({num_val_shards} 个)")

    # 创建训练集的包装类
    class ShardSubset:
        """MultiShardDataset 的子集包装器"""
        def __init__(self, parent_multi_shard, start_idx, end_idx):
            self.parent = parent_multi_shard
            self.start_idx = start_idx
            self.end_idx = end_idx
            self.num_shards = end_idx - start_idx

        def get_single_shard_dataset(self, shard_idx):
            return self.parent.get_single_shard_dataset(self.start_idx + shard_idx)

        def iter_shards(self, shuffle_shards=False):
            indices = list(range(self.num_shards))
            if shuffle_shards:
                np.random.shuffle(indices)
            return indices

    # 创建训练集和验证集
    train_multi_shard = ShardSubset(multi_shard, 0, num_train_shards)
    val_multi_shard = ShardSubset(multi_shard, num_train_shards, multi_shard.num_shards)

    train_dataset = RotatingShardDataset(
        train_multi_shard,
        shuffle_shards=True,
        shuffle_within_shard=False  # 数据已预先 shuffle
    )

    val_dataset = RotatingShardDataset(
        val_multi_shard,
        shuffle_shards=False,
        shuffle_within_shard=False
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=0,
        pin_memory=True
    )

    print(f"\n训练集: ~{num_train_shards * multi_shard.total_samples // multi_shard.num_shards:,} 样本")
    print(f"验证集: ~{num_val_shards * multi_shard.total_samples // multi_shard.num_shards:,} 样本")

    # 创建模型（Transformer with Attention）
    model = TransformerClassifier(input_size=input_size, **config['transformer_config'])

    # 创建 TensorBoard 日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_dir = f'./runs/{config["model_type"]}_{timestamp}'

    # 创建训练器
    trainer = Trainer(
        model,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        tensorboard_dir=tensorboard_dir
    )

    # 训练
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )

    # 保存配置
    config['timestamp'] = datetime.now().isoformat()
    config['total_samples'] = multi_shard.total_samples
    config['num_shards'] = multi_shard.num_shards
    config['input_size'] = int(input_size)
    config['seq_len'] = int(seq_len)

    config_path = Path('./checkpoints/model_config.json')
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n模型配置已保存到 {config_path}")


if __name__ == '__main__':
    main()