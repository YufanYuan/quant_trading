"""
使用HDF5数据集的模型训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import json
from datetime import datetime
from tqdm import tqdm

from model import TransformerClassifier
from hdf5_dataloader import create_dataloader


class Trainer:
    """模型训练器"""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        tensorboard_dir: Optional[str] = None,
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
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        # 创建 TensorBoard writer
        self.writer = None
        if tensorboard_dir is not None:
            self.writer = SummaryWriter(tensorboard_dir)
            print(f"TensorBoard 日志目录: {tensorboard_dir}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int = 0,
        num_epochs: int = 0,
        steps_per_epoch: int = 1000,
        global_step: int = 0,
    ) -> Tuple[float, int]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # 创建进度条
        desc = f"Epoch {epoch}/{num_epochs}" if num_epochs > 0 else "Training"
        pbar = tqdm(total=steps_per_epoch, desc=desc, leave=False)

        for step, (features, labels) in enumerate(train_loader):
            if step >= steps_per_epoch:
                break

            features = features.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录到 TensorBoard（每个step）
            if self.writer is not None:
                self.writer.add_scalar("Loss/train_step", loss.item(), global_step)
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("LearningRate", current_lr, global_step)

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            # 更新进度条
            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        pbar.close()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, global_step

    def validate(
        self, val_loader: DataLoader, steps_per_epoch: int = 200
    ) -> Tuple[float, float, Dict]:
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
        pbar = tqdm(total=steps_per_epoch, desc="Validating", leave=False)

        with torch.no_grad():
            for step, (features, labels) in enumerate(val_loader):
                if step >= steps_per_epoch:
                    break

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

                # 更新进度条
                current_acc = correct / total if total > 0 else 0.0
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.4f}"}
                )

        pbar.close()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        # 计算每个类别的准确率
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        class_metrics = {}
        for label in [0, 1, 2]:  # 三分类：横盘(0)、上涨(1)、下跌(2)
            mask = all_labels == label
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_labels[mask]).mean()
                class_metrics[f"class_{label}_accuracy"] = class_acc
                class_metrics[f"class_{label}_count"] = int(mask.sum())

        return avg_loss, accuracy, class_metrics

    def train(
        self,
        loader: DataLoader,
        dataset,
        num_epochs: int = 50,
        steps_per_epoch: int = 1000,
        val_steps_per_epoch: int = 200,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "./checkpoints",
    ):
        """训练模型"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        patience_counter = 0
        global_step = 0

        print(f"开始训练，设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)

        for epoch in range(num_epochs):
            # 训练
            dataset.set_mode('train')
            train_loss, global_step = self.train_epoch(
                loader, epoch + 1, num_epochs, steps_per_epoch, global_step
            )
            self.train_losses.append(train_loss)

            # 验证
            dataset.set_mode('val')
            val_loss, val_accuracy, class_metrics = self.validate(
                loader, val_steps_per_epoch
            )
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            # 更新学习率（基于验证loss）
            self.scheduler.step(val_loss)

            # 记录到 TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("Loss/train_epoch", train_loss, epoch)
                self.writer.add_scalar("Loss/val_epoch", val_loss, epoch)
                self.writer.add_scalar("Accuracy/val_epoch", val_accuracy, epoch)

                # 记录每个类别的准确率
                for key, value in class_metrics.items():
                    if "accuracy" in key:
                        class_label = key.split("_")[1]
                        self.writer.add_scalar(
                            f"ClassAccuracy/class_{class_label}", value, epoch
                        )

            # 打印信息
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            for key, value in class_metrics.items():
                if "accuracy" in key:
                    class_name = ["Sideways", "Up", "Down"][int(key.split("_")[1])]
                    print(
                        f"  {class_name} Acc: {value:.4f} (n={class_metrics[key.replace('accuracy', 'count')]})"
                    )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_checkpoint(checkpoint_dir / "best_model.pth")
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
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: Path):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.val_accuracies = checkpoint.get("val_accuracies", [])


def main():
    """主训练函数"""
    # 配置
    config = {
        "model_type": "transformer",
        "h5_path": "./data/all_data.h5",
        "datasets": None,  # None表示使用所有datasets，也可以指定列表
        "val_ratio": 0.2,  # 验证集比例
        "batch_size": 256,
        "num_epochs": 100,
        "steps_per_epoch": 1000,
        "val_steps_per_epoch": 200,
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
        "early_stopping_patience": 15,
        "num_workers": 0,  # Windows上建议使用0，避免多进程问题
        "transformer_config": {
            "d_model": 128,
            "nhead": 8,
            "num_layers": 3,
            "dim_feedforward": 512,
            "dropout": 0.3,
        },
    }

    print("=" * 60)
    print("使用HDF5数据集训练模型")
    print("=" * 60)

    # 检查数据文件
    h5_path = Path(config["h5_path"])
    if not h5_path.exists():
        print(f"错误: 找不到HDF5文件 {h5_path}")
        print("请先运行 data_loader.py 生成数据")
        return

    # 创建DataLoader
    print(f"\n创建DataLoader...")
    print(f"  使用datasets: {config['datasets'] if config['datasets'] else '全部'}")
    print(f"  验证集比例: {config['val_ratio']:.1%}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Num workers: {config['num_workers']}")

    loader, dataset = create_dataloader(
        h5_path=str(h5_path),
        dataset_names=config["datasets"],
        batch_size=config["batch_size"],
        val_ratio=config["val_ratio"],
        num_workers=config["num_workers"],
    )

    # 获取特征维度（从第一个batch取样本）
    print(f"\n获取数据维度...")
    dataset.set_mode('train')
    sample_features, _ = next(iter(loader))
    input_size = sample_features.shape[-1]
    seq_len = sample_features.shape[1]

    print(f"  输入特征数: {input_size}")
    print(f"  序列长度: {seq_len}")

    # 创建模型
    print(f"\n创建{config['model_type']}模型...")
    model = TransformerClassifier(input_size=input_size, **config["transformer_config"])

    # 创建 TensorBoard 日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_dir = f'./runs/{config["model_type"]}_{timestamp}'

    # 创建训练器
    trainer = Trainer(
        model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        tensorboard_dir=tensorboard_dir,
    )

    # 训练
    trainer.train(
        loader,
        dataset,
        num_epochs=config["num_epochs"],
        steps_per_epoch=config["steps_per_epoch"],
        val_steps_per_epoch=config["val_steps_per_epoch"],
        early_stopping_patience=config["early_stopping_patience"],
    )

    # 保存配置
    config["timestamp"] = datetime.now().isoformat()
    config["input_size"] = int(input_size)
    config["seq_len"] = int(seq_len)
    config["h5_path"] = str(h5_path)

    config_path = Path("./checkpoints/model_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n模型配置已保存到 {config_path}")
    print(f"TensorBoard日志: tensorboard --logdir={tensorboard_dir}")


if __name__ == "__main__":
    main()
