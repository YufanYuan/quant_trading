"""
数据预处理模块

负责将原始OHLC数据转换为训练特征和标签
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Optional

from ml_prediction.features import FeatureCalculator
from ml_prediction.label_generator import LabelGenerator


class DatasetPreparator:
    """数据集准备器，将原始数据转换为训练数据"""

    def __init__(
        self, window_size: int = 120, threshold: float = 0.005, horizon: int = 30
    ):
        """
        Args:
            window_size: 输入窗口大小（分钟数）
            threshold: 涨跌幅阈值
            horizon: 预测时间窗口（分钟数）
        """
        self.feature_calc = FeatureCalculator(window_size=window_size)
        self.label_gen = LabelGenerator(threshold=threshold, horizon=horizon)
        self.window_size = window_size

    def prepare_dataset(
        self, df: pd.DataFrame, warmup_df: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        准备完整数据集（支持warm-up数据）

        Args:
            df: 原始OHLC数据（当前年份）
            warmup_df: 可选的warm-up数据（上一年末尾数据）

        Returns:
            合并后的数据数组，形状为 (num_samples, window_size, num_features+1)
            最后一列为label，存储在 [:, 0, -1] 位置
        """
        print(f"准备数据集，原始数据量: {len(df)}")

        # 1. 处理warm-up数据
        warmup_size = 0
        if warmup_df is not None and not warmup_df.empty:
            warmup_size = len(warmup_df)
            # 合并warm-up和当前数据
            full_df = pd.concat([warmup_df, df], ignore_index=True)
            print(f"  包含{warmup_size}条warm-up数据，总数据量: {len(full_df)}")
        else:
            full_df = df.copy()
            print(f"  无warm-up数据")

        # 2. 计算特征（在完整序列上计算，包括warm-up）
        features, feature_names = self.feature_calc.prepare_features(
            full_df, warmup_size=warmup_size
        )
        print(f"特征计算完成，特征维度: {features.shape}")

        # 3. 生成标签（也是在完整序列上，然后对齐）
        labels = self.label_gen.generate_labels(full_df)

        # 对齐特征和标签
        # features[i] 对应 full_df[i:i+window_size]，即结束于 full_df[i+window_size-1]
        # 需要该位置的标签，但要考虑warm-up偏移
        if warmup_size > 0:
            # features已经排除了warm-up部分的窗口
            # 第一个feature窗口的结束位置大约在warmup_size附近
            # 我们需要对应位置的labels
            valid_start = max(0, warmup_size - self.window_size + 1)
            label_start = valid_start + self.window_size - 1
        else:
            label_start = self.window_size - 1

        aligned_labels = labels[label_start : label_start + len(features)]

        print(f"标签生成完成，标签数量: {len(aligned_labels)}")

        # 确保特征和标签数量一致
        min_len = min(len(features), len(aligned_labels))
        features = features[:min_len]
        aligned_labels = aligned_labels[:min_len]

        # 4. 显示原始标签分布
        distribution = self.label_gen.get_label_distribution(aligned_labels)
        print("\n原始标签分布 (-1/0/1):")
        for label_name, stats in distribution.items():
            print(f"  {label_name}: {stats['count']} ({stats['ratio']*100:.2f}%)")

        # 5. 转换标签为训练格式 (0/1/2)
        labels_converted = self.label_gen.convert_labels_for_training(aligned_labels)
        print(f"\n标签已转换为训练格式: -1→2, 0→0, 1→1")

        # 6. 合并features和labels为单个数组
        # features shape: (N, 120, 10)
        # labels shape: (N,) -> (N, 1, 1) -> 然后在时间维度上扩展
        # 结果: (N, 120, 11)，最后一列是label（只在第一个时间步有值）
        labels_expanded = np.full(
            (len(features), self.window_size, 1), np.nan, dtype=np.float32
        )
        labels_expanded[:, 0, 0] = labels_converted.astype(
            np.float32
        )  # 只在第一个位置存储label

        combined_data = np.concatenate(
            [features, labels_expanded], axis=2
        )  # (N, 120, 11)
        print(f"数据已合并，形状: {combined_data.shape}")

        # 7. 检测并移除包含NaN的样本
        # 检查features部分（前10列）是否有NaN
        nan_mask = np.isnan(combined_data[:, :, :-1]).any(axis=(1, 2))
        num_nan = nan_mask.sum()

        if num_nan > 0:
            print(f"\n警告：检测到{num_nan}个包含NaN的样本，将被丢弃")
            combined_data = combined_data[~nan_mask]
        else:
            print(f"\n[OK] 未检测到NaN，数据质量良好")

        print(f"最终样本数: {len(combined_data)}")
        print(f"数据类型: {combined_data.dtype}")

        return combined_data
