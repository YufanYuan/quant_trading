"""
标签生成器模块

生成训练标签（三分类）：
- 预测未来30分钟内先涨0.5%
- 预测未来30分钟内先跌0.5%
- 预测未来30分钟内两者都不达到（横盘）
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class LabelGenerator:
    """
    标签生成器（三分类）

    标签定义：
    - 0: 先达到-0.5% (下跌0.5%)
    - 1: 先达到+0.5% (上涨0.5%)
    - -1: 30分钟内都未达到 (横盘/小幅波动，不交易)

    注意：所有样本都是有效的，-1不是无效标签而是"不交易"的信号
    """

    def __init__(self, threshold: float = 0.005, horizon: int = 30):
        """
        Args:
            threshold: 涨跌幅阈值（默认0.005即0.5%）
            horizon: 预测时间窗口（分钟数，默认30）
        """
        self.threshold = threshold
        self.horizon = horizon

    def generate_label(self, future_prices: np.ndarray, current_price: float) -> int:
        """
        生成单个标签

        Args:
            future_prices: 未来价格序列（最多horizon个）
            current_price: 当前价格

        Returns:
            标签: 0 (先跌), 1 (先涨), -1 (横盘)
        """
        if len(future_prices) == 0:
            return -1

        # 计算收益率
        returns = (future_prices - current_price) / current_price

        # 找到首次达到阈值的位置
        up_hit = np.where(returns >= self.threshold)[0]
        down_hit = np.where(returns <= -self.threshold)[0]

        # 判断哪个先达到
        if len(up_hit) == 0 and len(down_hit) == 0:
            return -1  # 都未达到
        elif len(up_hit) > 0 and len(down_hit) == 0:
            return 1  # 只有上涨达到
        elif len(up_hit) == 0 and len(down_hit) > 0:
            return 0  # 只有下跌达到
        else:
            # 都达到了，看哪个先
            first_up = up_hit[0]
            first_down = down_hit[0]
            return 1 if first_up < first_down else 0

    def generate_labels(self, df: pd.DataFrame, price_col: str = 'close') -> np.ndarray:
        """
        为整个数据集生成标签

        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名

        Returns:
            标签数组
        """
        prices = df[price_col].values
        labels = []

        for i in range(len(prices)):
            # 获取当前价格
            current_price = prices[i]

            # 获取未来价格（最多horizon个）
            future_end = min(i + 1 + self.horizon, len(prices))
            future_prices = prices[i + 1:future_end]

            # 生成标签
            label = self.generate_label(future_prices, current_price)
            labels.append(label)

        return np.array(labels)

    def convert_labels_for_training(self, labels: np.ndarray) -> np.ndarray:
        """
        将标签转换为训练格式（PyTorch需要标签在[0, num_classes-1]范围）

        转换规则：
        - 0 (下跌) -> 0
        - 1 (上涨) -> 1
        - -1 (横盘) -> 2

        Args:
            labels: 原始标签数组

        Returns:
            转换后的标签数组
        """
        converted = labels.copy()
        converted[labels == -1] = 2
        return converted

    def get_label_distribution(self, labels: np.ndarray) -> dict:
        """
        获取标签分布统计

        Args:
            labels: 标签数组

        Returns:
            标签分布字典
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        distribution = {}
        for label, count in zip(unique, counts):
            if label == -1:
                distribution['sideways'] = {'count': count, 'ratio': count / total}
            elif label == 0:
                distribution['down'] = {'count': count, 'ratio': count / total}
            elif label == 1:
                distribution['up'] = {'count': count, 'ratio': count / total}

        return distribution


def test_label_generator():
    """测试标签生成器"""
    # 创建测试数据
    np.random.seed(42)
    n = 1000
    base_price = 100.0

    # 生成价格数据（带有一些趋势和波动）
    returns = np.random.randn(n) * 0.002  # 0.2%的波动
    returns[100:150] += 0.006  # 添加一个上涨趋势
    returns[200:250] -= 0.006  # 添加一个下跌趋势

    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1T'),
        'close': prices
    })

    # 生成标签
    label_gen = LabelGenerator(threshold=0.005, horizon=30)
    labels = label_gen.generate_labels(df)

    print(f"总样本数: {len(labels)}")

    # 显示标签分布
    distribution = label_gen.get_label_distribution(labels)
    print("\n标签分布 (原始):")
    for label_name, stats in distribution.items():
        print(f"  {label_name}: {stats['count']} ({stats['ratio']*100:.2f}%)")

    # 测试标签转换
    converted_labels = label_gen.convert_labels_for_training(labels)
    print(f"\n标签转换测试:")
    print(f"  原始标签范围: [{labels.min()}, {labels.max()}]")
    print(f"  转换后标签范围: [{converted_labels.min()}, {converted_labels.max()}]")
    print(f"  样本总数: {len(labels)} (所有样本都有效，包括横盘)")


if __name__ == '__main__':
    test_label_generator()
