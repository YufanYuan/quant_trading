"""
特征工程模块

计算并归一化以下特征：
- OHLC LogReturn (相对于最开始的)
- EMA5, EMA12, EMA26, EMA50, EMA200 (相对于最开始的LogReturn)
- RSI (归一化到0-1)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class FeatureCalculator:
    """特征计算器，处理2小时（120个数据点）的分钟级数据"""

    def __init__(self, window_size: int = 120):
        """
        Args:
            window_size: 输入窗口大小（分钟数），默认120（2小时）
        """
        self.window_size = window_size

    def calculate_log_return(self, prices: np.ndarray, base_price: float) -> np.ndarray:
        """
        计算相对于基准价格的对数收益率

        Args:
            prices: 价格数组
            base_price: 基准价格（序列的第一个价格）

        Returns:
            对数收益率数组
        """
        return np.log(prices / base_price)

    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        计算指数移动平均线

        Args:
            prices: 价格数组
            period: EMA周期

        Returns:
            EMA数组
        """
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        计算相对强弱指标RSI，并归一化到0-1范围

        Args:
            prices: 价格数组
            period: RSI周期，默认14

        Returns:
            归一化的RSI数组 (0-1范围)
        """
        if len(prices) < period + 1:
            # 数据不足，返回中性值0.5
            return np.full_like(prices, 0.5, dtype=float)

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # 初始化RSI数组
        rsi = np.zeros(len(prices))
        rsi[:period] = 0.5  # 前period个值设为中性值

        # 计算第一个平均值
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        # 计算RSI
        for i in range(period, len(prices)):
            if i > period:
                avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

            if avg_loss == 0:
                rsi[i] = 1.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = rs / (1 + rs)  # 直接归一化到0-1

        return rsi

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        准备所有特征

        Args:
            df: 包含OHLC数据的DataFrame，列名为 ['open', 'high', 'low', 'close']

        Returns:
            features: 特征数组，形状为 (num_samples, window_size, num_features)
            feature_names: 特征名称列表
        """
        if len(df) < self.window_size:
            raise ValueError(f"数据长度 {len(df)} 小于窗口大小 {self.window_size}")

        # 获取价格数据
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values

        # 准备滑动窗口特征
        num_samples = len(df) - self.window_size + 1
        features_list = []

        for i in range(num_samples):
            window_start = i
            window_end = i + self.window_size

            # 提取窗口内的数据
            window_open = open_prices[window_start:window_end]
            window_high = high_prices[window_start:window_end]
            window_low = low_prices[window_start:window_end]
            window_close = close_prices[window_start:window_end]

            # 基准价格（窗口第一个收盘价）
            base_price = window_close[0]

            # 1. OHLC LogReturn (相对于基准)
            open_logret = self.calculate_log_return(window_open, base_price)
            high_logret = self.calculate_log_return(window_high, base_price)
            low_logret = self.calculate_log_return(window_low, base_price)
            close_logret = self.calculate_log_return(window_close, base_price)

            # 2. EMA (相对于基准的LogReturn)
            ema5 = self.calculate_ema(window_close, 5)
            ema5_logret = self.calculate_log_return(ema5, base_price)

            ema12 = self.calculate_ema(window_close, 12)
            ema12_logret = self.calculate_log_return(ema12, base_price)

            ema26 = self.calculate_ema(window_close, 26)
            ema26_logret = self.calculate_log_return(ema26, base_price)

            ema50 = self.calculate_ema(window_close, 50)
            ema50_logret = self.calculate_log_return(ema50, base_price)

            ema200 = self.calculate_ema(window_close, 200)
            ema200_logret = self.calculate_log_return(ema200, base_price)

            # 3. RSI (归一化到0-1)
            rsi = self.calculate_rsi(window_close, period=14)

            # 堆叠所有特征
            window_features = np.stack([
                open_logret,
                close_logret,
                high_logret,
                low_logret,
                ema5_logret,
                ema12_logret,
                ema26_logret,
                ema50_logret,
                ema200_logret,
                rsi
            ], axis=1)  # 形状: (window_size, num_features)

            features_list.append(window_features)

        features = np.array(features_list)  # 形状: (num_samples, window_size, num_features)

        feature_names = [
            'open_logret', 'close_logret', 'high_logret', 'low_logret',
            'ema5_logret', 'ema12_logret', 'ema26_logret', 'ema50_logret', 'ema200_logret',
            'rsi'
        ]

        return features, feature_names


def test_feature_calculator():
    """测试特征计算器"""
    # 创建测试数据
    np.random.seed(42)
    n = 200
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.005),
        'close': prices
    })

    calc = FeatureCalculator(window_size=120)
    features, feature_names = calc.prepare_features(df)

    print(f"特征形状: {features.shape}")
    print(f"特征名称: {feature_names}")
    print(f"\n第一个样本的最后一行特征值:")
    print(dict(zip(feature_names, features[0, -1, :])))

    # 检查RSI是否在0-1范围内
    rsi_values = features[:, :, -1]
    print(f"\nRSI范围: [{rsi_values.min():.4f}, {rsi_values.max():.4f}]")

    # 检查LogReturn
    logret_values = features[:, :, :4]
    print(f"LogReturn范围: [{logret_values.min():.4f}, {logret_values.max():.4f}]")


if __name__ == '__main__':
    test_feature_calculator()
