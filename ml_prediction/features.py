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

    def calculate_ema(self, series: pd.Series, period: int) -> np.ndarray:
        """
        计算指数移动平均线（使用pandas ewm）

        Args:
            series: 价格Series
            period: EMA周期

        Returns:
            EMA数组
        """
        return series.ewm(span=period, adjust=False).mean().values

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> np.ndarray:
        """
        计算相对强弱指标RSI，并归一化到0-1范围（使用pandas rolling）

        Args:
            series: 价格Series
            period: RSI周期，默认14

        Returns:
            归一化的RSI数组 (0-1范围)
        """
        if len(series) < period + 1:
            # 数据不足，返回中性值0.5
            return np.full(len(series), 0.5, dtype=np.float32)

        # 计算价格变化
        delta = series.diff()

        # 分离涨跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 使用ewm计算平均涨跌（Wilder's smoothing）
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        # 计算RS和RSI
        rs = avg_gain / avg_loss
        rsi = rs / (1 + rs)  # 直接归一化到0-1

        # 填充前period个NaN值为中性值0.5
        rsi = rsi.fillna(0.5)

        return rsi.values.astype(np.float32)

    def prepare_features(
        self,
        df: pd.DataFrame,
        warmup_size: int = 0
    ) -> Tuple[np.ndarray, List[str]]:
        """
        准备所有特征（在完整序列上计算指标，然后提取滑动窗口）

        Args:
            df: 包含OHLC数据的DataFrame，列名为 ['open', 'high', 'low', 'close']
                可能包含warm-up数据在前面
            warmup_size: warm-up数据的行数，返回时会排除这部分的窗口

        Returns:
            features: 特征数组，形状为 (num_samples, window_size, num_features)
            feature_names: 特征名称列表
        """
        if len(df) < self.window_size:
            raise ValueError(f"数据长度 {len(df)} 小于窗口大小 {self.window_size}")

        # 重置索引以确保连续性
        df = df.reset_index(drop=True)

        # ===== 第一步：在整个序列上计算所有技术指标 =====
        # 使用pandas方法，利用完整历史信息
        close_series = df['close'].astype(np.float32)

        # EMA (直接在完整序列上计算)
        ema5 = self.calculate_ema(close_series, 5)
        ema12 = self.calculate_ema(close_series, 12)
        ema26 = self.calculate_ema(close_series, 26)
        ema50 = self.calculate_ema(close_series, 50)
        ema200 = self.calculate_ema(close_series, 200)

        # RSI
        rsi = self.calculate_rsi(close_series, period=14)

        # 将计算好的指标添加到DataFrame
        df_features = df.copy()
        df_features['ema5'] = ema5
        df_features['ema12'] = ema12
        df_features['ema26'] = ema26
        df_features['ema50'] = ema50
        df_features['ema200'] = ema200
        df_features['rsi'] = rsi

        # ===== 第二步：提取滑动窗口特征 =====
        num_samples = len(df) - self.window_size + 1
        features_list = []

        for i in range(num_samples):
            window_start = i
            window_end = i + self.window_size

            # 提取窗口数据
            window_df = df_features.iloc[window_start:window_end]

            # 基准价格（窗口第一个收盘价）
            base_price = window_df['close'].iloc[0]

            # 1. OHLC LogReturn (相对于基准)
            open_logret = self.calculate_log_return(window_df['open'].values, base_price)
            close_logret = self.calculate_log_return(window_df['close'].values, base_price)
            high_logret = self.calculate_log_return(window_df['high'].values, base_price)
            low_logret = self.calculate_log_return(window_df['low'].values, base_price)

            # 2. EMA LogReturn (相对于基准)
            ema5_logret = self.calculate_log_return(window_df['ema5'].values, base_price)
            ema12_logret = self.calculate_log_return(window_df['ema12'].values, base_price)
            ema26_logret = self.calculate_log_return(window_df['ema26'].values, base_price)
            ema50_logret = self.calculate_log_return(window_df['ema50'].values, base_price)
            ema200_logret = self.calculate_log_return(window_df['ema200'].values, base_price)

            # 3. RSI (已经是0-1范围)
            window_rsi = window_df['rsi'].values

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
                window_rsi
            ], axis=1).astype(np.float32)  # 形状: (window_size, num_features)

            features_list.append(window_features)

        features = np.array(features_list, dtype=np.float32)  # 形状: (num_samples, window_size, num_features)

        # ===== 第三步：如果有warm-up，排除warm-up部分的窗口 =====
        if warmup_size > 0:
            # 只保留窗口结束位置在warm-up之后的样本
            # 窗口[i, i+window_size)，结束位置是i+window_size-1
            # 我们要 i+window_size-1 >= warmup_size，即 i >= warmup_size - window_size + 1
            valid_start = max(0, warmup_size - self.window_size + 1)
            features = features[valid_start:]

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
