"""
策略执行器

使用训练好的模型进行实时预测和交易信号生成
"""

import torch
import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, Dict, Any
from pathlib import Path
import json

from model import LSTMClassifier, TransformerClassifier


class MLPredictionStrategy:
    """
    机器学习预测策略

    使用训练好的模型预测未来30分钟内首先出现的是涨幅0.5%还是跌幅0.5%
    """

    def __init__(
        self,
        model_path: str = 'best_model.pth',
        config_path: str = 'model_config.json',
        threshold: float = 0.005,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model_path: 模型权重路径
            config_path: 模型配置路径
            threshold: 交易阈值（默认0.005即0.5%）
            device: 计算设备
        """
        self.threshold = threshold
        self.device = device

        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.window_size = self.config['seq_len']
        self.num_features = self.config['num_features']

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        # 数据缓冲区
        self.buffer = deque(maxlen=self.window_size)

        # 当前持仓信息
        self.position = None  # None, 'long', or 'short'
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

        print(f"策略初始化完成，模型类型: {self.config['model_type']}")
        print(f"窗口大小: {self.window_size}, 特征数: {self.num_features}")

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """加载模型"""
        model_type = self.config['model_type']
        input_size = self.config['input_size']

        if model_type == 'lstm':
            model = LSTMClassifier(
                input_size=input_size,
                **self.config['lstm_config']
            )
        else:
            model = TransformerClassifier(
                input_size=input_size,
                **self.config['transformer_config']
            )

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        return model

    def _calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        计算特征

        Args:
            df: 包含最近window_size条数据的DataFrame

        Returns:
            特征向量
        """
        # 获取价格数据
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values

        # 基准价格（窗口第一个收盘价）
        base_price = close_prices[0]

        # 1. OHLC LogReturn
        open_logret = np.log(open_prices / base_price)
        high_logret = np.log(high_prices / base_price)
        low_logret = np.log(low_prices / base_price)
        close_logret = np.log(close_prices / base_price)

        # 2. EMA (相对于基准的LogReturn)
        def calculate_ema(prices, period):
            alpha = 2 / (period + 1)
            ema = np.zeros_like(prices)
            ema[0] = prices[0]
            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
            return ema

        ema5 = calculate_ema(close_prices, 5)
        ema5_logret = np.log(ema5 / base_price)

        ema12 = calculate_ema(close_prices, 12)
        ema12_logret = np.log(ema12 / base_price)

        ema26 = calculate_ema(close_prices, 26)
        ema26_logret = np.log(ema26 / base_price)

        ema50 = calculate_ema(close_prices, 50)
        ema50_logret = np.log(ema50 / base_price)

        ema200 = calculate_ema(close_prices, 200)
        ema200_logret = np.log(ema200 / base_price)

        # 3. RSI
        def calculate_rsi(prices, period=14):
            if len(prices) < period + 1:
                return np.full_like(prices, 0.5, dtype=float)

            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            rsi = np.zeros(len(prices))
            rsi[:period] = 0.5

            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])

            for i in range(period, len(prices)):
                if i > period:
                    avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
                    avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

                if avg_loss == 0:
                    rsi[i] = 1.0
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = rs / (1 + rs)

            return rsi

        rsi = calculate_rsi(close_prices, period=14)

        # 堆叠所有特征
        features = np.stack([
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

        return features

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        进行预测

        Args:
            df: 包含最近window_size条数据的DataFrame

        Returns:
            预测结果字典，包含预测类别、概率等
        """
        if len(df) < self.window_size:
            return {
                'prediction': None,
                'confidence': 0.0,
                'message': f'数据不足，需要{self.window_size}条'
            }

        # 计算特征
        features = self._calculate_features(df)

        # 转换为张量
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        return {
            'prediction': predicted_class,  # 0: 先跌, 1: 先涨, 2: 横盘
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()[0].tolist(),
            'message': 'success'
        }

    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[Dict[str, Any]]:
        """
        生成交易信号

        Args:
            df: 包含最近window_size条数据的DataFrame
            current_price: 当前价格

        Returns:
            交易信号字典，如果没有信号则返回None
        """
        # 如果已有持仓，先检查止损止盈
        if self.position is not None:
            if self.position == 'long':
                if current_price >= self.take_profit:
                    self.position = None
                    return {
                        'action': 'close_long',
                        'reason': 'take_profit',
                        'price': current_price
                    }
                elif current_price <= self.stop_loss:
                    self.position = None
                    return {
                        'action': 'close_long',
                        'reason': 'stop_loss',
                        'price': current_price
                    }
            elif self.position == 'short':
                if current_price <= self.take_profit:
                    self.position = None
                    return {
                        'action': 'close_short',
                        'reason': 'take_profit',
                        'price': current_price
                    }
                elif current_price >= self.stop_loss:
                    self.position = None
                    return {
                        'action': 'close_short',
                        'reason': 'stop_loss',
                        'price': current_price
                    }
            return None

        # 没有持仓时，进行预测
        result = self.predict(df)

        if result['prediction'] is None:
            return None

        # 只在置信度足够高时开仓
        if result['confidence'] < 0.6:
            return None

        prediction = result['prediction']
        signal = None

        if prediction == 1:  # 预测上涨
            self.position = 'long'
            self.entry_price = current_price
            self.stop_loss = current_price * (1 - self.threshold)
            self.take_profit = current_price * (1 + self.threshold)

            signal = {
                'action': 'open_long',
                'price': current_price,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'confidence': result['confidence']
            }

        elif prediction == 0:  # 预测下跌
            self.position = 'short'
            self.entry_price = current_price
            self.stop_loss = current_price * (1 + self.threshold)
            self.take_profit = current_price * (1 - self.threshold)

            signal = {
                'action': 'open_short',
                'price': current_price,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'confidence': result['confidence']
            }

        elif prediction == 2:  # 预测横盘 - 不交易
            return None

        return signal


def test_strategy():
    """测试策略"""
    # 注意：需要先训练模型并保存
    # 这里只是演示如何使用策略

    print("测试策略（需要先训练模型）")

    # 创建测试数据
    np.random.seed(42)
    n = 200
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1T'),
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.005),
        'close': prices
    })

    # 测试特征计算
    strategy = MLPredictionStrategy()
    window_df = df.iloc[:120]
    features = strategy._calculate_features(window_df)

    print(f"特征形状: {features.shape}")
    print(f"特征范围: [{features.min():.4f}, {features.max():.4f}]")


if __name__ == '__main__':
    test_strategy()
