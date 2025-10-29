"""
数据下载和准备模块

从Binance下载分钟级K线数据，并准备用于训练的特征和标签
"""

import sys
from pathlib import Path

# 添加项目根目录到路径，以便导入market_data
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional

from market_data import get_binance_data
from features import FeatureCalculator
from label_generator import LabelGenerator


class BinanceDataLoader:
    """Binance数据下载器（使用项目的market_data模块）"""

    def __init__(self, data_dir: str = './data'):
        """
        Args:
            data_dir: 数据保存目录（用于缓存处理后的数据）
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_data(self, df: pd.DataFrame, symbol: str, year: int):
        """
        保存数据到CSV文件

        Args:
            df: DataFrame
            symbol: 交易对符号
            year: 年份
        """
        filename = self.data_dir / f"{symbol}_{year}_1m.csv"
        df.to_csv(filename, index=False)
        print(f"数据已保存到: {filename}")

    def load_data(self, symbol: str, year: int) -> Optional[pd.DataFrame]:
        """
        从CSV文件加载数据

        Args:
            symbol: 交易对符号
            year: 年份

        Returns:
            DataFrame或None（如果文件不存在）
        """
        filename = self.data_dir / f"{symbol}_{year}_1m.csv"
        if not filename.exists():
            return None

        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def download_year_data(self, symbol: str, year: int) -> pd.DataFrame:
        """
        下载指定年份的完整数据

        Args:
            symbol: 交易对符号
            year: 年份

        Returns:
            DataFrame
        """
        # 先尝试从本地加载
        df = self.load_data(symbol, year)
        if df is not None:
            print(f"{symbol} {year}年数据已存在，直接加载")
            return df

        # 不存在则使用market_data下载
        print(f"下载 {symbol} {year}年数据...")
        start_time = datetime(year, 1, 1)
        end_time = datetime(year, 12, 31, 23, 59, 59)

        # 使用market_data的get_binance_data函数
        raw_df = get_binance_data(symbol, start_time, end_time, '1m')

        if raw_df.empty:
            print(f"警告：无法获取 {symbol} {year}年数据")
            return pd.DataFrame()

        # 转换为我们需要的格式
        df = pd.DataFrame({
            'timestamp': raw_df['open_time'],
            'open': raw_df['open'].astype(float),
            'high': raw_df['high'].astype(float),
            'low': raw_df['low'].astype(float),
            'close': raw_df['close'].astype(float),
            'volume': raw_df['volume'].astype(float)
        })

        print(f"成功获取 {len(df)} 条数据")

        # 保存到本地
        if not df.empty:
            self.save_data(df, symbol, year)

        return df


class DatasetPreparator:
    """数据集准备器，将原始数据转换为训练数据"""

    def __init__(
        self,
        window_size: int = 120,
        threshold: float = 0.005,
        horizon: int = 30
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
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备完整数据集

        Args:
            df: 原始OHLC数据

        Returns:
            (features, labels)元组
        """
        print(f"准备数据集，原始数据量: {len(df)}")

        # 1. 计算特征
        features, feature_names = self.feature_calc.prepare_features(df)
        print(f"特征计算完成，特征维度: {features.shape}")

        # 2. 生成标签
        # 注意：标签对应的是每个窗口结束时刻的价格
        # 我们需要从window_size-1位置开始生成标签
        labels = self.label_gen.generate_labels(df)

        # 对齐特征和标签
        # features[i] 对应 df[i:i+window_size]，即结束于 df[i+window_size-1]
        # 我们需要该位置的标签
        aligned_labels = labels[self.window_size - 1:self.window_size - 1 + len(features)]

        print(f"标签生成完成，标签数量: {len(aligned_labels)}")

        # 3. 显示原始标签分布
        distribution = self.label_gen.get_label_distribution(aligned_labels)
        print("\n原始标签分布 (-1/0/1):")
        for label_name, stats in distribution.items():
            print(f"  {label_name}: {stats['count']} ({stats['ratio']*100:.2f}%)")

        # 4. 转换标签为训练格式 (0/1/2)
        labels = self.label_gen.convert_labels_for_training(aligned_labels)
        print(f"\n标签已转换为训练格式: -1→2, 0→0, 1→1")
        print(f"最终样本数: {len(labels)} (包含横盘样本)")

        # 5. 优化数据类型以节省内存 (float64->float32, int64->int32)
        features = features.astype(np.float32)
        labels = labels.astype(np.int32)
        print(f"数据类型已优化: features={features.dtype}, labels={labels.dtype}")

        return features, labels

    def prepare_multiple_symbols(
        self,
        data_loader: BinanceDataLoader,
        symbols: List[str],
        years: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备多个交易对的数据集

        Args:
            data_loader: 数据加载器
            symbols: 交易对列表
            years: 年份列表

        Returns:
            合并后的(features, labels)元组
        """
        all_features = []
        all_labels = []

        for symbol in symbols:
            for year in years:
                print(f"\n处理 {symbol} {year}年数据...")

                # 加载数据
                df = data_loader.download_year_data(symbol, year)
                if df is None or df.empty:
                    print(f"  跳过 {symbol} {year}年（无数据）")
                    continue

                # 准备数据集
                features, labels = self.prepare_dataset(df)

                all_features.append(features)
                all_labels.append(labels)

        # 合并所有数据
        if all_features:
            features = np.concatenate(all_features, axis=0)
            labels = np.concatenate(all_labels, axis=0)

            print(f"\n总计样本数: {len(labels)}")
            print(f"特征维度: {features.shape}")

            return features, labels
        else:
            return np.array([]), np.array([])


def main():
    """主函数：下载并准备训练数据"""
    # 配置
    symbols = ['BTCUSDT', 'DOGEUSDT', 'ETHUSDT']
    train_years = [2022, 2023, 2024]

    # 初始化
    data_loader = BinanceDataLoader(data_dir='./data')
    preparator = DatasetPreparator(
        window_size=120,
        threshold=0.005,
        horizon=30
    )

    # 准备训练数据
    print("=" * 60)
    print("准备训练数据")
    print("=" * 60)

    features, labels = preparator.prepare_multiple_symbols(
        data_loader,
        symbols,
        train_years
    )

    # 保存处理后的数据
    if len(features) > 0:
        save_path = Path('./data')
        np.save(save_path / 'train_features.npy', features)
        np.save(save_path / 'train_labels.npy', labels)
        print(f"\n训练数据已保存到 {save_path}")


if __name__ == '__main__':
    main()
