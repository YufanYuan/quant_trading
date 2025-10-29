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
import h5py
from datetime import datetime
from typing import List, Tuple, Optional

from market_data import get_binance_data
from .features import FeatureCalculator
from .label_generator import LabelGenerator


class BinanceDataLoader:
    """Binance数据下载器（使用项目的market_data模块）"""

    def __init__(self, data_dir: str = "./data"):
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
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def load_warmup_data(
        self, symbol: str, year: int, warmup_size: int = 200
    ) -> Optional[pd.DataFrame]:
        """
        加载上一年的最后N条数据作为warm-up

        Args:
            symbol: 交易对符号
            year: 当前年份（会尝试加载year-1的数据）
            warmup_size: 需要的warm-up数据条数

        Returns:
            DataFrame或None（如果上一年数据不存在）
        """
        prev_year = year - 1
        prev_df = self.load_data(symbol, prev_year)

        if prev_df is None or prev_df.empty:
            return None

        # 返回最后warmup_size条数据
        warmup_df = prev_df.tail(warmup_size).copy()
        print(f"  已加载 {symbol} {prev_year}年的最后{len(warmup_df)}条数据作为warm-up")
        return warmup_df

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
        raw_df = get_binance_data(symbol, start_time, end_time, "1m")

        if raw_df.empty:
            print(f"警告：无法获取 {symbol} {year}年数据")
            return pd.DataFrame()

        # 转换为我们需要的格式
        df = pd.DataFrame(
            {
                "timestamp": raw_df["open_time"],
                "open": raw_df["open"].astype(float),
                "high": raw_df["high"].astype(float),
                "low": raw_df["low"].astype(float),
                "close": raw_df["close"].astype(float),
                "volume": raw_df["volume"].astype(float),
            }
        )

        print(f"成功获取 {len(df)} 条数据")

        # 保存到本地
        if not df.empty:
            self.save_data(df, symbol, year)

        return df


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

    def prepare_multiple_symbols(
        self,
        data_loader: BinanceDataLoader,
        symbols: List[str],
        years: List[int],
        output_file: str = "./data/all_data.h5",
        warmup_size: int = 200,
    ) -> None:
        """
        准备多个交易对的数据集，并逐年写入HDF5文件

        Args:
            data_loader: 数据加载器
            symbols: 交易对列表
            years: 年份列表
            output_file: HDF5输出文件路径
            warmup_size: warm-up数据条数
        """
        print(f"\n{'='*60}")
        print(f"开始处理数据，输出到: {output_file}")
        print(f"{'='*60}\n")

        total_samples = 0

        # 打开HDF5文件（写模式）
        with h5py.File(output_file, "w") as f:
            # 存储元数据
            f.attrs["window_size"] = self.window_size
            f.attrs["threshold"] = self.label_gen.threshold
            f.attrs["horizon"] = self.label_gen.horizon
            f.attrs["feature_names"] = (
                "open_logret,close_logret,high_logret,low_logret,ema5_logret,ema12_logret,ema26_logret,ema50_logret,ema200_logret,rsi,label"
            )

            for symbol in symbols:
                print(f"\n{'='*60}")
                print(f"处理交易对: {symbol}")
                print(f"{'='*60}")

                for year in years:
                    print(f"\n--- 处理 {symbol} {year}年数据 ---")

                    # 1. 加载当前年份数据
                    df = data_loader.download_year_data(symbol, year)
                    if df is None or df.empty:
                        print(f"  跳过 {symbol} {year}年（无数据）")
                        continue

                    # 2. 尝试加载warm-up数据
                    warmup_df = data_loader.load_warmup_data(symbol, year, warmup_size)

                    # 3. 准备数据集
                    combined_data = self.prepare_dataset(df, warmup_df)

                    if len(combined_data) == 0:
                        print(f"  跳过 {symbol} {year}年（处理后无有效数据）")
                        continue

                    # 4. 立即写入HDF5
                    dataset_name = f"{symbol}_{year}"
                    f.create_dataset(
                        dataset_name,
                        data=combined_data,
                        dtype="float32",
                        compression="gzip",
                        compression_opts=4,
                    )

                    total_samples += len(combined_data)
                    print(
                        f"[OK] 已保存到HDF5: {dataset_name}, 样本数: {len(combined_data)}"
                    )

                    # 5. 释放内存
                    del combined_data, df, warmup_df

        print(f"\n{'='*60}")
        print(f"数据处理完成！")
        print(f"总样本数: {total_samples}")
        print(f"输出文件: {output_file}")
        print(f"{'='*60}\n")


def main():
    """主函数：下载并准备训练数据"""
    # 配置
    symbols = ["BTCUSDT", "DOGEUSDT", "ETHUSDT"]
    train_years = [2022, 2023, 2024]

    # 初始化
    data_loader = BinanceDataLoader(data_dir="./data")
    preparator = DatasetPreparator(window_size=120, threshold=0.005, horizon=30)

    # 准备训练数据并保存到HDF5
    preparator.prepare_multiple_symbols(
        data_loader,
        symbols,
        train_years,
        output_file="./data/all_data.h5",
        warmup_size=200,
    )

    print("\n完成！可以使用以下代码读取数据：")
    print("```python")
    print("import h5py")
    print("with h5py.File('./data/all_data.h5', 'r') as f:")
    print("    data = f['BTCUSDT_2023'][:]  # 读取特定dataset")
    print("    features = data[:, :, :-1]    # 前10列是特征")
    print("    labels = data[:, 0, -1]       # 最后一列是label")
    print("```")


if __name__ == "__main__":
    main()
