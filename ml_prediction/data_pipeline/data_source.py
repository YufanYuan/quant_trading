"""
数据源模块

负责从Binance下载分钟级K线数据
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime
from typing import Optional

from market_data import get_binance_data


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
