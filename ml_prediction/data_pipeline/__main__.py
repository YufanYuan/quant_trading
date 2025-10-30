"""
数据处理流水线主入口

使用方法:
    python -m ml_prediction.data_pipeline
"""

from .data_source import BinanceDataLoader
from .preprocessor import DatasetPreparator
from .storage_manager import ShardedHDF5Writer, verify_shards


def main():
    """主函数：下载并准备训练数据，直接写入分片文件"""

    # 配置
    symbols = ["BTCUSDT", "DOGEUSDT", "ETHUSDT"]
    train_years = [2022, 2023, 2024]

    # 数据处理参数
    window_size = 120
    threshold = 0.005
    horizon = 30
    warmup_size = 200

    # 存储参数
    output_dir = "./data/sharded"
    num_shards = 64
    chunk_size = 128
    batch_size = 512

    print("=" * 70)
    print("数据处理流水线 - 直接生成分片文件")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  交易对: {', '.join(symbols)}")
    print(f"  年份: {', '.join(map(str, train_years))}")
    print(f"  窗口大小: {window_size}")
    print(f"  标签阈值: {threshold}")
    print(f"  预测窗口: {horizon}")
    print(f"  Warm-up大小: {warmup_size}")
    print(f"\n存储配置:")
    print(f"  输出目录: {output_dir}")
    print(f"  分片数量: {num_shards}")
    print(f"  Chunk大小: {chunk_size}")
    print(f"  Batch对齐: {batch_size}")
    print("=" * 70)

    # 初始化组件
    data_loader = BinanceDataLoader(data_dir="./data")
    preparator = DatasetPreparator(
        window_size=window_size,
        threshold=threshold,
        horizon=horizon
    )

    # 创建存储管理器
    feature_names = "open_logret,close_logret,high_logret,low_logret,ema5_logret,ema12_logret,ema26_logret,ema50_logret,ema200_logret,rsi,label"

    with ShardedHDF5Writer(
        output_dir=output_dir,
        num_shards=num_shards,
        batch_size=batch_size,
        chunk_size=chunk_size,
        window_size=window_size,
        feature_names=feature_names,
        metadata={
            "threshold": threshold,
            "horizon": horizon,
        }
    ) as storage:

        # 处理每个交易对的每一年数据
        for symbol in symbols:
            print(f"\n{'='*70}")
            print(f"处理交易对: {symbol}")
            print(f"{'='*70}")

            for year in train_years:
                print(f"\n--- 处理 {symbol} {year}年数据 ---")

                # 1. 下载/加载当前年份数据
                df = data_loader.download_year_data(symbol, year)
                if df is None or df.empty:
                    print(f"  跳过 {symbol} {year}年（无数据）")
                    continue

                # 2. 尝试加载warm-up数据
                warmup_df = data_loader.load_warmup_data(symbol, year, warmup_size)

                # 3. 准备数据集（特征工程 + 标签生成）
                combined_data = preparator.prepare_dataset(df, warmup_df)

                if len(combined_data) == 0:
                    print(f"  跳过 {symbol} {year}年（处理后无有效数据）")
                    continue

                # 4. 直接写入分片存储（按round-robin策略分配）
                storage.add_dataset(symbol, year, combined_data)

                # 5. 释放内存
                del combined_data, df, warmup_df

        # 6. 完成写入（shuffle + 对齐 + 关闭文件）
        print(f"\n{'='*70}")
        print("完成所有数据处理，正在finalize...")
        print(f"{'='*70}")

    # 验证生成的文件
    print(f"\n{'='*70}")
    print("验证分片文件")
    print(f"{'='*70}")
    verify_shards(output_dir)

    print(f"\n{'='*70}")
    print("✓ 数据处理完成！")
    print(f"{'='*70}")
    print(f"\n输出目录: {output_dir}")
    print(f"可以使用以下命令开始训练:")
    print(f"  python -m ml_prediction.train")
    print()


if __name__ == "__main__":
    main()
