"""
回测脚本

使用2025年数据对训练好的策略进行回测
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import json

from data_loader import BinanceDataLoader
from strategy import MLPredictionStrategy


class Trade:
    """交易记录"""

    def __init__(
        self,
        entry_time: datetime,
        entry_price: float,
        direction: str,  # 'long' or 'short'
        stop_loss: float,
        take_profit: float
    ):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0.0
        self.pnl_pct = 0.0

    def close(self, exit_time: datetime, exit_price: float, exit_reason: str):
        """平仓"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason

        if self.direction == 'long':
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price

        self.pnl = self.pnl_pct

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'direction': self.direction,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct
        }


class Backtester:
    """回测引擎"""

    def __init__(
        self,
        strategy: MLPredictionStrategy,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.0004  # 0.04% 手续费
    ):
        """
        Args:
            strategy: 策略实例
            initial_capital: 初始资金
            fee_rate: 手续费率
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate

        self.trades: List[Trade] = []
        self.current_trade = None
        self.equity_curve = []

    def run(self, df: pd.DataFrame):
        """
        运行回测

        Args:
            df: 包含OHLC数据的DataFrame
        """
        print(f"开始回测，数据量: {len(df)}")

        window_size = self.strategy.window_size
        capital = self.initial_capital

        for i in range(window_size, len(df)):
            # 获取当前窗口数据
            window_df = df.iloc[i - window_size:i].reset_index(drop=True)
            current_row = df.iloc[i]
            current_price = current_row['close']
            current_time = current_row['timestamp']

            # 如果有持仓，检查止损止盈
            if self.current_trade is not None:
                trade = self.current_trade

                if trade.direction == 'long':
                    if current_price >= trade.take_profit:
                        # 止盈
                        fee = current_price * self.fee_rate
                        trade.close(current_time, current_price - fee, 'take_profit')
                        capital *= (1 + trade.pnl)
                        self.trades.append(trade)
                        self.current_trade = None

                    elif current_price <= trade.stop_loss:
                        # 止损
                        fee = current_price * self.fee_rate
                        trade.close(current_time, current_price - fee, 'stop_loss')
                        capital *= (1 + trade.pnl)
                        self.trades.append(trade)
                        self.current_trade = None

                elif trade.direction == 'short':
                    if current_price <= trade.take_profit:
                        # 止盈
                        fee = current_price * self.fee_rate
                        trade.close(current_time, current_price + fee, 'take_profit')
                        capital *= (1 + trade.pnl)
                        self.trades.append(trade)
                        self.current_trade = None

                    elif current_price >= trade.stop_loss:
                        # 止损
                        fee = current_price * self.fee_rate
                        trade.close(current_time, current_price + fee, 'stop_loss')
                        capital *= (1 + trade.pnl)
                        self.trades.append(trade)
                        self.current_trade = None

            # 如果没有持仓，生成新信号
            if self.current_trade is None:
                signal = self.strategy.generate_signal(window_df, current_price)

                if signal is not None:
                    if signal['action'] == 'open_long':
                        fee = current_price * self.fee_rate
                        self.current_trade = Trade(
                            entry_time=current_time,
                            entry_price=current_price + fee,
                            direction='long',
                            stop_loss=signal['stop_loss'],
                            take_profit=signal['take_profit']
                        )

                    elif signal['action'] == 'open_short':
                        fee = current_price * self.fee_rate
                        self.current_trade = Trade(
                            entry_time=current_time,
                            entry_price=current_price - fee,
                            direction='short',
                            stop_loss=signal['stop_loss'],
                            take_profit=signal['take_profit']
                        )

            # 记录权益曲线
            self.equity_curve.append({
                'timestamp': current_time,
                'capital': capital
            })

        # 如果回测结束时还有持仓，强制平仓
        if self.current_trade is not None:
            final_row = df.iloc[-1]
            final_price = final_row['close']
            final_time = final_row['timestamp']

            fee = final_price * self.fee_rate
            exit_price = final_price - fee if self.current_trade.direction == 'long' else final_price + fee

            self.current_trade.close(final_time, exit_price, 'forced_close')
            capital *= (1 + self.current_trade.pnl)
            self.trades.append(self.current_trade)
            self.current_trade = None

        print(f"回测完成，共执行 {len(self.trades)} 笔交易")

        return capital

    def get_statistics(self) -> Dict[str, Any]:
        """计算回测统计指标"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }

        # 基本统计
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        # 收益统计
        pnls = [t.pnl for t in self.trades]
        avg_pnl = np.mean(pnls)
        total_pnl = np.sum(pnls)

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        # 最大回撤
        equity = [self.initial_capital]
        for trade in self.trades:
            equity.append(equity[-1] * (1 + trade.pnl))

        max_equity = equity[0]
        max_drawdown = 0.0
        for e in equity:
            if e > max_equity:
                max_equity = e
            drawdown = (max_equity - e) / max_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # 夏普比率（假设无风险利率为0）
        if len(pnls) > 1:
            sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)) if np.std(pnls) > 0 else 0
        else:
            sharpe_ratio = 0

        # 最终资金
        final_capital = self.initial_capital * (1 + total_pnl)

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'profit': final_capital - self.initial_capital
        }

    def save_results(self, output_dir: str = './results'):
        """保存回测结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存交易记录
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        trades_df.to_csv(output_path / 'trades.csv', index=False)

        # 保存权益曲线
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv(output_path / 'equity_curve.csv', index=False)

        # 保存统计信息
        stats = self.get_statistics()
        with open(output_path / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        print(f"回测结果已保存到 {output_path}")


def main():
    """主回测函数"""
    # 配置
    symbol = 'BTCUSDT'
    test_year = 2025

    # 加载数据
    print("=" * 60)
    print("加载回测数据")
    print("=" * 60)

    data_loader = BinanceDataLoader(data_dir='./data')
    df = data_loader.download_year_data(symbol, test_year)

    if df is None or df.empty:
        print(f"无法加载 {symbol} {test_year}年数据")
        return

    print(f"数据加载完成，共 {len(df)} 条")

    # 初始化策略
    print("\n" + "=" * 60)
    print("初始化策略")
    print("=" * 60)

    strategy = MLPredictionStrategy(
        model_path='best_model.pth',
        config_path='model_config.json'
    )

    # 运行回测
    print("\n" + "=" * 60)
    print("运行回测")
    print("=" * 60)

    backtester = Backtester(
        strategy=strategy,
        initial_capital=10000.0,
        fee_rate=0.0004
    )

    final_capital = backtester.run(df)

    # 显示结果
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)

    stats = backtester.get_statistics()

    print(f"\n总交易次数: {stats['total_trades']}")
    print(f"胜率: {stats['win_rate']*100:.2f}%")
    print(f"盈利交易: {stats['winning_trades']}")
    print(f"亏损交易: {stats['losing_trades']}")
    print(f"\n平均盈亏: {stats['avg_pnl']*100:.2f}%")
    print(f"平均盈利: {stats['avg_win']*100:.2f}%")
    print(f"平均亏损: {stats['avg_loss']*100:.2f}%")
    print(f"\n总收益率: {stats['total_pnl_pct']:.2f}%")
    print(f"最大回撤: {stats['max_drawdown_pct']:.2f}%")
    print(f"夏普比率: {stats['sharpe_ratio']:.2f}")
    print(f"\n初始资金: ${stats['initial_capital']:.2f}")
    print(f"最终资金: ${stats['final_capital']:.2f}")
    print(f"净利润: ${stats['profit']:.2f}")

    # 保存结果
    backtester.save_results('./results')


if __name__ == '__main__':
    main()
