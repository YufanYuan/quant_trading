import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
import pandas as pd
from datetime import datetime
from .backtrader_adapter import BacktraderAdapter
from market_data import get_binance_data


def run_backtest(
        symbol: str = "BTCUSDT",
        start_date: datetime = datetime(2025, 1, 1, 0, 0),
        end_date: datetime = datetime(2025, 7, 31, 23, 59),
):
    """运行回测"""
    print("开始回测...")
    
    # 获取数据
    interval = "1m"
    
    print(f"获取 {symbol} {interval} 数据，时间范围：{start_date} - {end_date}")
    data = get_binance_data(symbol, start_date, end_date, interval)
    
    if data.empty:
        print("未获取到数据，退出")
        return
    
    print(f"数据获取完成，共 {len(data)} 条记录")
    
    # 准备backtrader数据
    data['datetime'] = pd.to_datetime(data['open_time'], unit='ms')
    data.set_index('datetime', inplace=True)
    data = data[['open', 'high', 'low', 'close', 'volume']]
    
    # 创建backtrader数据源
    bt_data = bt.feeds.PandasData(dataname=data)
    
    # 初始化Cerebro引擎
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(
        BacktraderAdapter,
        enable_session_filter=False,
        debug=True,  # 设置为True可查看详细日志
        initial_balance=10000,
        max_positions=1,
        position_size=1.0,
        fee_rate=0.0
    )
    
    # 添加数据
    cerebro.adddata(bt_data)
    
    # 设置初始资金
    cerebro.broker.setcash(10000.0)
    
    # 设置手续费为0
    cerebro.broker.setcommission(commission=0.0)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")  # System Quality Number
    
    print("开始策略执行...")
    print(f"数据时间范围: {data.index[0]} 至 {data.index[-1]}")
    print(f"总数据点: {len(data):,} 条")
    initial_value = cerebro.broker.getvalue()
    
    # 运行回测
    print("正在执行回测...")
    print("注意: 1分钟数据量较大，请耐心等待...")
    
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    
    print("回测执行完成!")
    
    # 获取结果
    strategy = results[0]
    
    # 打印回测结果
    print("\n" + "="*60)
    print("BACKTRADER 回测结果:")
    print("="*60)
    print(f"初始资金: {initial_value:.2f}")
    print(f"最终资金: {final_value:.2f}")
    print(f"总收益: {final_value - initial_value:.2f}")
    print(f"总收益率: {((final_value - initial_value) / initial_value * 100):.2f}%")
    
    # 交易分析
    trades_analyzer = strategy.analyzers.trades.get_analysis()
    if trades_analyzer and hasattr(trades_analyzer, 'total') and trades_analyzer.total.total > 0:
        print(f"\n交易统计:")
        print(f"总交易次数: {trades_analyzer.total.total}")
        print(f"盈利交易: {trades_analyzer.won.total}")
        print(f"亏损交易: {trades_analyzer.lost.total}")
        print(f"胜率: {(trades_analyzer.won.total / trades_analyzer.total.total * 100):.2f}%")
        
        if hasattr(trades_analyzer.won, 'pnl') and trades_analyzer.won.total > 0:
            print(f"平均盈利: {trades_analyzer.won.pnl.average:.4f}")
        if hasattr(trades_analyzer.lost, 'pnl') and trades_analyzer.lost.total > 0:
            print(f"平均亏损: {trades_analyzer.lost.pnl.average:.4f}")
    else:
        print("\n没有完成的交易")
    
    # 其他分析指标
    try:
        sharpe_analyzer = strategy.analyzers.sharpe.get_analysis()
        if sharpe_analyzer and 'sharperatio' in sharpe_analyzer:
            sharpe_ratio = sharpe_analyzer['sharperatio']
            if sharpe_ratio is not None:
                print(f"夏普比率: {sharpe_ratio:.4f}")
    except:
        pass
    
    try:
        drawdown_analyzer = strategy.analyzers.drawdown.get_analysis()
        if drawdown_analyzer and 'max' in drawdown_analyzer:
            print(f"最大回撤: {drawdown_analyzer['max']['drawdown']:.2f}%")
            print(f"最大回撤金额: {drawdown_analyzer['max']['moneydown']:.2f}")
    except:
        pass
    
    try:
        sqn_analyzer = strategy.analyzers.sqn.get_analysis()
        if sqn_analyzer and 'sqn' in sqn_analyzer:
            sqn_value = sqn_analyzer['sqn']
            if sqn_value is not None:
                print(f"系统质量数(SQN): {sqn_value:.4f}")
    except:
        pass
    
    # 获取核心执行器的详细指标
    core_metrics = strategy.core_executor.get_metrics()
    print(f"\n核心执行器统计:")
    print(f"信号生成数: {core_metrics['total_signals']}")
    print(f"信号执行数: {core_metrics['executed_signals']}")
    print(f"信号拒绝数: {core_metrics['rejected_signals']}")
    print(f"信号执行率: {(core_metrics['executed_signals'] / core_metrics['total_signals'] * 100) if core_metrics['total_signals'] > 0 else 0:.1f}%")
    
    # 保存交易记录
    if strategy.core_executor.closed_trades:
        trades_data = []
        for trade in strategy.core_executor.closed_trades:
            trades_data.append({
                'signal_id': trade.signal_id,
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'exit_reason': trade.exit_reason,
                'pnl': trade.pnl,
                'size': trade.size
            })
        
        trades_df = pd.DataFrame(trades_data)
        output_file = f'doji_scalp_trades_{symbol}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
        trades_df.to_csv(output_file, index=False)
        print(f"\n交易记录已保存到: {output_file}")
        
        # 显示前几笔交易
        if len(trades_df) > 0:
            print(f"\n前5笔交易:")
            print(trades_df[['signal_id', 'direction', 'entry_price', 'exit_price', 'exit_reason', 'pnl']].head().to_string(index=False))


if __name__ == "__main__":
    run_backtest()