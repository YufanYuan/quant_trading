import backtrader as bt
from datetime import datetime
from .strategy import DojiScalpSignalGenerator
from .strategy_executor import CoreExecutor, FrameworkAdapter, Position


class BacktraderAdapter(bt.Strategy):
    """Backtrader框架适配器"""
    
    params = (
        ('enable_session_filter', False),
        ('debug', True),
        ('initial_balance', 10000),
        ('max_positions', 1),
        ('position_size', 1.0),
        ('fee_rate', 0.0),
    )
    
    def __init__(self):
        # 初始化核心执行器
        self.core_executor = CoreExecutor(
            initial_balance=self.params.initial_balance,
            max_positions=self.params.max_positions,
            position_size=self.params.position_size,
            fee_rate=self.params.fee_rate,
        )
        
        # 初始化信号生成器
        self.signal_generator = DojiScalpSignalGenerator(
            enable_session_filter=self.params.enable_session_filter
        )
        
        # Backtrader订单管理
        self.pending_orders = {}  # signal_id -> order list
        self.position_orders = {}  # signal_id -> [main_order, stop_order, limit_order]
        self.trade_to_position = {}  # tradeid -> position
        self.processed_orders = set()  # 已处理的订单ref，避免重复处理
        
        # 进度追踪
        self.bar_count = 0
        self.last_progress_report = 0
        self.progress_interval = 500  # 每50000条数据报告一次进度
    
    def process_signal(self, signal, current_time):
        """处理信号的通用流程"""
        position = self.core_executor.process_signal(signal, current_time)
        if position:
            success = self.execute_position(position)
            if success:
                self.log(f"执行信号: {signal.signal_id} - {signal.direction} @ {signal.entry_price:.4f}")
                return True
            else:
                # 如果执行失败，从持仓列表中移除
                self.core_executor.positions.remove(position)
                self.log(f"信号执行失败: {signal.signal_id}")
                return False
        else:
            open_positions = self.core_executor.get_open_positions()
            self.log(f"信号被拒绝: {signal.signal_id}, 当前开仓数: {len(open_positions)}")
            if open_positions:
                for pos in open_positions:
                    self.log(f"  - 开仓位置: {pos.signal_id}, 方向: {pos.direction}, 状态: {pos.status}")
            return False
    
    def update_positions(self, timestamp, high, low, close):
        """更新持仓状态（OCO订单会自动处理平仓）"""
        # 在使用OCO订单的情况下，平仓由backtrader自动处理
        # 这里只需要保持接口兼容性，实际平仓逻辑在notify_trade中处理
        pass
    
    def log(self, message: str, dt=None):
        """日志输出"""
        if self.params.debug:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {message}')
    
    def execute_position(self, position: Position) -> bool:
        """执行开仓（使用bracket orders）"""
        try:
            current_price = self.data.close[0]
            
            # 计算全仓买入数量：使用当前可用资金除以当前价格
            available_cash = self.broker.getcash()
            if position.direction == 'long':
                position.size = available_cash / current_price * 0.99  # 预留1%缓冲
            else:  # short
                position.size = available_cash / current_price * 0.99  # 预留1%缓冲
                
            self.log(f"开仓详情: {position.direction} @ entry={position.entry_price:.4f}, "
                    f"stop={position.stop_loss:.4f}, target={position.take_profit:.4f}, "
                    f"current={current_price:.4f}, size={position.size:.6f}, cash={available_cash:.2f}")
            
            if position.direction == 'long':
                # 做多：使用buy_bracket创建OCO订单组
                main_order, stop_order, limit_order = self.buy_bracket(
                    size=position.size,
                    price=None,  # 市价单
                    stopprice=position.stop_loss,
                    limitprice=position.take_profit,
                    exectype=bt.Order.Market
                )
                self.log(f"创建OCO订单组: BUY市价 {position.size:.6f}, STOP @ {position.stop_loss:.4f}, LIMIT @ {position.take_profit:.4f}")
                
            else:  # short
                # 做空：使用sell_bracket创建OCO订单组  
                main_order, stop_order, limit_order = self.sell_bracket(
                    size=position.size,
                    price=None,  # 市价单
                    stopprice=position.stop_loss,
                    limitprice=position.take_profit,
                    exectype=bt.Order.Market
                )
                self.log(f"创建OCO订单组: SELL市价 {position.size:.6f}, STOP @ {position.stop_loss:.4f}, LIMIT @ {position.take_profit:.4f}")
            
            # 记录订单和位置的映射关系
            self.position_orders[position.signal_id] = [main_order, stop_order, limit_order]
            return True
            
        except Exception as e:
            self.log(f"开仓失败: {e}")
            return False
    
    def close_position(self, position: Position) -> bool:
        """执行平仓（在backtrader中通常由止盈止损订单自动执行）"""
        # 在backtrader中，平仓通常由止盈止损订单自动执行
        # 这里主要是清理订单记录
        if position.signal_id in self.position_orders:
            orders = self.position_orders[position.signal_id]
            for order in orders:
                if order.status not in [bt.Order.Completed, bt.Order.Cancelled, bt.Order.Rejected]:
                    self.cancel(order)
            del self.position_orders[position.signal_id]
        return True
    
    def notify_order(self, order):
        """订单状态通知"""
        # 避免重复处理同一订单的同一状态
        order_key = (order.ref, order.status)
        if order_key in self.processed_orders:
            return
        self.processed_orders.add(order_key)
        
        # 记录所有订单状态变化
        status_name = {
            order.Created: 'Created',
            order.Submitted: 'Submitted', 
            order.Accepted: 'Accepted',
            order.Partial: 'Partial',
            order.Completed: 'Completed',
            order.Cancelled: 'Cancelled',
            order.Expired: 'Expired',
            order.Margin: 'Margin',
            order.Rejected: 'Rejected'
        }.get(order.status, f'Unknown({order.status})')
        
        order_type = "BUY" if order.isbuy() else "SELL"
        exec_type = {
            order.Market: 'Market',
            order.Limit: 'Limit', 
            order.Stop: 'Stop',
            order.StopLimit: 'StopLimit'
        }.get(order.exectype, f'Unknown({order.exectype})')
        
        self.log(f'订单状态: {order_type} {exec_type} ref={order.ref} status={status_name} price={order.price} size={order.size}')
        
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: 价格 {order.executed.price:.4f}, 数量 {order.executed.size:.2f}')
            else:
                self.log(f'卖出执行: 价格 {order.executed.price:.4f}, 数量 {order.executed.size:.2f}')
            
            # 如果是主订单完成，建立trade到position的映射
            for signal_id, orders in self.position_orders.items():
                if order in orders and orders[0] == order:  # 主订单（第一个订单）
                    # 找到对应的position
                    for position in self.core_executor.positions:
                        if position.signal_id == signal_id:
                            # 更新position的实际成交价格
                            position.entry_price = order.executed.price
                            self.trade_to_position[order.tradeid] = position
                            self.log(f'建立映射: tradeid={order.tradeid} -> position={position.signal_id}')
                            break
                    break
        
        elif order.status in [order.Cancelled, order.Margin, order.Rejected]:
            self.log(f'订单{status_name}: {order_type} {exec_type} ref={order.ref}')
            
            # 检查是否是主订单被拒绝，如果是则清理position
            # 但要排除已经完成过的订单（完成后又被取消是正常的）
            for signal_id, orders in list(self.position_orders.items()):
                if order in orders:
                    # 找到对应的position
                    position_to_remove = None
                    for position in self.core_executor.positions:
                        if position.signal_id == signal_id:
                            position_to_remove = position
                            break
                    
                    # 只有主订单且没有完成过才清理position
                    order_completed_key = (order.ref, order.Completed)
                    if (orders[0] == order and position_to_remove and 
                        order_completed_key not in self.processed_orders):
                        self.log(f'主订单被{status_name}（未完成），清理position: {signal_id}')
                        self.core_executor.positions.remove(position_to_remove)
                        del self.position_orders[signal_id]
                    elif orders[0] == order and order_completed_key in self.processed_orders:
                        self.log(f'主订单被{status_name}（已完成过），忽略清理')
                    break
    
    def notify_trade(self, trade):
        """交易通知 - 监控交易状态并同步position"""
        self.log(f'交易通知: tradeid={trade.tradeid}, size={trade.size}, price={trade.price:.4f}, '
                f'open={trade.isopen}, closed={trade.isclosed}, pnl={trade.pnlcomm:.4f}')
        
        # 只在交易完全关闭时更新position状态
        if trade.isclosed and trade.tradeid in self.trade_to_position:
            position = self.trade_to_position[trade.tradeid]
            
            # 更新position状态为已平仓
            position.status = 'closed'
            
            # 获取最后一次成交的价格作为平仓价格
            # trade.price是加权平均成交价，不一定是平仓价
            # 我们需要根据止盈止损逻辑来确定实际的平仓价格
            exit_price = trade.price  # 默认使用成交价
            
            # 根据价格判断退出原因，并使用相应的止盈止损价格
            if position.direction == 'long':
                if abs(trade.price - position.take_profit) < abs(trade.price - position.stop_loss):
                    position.exit_reason = 'take_profit'
                    exit_price = position.take_profit
                else:
                    position.exit_reason = 'stop_loss'
                    exit_price = position.stop_loss
            else:  # short
                if abs(trade.price - position.take_profit) < abs(trade.price - position.stop_loss):
                    position.exit_reason = 'take_profit'
                    exit_price = position.take_profit
                else:
                    position.exit_reason = 'stop_loss'
                    exit_price = position.stop_loss
            
            position.exit_price = exit_price
            position.exit_time = self.datas[0].datetime.datetime(0)
            position.pnl = trade.pnlcomm
            
            # 移动到已平仓列表
            self.core_executor.positions.remove(position)
            self.core_executor.closed_trades.append(position)
            
            # 清理映射
            if position.signal_id in self.position_orders:
                del self.position_orders[position.signal_id]
            del self.trade_to_position[trade.tradeid]
            
            self.log(f'Position平仓完成: {position.signal_id} - {position.exit_reason} @ {position.exit_price:.4f}, PnL: {position.pnl:.4f}')
    
    def next(self):
        """主要策略逻辑"""
        self.bar_count += 1
        
        # 进度报告
        if self.bar_count - self.last_progress_report >= self.progress_interval:
            metrics = self.core_executor.get_metrics()
            print(f"进度: {self.bar_count:,} 条数据已处理 | 信号: {metrics['total_signals']} | 完成交易: {metrics['total_trades']}")
            self.last_progress_report = self.bar_count
        
        # 获取当前K线数据
        current_time = self.datas[0].datetime.datetime(0)
        open_price = self.data.open[0]
        high = self.data.high[0]
        low = self.data.low[0]
        close = self.data.close[0]
        volume = self.data.volume[0]
        
        # 更新持仓状态（检查止盈止损）
        self.update_positions(current_time, high, low, close)
        
        # 生成交易信号
        signal = self.signal_generator.on_kline(
            current_time, open_price, high, low, close, volume
        )
        
        # 处理信号
        if signal:
            self.process_signal(signal, current_time)
    
    def stop(self):
        """策略结束时的清理"""
        self.log(f'期末资金: {self.broker.getvalue():.2f}')
        
        # 打印策略指标
        metrics = self.core_executor.get_metrics()
        self.log("="*50)
        self.log("策略执行统计:")
        self.log(f"总信号数: {metrics['total_signals']}")
        self.log(f"执行信号数: {metrics['executed_signals']}")
        self.log(f"拒绝信号数: {metrics['rejected_signals']}")
        self.log(f"完成交易数: {metrics['total_trades']}")
        self.log(f"胜率: {metrics['win_rate']:.2%}")
        if metrics['total_trades'] > 0:
            self.log(f"平均盈利: {metrics['avg_win']:.4f}")
            self.log(f"平均亏损: {metrics['avg_loss']:.4f}")
            self.log(f"盈亏比: {metrics['profit_factor']:.2f}")
        self.log("="*50)