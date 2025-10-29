import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
from .strategy import TradingSignal


@dataclass
class Position:
    """持仓信息"""
    signal_id: str
    direction: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    size: float = 1.0
    status: str = 'open'  # 'open', 'closed'
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None  # 'stop_loss', 'take_profit'
    pnl: float = 0.0


class BrokerInterface(Protocol):
    """券商接口协议"""
    
    def submit_order(self, signal: TradingSignal) -> bool:
        """提交订单"""
        ...
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        ...
    
    def get_current_price(self) -> Dict[str, float]:
        """获取当前价格（high, low, close）"""
        ...


class CoreExecutor:
    """核心执行器，处理通用的头寸管理和交易逻辑"""
    
    def __init__(
        self,
        initial_balance: float = 10000,
        max_positions: int = 1,
        position_size: float = 1.0,
        fee_rate: float = 0.0,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_positions = max_positions
        self.position_size = position_size
        self.fee_rate = fee_rate
        
        # 持仓和交易记录
        self.positions: List[Position] = []
        self.closed_trades: List[Position] = []
        self.signal_history: List[TradingSignal] = []
        
        # 统计
        self.total_signals = 0
        self.executed_signals = 0
        self.rejected_signals = 0
    
    def get_open_positions(self) -> List[Position]:
        """获取当前开仓头寸"""
        return [p for p in self.positions if p.status == 'open']
    
    def can_open_position(self, signal: TradingSignal) -> bool:
        """检查是否可以开仓"""
        open_positions = self.get_open_positions()
        
        # 检查最大持仓数限制
        if len(open_positions) >= self.max_positions:
            return False
        
        # 检查是否已有相同方向的持仓
        for pos in open_positions:
            if pos.direction == signal.direction:
                return False  # 同方向已有持仓
        
        return True
    
    def process_signal(self, signal: TradingSignal, current_time: datetime) -> Optional[Position]:
        """处理信号，返回需要执行的头寸（如果有的话）"""
        self.total_signals += 1
        self.signal_history.append(signal)
        
        if not self.can_open_position(signal):
            self.rejected_signals += 1
            return None
        
        # 创建新持仓
        position = Position(
            signal_id=signal.signal_id,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_time=current_time,
            size=self.position_size
        )
        
        self.positions.append(position)
        self.executed_signals += 1
        
        return position
    
    def update_positions(self, timestamp: datetime, high: float, low: float, close: float) -> List[Position]:
        """更新持仓状态，返回需要平仓的头寸列表"""
        positions_to_close = []
        
        for position in self.get_open_positions():
            exit_price = None
            exit_reason = None
            
            if position.direction == 'long':
                # 多头：检查止损和止盈
                if low <= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = 'stop_loss'
                elif high >= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = 'take_profit'
            
            elif position.direction == 'short':
                # 空头：检查止损和止盈
                if high >= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = 'stop_loss'
                elif low <= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = 'take_profit'
            
            if exit_price is not None:
                closed_position = self._close_position(position, exit_price, exit_reason, timestamp)
                positions_to_close.append(closed_position)
        
        return positions_to_close
    
    def _close_position(self, position: Position, exit_price: float, exit_reason: str, timestamp: datetime) -> Position:
        """平仓"""
        position.exit_price = exit_price
        position.exit_time = timestamp
        position.exit_reason = exit_reason
        position.status = 'closed'
        
        # 计算盈亏
        if position.direction == 'long':
            position.pnl = (exit_price - position.entry_price) * position.size
        else:  # short
            position.pnl = (position.entry_price - exit_price) * position.size
        
        # 扣除手续费
        entry_fee = position.entry_price * position.size * self.fee_rate
        exit_fee = exit_price * position.size * self.fee_rate
        position.pnl -= (entry_fee + exit_fee)
        
        # 更新余额
        self.balance += position.pnl
        
        # 移动到已平仓列表
        self.positions.remove(position)
        self.closed_trades.append(position)
        
        return position
    
    def get_metrics(self) -> Dict[str, Any]:
        """计算策略指标"""
        if not self.closed_trades:
            return {
                'total_signals': self.total_signals,
                'executed_signals': self.executed_signals,
                'rejected_signals': self.rejected_signals,
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'current_balance': self.balance
            }
        
        df = pd.DataFrame([
            {
                'signal_id': t.signal_id,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'exit_reason': t.exit_reason,
                'pnl': t.pnl
            }
            for t in self.closed_trades
        ])
        
        total_trades = len(df)
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        total_pnl = df['pnl'].sum()
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # 计算最大回撤
        if len(df) > 0:
            df = df.sort_values('exit_time')
            df['cumulative_pnl'] = df['pnl'].cumsum()
            df['running_max'] = df['cumulative_pnl'].expanding().max()
            df['drawdown'] = df['cumulative_pnl'] - df['running_max']
            max_drawdown = abs(df['drawdown'].min())
        else:
            max_drawdown = 0
        
        return {
            'total_signals': self.total_signals,
            'executed_signals': self.executed_signals,
            'rejected_signals': self.rejected_signals,
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'max_drawdown': max_drawdown,
            'current_balance': self.balance
        }


class FrameworkAdapter(ABC):
    """框架适配器抽象基类"""
    
    def __init__(self, core_executor: CoreExecutor):
        self.core_executor = core_executor
    
    @abstractmethod
    def execute_position(self, position: Position) -> bool:
        """执行开仓"""
        pass
    
    @abstractmethod
    def close_position(self, position: Position) -> bool:
        """执行平仓"""
        pass
    
    @abstractmethod
    def log(self, message: str):
        """日志输出"""
        pass
    
    def process_signal(self, signal: TradingSignal, current_time: datetime) -> bool:
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
            self.log(f"信号被拒绝: {signal.signal_id}")
            return False
    
    def update_positions(self, timestamp: datetime, high: float, low: float, close: float):
        """更新持仓的通用流程"""
        positions_to_close = self.core_executor.update_positions(timestamp, high, low, close)
        for position in positions_to_close:
            self.close_position(position)
            self.log(f"平仓: {position.signal_id} - {position.exit_reason} @ {position.exit_price:.4f}, PnL: {position.pnl:.4f}")