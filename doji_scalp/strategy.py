import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Optional, Callable, Dict, Any, NamedTuple
from collections import deque
from enum import Enum


class StrategyState(Enum):
    """策略状态枚举"""
    FINDING_STRUCTURE = "finding_structure"  # 寻找市场结构（回调+修复）
    FINDING_ENTRY_SIGNAL = "finding_entry_signal"  # 寻找入场信号


class TradingSignal(NamedTuple):
    """交易信号"""
    direction: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    signal_id: str


class HeikinAshiCalculator:
    """Heikin Ashi计算器"""
    
    def __init__(self):
        self.prev_ha_open = None
        self.prev_ha_close = None
    
    def calculate(self, open_price: float, high: float, low: float, close: float) -> Dict[str, float]:
        """
        计算Heikin Ashi OHLC
        
        Returns:
            Dict containing ha_open, ha_high, ha_low, ha_close
        """
        # HA_Close = (O + H + L + C) / 4
        ha_close = (open_price + high + low + close) / 4
        
        # HA_Open = (prev_HA_O + prev_HA_C) / 2 (第一根用原始Open)
        if self.prev_ha_open is None or self.prev_ha_close is None:
            ha_open = open_price
        else:
            ha_open = (self.prev_ha_open + self.prev_ha_close) / 2
        
        # HA_High = max(H, HA_O, HA_C)
        ha_high = max(high, ha_open, ha_close)
        
        # HA_Low = min(L, HA_O, HA_C)
        ha_low = min(low, ha_open, ha_close)
        
        # 更新前一根数据
        self.prev_ha_open = ha_open
        self.prev_ha_close = ha_close
        
        return {
            'ha_open': ha_open,
            'ha_high': ha_high,
            'ha_low': ha_low,
            'ha_close': ha_close
        }


class EMACalculator:
    """EMA指标计算器"""
    
    def __init__(self, period: int = 100):
        self.period = period
        self.alpha = 2 / (period + 1)
        self.ema = None
    
    def update(self, price: float) -> float:
        """更新EMA值"""
        if self.ema is None:
            self.ema = price
        else:
            self.ema = self.alpha * price + (1 - self.alpha) * self.ema
        return self.ema


class DojiScalpSignalGenerator:
    """1分钟剥头皮信号生成器"""
    
    def __init__(
        self,
        enable_session_filter: bool = False,
        session_start: time = time(9, 0),
        session_end: time = time(11, 0),
    ):
        # 指标计算器
        self.ha_calc = HeikinAshiCalculator()
        self.ema_calc = EMACalculator(100)
        
        # 策略参数
        self.enable_session_filter = enable_session_filter
        self.session_start = session_start
        self.session_end = session_end
        
        # 状态变量
        self.state = StrategyState.FINDING_STRUCTURE
        self.trend_direction = None  # 'up' or 'down' or None
        self.pullback_count = 0
        self.recovery_count = 0
        self.clean_counter_candles = []  # 存储连续的干净反色蜡烛
        
        # 信号计数器
        self.signal_counter = 0
    
    def _is_in_session(self, ts: datetime) -> bool:
        """检查是否在交易时段内"""
        if not self.enable_session_filter:
            return True
        current_time = ts.time()
        return self.session_start <= current_time <= self.session_end
    
    def _get_trend_direction(self, ha_data: Dict[str, float], ema: float) -> Optional[str]:
        """判断主趋势方向"""
        ha_open = ha_data['ha_open']
        ha_close = ha_data['ha_close']
        
        if ha_open > ema and ha_close > ema:
            return 'up'
        elif ha_open < ema and ha_close < ema:
            return 'down'
        else:
            return None
    
    def _is_clean_counter_candle(self, ha_data: Dict[str, float], trend: str) -> bool:
        """判断是否为干净反色蜡烛"""
        ha_open = ha_data['ha_open']
        ha_close = ha_data['ha_close']
        ha_high = ha_data['ha_high']
        ha_low = ha_data['ha_low']
        
        if trend == 'up':
            # 上升趋势中的反色蜡烛：红色（下跌）且无上影线
            is_red = ha_close < ha_open
            no_upper_shadow = ha_high == max(ha_open, ha_close)
            return is_red and no_upper_shadow
        elif trend == 'down':
            # 下降趋势中的反色蜡烛：绿色（上涨）且无下影线
            is_green = ha_close > ha_open
            no_lower_shadow = ha_low == min(ha_open, ha_close)
            return is_green and no_lower_shadow
        
        return False
    
    def _is_doji(self, ha_data: Dict[str, float]) -> bool:
        """判断是否为Doji"""
        ha_open = ha_data['ha_open']
        ha_close = ha_data['ha_close']
        ha_high = ha_data['ha_high']
        ha_low = ha_data['ha_low']
        
        if ha_high == ha_low:  # 避免除零
            return False
        
        # 实体高度占全场 <= 10%
        body_ratio = abs(ha_close - ha_open) / (ha_high - ha_low)
        if body_ratio > 0.1:
            return False
        
        # 上下影线长度比例在[0.8, 1.25]范围内
        upper_shadow = ha_high - max(ha_open, ha_close)
        lower_shadow = min(ha_open, ha_close) - ha_low
        
        if upper_shadow == 0 or lower_shadow == 0:
            return False
        
        shadow_ratio = upper_shadow / lower_shadow
        return 0.8 <= shadow_ratio <= 1.25
    
    def _check_high_volume_doji_entry(self, current_ha: Dict[str, float], volume: float) -> bool:
        """检查高量Doji入场条件"""
        if len(self.clean_counter_candles) < 2:
            return False
        
        # 当前K线必须是Doji
        if not self._is_doji(current_ha):
            return False
        
        # 取最近的两根干净反色蜡烛
        prev_two = self.clean_counter_candles[-2:]
        
        # 检查high-low差值条件
        current_range = current_ha['ha_high'] - current_ha['ha_low']
        for prev_candle in prev_two:
            prev_range = prev_candle['ha_high'] - prev_candle['ha_low']
            if current_range <= prev_range * 0.8:
                return False
        
        # 检查成交量条件
        for prev_candle in prev_two:
            if volume <= prev_candle['volume'] * 0.8:
                return False
        
        return True
    
    def _generate_signal(self, ha_data: Dict[str, float], trend: str, timestamp: datetime) -> TradingSignal:
        """生成交易信号"""
        ha_high = ha_data['ha_high']
        ha_low = ha_data['ha_low']
        range_size = ha_high - ha_low
        
        self.signal_counter += 1
        signal_id = f"doji_scalp_{self.signal_counter}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        if trend == 'up':
            # 做多信号
            entry_price = ha_high
            stop_loss = ha_low
            take_profit = ha_high + range_size  # 1:1 RR
            return TradingSignal('long', entry_price, stop_loss, take_profit, timestamp, signal_id)
            
        elif trend == 'down':
            # 做空信号
            entry_price = ha_low
            stop_loss = ha_high
            take_profit = ha_low - range_size  # 1:1 RR
            return TradingSignal('short', entry_price, stop_loss, take_profit, timestamp, signal_id)
    
    def _reset_state(self):
        """重置状态到寻找市场结构"""
        self.state = StrategyState.FINDING_STRUCTURE
        self.pullback_count = 0
        self.recovery_count = 0
        self.clean_counter_candles.clear()
    
    def _process_finding_structure(self, ha_data: Dict[str, float]):
        """处理寻找市场结构状态"""
        is_clean_counter = self._is_clean_counter_candle(ha_data, self.trend_direction)
        
        if is_clean_counter:
            # 发现干净反色蜡烛
            self.pullback_count += 1
            self.clean_counter_candles.append(ha_data)
        else:
            # 非干净反色蜡烛
            if self.pullback_count >= 2:  # 回调已确认（>=2根干净反色）
                # 检查是否为修复（主趋势颜色的K线）
                is_trend_color = False
                if self.trend_direction == 'up':
                    is_trend_color = ha_data['ha_close'] > ha_data['ha_open']
                elif self.trend_direction == 'down':
                    is_trend_color = ha_data['ha_close'] < ha_data['ha_open']
                
                if is_trend_color:
                    self.recovery_count += 1
                    if self.recovery_count >= 2:
                        # 市场结构确认：回调+修复完成
                        self.state = StrategyState.FINDING_ENTRY_SIGNAL
                        return
                else:
                    # 不是主趋势颜色，重置修复计数
                    self.recovery_count = 0
            else:
                # 回调未确认就结束，重置
                self.pullback_count = 0
                self.clean_counter_candles.clear()
                self.recovery_count = 0
    
    def _process_finding_entry_signal(self, ha_data: Dict[str, float], volume: float, timestamp: datetime) -> Optional[TradingSignal]:
        """处理寻找入场信号状态"""
        # 检查高量Doji入场条件
        if self._check_high_volume_doji_entry(ha_data, volume):
            return self._generate_signal(ha_data, self.trend_direction, timestamp)
        return None
    
    def on_kline(
        self,
        ts: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> Optional[TradingSignal]:
        """主要的K线处理函数，返回交易信号（如果有的话）"""
        if not self._is_in_session(ts):
            return None
        
        # 计算HA K线
        ha_data = self.ha_calc.calculate(open_price, high, low, close)
        ha_data['volume'] = volume
        ha_data['timestamp'] = ts
        
        # 更新EMA
        ema = self.ema_calc.update(ha_data['ha_close'])
        
        # 判断趋势方向
        current_trend = self._get_trend_direction(ha_data, ema)
        
        # 如果穿越EMA，重置所有状态
        if current_trend != self.trend_direction:
            self.trend_direction = current_trend
            self._reset_state()
            return None
        
        # 如果没有明确趋势，跳过
        if self.trend_direction is None:
            return None
        
        # 根据当前状态处理
        if self.state == StrategyState.FINDING_STRUCTURE:
            self._process_finding_structure(ha_data)
            return None
        elif self.state == StrategyState.FINDING_ENTRY_SIGNAL:
            return self._process_finding_entry_signal(ha_data, volume, ts)
        
        return None