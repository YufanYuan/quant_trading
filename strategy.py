import pandas as pd
import pytz
from collections import deque
from datetime import datetime, time
from typing import Optional, Callable, Dict, Any, NamedTuple
from enum import Enum, auto
from dataclasses import dataclass

# ==============================================================================
# 0. 辅助类与数据结构 (Helpers and Data Structures)
# ==============================================================================


class Broker:
    """一个模拟的 Broker，现在增加了 close_position 的日志记录。"""

    def open_long(self, price: float, sl: float, tp: float) -> str:
        pos_id = f"long_{datetime.now().timestamp()}"
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] --- BROKER: Opening LONG @ {price} (SL={sl}, TP={tp}). ID: {pos_id} ---"
        )
        return pos_id

    def open_short(self, price: float, sl: float, tp: float) -> str:
        pos_id = f"short_{datetime.now().timestamp()}"
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] --- BROKER: Opening SHORT @ {price} (SL={sl}, TP={tp}). ID: {pos_id} ---"
        )
        return pos_id

    def close_position(self, position_id: str, price: float):
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] --- BROKER: Closing position {position_id} @ {price} ---"
        )


# 新增: 策略内部状态枚举
class StrategyState(Enum):
    SEARCHING_TREND = auto()  # 寻找主趋势
    TRACKING_TREND = auto()  # 趋势已确立，等待回调
    CONFIRMING_CALLBACK = auto()  # 确认回调段 (计数反色K线)
    AWAITING_ENTRY_OR_REPAIR = auto()  # 等待入场信号或修复信号
    CONFIRMING_REPAIR = auto()  # 确认修复段 (计数主趋势色K线)
    IN_POSITION = auto()  # 已持仓，监控止损止盈


# 新增: 持仓信息数据类
@dataclass
class Position:
    position_id: str
    direction: str  # 'LONG' or 'SHORT'
    sl_price: float
    tp_price: float


# 新增: K线和HA-K线的数据结构
class Candle(NamedTuple):
    ts: datetime
    o: float
    h: float
    l: float
    c: float
    v: float = 0


class HACandle(NamedTuple):
    ts: datetime
    o: float
    h: float
    l: float
    c: float

    @property
    def color(self) -> str:
        return "green" if self.c > self.o else "red"

    @property
    def body_size(self) -> float:
        return abs(self.c - self.o)

    @property
    def full_range(self) -> float:
        return self.h - self.l

    @property
    def upper_shadow(self) -> float:
        return self.h - max(self.o, self.c)

    @property
    def lower_shadow(self) -> float:
        return min(self.o, self.c) - self.l


class Trend(Enum):
    NONE = 0
    UP = 1
    DOWN = 2


# ==============================================================================
# 5. 策略类接口实现 (Refactored Scalp1MStrategy)
# ==============================================================================


class Scalp1MStrategy:
    """
    【状态机重构版】1分钟剥头皮策略
    """

    def __init__(self, broker: Broker, **kwargs):
        self.broker = broker
        # --- 参数设置 ---
        self.ema_period = kwargs.get("ema_period", 100)
        self.callback_candle_count_req = kwargs.get("callback_candle_count_req", 2)
        self.repair_candle_count_req = kwargs.get("repair_candle_count_req", 2)
        self.doji_body_ratio = kwargs.get("doji_body_ratio", 0.2)
        self.doji_shadow_ratio_min = kwargs.get("doji_shadow_ratio_min", 0.8)
        self.doji_shadow_ratio_max = kwargs.get("doji_shadow_ratio_max", 1.25)
        # --- 交易时段 ---
        self.enable_session_filter = kwargs.get("enable_session_filter", True)
        self.session_start = kwargs.get("session_start", time(9, 0))
        self.session_end = kwargs.get("session_end", time(11, 0))
        self.central_timezone = pytz.timezone("America/Chicago")
        # --- 内部状态 ---
        self.ha_close_history = []
        self.ema100: Optional[float] = None
        self.prev_ha_open: Optional[float] = None
        self.prev_ha_close: Optional[float] = None
        self._reset_state_machine()

    def _reset_state_machine(
        self, new_state: StrategyState = StrategyState.SEARCHING_TREND
    ):
        """重置状态机和所有计数器"""
        # print(
        #     f"[{datetime.now().strftime('%H:%M:%S')}] ### STATE TRANSITION -> {new_state.name} ###"
        # )
        self.current_state = new_state
        self.main_trend: Trend = Trend.NONE
        self.callback_count = 0
        self.counter_trend_candle_count = 0
        self.repair_candle_count = 0
        self.clean_counter_trend_candles = deque(maxlen=2)
        self.active_position: Optional[Position] = None

    def on_kline(
        self,
        ts: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ):
        """策略主入口，已重构"""
        if self.enable_session_filter and not self._is_in_session(ts):
            return

        # 步骤 1: 检查并处理止损止盈 (如果持仓)
        if self._check_sl_tp(high, low):
            return  # 如果头寸被关闭，则结束当前K线的处理

        # 步骤 2: 计算指标
        ha_candle = self._calculate_heikin_ashi(ts, open, high, low, close)
        if ha_candle is None:
            return
        self._update_ema(ha_candle.c)
        if self.ema100 is None:
            return

        # 步骤 3: 检查全局趋势是否被打破 (最优先规则)
        if self.current_state != StrategyState.SEARCHING_TREND:
            current_main_trend = self._get_main_trend(ha_candle)
            if current_main_trend != self.main_trend:
                # print(f"[{ts}] Main trend broken. Resetting state machine.")
                self._reset_state_machine()
                # 重新运行状态机以确定新趋势
                self._run_state_machine(ha_candle)
                return

        # 步骤 4: 运行当前状态对应的逻辑
        self._run_state_machine(ha_candle)

    def _run_state_machine(self, ha_candle: HACandle):
        """根据当前状态分发任务"""
        handlers = {
            StrategyState.SEARCHING_TREND: self._handle_searching_trend,
            StrategyState.TRACKING_TREND: self._handle_tracking_trend,
            StrategyState.CONFIRMING_CALLBACK: self._handle_confirming_callback,
            StrategyState.AWAITING_ENTRY_OR_REPAIR: self._handle_awaiting_entry_or_repair,
            StrategyState.CONFIRMING_REPAIR: self._handle_confirming_repair,
        }
        handler = handlers.get(self.current_state)
        if handler:
            handler(ha_candle)

    # --- 状态处理函数 ---
    def _handle_searching_trend(self, ha_candle: HACandle):
        """状态: 寻找主趋势"""
        trend = self._get_main_trend(ha_candle)
        if trend != Trend.NONE:
            self.main_trend = trend
            self._transition_to(StrategyState.TRACKING_TREND)

    def _handle_tracking_trend(self, ha_candle: HACandle):
        """状态: 跟踪趋势，等待回调信号"""
        main_trend_color = "green" if self.main_trend == Trend.UP else "red"
        if ha_candle.color != main_trend_color:
            self.counter_trend_candle_count = 1
            self._transition_to(StrategyState.CONFIRMING_CALLBACK)

    def _handle_confirming_callback(self, ha_candle: HACandle):
        """状态: 确认回调段 (连续3根)"""
        main_trend_color = "green" if self.main_trend == Trend.UP else "red"
        if ha_candle.color != main_trend_color:
            self.counter_trend_candle_count += 1
            if self.counter_trend_candle_count >= self.callback_candle_count_req:
                self._transition_to(StrategyState.AWAITING_ENTRY_OR_REPAIR)
        else:  # 序列被打破
            self.counter_trend_candle_count = 0
            self._transition_to(StrategyState.TRACKING_TREND)

    def _handle_awaiting_entry_or_repair(self, ha_candle: HACandle):
        """状态: 等待入场或修复信号"""
        main_trend_color = "green" if self.main_trend == Trend.UP else "red"
        if ha_candle.color == main_trend_color:
            self.repair_candle_count = 1
            self._transition_to(StrategyState.CONFIRMING_REPAIR)
            return

        # 检查严格的 `干净 -> 干净 -> Doji` 序列
        if self._is_clean_counter_trend_candle(ha_candle):
            self.clean_counter_trend_candles.append(ha_candle)
            return

        if len(self.clean_counter_trend_candles) == 2 and self._is_high_volume_doji(
            ha_candle
        ):
            self._execute_trade(ha_candle)
            return

        # 如果不是干净K线，也不是Doji信号，则序列被打破
        if len(self.clean_counter_trend_candles) > 0:
            self.clean_counter_trend_candles.clear()

    def _handle_confirming_repair(self, ha_candle: HACandle):
        """状态: 确认修复段 (连续3根)"""
        main_trend_color = "green" if self.main_trend == Trend.UP else "red"
        if ha_candle.color == main_trend_color:
            self.repair_candle_count += 1
            if self.repair_candle_count >= self.repair_candle_count_req:
                self.callback_count += 1
                self.counter_trend_candle_count = 0
                self.repair_candle_count = 0
                self._transition_to(StrategyState.TRACKING_TREND)
        else:  # 修复被中断，回到等待状态
            self.repair_candle_count = 0
            # 重新检查入场或修复
            self._handle_awaiting_entry_or_repair(ha_candle)

    # --- 核心行为函数 ---
    def _execute_trade(self, doji_candle: HACandle):
        """执行开仓，并进入 IN_POSITION 状态"""
        if self.callback_count < 2:
            return  # 双重检查

        doji_high, doji_low = doji_candle.h, doji_candle.l
        doji_range = doji_high - doji_low
        if doji_range < 1e-9:
            return

        pos_id = None
        if self.main_trend == Trend.UP:
            sl, tp = doji_low, doji_high + doji_range
            pos_id = self.broker.open_long(price=doji_high, sl=sl, tp=tp)
            self.active_position = Position(pos_id, "LONG", sl, tp)
        elif self.main_trend == Trend.DOWN:
            sl, tp = doji_high, doji_low - doji_range
            pos_id = self.broker.open_short(price=doji_low, sl=sl, tp=tp)
            self.active_position = Position(pos_id, "SHORT", sl, tp)

        if pos_id:
            self._transition_to(StrategyState.IN_POSITION)

    def _check_sl_tp(self, high: float, low: float) -> bool:
        """检查并处理止损止盈，返回是否有关仓发生"""
        if (
            self.current_state != StrategyState.IN_POSITION
            or self.active_position is None
        ):
            return False

        pos = self.active_position
        closed = False
        close_price = 0.0

        if pos.direction == "LONG":
            if low <= pos.sl_price:
                closed, close_price = True, pos.sl_price
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] STOP-LOSS triggered for LONG position."
                )
            elif high >= pos.tp_price:
                closed, close_price = True, pos.tp_price
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] TAKE-PROFIT triggered for LONG position."
                )
        elif pos.direction == "SHORT":
            if high >= pos.sl_price:
                closed, close_price = True, pos.sl_price
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] STOP-LOSS triggered for SHORT position."
                )
            elif low <= pos.tp_price:
                closed, close_price = True, pos.tp_price
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] TAKE-PROFIT triggered for SHORT position."
                )

        if closed:
            self.broker.close_position(pos.position_id, close_price)
            self._reset_state_machine()  # 关仓后重置状态机
            return True
        return False

    def _transition_to(self, new_state: StrategyState):
        """状态转移函数，方便调试"""
        if self.current_state != new_state:
            # print(
            #     f"[{datetime.now().strftime('%H:%M:%S')}] ### STATE TRANSITION: {self.current_state.name} -> {new_state.name} ###"
            # )
            self.current_state = new_state

    # --- 辅助与计算函数 ---
    def _get_main_trend(self, ha_candle: HACandle) -> Trend:
        is_above = ha_candle.o > self.ema100 and ha_candle.c > self.ema100
        is_below = ha_candle.o < self.ema100 and ha_candle.c < self.ema100
        if is_above:
            return Trend.UP
        if is_below:
            return Trend.DOWN
        return Trend.NONE

    # ... (其他辅助函数 _is_doji, _is_clean_counter_trend_candle 等保持不变) ...
    def _is_clean_counter_trend_candle(self, ha_candle: HACandle) -> bool:
        if self.main_trend == Trend.UP:
            return ha_candle.color == "red" and ha_candle.upper_shadow <= 1e-9
        elif self.main_trend == Trend.DOWN:
            return ha_candle.color == "green" and ha_candle.lower_shadow <= 1e-9
        return False

    def _is_high_volume_doji(self, ha_candle: HACandle) -> bool:
        if not self._is_doji(ha_candle):
            return False
        clean_candle1, clean_candle2 = self.clean_counter_trend_candles
        return ha_candle.full_range >= (
            clean_candle1.full_range * 0.9
        ) and ha_candle.full_range >= (clean_candle2.full_range * 0.9)
        # if self.main_trend == Trend.UP:
        #     return (
        #         ha_candle.lower_shadow > clean_candle1.lower_shadow
        #         and ha_candle.lower_shadow > clean_candle2.lower_shadow
        #         and ha_candle.upper_shadow > 0
        #     )
        # elif self.main_trend == Trend.DOWN:
        #     return (
        #         ha_candle.upper_shadow > clean_candle1.upper_shadow
        #         and ha_candle.upper_shadow > clean_candle2.upper_shadow
        #         and ha_candle.lower_shadow > 0
        #     )
        # return False

    def _is_doji(self, ha_candle: HACandle) -> bool:
        if ha_candle.full_range < 1e-9:
            return False
        if (ha_candle.body_size / ha_candle.full_range) > self.doji_body_ratio:
            return False
        l_s, u_s = ha_candle.lower_shadow, ha_candle.upper_shadow
        if l_s > 1e-9 and u_s > 1e-9:
            return self.doji_shadow_ratio_min <= u_s / l_s <= self.doji_shadow_ratio_max
        return l_s <= 1e-9 and u_s <= 1e-9

    # ... (其他计算函数 _calculate_heikin_ashi, _update_ema 等保持不变) ...
    def _is_in_session(self, ts: datetime) -> bool:
        if ts.tzinfo is None:
            ts = pytz.utc.localize(ts)
        return (
            self.session_start
            <= ts.astimezone(self.central_timezone).time()
            < self.session_end
        )

    def _calculate_heikin_ashi(
        self, ts: datetime, o: float, h: float, l: float, c: float
    ) -> Optional[HACandle]:
        ha_c = (o + h + l + c) / 4
        ha_o = self.prev_ha_open if self.prev_ha_open is not None else o
        if self.prev_ha_open is not None and self.prev_ha_close is not None:
            ha_o = (self.prev_ha_open + self.prev_ha_close) / 2
        self.prev_ha_open, self.prev_ha_close = ha_o, ha_c
        return HACandle(ts, ha_o, max(h, ha_o, ha_c), min(l, ha_o, ha_c), ha_c)

    def _update_ema(self, ha_close: float):
        self.ha_close_history.append(ha_close)
        if len(self.ha_close_history) >= self.ema_period:
            self.ema100 = (
                pd.Series(self.ha_close_history)
                .ewm(span=self.ema_period, adjust=False)
                .mean()
                .iloc[-1]
            )
