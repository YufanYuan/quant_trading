from datetime import datetime, time
from typing import Optional, Callable
import pandas as pd
import pytz

class Scalp1MinStrategy:
    def __init__(
        self,
        broker, # Placeholder for the broker object
        fee_handler: Optional[Callable[[dict], float]] = None,
        enable_session_filter: bool = True,
        session_start: time = time(9, 0),
        session_end: time = time(11, 0),
    ):
        self.broker = broker
        self.fee_handler = fee_handler
        self.enable_session_filter = enable_session_filter
        self.session_start = session_start
        self.session_end = session_end
        
        # Timezone for session filtering
        self.timezone = pytz.timezone('US/Central')

        # Data storage
        self.kline_data = pd.DataFrame(columns=[
            'ts', 'open', 'high', 'low', 'close', 'volume',
            'ha_open', 'ha_high', 'ha_low', 'ha_close', 'ema100'
        ])

        # State management
        self.position = None # To store current position details
        self.state = "SEARCHING_TREND" 
        self.pullback_peaks = []


    def on_kline(
        self,
        ts: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ):
        """
        Processes each incoming 1-minute candlestick.
        """
        # Session filtering
        if self.enable_session_filter:
            ts_ct = ts.astimezone(self.timezone)
            if not (self.session_start <= ts_ct.time() <= self.session_end):
                return

        # Step 1: Update data and indicators
        self._update_indicators(ts, open, high, low, close, volume)

        # Step 2: Check for position closing
        self._check_close_position()

        # Step 3: Run the state machine for trading logic
        self._run_state_machine()

    def _update_indicators(self, ts, open, high, low, close, volume):
        """Calculates Heikin Ashi candles and EMA."""
        # HA calculation
        ha_close = (open + high + low + close) / 4
        if len(self.kline_data) == 0:
            ha_open = open
        else:
            prev_ha_open = self.kline_data['ha_open'].iloc[-1]
            prev_ha_close = self.kline_data['ha_close'].iloc[-1]
            ha_open = (prev_ha_open + prev_ha_close) / 2
        
        ha_high = max(high, ha_open, ha_close)
        ha_low = min(low, ha_open, ha_close)

        # EMA calculation
        # We need at least 100 data points to calculate EMA
        ema100 = None
        if len(self.kline_data) >= 100:
            # Using pandas for EMA calculation on ha_close
            ema100 = self.kline_data['ha_close'].ewm(span=100, adjust=False).mean().iloc[-1]

        new_kline = {
            'ts': ts, 'open': open, 'high': high, 'low': low, 'close': close, 'volume': volume,
            'ha_open': ha_open, 'ha_high': ha_high, 'ha_low': ha_low, 'ha_close': ha_close,
            'ema100': ema100
        }
        
        # self.kline_data = self.kline_data.append(new_kline, ignore_index=True)
        self.kline_data = pd.concat([self.kline_data, pd.DataFrame([new_kline])], ignore_index=True)


    def _check_close_position(self):
        """Checks if the current position should be closed."""
        if self.position:
            last_kline = self.kline_data.iloc[-1]
            # Check for stop loss or take profit
            if self.position['direction'] == 'long':
                if last_kline['low'] <= self.position['sl']:
                    self.broker.close_position(self.position['id'], self.position['sl'])
                    self.position = None
                elif last_kline['high'] >= self.position['tp']:
                    self.broker.close_position(self.position['id'], self.position['tp'])
                    self.position = None
            elif self.position['direction'] == 'short':
                if last_kline['high'] >= self.position['sl']:
                    self.broker.close_position(self.position['id'], self.position['sl'])
                    self.position = None
                elif last_kline['low'] <= self.position['tp']:
                    self.broker.close_position(self.position['id'], self.position['tp'])
                    self.position = None

    def _run_state_machine(self):
        """Runs the core trading logic based on the current state."""
        if len(self.kline_data) < 103: # Need at least 100+3 bars
            return

        # Universal check: if price crosses EMA, reset the state machine.
        if self.state != "SEARCHING_TREND":
            last_candle = self.kline_data.iloc[-1]
            ema = last_candle['ema100']
            if ema is None: return

            price_crossed_down = self.trend_direction == 'up' and last_candle['ha_close'] < ema
            price_crossed_up = self.trend_direction == 'down' and last_candle['ha_close'] > ema

            if price_crossed_down or price_crossed_up:
                reason = "Price crossed EMA against the trend"
                self._reset_state(reason)
                # After reset, immediately try to find a new trend with the current candle
                self._handle_state_searching_trend()
                return

        if self.state == "SEARCHING_TREND":
            self._handle_state_searching_trend()
        elif self.state == "CONFIRMING_PULLBACKS":
            self._handle_state_confirming_pullbacks()
        elif self.state == "SEARCHING_CLEAN_CANDLES":
            self._handle_state_searching_clean_candles()
        elif self.state == "SEARCHING_DOJI_ENTRY":
            self._handle_state_searching_doji_entry()

    def _reset_state(self, reason=""):
        """Resets the state machine to its initial state."""
        if self.state != "SEARCHING_TREND":
            ts = self.kline_data.iloc[-1]['ts'] if not self.kline_data.empty else 'N/A'
            # print(f"[{ts}] State reset to SEARCHING_TREND. Reason: {reason}")
        self.state = "SEARCHING_TREND"
        self.pullback_peaks = []
        self.trend_direction = None
        self.clean_candle_count = 0

    def _get_ha_color(self, index):
        kline = self.kline_data.iloc[index]
        if kline['ha_close'] > kline['ha_open']: return 'green'
        if kline['ha_close'] < kline['ha_open']: return 'red'
        return 'neutral'

    def _is_doji(self, index):
        kline = self.kline_data.iloc[index]
        body_size = abs(kline['ha_close'] - kline['ha_open'])
        total_range = kline['ha_high'] - kline['ha_low']
        if total_range == 0: return False
        
        upper_shadow = kline['ha_high'] - max(kline['ha_open'], kline['ha_close'])
        lower_shadow = min(kline['ha_open'], kline['ha_close']) - kline['ha_low']
        
        if lower_shadow == 0: # Avoid division by zero
            return body_size / total_range <= 0.1 and upper_shadow == 0
        
        shadow_ratio = upper_shadow / lower_shadow
        return (body_size / total_range <= 0.1) and (0.8 <= shadow_ratio <= 1.25)

    def _is_clean_red_candle(self, index):
        kline = self.kline_data.iloc[index]
        # Red candle with no upper shadow
        return self._get_ha_color(index) == 'red' and kline['ha_high'] == max(kline['ha_open'], kline['ha_close'])

    def _is_clean_green_candle(self, index):
        kline = self.kline_data.iloc[index]
        # Green candle with no lower shadow
        return self._get_ha_color(index) == 'green' and kline['ha_low'] == min(kline['ha_open'], kline['ha_close'])

    def _handle_state_searching_trend(self):
        last_3_candles = self.kline_data.iloc[-3:]
        if last_3_candles['ema100'].isnull().any(): return

        all_above_ema = (last_3_candles['ha_close'] > last_3_candles['ema100']).all()
        all_below_ema = (last_3_candles['ha_close'] < last_3_candles['ema100']).all()
        
        colors = [self._get_ha_color(i) for i in last_3_candles.index]
        is_all_green = all(c == 'green' for c in colors)
        is_all_red = all(c == 'red' for c in colors)

        if all_above_ema and is_all_green:
            self.trend_direction = 'up'
            self.state = "CONFIRMING_PULLBACKS"
            self.pullback_peaks = []
            print(f"[{last_3_candles.iloc[-1]['ts']}] Trend UP detected. Moving to CONFIRMING_PULLBACKS.")
        elif all_below_ema and is_all_red:
            self.trend_direction = 'down'
            self.state = "CONFIRMING_PULLBACKS"
            self.pullback_peaks = []
            print(f"[{last_3_candles.iloc[-1]['ts']}] Trend DOWN detected. Moving to CONFIRMING_PULLBACKS.")

    def _handle_state_confirming_pullbacks(self):
        prev_color = self._get_ha_color(-2)
        curr_color = self._get_ha_color(-1)

        if prev_color == curr_color: return

        if self.trend_direction == 'up' and prev_color == 'green' and curr_color == 'red':
            peak = self.kline_data.iloc[-2][['ha_open', 'ha_close']].max()
            self.pullback_peaks.append(peak)
            print(f"[{self.kline_data.iloc[-1]['ts']}] Pullback peak detected: {peak}")
        elif self.trend_direction == 'down' and prev_color == 'red' and curr_color == 'green':
            valley = self.kline_data.iloc[-2][['ha_open', 'ha_close']].min()
            self.pullback_peaks.append(valley)
            print(f"[{self.kline_data.iloc[-1]['ts']}] Pullback valley detected: {valley}")

        if len(self.pullback_peaks) >= 2:
            self.state = "SEARCHING_CLEAN_CANDLES"
            self.clean_candle_count = 0
            print(f"[{self.kline_data.iloc[-1]['ts']}] Two pullbacks confirmed. Moving to SEARCHING_CLEAN_CANDLES.")
            self._handle_state_searching_clean_candles() # Immediately check current candle

    def _handle_state_searching_clean_candles(self):
        if self._is_doji(-1):
            self.clean_candle_count = 0
            return

        is_long_setup_candle = self.trend_direction == 'up' and self._is_clean_red_candle(-1)
        is_short_setup_candle = self.trend_direction == 'down' and self._is_clean_green_candle(-1)

        if is_long_setup_candle or is_short_setup_candle:
            self.clean_candle_count += 1
        else:
            self.clean_candle_count = 0

        if self.clean_candle_count >= 2:
            self.state = "SEARCHING_DOJI_ENTRY"
            print(f"[{self.kline_data.iloc[-1]['ts']}] Two clean candles found. Moving to SEARCHING_DOJI_ENTRY.")

    def _handle_state_searching_doji_entry(self):
        # This state is active for the candle immediately following the 2 clean candles.
        # If this candle is not a valid Doji, the pattern is broken.
        if not self._is_doji(-1):
            self._reset_state("No Doji appeared after clean candles.")
            return

        # Conditions met, check shadows and trade
        doji_kline = self.kline_data.iloc[-1]
        clean_candle1 = self.kline_data.iloc[-2]
        clean_candle2 = self.kline_data.iloc[-3]

        doji_upper_shadow = doji_kline['ha_high'] - max(doji_kline['ha_open'], doji_kline['ha_close'])
        doji_lower_shadow = min(doji_kline['ha_open'], doji_kline['ha_close']) - doji_kline['ha_low']

        if self.trend_direction == 'up':
            shadow1 = min(clean_candle1['ha_open'], clean_candle1['ha_close']) - clean_candle1['ha_low']
            shadow2 = min(clean_candle2['ha_open'], clean_candle2['ha_close']) - clean_candle2['ha_low']
            if doji_upper_shadow > shadow1 and doji_upper_shadow > shadow2 and \
               doji_lower_shadow > shadow1 and doji_lower_shadow > shadow2:
                if not self.position:
                    entry_price = doji_kline['high']
                    sl = doji_kline['low']
                    tp = entry_price + (entry_price - sl)
                    self.broker.open_long(entry_price, sl, tp)
                    self.position = {'id': 'pos1', 'direction': 'long', 'sl': sl, 'tp': tp}
                self._reset_state("Long trade triggered.")
            else:
                self._reset_state("Doji shadows condition not met.")

        elif self.trend_direction == 'down':
            shadow1 = clean_candle1['ha_high'] - max(clean_candle1['ha_open'], clean_candle1['ha_close'])
            shadow2 = clean_candle2['ha_high'] - max(clean_candle2['ha_open'], clean_candle2['ha_close'])
            if doji_upper_shadow > shadow1 and doji_upper_shadow > shadow2 and \
               doji_lower_shadow > shadow1 and doji_lower_shadow > shadow2:
                if not self.position:
                    entry_price = doji_kline['low']
                    sl = doji_kline['high']
                    tp = entry_price - (sl - entry_price)
                    self.broker.open_short(entry_price, sl, tp)
                    self.position = {'id': 'pos1', 'direction': 'short', 'sl': sl, 'tp': tp}
                self._reset_state("Short trade triggered.")
            else:
                self._reset_state("Doji shadows condition not met.")

# --- Mock Broker for testing ---
class MockBroker:
    def open_long(self, price, sl, tp):
        print(f"--- BROKER: Opening LONG at {price}, SL: {sl}, TP: {tp} ---")
    
    def open_short(self, price, sl, tp):
        print(f"--- BROKER: Opening SHORT at {price}, SL: {sl}, TP: {tp} ---")

    def close_position(self, position_id, price):
        print(f"--- BROKER: Closing position {position_id} at {price} ---")

if __name__ == '__main__':
    # Example usage (for testing)
    broker = MockBroker()
    strategy = Scalp1MinStrategy(broker)

    # Example of feeding a kline
    # In a real scenario, this would come from a live feed or historical data
    strategy.on_kline(
        ts=datetime.now(pytz.utc),
        open=100, high=105, low=95, close=102, volume=1000
    )