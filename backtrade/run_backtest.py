
import backtrader as bt
from datetime import datetime
import pandas as pd
import sys
import os

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_data import get_binance_data
from strategy import Scalp1MinStrategy # We need to adapt this for backtrader

# --- Backtrader-compatible Strategy ---
# The original strategy is callback-based. We need to wrap it for backtrader.
class BtScalpStrategy(bt.Strategy):
    params = (
        ('fee_handler', None),
        ('enable_session_filter', True),
        ('session_start_hour', 9),
        ('session_end_hour', 11),
    )

    def __init__(self):
        # We instantiate our original strategy logic here
        self.core_strategy = Scalp1MinStrategy(
            broker=self, # The backtrader strategy itself will act as the broker
            fee_handler=self.p.fee_handler,
            enable_session_filter=self.p.enable_session_filter,
            session_start=datetime.strptime(f"{self.p.session_start_hour}:00", '%H:%M').time(),
            session_end=datetime.strptime(f"{self.p.session_end_hour}:00", '%H:%M').time(),
        )
        self.order = None

    def next(self):
        # Feed data to the core strategy
        self.core_strategy.on_kline(
            ts=self.data.datetime.datetime(0),
            open=self.data.open[0],
            high=self.data.high[0],
            low=self.data.low[0],
            close=self.data.close[0],
            volume=self.data.volume[0]
        )

    # --- Broker implementation for the core strategy ---
    def open_long(self, price, sl, tp):
        if self.position: # Prevent new trades if already in a position
            return
        self.order = self.buy(exectype=bt.Order.Limit, price=price, valid=self.data.datetime.datetime(1))
        # Note: Backtrader's native SL/TP is more robust, but we follow the spec.
        print(f"--- BT BROKER: Placing LONG order at {price}, SL: {sl}, TP: {tp} ---")

    def open_short(self, price, sl, tp):
        if self.position:
            return
        self.order = self.sell(exectype=bt.Order.Limit, price=price, valid=self.data.datetime.datetime(1))
        print(f"--- BT BROKER: Placing SHORT order at {price}, SL: {sl}, TP: {tp} ---")

    def close_position(self, position_id, price):
        if self.position:
            self.close()
            print(f"--- BT BROKER: Closing position {position_id} at {price} ---")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                print(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            print('Order Canceled/Margin/Rejected/Expired')

        self.order = None


if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # --- Data Fetching ---
    symbol = "DOGEUSDT"
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    interval = "1m"

    print("Fetching data...")
    df = get_binance_data(symbol, start_date, end_date, interval)
    
    if df.empty:
        print("Could not fetch data. Exiting.")
        sys.exit(1)

    # Convert to backtrader format
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None, # Use index
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=None
    )

    # --- Cerebro Setup ---
    cerebro.adddata(data)
    cerebro.addstrategy(BtScalpStrategy)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001) # 0.1% fee
    cerebro.addsizer(bt.sizers.FixedSize, stake=1) # Trade a fixed size

    # --- Run Backtest ---
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # --- Plotting ---
    cerebro.plot()
