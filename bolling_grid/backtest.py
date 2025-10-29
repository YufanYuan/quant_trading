import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path to allow sibling imports, making the script runnable from any location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from bolling_grid.strategy import BollingerGridStrategy
from market_data import get_binance_data


def run_backtest():
    """Main function to run the backtesting process."""
    # --- Task 1: Environment Setup & Data Preparation ---
    print("Starting backtest...")
    print("Step 1: Setting up environment and fetching data.")

    # 1a. Backtest Parameters
    SYMBOL = "BTCUSDT"
    START_TIME = datetime(2024, 1, 1)
    END_TIME = datetime(2024, 12, 31, 23, 59)
    MA_PERIOD = 20

    # 1b. Strategy Configuration
    strategy_config = {
        "ma_period": MA_PERIOD,
        "sigma_k": 1,
        "grid1_num": 10,
        "grid2_num": 6,
        "grid3_num": 2,
        "regrid_interval_min": 5,
        "lot_size": 0.001,  # Assuming BTC allows fractional lots
    }

    # 1c. Fetch Data
    print(f"Fetching 1-minute data for {SYMBOL}...")
    df_minute = get_binance_data(SYMBOL, START_TIME, END_TIME, "1m")
    if df_minute.empty:
        print("Could not fetch 1-minute data. Aborting.")
        return

    print(f"Fetching 1-day data for {SYMBOL}...")
    df_day = get_binance_data(SYMBOL, START_TIME, END_TIME, "1d")
    if df_day.empty:
        print("Could not fetch 1-day data. Aborting.")
        return

    print(f"Loaded {len(df_minute)} minute bars and {len(df_day)} daily bars.")

    # 1d. Calculate Indicators on Daily Data
    print("Calculating daily indicators (MA, Sigma)...")
    df_day["ma"] = df_day["close"].rolling(window=MA_PERIOD).mean()
    df_day["sigma"] = df_day["close"].rolling(window=MA_PERIOD).std()

    # To avoid lookahead bias, we use the indicator from the previous day's close.
    # The signal for day D is based on data available at the end of day D-1.
    df_day["ma"] = df_day["ma"].shift(1)
    df_day["sigma"] = df_day["sigma"].shift(1)

    # 1e. Merge Daily Indicators into Minute Data
    print("Merging daily indicators into minute data...")
    daily_indicators = df_day[["open_time", "ma", "sigma"]].copy()
    daily_indicators.dropna(inplace=True)

    # Use merge_asof to map the daily indicator to all minutes of that day
    df_merged = pd.merge_asof(
        df_minute.sort_values("open_time"),
        daily_indicators.sort_values("open_time"),
        on="open_time",
        direction="backward",  # Use the most recent indicator
    )
    df_merged.dropna(inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    print(
        f"Data preparation complete. {len(df_merged)} minute bars ready for backtest."
    )

    # --- Task 2: Backtest Simulation Loop ---
    print("\nStep 2: Starting simulation loop...")

    # 2a. Initialize Portfolio and Strategy
    portfolio = {
        "cash": 10000.0,  # Starting with $10,000
        "position_size": 0.0,
        "total_value": 10000.0,
        "history": [],
    }
    strategy = BollingerGridStrategy(strategy_config)

    # 2b. Main Loop
    total_bars = len(df_merged)
    for i, row in df_merged.iterrows():
        # Prepare data for the strategy
        price = row["close"]
        ts = row["open_time"]
        ma = row["ma"]
        sigma = row["sigma"]
        bar_1m = {"close": price, "ts": ts.timestamp()}

        position_before = strategy.position

        # 2c. Call strategy's on_bar method
        strategy.on_bar(bar_1m, ma, sigma)

        position_after = strategy.position

        # 2d. Execute trades if position changed
        if position_after != position_before:
            trade_size = position_after - position_before
            trade_value = trade_size * price
            fee = abs(trade_value) * strategy.fee_rate

            portfolio["cash"] -= trade_value + fee
            portfolio["position_size"] += trade_size

        # 2e. Update portfolio value for every bar
        position_value = portfolio["position_size"] * price
        portfolio["total_value"] = portfolio["cash"] + position_value
        portfolio["history"].append((ts, portfolio["total_value"]))

        if (i + 1) % 50000 == 0:
            print(f"  Processed {i + 1}/{total_bars} bars...")

    print("Simulation loop finished.")

    # --- Task 3: Metrics Calculation ---
    print("\nStep 3: Calculating performance metrics...")
    price_series = df_merged.set_index("open_time")["close"]
    calculate_and_print_metrics(
        portfolio, initial_cash=10000.0, price_series=price_series
    )


def calculate_and_print_metrics(
    portfolio: dict, initial_cash: float, price_series: pd.Series
):
    """
    Calculates and prints key performance metrics for the strategy and a buy-and-hold benchmark.
    """
    if not portfolio["history"]:
        print("No trading history to analyze.")
        return

    # --- Strategy Performance ---
    history_df = pd.DataFrame(portfolio["history"], columns=["timestamp", "value"])
    history_df.set_index("timestamp", inplace=True)
    strategy_value = history_df["value"]

    # 1. Total Return
    strategy_end_value = strategy_value.iloc[-1]
    strategy_total_return_pct = (strategy_end_value / initial_cash - 1) * 100

    # 2. Max Drawdown
    strategy_running_max = strategy_value.cummax()
    strategy_drawdown = (strategy_value - strategy_running_max) / strategy_running_max
    strategy_max_drawdown_pct = strategy_drawdown.min() * 100

    # 3. Sharpe Ratio (annualized)
    strategy_daily_values = strategy_value.resample("D").last()
    strategy_daily_returns = strategy_daily_values.pct_change().dropna()

    if strategy_daily_returns.std() > 0:
        strategy_sharpe_ratio = (
            strategy_daily_returns.mean() / strategy_daily_returns.std() * np.sqrt(365)
        )
    else:
        strategy_sharpe_ratio = 0.0

    # --- Buy and Hold Performance ---
    # Align price series with strategy's trading period
    price_series = price_series[strategy_value.index.min() : strategy_value.index.max()]

    initial_price = price_series.iloc[0]
    final_price = price_series.iloc[-1]

    # 1. Total Return
    bh_end_value = (initial_cash / initial_price) * final_price
    bh_total_return_pct = (bh_end_value / initial_cash - 1) * 100

    # 2. Max Drawdown
    bh_value = (initial_cash / initial_price) * price_series
    bh_running_max = bh_value.cummax()
    bh_drawdown = (bh_value - bh_running_max) / bh_running_max
    bh_max_drawdown_pct = bh_drawdown.min() * 100

    # 3. Sharpe Ratio (annualized)
    bh_daily_values = bh_value.resample("D").last()
    bh_daily_returns = bh_daily_values.pct_change().dropna()

    if bh_daily_returns.std() > 0:
        bh_sharpe_ratio = (
            bh_daily_returns.mean() / bh_daily_returns.std() * np.sqrt(365)
        )
    else:
        bh_sharpe_ratio = 0.0

    # --- Print Results ---
    print("\n--- Backtest Results ---")
    print(
        f"Period: {strategy_value.index.min().date()} to {strategy_value.index.max().date()}"
    )
    print(f"Initial Portfolio Value: ${initial_cash:,.2f}")

    print("\n--- Strategy Performance ---")
    print(f"Final Portfolio Value:   ${strategy_end_value:,.2f}")
    print(f"Total Return: {strategy_total_return_pct:.2f}%")
    print(f"Max Drawdown: {strategy_max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio (Annualized): {strategy_sharpe_ratio:.2f}")

    print("\n--- Buy and Hold Performance ---")
    print(f"Final Portfolio Value:   ${bh_end_value:,.2f}")
    print(f"Total Return: {bh_total_return_pct:.2f}%")
    print(f"Max Drawdown: {bh_max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio (Annualized): {bh_sharpe_ratio:.2f}")
    print("------------------------")


if __name__ == "__main__":
    run_backtest()
