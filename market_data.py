import requests
import zipfile
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path("data_cache")
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
BINANCE_KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


def get_binance_data(
    symbol: str, start_time: datetime, end_time: datetime, interval: str
) -> pd.DataFrame:
    """
    Fetches Binance kline data for a specific symbol and time range, with caching.

    Args:
        symbol (str): The trading symbol (e.g., 'BTCUSDT').
        start_time (datetime): The start of the desired time range.
        end_time (datetime): The end of the desired time range.
        interval (str): The kline interval (e.g., '1h', '1m', '1d').

    Returns:
        pd.DataFrame: A DataFrame containing the requested kline data.
    """
    symbol = symbol.upper()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Generate the list of year/month pairs needed for the download
    date_range = pd.date_range(start=start_time.replace(day=1), end=end_time, freq="MS")

    all_monthly_data = []

    for dt in date_range:
        year = dt.year
        month = dt.month
        month_str = f"{month:02d}"

        csv_filename = f"{symbol}-{interval}-{year}-{month_str}.csv"
        csv_path = CACHE_DIR / csv_filename

        if csv_path.exists():
            print(f"Loading from cache: {csv_path}")
            monthly_df = pd.read_csv(csv_path, header=None, names=BINANCE_KLINE_COLS)
            if year >= 2025:
                # 将 open_time 和 close_time 列除以 1000
                monthly_df["open_time"] = monthly_df["open_time"] // 1000
                monthly_df["close_time"] = monthly_df["close_time"] // 1000
            all_monthly_data.append(monthly_df)
            continue

        # If not in cache, download
        zip_filename = f"{symbol}-{interval}-{year}-{month_str}.zip"
        zip_path = CACHE_DIR / zip_filename
        url = f"{BASE_URL}/{symbol}/{interval}/{zip_filename}"

        print(f"Attempting to download from: {url}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            print(f"Downloading {zip_filename}...")
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Unzipping {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(CACHE_DIR)
            print(f"Successfully extracted {csv_filename}")

            # Now read the extracted CSV
            if csv_path.exists():
                monthly_df = pd.read_csv(
                    csv_path, header=None, names=BINANCE_KLINE_COLS
                )
                all_monthly_data.append(monthly_df)
            else:
                print(f"Warning: Could not find {csv_filename} after extraction.")

        except requests.exceptions.HTTPError as http_err:
            print(
                f"Warning: Could not download {zip_filename}. HTTP error: {http_err}. Maybe data doesn't exist for this period."
            )
        except Exception as err:
            print(f"An error occurred while processing {zip_filename}: {err}")
        finally:
            if zip_path.exists():
                os.remove(zip_path)

    if not all_monthly_data:
        return pd.DataFrame()

    # Combine all dataframes and filter to the precise time range
    full_df = pd.concat(all_monthly_data, ignore_index=True)
    full_df = full_df[full_df["ignore"] == 0]
    full_df["open_time"] = pd.to_datetime(full_df["open_time"], unit="ms")

    # Ensure start_time and end_time are timezone-aware if the dataframe's index is
    # For this implementation, we assume naive datetimes or consistent timezones
    mask = (full_df["open_time"] >= start_time) & (full_df["open_time"] <= end_time)

    return full_df.loc[mask].reset_index(drop=True)


if __name__ == "__main__":
    # Example usage:
    # Get 1-minute data for BTCUSDT for a specific period in 2022
    symbol_to_fetch = "BTCUSDT"
    start_date = datetime(2022, 1, 1, 0, 0)
    end_date = datetime(2022, 2, 28, 23, 59)
    kline_interval = "1m"

    print(
        f"Fetching {kline_interval} data for {symbol_to_fetch} from {start_date} to {end_date}"
    )

    data = get_binance_data(symbol_to_fetch, start_date, end_date, kline_interval)

    if not data.empty:
        print("Successfully fetched data:")
        print(f"Shape: {data.shape}")
        print("First 5 rows:")
        print(data.head())
        print("\nLast 5 rows:")
        print(data.tail())
    else:
        print("Could not fetch any data for the specified range.")
