# trading_engine/data/price_loader.py

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from trading_engine.config.config_loader import load_config

# Load config
_cfg = load_config()

analysis_cfg = _cfg.get("analysis_date", {})
USE_CUSTOM_ANALYSIS_DATE = analysis_cfg.get("use_custom", False)

_custom_date_str = analysis_cfg.get("custom_date", None)
if _custom_date_str:
    CUSTOM_ANALYSIS_DATE = datetime.strptime(_custom_date_str, "%Y-%m-%d").date()
else:
    CUSTOM_ANALYSIS_DATE = None

DATA_DIR = "data"
CACHE_PATH = os.path.join(DATA_DIR, "historical_prices.parquet")

LOOKBACK_YEARS = 1
DATA_INTERVAL = "1d"
BATCH_SIZE = 6
MAX_RETRIES = 3


def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def _download_ohlcv_batch(tickers, start, end, interval="1d", max_retries=3):
    """Download a batch of tickers with retry logic."""
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                timeout=20,
            )
            if data.empty:
                return pd.DataFrame()

            # Ensure MultiIndex columns: (field, ticker)
            if not isinstance(data.columns, pd.MultiIndex):
                ticker = tickers[0] if isinstance(tickers, list) else tickers
                data.columns = pd.MultiIndex.from_product([data.columns, [ticker]])

            # Keep only OHLCV
            wanted = ["Open", "High", "Low", "Close", "Volume"]
            mask = data.columns.get_level_values(0).isin(wanted)
            return data.loc[:, mask]

        except Exception as e:
            print(f"Batch {tickers} failed (attempt {attempt+1}/{max_retries}): {e}")

    print(f"Batch {tickers} failed after {max_retries} attempts.")
    return pd.DataFrame()


def load_or_update_price_cache(universe):
    """Load cached prices or download missing data."""
    ensure_data_dir()

    # Determine analysis end date
    if USE_CUSTOM_ANALYSIS_DATE and CUSTOM_ANALYSIS_DATE:
        target_end_date = CUSTOM_ANALYSIS_DATE
    else:
        target_end_date = datetime.today().date()

    target_start_date = target_end_date - timedelta(days=365 * LOOKBACK_YEARS + 10)

    all_tickers = sorted(set(t for group in universe.values() for t in group))

    print("\n=== DATA CACHE INFO ===")
    print(f"Tickers in universe: {len(all_tickers)}")
    print(f"Required window: {target_start_date} → {target_end_date}")

    # Load existing cache
    if os.path.exists(CACHE_PATH):
        print(f"Cache found at {CACHE_PATH}. Loading...")
        prices = pd.read_parquet(CACHE_PATH)
        prices.index = pd.to_datetime(prices.index)

        cache_end = prices.index.max().date()
        print(f"Cache currently covers: {prices.index.min().date()} → {cache_end}")

        # Update if needed
        if cache_end < target_end_date:
            download_start = cache_end + timedelta(days=1)
            download_end = target_end_date + timedelta(days=1)
            print(f"Updating cache: {download_start} → {download_end}")

            new_batches = []
            for i in range(0, len(all_tickers), BATCH_SIZE):
                batch = all_tickers[i:i + BATCH_SIZE]
                print(f"Updating batch {i//BATCH_SIZE + 1}: {batch}")
                data = _download_ohlcv_batch(batch, download_start, download_end)
                if not data.empty:
                    new_batches.append(data)

            if new_batches:
                new_data = pd.concat(new_batches, axis=1)
                prices = pd.concat([prices, new_data], axis=0)
                prices = prices[~prices.index.duplicated(keep="last")]
                prices = prices.sort_index()

        # Trim to required window
        prices = prices.loc[prices.index.date >= target_start_date]
        prices.to_parquet(CACHE_PATH)
        print("Cache updated.")
        print(f"Cache now covers: {prices.index.min().date()} → {prices.index.max().date()}")
        return prices

    # No cache → full download
    print("No cache found — downloading full required history...")
    all_batches = []
    download_start = target_start_date
    download_end = target_end_date + timedelta(days=1)

    for i in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[i:i + BATCH_SIZE]
        print(f"Downloading batch {i//BATCH_SIZE + 1}: {batch}")
        data = _download_ohlcv_batch(batch, download_start, download_end)
        if not data.empty:
            all_batches.append(data)
        print("Sleeping 25 seconds to avoid Yahoo rate limits...")
        time.sleep(25)

    if not all_batches:
        raise RuntimeError("No price data downloaded — all batches failed.")

    prices = pd.concat(all_batches, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index()
    prices = prices.loc[prices.index.date >= target_start_date]
    prices = prices.dropna(how="all", axis=0)

    prices.to_parquet(CACHE_PATH)
    print(f"Cache created at {CACHE_PATH}.")
    print(f"Cache covers: {prices.index.min().date()} → {prices.index.max().date()}")
    return prices
