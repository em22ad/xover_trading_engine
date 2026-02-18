# trading_engine/signals/lag_detector.py

import pandas as pd
import numpy as np

from trading_engine.backtest.adaptive_grid import build_sector_param_grid

# Extra signal-quality filters
MIN_LEADERS_PER_SIGNAL = 2
MIN_LAGGERS_PER_SIGNAL = 1
MIN_WINDOW_STD = 0.01  # require some dispersion in the group move


def prepare_normalized_series(ohlcv: pd.DataFrame, mode: str = "HL2") -> pd.DataFrame:
    """
    Convert OHLCV MultiIndex data into a normalized price series.

    price_loader.py creates MultiIndex columns:
        level 0 = field (Open, High, Low, Close, Volume)
        level 1 = ticker

    mode options:
        "High"  → use High
        "Low"   → use Low
        "Close" → use Close
        "HL2"   → (High + Low) / 2
        "HLC3"  → (High + Low + Close) / 3
    """

    # Extract fields from MultiIndex level 0
    if mode == "High":
        series = ohlcv.xs("High", level=0, axis=1)

    elif mode == "Low":
        series = ohlcv.xs("Low", level=0, axis=1)

    elif mode == "Close":
        series = ohlcv.xs("Close", level=0, axis=1)

    elif mode == "HL2":
        high = ohlcv.xs("High", level=0, axis=1)
        low = ohlcv.xs("Low", level=0, axis=1)
        series = (high + low) / 2

    elif mode == "HLC3":
        high = ohlcv.xs("High", level=0, axis=1)
        low = ohlcv.xs("Low", level=0, axis=1)
        close = ohlcv.xs("Close", level=0, axis=1)
        series = (high + low + close) / 3

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalize each ticker to start at 100
    series = series / series.iloc[0] * 100.0
    return series


def _detect_lag_signals_for_group(
    series_group: pd.DataFrame,
    group_name: str,
    lookback: int,
    group_thresh: float,
    participation: float,
    lagger_max_move: float,
    direction: str = "both",
):
    """
    Detect lag signals for a single sector (group) and a single parameter set.
    Applies additional quality filters:
      - minimum dispersion in the window
      - minimum number of leaders
      - minimum number of laggers
    """
    signals = []
    tickers = list(series_group.columns)

    for end_idx in range(lookback, len(series_group)):
        end_date = series_group.index[end_idx]
        start_idx = end_idx - lookback
        start_date = series_group.index[start_idx]

        window_prices = series_group.iloc[start_idx : end_idx + 1]
        window_rets = window_prices.iloc[-1] / window_prices.iloc[0] - 1.0

        # Require some dispersion in the group move; skip flat windows
        if window_rets.std() < MIN_WINDOW_STD:
            continue

        movers_up = pd.Series(dtype=float)
        movers_down = pd.Series(dtype=float)
        frac_up = frac_down = 0.0

        if direction in ("up", "both"):
            movers_up = window_rets[window_rets >= group_thresh]
            frac_up = len(movers_up) / len(tickers)

        if direction in ("down", "both"):
            movers_down = window_rets[window_rets <= -group_thresh]
            frac_down = len(movers_down) / len(tickers)

        # Decide direction of the sector move
        if frac_up >= participation and frac_up >= frac_down:
            dir_sign = 1
            leaders = list(movers_up.index)
        elif frac_down >= participation and frac_down > frac_up:
            dir_sign = -1
            leaders = list(movers_down.index)
        else:
            continue

        # Require a minimum number of leaders
        if len(leaders) < MIN_LEADERS_PER_SIGNAL:
            continue

        # Identify laggers: those that barely moved
        laggers = window_rets[window_rets.abs() <= lagger_max_move]

        # Exclude leaders from laggers (we want true followers)
        laggers = laggers[~laggers.index.isin(leaders)]

        # Require at least one lagger
        if len(laggers) < MIN_LAGGERS_PER_SIGNAL:
            continue

        for lagger_ticker, lagger_ret in laggers.items():
            signals.append(
                {
                    "group": group_name,
                    "ticker": lagger_ticker,
                    "direction": dir_sign,
                    "lookback": lookback,
                    "group_thresh": group_thresh,
                    "participation": participation,
                    "lagger_max_move": lagger_max_move,
                    "signal_date": end_date,
                    "start_date": start_date,
                    "leaders": leaders,
                    "lagger_ret": lagger_ret,
                    "window_rets": window_rets.to_dict(),
                }
            )

    return signals


def detect_lag_signals(series: pd.DataFrame, universe: dict) -> list:
    """
    Detect lag signals across all sectors using a sector-adaptive parameter grid.

    series: normalized price series (rows: dates, cols: tickers)
    universe: {group_name: [tickers]}
    """
    all_signals: list[dict] = []

    # Build sector-specific parameter grid from historical behavior
    sector_param_grid = build_sector_param_grid(series, universe)

    for group_name, tickers in universe.items():
        tickers = [t for t in tickers if t in series.columns]
        if not tickers:
            continue

        group_series = series[tickers]

        param_list = sector_param_grid.get(group_name, [])
        if not param_list:
            continue

        for params in param_list:
            lookback = params["lookback"]
            group_thresh = params["group_thresh"]
            participation = params["participation"]
            lagger_max = params["lagger_max_move"]

            sigs = _detect_lag_signals_for_group(
                group_series,
                group_name,
                lookback,
                group_thresh,
                participation,
                lagger_max,
                direction="both",
            )
            all_signals.extend(sigs)

    # Debug: latest signal date
    if all_signals:
        print("Latest signal date:", max(sig["signal_date"] for sig in all_signals))
    else:
        print("Latest signal date: None (no signals generated)")

    return all_signals
