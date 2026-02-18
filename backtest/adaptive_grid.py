# trading_engine/backtest/adaptive_grid.py

import pandas as pd


def _safe_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change(fill_method=None).replace([float("inf"), float("-inf")], pd.NA).dropna(how="all")


def build_sector_param_grid(series: pd.DataFrame, universe: dict) -> dict:
    """
    Build a sector-specific parameter grid based on simple, deterministic
    sector fingerprints (volatility + cross-sectional dispersion).

    Returns:
        {
            group_name: [
                {
                    "lookback": ...,
                    "group_thresh": ...,
                    "participation": ...,
                    "lagger_max_move": ...,
                    "entry_lag": ...,
                    "hold": ...,
                },
                ...
            ],
            ...
        }
    """
    sector_param_grid: dict[str, list[dict]] = {}

    for group_name, tickers in universe.items():
        tickers = [t for t in tickers if t in series.columns]
        if not tickers:
            continue

        sub = series[tickers]
        rets = _safe_pct_change(sub)
        if rets.empty:
            continue

        # Sector fingerprint
        vol = rets.std().mean()  # average per-ticker volatility
        cs_disp = rets.std(axis=1).mean()  # average cross-sectional dispersion

        # --- Lookback selection ---
        if vol > 0.03:
            lookbacks = [2, 3]
            holds = [3, 5]
        elif vol > 0.02:
            lookbacks = [3, 5]
            holds = [3, 5, 7]
        else:
            lookbacks = [5, 10]
            holds = [5, 7, 10]

        # --- Group threshold selection (sector move strength) ---
        if cs_disp > 0.025:
            group_thresh_vals = [0.03, 0.05]
        elif cs_disp > 0.015:
            group_thresh_vals = [0.02, 0.03]
        else:
            group_thresh_vals = [0.015, 0.02]

        # --- Participation selection ---
        if cs_disp > 0.02:
            participation_vals = [0.6, 0.7]
        else:
            participation_vals = [0.5, 0.6]

        # --- Lagger threshold selection ---
        # Keep laggers relatively tight vs group move thresholds
        lagger_max_vals = [min(gt * 0.75, 0.04) for gt in group_thresh_vals]

        entry_lags = [0, 1]

        params_list: list[dict] = []
        for lb in lookbacks:
            for gt in group_thresh_vals:
                for part in participation_vals:
                    for lag in lagger_max_vals:
                        for elag in entry_lags:
                            for hold in holds:
                                params_list.append(
                                    {
                                        "lookback": lb,
                                        "group_thresh": gt,
                                        "participation": part,
                                        "lagger_max_move": lag,
                                        "entry_lag": elag,
                                        "hold": hold,
                                    }
                                )

        sector_param_grid[group_name] = params_list

    return sector_param_grid
