# trading_engine/backtest/sector_analysis.py

import numpy as np
import pandas as pd

from .portfolio import compute_max_drawdown


def compute_sector_equity_curves(trades_df: pd.DataFrame) -> dict:
    """
    Build an equity curve per sector by compounding trade returns in time order.
    This is rule-agnostic: it just shows how each sector behaves under all rules.
    """
    sector_curves = {}

    if trades_df.empty:
        return sector_curves

    for group, df_g in trades_df.groupby("group"):
        df_g = df_g.sort_values("entry_date")
        eq = (1.0 + df_g["ret"]).cumprod()
        eq.index = df_g["entry_date"]
        sector_curves[group] = eq

    return sector_curves


def summarize_sector_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize sector performance:
    - count
    - mean (%)
    - win_rate
    - max_drawdown (from simple compounded equity)
    - sharpe (approx, using trade-level returns)
    """
    if trades_df.empty:
        return pd.DataFrame()

    stats = trades_df.groupby("group")["ret"].agg(["count", "mean"])
    stats["win_rate"] = trades_df.groupby("group")["ret"].apply(lambda x: (x > 0).mean())

    # Convert mean to % for readability
    stats["mean"] = stats["mean"] * 100.0

    # Compute simple equity-based metrics per sector
    max_dd_list = []
    sharpe_list = []

    for group, df_g in trades_df.groupby("group"):
        df_g = df_g.sort_values("entry_date")
        eq = (1.0 + df_g["ret"]).cumprod()
        if eq.empty:
            max_dd_list.append(np.nan)
            sharpe_list.append(np.nan)
            continue

        dd = compute_max_drawdown(eq)
        max_dd_list.append(dd)

        # Approx Sharpe using trade-level returns (not daily)
        mu = df_g["ret"].mean()
        sigma = df_g["ret"].std()
        sharpe = (mu / sigma) if sigma > 0 else np.nan
        sharpe_list.append(sharpe)

    stats["max_drawdown"] = max_dd_list
    stats["sharpe"] = sharpe_list

    # Sort by mean return descending
    stats = stats.sort_values("mean", ascending=False)

    return stats
