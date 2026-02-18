# trading_engine/reporting/rule_stability.py

import pandas as pd
import numpy as np

from trading_engine.backtest.rule_scoring import RULE_KEY_COLS


def compute_rule_stability(trades_df: pd.DataFrame, scores_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty or scores_df.empty:
        return pd.DataFrame()

    df = trades_df.copy()
    df = df[df["exit_date"].notna()].copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    # Full-period stats
    full_stats = (
        df.groupby(RULE_KEY_COLS)
        .agg(
            n_trades=("ret", "count"),
            avg_ret_full=("ret", "mean"),
            win_rate=("ret", lambda x: (x > 0).mean()),
            max_dd=("ret", "min"),
        )
        .reset_index()
    )

    # Recent windows
    last_date = df["exit_date"].max()
    prev_90 = df[df["exit_date"] >= last_date - pd.Timedelta(days=90)]
    prev_30 = df[df["exit_date"] >= last_date - pd.Timedelta(days=30)]

    prev_90_stats = (
        prev_90.groupby(RULE_KEY_COLS)
        .agg(avg_ret_prev_90d=("ret", "mean"))
        .reset_index()
    )

    prev_30_stats = (
        prev_30.groupby(RULE_KEY_COLS)
        .agg(avg_ret_prev_30d=("ret", "mean"))
        .reset_index()
    )

    stab = full_stats.merge(prev_90_stats, on=RULE_KEY_COLS, how="left")
    stab = stab.merge(prev_30_stats, on=RULE_KEY_COLS, how="left")

    # Merge rule_id + rule_quality_score
    stab = stab.merge(
        scores_df[RULE_KEY_COLS + ["rule_id", "rule_quality_score"]],
        on=RULE_KEY_COLS,
        how="left",
    )

    # DO NOT convert to percentages here.
    # Leave avg_ret_full, win_rate, max_dd, etc. as decimals.
    # main.py handles formatting.

    stab["is_investable"] = stab["rule_quality_score"] > 0.0

    return stab
