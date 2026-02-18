# trading_engine/backtest/rule_scoring.py

import hashlib
import json
from typing import Dict, List

import numpy as np
import pandas as pd

# Columns that uniquely define a rule (NO rule_id here)
RULE_KEY_COLS: List[str] = [
    "lookback",
    "group_thresh",
    "participation",
    "lagger_max_move",
    "entry_lag",
    "hold_days",
    "group",
]


def generate_rule_id(rule_params: Dict) -> str:
    """
    Generate a fully deterministic, fully unique rule_id based on ALL rule parameters.
    """
    canonical = json.dumps(rule_params, sort_keys=True)
    h = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
    short = h[:8].upper()
    return f"R_{short}"


def _compute_rule_quality(avg_ret: float, win_rate: float, max_dd: float, n_trades: int) -> float:
    """
    Simple, stable rule quality metric.
    """
    if n_trades < 50:
        penalty = 0.5
    elif n_trades < 200:
        penalty = 0.8
    else:
        penalty = 1.0

    dd_term = max(1e-6, abs(max_dd))
    quality = (avg_ret * win_rate * penalty) / dd_term
    return float(quality)


def score_rules(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rule-level metrics from trade-level data.
    """

    required_cols = RULE_KEY_COLS + ["ret"]
    missing = [c for c in required_cols if c not in trades_df.columns]
    if missing:
        raise ValueError(f"score_rules: trades_df missing required columns: {missing}")

    grouped = trades_df.groupby(RULE_KEY_COLS, as_index=False)

    agg = grouped.agg(
        n_trades=("ret", "size"),
        avg_ret_full=("ret", "mean"),
        win_rate=("ret", lambda x: np.mean(x > 0.0)),
        max_dd=("ret", lambda x: x.min()),
    )

    # Generate rule_id
    rule_ids = []
    for _, row in agg.iterrows():
        rule_params = {k: row[k] for k in RULE_KEY_COLS}
        rule_ids.append(generate_rule_id(rule_params))

    agg["rule_id"] = rule_ids

    # Compute rule quality
    qualities = []
    for _, row in agg.iterrows():
        q = _compute_rule_quality(
            avg_ret=float(row["avg_ret_full"]),
            win_rate=float(row["win_rate"]),
            max_dd=float(row["max_dd"]),
            n_trades=int(row["n_trades"]),
        )
        qualities.append(q)

    agg["rule_quality_score"] = qualities

    ordered_cols = (
        RULE_KEY_COLS
        + ["rule_id", "n_trades", "avg_ret_full", "win_rate", "max_dd", "rule_quality_score"]
    )
    agg = agg[ordered_cols]

    return agg
