# trading_engine/backtest/portfolio.py

import numpy as np
import pandas as pd

from .rule_scoring import RULE_KEY_COLS

MAX_CONCURRENT_TRADES = 3


def compute_max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd.min()


def _daily_returns_from_trades(trades: pd.DataFrame) -> pd.Series:
    """
    Only CLOSED trades contribute to P&L.
    OPEN trades are ignored.
    """
    closed = trades[trades["exit_date"].notna()].copy()
    if closed.empty:
        return pd.Series(dtype=float)

    closed["entry_date"] = pd.to_datetime(closed["entry_date"])
    closed["exit_date"] = pd.to_datetime(closed["exit_date"])

    start = closed["entry_date"].min()
    end = closed["exit_date"].max()

    all_days = pd.date_range(start, end, freq="B")
    daily_ret = pd.Series(0.0, index=all_days)

    for day in all_days:
        open_mask = (closed["entry_date"] <= day) & (closed["exit_date"] >= day)
        open_trades = closed[open_mask]

        if open_trades.empty:
            continue

        weight = 1.0 / len(open_trades)

        exit_mask = closed["exit_date"] == day
        exiting = closed[exit_mask]

        if not exiting.empty:
            daily_ret.loc[day] = (exiting["ret"] * weight).sum()

    return daily_ret


def run_portfolio_for_trade_suggestions(
    trades: pd.DataFrame,
    best_rules: pd.DataFrame,
) -> dict:
    """
    Portfolio engine:

    - Uses ONLY the globally top rules (already filtered upstream)
    - Trades matched via RULE_KEY_COLS
    - rule_quality_score drives priority when capacity is limited
    """

    if trades.empty or best_rules.empty:
        return {
            "best_rules": pd.DataFrame(),
            "equity_curve": pd.Series(dtype=float),
            "metrics": {},
            "used_trades": pd.DataFrame(),
        }

    # Ensure required columns exist
    for col in RULE_KEY_COLS:
        if col not in trades.columns:
            raise KeyError(f"Column '{col}' missing from trades_df")
        if col not in best_rules.columns:
            raise KeyError(f"Column '{col}' missing from best_rules")

    merge_cols = RULE_KEY_COLS + ["rule_id", "rule_quality_score"]

    # Filter trades to only those belonging to selected rules
    filtered = trades.merge(
        best_rules[merge_cols],
        on=RULE_KEY_COLS,
        how="inner",
        suffixes=("", "_rule"),
    )

    # Include BOTH closed and open trades for selection and display
    selected = filtered.copy()

    # But PnL will still be computed only from closed trades inside _daily_returns_from_trades()
    results = _run_static_portfolio(selected)

    results["best_rules"] = best_rules
    return results


def _run_static_portfolio(trades: pd.DataFrame) -> dict:
    """
    Simple K-constrained portfolio engine using CLOSED trades only.

    - Trades sorted by entry_date
    - If capacity limited, higher rule_quality_score preferred
    """
    trades = trades.sort_values("entry_date").reset_index(drop=True)

    open_trades = []
    used_trades = []

    all_dates = sorted(set(trades["entry_date"].tolist()) | set(trades["exit_date"].dropna().tolist()))

    start = all_dates[0]
    end = all_dates[-1]

    for day in pd.date_range(start, end, freq="B"):
        still_open = []
        for tr in open_trades:
            if tr["exit_date"] < day:
                used_trades.append(tr)
            else:
                still_open.append(tr)
        open_trades = still_open

        todays = trades[trades["entry_date"] == day].copy()
        if not todays.empty:
            if "rule_quality_score" in todays.columns and not todays["rule_quality_score"].isna().all():
                todays = todays.sort_values("rule_quality_score", ascending=False)
            elif "avg_ret" in todays.columns:
                todays = todays.sort_values("avg_ret", ascending=False)

            for _, tr in todays.iterrows():
                if len(open_trades) < MAX_CONCURRENT_TRADES:
                    open_trades.append(tr)

    for tr in open_trades:
        used_trades.append(tr)

    used_trades_df = pd.DataFrame(used_trades)

    if used_trades_df.empty:
        return {
            "equity_curve": pd.Series(dtype=float),
            "metrics": {},
            "used_trades": used_trades_df,
        }

    daily_ret = _daily_returns_from_trades(used_trades_df)
    equity = (1.0 + daily_ret).cumprod()

    if equity.empty:
        metrics = {}
    else:
        total_ret = equity.iloc[-1] - 1.0
        max_dd = compute_max_drawdown(equity)
        ann_factor = 252.0 / max(len(equity), 1)
        cagr = (1.0 + total_ret) ** ann_factor - 1.0 if len(equity) > 0 else 0.0
        vol = daily_ret.std() * np.sqrt(252.0) if len(daily_ret) > 1 else 0.0
        sharpe = (cagr / vol) if vol > 0 else 0.0
        downside = (
            daily_ret[daily_ret < 0].std() * np.sqrt(252.0)
            if (daily_ret < 0).any()
            else 0.0
        )
        sortino = (cagr / downside) if downside > 0 else 0.0

        open_counts = []
        for d in equity.index:
            mask = (used_trades_df["entry_date"] <= d) & (
                used_trades_df["exit_date"] >= d
            )
            open_counts.append(mask.sum())
        avg_concurrent = float(np.mean(open_counts)) if open_counts else 0.0

        metrics = {
            "total_return": float(total_ret),
            "max_drawdown": float(max_dd),
            "cagr": float(cagr),
            "volatility": float(vol),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "n_trades": int(len(used_trades_df)),
            "avg_concurrent_trades": avg_concurrent,
        }

    return {
        "equity_curve": equity,
        "metrics": metrics,
        "used_trades": used_trades_df,
    }
