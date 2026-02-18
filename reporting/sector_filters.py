# trading_engine/reporting/sector_filters.py

import pandas as pd
import numpy as np
from typing import List, Tuple

from trading_engine.config.config_loader import load_config

# Load strict-mode thresholds from config.yaml
config = load_config()
SECTOR_STRICT_PARAMS = config["strict_mode"]["sector_filters"]


def _compute_sector_daily_returns(trades: pd.DataFrame) -> pd.DataFrame:
    closed = trades[trades["exit_date"].notna()].copy()
    if closed.empty:
        return pd.DataFrame()

    closed["entry_date"] = pd.to_datetime(closed["entry_date"])
    closed["exit_date"] = pd.to_datetime(closed["exit_date"])

    start = closed["entry_date"].min()
    end = closed["exit_date"].max()

    all_days = pd.date_range(start, end, freq="B")
    sectors = closed["group"].unique()

    daily = pd.DataFrame(0.0, index=all_days, columns=sectors)

    for day in all_days:
        day_trades = closed[(closed["entry_date"] <= day) & (closed["exit_date"] >= day)]
        if day_trades.empty:
            continue

        for sector in sectors:
            sec_trades = day_trades[day_trades["group"] == sector]
            if sec_trades.empty:
                continue

            weight = 1.0 / len(sec_trades)
            exits = sec_trades[sec_trades["exit_date"] == day]

            if not exits.empty:
                daily.loc[day, sector] = (exits["ret"] * weight).sum()

    return daily


def _compute_sector_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()

    metrics = []

    for sector in daily.columns:
        series = daily[sector].dropna()
        if series.empty:
            continue

        mean_ret = series.mean()
        win_rate = (series > 0).mean()

        vol = series.std()
        downside = series[series < 0].std()

        sharpe = (mean_ret / vol) if vol > 0 else 0.0
        sortino = (mean_ret / downside) if downside > 0 else 0.0

        equity = (1 + series).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / peak
        max_dd = dd.min()

        metrics.append({
            "group": sector,
            "mean": mean_ret,
            "win_rate": win_rate,
            "volatility": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_dd": max_dd,
        })

    return pd.DataFrame(metrics)


def _compute_stability_score(daily: pd.DataFrame, sector: str) -> float:
    if daily.empty or sector not in daily.columns:
        return 0.0

    series = daily[sector].dropna()
    if series.empty:
        return 0.0

    last_date = series.index.max()

    last_30 = last_date - pd.Timedelta(days=30)
    prev_30 = last_date - pd.Timedelta(days=60)
    prev_90 = last_date - pd.Timedelta(days=150)

    def avg_between(start, end):
        mask = (series.index > start) & (series.index <= end)
        return series[mask].mean() if mask.any() else 0.0

    s_last_30 = avg_between(last_30, last_date)
    s_prev_30 = avg_between(prev_30, last_30)
    s_prev_90 = avg_between(prev_90, prev_30)

    return 0.5 * s_last_30 + 0.3 * s_prev_30 + 0.2 * s_prev_90


def _evaluate_sector_weighted_strict(row: pd.Series) -> Tuple[bool, List[str]]:
    p = SECTOR_STRICT_PARAMS
    reasons = []
    ok = True

    if row["mean"] < p["min_mean"]:
        ok = False
        reasons.append(f"- Sector mean return {row['mean']:.4%} is not positive.")

    if row["win_rate"] < p["min_win_rate"]:
        ok = False
        reasons.append(f"- Sector win rate {row['win_rate']:.2%} is below 50%.")

    if row["sharpe"] < p["min_sharpe"]:
        ok = False
        reasons.append(f"- Sector Sharpe {row['sharpe']:.4f} is too weak.")

    if row["sortino"] < p["min_sortino"]:
        ok = False
        reasons.append(f"- Sector Sortino {row['sortino']:.4f} is too weak.")

    if row["stability"] < p["min_stability"]:
        ok = False
        reasons.append(f"- Sector stability score {row['stability']:.4f} is too weak.")

    if row["max_dd"] < p["min_max_dd"]:
        ok = False
        reasons.append(f"- Sector max drawdown {row['max_dd']:.2%} is too deep.")

    return ok, reasons


def compute_investable_sectors(trades: pd.DataFrame) -> pd.DataFrame:
    daily = _compute_sector_daily_returns(trades)
    metrics = _compute_sector_metrics(daily)

    if metrics.empty:
        return metrics

    metrics["stability"] = metrics["group"].apply(lambda g: _compute_stability_score(daily, g))

    investable_flags = []
    investable_reasons = []

    for _, row in metrics.iterrows():
        ok, reasons = _evaluate_sector_weighted_strict(row)
        investable_flags.append(ok)
        investable_reasons.append(reasons)

    metrics["is_investable"] = investable_flags
    metrics["investable_reasons"] = investable_reasons

    return metrics


def print_sector_investability_report(sector_df: pd.DataFrame) -> None:
    print("\n=== INVESTABLE SECTORS (Weighted Strict Mode) ===\n")

    if sector_df.empty:
        print("No sector data available.")
        return

    investable = sector_df[sector_df["is_investable"]].copy()

    if investable.empty:
        print("No sectors meet Weighted Strict stability criteria.\n")
        print("=== SAMPLE REJECTED SECTORS (with reasons) ===\n")
        for _, row in sector_df.head(5).iterrows():
            print(f"Sector {row['group']} — Rejected\n")
            print("Reason:")
            for r in row["investable_reasons"]:
                print(r)
            print()
        return

    for _, row in investable.iterrows():
        print(
            f"{row['group']} — Mean: {row['mean']:.4%}, "
            f"WinRate: {row['win_rate']:.2%}, Sharpe: {row['sharpe']:.3f}, "
            f"Sortino: {row['sortino']:.3f}, Stability: {row['stability']:.3f}"
        )
        print()
