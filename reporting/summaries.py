# trading_engine/reporting/summaries.py

import pandas as pd
import numpy as np

from ..backtest.portfolio import compute_max_drawdown
from ..backtest.sector_analysis import summarize_sector_performance


def summarize_overall_backtest(trades_df: pd.DataFrame) -> None:
    """
    Print a high-level system summary based on trade-level returns
    (not portfolio-level).
    """
    if trades_df.empty:
        print("No trades to summarize.")
        return

    df = trades_df.sort_values("entry_date").copy()

    n_trades = len(df)
    win_rate = (df["ret"] > 0).mean()
    avg_ret = df["ret"].mean()
    median_ret = df["ret"].median()
    profit_factor = (
        df[df.ret > 0].ret.sum() / abs(df[df.ret < 0].ret.sum())
        if (df.ret < 0).any()
        else np.nan
    )
    expectancy = df["ret"].mean()

    # These are trade-level stats, not a true compounded equity curve
    total_ret = df["ret"].sum()
    max_dd = np.nan  # not meaningful with overlapping trades

    print("\n=== SYSTEM SUMMARY (trade-level, not portfolio) ===")
    print(f"Total trades      : {n_trades}")
    print(f"Win rate          : {win_rate:.2%}")
    print(f"Average trade ret : {avg_ret:.2%}")
    print(f"Median trade ret  : {median_ret:.2%}")
    print(f"Profit factor     : {profit_factor:.2f}")
    print(f"Expectancy        : {expectancy:.4f} per trade")
    print(f"Total return      : {total_ret:.2%} (sum of trade returns, not compounded)")
    print(f"Max drawdown      : n/a for overlapping trades")


def summarize_portfolio(metrics: dict) -> None:
    """
    Print portfolio-level metrics from run_portfolio_backtest.
    """
    print("\n=== PORTFOLIO SUMMARY (max K concurrent trades) ===")
    if not metrics:
        print("No portfolio metrics available.")
        return

    print(f"Total return      : {metrics.get('total_return', 0.0):.2%}")
    print(f"CAGR              : {metrics.get('cagr', 0.0):.2%}")
    print(f"Volatility        : {metrics.get('volatility', 0.0):.2%}")
    print(f"Sharpe            : {metrics.get('sharpe', 0.0):.2f}")
    print(f"Sortino           : {metrics.get('sortino', 0.0):.2f}")
    print(f"Max drawdown      : {metrics.get('max_drawdown', 0.0):.2%}")
    print(f"Trades used       : {metrics.get('n_trades', 0)}")
    print(
        f"Avg concurrent    : {metrics.get('avg_concurrent_trades', 0.0):.2f} trades"
    )


def print_sector_leaderboard(trades_df: pd.DataFrame) -> None:
    """
    Print ranked sector performance.
    """
    print("\n=== SECTOR PERFORMANCE (ranked) ===")
    stats = summarize_sector_performance(trades_df)
    if stats.empty:
        print("No sector stats available.")
        return
    print(stats)
