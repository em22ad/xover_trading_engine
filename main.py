import os
from datetime import datetime

import pandas as pd

from trading_engine.config.config_loader import load_config
from trading_engine.data.universe import UNIVERSE
from trading_engine.data.price_loader import load_or_update_price_cache

from trading_engine.signals.lag_detector import (
    prepare_normalized_series,
    detect_lag_signals,
)

from trading_engine.backtest.trade_generator import backtest_signals
from trading_engine.backtest.rule_scoring import (
    score_rules,
    RULE_KEY_COLS,
)

from trading_engine.backtest.portfolio import (
    run_portfolio_for_trade_suggestions,
)

from trading_engine.reporting.summaries import (
    summarize_overall_backtest,
    summarize_portfolio,
)

from trading_engine.reporting.rule_stability import (
    compute_rule_stability,
)

from trading_engine.reporting.sector_filters import (
    compute_investable_sectors,
)

from trading_engine.reporting.portfolio_trade_table import (
    print_portfolio_trade_table,
)


# ----------------------------------------------------------------------
# Helpers for clean, decision-ready output
# ----------------------------------------------------------------------


def print_investable_sectors_ranked(sector_df: pd.DataFrame) -> None:
    """
    Print only investable sectors, ranked by win rate, in a compact table.
    """
    investable = sector_df.copy()
    if investable.empty:
        print("\n=== INVESTABLE SECTORS (Ranked by Win Rate) ===")
        print("No sectors found.\n")
        return

    investable = investable.sort_values("win_rate", ascending=False)

    print("\n=== INVESTABLE SECTORS (Ranked by Win Rate) ===\n")
    header = (
        f"{'Sector':<24} {'Mean':>8} {'WinRate':>9} "
        f"{'Sharpe':>8} {'Sortino':>8} {'Stability':>10}"
    )
    print(header)
    print("-" * len(header))

    for _, row in investable.iterrows():
        group = row["group"]
        mean = float(row["mean"]) * 100.0
        win = float(row["win_rate"]) * 100.0
        sharpe = float(row["sharpe"])
        sortino = float(row["sortino"])
        stability = float(row["stability"])

        print(
            f"{group:<24} "
            f"{mean:7.4f}% "
            f"{win:8.2f}% "
            f"{sharpe:8.3f} "
            f"{sortino:8.3f} "
            f"{stability:10.3f}"
        )
    print()


def build_rule_narrative(row: pd.Series) -> str:
    """
    Build a clean, human-readable narrative for a rule.
    All percentages are formatted properly.
    No engine logic is changed.
    """

    sector = row.get("group", "UNKNOWN_SECTOR")
    lookback = row.get("lookback")
    participation = row.get("participation")
    lagger_max_move = row.get("lagger_max_move")
    entry_lag = row.get("entry_lag")
    hold_days = row.get("hold_days")

    parts = []

    # Participation threshold
    if participation is not None and lookback is not None:
        parts.append(
            f"If at least {participation * 100:.0f}% of tickers in {sector} "
            f"show a strong move over a {int(lookback)}‑day lookback"
        )
    elif lookback is not None:
        parts.append(
            f"If tickers in {sector} show a strong move over a {int(lookback)}‑day lookback"
        )
    else:
        parts.append(f"If tickers in {sector} show a strong move")

    # Lagger threshold
    if lagger_max_move is not None:
        parts.append(
            f"while the lagging ticker moves less than {lagger_max_move * 100:.2f}%"
        )

    # Entry lag
    if entry_lag is not None:
        parts.append(f"then enter after {int(entry_lag)} day(s)")

    # Holding period
    if hold_days is not None:
        parts.append(f"and hold for {int(hold_days)} day(s)")

    desc = ", ".join(parts).rstrip(", ")
    if not desc.endswith("."):
        desc += "."

    return desc


def print_top_global_rules_table(top_rules: pd.DataFrame) -> None:
    """
    Print the top global rules in a ranked table, using only fields that
    already exist in stability_df.
    """
    if top_rules.empty:
        print("\n=== TOP 10 GLOBAL RULES (Strict, Ranked by Rule Quality) ===")
        print("No investable rules selected.\n")
        return

    for col in [
        "rule_quality_score",
        "avg_ret_full",
        "win_rate",
        "max_dd",
        "n_trades",
        "lookback",
        "participation",
        "lagger_max_move",
        "entry_lag",
        "hold_days",
    ]:
        if col not in top_rules.columns:
            top_rules[col] = pd.NA

    top_rules = top_rules.sort_values("rule_quality_score", ascending=False).head(10)

    print("\n=== TOP 10 GLOBAL RULES (Strict, Ranked by Rule Quality) ===\n")

    header = (
        f"{'Rule ID':<14} {'Sector':<20} {'Score':>8} "
        f"{'AvgRet':>8} {'WinRate':>8} {'MaxDD':>8} {'Trades':>8}"
    )
    print(header)
    print("-" * len(header))

    for _, row in top_rules.iterrows():
        rid = row["rule_id"]
        sector = row["group"]
        score = float(row["rule_quality_score"])
        avg_ret = float(row["avg_ret_full"]) * 100.0
        win = float(row["win_rate"]) * 100.0
        max_dd = float(row["max_dd"]) * 100.0
        n_trades = int(row["n_trades"])

        print(
            f"{rid:<14} {sector:<20} "
            f"{score:8.4f} {avg_ret:8.2f}% {win:8.2f}% {max_dd:8.2f}% {n_trades:8d}"
        )

        desc = build_rule_narrative(row)
        print(f"    Rule logic: {desc}\n")


# ----------------------------------------------------------------------
# Main engine
# ----------------------------------------------------------------------


def run_engine():

    # Load config
    config = load_config()
    price_field = config.get("price_field", "HL2")

    analysis_cfg = config.get("analysis_date", {})
    use_custom_date = analysis_cfg.get("use_custom", False)
    custom_date_str = analysis_cfg.get("custom_date", None)

    print("\n=== LOADING PRICE DATA ===")
    prices_ohlcv = load_or_update_price_cache(UNIVERSE)

    # Apply custom analysis date if configured
    if use_custom_date and custom_date_str:
        custom_date = datetime.strptime(custom_date_str, "%Y-%m-%d").date()
        prices_ohlcv = prices_ohlcv.loc[prices_ohlcv.index.date <= custom_date]

    print("\n=== PREPARING NORMALIZED SERIES ===")
    series = prepare_normalized_series(prices_ohlcv, mode=price_field)

    print("\n=== DETECTING LAG SIGNALS ===")
    signals = detect_lag_signals(series, UNIVERSE)
    print(f"Detected {len(signals):,} lag signals.")

    print("\n=== GENERATING TRADES ===")
    # First pass: generate trades WITHOUT rule_id
    trades_df = backtest_signals(series, signals)
    print(f"Generated {len(trades_df):,} trades.")

    os.makedirs("research", exist_ok=True)
    trades_df.to_csv("research/all_trades_initial.csv", index=False)

    print("\n=== RULE SCORING ===")
    scores_df = score_rules(trades_df)
    scores_df.to_csv("research/rule_scores.csv", index=False)

    # Second pass: attach rule_id + rule_quality_score to trades
    trades_df = backtest_signals(series, signals, scored_rules=scores_df)
    trades_df.to_csv("research/all_trades_with_rule_id.csv", index=False)

    print("\n=== SYSTEM SUMMARY (trade-level) ===")
    summarize_overall_backtest(trades_df)

    # -------------------------------------------------
    # Compute rule stability and sector investability
    # -------------------------------------------------
    stability_df = compute_rule_stability(trades_df, scores_df)
    stability_df.to_csv("research/rule_stability.csv", index=False)

    sector_df = compute_investable_sectors(trades_df)
    sector_df.to_csv("research/sector_investability.csv", index=False)

    # -------------------------------------------------
    # 1) INVESTABLE SECTORS (ranked by win rate)
    # -------------------------------------------------
    print_investable_sectors_ranked(sector_df)

    # -------------------------------------------------
    # 2) TOP 10 GLOBAL RULES (strict, from sectors with win_rate >= 55%)
    # -------------------------------------------------
    high_win_sectors = set(
        sector_df.loc[sector_df["win_rate"] >= 0.51, "group"].unique()
    )

    candidate_rules = stability_df[
        (stability_df["group"].isin(high_win_sectors))
    ].copy()

    candidate_rules = candidate_rules.dropna(subset=["rule_quality_score"])

    if candidate_rules.empty:
        print("No investable rules in high-win-rate sectors — skipping portfolio backtest.\n")
        top_rules = pd.DataFrame()
    else:
        N_GLOBAL_RULES = 10
        top_rules = (
            candidate_rules
            .sort_values("rule_quality_score", ascending=False)
            .head(N_GLOBAL_RULES)
            .copy()
        )

    print_top_global_rules_table(top_rules)

    # -------------------------------------------------
    # 3) PORTFOLIO BACKTEST (using top global rules)
    # -------------------------------------------------
    if top_rules.empty:
        print("No investable rules — skipping portfolio backtest.\n")
        portfolio_results = None
    else:
        filtered_trades = trades_df.merge(
            top_rules[RULE_KEY_COLS],
            on=RULE_KEY_COLS,
            how="inner",
        )

        if filtered_trades.empty:
            print("No trades match selected top rules — skipping portfolio backtest.\n")
            portfolio_results = None
        else:
            portfolio_results = run_portfolio_for_trade_suggestions(
                filtered_trades,
                top_rules,
            )

    if portfolio_results is not None:
        equity = portfolio_results["equity_curve"]
        metrics = portfolio_results["metrics"]
        used_trades = portfolio_results["used_trades"]

        equity.to_csv("research/portfolio_equity_curve.csv")
        used_trades.to_csv("research/portfolio_used_trades.csv", index=False)

        print("\n=== PORTFOLIO SUMMARY ===")
        summarize_portfolio(metrics)

        # -------------------------------------------------
        # 4) ALL TRADES USED IN THE PORTFOLIO
        # -------------------------------------------------
        print_portfolio_trade_table(used_trades, equity)


if __name__ == "__main__":
    run_engine()
