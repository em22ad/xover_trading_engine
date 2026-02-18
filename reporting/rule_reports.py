# trading_engine/reporting/rule_reports.py

import pandas as pd
from typing import Optional

from trading_engine.backtest.rule_scoring import RULE_KEY_COLS


def _print_top_rules_block(
    title: str,
    rules_df: pd.DataFrame,
    sector_breakdown_df: Optional[pd.DataFrame] = None,
    top_n: int = 10,
) -> None:
    if rules_df.empty:
        print(f"\n=== {title} ===")
        print("No rules available.\n")
        return

    print(f"\n=== {title} ===\n")

    # Expect columns: rule_id, win_rate, avg_ret, max_dd, trade_count
    cols = {
        "rule_id": "rule_id",
        "win_rate": "win_rate",
        "avg_ret": "avg_ret",
        "max_dd": "max_dd",
        "trade_count": "trade_count",
    }
    for c in cols.values():
        if c not in rules_df.columns:
            # If structure changed, just show head
            print(rules_df.head(top_n))
            return

    top = rules_df.sort_values("avg_ret", ascending=False).head(top_n)

    for _, row in top.iterrows():
        rule_id = row["rule_id"]
        print(f"R_{rule_id}:")
        print(f"Win rate          : {row['win_rate']:.2f}%")
        print(f"Avg return        : {row['avg_ret']:.2f}%")
        print(f"Max DD            : {row['max_dd']:.2f}%")
        print(f"Trades            : {int(row['trade_count'])}\n")

        if sector_breakdown_df is not None and "rule_id" in sector_breakdown_df.columns:
            sb = sector_breakdown_df[sector_breakdown_df["rule_id"] == rule_id]
            if not sb.empty and {"group", "mean"}.issubset(sb.columns):
                print("  Sector performance for this rule (mean % return):")
                print(sb[["group", "mean"]].set_index("group"))
                print()


def _print_rule_stability_block(rule_stability_df: pd.DataFrame, top_n: int = 10) -> None:
    print("\n=== RULE STABILITY (Weighted Strict) ===\n")

    if rule_stability_df.empty:
        print("No rule stability data available.\n")
        return

    # Expect columns: rule_id, group, avg_ret_full, avg_ret_90d, avg_ret_30d,
    # win_rate, max_dd_full, rule_quality, is_investable
    col_map = {
        "rule_id": "rule_id",
        "group": "group",
        "avg_ret_full": "avg_ret_full",
        "avg_ret_90d": "avg_ret_90d",
        "avg_ret_30d": "avg_ret_30d",
        "win_rate": "win_rate",
        "max_dd_full": "max_dd_full",
        "rule_quality": "rule_quality",
        "is_investable": "is_investable",
    }

    missing = [c for c in col_map.values() if c not in rule_stability_df.columns]
    if missing:
        # Fallback: just show head
        print(rule_stability_df.head(top_n))
        return

    investable = rule_stability_df[rule_stability_df[col_map["is_investable"]]].copy()
    if investable.empty:
        print("No rules qualify under Weighted Strict filters.\n")
        return

    investable = investable.sort_values(col_map["rule_quality"], ascending=False).head(top_n)

    print("\n=== INVESTABLE RULES (Weighted Strict Mode) — Top 10 ===\n")
    for _, row in investable.iterrows():
        rid = row[col_map["rule_id"]]
        grp = row[col_map["group"]]
        print(f"Rule R_{rid} — group={grp}")
        print(f"  AvgRet (full)     : {row[col_map['avg_ret_full']]:.4f}% (n={int(row.get('trade_count', 0))})")
        print(f"  AvgRet (prev 90d) : {row[col_map['avg_ret_90d']]:.4f}%")
        print(f"  AvgRet (prev 30d) : {row[col_map['avg_ret_30d']]:.4f}%")
        print(f"  Win rate          : {row[col_map['win_rate']]:.2f}%")
        print(f"  Max DD (full)     : {row[col_map['max_dd_full']]:.2f}%")
        print(f"  Rule quality      : {row[col_map['rule_quality']]:.6f}\n")


def _print_sector_stability_block(sector_investability_df: pd.DataFrame) -> None:
    print("\n=== SECTOR STABILITY (Weighted Strict) ===\n")

    if sector_investability_df.empty:
        print("No sector stability data available.\n")
        return

    # Expect columns: group, mean_ret, win_rate, sharpe, sortino, stability, is_investable
    col_map = {
        "group": "group",
        "mean_ret": "mean_ret",
        "win_rate": "win_rate",
        "sharpe": "sharpe",
        "sortino": "sortino",
        "stability": "stability",
        "is_investable": "is_investable",
    }

    missing = [c for c in col_map.values() if c not in sector_investability_df.columns]
    if missing:
        print(sector_investability_df.head())
        return

    investable = sector_investability_df[sector_investability_df[col_map["is_investable"]]].copy()

    print("\n=== INVESTABLE SECTORS (Weighted Strict Mode) ===\n")
    for _, row in investable.iterrows():
        grp = row[col_map["group"]]
        print(
            f"{grp} — Mean: {row[col_map['mean_ret']]:.4f}%, "
            f"WinRate: {row[col_map['win_rate']]:.2f}%, "
            f"Sharpe: {row[col_map['sharpe']]:.3f}, "
            f"Sortino: {row[col_map['sortino']]:.3f}, "
            f"Stability: {row[col_map['stability']]:.3f}"
        )
        print()


def print_rule_leaderboard(
    scores_df: pd.DataFrame,
    cleaned_scores_df: pd.DataFrame,
    rule_stability_df: pd.DataFrame,
    sector_investability_df: pd.DataFrame,
    sector_breakdown_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Console reporting:
      - Top rules (research)
      - Top rules (cleaned)
      - Rule stability (Weighted Strict)
      - Sector stability (Weighted Strict)
    """

    _print_top_rules_block("TOP RULES (research)", scores_df, sector_breakdown_df)
    _print_top_rules_block("TOP RULES (cleaned)", cleaned_scores_df, sector_breakdown_df)
    _print_rule_stability_block(rule_stability_df)
    _print_sector_stability_block(sector_investability_df)
