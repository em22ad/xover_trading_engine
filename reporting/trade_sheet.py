# trading_engine/reporting/trade_sheet.py

import pandas as pd
from typing import List, Optional

from trading_engine.backtest.rule_scoring import RULE_KEY_COLS


def _pick_score_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Pick the first available score-like column from the given candidates.
    Returns None if none are present.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def suggest_trades_from_top_rules(
    series: pd.DataFrame,
    trades_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    rule_stability_df: pd.DataFrame,
    sector_investability_df: pd.DataFrame,
) -> None:
    """
    Strict‑mode trade sheet:
      - Only investable rules (Weighted Strict)
      - Only investable sectors (Weighted Strict)
      - Only trades belonging to the TOP 5 RULES globally
      - No UNKNOWN rules
      - Clean, compact output
    """

    print("\n=== TODAY'S INVESTABLE TRADE SETUPS (Top 5 Rules Only) ===\n")

    if trades_df.empty or scores_df.empty:
        print("No trades or scores available.")
        return

    # -------------------------------
    # 1. Filter to investable rules
    # -------------------------------
    investable_rules = rule_stability_df[rule_stability_df["is_investable"]].copy()
    if investable_rules.empty:
        print("No rules qualify under Weighted Strict filters.\n")
        return

    # Determine score column
    stability_score_col = _pick_score_column(
        investable_rules,
        ["rule_quality_score", "rule_quality", "avg_ret_full"],
    )

    # -------------------------------
    # 2. Filter to investable sectors
    # -------------------------------
    investable_sectors = sector_investability_df[
        sector_investability_df["is_investable"]
    ].copy()

    if investable_sectors.empty:
        print("No sectors qualify under Weighted Strict filters.\n")
        return

    investable_sector_names = set(investable_sectors["group"].unique())

    # -------------------------------
    # 3. Identify today's trades
    # -------------------------------
    if "entry_date" not in trades_df.columns:
        print("Trades DataFrame missing 'entry_date' column.")
        return

    trades_df = trades_df.copy()
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
    latest_entry = trades_df["entry_date"].max()

    todays_trades = trades_df[trades_df["entry_date"] == latest_entry].copy()
    if todays_trades.empty:
        print("No trades for the latest date.")
        return

    # -------------------------------
    # 4. Merge today's trades with investable rules
    # -------------------------------
    base_cols = RULE_KEY_COLS + ["rule_id"]
    extra_cols = ["rule_description", "positive_sectors", "tickers_traded"]

    cols_to_take = base_cols.copy()
    if stability_score_col:
        cols_to_take.append(stability_score_col)
    for c in extra_cols:
        if c in investable_rules.columns:
            cols_to_take.append(c)

    cols_to_take = list(dict.fromkeys(cols_to_take))

    rule_meta = investable_rules[cols_to_take].drop_duplicates(
        subset=RULE_KEY_COLS + ["rule_id"]
    )

    filtered = todays_trades.merge(rule_meta, on=RULE_KEY_COLS, how="inner")

    # -------------------------------
    # 5. Keep only trades in investable sectors
    # -------------------------------
    if "group" not in filtered.columns:
        print("Merged trades missing 'group' column.")
        return

    filtered = filtered[filtered["group"].isin(investable_sector_names)].copy()
    if filtered.empty:
        print("No trades match investable rules + sectors.")
        return

    # -------------------------------
    # 6. Merge sector explanations
    # -------------------------------
    sector_meta = investable_sectors[["group", "investable_reasons"]].rename(
        columns={"investable_reasons": "sector_reasons"}
    )
    filtered = filtered.merge(sector_meta, on="group", how="left")

    # -------------------------------
    # 7. Sort trades by score
    # -------------------------------
    score_col_for_sort = _pick_score_column(
        filtered,
        ([stability_score_col] if stability_score_col else [])
        + ["rule_quality_score", "rule_quality", "avg_ret_full"],
    )

    if score_col_for_sort:
        filtered = filtered.sort_values(
            ["group", score_col_for_sort], ascending=[True, False]
        )
    else:
        filtered = filtered.sort_values(["group"], ascending=[True])

    # -------------------------------
    # 8. Limit to TOP 5 RULES ONLY
    # -------------------------------
    if "rule_id" in filtered.columns:
        score_col_for_rank = score_col_for_sort or stability_score_col
        if score_col_for_rank is None:
            score_col_for_rank = _pick_score_column(
                filtered,
                ["rule_quality_score", "rule_quality", "avg_ret_full"],
            )

        if score_col_for_rank:
            rule_scores = (
                filtered[["rule_id", score_col_for_rank]]
                .dropna(subset=[score_col_for_rank])
                .groupby("rule_id", as_index=False)[score_col_for_rank]
                .max()
                .sort_values(score_col_for_rank, ascending=False)
            )
            top_rule_ids = set(rule_scores["rule_id"].head(5).tolist())
            filtered = filtered[filtered["rule_id"].isin(top_rule_ids)].copy()
        else:
            top_rule_ids = (
                filtered["rule_id"].dropna().drop_duplicates().head(5).tolist()
            )
            filtered = filtered[filtered["rule_id"].isin(top_rule_ids)].copy()

    if filtered.empty:
        print("No trades remain after limiting to top 5 rules.")
        return

