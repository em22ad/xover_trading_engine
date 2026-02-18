# trading_engine/backtest/best_rules_selector.py

import pandas as pd


def select_best_rules_per_sector(
    rule_stability_df: pd.DataFrame,
    sector_investability_df: pd.DataFrame,
    top_rules_per_sector: int = 2,
) -> pd.DataFrame:
    """
    Select top N rules per investable sector.

    Assumptions:
      - rule_stability_df has columns:
          'rule_id', 'group', 'rule_quality_score', 'avg_ret_prev_90d'
      - sector_investability_df has column:
          'group' (sector name)
      - 'group' is the sector key in both tables
    """

    if rule_stability_df.empty or sector_investability_df.empty:
        return pd.DataFrame()

    # Investable sectors (already filtered by sector strict mode upstream)
    investable_sectors = sector_investability_df["group"].unique().tolist()

    df = rule_stability_df.copy()

    # Keep only rules from investable sectors
    df = df[df["group"].isin(investable_sectors)]

    if df.empty:
        return pd.DataFrame()

    # Drop rules without a quality score
    df = df[df["rule_quality_score"].notna()]

    if df.empty:
        return pd.DataFrame()

    # Sort within each sector:
    #   1) rule_quality_score (desc)
    #   2) avg_ret_prev_90d (desc) as tie-breaker
    df = df.sort_values(
        by=["group", "rule_quality_score", "avg_ret_prev_90d"],
        ascending=[True, False, False],
    )

    # Take top N per sector
    best_rules = (
        df.groupby("group", as_index=False)
        .head(top_rules_per_sector)
        .reset_index(drop=True)
    )

    return best_rules
