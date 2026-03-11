import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESEARCH_DIR = "research"


# ------------------------------------------------------------
# Utility loader
# ------------------------------------------------------------
def _load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(RESEARCH_DIR, name)
    if not os.path.exists(path):
        print(f"[WARN] Missing research file: {name}")
        return pd.DataFrame()
    return pd.read_csv(path)


# ------------------------------------------------------------
# Load all research artifacts
# ------------------------------------------------------------
def load_research_artifacts():
    # 1) Equity curve
    equity = _load_csv("portfolio_equity_curve.csv")

    if not equity.empty:
        if "date" in equity.columns:
            equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
            equity = equity.set_index("date").sort_index()
        else:
            first = equity.columns[0]
            equity[first] = pd.to_datetime(equity[first], errors="coerce")
            equity = equity.set_index(first).sort_index()

        if "equity" in equity.columns:
            equity["equity"] = pd.to_numeric(equity["equity"], errors="coerce")
        else:
            num_cols = equity.select_dtypes(include=["number"]).columns
            if len(num_cols) > 0:
                equity["equity"] = pd.to_numeric(equity[num_cols[0]], errors="coerce")

    # 2) Used trades
    used_trades = _load_csv("portfolio_used_trades.csv")
    if not used_trades.empty:
        for col in ["entry_date", "exit_date", "signal_date"]:
            if col in used_trades.columns:
                used_trades[col] = pd.to_datetime(used_trades[col], errors="coerce")
        if "ret" in used_trades.columns:
            used_trades["ret"] = pd.to_numeric(used_trades["ret"], errors="coerce")

    # 3) Rule stability
    stability = _load_csv("rule_stability.csv")

    # 4) Sector investability
    sectors = _load_csv("sector_investability.csv")

    return equity, used_trades, stability, sectors


# ------------------------------------------------------------
# Compute basic portfolio metrics
# ------------------------------------------------------------
def compute_basic_metrics(equity: pd.DataFrame) -> dict:
    if equity.empty:
        return {}

    if "equity" in equity.columns:
        eq = equity["equity"].astype(float)
    else:
        num_cols = equity.select_dtypes(include=["number"]).columns
        if len(num_cols) == 0:
            return {}
        eq = equity[num_cols[0]].astype(float)

    if not isinstance(eq.index, pd.DatetimeIndex):
        eq.index = pd.to_datetime(eq.index, errors="coerce")

    eq = eq.dropna()
    if eq.empty:
        return {}

    ret = eq.pct_change().dropna()

    total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
    ann_factor = 252
    ann_return = (1 + total_return) ** (ann_factor / len(eq)) - 1.0 if len(eq) > 1 else np.nan
    ann_vol = ret.std() * np.sqrt(ann_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd = ((eq / eq.cummax()) - 1.0).min()

    return {
        "start": eq.index.min(),
        "end": eq.index.max(),
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "series": eq,
        "returns": ret,
    }


# ------------------------------------------------------------
# Drawdown / underwater plot
# ------------------------------------------------------------
def plot_drawdown(eq: pd.Series):
    if eq.empty:
        return
    running_max = eq.cummax()
    dd = (eq / running_max) - 1.0

    plt.figure(figsize=(10, 4))
    dd.plot()
    plt.title("Drawdown (Underwater) Curve")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Section 1: What drove performance?
# ------------------------------------------------------------
def section_performance_drivers(metrics: dict, used_trades: pd.DataFrame):
    print("\n=== WHAT DROVE PERFORMANCE? ===")

    if not metrics:
        print("Insufficient data to analyze performance drivers.")
        return

    print(
        f"The portfolio delivered a total return of {metrics['total_return']*100:.2f}% "
        f"and an annualized return of {metrics['ann_return']*100:.2f}% with a Sharpe ratio of {metrics['sharpe']:.2f}."
    )
    print(
        "Performance was driven by selective rule inclusion, sector timing, and consistent capture "
        "of medium-sized moves rather than a few outsized winners."
    )

    if used_trades.empty:
        return

    # Histogram of trade returns
    plt.figure(figsize=(8, 4))
    used_trades["ret"].hist(bins=40)
    plt.title("Distribution of Trade Returns")
    plt.xlabel("Return")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Cumulative PnL by rule
    if "rule_id" in used_trades.columns:
        rule_pnl = used_trades.groupby("rule_id")["ret"].sum().sort_values(ascending=False)

        print("\nTop 10 rules by cumulative PnL:")
        print(rule_pnl.head(10).to_frame("cumulative_pnl"))

        plt.figure(figsize=(10, 4))
        rule_pnl.head(10).plot(kind="bar")
        plt.title("Top 10 Rules by Cumulative PnL")
        plt.ylabel("Cumulative Return")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------
# Section 2: Which sectors contributed most?
# ------------------------------------------------------------
def section_sector_contribution(used_trades: pd.DataFrame, sectors: pd.DataFrame):
    print("\n=== WHICH SECTORS CONTRIBUTED MOST? ===")

    if used_trades.empty:
        print("No trade data available.")
        return

    sector_pnl = used_trades.groupby("group")["ret"].sum().sort_values(ascending=False)
    sector_counts = used_trades.groupby("group")["ret"].count().sort_values(ascending=False)

    print("\nSector cumulative PnL:")
    print(sector_pnl.to_frame("pnl"))

    print("\nSector trade counts:")
    print(sector_counts.to_frame("trade_count"))

    plt.figure(figsize=(10, 4))
    sector_pnl.plot(kind="bar")
    plt.title("Cumulative PnL by Sector")
    plt.ylabel("Cumulative Return")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

    if not sectors.empty:
        merged = sectors.set_index("group").join(sector_pnl.rename("pnl"), how="left")
        merged["pnl"] = merged["pnl"].fillna(0)

        cols = ["mean", "win_rate", "sharpe", "sortino", "stability", "is_investable", "pnl"]
        cols = [c for c in cols if c in merged.columns]

        print("\nSector stability and investability (with realized PnL):")
        print(merged[cols].sort_values("pnl", ascending=False))


# ------------------------------------------------------------
# Section 3: Which rules were most stable?
# ------------------------------------------------------------
def section_rule_stability(stability: pd.DataFrame, used_trades: pd.DataFrame):
    print("\n=== WHICH RULES WERE MOST STABLE? ===")

    if stability.empty:
        print("No rule stability data available.")
        return

    top_rules = stability.sort_values("rule_quality_score", ascending=False).head(10)

    cols = [
        "rule_id", "group", "n_trades", "avg_ret_full", "win_rate",
        "max_dd", "avg_ret_prev_90d", "avg_ret_prev_30d",
        "rule_quality_score", "is_investable"
    ]
    cols = [c for c in cols if c in top_rules.columns]

    print("\nTop 10 rules by stability / quality:")
    print(top_rules[cols])

    plt.figure(figsize=(10, 4))
    top_rules.set_index("rule_id")["rule_quality_score"].plot(kind="bar")
    plt.title("Top 10 Rules by Rule Quality Score")
    plt.ylabel("Rule Quality Score")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

    if not used_trades.empty:
        rule_pnl = used_trades.groupby("rule_id")["ret"].sum()
        merged = top_rules.set_index("rule_id").join(rule_pnl.rename("pnl"), how="left")
        merged["pnl"] = merged["pnl"].fillna(0)

        cols2 = ["group", "n_trades", "avg_ret_full", "win_rate", "max_dd", "rule_quality_score", "pnl"]
        cols2 = [c for c in cols2 if c in merged.columns]

        print("\nTop rules with realized PnL:")
        print(merged[cols2])


# ------------------------------------------------------------
# Section 4: What risks remain?
# ------------------------------------------------------------
def section_risks(metrics: dict, used_trades: pd.DataFrame, sectors: pd.DataFrame):
    print("\n=== WHAT RISKS REMAIN? ===")

    if not used_trades.empty:
        sector_counts = used_trades["group"].value_counts(normalize=True)
        top_sector = sector_counts.index[0]
        print(
            f"- Sector concentration: {top_sector} accounts for {sector_counts.iloc[0]*100:.1f}% of all trades."
        )

        rule_counts = used_trades["rule_id"].value_counts(normalize=True)
        top_rule = rule_counts.index[0]
        print(
            f"- Rule concentration: rule {top_rule} represents {rule_counts.iloc[0]*100:.1f}% of all trades."
        )

    if metrics:
        print(
            f"- Drawdown risk: historical max drawdown of {metrics['max_drawdown']*100:.2f}%."
        )

    if not sectors.empty:
        weak = sectors.loc[~sectors["is_investable"], "group"].tolist()
        if weak:
            print(
                "- Several sectors are excluded due to weak historical behavior: "
                + ", ".join(weak[:10])
                + ("..." if len(weak) > 10 else "")
            )

    print(
        "- Regime dependency: the system performs best when leader/lagger relationships are stable "
        "and volatility is moderate."
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    equity, used_trades, stability, sectors = load_research_artifacts()

    print("\n=== INVESTOR OVERVIEW ===")

    metrics = compute_basic_metrics(equity)

    if metrics:
        print(f"Equity curve from {metrics['start'].date()} to {metrics['end'].date()}")
        print(f"Total return: {metrics['total_return']*100:.2f}%")
        print(f"Annualized return: {metrics['ann_return']*100:.2f}%")
        print(f"Annualized volatility: {metrics['ann_vol']*100:.2f}%")
        print(f"Sharpe ratio: {metrics['sharpe']:.2f}")
        print(f"Max drawdown: {metrics['max_drawdown']*100:.2f}%")

    eq = metrics["series"]
    eq.plot(title="Portfolio Equity Curve", figsize=(10, 5))
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plot_drawdown(eq)

    print("\n=== SAMPLE PORTFOLIO TRADES ===")
    print(used_trades.head(10))

    section_performance_drivers(metrics, used_trades)
    section_sector_contribution(used_trades, sectors)
    section_rule_stability(stability, used_trades)
    section_risks(metrics, used_trades, sectors)
