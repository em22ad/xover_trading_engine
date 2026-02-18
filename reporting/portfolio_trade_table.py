# trading_engine/reporting/portfolio_trade_table.py

import pandas as pd

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def print_portfolio_trade_table(used_trades: pd.DataFrame, equity_curve: pd.Series):
    """
    Color-coded table of trades used in the portfolio backtest.
    Now includes OPEN trades (exit_date = None), flagged clearly.
    """

    if used_trades.empty:
        print("\nNo trades used in portfolio.\n")
        return

    df = used_trades.copy()

    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")

    # Closed trades: compute PnL
    df["pnl_pct"] = df["ret"] * 100

    # Portfolio % Since First only for closed trades
    first_equity = equity_curve.iloc[0]
    df["portfolio_change_pct"] = None
    closed_mask = df["exit_date"].notna()

    if closed_mask.any():
        df.loc[closed_mask, "portfolio_change_pct"] = (
            (equity_curve.loc[df.loc[closed_mask, "exit_date"]].values - first_equity)
            / first_equity * 100
        )

    # Build printable table
    table = pd.DataFrame({
        "Rule": df["rule_id"],
        "Leaders": df["leaders"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x),
        "Lagger": df["ticker"],
        "Entry": df["entry_date"].dt.strftime("%Y-%m-%d"),
        "Exit": df["exit_date"].dt.strftime("%Y-%m-%d").fillna("OPEN"),
        "PnL %": df["pnl_pct"].round(2),
        "Portfolio % Since First": df["portfolio_change_pct"].round(2),
        "is_open": df["exit_date"].isna(),
    })

    print("\n=== PORTFOLIO TRADE-BY-TRADE TABLE ===\n")

    for _, row in table.iterrows():
        if row["is_open"]:
            # OPEN TRADE — no PnL, no portfolio % change
            print(
                f"{row['Rule']:6}  "
                f"{row['Leaders'][:25]:25}  "
                f"{row['Lagger']:6}  "
                f"{row['Entry']}  "
                f"{'OPEN':10}  "
                f"{'   n/a':>7}  "
                f"{'   n/a':>7}"
            )
        else:
            # CLOSED TRADE — color-coded PnL
            pnl = row["PnL %"]
            if pnl > 0:
                color = GREEN
            elif pnl < 0:
                color = RED
            else:
                color = RESET

            print(
                f"{row['Rule']:6}  "
                f"{row['Leaders'][:25]:25}  "
                f"{row['Lagger']:6}  "
                f"{row['Entry']}  "
                f"{row['Exit']:10}  "
                f"{color}{pnl:6.2f}%{RESET}  "
                f"{row['Portfolio % Since First']:6.2f}%"
            )

    print()


