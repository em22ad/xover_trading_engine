# trading_engine/backtest/trade_generator.py

import pandas as pd
from trading_engine.backtest.rule_scoring import RULE_KEY_COLS


def backtest_signals(series: pd.DataFrame, signals: list, scored_rules=None) -> pd.DataFrame:
    trades = []

    if not signals:
        return _empty_trades_df()

    # Unique parameter sets from signals
    unique_param_sets = {
        (
            sig["lookback"],
            sig["group_thresh"],
            sig["participation"],
            sig["lagger_max_move"],
        )
        for sig in signals
    }

    entry_lags = [0, 1, 2, 3]
    holds = [3, 5, 7, 10]

    adaptive_rule_grid = []
    for (lb, gt, part, lag) in unique_param_sets:
        for elag in entry_lags:
            for hold in holds:
                adaptive_rule_grid.append(
                    {
                        "lookback": lb,
                        "group_thresh": gt,
                        "participation": part,
                        "lagger_max_move": lag,
                        "entry_lag": elag,
                        "hold_days": hold,
                    }
                )

    for sig in signals:
        ticker = sig["ticker"]
        group = sig["group"]
        dir_sign = sig["direction"]
        signal_date = sig["signal_date"]

        if ticker not in series.columns:
            continue

        ticker_series = series[ticker]

        try:
            signal_loc = ticker_series.index.get_loc(signal_date)
        except KeyError:
            continue

        for params in adaptive_rule_grid:
            if (
                params["lookback"] != sig["lookback"]
                or params["group_thresh"] != sig["group_thresh"]
                or params["participation"] != sig["participation"]
                or params["lagger_max_move"] != sig["lagger_max_move"]
            ):
                continue

            entry_lag = params["entry_lag"]
            hold_days = params["hold_days"]

            entry_idx = signal_loc + entry_lag
            if entry_idx >= len(ticker_series):
                continue

            entry_date = ticker_series.index[entry_idx]
            exit_idx = entry_idx + hold_days

            if exit_idx < len(ticker_series):
                exit_date = ticker_series.index[exit_idx]
                entry_price = ticker_series.iloc[entry_idx]
                exit_price = ticker_series.iloc[exit_idx]
                ret = (exit_price / entry_price - 1.0) * dir_sign
            else:
                exit_date = None
                ret = None

            trades.append(
                {
                    "group": group,
                    "ticker": ticker,
                    "direction": dir_sign,
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_lag": entry_lag,
                    "hold_days": hold_days,
                    "lookback": sig["lookback"],
                    "group_thresh": sig["group_thresh"],
                    "participation": sig["participation"],
                    "lagger_max_move": sig["lagger_max_move"],
                    "ret": ret,
                    "leaders": ",".join(sig["leaders"]),
                    "is_open": exit_date is None,
                }
            )

    df = pd.DataFrame(trades)
    if df.empty:
        return _empty_trades_df()

    # Attach rule_id + rule_quality_score if provided
    if scored_rules is not None and not scored_rules.empty:
        merge_cols = RULE_KEY_COLS + ["rule_id", "rule_quality_score"]
        df = df.merge(
            scored_rules[merge_cols],
            on=RULE_KEY_COLS,
            how="left",
        )

    return df


def _empty_trades_df():
    return pd.DataFrame(
        columns=[
            "group",
            "ticker",
            "direction",
            "signal_date",
            "entry_date",
            "exit_date",
            "entry_lag",
            "hold_days",
            "lookback",
            "group_thresh",
            "participation",
            "lagger_max_move",
            "ret",
            "leaders",
            "is_open",
            "rule_id",
            "rule_quality_score",
        ]
    )
