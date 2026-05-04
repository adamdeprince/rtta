#!/usr/bin/env python
"""Run MatchedFlowConformalSignal on 5-minute massive_speedup aggregates."""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path

import massive_speedup
import rtta


def date_range(start: dt.date, stop: dt.date):
    for offset in range((stop - start).days + 1):
        yield start + dt.timedelta(days=offset)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--stop-date", required=True)
    parser.add_argument("--market-cap", type=float, default=float("nan"))
    parser.add_argument("--normal-dollar-alpha", type=float, default=0.05)
    parser.add_argument("--horizon-bars", type=int, default=12)
    parser.add_argument("--calibration-window", type=int, default=250)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--min-estimated-return", type=float, default=0.0)
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    stop_date = dt.date.fromisoformat(args.stop_date)
    if stop_date < start_date:
        parser.error("--stop-date must be on or after --start-date")

    signal = rtta.MatchedFlowConformalSignal(
        horizon_bars=args.horizon_bars,
        calibration_window=args.calibration_window,
        fillna=True,
    )
    normal_dollar_volume = float("nan")
    cash = float(args.initial_capital)
    inventory = 0.0
    trade_count = 0

    print(
        "date,window_start,open,high,low,close,volume,prediction,radius,score,"
        "signal,target_fraction,max_trade_dollars,realized_error,estimated_return,"
        "inventory,cash,equity,estimated_equity,trade_count"
    )
    for current_date in date_range(start_date, stop_date):
        try:
            trades = massive_speedup.StockTradeDatabase(
                args.database,
                current_date.isoformat(),
                args.symbol,
            )
        except Exception:
            continue

        first_bar = True
        for bar in massive_speedup.StockTradeAggregator(trades, interval_seconds=300):
            if bar.transactions == 0 or bar.volume == 0:
                continue

            result = signal.update(
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(bar.volume),
                normal_dollar_volume=normal_dollar_volume,
                market_cap=args.market_cap,
                reset_session=first_bar,
            )
            dollar_volume = float(bar.dollar_volume)
            if dollar_volume > 0.0:
                if normal_dollar_volume != normal_dollar_volume:
                    normal_dollar_volume = dollar_volume
                else:
                    normal_dollar_volume = (
                        args.normal_dollar_alpha * dollar_volume
                        + (1.0 - args.normal_dollar_alpha) * normal_dollar_volume
                    )

            close = float(bar.close)
            prediction = float(result.prediction)
            estimated_return = (
                prediction / close - 1.0
                if close > 0.0 and math.isfinite(prediction)
                else float("nan")
            )
            desired_dollars = 0.0
            if math.isfinite(estimated_return) and abs(estimated_return) >= args.min_estimated_return:
                desired_dollars = float(result.target_fraction) * args.initial_capital

            current_dollars = inventory * close
            order_dollars = desired_dollars - current_dollars
            max_trade_dollars = float(result.max_trade_dollars)
            if math.isfinite(max_trade_dollars) and max_trade_dollars > 0.0:
                order_dollars = max(-max_trade_dollars, min(max_trade_dollars, order_dollars))

            if close > 0.0 and math.isfinite(order_dollars) and abs(order_dollars) > 0.0:
                shares = order_dollars / close
                inventory += shares
                cash -= shares * close
                trade_count += 1

            equity = cash + inventory * close
            estimated_equity = (
                cash + inventory * prediction
                if math.isfinite(prediction)
                else float("nan")
            )

            print(
                f"{current_date.isoformat()},{int(bar.window_start)},"
                f"{float(bar.open):.6f},{float(bar.high):.6f},{float(bar.low):.6f},"
                f"{close:.6f},{int(bar.volume)},"
                f"{prediction:.10g},{float(result.radius):.10g},"
                f"{float(result.score):.10g},{float(result.signal):.0f},"
                f"{float(result.target_fraction):.10g},"
                f"{float(result.max_trade_dollars):.10g},"
                f"{float(result.realized_error):.10g},"
                f"{estimated_return:.10g},{inventory:.10g},{cash:.10g},"
                f"{equity:.10g},{estimated_equity:.10g},{trade_count}"
            )
            first_bar = False

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
