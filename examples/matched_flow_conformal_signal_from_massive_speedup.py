#!/usr/bin/env python
"""Run MatchedFlowConformalSignal on 5-minute massive_speedup aggregates."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import massive_speedup
import rtta


BAR_INTERVAL_SECONDS = 300
BAR_INTERVAL_NS = BAR_INTERVAL_SECONDS * 1_000_000_000


def date_range(start: dt.date, stop: dt.date):
    for offset in range((stop - start).days + 1):
        yield start + dt.timedelta(days=offset)


def quote_at_or_after(quotes, timestamp_ns: int):
    quote_index = quotes.index_before_timestamp(timestamp_ns)
    if quote_index < 0:
        quote_index = 0
    elif quotes[quote_index].sip_timestamp < timestamp_ns:
        quote_index += 1
    if quote_index >= len(quotes):
        return None
    return quotes[quote_index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--trade-delay-ns", type=int, default=150_000_000)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--stop-date", required=True)
    parser.add_argument("--market-cap", type=float, default=float("nan"))
    parser.add_argument("--normal-dollar-alpha", type=float, default=0.05)
    parser.add_argument("--horizon-bars", type=int, default=12)
    parser.add_argument("--calibration-window", type=int, default=250)
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
    holding = False
    profit = 0.0
    trade_count = 0

    print(
        "date,window_start,open,high,low,close,volume,prediction,radius,score,"
        "signal,target_fraction,max_trade_dollars,realized_error,execution_timestamp,"
        "execution_bid,execution_ask,profit_delta,profit,holding,trade_count"
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
        try:
            quotes = massive_speedup.StockQuoteDatabase(
                args.database,
                current_date.isoformat(),
                args.symbol,
            )
        except Exception:
            continue
        if len(quotes) == 0:
            continue

        first_bar = True
        for bar in massive_speedup.StockTradeAggregator(trades, interval_seconds=BAR_INTERVAL_SECONDS):
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
            direction = float(result.signal)
            execution_timestamp = int(bar.window_start) + BAR_INTERVAL_NS + args.trade_delay_ns
            delayed_quote = quote_at_or_after(quotes, execution_timestamp)
            execution_bid = float("nan")
            execution_ask = float("nan")
            profit_delta = 0.0

            if delayed_quote is not None:
                execution_bid = float(delayed_quote.bid_price)
                execution_ask = float(delayed_quote.ask_price)
                if holding:
                    if direction < 0.0 and execution_bid > 0.0:
                        profit_delta = execution_bid
                        profit += profit_delta
                        holding = False
                else:
                    if direction > 0.0 and execution_ask > 0.0:
                        profit_delta = -execution_ask
                        profit += profit_delta
                        holding = True
                        trade_count += 1
            if profit_delta == 0.0:
                execution_timestamp = 0

            print(
                f"{current_date.isoformat()},{int(bar.window_start)},"
                f"{float(bar.open):.6f},{float(bar.high):.6f},{float(bar.low):.6f},"
                f"{close:.6f},{int(bar.volume)},"
                f"{float(result.prediction):.10g},{float(result.radius):.10g},"
                f"{float(result.score):.10g},{direction:.0f},"
                f"{float(result.target_fraction):.10g},"
                f"{float(result.max_trade_dollars):.10g},"
                f"{float(result.realized_error):.10g},"
                f"{execution_timestamp},{execution_bid:.10g},{execution_ask:.10g},"
                f"{profit_delta:.10g},{profit:.10g},{int(holding)},{trade_count}"
            )
            first_bar = False

        if holding:
            final_quote = quotes[len(quotes) - 1]
            execution_timestamp = int(final_quote.sip_timestamp)
            execution_bid = float(final_quote.bid_price)
            if execution_bid > 0.0:
                profit_delta = execution_bid
                profit += profit_delta
                holding = False
                print(
                    f"{current_date.isoformat()},closeout,"
                    f"nan,nan,nan,nan,0,"
                    f"nan,nan,nan,0,nan,nan,nan,"
                    f"{execution_timestamp},{execution_bid:.10g},nan,"
                    f"{profit_delta:.10g},{profit:.10g},0,{trade_count}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
