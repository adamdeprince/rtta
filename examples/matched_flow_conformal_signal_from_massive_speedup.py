#!/usr/bin/env python
"""Run MatchedFlowConformalSignal on 5-minute massive_speedup aggregates."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import massive_speedup
import rtta



def date_range(start: dt.date, stop: dt.date):
    for offset in range((stop - start).days + 1):
        yield start + dt.timedelta(days=offset)


def delayed_quote(quotes, timestamp_ns: int, delay_ns: int):
    execution_timestamp = timestamp_ns + delay_ns
    quote_index = quotes.index_before_timestamp(execution_timestamp)
    if quote_index < 0:
        quote_index = 0
    elif quotes[quote_index].sip_timestamp < execution_timestamp:
        quote_index += 1
    if quote_index >= len(quotes):
        return execution_timestamp, None
    return execution_timestamp, quotes[quote_index]


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
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--offset', type=int, default=0)
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
    cash = 0.0
    first_seen_price = None
    trade_count = 0

    print(
        "date,window_start,open,high,low,close,volume,prediction,radius,score,"
        "signal,target_fraction,max_trade_dollars,realized_error,execution_timestamp,"
        "execution_bid,execution_ask,cash_delta,cash,current_value,"
        "current_value_pct_initial,holding,trade_count"
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
        for bar in massive_speedup.StockTradeAggregator(trades, interval_seconds=args.interval, offset_seconds=args.offset):
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
            if first_seen_price is None and close > 0.0:
                first_seen_price = close
            direction = float(result.signal)
            execution_timestamp, execution_quote = delayed_quote(
                quotes,
                int(bar.window_start) + args.interval * 1_000_000_000,
                args.trade_delay_ns,
            )
            execution_bid = float("nan")
            execution_ask = float("nan")
            cash_delta = 0.0

            if execution_quote is not None:
                execution_bid = float(execution_quote.bid_price)
                execution_ask = float(execution_quote.ask_price)
                if holding:
                    if direction < 0.0 and execution_bid > 0.0:
                        cash_delta = execution_bid
                        cash += cash_delta
                        holding = False
                else:
                    if direction > 0.0 and execution_ask > 0.0:
                        cash_delta = -execution_ask
                        cash += cash_delta
                        holding = True
                        trade_count += 1
            if cash_delta == 0.0:
                execution_timestamp = 0
            current_value = cash + close if holding else cash
            current_value_pct_initial = (
                current_value / first_seen_price
                if first_seen_price is not None and first_seen_price > 0.0
                else float("nan")
            )

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
                f"{cash_delta:.10g},{cash:.10g},{current_value:.10g},"
                f"{current_value_pct_initial:.10g},{int(holding)},{trade_count}"
            )
            first_bar = False

        if holding:
            final_quote = quotes[len(quotes) - 1]
            execution_timestamp = int(final_quote.sip_timestamp)
            execution_bid = float(final_quote.bid_price)
            if execution_bid > 0.0:
                cash_delta = execution_bid
                cash += cash_delta
                holding = False
                current_value = cash
                current_value_pct_initial = (
                    current_value / first_seen_price
                    if first_seen_price is not None and first_seen_price > 0.0
                    else float("nan")
                )
                print(
                    f"{current_date.isoformat()},closeout,"
                    f"nan,nan,nan,nan,0,"
                    f"nan,nan,nan,0,nan,nan,nan,"
                    f"{execution_timestamp},{execution_bid:.10g},nan,"
                    f"{cash_delta:.10g},{cash:.10g},{current_value:.10g},"
                    f"{current_value_pct_initial:.10g},0,{trade_count}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
