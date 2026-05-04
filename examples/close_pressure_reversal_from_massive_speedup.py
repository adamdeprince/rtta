#!/usr/bin/env python
"""Run ClosePressureReversalSignal cross-sectionally on massive_speedup aggregates."""

from __future__ import annotations

import argparse
import datetime as dt
import heapq
import math
from pathlib import Path

import massive_speedup
import rtta


# Populate this with the production universe, e.g. 100-300 liquid stock symbols.
SYMBOLS: list[str] = []

NANOSECONDS_PER_SECOND = 1_000_000_000


def date_range(start: dt.date, stop: dt.date):
    for offset in range((stop - start).days + 1):
        yield start + dt.timedelta(days=offset)


def load_symbols(args) -> list[str]:
    symbols = list(SYMBOLS)
    if args.symbols:
        symbols.extend(symbol.strip() for symbol in args.symbols.split(",") if symbol.strip())
    if args.symbols_file:
        with args.symbols_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                symbol = line.strip()
                if symbol and not symbol.startswith("#"):
                    symbols.append(symbol)

    seen = set()
    unique = []
    for symbol in symbols:
        if symbol not in seen:
            unique.append(symbol)
            seen.add(symbol)
    if not unique:
        raise SystemExit("populate SYMBOLS or pass --symbols/--symbols-file")
    return unique


def quote_at_or_after(quotes, timestamp_ns: int):
    quote_index = quotes.index_before_timestamp(timestamp_ns)
    if quote_index < 0:
        quote_index = 0
    elif quotes[quote_index].sip_timestamp < timestamp_ns:
        quote_index += 1
    if quote_index >= len(quotes):
        return None
    return quotes[quote_index]


def push_next(heap, symbol, iterator):
    try:
        bar = next(iterator)
    except StopIteration:
        return
    heapq.heappush(heap, (int(bar.window_start), symbol, bar, iterator))


def aggregate_groups(trade_databases, *, interval_seconds: int, offset_seconds: int):
    heap = []
    for symbol, trades in trade_databases.items():
        iterator = iter(
            massive_speedup.StockTradeAggregator(
                trades,
                interval_seconds=interval_seconds,
                offset_seconds=offset_seconds,
            )
        )
        push_next(heap, symbol, iterator)

    while heap:
        window_start = heap[0][0]
        bars = {}
        while heap and heap[0][0] == window_start:
            _, symbol, bar, iterator = heapq.heappop(heap)
            bars[symbol] = bar
            push_next(heap, symbol, iterator)
        yield window_start, bars


def finite_score(result) -> float:
    score = float(result.score)
    if math.isfinite(score):
        return score
    return -math.inf


def current_portfolio_value(cash: float, holdings: set[str], latest_close: dict[str, float]) -> float:
    return cash + sum(latest_close.get(symbol, 0.0) for symbol in holdings)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--symbols")
    parser.add_argument("--symbols-file", type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--stop-date", required=True)
    parser.add_argument("--trade-delay-ns", type=int, default=150_000_000)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--cutoff-after-bars", type=int, default=66)
    parser.add_argument("--entry-start-after-bars", type=int, default=72)
    parser.add_argument("--entry-end-after-bars", type=int, default=75)
    parser.add_argument("--exit-after-bars", type=int, default=77)
    parser.add_argument("--calibration-window", type=int, default=120)
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    stop_date = dt.date.fromisoformat(args.stop_date)
    if stop_date < start_date:
        parser.error("--stop-date must be on or after --start-date")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if not (0.0 < args.top_fraction <= 1.0):
        parser.error("--top-fraction must be in (0, 1]")

    symbols = load_symbols(args)
    indicators = {
        symbol: rtta.ClosePressureReversalSignal(
            cutoff_after_bars=args.cutoff_after_bars,
            entry_start_after_bars=args.entry_start_after_bars,
            entry_end_after_bars=args.entry_end_after_bars,
            exit_after_bars=args.exit_after_bars,
            calibration_window=args.calibration_window,
            fillna=True,
        )
        for symbol in symbols
    }
    first_bar = {symbol: True for symbol in symbols}
    previous_session_close: dict[str, float] = {}
    latest_close: dict[str, float] = {}
    first_seen_price: dict[str, float] = {}
    holdings: set[str] = set()
    cash = 0.0
    trade_count = 0

    print(
        "date,window_start,symbol,rank,selected,open,high,low,close,volume,"
        "rod_return,prediction,radius,score,signal,target_fraction,"
        "entry_window,exit_window,news_guard,execution_timestamp,execution_bid,"
        "execution_ask,action,cash_delta,cash,current_value,"
        "current_value_pct_initial,holding,trade_count"
    )

    interval_ns = args.interval * NANOSECONDS_PER_SECOND
    for current_date in date_range(start_date, stop_date):
        trade_databases = {}
        quote_databases = {}
        for symbol in symbols:
            try:
                trades = massive_speedup.StockTradeDatabase(args.database, current_date.isoformat(), symbol)
                quotes = massive_speedup.StockQuoteDatabase(args.database, current_date.isoformat(), symbol)
            except Exception:
                continue
            if len(quotes) == 0:
                continue
            trade_databases[symbol] = trades
            quote_databases[symbol] = quotes

        if not trade_databases:
            continue

        for window_start, bars in aggregate_groups(
            trade_databases,
            interval_seconds=args.interval,
            offset_seconds=args.offset,
        ):
            outputs = {}
            for symbol, bar in bars.items():
                if bar.transactions == 0 or bar.volume == 0:
                    continue

                close = float(bar.close)
                if close > 0.0:
                    latest_close[symbol] = close
                    first_seen_price.setdefault(symbol, close)

                outputs[symbol] = indicators[symbol].update(
                    float(bar.open),
                    float(bar.high),
                    float(bar.low),
                    close,
                    float(bar.volume),
                    vwap=float(bar.volume_weighted_avg),
                    transactions=float(bar.transactions),
                    previous_session_close=previous_session_close.get(symbol, float("nan")),
                    reset_session=first_bar.get(symbol, True),
                )
                first_bar[symbol] = False
                previous_session_close[symbol] = close

            candidates = [
                (symbol, result)
                for symbol, result in outputs.items()
                if float(result.signal) > 0.0 and not bool(result.news_guard)
            ]
            candidates.sort(key=lambda item: finite_score(item[1]), reverse=True)
            selected_count = math.ceil(len(candidates) * args.top_fraction)
            selected = {symbol for symbol, _ in candidates[:selected_count]}
            ranks = {symbol: index + 1 for index, (symbol, _) in enumerate(candidates)}

            for symbol, result in sorted(outputs.items()):
                bar = bars[symbol]
                close = float(bar.close)
                quotes = quote_databases.get(symbol)
                execution_timestamp = int(window_start) + interval_ns + args.trade_delay_ns
                delayed_quote = quote_at_or_after(quotes, execution_timestamp) if quotes is not None else None
                execution_bid = float("nan")
                execution_ask = float("nan")
                cash_delta = 0.0
                action = "none"

                if delayed_quote is not None:
                    execution_bid = float(delayed_quote.bid_price)
                    execution_ask = float(delayed_quote.ask_price)
                    if symbol in holdings and symbol not in selected and execution_bid > 0.0:
                        cash_delta = execution_bid
                        cash += cash_delta
                        holdings.remove(symbol)
                        action = "sell"
                    elif symbol not in holdings and symbol in selected and execution_ask > 0.0:
                        cash_delta = -execution_ask
                        cash += cash_delta
                        holdings.add(symbol)
                        trade_count += 1
                        action = "buy"

                if cash_delta == 0.0:
                    execution_timestamp = 0
                initial_reference_value = sum(first_seen_price.get(symbol, 0.0) for symbol in first_seen_price)
                current_value = current_portfolio_value(cash, holdings, latest_close)
                current_value_pct_initial = (
                    current_value / initial_reference_value
                    if initial_reference_value > 0.0
                    else float("nan")
                )

                print(
                    f"{current_date.isoformat()},{window_start},{symbol},"
                    f"{ranks.get(symbol, 0)},{int(symbol in selected)},"
                    f"{float(bar.open):.6f},{float(bar.high):.6f},"
                    f"{float(bar.low):.6f},{close:.6f},{int(bar.volume)},"
                    f"{float(result.rod_return):.10g},{float(result.prediction):.10g},"
                    f"{float(result.radius):.10g},{float(result.score):.10g},"
                    f"{float(result.signal):.0f},{float(result.target_fraction):.10g},"
                    f"{int(bool(result.entry_window))},{int(bool(result.exit_window))},"
                    f"{int(bool(result.news_guard))},{execution_timestamp},"
                    f"{execution_bid:.10g},{execution_ask:.10g},{action},"
                    f"{cash_delta:.10g},{cash:.10g},{current_value:.10g},"
                    f"{current_value_pct_initial:.10g},{int(symbol in holdings)},"
                    f"{trade_count}"
                )

        for symbol in sorted(list(holdings)):
            quotes = quote_databases.get(symbol)
            if quotes is None or len(quotes) == 0:
                continue
            final_quote = quotes[len(quotes) - 1]
            execution_bid = float(final_quote.bid_price)
            if execution_bid <= 0.0:
                continue
            execution_timestamp = int(final_quote.sip_timestamp)
            cash_delta = execution_bid
            cash += cash_delta
            holdings.remove(symbol)
            initial_reference_value = sum(first_seen_price.get(symbol, 0.0) for symbol in first_seen_price)
            current_value = current_portfolio_value(cash, holdings, latest_close)
            current_value_pct_initial = (
                current_value / initial_reference_value
                if initial_reference_value > 0.0
                else float("nan")
            )
            print(
                f"{current_date.isoformat()},closeout,{symbol},0,0,"
                f"nan,nan,nan,nan,0,nan,nan,nan,nan,0,nan,0,1,0,"
                f"{execution_timestamp},{execution_bid:.10g},nan,closeout,"
                f"{cash_delta:.10g},{cash:.10g},{current_value:.10g},"
                f"{current_value_pct_initial:.10g},0,{trade_count}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
