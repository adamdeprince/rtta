#!/usr/bin/env python
"""Run IntradayClockEchoSignal on massive_speedup aggregate bars."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import massive_speedup
import rtta


# Populate this with the production universe, or pass --symbols/--symbols-file.
SYMBOLS: list[str] = []

NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class ClockBar:
    window_start: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    transactions: float
    market_return: float
    slot: int


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


def quote_at_or_before(quotes, timestamp_ns: int):
    quote_index = quotes.index_before_timestamp(timestamp_ns)
    if quote_index < 0:
        return None
    return quotes[quote_index]


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def slot_for_window(market_open: int, window_start: int, interval_ns: int, slots_per_session: int) -> int | None:
    slot = (int(window_start) - market_open) // interval_ns
    if slot < 0 or slot >= slots_per_session:
        return None
    return int(slot)


def trade_aggregates(trades, *, interval_seconds: int, offset_seconds: int):
    market_open = int(trades.market_open)
    market_close = int(trades.market_close)
    for bar in massive_speedup.StockTradeAggregator(
        trades,
        interval_seconds=interval_seconds,
        offset_seconds=offset_seconds,
    ):
        window_start = int(bar.window_start)
        if window_start > market_close:
            break
        if window_start >= market_open:
            yield bar


def market_returns_for_day(args, current_date: dt.date, interval_ns: int, slots_per_session: int) -> dict[int, float]:
    if not args.market_symbol:
        return {}
    try:
        trades = massive_speedup.StockTradeDatabase(args.database, current_date.isoformat(), args.market_symbol)
    except Exception:
        return {}

    returns = {}
    previous_close = None
    market_open = int(trades.market_open)
    for bar in trade_aggregates(trades, interval_seconds=args.interval, offset_seconds=args.offset):
        if bar.transactions == 0 or bar.volume == 0:
            continue
        slot = slot_for_window(market_open, int(bar.window_start), interval_ns, slots_per_session)
        if slot is None:
            continue
        close = float(bar.close)
        returns[int(bar.window_start)] = (
            math.log(close / previous_close)
            if previous_close is not None and previous_close > 0.0 and close > 0.0
            else 0.0
        )
        previous_close = close
    return returns


def aggregate_day(trades, args, interval_ns: int, slots_per_session: int, market_returns: dict[int, float]):
    bars = []
    market_open = int(trades.market_open)
    for bar in trade_aggregates(trades, interval_seconds=args.interval, offset_seconds=args.offset):
        if bar.transactions == 0 or bar.volume == 0:
            continue
        window_start = int(bar.window_start)
        slot = slot_for_window(market_open, window_start, interval_ns, slots_per_session)
        if slot is None:
            continue
        bars.append(
            ClockBar(
                window_start=window_start,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                vwap=float(bar.volume_weighted_avg),
                transactions=float(bar.transactions),
                market_return=market_returns.get(window_start, 0.0),
                slot=slot,
            )
        )
    return bars


def write_trade(
    writer,
    date: dt.date,
    symbol: str,
    side: str,
    entry_time: int,
    entry_price: float,
    exit_time: int,
    exit_price: float,
    daily_profit: float,
    total_profit: float,
) -> None:
    profit = exit_price - entry_price if side == "long" else entry_price - exit_price
    profit_pct = 100.0 * profit / entry_price if entry_price > 0.0 else float("nan")
    writer.writerow(
        [
            "trade",
            date.isoformat(),
            symbol,
            side,
            entry_time,
            fmt(entry_price),
            exit_time,
            fmt(exit_price),
            fmt(profit),
            fmt(profit_pct),
            fmt(daily_profit),
            fmt(total_profit),
        ]
    )


def write_daily_summary(writer, date: dt.date, daily_profit: float, total_profit: float) -> None:
    writer.writerow(["daily", date.isoformat(), "", "", "", "", "", "", "", "", fmt(daily_profit), fmt(total_profit)])


def write_total_summary(writer, total_profit: float) -> None:
    writer.writerow(["total", "", "", "", "", "", "", "", "", "", "", fmt(total_profit)])


def close_position(writer, current_date, symbols, quotes, positions, index, exit_time, exit_price, daily_profit, total_profit):
    side, entry_time, entry_price = positions.pop(index)
    profit = exit_price - entry_price if side == "long" else entry_price - exit_price
    daily_profit += profit
    total_profit += profit
    write_trade(
        writer,
        current_date,
        symbols[index],
        side,
        entry_time,
        entry_price,
        exit_time,
        exit_price,
        daily_profit,
        total_profit,
    )
    return daily_profit, total_profit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--symbols")
    parser.add_argument("--symbols-file", type=Path)
    parser.add_argument("--market-symbol", default="")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--stop-date", required=True)
    parser.add_argument("--training-days", type=int, default=40)
    parser.add_argument("--trade-delay-ns", type=int, default=150_000_000)
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--slots-per-session", type=int, default=0)
    parser.add_argument("--horizon-bars", type=int, default=15)
    parser.add_argument("--lookback-days", type=int, default=40)
    parser.add_argument("--min-slot-samples", type=int)
    parser.add_argument("--calibration-window", type=int, default=500)
    parser.add_argument("--entry-z", type=float, default=1.0)
    parser.add_argument("--cost-buffer", type=float, default=0.0005)
    parser.add_argument("--max-abs-target-fraction", type=float, default=0.03)
    parser.add_argument("--participation-cap", type=float, default=0.02)
    parser.add_argument("--max-names", type=int, default=10)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--fillna", action="store_true")
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    stop_date = dt.date.fromisoformat(args.stop_date)
    if stop_date < start_date:
        parser.error("--stop-date must be on or after --start-date")
    if args.training_days <= 0:
        parser.error("--training-days must be positive")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.max_names <= 0:
        parser.error("--max-names must be positive")
    if args.min_slot_samples is not None and args.min_slot_samples <= 0:
        parser.error("--min-slot-samples must be positive")

    symbols = load_symbols(args)
    min_slot_samples = args.min_slot_samples or min(10, args.training_days)
    interval_ns = args.interval * NANOSECONDS_PER_SECOND
    slots_per_session = args.slots_per_session or max(1, int(round(390 * 60 / args.interval)))
    training_history: dict[int, list[list[ClockBar]]] = defaultdict(list)
    total_profit = 0.0
    loaded_days = 0

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        [
            "row_type",
            "date",
            "symbol",
            "side",
            "entry_time_ns",
            "entry_price",
            "exit_time_ns",
            "exit_price",
            "profit",
            "profit_pct",
            "daily_profit",
            "total_profit",
        ]
    )

    for current_date in date_range(start_date, stop_date):
        market_returns = market_returns_for_day(args, current_date, interval_ns, slots_per_session)
        quote_databases = {}
        trade_databases = {}
        day_bars: dict[int, list[ClockBar]] = {}
        for index, symbol in enumerate(symbols):
            try:
                trades = massive_speedup.StockTradeDatabase(args.database, current_date.isoformat(), symbol)
                quotes = massive_speedup.StockQuoteDatabase(args.database, current_date.isoformat(), symbol)
            except Exception:
                continue
            if len(quotes) == 0:
                continue
            bars = aggregate_day(trades, args, interval_ns, slots_per_session, market_returns)
            if not bars:
                continue
            trade_databases[index] = trades
            quote_databases[index] = quotes
            day_bars[index] = bars

        if not day_bars:
            continue
        loaded_days += 1

        can_trade_date = current_date >= start_date + dt.timedelta(days=args.training_days)
        indicators = {}
        if can_trade_date:
            for index in day_bars:
                history = training_history[index][-args.training_days :]
                if len(history) < args.training_days:
                    continue
                indicator = rtta.IntradayClockEchoSignal(
                    slots_per_session=slots_per_session,
                    horizon_bars=args.horizon_bars,
                    lookback_days=args.lookback_days,
                    min_slot_samples=min_slot_samples,
                    calibration_window=args.calibration_window,
                    entry_z=args.entry_z,
                    cost_buffer=args.cost_buffer,
                    max_abs_target_fraction=args.max_abs_target_fraction,
                    participation_cap=args.participation_cap,
                    allow_short=args.allow_short,
                    fillna=args.fillna,
                )
                indicator.train(history)
                indicators[index] = indicator

        daily_profit = 0.0
        positions: dict[int, tuple[str, int, float]] = {}
        if indicators:
            bars_by_window = defaultdict(list)
            for index, bars in day_bars.items():
                if index not in indicators:
                    continue
                for bar in bars:
                    bars_by_window[bar.window_start].append((index, bar))

            first_bar = {index: True for index in indicators}
            for window_start in sorted(bars_by_window):
                long_candidates = []
                short_candidates = []
                for index, bar in bars_by_window[window_start]:
                    result = indicators[index].update(
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                        vwap=bar.vwap,
                        transactions=bar.transactions,
                        market_return=bar.market_return,
                        slot=bar.slot,
                        reset_session=first_bar[index],
                    )
                    first_bar[index] = False
                    if result.signal > 0.0:
                        long_candidates.append((abs(float(result.score)), index))
                    elif args.allow_short and result.signal < 0.0:
                        short_candidates.append((abs(float(result.score)), index))

                long_candidates.sort(reverse=True)
                short_candidates.sort(reverse=True)
                desired = {index: "long" for _, index in long_candidates[: args.max_names]}
                if args.allow_short:
                    desired.update({index: "short" for _, index in short_candidates[: args.max_names]})

                execution_timestamp = int(window_start) + interval_ns + args.trade_delay_ns
                for index, (side, _, _) in list(positions.items()):
                    if desired.get(index) == side:
                        continue
                    quote = quote_at_or_after(quote_databases[index], execution_timestamp)
                    if quote is None:
                        continue
                    exit_price = float(quote.bid_price if side == "long" else quote.ask_price)
                    if exit_price <= 0.0:
                        continue
                    daily_profit, total_profit = close_position(
                        writer,
                        current_date,
                        symbols,
                        quote_databases[index],
                        positions,
                        index,
                        execution_timestamp,
                        exit_price,
                        daily_profit,
                        total_profit,
                    )

                for index, side in desired.items():
                    if index in positions:
                        continue
                    quote = quote_at_or_after(quote_databases[index], execution_timestamp)
                    if quote is None:
                        continue
                    entry_price = float(quote.ask_price if side == "long" else quote.bid_price)
                    if entry_price > 0.0:
                        positions[index] = (side, execution_timestamp, entry_price)

            for index, (side, _, _) in list(positions.items()):
                trades = trade_databases.get(index)
                quotes = quote_databases.get(index)
                if trades is None or quotes is None or len(quotes) == 0:
                    continue
                quote = quote_at_or_before(quotes, int(trades.market_close))
                if quote is None:
                    continue
                exit_price = float(quote.bid_price if side == "long" else quote.ask_price)
                if exit_price <= 0.0:
                    continue
                daily_profit, total_profit = close_position(
                    writer,
                    current_date,
                    symbols,
                    quotes,
                    positions,
                    index,
                    int(quote.sip_timestamp),
                    exit_price,
                    daily_profit,
                    total_profit,
                )

        write_daily_summary(writer, current_date, daily_profit, total_profit)

        for index, bars in day_bars.items():
            history = training_history[index]
            history.append(bars)
            if len(history) > args.training_days:
                del history[:-args.training_days]

    write_total_summary(writer, total_profit)
    if loaded_days == 0:
        print(
            f"warning: no trade/quote data found for {start_date.isoformat()} through {stop_date.isoformat()}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
