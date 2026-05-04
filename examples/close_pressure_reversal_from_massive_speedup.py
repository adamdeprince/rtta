#!/usr/bin/env python
"""Run ClosePressureReversalSignal cross-sectionally on massive_speedup aggregates."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import heapq
import math
import sys
from pathlib import Path

import massive_speedup
import numpy as np
import rtta


# Populate this with the production universe, e.g. 100-300 liquid stock symbols.
SYMBOLS: list[str] = [
    'TNA', 'TQQQ', 'DIA', 'XLV', 'VOO', 'SOXX', 'RSP', 'IVV', 'UDOW', 'USO', 'XLI',
    'UCO', 'QQQM', 'SPXU', 'VTV', 'TSLA', 'ARKK', 'IAU', 'QLD', 'KRE', 'EFA',
    'XLY', 'MSTZ', 'SIVR', 'VTWO', 'IBIT', 'FBTC', 'SPXS', 'GOOGL', 'SPXL', 'ITOT',
    'VXX', 'NVDL', 'AMZU', 'VT', 'IGV', 'IVW', 'EMXC', 'GLDM', 'TSLL', 'GOOG',
    'BNO', 'TSLQ', 'SDS', 'XBI', 'SMH', 'ACWI', 'PLTR', 'NFLX', 'ETHU', 'INTC',
    'XLC', 'IXUS', 'AMZN', 'SPYG', 'GBTC', 'AAPL', 'IEMG', 'IYR', 'VIXY', 'SDOW',
    'GLD', 'IEFA', 'AAPU', 'SPYM', 'IWD', 'ZSL', 'URTY', 'SSO', 'SOXQ', 'VNQ',
    'ETHA', 'FNGU', 'SVXY', 'EWJ', 'UPRO', 'BITX', 'EFG', 'MAGS', 'VONG', 'UGL',
    'BITB', 'USMV', 'XLE', 'TECS', 'GDX', 'IJH', 'XLF', 'VTI', 'BBJP', 'PTIR',
    'SCZ', 'SILJ', 'TLH', 'SRTY', 'LABD', 'VEA', 'EFV', 'KBE', 'XRT', 'TSLT',
    'EEM', 'IWR', 'BMNR', 'VGT', 'ETH', 'AMDL', 'MOAT', 'METU', 'ACWX', 'IYW',
    'ARKG', 'PFSA', 'QID', 'TECL', 'VPL', 'AIXI', 'VGK', 'ETHE', 'DGRO', 'KOLD',
    'VXUS', 'XLP', 'EWZ', 'TSLR', 'NVDQ', 'IUSG', 'MSFU', 'BTC', 'SH', 'FETH',
    'OEF', 'TSLZ', 'MU', 'DYNF', 'IREN', 'AVDE', 'PLTZ', 'EWY', 'IWB', 'SPHQ',
    'UVIX', 'USB', 'AG', 'AVEM', 'MDY', 'SBIT', 'SPTM', 'PSLV', 'RDVY', 'SPMD',
    'SVIX', 'TZA', 'ARKB', 'GLL', 'EWT', 'TSDD', 'NVDX', 'FBL', 'IAUM', 'IYK',
    'IDEV', 'SCHG', 'XLG', 'COWZ', 'QBTS', 'YANG', 'KGC', 'SGOL', 'XLB', 'FCX',
    'ILF', 'STM', 'IYE', 'HODL', 'FEZ', 'VIG', 'BOIL', 'RGTI', 'UWM', 'BITO',
    'BITU', 'TWM', 'IUSV', 'TSLS', 'HIBS', 'VUG', 'NOBL', 'SPYU', 'HOOD', 'DGRW',
    'PAVE', 'XOM', 'SPEM', 'VYMI', 'PAAS', 'KBWB', 'IVE', 'IWY', 'VLUE', 'IVES',
    'CIFR', 'DFUS', 'SMCI', 'SARK', 'EWQ', 'BAC', 'YINN', 'PLTU', 'CDE', 'SPSM',
    'IWF', 'SPYV', 'EZU', 'WULF', 'XLRE', 'BITI', 'QUAL', 'LUNR', 'JEPQ', 'FDL',
    'DFUV', 'COMT', 'TLT', 'ETHW', 'WCLD', 'MRVL', 'BULZ', 'IYH', 'CNQ', 'FPX',
    'ORCX', 'VDE', 'GGLL', 'HIMS', 'AIQ', 'MGK', 'PSQ', 'IONQ', 'EWU', 'EUFN',
    'AGQ', 'SPLV', 'EWC', 'PHYS', 'ESGU', 'RWM', 'VEU', 'ETHV', 'HL', 'IEUR', 'B',
    'CONL', 'THRO', 'SOFI', 'LABU', 'MSTU', 'AAAU', 'BTCO', 'CVE', 'DFAI', 'GSG',
    'EZBC', 'RDW', 'CSCO', 'DUST', 'PBW', 'IHI', 'CMCI', 'VWO', 'QQQI', 'TOPT',
    'EZET', 'EWW', 'SPYD', 'LQD', 'JQUA', 'TFC', 'HIBL', 'VYM', 'SPMO', 'CGGR',
    'KO', 'ETHT', 'RPG', 'RWR', 'NKE', 'TMF', 'DFAC', 'AIB'][:200]


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


def quote_at_or_before(quotes, timestamp_ns: int):
    quote_index = quotes.index_before_timestamp(timestamp_ns)
    if quote_index < 0:
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
        rows = trades.iterate_bounded(trades.market_open, trades.market_close)
        iterator = iter(
            massive_speedup.StockTradeAggregator(
                rows,
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


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def write_trade(
    writer,
    date: dt.date,
    symbol: str,
    buy_time: int,
    buy_price: float,
    sell_time: int,
    sell_price: float,
    daily_profit: float,
    total_profit: float,
) -> None:
    profit = sell_price - buy_price
    profit_pct = 100.0 * profit / buy_price if buy_price > 0.0 else float("nan")
    writer.writerow(
        [
            "trade",
            date.isoformat(),
            symbol,
            buy_time,
            fmt(buy_price),
            sell_time,
            fmt(sell_price),
            fmt(profit),
            fmt(profit_pct),
            fmt(daily_profit),
            fmt(total_profit),
        ]
    )


def write_daily_summary(writer, date: dt.date, daily_profit: float, total_profit: float) -> None:
    writer.writerow(
        [
            "daily",
            date.isoformat(),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            fmt(daily_profit),
            fmt(total_profit),
        ]
    )


def write_total_summary(writer, total_profit: float) -> None:
    writer.writerow(["total", "", "", "", "", "", "", "", "", "", fmt(total_profit)])


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
    universe = rtta.ClosePressureReversalUniverse(
        len(symbols),
        cutoff_after_bars=args.cutoff_after_bars,
        entry_start_after_bars=args.entry_start_after_bars,
        entry_end_after_bars=args.entry_end_after_bars,
        exit_after_bars=args.exit_after_bars,
        calibration_window=args.calibration_window,
        fillna=True,
    )
    positions: dict[int, tuple[int, float]] = {}
    total_profit = 0.0

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        [
            "row_type",
            "date",
            "symbol",
            "buy_time_ns",
            "buy_price",
            "sell_time_ns",
            "sell_price",
            "profit",
            "profit_pct",
            "daily_profit",
            "total_profit",
        ]
    )

    interval_ns = args.interval * NANOSECONDS_PER_SECOND
    for current_date in date_range(start_date, stop_date):
        trade_databases = {}
        quote_databases = {}
        for index, symbol in enumerate(symbols):
            try:
                trades = massive_speedup.StockTradeDatabase(args.database, current_date.isoformat(), symbol)
                quotes = massive_speedup.StockQuoteDatabase(args.database, current_date.isoformat(), symbol)
            except Exception:
                continue
            if len(quotes) == 0:
                continue
            trade_databases[index] = trades
            quote_databases[index] = quotes

        if not trade_databases:
            continue

        universe.begin_session(np.fromiter(trade_databases.keys(), dtype=np.int64))
        daily_profit = 0.0
        for window_start, bars in aggregate_groups(
            trade_databases,
            interval_seconds=args.interval,
            offset_seconds=args.offset,
        ):
            valid_bars = [(index, bar) for index, bar in bars.items() if bar.transactions != 0 and bar.volume != 0]
            if not valid_bars:
                continue

            indices = np.empty(len(valid_bars), dtype=np.int64)
            open_values = np.empty(len(valid_bars), dtype=np.float64)
            high_values = np.empty(len(valid_bars), dtype=np.float64)
            low_values = np.empty(len(valid_bars), dtype=np.float64)
            close_values = np.empty(len(valid_bars), dtype=np.float64)
            volume_values = np.empty(len(valid_bars), dtype=np.float64)
            vwap_values = np.empty(len(valid_bars), dtype=np.float64)
            transaction_values = np.empty(len(valid_bars), dtype=np.float64)
            for offset, (index, bar) in enumerate(valid_bars):
                indices[offset] = index
                open_values[offset] = float(bar.open)
                high_values[offset] = float(bar.high)
                low_values[offset] = float(bar.low)
                close_values[offset] = float(bar.close)
                volume_values[offset] = float(bar.volume)
                vwap_values[offset] = float(bar.volume_weighted_avg)
                transaction_values[offset] = float(bar.transactions)

            selected, exits = universe.update(
                indices,
                open_values,
                high_values,
                low_values,
                close_values,
                volume_values,
                vwap_values,
                transaction_values,
                args.top_fraction,
            )
            execution_timestamp = int(window_start) + interval_ns + args.trade_delay_ns

            for index in exits:
                index = int(index)
                if index not in positions:
                    continue
                quotes = quote_databases.get(index)
                if quotes is None or len(quotes) == 0:
                    continue
                delayed_quote = quote_at_or_after(quotes, execution_timestamp)
                if delayed_quote is None:
                    continue
                execution_bid = float(delayed_quote.bid_price)
                if execution_bid <= 0.0:
                    continue
                buy_time, buy_price = positions.pop(index)
                profit = execution_bid - buy_price
                daily_profit += profit
                total_profit += profit
                write_trade(
                    writer,
                    current_date,
                    symbols[index],
                    buy_time,
                    buy_price,
                    execution_timestamp,
                    execution_bid,
                    daily_profit,
                    total_profit,
                )

            for index in selected:
                index = int(index)
                if index in positions:
                    continue
                quotes = quote_databases.get(index)
                if quotes is None or len(quotes) == 0:
                    continue
                delayed_quote = quote_at_or_after(quotes, execution_timestamp)
                if delayed_quote is None:
                    continue
                execution_ask = float(delayed_quote.ask_price)
                if execution_ask > 0.0:
                    positions[index] = (execution_timestamp, execution_ask)

        for index in sorted(list(positions)):
            quotes = quote_databases.get(index)
            if quotes is None or len(quotes) == 0:
                continue
            trades = trade_databases.get(index)
            if trades is None:
                continue
            final_quote = quote_at_or_before(quotes, int(trades.market_close))
            if final_quote is None:
                continue
            execution_bid = float(final_quote.bid_price)
            if execution_bid <= 0.0:
                continue
            execution_timestamp = int(final_quote.sip_timestamp)
            buy_time, buy_price = positions.pop(index)
            profit = execution_bid - buy_price
            daily_profit += profit
            total_profit += profit
            write_trade(
                writer,
                current_date,
                symbols[index],
                buy_time,
                buy_price,
                execution_timestamp,
                execution_bid,
                daily_profit,
                total_profit,
            )

        write_daily_summary(writer, current_date, daily_profit, total_profit)

    write_total_summary(writer, total_profit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
