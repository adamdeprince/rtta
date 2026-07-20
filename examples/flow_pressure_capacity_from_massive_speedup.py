#!/usr/bin/env python3
"""Trade FlowPressureCapacitySignal on causally merged Massive trades and quotes.

Why this entry is different
---------------------------
Raw queue imbalance says where displayed size sits. Raw trade imbalance says who
crossed the spread. Neither tells whether aggressive flow is consuming scarce
liquidity or being harmlessly absorbed by a queue that keeps refilling.

``rtta.FlowPressureCapacitySignal`` reconciles each pair of consecutive L1 quote
snapshots with the buyer- and seller-initiated volume printed between them. It
combines persistent queue imbalance, flow/opposing-capacity pressure, inferred
same-price replenishment, and liquidity withdrawal. The update is event-time,
causal, symmetric, allocation-free, and O(1).

Research motivation
-------------------
* Chang (2026), "Do Order-Book States Predict Passive-Buy Toxicity?", reports
  that directional flow relative to near-touch capacity is more informative
  than raw flow alone for short-horizon returns and adverse-selection risk.
* Nittur Anantha, Jain & Maiti (2025), "Order Book Filtration and Directional
  Signal Extraction at High Frequency", find that transient book activity
  weakens raw imbalance and trade-event imbalance has stronger causal alignment.
* Stoikov's microprice and recent high-resolution extensions motivate treating
  the weighted mid as a starting point and correcting it with book dynamics.

What the example measures
-------------------------
This script does two deliberately separate things:

1. Samples ``score`` without overlap finer than ``--sample-ms`` and resolves it
   against the midpoint at ``--horizon-ms``. It reports continuous score IC and
   active-signal signed edge/hit rate, alongside raw queue imbalance as a baseline.
2. Paper-trades the discrete ``signal`` with ``--latency-ms`` delay at the then
   visible ask (buy/cover) or bid (sell/short), so quoted spread and optional fees
   are paid. Orders are not filled at the signal-time midpoint.

The forward diagnostic observations may still overlap when ``sample-ms`` is
shorter than ``horizon-ms``; their mean is useful, but do not read it as an IID
t-statistic. Run multiple symbols, dates, horizons, and latency assumptions.

Massive changed stock quote sizes from round lots to shares in November 2025
and subsequently regenerated its historical flat files. The current files need
no conversion. For an older cached quote file whose sizes are still round lots,
pass ``--quote-size-multiplier 100`` so quote capacity and trade volume use the
same unit.

Examples
--------
::

    python examples/flow_pressure_capacity_from_massive_speedup.py \
        --database /path/to/massive-db --symbol AAPL \
        --start-date 2026-01-02 --stop-date 2026-03-31 --quiet

    # More selective entries, 250 ms latency, long/short:
    python examples/flow_pressure_capacity_from_massive_speedup.py ... \
        --entry-threshold 0.50 --latency-ms 250 --allow-short

CSV output contains signal changes and delayed fills only. Use ``--quiet`` for
the closing alpha/P&L summary alone.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import massive_speedup
import rtta


def date_range(start: dt.date, stop: dt.date):
    for offset in range((stop - start).days + 1):
        yield start + dt.timedelta(days=offset)


def valid_quote(quote) -> bool:
    bid = float(quote.bid_price)
    ask = float(quote.ask_price)
    return (
        math.isfinite(bid)
        and math.isfinite(ask)
        and bid > 0.0
        and ask > bid
        and int(quote.bid_size) > 0
        and int(quote.ask_size) > 0
    )


def classify_trade(trade, quote, previous_price, previous_side):
    """Classify a print using only the quote already known at its SIP time."""
    price = float(trade.price)
    size = float(trade.size)
    if int(trade.correction) != 0 or not math.isfinite(price) or price <= 0.0 or size <= 0.0:
        return 0.0, previous_price, previous_side

    bid = float(quote.bid_price)
    ask = float(quote.ask_price)
    midpoint = 0.5 * (bid + ask)
    if price >= ask:
        side = 1.0
    elif price <= bid:
        side = -1.0
    elif price > midpoint:
        side = 1.0
    elif price < midpoint:
        side = -1.0
    elif previous_price is not None and price > previous_price:
        side = 1.0
    elif previous_price is not None and price < previous_price:
        side = -1.0
    else:
        side = previous_side
    return side, price, side


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total2: float = 0.0
    positive: int = 0

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.total2 += value * value
        self.positive += value > 0.0

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    @property
    def hit_rate(self) -> float:
        return self.positive / self.count if self.count else float("nan")


@dataclass
class CorrelationStats:
    count: int = 0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    syy: float = 0.0
    sxy: float = 0.0

    def add(self, x: float, y: float) -> None:
        if not math.isfinite(x) or not math.isfinite(y):
            return
        self.count += 1
        self.sx += x
        self.sy += y
        self.sxx += x * x
        self.syy += y * y
        self.sxy += x * y

    @property
    def value(self) -> float:
        if self.count < 2:
            return float("nan")
        n = float(self.count)
        vx = n * self.sxx - self.sx * self.sx
        vy = n * self.syy - self.sy * self.sy
        if vx <= 0.0 or vy <= 0.0:
            return float("nan")
        return (n * self.sxy - self.sx * self.sy) / math.sqrt(vx * vy)


@dataclass(frozen=True)
class PendingForecast:
    due_ns: int
    midpoint: float
    score: float
    signal: float
    raw_queue: float


class ForwardDiagnostics:
    def __init__(self, horizon_ns: int, sample_ns: int, baseline_threshold: float):
        self.horizon_ns = horizon_ns
        self.sample_ns = sample_ns
        self.baseline_threshold = baseline_threshold
        self.pending = deque()
        self.next_sample_ns = None
        self.score_ic = CorrelationStats()
        self.queue_ic = CorrelationStats()
        self.signal_edge = RunningStats()
        self.queue_edge = RunningStats()
        self.resolved = 0

    def begin_session(self) -> None:
        self.pending.clear()
        self.next_sample_ns = None

    def advance(self, timestamp_ns: int, midpoint: float) -> None:
        while self.pending and self.pending[0].due_ns <= timestamp_ns:
            forecast = self.pending.popleft()
            move_bps = (midpoint / forecast.midpoint - 1.0) * 10_000.0
            self.score_ic.add(forecast.score, move_bps)
            self.queue_ic.add(forecast.raw_queue, move_bps)
            if forecast.signal != 0.0:
                self.signal_edge.add(forecast.signal * move_bps)
            if abs(forecast.raw_queue) >= self.baseline_threshold:
                queue_side = 1.0 if forecast.raw_queue > 0.0 else -1.0
                self.queue_edge.add(queue_side * move_bps)
            self.resolved += 1

    def sample(self, timestamp_ns: int, midpoint: float, result) -> None:
        if self.next_sample_ns is not None and timestamp_ns < self.next_sample_ns:
            return
        self.pending.append(
            PendingForecast(
                timestamp_ns + self.horizon_ns,
                midpoint,
                float(result.score),
                float(result.signal),
                float(result.raw_queue_imbalance),
            )
        )
        self.next_sample_ns = timestamp_ns + self.sample_ns


class PaperBook:
    """One-share long/short paper book with delayed marketable executions."""

    def __init__(self, latency_ns: int, allow_short: bool, max_spread_bps: float, fee_bps: float):
        self.latency_ns = latency_ns
        self.allow_short = allow_short
        self.max_spread_bps = max_spread_bps
        self.fee_rate = fee_bps / 10_000.0
        self.cash = 0.0
        self.position = 0.0
        self.desired = 0.0
        self.pending_due_ns = None
        self.entries = 0
        self.rebalances = 0
        self.entry_notional = 0.0

    def _target(self, signal: float, spread_bps: float) -> float:
        if spread_bps > self.max_spread_bps or signal == 0.0:
            return 0.0
        if signal > 0.0:
            return 1.0
        return -1.0 if self.allow_short else 0.0

    def request(self, timestamp_ns: int, signal: float, spread_bps: float) -> bool:
        target = self._target(signal, spread_bps)
        if target == self.desired:
            return False
        self.desired = target
        self.pending_due_ns = timestamp_ns + self.latency_ns
        return True

    def _fee(self, price: float) -> float:
        return price * self.fee_rate

    def on_quote(self, timestamp_ns: int, bid: float, ask: float):
        if self.pending_due_ns is None or timestamp_ns < self.pending_due_ns:
            return ""
        self.pending_due_ns = None
        target = self.desired
        if target == self.position:
            return ""

        actions = []
        if self.position > 0.0:
            self.cash += bid - self._fee(bid)
            actions.append("SELL")
        elif self.position < 0.0:
            self.cash -= ask + self._fee(ask)
            actions.append("COVER")

        if target > 0.0:
            self.cash -= ask + self._fee(ask)
            self.entry_notional += ask
            self.entries += 1
            actions.append("BUY")
        elif target < 0.0:
            self.cash += bid - self._fee(bid)
            self.entry_notional += bid
            self.entries += 1
            actions.append("SHORT")

        self.position = target
        self.rebalances += 1
        return "+".join(actions)

    def close_session(self, bid: float, ask: float) -> str:
        self.desired = 0.0
        self.pending_due_ns = None
        if self.position > 0.0:
            self.cash += bid - self._fee(bid)
            action = "SELL_CLOSE"
        elif self.position < 0.0:
            self.cash -= ask + self._fee(ask)
            action = "COVER_CLOSE"
        else:
            return ""
        self.position = 0.0
        self.rebalances += 1
        return action

    @property
    def deployment_return_bps(self) -> float:
        return (
            self.cash / self.entry_notional * 10_000.0
            if self.entry_notional > 0.0
            else float("nan")
        )


def first_valid_quote_index(quotes, start_ns, stop_ns):
    for index in range(len(quotes)):
        quote = quotes[index]
        timestamp = int(quote.sip_timestamp)
        if timestamp < start_ns:
            continue
        if timestamp > stop_ns:
            break
        if valid_quote(quote):
            return index
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Causally merge Massive L1 quotes and trades into RTTA's flow-versus-capacity "
            "signal, forward diagnostics, and delayed bid/ask paper execution."
        )
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--stop-date", required=True)
    parser.add_argument("--all-hours", action="store_true", help="Include pre/post market")
    parser.add_argument(
        "--quote-size-multiplier",
        type=float,
        default=1.0,
        help="Multiply quote sizes into shares; use 100 for pre-migration round-lot files",
    )
    parser.add_argument("--half-life-updates", type=float, default=32.0)
    parser.add_argument("--queue-weight", type=float, default=0.25)
    parser.add_argument("--pressure-weight", type=float, default=1.0)
    parser.add_argument("--replenishment-weight", type=float, default=0.75)
    parser.add_argument("--fragility-weight", type=float, default=0.25)
    parser.add_argument("--score-scale", type=float, default=1.0)
    parser.add_argument("--entry-threshold", type=float, default=0.35)
    parser.add_argument("--exit-threshold", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--horizon-ms", type=float, default=1_000.0)
    parser.add_argument("--sample-ms", type=float, default=100.0)
    parser.add_argument("--baseline-threshold", type=float, default=0.50)
    parser.add_argument("--latency-ms", type=float, default=150.0)
    parser.add_argument("--max-spread-bps", type=float, default=3.0)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--max-quotes", type=int, default=0, help="Debug cap; 0 means unlimited")
    parser.add_argument("--quiet", action="store_true", help="Print summary only")
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    stop_date = dt.date.fromisoformat(args.stop_date)
    if stop_date < start_date:
        parser.error("--stop-date must be on or after --start-date")
    for name in ("horizon_ms", "sample_ms", "latency_ms"):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if not math.isfinite(args.quote_size_multiplier) or args.quote_size_multiplier <= 0.0:
        parser.error("--quote-size-multiplier must be finite and positive")

    indicator = rtta.FlowPressureCapacitySignal(
        half_life_updates=args.half_life_updates,
        queue_weight=args.queue_weight,
        pressure_weight=args.pressure_weight,
        replenishment_weight=args.replenishment_weight,
        fragility_weight=args.fragility_weight,
        score_scale=args.score_scale,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        warmup=args.warmup,
        fillna=True,
    )
    diagnostics = ForwardDiagnostics(
        int(args.horizon_ms * 1_000_000.0),
        max(1, int(args.sample_ms * 1_000_000.0)),
        abs(args.baseline_threshold),
    )
    book = PaperBook(
        int(args.latency_ms * 1_000_000.0),
        args.allow_short,
        args.max_spread_bps,
        args.fee_bps,
    )

    if not args.quiet:
        print(
            "date,sip_timestamp,bid,ask,bid_size,ask_size,buy_volume,sell_volume,"
            "signal,score,fair_value,microprice,raw_queue,filtered_queue,flow_imbalance,"
            "pressure,replenishment,fragility,spread_bps,desired,position,event"
        )

    sessions = 0
    quote_updates = 0
    eligible_trades = 0
    first_midpoint = None
    last_midpoint = None
    stop_requested = False

    for day in date_range(start_date, stop_date):
        try:
            quotes = massive_speedup.StockQuoteDatabase(
                str(args.database), day.isoformat(), args.symbol
            )
            trades = massive_speedup.StockTradeDatabase(
                str(args.database), day.isoformat(), args.symbol
            )
        except Exception:
            continue
        if len(quotes) == 0:
            continue

        if args.all_hours:
            session_start = 0
            session_stop = (1 << 63) - 1
        else:
            session_start = int(quotes.market_open)
            session_stop = int(quotes.market_close)

        quote_index = first_valid_quote_index(quotes, session_start, session_stop)
        if quote_index is None:
            continue
        current_quote = quotes[quote_index]
        current_timestamp = int(current_quote.sip_timestamp)
        try:
            trade_index = int(trades.index_before_timestamp(current_timestamp)) + 1
        except Exception:
            trade_index = 0
        trade_index = max(trade_index, 0)
        previous_trade_price = None
        previous_trade_side = 0.0

        indicator.reset()
        diagnostics.begin_session()
        sessions += 1
        last_signal = 0.0
        last_bid = float(current_quote.bid_price)
        last_ask = float(current_quote.ask_price)

        initial = indicator.update(
            last_bid,
            float(current_quote.bid_size) * args.quote_size_multiplier,
            last_ask,
            float(current_quote.ask_size) * args.quote_size_multiplier,
            0.0,
            0.0,
        )
        midpoint = 0.5 * (last_bid + last_ask)
        first_midpoint = midpoint if first_midpoint is None else first_midpoint
        last_midpoint = midpoint
        diagnostics.advance(current_timestamp, midpoint)
        diagnostics.sample(current_timestamp, midpoint, initial)
        quote_updates += 1

        for quote_index in range(quote_index + 1, len(quotes)):
            quote = quotes[quote_index]
            timestamp = int(quote.sip_timestamp)
            if timestamp > session_stop:
                break
            if timestamp < session_start or not valid_quote(quote):
                continue

            buy_volume = 0.0
            sell_volume = 0.0
            while trade_index < len(trades):
                trade = trades[trade_index]
                trade_timestamp = int(trade.sip_timestamp)
                if trade_timestamp > timestamp:
                    break
                side, previous_trade_price, previous_trade_side = classify_trade(
                    trade, current_quote, previous_trade_price, previous_trade_side
                )
                if side > 0.0:
                    buy_volume += float(trade.size)
                    eligible_trades += 1
                elif side < 0.0:
                    sell_volume += float(trade.size)
                    eligible_trades += 1
                trade_index += 1

            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
            execution = book.on_quote(timestamp, bid, ask)
            out = indicator.update(
                bid,
                float(quote.bid_size) * args.quote_size_multiplier,
                ask,
                float(quote.ask_size) * args.quote_size_multiplier,
                buy_volume,
                sell_volume,
            )
            midpoint = 0.5 * (bid + ask)
            diagnostics.advance(timestamp, midpoint)
            diagnostics.sample(timestamp, midpoint, out)
            requested = book.request(timestamp, float(out.signal), float(out.spread_bps))

            signal_changed = float(out.signal) != last_signal
            if not args.quiet and (signal_changed or execution):
                event = execution or ("REQUEST" if requested else "SIGNAL")
                print(
                    f"{day.isoformat()},{timestamp},{bid:.6f},{ask:.6f},"
                    f"{float(quote.bid_size) * args.quote_size_multiplier:.0f},"
                    f"{float(quote.ask_size) * args.quote_size_multiplier:.0f},"
                    f"{buy_volume:.0f},{sell_volume:.0f},{float(out.signal):.0f},"
                    f"{float(out.score):.8f},{float(out.fair_value):.6f},"
                    f"{float(out.microprice):.6f},{float(out.raw_queue_imbalance):.8f},"
                    f"{float(out.queue_imbalance):.8f},{float(out.flow_imbalance):.8f},"
                    f"{float(out.pressure):.8f},{float(out.replenishment):.8f},"
                    f"{float(out.fragility):.8f},{float(out.spread_bps):.4f},"
                    f"{book.desired:.0f},{book.position:.0f},{event}"
                )

            last_signal = float(out.signal)
            current_quote = quote
            current_timestamp = timestamp
            last_bid = bid
            last_ask = ask
            last_midpoint = midpoint
            quote_updates += 1
            if args.max_quotes and quote_updates >= args.max_quotes:
                stop_requested = True
                break

        book.close_session(last_bid, last_ask)
        if stop_requested:
            break

    buy_hold_bps = (
        (last_midpoint / first_midpoint - 1.0) * 10_000.0
        if first_midpoint is not None and last_midpoint is not None
        else float("nan")
    )
    print(
        "# summary,"
        f"symbol={args.symbol},sessions={sessions},quotes={quote_updates},"
        f"classified_trades={eligible_trades},resolved_forecasts={diagnostics.resolved},"
        f"score_ic={diagnostics.score_ic.value:.6g},"
        f"raw_queue_ic={diagnostics.queue_ic.value:.6g},"
        f"signal_active={diagnostics.signal_edge.count},"
        f"signal_mean_signed_bps={diagnostics.signal_edge.mean:.6g},"
        f"signal_hit_rate={diagnostics.signal_edge.hit_rate:.6g},"
        f"raw_queue_active={diagnostics.queue_edge.count},"
        f"raw_queue_mean_signed_bps={diagnostics.queue_edge.mean:.6g},"
        f"raw_queue_hit_rate={diagnostics.queue_edge.hit_rate:.6g},"
        f"paper_entries={book.entries},paper_rebalances={book.rebalances},"
        f"paper_pnl_dollars_per_share={book.cash:.6g},"
        f"paper_deployment_return_bps={book.deployment_return_bps:.6g},"
        f"buy_hold_first_to_last_bps={buy_hold_bps:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
