#!/usr/bin/env python
"""Run SqrtImpactFlowSignal on massive_speedup aggregates and map it to trades.

Data
----
Uses Polygon/Massive-style bars: close, volume, bar VWAP. Optionally builds
signed dollar volume from the trade tape (tick rule) for a cleaner flow sign.

How to buy / sell from the indicator
------------------------------------
``SqrtImpactFlowSignal.update(...)`` returns a result with:

* ``signal``  — discrete stance in ``{-1, 0, +1}`` (**this is the trade directive**)
* ``score``   — continuous strength in roughly ``(-1, 1)`` (optional size/confidence)
* diagnostics — ``impact``, ``continuation``, ``reversion``, ``flow``, ``vwap_gap``, ...

Trading map (what this example implements):

+-----------+---------------------------+----------------------------------+
| signal    | Meaning                   | Action                           |
+===========+===========================+==================================+
| ``+1``    | Bullish impact residual   | BUY  (enter/hold long)           |
| ``-1``    | Bearish impact residual   | SELL short (or exit if long-only)|
| ``0``     | No edge / hysteresis exit | FLAT (exit to cash)              |
+-----------+---------------------------+----------------------------------+

Rules:

1. Call ``update`` once per finished bar.
2. Read ``out.signal`` (hysteresis is already inside the indicator via
   ``entry_z`` / ``exit_z``).
3. Rebalance when the target position changes:
   - **long-only** (default): ``+1`` → long; ``0`` or ``-1`` → flat.
   - **long/short** (``--allow-short``): target position equals ``signal``.
4. Optional: require ``|score| >= --min-score`` to **open** risk (exits always
   honor ``signal == 0`` / side change).

Economic reading (debugging, not required to fire trades):

* High ``continuation`` + same-sign ``flow`` → volume arrived, price lagged →
  lean *with* flow (unfinished square-root impact).
* High ``reversion`` → price overshot volume-implied impact → lean *against*
  the move (temporary impact).
* ``vwap_gap`` reinforces when close sits on the same side of VWAP as flow.

Example
-------
::

    python examples/sqrt_impact_flow_from_massive_speedup.py \\
        --database /path/to/db --symbol AAPL \\
        --start-date 2024-01-02 --stop-date 2024-01-31 \\
        --interval 300 --use-tape-sign

    # long-only, fewer trades:
    python examples/sqrt_impact_flow_from_massive_speedup.py ... \\
        --entry-z 1.0 --min-score 0.15

    # allow shorts:
    python examples/sqrt_impact_flow_from_massive_speedup.py ... --allow-short
"""

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


def tick_rule_signed_dollar(trades) -> float:
    """Sum signed notional over a trade list using the tick rule."""
    signed = 0.0
    prev = None
    for tr in trades:
        price = float(tr.price)
        size = float(tr.size)
        if prev is None:
            side = 0.0
        elif price > prev:
            side = 1.0
        elif price < prev:
            side = -1.0
        else:
            side = 0.0
        signed += side * price * size
        prev = price
    return signed


def desired_position(
    signal: float,
    score: float,
    current: float,
    *,
    allow_short: bool,
    min_score: float,
) -> float:
    """Map ``signal`` / ``score`` → target position in ``{-1, 0, +1}``.

    * ``signal == +1`` → long (if ``|score|`` ok for new entries, or already long).
    * ``signal == -1`` → short if ``allow_short``, else flat.
    * ``signal ==  0`` → flat.

    ``min_score`` only blocks *new* risk. Holding an existing side while
    ``signal`` still agrees does not require ``|score| >= min_score``.
    """
    # Explicit flat from the indicator (hysteresis exit).
    if signal == 0.0:
        return 0.0

    want_long = signal > 0.0
    want_short = signal < 0.0

    # Still on the same side → hold without re-checking min_score.
    if want_long and current > 0.0:
        return 1.0
    if want_short and current < 0.0 and allow_short:
        return -1.0

    # Exits / flips from an existing position always allowed.
    if current > 0.0 and not want_long:
        if want_short and allow_short:
            if abs(score) < min_score:
                return 0.0  # exit long but block weak short entry
            return -1.0
        return 0.0
    if current < 0.0 and not want_short:
        if want_long:
            if abs(score) < min_score:
                return 0.0
            return 1.0
        return 0.0

    # Flat → open only if score clears the threshold.
    if abs(score) < min_score:
        return 0.0
    if want_long:
        return 1.0
    if want_short and allow_short:
        return -1.0
    return 0.0  # long-only: signal=-1 while flat → stay flat


def action_name(prev_pos: float, new_pos: float) -> str:
    """Human-readable trade label for the CSV."""
    if prev_pos == new_pos:
        return "HOLD"
    if prev_pos == 0.0 and new_pos > 0.0:
        return "BUY"
    if prev_pos == 0.0 and new_pos < 0.0:
        return "SELL_SHORT"
    if prev_pos > 0.0 and new_pos == 0.0:
        return "SELL"  # exit long → flat
    if prev_pos < 0.0 and new_pos == 0.0:
        return "COVER"  # exit short → flat
    if prev_pos > 0.0 and new_pos < 0.0:
        return "SELL_REVERSE"  # long → short
    if prev_pos < 0.0 and new_pos > 0.0:
        return "BUY_REVERSE"  # short → long
    return "REBALANCE"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "SqrtImpactFlowSignal on Massive/Polygon bars with explicit BUY/SELL mapping. "
            "signal=+1 buy/long, signal=-1 sell/short (or exit), signal=0 flat. "
            "See module docstring."
        )
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--stop-date", required=True)
    parser.add_argument("--interval", type=int, default=300, help="Bar seconds")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--use-tape-sign",
        action="store_true",
        help="Build signed dollar volume from trades in each bar (else tick-rule on close)",
    )
    parser.add_argument(
        "--impact-coefficient",
        type=float,
        default=1.0,
        help="Y in I = Y * sigma * sqrt(Q/V)",
    )
    parser.add_argument(
        "--entry-z",
        type=float,
        default=0.75,
        help="Higher → fewer entries (harder for signal to leave 0)",
    )
    parser.add_argument(
        "--exit-z",
        type=float,
        default=0.25,
        help="Lower → hold longer once in (hysteresis exit threshold)",
    )
    parser.add_argument(
        "--continuation-weight",
        type=float,
        default=1.0,
        help="Weight on unfinished impact (with-flow) leg",
    )
    parser.add_argument(
        "--reversion-weight",
        type=float,
        default=0.5,
        help="Weight on overshoot fade leg (0 disables mean-reversion leg)",
    )
    parser.add_argument(
        "--vwap-weight",
        type=float,
        default=0.25,
        help="Weight on close-vs-VWAP alignment",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Require |score| >= this to open new risk (0 = use signal only)",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="If set, signal=-1 opens/holds a short; else -1 only exits longs",
    )
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    stop_date = dt.date.fromisoformat(args.stop_date)
    if stop_date < start_date:
        parser.error("--stop-date must be on or after --start-date")

    # Discrete buy/sell stance is result.signal (see module docstring).
    indicator = rtta.SqrtImpactFlowSignal(
        impact_coefficient=args.impact_coefficient,
        continuation_weight=args.continuation_weight,
        reversion_weight=args.reversion_weight,
        vwap_weight=args.vwap_weight,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        fillna=True,
    )

    # Paper state driven by signal → desired_position → action_name.
    position = 0.0  # -1 short, 0 flat, +1 long
    trade_count = 0
    entry_price = float("nan")
    realized_pnl = 0.0

    print(
        "date,window_start,close,volume,vwap,signed_dollar,"
        "signal,score,impact,residual,continuation,reversion,"
        "participation,flow,volatility,vwap_gap,"
        "position_before,position_after,action,trade_count,entry_price,realized_pnl"
    )

    for current_date in date_range(start_date, stop_date):
        try:
            trades = massive_speedup.StockTradeDatabase(
                args.database, current_date.isoformat(), args.symbol
            )
        except Exception:
            continue

        for bar in massive_speedup.StockTradeAggregator(
            trades, interval_seconds=args.interval, offset_seconds=args.offset
        ):
            if bar.transactions == 0 or bar.volume == 0:
                continue
            close = float(bar.close)
            volume = float(bar.volume)
            vwap = float(bar.volume_weighted_avg)
            signed = float("nan")
            if args.use_tape_sign:
                bar_trades = getattr(bar, "trades", None)
                if bar_trades is not None:
                    signed = tick_rule_signed_dollar(bar_trades)

            # 1) Indicator step — one finished bar.
            out = indicator.update(close, volume, signed, vwap)

            # 2) Map signal (+ score filter) → target position.
            #    signal +1 → BUY/long, -1 → SELL/short or exit, 0 → FLAT.
            position_before = position
            position_after = desired_position(
                out.signal,
                out.score,
                position_before,
                allow_short=args.allow_short,
                min_score=args.min_score,
            )
            action = action_name(position_before, position_after)

            # 3) Toy fill at bar close when the target changes.
            if position_after != position_before:
                if position_before != 0.0 and math.isfinite(entry_price):
                    realized_pnl += position_before * (close - entry_price)
                if position_after != 0.0:
                    entry_price = close
                else:
                    entry_price = float("nan")
                trade_count += 1
                position = position_after

            print(
                f"{current_date.isoformat()},{getattr(bar, 'window_start', '')},"
                f"{close},{volume},{vwap},{signed},"
                f"{out.signal},{out.score},{out.impact},{out.residual},"
                f"{out.continuation},{out.reversion},{out.participation},"
                f"{out.flow},{out.volatility},{out.vwap_gap},"
                f"{position_before},{position},{action},{trade_count},"
                f"{entry_price},{realized_pnl}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
