#!/usr/bin/env python
"""Diagnose intraday mean reversion with the Fourier-Residue Identity.

Run ``rtta.FourierResidueIdentity`` over bars aggregated from the Massive/Polygon
trade tape and answer the question a scalar autocorrelation cannot: when the
series mean-reverts, is the *direction* predictable, or only the *size*?

Research
--------
V. Portnaya, "The Bounce Has No Direction: Sign, Magnitude, and the
Microstructure of Equity Return Predictability", arXiv:2606.29591 (June 2026).

The paper's headline: SPY's lag-1 autocorrelation is -0.081, which is 7.4
standard errors below zero, yet the FRI *sign* test on the same data returns
p = 0.11. The reversal is entirely in the magnitude channel - the fingerprint of
bid-ask bounce and non-synchronous constituent pricing, not directional
reversal. A contrarian strategy built on that -0.081 has no statistical warrant.

What this example demonstrates
------------------------------
It paper-trades two strategies side by side on the same bars:

* ``naive``  - the textbook mistake. Trades contrarian whenever the scalar
  autocorrelation is significantly negative (``z_rho <= -entry_z``), i.e. fades
  the last bar's move. This is what most "mean reversion" indicators license.
* ``fri``    - trades only when the **sign channel** itself clears significance
  (``|z_sign| >= entry_z``). This is ``result.signal``, straight from the
  indicator.

The two agree whenever both channels are significant, and diverge exactly when
the scalar ACF is significant but the direction channel is not - the SPY case,
where ``naive`` keeps trading a direction that is a coin flip and ``fri`` stands
down. The comparison is the point: the FRI tells you *which regime you are in*
before you commit risk.

Honest limits: ``signal`` is not a bid-ask-bounce filter. A simulated Roll
bounce produces a thoroughly significant sign channel, and no statistic computed
from close prices alone can tell you the reversal is uncapturable - what makes it
uncapturable is the spread you cross, which is not in this data. What the FRI
does give you is the separation: large ``|z_rho|`` with small ``|z_sign|`` means
size the position, do not bet the direction.

Reading the diagnostic columns
------------------------------
+----------------------+------------------------------------------------------+
| ``z_rho``            | evidence that *something* is predictable             |
| ``z_sign``           | evidence that the *direction* is predictable         |
| ``elliptical_ratio`` | observed sign channel / Gaussian benchmark; ~1 means |
|                      | as directional as an elliptical process with that    |
|                      | rho, ~0 means magnitude carries all of it            |
| ``variance_ratio``   | VR(q) < 1 reversion, > 1 momentum                    |
| ``..._sign``         | VR_2(q), the direction channel alone                 |
| ``..._magnitude``    | VR_4(q), the magnitude channel alone                 |
| ``persistence``      | half-period ratio: ~1.41 sampling noise, ~1 real     |
| ``magnitude_forecast``| conditional E|next return| - use it for sizing      |
+----------------------+------------------------------------------------------+

Overnight gaps
--------------
Intraday microstructure statistics must not be contaminated by the overnight
return. Rather than resetting the estimator each session (which would throw away
the memory the statistics are built from), this script chains *intraday* log
returns into a continuous synthetic price index and feeds that to the indicator.

Examples
--------
::

    python examples/fourier_residue_identity_from_massive_speedup.py \\
        --database /path/to/db --symbol SPY \\
        --start-date 2026-01-02 --stop-date 2026-03-31 --interval 60

    # Test the lag-3 partial-price-adjustment channel the scalar ACF misses:
    python examples/fourier_residue_identity_from_massive_speedup.py ... \\
        --test-lag 3 --horizon 5

    # Summary only, at several sampling frequencies:
    python examples/fourier_residue_identity_from_massive_speedup.py ... \\
        --quiet --interval 300
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


def session_bars(database: Path, symbol: str, day: dt.date, interval: int,
                 offset: int, regular_hours: bool):
    """Yield aggregated trade bars for one session, optionally RTH-only."""
    try:
        trades = massive_speedup.StockTradeDatabase(
            str(database), day.isoformat(), symbol
        )
    except Exception:
        return

    open_ns = getattr(trades, "market_open", None)
    close_ns = getattr(trades, "market_close", None)

    for bar in massive_speedup.StockTradeAggregator(
        trades, interval_seconds=interval, offset_seconds=offset
    ):
        if bar.volume == 0 or bar.transactions == 0:
            continue
        if regular_hours and open_ns is not None and close_ns is not None:
            if not (open_ns <= bar.window_start < close_ns):
                continue
        yield bar


class PaperBook:
    """Minimal long/short paper book marked at bar closes."""

    def __init__(self, name: str, allow_short: bool):
        self.name = name
        self.allow_short = allow_short
        self.position = 0.0
        self.entry = float("nan")
        self.trades = 0
        self.pnl = 0.0

    def target(self, stance: float) -> float:
        if stance > 0.0:
            return 1.0
        if stance < 0.0:
            return -1.0 if self.allow_short else 0.0
        return 0.0

    def rebalance(self, stance: float, price: float) -> None:
        want = self.target(stance)
        if want == self.position:
            return
        if self.position != 0.0 and math.isfinite(self.entry):
            self.pnl += self.position * (price / self.entry - 1.0)
        self.entry = price if want != 0.0 else float("nan")
        self.position = want
        self.trades += 1

    def close_out(self, price: float) -> None:
        self.rebalance(0.0, price)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fourier-Residue Identity on Massive/Polygon trade bars. Separates "
            "directional reversal from magnitude shrinkage and only trades "
            "direction when the sign channel is statistically warranted."
        )
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--stop-date", required=True)
    parser.add_argument("--interval", type=int, default=60, help="Bar seconds")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--all-hours",
        action="store_true",
        help="Include pre/post market bars (default: regular session only)",
    )
    parser.add_argument("--max-lag", type=int, default=8)
    parser.add_argument(
        "--horizon", type=int, default=2, help="Variance-ratio horizon q"
    )
    parser.add_argument(
        "--test-lag", type=int, default=1, help="Lag m for the scalar channels"
    )
    parser.add_argument(
        "--span", type=float, default=2048.0, help="EWMA memory in bars"
    )
    parser.add_argument("--median-window", type=int, default=512)
    parser.add_argument(
        "--entry-z",
        type=float,
        default=2.0,
        help="Evidence needed to open directional risk in either strategy",
    )
    parser.add_argument("--exit-z", type=float, default=1.0)
    parser.add_argument(
        "--allow-short", action="store_true", help="Permit short positions"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Print only the closing summary"
    )
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    stop_date = dt.date.fromisoformat(args.stop_date)
    if stop_date < start_date:
        parser.error("--stop-date must be on or after --start-date")

    indicator = rtta.FourierResidueIdentity(
        max_lag=args.max_lag,
        horizon=args.horizon,
        test_lag=args.test_lag,
        span=args.span,
        median_window=args.median_window,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        fillna=False,
    )

    naive = PaperBook("naive", args.allow_short)
    fri = PaperBook("fri", args.allow_short)

    # Synthetic index chaining intraday returns only, so the overnight gap never
    # enters the autocorrelation estimates.
    chained = 100.0
    prev_close = None
    prev_bar_sign = 0.0
    last_price = float("nan")
    bar_count = 0

    if not args.quiet:
        print(
            "date,window_start,close,rho,z_rho,rho_sign,z_sign,elliptical_ratio,"
            "variance_ratio,variance_ratio_sign,variance_ratio_magnitude,"
            "z_variance_ratio,persistence,magnitude_forecast,"
            "fri_signal,fri_position,naive_position"
        )

    for day in date_range(start_date, stop_date):
        prev_close = None  # break the chain at each session boundary
        for bar in session_bars(
            args.database, args.symbol, day, args.interval,
            args.offset, not args.all_hours,
        ):
            close = float(bar.close)
            if close <= 0.0:
                continue
            bar_sign = 0.0
            if prev_close is not None:
                chained *= close / prev_close
                bar_sign = math.copysign(1.0, close - prev_close) if close != prev_close else 0.0
            prev_close = close
            last_price = close

            out = indicator.update(chained)
            bar_count += 1
            if not math.isfinite(out.rho):
                prev_bar_sign = bar_sign
                continue

            # The textbook mistake: fade the last bar's move whenever the scalar
            # ACF says the series reverts, with no check that *direction* is what
            # is actually predictable.
            naive_stance = 0.0
            if math.isfinite(out.z_rho) and out.z_rho <= -args.entry_z:
                naive_stance = -prev_bar_sign

            naive.rebalance(naive_stance, close)
            fri.rebalance(out.signal, close)

            if not args.quiet:
                print(
                    f"{day.isoformat()},{bar.window_start},{close},"
                    f"{out.rho:.6f},{out.z_rho:.3f},{out.rho_sign:.6f},"
                    f"{out.z_sign:.3f},{out.elliptical_ratio:.4f},"
                    f"{out.variance_ratio:.6f},{out.variance_ratio_sign:.6f},"
                    f"{out.variance_ratio_magnitude:.6f},"
                    f"{out.z_variance_ratio:.3f},{out.persistence:.4f},"
                    f"{out.magnitude_forecast:.8f},"
                    f"{out.signal:.0f},{fri.position:.0f},{naive.position:.0f}"
                )
            prev_bar_sign = bar_sign

    if math.isfinite(last_price):
        naive.close_out(last_price)
        fri.close_out(last_price)

    final = indicator.last()
    print()
    print(f"# {args.symbol}  {args.start_date}..{args.stop_date}  "
          f"{args.interval}s bars  n={bar_count}")
    if not math.isfinite(final.rho):
        print("# not enough bars to form the FRI channels")
        return 0

    print(f"#   rho({args.test_lag})        = {final.rho:+.5f}   "
          f"z_rho  = {final.z_rho:+.2f}")
    print(f"#   rho_sign({args.test_lag})   = {final.rho_sign:+.5f}   "
          f"z_sign = {final.z_sign:+.2f}")
    print(f"#   rho_magnitude   = {final.rho_magnitude:+.5f}")
    print(f"#   elliptical_ratio= {final.elliptical_ratio:+.3f}  "
          f"(1 = as directional as Gaussian, 0 = magnitude only)")
    print(f"#   VR({args.horizon})           = {final.variance_ratio:.5f}   "
          f"z* = {final.z_variance_ratio:+.2f}")
    print(f"#   VR_2 (direction)= {final.variance_ratio_sign:.5f}")
    print(f"#   VR_4 (magnitude)= {final.variance_ratio_magnitude:.5f}")
    print(f"#   persistence R_N = {final.persistence:.3f}   "
          f"(1.41 = sampling noise, 1.0 = structural)")

    verdict = (
        "DIRECTIONAL - the sign channel is significant; a contrarian/momentum "
        "bet is warranted"
        if abs(final.z_sign) >= args.entry_z
        else "MAGNITUDE ONLY - direction is a coin flip; size with "
             "magnitude_forecast, do not bet on direction"
    )
    print(f"#   verdict: {verdict}")
    print()
    print(f"# {'strategy':<10} {'trades':>8} {'return':>12}")
    for book in (naive, fri):
        print(f"# {book.name:<10} {book.trades:>8} {book.pnl:>11.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
