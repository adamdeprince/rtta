# RTTA Research Signal Notes

This directory documents the research-prototype trading signals that were added
to RTTA as higher-conviction, higher-complexity "moonshot" indicators. These are
not generic chart indicators. They combine market microstructure papers, online
state tracking, and conservative uncertainty gates into incremental RTTA APIs.

Each note covers:

- the paper or research idea behind the signal,
- the implemented algorithm,
- the emitted fields,
- expected data shape and session handling,
- where the implementation intentionally departs from the paper.

## Signals

- [MatchedFlowConformalSignal](matched_flow_conformal_signal.md)
- [ClosePressureReversalSignal](close_pressure_reversal_signal.md)
- [IntradayClockEchoSignal](intraday_clock_echo_signal.md)

## Algorithm Pages

Detailed per-algorithm pages live under [algorithms](algorithms/README.md).
The first pilot set documents ATR, EMA, MACD, RSI, and SMA directly from the
C++ `update(...)` recurrences.

## Benchmarks

CPU-specific latency pages are linked from the [benchmark overview](../BENCHMARK.md).

## General Caveat

These indicators are research prototypes. They produce signals, scores, target
fractions, and liquidity diagnostics; they do not perform execution, portfolio
construction, risk limits, borrow checks, or transaction-cost modeling beyond the
small internal `cost_buffer` threshold. Treat the outputs as features or strategy
components, not as standalone trading systems.
