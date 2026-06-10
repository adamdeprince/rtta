# RTTA

`pyrtta` is a low-latency C++23/nanobind library for tick-by-tick technical
analysis, online change detection, market regime monitoring, and research
signals. The Python import package is `rtta`.

The design goal is to make accidental crystal-balling hard. Algorithms are
stateful, causal objects: callers feed one tick at a time through `update(...)`
or `advance(...)`, and the object can only react to data it has already seen.
That surface is meant to fit live systems, interactive research modes, and
market simulators where orders must be decided before the next observation is
available.

## Scope

The current benchmark registry covers 188 algorithms:

- Classic technical indicators: moving averages, oscillators, trend,
  volatility, price transforms, bands, channels, volume indicators, and returns.
- State-space and adaptive filters: Kalman variants, particle filters,
  interacting multiple models, Gaussian-process and kernel envelopes, and
  tracking filters.
- Market microstructure and liquidity widgets: order-flow imbalance, bid-ask
  bounce, spread features, quote/trade intensity, VPIN, Amihud, Kyle lambda,
  liquidity drought, spread explosion, and execution-cost/slippage regimes.
- Online change and drift detection: CUSUM, Page-Hinkley, ADWIN, DDM, EDDM,
  HDDM, KSWIN, EWMA z-score shifts, residual/error/hit-rate/calibration/feature
  drift, and rolling two-window mean, variance, correlation, beta, and
  spread/liquidity shift detectors.
- Online regime filters: threshold and hysteresis regimes, volatility/ATR/
  realized-variance/trend-chop/liquidity/spread/volume/order-flow/correlation/
  beta/pairs-spread regimes, bounded BOCPD, online HMM, sticky HMM-style,
  Markov-switching volatility, Gaussian mixture, and semi-Markov-style filters.
- Finance-specific live widgets: volatility breakout, compression/expansion,
  microstructure-noise, quote-stuffing, lead-lag, open/close and
  auction/continuous-market transitions, cross-asset correlation breaks, and
  streaming residual-based cointegration breakdown monitoring.

## Installation

```bash
pip install pyrtta
```

## Usage

```python
import rtta

rsi = rtta.RSI()

for close in close_stream:
    value = rsi.update(close)
    if value > 70.0:
        reduce_position()
```

Use `advance(...)` when the caller only needs to update state and does not need
a Python result object for that tick:

```python
ema = rtta.EMA(window=30.0)

for close in warmup_ticks:
    ema.advance(close)

current = ema.update(next_close)
```

Most indicators expose:

- `update(...)`: consume one sample and return the current value/result.
- `advance(...)`: consume one sample and return `None`.
- `last()` or `last_<field>()`: read the most recent state without advancing.
- `batch(...)`: causal bulk catch-up for restart/research workflows. It consumes
  input in chronological order and leaves the object ready for the next live
  tick.
- `replay_update(...)`, `replay_advance(...)`, and
  `replay_update_outputs(...)`: C++ replay paths for causal catch-up and
  latency benchmarking.

Multi-output indicators return immutable C++ result structs with read-only
fields. Scalar convenience methods such as `update_upper(...)` and
`last_upper()` are available where an indicator has named fields.

## Benchmarks

Latency results are maintained on the standalone [benchmark page](BENCHMARK.md).
The benchmark page records CPU and runtime metadata for current Intel, Apple
Silicon, and Loongson runs.

## Development

This project is built as a C++23 nanobind extension with CMake through
`scikit-build-core`.

```bash
poetry install --with build,dev --no-root
poetry run python -m pip install --no-build-isolation -e .
poetry run pytest
```

To build a wheel:

```bash
poetry run python -m build --wheel
```

## Citation

If you use RTTA in research, benchmarks, or published work, cite it with:

```bibtex
@misc{deprince2026pyrtta,
  author       = {DePrince, Adam},
  title        = {{pyrtta}: Low Latency Incremental Technical Analysis},
  year         = {2026},
  version      = {0.2.1},
  howpublished = {\url{https://github.com/adamdeprince/rtta}},
  note         = {Python package name: pyrtta; import package name: rtta}
}
```

The same entry is available in `CITATION.bib`.
