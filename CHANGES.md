# Changes

## 0.2.0

- Renamed the Python distribution to `pyrtta` while preserving `rtta` as the
  import package name.
- Added pinned `fast-kalman==0.2.2` Python and build dependencies for Kalman
  filter indicators.
- Removed the vendored `third_party/fast-kalman` submodule path; CMake now
  consumes the C++ headers from the pinned `fast-kalman` build dependency so
  source distributions build cleanly without submodule-specific setup.
- Added `KalmanMovingAverage`, a constant-velocity Kalman price filter with
  update, advance, replay, batch, pandas-table batch, record-list batch, and
  indicator-specific immutable tuning output.
- Added `KalmanLocalLinearTrend`, a Kalman level/trend state-space indicator
  with configurable and trainable tuning parameters.
- Added `KalmanVelocityOscillator`, exposing the velocity state from the same
  constant-velocity Kalman price model with configurable and trainable tuning
  parameters.
- Added `KalmanInnovationZScore`, exposing the signed Kalman measurement
  innovation normalized by its predicted standard deviation.
- Added `KalmanPredictionBands`, exposing one-step predicted price bands from
  Kalman predicted measurement uncertainty.
- Added `KalmanTrendSignal`, exposing a Kalman-filtered trend line and
  buy/sell signal based on price crossing the filtered trend.
- Added `ConnorsRSI`, `RelativeVigorIndex`, `KlingerVolumeOscillator`,
  `ElderRayIndex`, `CoppockCurve`, `FisherTransform`,
  `FractalAdaptiveMovingAverage`, `MesaAdaptiveMovingAverage`, and
  `EhlersOptimalTrackingFilter`.
- Added `OrderFlowImbalance`, a quote-level best bid/ask price and size
  pressure measure with incremental update/advance, replay, array batch,
  record-list batch, pandas-table batch, benchmark, and correctness coverage.
- Added `AlphaBetaGammaTrackingFilter`, a steady-state Kalman-like
  price/velocity/acceleration tracker with immutable multi-output results,
  scalar field accessors, replay-output batches, and array/table/record batch
  support.
- Added `InteractingMultipleModelFilter`, a four-regime IMM Kalman tracker
  that mixes low-volatility, high-volatility, trend, and chop models with
  online model probabilities.
- Added `ParticleFilterTrend`, a deterministic-seed non-Gaussian particle
  trend filter with Laplace measurement likelihood, systematic resampling,
  signal output, and effective sample size diagnostics.
- Added `SavitzkyGolayFilter`, a causal rolling polynomial smoother with
  precomputed endpoint convolution coefficients for smooth price, first
  derivative, and second derivative outputs.
- Added `NadarayaWatsonEnvelope`, a Gaussian-kernel nonparametric smoother
  with weighted residual upper/lower bands.
- Added `GaussianProcessRegressionBands`, a rolling RBF-kernel Gaussian
  process smoother that emits posterior mean and uncertainty bands.
- Added `ZigZagSwingDetector`, a close-based percentage swing detector for
  filtering noise and labeling confirmed pivots.
- Added `RenkoBrickGenerator`, an event-driven close-price transform that
  reports signed brick counts, current brick state, direction, and reversals.
- Added `HeikinAshiTransform`, an incremental OHLC candle transform that emits
  Heikin-Ashi open, high, low, and close values.
- Added `AnchoredVWAP`, a VWAP variant that resets cumulative price-volume
  state from arbitrary anchor events.
- Added `KalmanRegressionChannel`, `KalmanHedgeRatio`,
  `TwoFactorKalmanTrendFilter`, and `KalmanExtremumTrend` for online Kalman
  pair regression and hybrid trend filtering.
- Added `ALGOS.md`, cataloging public indicators with short descriptions and
  documentation links that prefer ChartSchool when a direct ChartSchool page
  exists.
- Added `CITATION.bib` and a README citation section with a BibTeX reference
  for users who cite the package.
- Added `VariableIndexDynamicAverage`, implementing VIDYA with CMO-modulated
  EMA smoothing for update, advance, replay, array batch, record-list batch,
  pandas-table batch, benchmarks, and correctness tests.
- Added a standalone benchmark utility at `benchmarks/benchmark_indicators.py`.
  It reports nanoseconds per sample with batch-vs-batch comparisons separated
  from RTTA `update()` latency.
- Added `benchmarks/benchmark_update_latency.py` for RTTA-first incremental
  latency tracking. It reports Python loop overhead, gross `update()` latency,
  loop-adjusted update latency, and future `advance()` latency columns.
- Added no-return `advance()` bindings for every public indicator with
  `update()`, plus coverage that verifies `advance()` performs the same state
  transition as `update()`.
- Split benchmark reporting into RTTA ndarray batch, RTTA record-list batch,
  RTTA pandas table batch, third-party batch, and RTTA `update()` timing
  columns.
- Added record-list batch overloads for EMA, SMA, Summation, ROC, Kama, and
  PercentagePrice.
- Added C++ batch loops for indicators that previously only exposed
  incremental `update()` calls, so NumPy/table batches iterate on the C++ side
  instead of crossing the Python boundary once per sample.
- Added zero-copy pandas table batch support. Table batches extract the
  expected column names and only accept contiguous float32/float64 columns.
- Added explicit float32 NumPy batch overloads alongside the existing float64
  overloads, with no-convert ndarray arguments so nanobind cannot silently
  cast or copy inputs into a slower hidden path.
- Switched pandas table dtype dispatch to the integer NumPy `dtype.num`
  interface and `switch` dispatch instead of dtype-name string comparisons.
- Specialized hot rolling batch cores for weighted moving average,
  beta/correlation, variance, linear regression outputs, and several
  extrema-derived indicators so batch runs avoid avoidable rolling-window
  rescans and unnecessary min/max/sum maintenance.
- Added formula-specific batch loops for UltimateOscillator, TrueRange,
  Momentum, ChandeMomentumOscillator, MoneyFlowIndex, HighLow, and
  HighLowIndex so these no longer route NumPy/table batches through generic
  incremental update wrappers.
- Added an explicit `batch_kernels` section in `indicator.cpp` and migrated
  the largest remaining TA-Lib benchmark gaps to raw-loop batch kernels while
  preserving the existing object-oriented `update()` and `batch()` APIs.
- Replaced deque-backed rolling min/max queues with fixed-capacity vector ring
  buffers and explicitly inlined the hot rolling helper accessors.
- Added a dedicated small-window raw-scan batch kernel for `MidPrice`, with
  state rebuild logic so `batch()` remains compatible with subsequent
  incremental `update()` calls.
- Added dedicated fresh-batch fast paths for small-window extrema-derived
  value indicators (`High`, `Low`, `HighLow`, `MidPoint`, `WilliamsR`, and
  stochastic fast-k), while keeping index/Aroon outputs on the faster
  monotonic-queue path; also flattened EMA chains for `T3MovingAverage`,
  direct momentum loops, prefix-style statistical kernels, and flattened
  Wilder smoothing loops for ATR and directional-movement indicators.
- Specialized EMA, SMA, Summation, ROC, and Kama batch loops to avoid
  per-sample nanobind indexing and unnecessary `update()` dispatch inside
  batch runs.
- Added `PercentagePrice.batch_ppo()` so the PPO-only benchmark compares the
  same output shape as TA-Lib's PPO batch function.
- Kept comparison libraries out of `pyproject.toml`; the benchmark documents
  optional `pip install ta==0.11.0 TA-Lib` usage instead.
- Updated the README performance section to point at the benchmark utility and
  its Markdown/CSV output modes.
- Changed `make_array` to transfer a moved `std::vector<double>` directly to
  NumPy ownership instead of allocating a second buffer and copying.
- Reworked `RollingWindow` to maintain an incremental sum and monotonic
  min/max queues, avoiding rescans for rolling `sum`, `min`, `max`, and
  min/max offset lookups.
- Replaced Python `dict` result objects with typed C++ result structs exposed
  through read-only nanobind fields for multi-output indicators.
- Added scalar `update_<field>()` and `last_<field>()` accessors for
  multi-output indicators so callers can avoid allocating immutable result
  objects when they only need one output.
- Added `replay_update_outputs()` for multi-output indicators, returning the
  same batch-result shape as `batch()` while iterating the incremental update
  kernel in C++.
- Documented the indicator API conventions for `update()`, `advance()`,
  immutable multi-output result structs, scalar field accessors, checksum
  replay methods, and replay-output batches in `README.md`.
- Added `SuperTrend`, with incremental `update()`/`advance()`, scalar field
  accessors, C++ replay-output batches, and array/table/record batch support.
- Added `ChoppinessIndex`, implementing CHOP from rolling true-range sums and
  rolling high/low range, with incremental `update()`/`advance()` and
  array/table/record batch support.
- Added `HullMovingAverage`, implementing HMA as
  `WMA(2*WMA(n/2) - WMA(n), sqrt(n))` with incremental `update()`/`advance()`
  and array/table/record batch support.
- Added `VolumeWeightedMovingAverage`, implementing VWMA as rolling
  `sum(close * volume) / sum(volume)` with incremental `update()`/`advance()`
  and array/table/record batch support.
- Added `FibonacciRetracementLevels`, exposing rolling 0%, 23.6%, 38.2%,
  50%, 61.8%, and 100% retracement levels for uptrend or downtrend anchors,
  with immutable multi-output results, scalar field accessors, replay-output
  batches, and array/table/record batch support.
- Corrected ATR/ATRP/NormalizedATR to use incremental Wilder smoothing that
  matches TA-Lib after warmup.
- Updated affected tests for the typed result API.
- Added 512-sample realistic-sequence tests for every benchmarked indicator
  and third-party correctness comparisons against TA-Lib and `ta` for matching
  technical-analysis indicators.
- Bumped the package version to `0.2.0`.
