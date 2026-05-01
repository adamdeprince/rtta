# Changes

## 0.2.0

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
- Corrected ATR/ATRP/NormalizedATR to use incremental Wilder smoothing that
  matches TA-Lib after warmup.
- Updated affected tests for the typed result API.
- Added 512-sample realistic-sequence tests for every benchmarked indicator
  and third-party correctness comparisons against TA-Lib and `ta` for matching
  technical-analysis indicators.
- Bumped the package version to `0.2.0`.
