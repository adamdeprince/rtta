RTTA
======================

Purpose
-------

The purpose of this package is to implement a very low latency
incremental technical analysis toolkit.  Most technical analysis
tool-kits work in a "batch mode" where you hand them a blob of data and
in a pandas series and they return a series with the computed data.
Incremental updates for these require O(n) work.  There is one tool,
[talipp](https://pypi.org/project/talipp/) that is designed to support
incremental updates, but it is implemented in pure python and is a
little more than an order of magnitude slower than rtta.  On a 5995WX
talipp's exponential moving average requires 465ns; rtta's requires
36ns.  A bare python function call requires 35ns, so we're about as
fast as fast can be.

Installation
------------

```bash
pip install pyrtta
```

Development
-----------

This project is built as a C++23 nanobind extension with CMake through
`scikit-build-core`. Poetry is used for dependency management. The Kalman
indicators include `fast-kalman` headers from the pinned build dependency.

```bash
poetry install --with build,dev --no-root
poetry run python -m pip install --no-build-isolation -e .
poetry run pytest
```

To build a wheel:

```bash
poetry run python -m build --wheel
```

Usage
-----

Each operator has a paramater fillna.  If set to false, nan values
will be returned until the operation is "populated".  If set to true,
best guesses will be returned until the operation is populated.

So for example, the simple moving average works sort of like this:

```python
>>> from rtta.indicator import SMA
>>> sma = SMA(window=4, fillna=True)
>>> sma.update(1)
1
>>> sma.update(2)
1.5
>>> sma.update(3)
2
>>> sma.update(2)
2
>>> sma.update(2)
2.25
```

Indicator API conventions
-------------------------

Kalman indicators expose algorithm-specific tuning helpers. For example,
`KalmanMovingAverage.tune(close)` estimates the price-filter noise and state
variance parameters for that indicator and returns an immutable
`KalmanMovingAverageTuning` object. Pass that object back to
`KalmanMovingAverage(...)` to build an indicator with the recommended
parameters for the data.

New indicators should follow the same surface area as the existing C++
nanobind indicators:

- `update(...)` advances state by one sample and returns the current result.
- `advance(...)` advances state by one sample and returns `None`; it is for
  callers that do not need a Python result object for that sample.
- Single-output indicators return a Python float from `update(...)`.
- Multi-output indicators return an immutable C++ result struct with read-only
  fields, for example `VortexResult.positive`.
- Multi-output indicators also expose scalar field updates named
  `update_<field>(...)`, such as `update_positive(...)`. These advance state and
  return only that field as a Python float.
- Multi-output indicators expose `last_<field>()`, such as `last_positive()`,
  to read one field from the most recent state without advancing.
- If more than one field is needed for the same sample, call `update(...)` once
  and read the immutable result fields, or call one `update_<field>(...)` and
  then read the other fields through `last_<field>()`. Calling multiple
  `update_<field>(...)` methods advances multiple samples.
- Incremental C++ replay methods use NumPy float64 or float32 arrays:
  `replay_update(...)` and `replay_advance(...)` return checksum floats for
  latency benchmarking, while `replay_update_outputs(...)` returns the same
  immutable batch-result shape as `batch(...)` after iterating the incremental
  update kernel in C++.
- `batch(...)` remains the normal array/table/record-list bulk API. It may use
  specialized batch kernels when that is algorithmically better, but it must
  leave object state compatible with subsequent incremental `update(...)` calls.

Performance
-----------

Use the benchmark utility to generate blog-ready Markdown tables with
nanoseconds per input sample. RTTA ndarray batch, RTTA pandas table batch,
RTTA record-list batch, TA-Lib batch, and `ta` batch timings are reported
together where equivalent indicators are available. RTTA `update()` latency is
reported separately.

RTTA batch methods accept contiguous NumPy float64 and float32 arrays. They
also accept pandas tables when the referenced columns can be read as contiguous
float64 or float32 arrays without copying.

The comparison packages are optional and intentionally not listed in
`pyproject.toml`:

```bash
python -m pip install ta==0.11.0 TA-Lib
python benchmarks/benchmark_indicators.py --samples 200000 --output benchmark.md
```

CSV output is also available:

```bash
python benchmarks/benchmark_indicators.py --format csv --output benchmark.csv
```
