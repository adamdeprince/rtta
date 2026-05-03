from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_indicators import (
    INDICATORS,
    generate_market_data,
    make_rtta_array_batch_runner,
    make_rtta_table_batch_runner,
)


RESULT_FIELDS = (
    "acceleration",
    "angle",
    "base",
    "bear_power",
    "bull_power",
    "chop_probability",
    "conversion",
    "difference",
    "direction",
    "down",
    "ease_of_movement",
    "effective_sample_size",
    "fama",
    "fastd",
    "fastk",
    "first_derivative",
    "histogram",
    "hedge_ratio",
    "high_vol_probability",
    "intercept",
    "kst",
    "kvo",
    "level",
    "level0",
    "level100",
    "level236",
    "level382",
    "level500",
    "level618",
    "low_vol_probability",
    "lower",
    "long_trend",
    "mama",
    "max",
    "max_index",
    "middle",
    "min",
    "min_index",
    "negative",
    "oscillator",
    "percent",
    "positive",
    "price",
    "pivot",
    "pivot_index",
    "ppo",
    "pvo",
    "residual",
    "rvi",
    "signal",
    "slope",
    "slowd",
    "slowk",
    "sma",
    "second_derivative",
    "smooth",
    "span_a",
    "span_b",
    "spread",
    "short_trend",
    "tsf",
    "trend",
    "trend_probability",
    "up",
    "upper",
    "value",
    "velocity",
    "width",
)


_DATA_CACHE = {}


def _market_data(dtype):
    dtype = np.dtype(dtype)
    cached = _DATA_CACHE.get(dtype)
    if cached is not None:
        return cached

    pandas = pytest.importorskip("pandas")
    data = generate_market_data(512, 42)
    arrays = {
        name: np.ascontiguousarray(values.astype(dtype, copy=True))
        for name, values in data.arrays.items()
    }
    lists = {name: values.tolist() for name, values in arrays.items()}
    series = {name: pandas.Series(values, copy=False) for name, values in arrays.items()}
    table = pandas.DataFrame(arrays, copy=False)
    data = replace(data, arrays=arrays, lists=lists, series=series, table=table)
    _DATA_CACHE[dtype] = data
    return data


def _normalise_output(output):
    if isinstance(output, np.ndarray):
        return np.asarray(output, dtype=np.float64)
    if isinstance(output, tuple):
        return tuple(_normalise_output(value) for value in output)

    fields = {
        field: _normalise_output(getattr(output, field))
        for field in RESULT_FIELDS
        if hasattr(output, field)
    }
    if fields:
        return fields

    return np.asarray(output, dtype=np.float64)


def _assert_output_close(actual, expected):
    actual = _normalise_output(actual)
    expected = _normalise_output(expected)

    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_output_close(actual[key], expected[key])
        return

    if isinstance(actual, tuple):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_output_close(actual_value, expected_value)
        return

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("dtype", (np.float64, np.float32), ids=("float64", "float32"))
@pytest.mark.parametrize("spec", INDICATORS, ids=lambda spec: spec.name)
def test_array_and_pandas_table_batches_match_for_float_dtypes(spec, dtype):
    import rtta

    data = _market_data(dtype)
    array_output = make_rtta_array_batch_runner(rtta, spec, data)()
    table_runner = make_rtta_table_batch_runner(rtta, spec, data)

    assert table_runner is not None
    _assert_output_close(table_runner(), array_output)


@pytest.mark.parametrize("dtype", (np.float64, np.float32), ids=("float64", "float32"))
def test_numpy_batch_inputs_do_not_accept_implicit_dtype_or_order_conversion(dtype):
    import rtta

    contiguous = np.ascontiguousarray(np.arange(64, dtype=dtype))
    assert rtta.SMA(3, fillna=True).batch(contiguous).shape == contiguous.shape

    non_contiguous = np.arange(128, dtype=dtype)[::2]
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    with pytest.raises(TypeError):
        rtta.SMA(3, fillna=True).batch(non_contiguous)

    integers = np.arange(64, dtype=np.int64)
    with pytest.raises(TypeError):
        rtta.SMA(3, fillna=True).batch(integers)


def test_pandas_table_batches_reject_inputs_that_would_require_a_copy():
    import rtta

    pandas = pytest.importorskip("pandas")

    class Table:
        columns = ("input",)

        def __init__(self, values):
            self.values = values

        def __getitem__(self, key):
            if key != "input":
                raise KeyError(key)
            return pandas.Series(self.values, copy=False)

    with pytest.raises(TypeError, match="float32 or float64"):
        rtta.SMA(3, fillna=True).batch(Table(np.arange(64, dtype=np.int64)))

    non_contiguous = np.arange(128, dtype=np.float64)[::2]
    with pytest.raises(TypeError, match="contiguous"):
        rtta.SMA(3, fillna=True).batch(Table(non_contiguous))
