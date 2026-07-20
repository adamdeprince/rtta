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
    "baseline",
    "continuation",
    "dark_cloud_cover",
    "doji",
    "engulfing",
    "evening_star",
    "excitation",
    "flow",
    "hammer",
    "hanging_man",
    "harami",
    "intensity",
    "inverted_hammer",
    "marubozu",
    "morning_star",
    "piercing",
    "reversion",
    "shooting_star",
    "spinning_top",
    "three_black_crows",
    "three_white_soldiers",
    "range",
    "hot",
    "ratio",
    "median",
    "l60",
    "l50",
    "l45",
    "l40",
    "l35",
    "l30",
    "s15",
    "s12",
    "s10",
    "s8",
    "s5",
    "peer_mean",
    "span_b_displaced",
    "span_a_displaced",
    "probability",
    "peer_ofi",
    "impact",
    "beta",
    "total",
    "statistic",
    "bars",
    "complete",
    "bar_volume",
    "bar_low",
    "bar_high",
    "bar_close",
    "bar_open",
    "extremum",
    "overshoot",
    "event",
    "trade",
    "cancel",
    "add",
    "acceleration",
    "alpha_flow",
    "angle",
    "base",
    "bar_number",
    "bear_power",
    "box_price",
    "boxes",
    "brick_close",
    "brick_open",
    "bricks",
    "bull_power",
    "cg",
    "chop_probability",
    "close",
    "conversion",
    "cycle",
    "decycle",
    "difference",
    "direction",
    "down",
    "ease_of_movement",
    "entry_window",
    "effective_spread",
    "effective_sample_size",
    "fama",
    "fastd",
    "fastk",
    "first_derivative",
    "flow_score",
    "frozen",
    "frozen_rod_return",
    "highpass",
    "histogram",
    "hedge_ratio",
    "high",
    "high_vol_probability",
    "inphase",
    "intercept",
    "jaw",
    "kst",
    "kvo",
    "lag",
    "lagging_span",
    "lead_sine",
    "level",
    "level0",
    "level100",
    "level236",
    "level382",
    "level500",
    "level618",
    "line",
    "lips",
    "long_average",
    "long_exit",
    "low_vol_probability",
    "low",
    "lower",
    "long_trend",
    "loser_z",
    "macd",
    "mama",
    "max",
    "max_index",
    "max_trade_dollars",
    "middle",
    "min",
    "min_index",
    "momentum",
    "negative",
    "on",
    "oscillator",
    "open",
    "participation",
    "percent",
    "positive",
    "pp",
    "pressure_score",
    "prediction",
    "price",
    "point_of_control",
    "pivot",
    "pivot_index",
    "ppo",
    "pvo",
    "quadrature",
    "quoted_spread",
    "r1",
    "r2",
    "r3",
    "radius",
    "realized_spread",
    "rel_dollar_volume",
    "residual",
    "reversal",
    "rod_return",
    "roof",
    "rvi",
    "s1",
    "s2",
    "s3",
    "score",
    "short_average",
    "short_exit",
    "signal",
    "sine",
    "slope",
    "slowd",
    "slowk",
    "sma",
    "smi",
    "second_derivative",
    "smooth",
    "span_a",
    "span_b",
    "spread",
    "short_trend",
    "target_fraction",
    "teeth",
    "transaction_shock",
    "trendline",
    "trigger",
    "tsf",
    "trend",
    "trend_probability",
    "up",
    "upper",
    "value",
    "value_area_high",
    "value_area_low",
    "velocity",
    "volatility",
    "volume_shock",
    "vwap_gap",
    "width",
    "winner_z",
    "wt1",
    "wt2",
    "news_guard",
    "exit_window",
    "range_z",
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
    lists = {}
    for name, values in arrays.items():
        if values.ndim == 2:
            lists[name] = [np.ascontiguousarray(values[i]) for i in range(values.shape[0])]
        else:
            lists[name] = values.tolist()
    series = {
        name: pandas.Series(values, copy=False)
        for name, values in arrays.items()
        if values.ndim == 1
    }
    table = pandas.DataFrame(
        {name: values for name, values in arrays.items() if values.ndim == 1},
        copy=False,
    )
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
    if getattr(spec, "depth_book", False):
        # Depth-book indicators use 2D arrays; pandas table batch is intentionally unsupported.
        assert array_output is not None
        return
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
