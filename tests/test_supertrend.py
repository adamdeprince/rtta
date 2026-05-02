import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rtta
from benchmarks.benchmark_indicators import generate_market_data


def _reference_supertrend(close, high, low, window=10, multiplier=3.0, fillna=True):
    value = []
    direction = []
    upper = []
    lower = []
    previous_close = 0.0
    tr_sum = 0.0
    atr = 0.0
    final_upper = math.nan
    final_lower = math.nan
    previous_value = math.nan
    initialized = False

    for index, (close_value, high_value, low_value) in enumerate(zip(close, high, low), start=1):
        tr = high_value - low_value if index == 1 else max(
            high_value - low_value,
            abs(high_value - previous_close),
            abs(low_value - previous_close),
        )
        previous_close_for_band = previous_close
        previous_close = close_value

        if index <= window:
            tr_sum += tr
            atr = tr_sum / index
            atr_value = math.nan if not fillna and index < window else atr
        else:
            atr = (atr * (window - 1.0) + tr) / window
            atr_value = atr

        midpoint = (high_value + low_value) * 0.5
        if math.isnan(atr_value):
            value.append(math.nan)
            direction.append(math.nan)
            upper.append(math.nan)
            lower.append(math.nan)
            continue

        basic_upper = midpoint + multiplier * atr_value
        basic_lower = midpoint - multiplier * atr_value

        if not initialized:
            final_upper = basic_upper
            final_lower = basic_lower
            uptrend = close_value >= midpoint
            initialized = True
        else:
            old_upper = final_upper
            old_lower = final_lower
            final_upper = basic_upper if basic_upper < old_upper or previous_close_for_band > old_upper else old_upper
            final_lower = basic_lower if basic_lower > old_lower or previous_close_for_band < old_lower else old_lower
            uptrend = close_value > final_upper if previous_value == old_upper else close_value >= final_lower

        current = final_lower if uptrend else final_upper
        previous_value = current
        value.append(current)
        direction.append(1.0 if uptrend else -1.0)
        upper.append(final_upper)
        lower.append(final_lower)

    return {
        "value": np.asarray(value, dtype=np.float64),
        "direction": np.asarray(direction, dtype=np.float64),
        "upper": np.asarray(upper, dtype=np.float64),
        "lower": np.asarray(lower, dtype=np.float64),
    }


def _incremental(close, high, low, **kwargs):
    indicator = rtta.SuperTrend(**kwargs)
    rows = {"value": [], "direction": [], "upper": [], "lower": []}
    for values in zip(close, high, low):
        out = indicator.update(*values)
        rows["value"].append(out.value)
        rows["direction"].append(out.direction)
        rows["upper"].append(out.upper)
        rows["lower"].append(out.lower)
    return {name: np.asarray(values, dtype=np.float64) for name, values in rows.items()}


@pytest.mark.parametrize("fillna", [True, False])
def test_supertrend_matches_reference_512_sample_sequence(fillna):
    data = generate_market_data(512, 20260502)
    kwargs = {"window": 10, "multiplier": 3.0, "fillna": fillna}
    expected = _reference_supertrend(data.arrays["close"], data.arrays["high"], data.arrays["low"], **kwargs)
    actual = _incremental(data.lists["close"], data.lists["high"], data.lists["low"], **kwargs)

    for field in expected:
        np.testing.assert_allclose(actual[field], expected[field], rtol=1e-12, atol=1e-12, equal_nan=True)


def test_supertrend_batch_replay_and_scalar_accessors_match_update():
    data = generate_market_data(128, 20260503)
    arrays = [data.arrays[name] for name in ("close", "high", "low")]
    lists = [data.lists[name] for name in ("close", "high", "low")]
    batch = rtta.SuperTrend().batch(*arrays)
    replay = rtta.SuperTrend().replay_update_outputs(*arrays)

    for field in ("value", "direction", "upper", "lower"):
        np.testing.assert_allclose(getattr(replay, field), getattr(batch, field), rtol=1e-12, atol=1e-12, equal_nan=True)

    result_indicator = rtta.SuperTrend()
    scalar_indicators = {field: rtta.SuperTrend() for field in ("value", "direction", "upper", "lower")}
    for index in range(96):
        args = [values[index] for values in lists]
        result = result_indicator.update(*args)
        for field, indicator in scalar_indicators.items():
            scalar = getattr(indicator, f"update_{field}")(*args)
            assert scalar == pytest.approx(getattr(result, field), rel=1e-12, abs=1e-12)
            assert getattr(indicator, f"last_{field}")() == pytest.approx(scalar, rel=1e-12, abs=1e-12)
