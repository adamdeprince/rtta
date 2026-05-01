import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rtta
from benchmarks.benchmark_indicators import INDICATORS, generate_market_data


def _result_values(value):
    if isinstance(value, (float, int)):
        return (float(value),)

    fields = [
        name
        for name in dir(value)
        if not name.startswith("_") and not callable(getattr(value, name))
    ]
    return tuple(float(getattr(value, name)) for name in fields)


def _assert_same_result(left, right):
    left_values = _result_values(left)
    right_values = _result_values(right)
    assert len(left_values) == len(right_values)
    for left_value, right_value in zip(left_values, right_values):
        if math.isnan(left_value) or math.isnan(right_value):
            assert math.isnan(left_value) and math.isnan(right_value)
        else:
            assert left_value == pytest.approx(right_value, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("spec", INDICATORS, ids=lambda spec: spec.name)
def test_advance_is_no_return_update_state_transition(spec):
    data = generate_market_data(128, 54321)
    indicator_cls = getattr(rtta, spec.name)
    update_indicator = indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)
    advance_indicator = indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)
    replay_update_indicator = indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)
    replay_advance_indicator = indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)
    inputs = [data.lists[name] for name in spec.update_inputs]
    arrays = [data.arrays[name] for name in spec.update_inputs]

    assert hasattr(advance_indicator, "advance")
    assert hasattr(replay_update_indicator, "replay_update")
    assert hasattr(replay_advance_indicator, "replay_advance")

    for index in range(64):
        args = [values[index] for values in inputs]
        update_indicator.update(*args)
        assert advance_indicator.advance(*args) is None

    replay_inputs = [values[:64] for values in arrays]
    assert isinstance(replay_update_indicator.replay_update(*replay_inputs), float)
    assert isinstance(replay_advance_indicator.replay_advance(*replay_inputs), float)

    final_args = [values[64] for values in inputs]
    expected = update_indicator.update(*final_args)
    _assert_same_result(expected, advance_indicator.update(*final_args))
    _assert_same_result(expected, replay_update_indicator.update(*final_args))
    _assert_same_result(expected, replay_advance_indicator.update(*final_args))
