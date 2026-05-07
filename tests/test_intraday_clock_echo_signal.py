import math

import numpy as np
import pytest


FIELDS = (
    "slot",
    "samples_for_slot",
    "bar_return",
    "residual_return",
    "clock_echo",
    "flow_confirm",
    "volume_sync",
    "prediction",
    "radius",
    "score",
    "signal",
    "target_fraction",
    "max_trade_dollars",
    "realized_error",
    "ready",
)


def _training_days(day_count=10):
    slot_returns = (0.0, 0.0040, 0.0030, -0.0010, 0.0005, 0.0002)
    days = []
    for day in range(day_count):
        close = 100.0 + day * 0.25
        day_records = []
        for slot, ret in enumerate(slot_returns):
            open_ = close
            close = close * math.exp(ret)
            day_records.append(
                {
                    "open": open_,
                    "high": max(open_, close) + 0.02,
                    "low": min(open_, close) - 0.02,
                    "close": close,
                    "volume": 100_000.0 + 1_000.0 * slot,
                    "vwap": 0.5 * (open_ + close),
                    "transactions": 1000.0 + slot,
                    "market_return": 0.0,
                    "slot": slot,
                }
            )
        days.append(day_records)
    return days


def _assert_same_result(actual, expected):
    for field in FIELDS:
        left = getattr(actual, field)
        right = getattr(expected, field)
        if isinstance(left, bool):
            assert left is right
        elif np.isnan(float(left)) or np.isnan(float(right)):
            assert np.isnan(float(left)) and np.isnan(float(right))
        else:
            assert float(left) == pytest.approx(float(right), rel=1e-12, abs=1e-12)


def test_train_from_day_lists_then_live_update_is_ready_and_predicts_clock_echo():
    import rtta

    indicator = rtta.IntradayClockEchoSignal(
        slots_per_session=6,
        horizon_bars=2,
        lookback_days=5,
        min_slot_samples=2,
        calibration_window=16,
        entry_z=0.1,
        fillna=True,
    )
    indicator.train(_training_days(12))

    result = indicator.update(
        101.0,
        101.1,
        100.9,
        101.0,
        120_000.0,
        vwap=101.0,
        transactions=1200.0,
        slot=0,
        reset_session=True,
    )

    assert result.slot == 0.0
    assert result.samples_for_slot >= 12.0
    assert result.ready
    assert result.clock_echo > 0.0
    assert result.prediction > 0.0
    assert result.signal == 1.0
    assert result.target_fraction > 0.0


def test_scalar_accessors_match_full_result_and_last():
    import rtta

    training = _training_days(8)
    full = rtta.IntradayClockEchoSignal(
        slots_per_session=6,
        horizon_bars=2,
        lookback_days=5,
        min_slot_samples=1,
        calibration_window=8,
        fillna=True,
    )
    scalar = rtta.IntradayClockEchoSignal(
        slots_per_session=6,
        horizon_bars=2,
        lookback_days=5,
        min_slot_samples=1,
        calibration_window=8,
        fillna=True,
    )
    full.train(training)
    scalar.train(training)

    args = (101.0, 101.1, 100.9, 101.0, 120_000.0)
    kwargs = dict(vwap=101.0, transactions=1200.0, slot=0, reset_session=True)
    result = full.update(*args, **kwargs)
    assert scalar.update_prediction(*args, **kwargs) == pytest.approx(result.prediction)
    assert scalar.last_prediction() == pytest.approx(result.prediction)
    _assert_same_result(full.last(), result)


def test_fillna_false_untrained_outputs_are_nan():
    import rtta

    result = rtta.IntradayClockEchoSignal(slots_per_session=6, horizon_bars=2, fillna=False).update(
        100.0,
        100.1,
        99.9,
        100.0,
        100_000.0,
        slot=0,
        reset_session=True,
    )
    assert not result.ready
    assert np.isnan(result.prediction)
    assert np.isnan(result.radius)
    assert np.isnan(result.score)
    assert result.signal == 0.0
