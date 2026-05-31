import math

import numpy as np
import pytest


def _cusum(close, threshold=1.0, drift=0.0):
    positive = 0.0
    negative = 0.0
    previous = None
    output = []

    for value in close:
        value = float(value)
        if previous is None:
            previous = value
            output.append(0.0)
            continue

        change = value - previous
        previous = value
        positive = max(0.0, positive + change - drift)
        negative = min(0.0, negative + change + drift)

        if positive > threshold:
            positive = 0.0
            output.append(1.0)
        elif negative < -threshold:
            negative = 0.0
            output.append(-1.0)
        else:
            output.append(0.0)

    return np.asarray(output, dtype=np.float64)


def test_update_matches_causal_cusum_filter_and_last():
    import rtta

    close = np.asarray([
        100.0,
        100.4,
        100.9,
        101.25,
        100.7,
        100.2,
        99.75,
        100.3,
        101.15,
        100.95,
        99.65,
    ])
    expected = _cusum(close, threshold=1.0, drift=0.1)

    indicator = rtta.CUSUM(threshold=1.0, drift=0.1)
    actual = []
    for value, expected_value in zip(close, expected):
        actual_value = indicator.update(float(value))
        actual.append(actual_value)
        assert indicator.last() == expected_value

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert np.any(expected == 1.0)
    assert np.any(expected == -1.0)


def test_batch_records_pandas_and_float32_match_reference():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(20260529)
    close = np.ascontiguousarray(
        100.0 + np.cumsum(rng.normal(0.0, 0.35, 512)),
        dtype=np.float64,
    )
    close32 = np.ascontiguousarray(close.astype(np.float32))
    records = [{"close": float(value)} for value in close]
    table = pandas.DataFrame({"close": close}, copy=False)

    np.testing.assert_allclose(
        rtta.CUSUM(threshold=0.9, drift=0.03).batch(close),
        _cusum(close, threshold=0.9, drift=0.03),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rtta.CUSUM(threshold=0.9, drift=0.03).batch(records),
        _cusum(close, threshold=0.9, drift=0.03),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rtta.CUSUM(threshold=0.9, drift=0.03).batch(table),
        _cusum(close, threshold=0.9, drift=0.03),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rtta.CUSUM(threshold=0.9, drift=0.03).batch(close32),
        _cusum(close32, threshold=0.9, drift=0.03),
        rtol=0.0,
        atol=0.0,
    )


def test_advance_last_and_replay_paths_match_update_checksum():
    import rtta

    close = np.ascontiguousarray(
        [100.0, 100.35, 100.85, 101.25, 100.4, 99.95, 99.5],
        dtype=np.float64,
    )
    expected = rtta.CUSUM(threshold=0.8, drift=0.05).batch(close)

    advance_indicator = rtta.CUSUM(threshold=0.8, drift=0.05)
    for value, expected_value in zip(close, expected):
        assert advance_indicator.advance(float(value)) is None
        assert advance_indicator.last() == expected_value

    assert rtta.CUSUM(threshold=0.8, drift=0.05).replay_update(close) == pytest.approx(
        float(expected.sum())
    )
    assert rtta.CUSUM(threshold=0.8, drift=0.05).replay_advance(close) == pytest.approx(
        float(expected.sum())
    )


def test_invalid_parameters_are_rejected():
    import rtta

    with pytest.raises(ValueError):
        rtta.CUSUM(threshold=0.0)
    with pytest.raises(ValueError):
        rtta.CUSUM(threshold=math.inf)
    with pytest.raises(ValueError):
        rtta.CUSUM(drift=-0.1)
    with pytest.raises(ValueError):
        rtta.CUSUM(drift=math.nan)
