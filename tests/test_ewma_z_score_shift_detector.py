import math

import numpy as np
import pytest


def _ewma_z_score_shift_detector(close, alpha=0.05, threshold=3.0, min_variance=1.0e-12):
    mean = 0.0
    variance = 0.0
    count = 0
    output = []

    for value in close:
        value = float(value)
        if count == 0:
            mean = value
            variance = 0.0
            count = 1
            output.append(0.0)
            continue

        signal = 0.0
        if count >= 2:
            z_score = (value - mean) / math.sqrt(max(variance, min_variance))
            if z_score > threshold:
                signal = 1.0
            elif z_score < -threshold:
                signal = -1.0

        if signal != 0.0:
            mean = value
            variance = 0.0
            count = 1
        else:
            delta = value - mean
            mean += alpha * delta
            variance = (1.0 - alpha) * (variance + alpha * delta * delta)
            count += 1

        output.append(signal)

    return np.asarray(output, dtype=np.float64)


def test_update_scores_current_tick_against_prior_ewma_state_and_last():
    import rtta

    close = np.asarray([
        100.0,
        100.1,
        99.9,
        100.05,
        100.0,
        102.0,
        102.1,
        101.9,
        102.0,
        99.5,
        99.6,
    ])
    expected = _ewma_z_score_shift_detector(close, alpha=0.2, threshold=3.0, min_variance=1.0e-4)

    indicator = rtta.EWMAZScoreShiftDetector(alpha=0.2, threshold=3.0, min_variance=1.0e-4)
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
    rng = np.random.default_rng(20260530)
    close = np.ascontiguousarray(
        100.0 + np.cumsum(rng.normal(0.0, 0.35, 512)),
        dtype=np.float64,
    )
    close32 = np.ascontiguousarray(close.astype(np.float32))
    records = [{"close": float(value)} for value in close]
    table = pandas.DataFrame({"close": close}, copy=False)

    np.testing.assert_allclose(
        rtta.EWMAZScoreShiftDetector(alpha=0.08, threshold=3.0).batch(close),
        _ewma_z_score_shift_detector(close, alpha=0.08, threshold=3.0),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rtta.EWMAZScoreShiftDetector(alpha=0.08, threshold=3.0).batch(records),
        _ewma_z_score_shift_detector(close, alpha=0.08, threshold=3.0),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rtta.EWMAZScoreShiftDetector(alpha=0.08, threshold=3.0).batch(table),
        _ewma_z_score_shift_detector(close, alpha=0.08, threshold=3.0),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rtta.EWMAZScoreShiftDetector(alpha=0.08, threshold=3.0).batch(close32),
        _ewma_z_score_shift_detector(close32, alpha=0.08, threshold=3.0),
        rtol=0.0,
        atol=0.0,
    )


def test_advance_last_and_replay_paths_match_update_checksum():
    import rtta

    close = np.ascontiguousarray(
        [100.0, 100.1, 99.9, 100.0, 102.2, 102.1, 99.4],
        dtype=np.float64,
    )
    expected = rtta.EWMAZScoreShiftDetector(
        alpha=0.25,
        threshold=2.5,
        min_variance=1.0e-4,
    ).batch(close)

    advance_indicator = rtta.EWMAZScoreShiftDetector(
        alpha=0.25,
        threshold=2.5,
        min_variance=1.0e-4,
    )
    for value, expected_value in zip(close, expected):
        assert advance_indicator.advance(float(value)) is None
        assert advance_indicator.last() == expected_value

    assert rtta.EWMAZScoreShiftDetector(
        alpha=0.25,
        threshold=2.5,
        min_variance=1.0e-4,
    ).replay_update(close) == pytest.approx(float(expected.sum()))
    assert rtta.EWMAZScoreShiftDetector(
        alpha=0.25,
        threshold=2.5,
        min_variance=1.0e-4,
    ).replay_advance(close) == pytest.approx(float(expected.sum()))


def test_invalid_parameters_are_rejected():
    import rtta

    with pytest.raises(ValueError):
        rtta.EWMAZScoreShiftDetector(alpha=0.0)
    with pytest.raises(ValueError):
        rtta.EWMAZScoreShiftDetector(alpha=math.inf)
    with pytest.raises(ValueError):
        rtta.EWMAZScoreShiftDetector(alpha=1.01)
    with pytest.raises(ValueError):
        rtta.EWMAZScoreShiftDetector(threshold=0.0)
    with pytest.raises(ValueError):
        rtta.EWMAZScoreShiftDetector(min_variance=0.0)
    with pytest.raises(ValueError):
        rtta.EWMAZScoreShiftDetector(min_variance=math.nan)
