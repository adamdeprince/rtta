import math

import numpy as np
import pytest


def _reference_nadaraya_watson(close, window=64, bandwidth=8.0, multiplier=2.0, fillna=True):
    close = np.asarray(close, dtype=np.float64)
    window = max(int(window), 1)
    bandwidth = bandwidth if math.isfinite(bandwidth) and bandwidth > 0.0 else 8.0
    multiplier = abs(multiplier) if math.isfinite(multiplier) else 2.0
    weights = np.exp(-0.5 * (np.arange(window, dtype=np.float64) / bandwidth) ** 2)
    middle = np.empty_like(close)
    upper = np.empty_like(close)
    lower = np.empty_like(close)
    history = []

    for index, value in enumerate(close):
        history.append(float(value))
        history = history[-window:]
        if not fillna and len(history) < window:
            middle[index] = math.nan
            upper[index] = math.nan
            lower[index] = math.nan
            continue
        values = np.asarray(history, dtype=np.float64)
        current_weights = weights[: len(values)][::-1]
        center = float(np.dot(current_weights, values) / current_weights.sum())
        variance = float(np.dot(current_weights, (values - center) ** 2) / current_weights.sum())
        band = multiplier * math.sqrt(max(variance, 0.0))
        middle[index] = center
        upper[index] = center + band
        lower[index] = center - band
    return middle, upper, lower


def _assert_result_close(result, expected):
    for field, values in zip(("middle", "upper", "lower"), expected):
        np.testing.assert_allclose(getattr(result, field), values, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_update_matches_gaussian_kernel_reference():
    import rtta

    close = np.asarray([100.0, 100.5, 100.2, 101.0, 101.4, 101.1], dtype=np.float64)
    expected = _reference_nadaraya_watson(close, window=4, bandwidth=1.5, multiplier=1.8)
    indicator = rtta.NadarayaWatsonEnvelope(window=4, bandwidth=1.5, multiplier=1.8)

    actual = [indicator.update(float(value)) for value in close]

    for index, result in enumerate(actual):
        assert result.middle == pytest.approx(expected[0][index], rel=1e-12, abs=1e-12)
        assert result.upper == pytest.approx(expected[1][index], rel=1e-12, abs=1e-12)
        assert result.lower == pytest.approx(expected[2][index], rel=1e-12, abs=1e-12)
    assert indicator.last().middle == pytest.approx(expected[0][-1], rel=1e-12, abs=1e-12)


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(222)
    close = np.ascontiguousarray(100.0 + np.cumsum(rng.normal(0.0, 0.35, 512)), dtype=np.float64)
    expected = _reference_nadaraya_watson(close, window=32, bandwidth=6.0, multiplier=2.5)
    records = [{"close": float(value)} for value in close]
    table = pandas.DataFrame({"close": close}, copy=False)
    close32 = np.ascontiguousarray(close.astype(np.float32))

    _assert_result_close(rtta.NadarayaWatsonEnvelope(window=32, bandwidth=6.0, multiplier=2.5).batch(close), expected)
    _assert_result_close(rtta.NadarayaWatsonEnvelope(window=32, bandwidth=6.0, multiplier=2.5).batch(records), expected)
    _assert_result_close(rtta.NadarayaWatsonEnvelope(window=32, bandwidth=6.0, multiplier=2.5).batch(table), expected)
    _assert_result_close(
        rtta.NadarayaWatsonEnvelope(window=32, bandwidth=6.0, multiplier=2.5).batch(close32),
        _reference_nadaraya_watson(close32, window=32, bandwidth=6.0, multiplier=2.5),
    )


def test_fillna_false_advance_last_scalar_accessors_and_replay_outputs():
    import rtta

    close = np.ascontiguousarray([100.0, 100.3, 100.7, 100.6], dtype=np.float64)
    batch = rtta.NadarayaWatsonEnvelope(window=4, fillna=False).batch(close)
    assert math.isnan(batch.middle[0])
    assert math.isnan(batch.middle[2])
    assert np.isfinite(batch.middle[3])

    update_indicator = rtta.NadarayaWatsonEnvelope(window=4)
    update_results = [update_indicator.update(float(value)) for value in close]
    advance_indicator = rtta.NadarayaWatsonEnvelope(window=4)
    for value, result in zip(close, update_results):
        assert advance_indicator.advance(float(value)) is None
        assert advance_indicator.last().middle == pytest.approx(result.middle)
        assert advance_indicator.last_middle() == pytest.approx(result.middle)
        assert advance_indicator.last_upper() == pytest.approx(result.upper)

    replay = rtta.NadarayaWatsonEnvelope(window=4).replay_update_outputs(close)
    np.testing.assert_allclose(replay.middle, [result.middle for result in update_results], rtol=1e-12, atol=1e-12)
    checksum = sum(result.middle + result.upper + result.lower for result in update_results)
    assert rtta.NadarayaWatsonEnvelope(window=4).replay_update(close) == pytest.approx(checksum)
    assert rtta.NadarayaWatsonEnvelope(window=4).replay_advance(close) == pytest.approx(checksum)
