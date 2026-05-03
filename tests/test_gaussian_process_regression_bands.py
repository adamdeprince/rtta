import math

import numpy as np
import pytest


def _reference_gpr(
    close,
    window=32,
    length_scale=8.0,
    signal_variance=1.0,
    noise_variance=0.25,
    multiplier=2.0,
    fillna=True,
):
    close = np.asarray(close, dtype=np.float64)
    window = max(int(window), 1)
    length_scale = length_scale if math.isfinite(length_scale) and length_scale > 0.0 else 8.0
    signal_variance = signal_variance if math.isfinite(signal_variance) and signal_variance > 0.0 else 1.0
    noise_variance = noise_variance if math.isfinite(noise_variance) and noise_variance > 0.0 else 0.25
    multiplier = abs(multiplier) if math.isfinite(multiplier) else 2.0
    middle = np.empty_like(close)
    upper = np.empty_like(close)
    lower = np.empty_like(close)
    history = []

    def kernel(left, right):
        scaled = (left - right) / length_scale
        return signal_variance * math.exp(-0.5 * scaled * scaled)

    for index, value in enumerate(close):
        history.append(float(value))
        history = history[-window:]
        if not fillna and len(history) < window:
            middle[index] = math.nan
            upper[index] = math.nan
            lower[index] = math.nan
            continue
        values = np.asarray(history, dtype=np.float64)
        size = len(values)
        x = np.arange(size, dtype=np.float64) - float(size - 1)
        matrix = np.empty((size, size), dtype=np.float64)
        target = np.empty(size, dtype=np.float64)
        for row in range(size):
            target[row] = kernel(x[row], 0.0)
            for col in range(size):
                matrix[row, col] = kernel(x[row], x[col])
            matrix[row, row] += noise_variance
        mean = float(values.mean())
        alpha = np.linalg.solve(matrix, values - mean)
        variance_weights = np.linalg.solve(matrix, target)
        center = mean + float(target @ alpha)
        variance = max(signal_variance - float(target @ variance_weights), 0.0)
        band = multiplier * math.sqrt(variance)
        middle[index] = center
        upper[index] = center + band
        lower[index] = center - band
    return middle, upper, lower


def _assert_result_close(result, expected):
    for field, values in zip(("middle", "upper", "lower"), expected):
        np.testing.assert_allclose(getattr(result, field), values, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_update_matches_gaussian_process_reference():
    import rtta

    close = np.asarray([100.0, 100.4, 100.2, 101.0, 101.5, 101.2], dtype=np.float64)
    expected = _reference_gpr(close, window=4, length_scale=1.7, signal_variance=1.4, noise_variance=0.2)
    indicator = rtta.GaussianProcessRegressionBands(window=4, length_scale=1.7, signal_variance=1.4, noise_variance=0.2)

    actual = [indicator.update(float(value)) for value in close]

    for index, result in enumerate(actual):
        assert result.middle == pytest.approx(expected[0][index], rel=1e-10, abs=1e-10)
        assert result.upper == pytest.approx(expected[1][index], rel=1e-10, abs=1e-10)
        assert result.lower == pytest.approx(expected[2][index], rel=1e-10, abs=1e-10)
    assert indicator.last().middle == pytest.approx(expected[0][-1], rel=1e-10, abs=1e-10)


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(333)
    close = np.ascontiguousarray(100.0 + np.cumsum(rng.normal(0.0, 0.35, 512)), dtype=np.float64)
    expected = _reference_gpr(close, window=12, length_scale=4.0, signal_variance=1.2, noise_variance=0.3)
    records = [{"close": float(value)} for value in close]
    table = pandas.DataFrame({"close": close}, copy=False)
    close32 = np.ascontiguousarray(close.astype(np.float32))

    _assert_result_close(rtta.GaussianProcessRegressionBands(window=12, length_scale=4.0, signal_variance=1.2, noise_variance=0.3).batch(close), expected)
    _assert_result_close(rtta.GaussianProcessRegressionBands(window=12, length_scale=4.0, signal_variance=1.2, noise_variance=0.3).batch(records), expected)
    _assert_result_close(rtta.GaussianProcessRegressionBands(window=12, length_scale=4.0, signal_variance=1.2, noise_variance=0.3).batch(table), expected)
    _assert_result_close(
        rtta.GaussianProcessRegressionBands(window=12, length_scale=4.0, signal_variance=1.2, noise_variance=0.3).batch(close32),
        _reference_gpr(close32, window=12, length_scale=4.0, signal_variance=1.2, noise_variance=0.3),
    )


def test_fillna_false_advance_last_scalar_accessors_and_replay_outputs():
    import rtta

    close = np.ascontiguousarray([100.0, 100.3, 100.7, 100.6], dtype=np.float64)
    batch = rtta.GaussianProcessRegressionBands(window=4, fillna=False).batch(close)
    assert math.isnan(batch.middle[0])
    assert math.isnan(batch.middle[2])
    assert np.isfinite(batch.middle[3])

    update_indicator = rtta.GaussianProcessRegressionBands(window=4)
    update_results = [update_indicator.update(float(value)) for value in close]
    advance_indicator = rtta.GaussianProcessRegressionBands(window=4)
    for value, result in zip(close, update_results):
        assert advance_indicator.advance(float(value)) is None
        assert advance_indicator.last().middle == pytest.approx(result.middle)
        assert advance_indicator.last_middle() == pytest.approx(result.middle)
        assert advance_indicator.last_lower() == pytest.approx(result.lower)

    replay = rtta.GaussianProcessRegressionBands(window=4).replay_update_outputs(close)
    np.testing.assert_allclose(replay.middle, [result.middle for result in update_results], rtol=1e-10, atol=1e-10)
    checksum = sum(result.middle + result.upper + result.lower for result in update_results)
    assert rtta.GaussianProcessRegressionBands(window=4).replay_update(close) == pytest.approx(checksum)
    assert rtta.GaussianProcessRegressionBands(window=4).replay_advance(close) == pytest.approx(checksum)
