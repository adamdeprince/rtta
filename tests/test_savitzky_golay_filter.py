import math

import numpy as np
import pytest


def _coefficients(window, polynomial_order, derivative_order, dt):
    columns = min(polynomial_order, window - 1) + 1
    if derivative_order >= columns:
        return np.zeros(window, dtype=np.float64)
    x = np.arange(window, dtype=np.float64) - float(window - 1)
    design = np.vstack([x**power for power in range(columns)]).T
    inverse = np.linalg.inv(design.T @ design)
    scale = math.factorial(derivative_order) / (dt**derivative_order)
    return scale * inverse[derivative_order] @ design.T


def _reference_savgol(close, window=7, polynomial_order=2, dt=1.0, fillna=True):
    close = np.asarray(close, dtype=np.float64)
    window = max(window, 3)
    polynomial_order = min(max(polynomial_order, 2), window - 1)
    output = [np.empty_like(close) for _ in range(3)]
    history = []

    for index, value in enumerate(close):
        history.append(float(value))
        history = history[-window:]
        if not fillna and len(history) < window:
            for target in output:
                target[index] = math.nan
            continue
        if len(history) == 1:
            values = (history[-1], 0.0, 0.0)
        else:
            order = min(polynomial_order, len(history) - 1)
            smooth = float(_coefficients(len(history), order, 0, dt) @ np.asarray(history))
            first = float(_coefficients(len(history), order, 1, dt) @ np.asarray(history))
            second = float(_coefficients(len(history), order, 2, dt) @ np.asarray(history)) if order >= 2 else 0.0
            values = (smooth, first, second)
        for target, result in zip(output, values):
            target[index] = result
    return tuple(output)


def _assert_result_close(result, expected):
    for field, values in zip(("smooth", "first_derivative", "second_derivative"), expected):
        np.testing.assert_allclose(getattr(result, field), values, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_update_matches_polynomial_fit_reference():
    import rtta

    close = np.asarray([100.0, 100.5, 101.4, 101.8, 102.1, 102.9, 103.2], dtype=np.float64)
    expected = _reference_savgol(close, window=5, polynomial_order=2, dt=0.5)
    indicator = rtta.SavitzkyGolayFilter(window=5, polynomial_order=2, dt=0.5)

    actual = [indicator.update(float(value)) for value in close]

    for index, result in enumerate(actual):
        assert result.smooth == pytest.approx(expected[0][index], rel=1e-11, abs=1e-11)
        assert result.first_derivative == pytest.approx(expected[1][index], rel=1e-11, abs=1e-11)
        assert result.second_derivative == pytest.approx(expected[2][index], rel=1e-11, abs=1e-11)
    assert indicator.last().smooth == pytest.approx(expected[0][-1], rel=1e-11, abs=1e-11)


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(111)
    close = np.ascontiguousarray(100.0 + np.cumsum(rng.normal(0.0, 0.35, 512)), dtype=np.float64)
    expected = _reference_savgol(close, window=9, polynomial_order=3)
    records = [{"close": float(value)} for value in close]
    table = pandas.DataFrame({"close": close}, copy=False)
    close32 = np.ascontiguousarray(close.astype(np.float32))

    _assert_result_close(rtta.SavitzkyGolayFilter(window=9, polynomial_order=3).batch(close), expected)
    _assert_result_close(rtta.SavitzkyGolayFilter(window=9, polynomial_order=3).batch(records), expected)
    _assert_result_close(rtta.SavitzkyGolayFilter(window=9, polynomial_order=3).batch(table), expected)
    _assert_result_close(
        rtta.SavitzkyGolayFilter(window=9, polynomial_order=3).batch(close32),
        _reference_savgol(close32, window=9, polynomial_order=3),
    )


def test_fillna_false_advance_last_scalar_accessors_and_replay_outputs():
    import rtta

    close = np.ascontiguousarray([100.0, 100.3, 100.7, 100.6, 101.2], dtype=np.float64)
    batch = rtta.SavitzkyGolayFilter(window=5, fillna=False).batch(close)
    assert math.isnan(batch.smooth[0])
    assert math.isnan(batch.smooth[3])
    assert np.isfinite(batch.smooth[4])

    update_indicator = rtta.SavitzkyGolayFilter(window=5)
    update_results = [update_indicator.update(float(value)) for value in close]
    advance_indicator = rtta.SavitzkyGolayFilter(window=5)
    for value, result in zip(close, update_results):
        assert advance_indicator.advance(float(value)) is None
        assert advance_indicator.last().smooth == pytest.approx(result.smooth)
        assert advance_indicator.last_smooth() == pytest.approx(result.smooth)
        assert advance_indicator.last_first_derivative() == pytest.approx(result.first_derivative)

    replay = rtta.SavitzkyGolayFilter(window=5).replay_update_outputs(close)
    np.testing.assert_allclose(replay.smooth, [result.smooth for result in update_results], rtol=1e-10, atol=1e-10)
    checksum = sum(result.smooth + result.first_derivative + result.second_derivative for result in update_results)
    assert rtta.SavitzkyGolayFilter(window=5).replay_update(close) == pytest.approx(checksum)
    assert rtta.SavitzkyGolayFilter(window=5).replay_advance(close) == pytest.approx(checksum)
