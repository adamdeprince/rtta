import math

import numpy as np
import pytest


def _reference_alpha_beta_gamma(
    close,
    alpha=0.65,
    beta=0.25,
    gamma=0.05,
    dt=1.0,
    initial_price=math.nan,
    initial_velocity=0.0,
    initial_acceleration=0.0,
    fillna=True,
):
    close = np.asarray(close, dtype=np.float64)
    price = np.empty_like(close)
    velocity = np.empty_like(close)
    acceleration = np.empty_like(close)
    residual = np.empty_like(close)
    state_price = 0.0
    state_velocity = initial_velocity
    state_acceleration = initial_acceleration
    initialized = False
    count = 0

    for i, value in enumerate(close):
        if not initialized:
            initialized = True
            if math.isnan(initial_price):
                state_price = value
                out = (state_price, state_velocity, state_acceleration, 0.0)
                count += 1
                if not fillna and count < 2:
                    out = (math.nan, math.nan, math.nan, math.nan)
                price[i], velocity[i], acceleration[i], residual[i] = out
                continue
            state_price = initial_price

        predicted_price = state_price + state_velocity * dt + 0.5 * state_acceleration * dt * dt
        predicted_velocity = state_velocity + state_acceleration * dt
        predicted_acceleration = state_acceleration
        innovation = value - predicted_price
        state_price = predicted_price + alpha * innovation
        state_velocity = predicted_velocity + beta * innovation / dt
        state_acceleration = predicted_acceleration + 2.0 * gamma * innovation / (dt * dt)
        count += 1
        if not fillna and count < 2:
            price[i] = math.nan
            velocity[i] = math.nan
            acceleration[i] = math.nan
            residual[i] = math.nan
        else:
            price[i] = state_price
            velocity[i] = state_velocity
            acceleration[i] = state_acceleration
            residual[i] = innovation

    return price, velocity, acceleration, residual


def _assert_result_close(result, expected):
    for field, values in zip(("price", "velocity", "acceleration", "residual"), expected):
        np.testing.assert_allclose(getattr(result, field), values, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_update_matches_reference_equations():
    import rtta

    close = np.asarray([100.0, 100.4, 100.9, 101.1, 100.8, 101.5], dtype=np.float64)
    expected = _reference_alpha_beta_gamma(close, alpha=0.7, beta=0.3, gamma=0.08, dt=0.5)
    indicator = rtta.AlphaBetaGammaTrackingFilter(alpha=0.7, beta=0.3, gamma=0.08, dt=0.5)

    actual = [indicator.update(float(value)) for value in close]

    for index, result in enumerate(actual):
        assert result.price == pytest.approx(expected[0][index])
        assert result.velocity == pytest.approx(expected[1][index])
        assert result.acceleration == pytest.approx(expected[2][index])
        assert result.residual == pytest.approx(expected[3][index])
    assert indicator.last().price == pytest.approx(expected[0][-1])


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(321)
    close = np.ascontiguousarray(100.0 + np.cumsum(rng.normal(0.0, 0.35, 512)), dtype=np.float64)
    expected = _reference_alpha_beta_gamma(close, alpha=0.6, beta=0.2, gamma=0.04)
    records = [{"close": float(value)} for value in close]
    table = pandas.DataFrame({"close": close}, copy=False)
    close32 = np.ascontiguousarray(close.astype(np.float32))

    _assert_result_close(rtta.AlphaBetaGammaTrackingFilter(0.6, 0.2, 0.04).batch(close), expected)
    _assert_result_close(rtta.AlphaBetaGammaTrackingFilter(0.6, 0.2, 0.04).batch(records), expected)
    _assert_result_close(rtta.AlphaBetaGammaTrackingFilter(0.6, 0.2, 0.04).batch(table), expected)
    _assert_result_close(
        rtta.AlphaBetaGammaTrackingFilter(0.6, 0.2, 0.04).batch(close32),
        _reference_alpha_beta_gamma(close32, alpha=0.6, beta=0.2, gamma=0.04),
    )


def test_fillna_false_advance_last_and_replay_outputs():
    import rtta

    close = np.ascontiguousarray([100.0, 100.3, 100.7, 100.6], dtype=np.float64)
    expected = _reference_alpha_beta_gamma(close, fillna=False)

    batch = rtta.AlphaBetaGammaTrackingFilter(fillna=False).batch(close)
    _assert_result_close(batch, expected)
    assert math.isnan(batch.price[0])
    assert np.isfinite(batch.price[1])

    update_indicator = rtta.AlphaBetaGammaTrackingFilter()
    update_results = [update_indicator.update(float(value)) for value in close]
    advance_indicator = rtta.AlphaBetaGammaTrackingFilter()
    for value, result in zip(close, update_results):
        assert advance_indicator.advance(float(value)) is None
        assert advance_indicator.last().price == pytest.approx(result.price)
        assert advance_indicator.last_price() == pytest.approx(result.price)
        assert advance_indicator.last_velocity() == pytest.approx(result.velocity)

    replay = rtta.AlphaBetaGammaTrackingFilter().replay_update_outputs(close)
    np.testing.assert_allclose(replay.price, [result.price for result in update_results], rtol=1e-12, atol=1e-12)
    checksum = sum(result.price + result.velocity + result.acceleration + result.residual for result in update_results)
    assert rtta.AlphaBetaGammaTrackingFilter().replay_update(close) == pytest.approx(checksum)
    assert rtta.AlphaBetaGammaTrackingFilter().replay_advance(close) == pytest.approx(checksum)
