import math

import numpy as np
import pytest


def _variance_floor(value, minimum=1.0e-12):
    return minimum if not math.isfinite(value) or value < minimum else value


def _reference_imm(
    close,
    initial_price=math.nan,
    initial_velocity=0.0,
    dt=1.0,
    low_vol_process_variance=1.0e-5,
    high_vol_process_variance=1.0e-2,
    trend_process_variance=1.0e-4,
    chop_process_variance=1.0e-3,
    measurement_variance=0.25,
    stickiness=0.96,
    fillna=True,
):
    close = np.asarray(close, dtype=np.float64)
    measurement = _variance_floor(measurement_variance)
    specs = (
        (1.0, 0.85, _variance_floor(low_vol_process_variance), _variance_floor(low_vol_process_variance), _variance_floor(0.5 * measurement)),
        (1.0, 0.85, _variance_floor(high_vol_process_variance), _variance_floor(high_vol_process_variance), _variance_floor(2.0 * measurement)),
        (1.0, 1.0, _variance_floor(0.25 * trend_process_variance), _variance_floor(trend_process_variance), measurement),
        (0.15, 0.20, _variance_floor(chop_process_variance), _variance_floor(2.0 * chop_process_variance), measurement),
    )
    stickiness = min(max(stickiness if math.isfinite(stickiness) else 0.96, 0.25), 0.999)
    probabilities = np.full(4, 0.25, dtype=np.float64)
    models = np.zeros((4, 5), dtype=np.float64)
    initialized = False
    count = 0
    outputs = [np.empty_like(close) for _ in range(6)]

    def transition(src, dst):
        return stickiness if src == dst else (1.0 - stickiness) / 3.0

    def combined():
        value = float(np.dot(probabilities, models[:, 0]))
        velocity = float(np.dot(probabilities, models[:, 1]))
        return value, velocity, *probabilities.tolist()

    def write(index, values):
        for target, value in zip(outputs, values):
            target[index] = value

    for index, value in enumerate(close):
        if not initialized:
            seed = value if math.isnan(initial_price) else initial_price
            models[:, 0] = seed
            models[:, 1] = initial_velocity
            models[:, 2] = 1.0
            models[:, 3] = 0.0
            models[:, 4] = 1.0
            initialized = True
            if math.isnan(initial_price):
                count += 1
                out = combined()
                write(index, (math.nan,) * 6 if not fillna and count < 2 else out)
                continue

        mixed = np.zeros_like(models)
        predicted_probabilities = np.zeros(4, dtype=np.float64)
        for dst in range(4):
            predicted_probabilities[dst] = sum(probabilities[src] * transition(src, dst) for src in range(4))
            predicted_probabilities[dst] = _variance_floor(float(predicted_probabilities[dst]))
            weights = np.asarray(
                [probabilities[src] * transition(src, dst) / predicted_probabilities[dst] for src in range(4)],
                dtype=np.float64,
            )
            price = float(np.dot(weights, models[:, 0]))
            velocity = float(np.dot(weights, models[:, 1]))
            p00 = p01 = p11 = 0.0
            for src, weight in enumerate(weights):
                dp = models[src, 0] - price
                dv = models[src, 1] - velocity
                p00 += weight * (models[src, 2] + dp * dp)
                p01 += weight * (models[src, 3] + dp * dv)
                p11 += weight * (models[src, 4] + dv * dv)
            mixed[dst] = (price, velocity, _variance_floor(p00), p01, _variance_floor(p11))

        log_likelihoods = np.empty(4, dtype=np.float64)
        for model_index, (loading_factor, persistence, q_price, q_velocity, r_value) in enumerate(specs):
            loading = loading_factor * dt
            price, velocity, p00, p01, p11 = mixed[model_index]
            predicted_price = price + loading * velocity
            predicted_velocity = persistence * velocity
            pred_p00 = p00 + loading * p01 + loading * p01 + loading * loading * p11 + q_price
            pred_p01 = persistence * (p01 + loading * p11)
            pred_p11 = persistence * persistence * p11 + q_velocity
            innovation = value - predicted_price
            innovation_variance = _variance_floor(pred_p00 + r_value)
            k0 = pred_p00 / innovation_variance
            k1 = pred_p01 / innovation_variance
            mixed[model_index] = (
                predicted_price + k0 * innovation,
                predicted_velocity + k1 * innovation,
                _variance_floor((1.0 - k0) * pred_p00),
                (1.0 - k0) * pred_p01,
                _variance_floor(pred_p11 - k1 * pred_p01),
            )
            log_likelihoods[model_index] = -0.5 * (
                math.log(2.0 * math.pi * innovation_variance) + innovation * innovation / innovation_variance
            )

        models = mixed
        weights = predicted_probabilities * np.exp(log_likelihoods - np.max(log_likelihoods))
        probabilities = weights / weights.sum() if np.isfinite(weights.sum()) and weights.sum() > 0.0 else predicted_probabilities / predicted_probabilities.sum()
        count += 1
        out = combined()
        write(index, (math.nan,) * 6 if not fillna and count < 2 else out)

    return tuple(outputs)


def _assert_result_close(result, expected):
    fields = (
        "value",
        "velocity",
        "low_vol_probability",
        "high_vol_probability",
        "trend_probability",
        "chop_probability",
    )
    for field, values in zip(fields, expected):
        np.testing.assert_allclose(getattr(result, field), values, rtol=1e-11, atol=1e-11, equal_nan=True)


def test_update_matches_reference_imm_cycle():
    import rtta

    close = np.asarray([100.0, 100.2, 100.4, 101.5, 101.7, 101.6], dtype=np.float64)
    expected = _reference_imm(close, stickiness=0.9, measurement_variance=0.3)
    indicator = rtta.InteractingMultipleModelFilter(stickiness=0.9, measurement_variance=0.3)

    actual = [indicator.update(float(value)) for value in close]

    for index, result in enumerate(actual):
        assert result.value == pytest.approx(expected[0][index], rel=1e-12, abs=1e-12)
        assert result.velocity == pytest.approx(expected[1][index], rel=1e-12, abs=1e-12)
        probability_sum = result.low_vol_probability + result.high_vol_probability + result.trend_probability + result.chop_probability
        assert probability_sum == pytest.approx(1.0, rel=1e-12, abs=1e-12)


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(654)
    close = np.ascontiguousarray(100.0 + np.cumsum(rng.normal(0.0, 0.35, 512)), dtype=np.float64)
    expected = _reference_imm(close)
    records = [{"close": float(value)} for value in close]
    table = pandas.DataFrame({"close": close}, copy=False)
    close32 = np.ascontiguousarray(close.astype(np.float32))

    _assert_result_close(rtta.InteractingMultipleModelFilter().batch(close), expected)
    _assert_result_close(rtta.InteractingMultipleModelFilter().batch(records), expected)
    _assert_result_close(rtta.InteractingMultipleModelFilter().batch(table), expected)
    _assert_result_close(rtta.InteractingMultipleModelFilter().batch(close32), _reference_imm(close32))


def test_fillna_false_advance_last_scalar_accessors_and_replay_outputs():
    import rtta

    close = np.ascontiguousarray([100.0, 100.1, 99.8, 100.5], dtype=np.float64)
    batch = rtta.InteractingMultipleModelFilter(fillna=False).batch(close)
    assert math.isnan(batch.value[0])
    assert np.isfinite(batch.value[1])

    update_indicator = rtta.InteractingMultipleModelFilter()
    update_results = [update_indicator.update(float(value)) for value in close]
    advance_indicator = rtta.InteractingMultipleModelFilter()
    for value, result in zip(close, update_results):
        assert advance_indicator.advance(float(value)) is None
        assert advance_indicator.last().value == pytest.approx(result.value)
        assert advance_indicator.last_value() == pytest.approx(result.value)
        assert advance_indicator.last_trend_probability() == pytest.approx(result.trend_probability)

    replay = rtta.InteractingMultipleModelFilter().replay_update_outputs(close)
    np.testing.assert_allclose(replay.value, [result.value for result in update_results], rtol=1e-12, atol=1e-12)
    checksum = sum(
        result.value
        + result.velocity
        + result.low_vol_probability
        + result.high_vol_probability
        + result.trend_probability
        + result.chop_probability
        for result in update_results
    )
    assert rtta.InteractingMultipleModelFilter().replay_update(close) == pytest.approx(checksum)
    assert rtta.InteractingMultipleModelFilter().replay_advance(close) == pytest.approx(checksum)
