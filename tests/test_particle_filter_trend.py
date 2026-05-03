import math

import numpy as np
import pytest


MASK = (1 << 64) - 1
MULTIPLIER = 2685821657736338717


class _ReferenceRng:
    def __init__(self, seed):
        self.state = 1 if seed == 0 else seed & MASK

    def uniform_open(self):
        self.state ^= self.state >> 12
        self.state &= MASK
        self.state ^= (self.state << 25) & MASK
        self.state &= MASK
        self.state ^= self.state >> 27
        self.state &= MASK
        value = (self.state * MULTIPLIER) & MASK
        return (((value >> 11) + 0.5) / float(1 << 53))

    def normal(self):
        u1 = max(self.uniform_open(), 1.0e-16)
        u2 = self.uniform_open()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _reference_particle_filter(
    close,
    particles=32,
    initial_price=math.nan,
    initial_velocity=0.0,
    dt=1.0,
    process_position_scale=0.05,
    process_velocity_scale=0.01,
    measurement_scale=0.5,
    resample_threshold=0.5,
    seed=7,
    fillna=True,
):
    close = np.asarray(close, dtype=np.float64)
    particles = max(int(particles), 8)
    rng = _ReferenceRng(seed)
    positions = np.zeros(particles, dtype=np.float64)
    velocities = np.zeros(particles, dtype=np.float64)
    weights = np.full(particles, 1.0 / particles, dtype=np.float64)
    initialized = False
    count = 0
    outputs = [np.empty_like(close) for _ in range(4)]

    def initialize(price):
        spread = 0.25 * measurement_scale
        for index in range(particles):
            positions[index] = price + spread * rng.normal()
            velocities[index] = initial_velocity + process_velocity_scale * rng.normal()
            weights[index] = 1.0 / particles

    def write(index, trend, velocity, signal, effective_sample_size):
        outputs[0][index] = trend
        outputs[1][index] = velocity
        outputs[2][index] = signal
        outputs[3][index] = effective_sample_size

    for index, value in enumerate(close):
        if not initialized:
            initialize(value if math.isnan(initial_price) else initial_price)
            initialized = True
            if math.isnan(initial_price):
                trend = float(np.dot(weights, positions))
                velocity = float(np.dot(weights, velocities))
                signal = 1.0 if value > trend else (-1.0 if value < trend else 0.0)
                count += 1
                if not fillna and count < 2:
                    write(index, math.nan, math.nan, math.nan, math.nan)
                else:
                    write(index, trend, velocity, signal, float(particles))
                continue

        sqrt_dt = math.sqrt(dt)
        log_weights = np.empty(particles, dtype=np.float64)
        for particle in range(particles):
            velocities[particle] += process_velocity_scale * sqrt_dt * rng.normal()
            positions[particle] += velocities[particle] * dt + process_position_scale * sqrt_dt * rng.normal()
            innovation = value - positions[particle]
            log_weights[particle] = math.log(max(weights[particle], 1.0e-300)) - abs(innovation) / measurement_scale

        scaled = np.exp(log_weights - np.max(log_weights))
        weights[:] = scaled / scaled.sum()
        effective_sample_size = 1.0 / float(np.dot(weights, weights))
        trend = float(np.dot(weights, positions))
        velocity = float(np.dot(weights, velocities))
        signal = 1.0 if value > trend else (-1.0 if value < trend else 0.0)

        if effective_sample_size < resample_threshold * particles:
            new_positions = np.empty_like(positions)
            new_velocities = np.empty_like(velocities)
            step = 1.0 / particles
            cursor = rng.uniform_open() * step
            cumulative = weights[0]
            source = 0
            for particle in range(particles):
                while cursor > cumulative and source + 1 < particles:
                    source += 1
                    cumulative += weights[source]
                new_positions[particle] = positions[source]
                new_velocities[particle] = velocities[source]
                cursor += step
            positions[:] = new_positions
            velocities[:] = new_velocities
            weights[:] = 1.0 / particles

        count += 1
        if not fillna and count < 2:
            write(index, math.nan, math.nan, math.nan, math.nan)
        else:
            write(index, trend, velocity, signal, effective_sample_size)

    return tuple(outputs)


def _assert_result_close(result, expected):
    for field, values in zip(("trend", "velocity", "signal", "effective_sample_size"), expected):
        np.testing.assert_allclose(getattr(result, field), values, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_update_matches_seeded_particle_reference():
    import rtta

    close = np.asarray([100.0, 100.2, 99.9, 100.6, 100.8, 100.4], dtype=np.float64)
    expected = _reference_particle_filter(close, particles=32, seed=11)
    indicator = rtta.ParticleFilterTrend(particles=32, seed=11)

    actual = [indicator.update(float(value)) for value in close]

    for index, result in enumerate(actual):
        assert result.trend == pytest.approx(expected[0][index], rel=1e-10, abs=1e-10)
        assert result.velocity == pytest.approx(expected[1][index], rel=1e-10, abs=1e-10)
        assert result.signal == expected[2][index]
        assert result.effective_sample_size == pytest.approx(expected[3][index], rel=1e-10, abs=1e-10)


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(987)
    close = np.ascontiguousarray(100.0 + np.cumsum(rng.normal(0.0, 0.35, 512)), dtype=np.float64)
    expected = _reference_particle_filter(close, particles=32, seed=5)
    records = [{"close": float(value)} for value in close]
    table = pandas.DataFrame({"close": close}, copy=False)
    close32 = np.ascontiguousarray(close.astype(np.float32))

    _assert_result_close(rtta.ParticleFilterTrend(particles=32, seed=5).batch(close), expected)
    _assert_result_close(rtta.ParticleFilterTrend(particles=32, seed=5).batch(records), expected)
    _assert_result_close(rtta.ParticleFilterTrend(particles=32, seed=5).batch(table), expected)
    _assert_result_close(
        rtta.ParticleFilterTrend(particles=32, seed=5).batch(close32),
        _reference_particle_filter(close32, particles=32, seed=5),
    )


def test_fillna_false_advance_last_scalar_accessors_and_replay_outputs():
    import rtta

    close = np.ascontiguousarray([100.0, 100.1, 99.8, 100.5], dtype=np.float64)
    batch = rtta.ParticleFilterTrend(particles=16, seed=3, fillna=False).batch(close)
    assert math.isnan(batch.trend[0])
    assert np.isfinite(batch.trend[1])

    update_indicator = rtta.ParticleFilterTrend(particles=16, seed=3)
    update_results = [update_indicator.update(float(value)) for value in close]
    advance_indicator = rtta.ParticleFilterTrend(particles=16, seed=3)
    for value, result in zip(close, update_results):
        assert advance_indicator.advance(float(value)) is None
        assert advance_indicator.last().trend == pytest.approx(result.trend)
        assert advance_indicator.last_trend() == pytest.approx(result.trend)
        assert advance_indicator.last_effective_sample_size() == pytest.approx(result.effective_sample_size)

    replay = rtta.ParticleFilterTrend(particles=16, seed=3).replay_update_outputs(close)
    np.testing.assert_allclose(replay.trend, [result.trend for result in update_results], rtol=1e-10, atol=1e-10)
    checksum = sum(result.trend + result.velocity + result.signal + result.effective_sample_size for result in update_results)
    assert rtta.ParticleFilterTrend(particles=16, seed=3).replay_update(close) == pytest.approx(checksum)
    assert rtta.ParticleFilterTrend(particles=16, seed=3).replay_advance(close) == pytest.approx(checksum)
