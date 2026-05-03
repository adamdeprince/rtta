import numpy as np
import pytest
from types import SimpleNamespace

import rtta
from rtta.indicator import (
    KalmanInnovationZScore,
    KalmanLocalLinearTrend,
    KalmanMovingAverage,
    KalmanPredictionBands,
    KalmanVelocityOscillator,
)

kalman = pytest.importorskip("kalman")


def _fast_kalman_outputs(close, tuning):
    kf = kalman.make_constant_velocity_1d(
        tuning.initial_price,
        tuning.initial_velocity,
        tuning.dt,
        tuning.position_variance,
        tuning.velocity_variance,
        tuning.process_position_variance,
        tuning.process_velocity_variance,
        tuning.measurement_variance,
    )
    output = []
    for value in close:
        kf.update([float(value)])
        output.append(float(kf.x[0]))
    return np.asarray(output)


def _fast_kalman_level_trend(close, tuning):
    kf = kalman.make_constant_velocity_1d(
        tuning.initial_level,
        tuning.initial_trend,
        tuning.dt,
        tuning.level_variance,
        tuning.trend_variance,
        tuning.process_level_variance,
        tuning.process_trend_variance,
        tuning.observation_variance,
    )
    level = []
    trend = []
    for value in close:
        kf.update([float(value)])
        level.append(float(kf.x[0]))
        trend.append(float(kf.x[1]))
    return np.asarray(level), np.asarray(trend)


def _fast_kalman_velocity(close, tuning):
    kf = kalman.make_constant_velocity_1d(
        tuning.initial_price,
        tuning.initial_velocity,
        tuning.dt,
        tuning.position_variance,
        tuning.velocity_variance,
        tuning.process_position_variance,
        tuning.process_velocity_variance,
        tuning.measurement_variance,
    )
    output = []
    for value in close:
        kf.update([float(value)])
        output.append(float(kf.x[1]))
    return np.asarray(output)


def _fast_kalman_innovation_zscore(close, tuning):
    kf = kalman.make_constant_velocity_1d(
        tuning.initial_price,
        tuning.initial_velocity,
        tuning.dt,
        tuning.position_variance,
        tuning.velocity_variance,
        tuning.process_position_variance,
        tuning.process_velocity_variance,
        tuning.measurement_variance,
    )
    output = []
    for value in close:
        stats = kf.update([float(value)])
        innovation = float(stats["innovation"][0])
        innovation_variance = float(stats["S"][0])
        output.append(innovation / np.sqrt(innovation_variance))
    return np.asarray(output)


def _fast_kalman_prediction_bands(close, tuning, multiplier=2.0):
    kf = kalman.make_constant_velocity_1d(
        tuning.initial_price,
        tuning.initial_velocity,
        tuning.dt,
        tuning.position_variance,
        tuning.velocity_variance,
        tuning.process_position_variance,
        tuning.process_velocity_variance,
        tuning.measurement_variance,
    )
    middle = []
    upper = []
    lower = []
    for value in close:
        stats = kf.update([float(value)])
        center = float(value) - float(stats["innovation"][0])
        band = multiplier * np.sqrt(float(stats["S"][0]))
        middle.append(center)
        upper.append(center + band)
        lower.append(center - band)
    return np.asarray(middle), np.asarray(upper), np.asarray(lower)


def _default_seed_outputs(close):
    tuning = SimpleNamespace(
        initial_price=float(close[0]),
        initial_velocity=0.0,
        dt=1.0,
        position_variance=1.0,
        velocity_variance=1.0,
        process_position_variance=1.0e-4,
        process_velocity_variance=1.0e-3,
        measurement_variance=0.25,
    )

    kf = kalman.make_constant_velocity_1d(
        tuning.initial_price,
        tuning.initial_velocity,
        tuning.dt,
        tuning.position_variance,
        tuning.velocity_variance,
        tuning.process_position_variance,
        tuning.process_velocity_variance,
        tuning.measurement_variance,
    )
    output = [float(close[0])]
    for value in close[1:]:
        kf.update([float(value)])
        output.append(float(kf.x[0]))
    return np.asarray(output)


def test_update_matches_fast_kalman_constant_velocity_filter():
    close = np.asarray([100.0, 100.4, 101.2, 101.0, 102.5, 103.0], dtype=np.float64)
    tuning = KalmanMovingAverage.tune(close)
    indicator = KalmanMovingAverage(tuning)

    actual = np.asarray([indicator.update(value) for value in close])
    expected = _fast_kalman_outputs(close, tuning)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_moving_average_tuning_can_be_splatted_into_constructor():
    close = np.asarray([100.0, 100.4, 101.2, 101.0, 102.5, 103.0], dtype=np.float64)
    tuning = KalmanMovingAverage.tune(close)
    via_tuning = KalmanMovingAverage(tuning).batch(close)
    via_args = KalmanMovingAverage(*tuning).batch(close)
    np.testing.assert_allclose(via_args, via_tuning, rtol=1e-12, atol=1e-12)


def test_default_batch_matches_seeded_fast_kalman_path_for_realistic_sequence():
    rng = np.random.default_rng(9876)
    close = 100.0 + np.cumsum(rng.normal(0.03, 0.7, 512))

    actual = KalmanMovingAverage().batch(close)
    expected = _default_seed_outputs(close)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_tune_matches_fast_kalman_linear_tuner_for_price_model():
    close = np.asarray([50.0, 50.2, 50.6, 50.7, 51.4, 51.8, 52.0], dtype=np.float64)
    tuning = KalmanMovingAverage.tune(close)
    generic = kalman.LinearKalmanTuner(2, 1, dt=1.0, model="constant-velocity").update(close)

    assert tuning.initial_price == pytest.approx(generic.x[0])
    assert tuning.initial_velocity == pytest.approx(generic.x[1])
    assert tuning.dt == pytest.approx(1.0)
    assert tuning.position_variance == pytest.approx(generic.P[0])
    assert tuning.velocity_variance == pytest.approx(generic.P[3])
    assert tuning.process_position_variance == pytest.approx(generic.Q[0])
    assert tuning.process_velocity_variance == pytest.approx(generic.Q[3])
    assert tuning.measurement_variance == pytest.approx(generic.R[0])


def test_tune_accepts_records_and_pandas_table():
    close = np.asarray([10.0, 10.4, 10.3, 11.0, 11.6], dtype=np.float64)
    expected = KalmanMovingAverage.tune(close)

    records = [{"close": float(value)} for value in close]
    from_records = KalmanMovingAverage.tune(records)
    assert from_records.measurement_variance == pytest.approx(expected.measurement_variance)

    pandas = pytest.importorskip("pandas")
    from_table = KalmanMovingAverage.tune(pandas.DataFrame({"close": close}))
    assert from_table.process_velocity_variance == pytest.approx(expected.process_velocity_variance)


def test_advance_replay_and_last_follow_update_state():
    close = np.asarray([30.0, 30.1, 30.7, 30.6, 31.2], dtype=np.float64)
    update_indicator = KalmanMovingAverage()
    advance_indicator = KalmanMovingAverage()

    for value in close[:-1]:
        expected = update_indicator.update(value)
        assert advance_indicator.advance(value) is None
        assert advance_indicator.last() == pytest.approx(expected)

    assert advance_indicator.update(close[-1]) == pytest.approx(update_indicator.update(close[-1]))
    assert isinstance(KalmanMovingAverage().replay_update(close), float)
    assert isinstance(KalmanMovingAverage().replay_advance(close), float)


def test_tuning_type_is_exported():
    tuning = rtta.KalmanMovingAverage.tune([1.0, 1.2, 1.4])
    assert isinstance(tuning, rtta.KalmanMovingAverageTuning)
    with pytest.raises(AttributeError):
        tuning.measurement_variance = 1.0


def test_innovation_zscore_update_matches_fast_kalman_innovation_statistics():
    close = np.asarray([100.0, 100.4, 101.2, 101.0, 102.5, 103.0], dtype=np.float64)
    tuning = KalmanInnovationZScore.tune(close)
    indicator = KalmanInnovationZScore(tuning)

    actual = np.asarray([indicator.update(value) for value in close])
    expected = _fast_kalman_innovation_zscore(close, tuning)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_innovation_zscore_batch_splat_and_512_sample_paths():
    rng = np.random.default_rng(13579)
    close = 85.0 + np.cumsum(rng.normal(0.01, 0.8, 512))
    tuning = KalmanInnovationZScore.tune(close)

    via_tuning = KalmanInnovationZScore(tuning).batch(close)
    via_args = KalmanInnovationZScore(*tuning).batch(close)
    expected = _fast_kalman_innovation_zscore(close, tuning)
    np.testing.assert_allclose(via_tuning, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(via_args, expected, rtol=1e-12, atol=1e-12)

    records = [{"close": float(value)} for value in close]
    from_records = KalmanInnovationZScore(tuning).batch(records)
    np.testing.assert_allclose(from_records, expected, rtol=1e-12, atol=1e-12)

    pandas = pytest.importorskip("pandas")
    from_table = KalmanInnovationZScore(tuning).batch(pandas.DataFrame({"close": close}))
    np.testing.assert_allclose(from_table, expected, rtol=1e-12, atol=1e-12)


def test_innovation_zscore_tune_accepts_records_pandas_and_is_immutable():
    close = np.asarray([10.0, 10.4, 10.3, 11.0, 11.6], dtype=np.float64)
    expected = KalmanInnovationZScore.tune(close)

    records = [{"close": float(value)} for value in close]
    from_records = KalmanInnovationZScore.tune(records)
    assert from_records.measurement_variance == pytest.approx(expected.measurement_variance)

    pandas = pytest.importorskip("pandas")
    from_table = KalmanInnovationZScore.tune(pandas.DataFrame({"close": close}))
    assert from_table.process_velocity_variance == pytest.approx(expected.process_velocity_variance)

    assert isinstance(from_table, rtta.KalmanInnovationZScoreTuning)
    with pytest.raises(AttributeError):
        from_table.measurement_variance = 1.0


def test_innovation_zscore_advance_replay_and_last_follow_update_state():
    close = np.asarray([30.0, 30.1, 30.7, 30.6, 31.2], dtype=np.float64)
    update_indicator = KalmanInnovationZScore()
    advance_indicator = KalmanInnovationZScore()

    for value in close[:-1]:
        expected = update_indicator.update(value)
        assert advance_indicator.advance(value) is None
        assert advance_indicator.last() == pytest.approx(expected)

    assert advance_indicator.update(close[-1]) == pytest.approx(update_indicator.update(close[-1]))
    assert isinstance(KalmanInnovationZScore().replay_update(close), float)
    assert isinstance(KalmanInnovationZScore().replay_advance(close), float)


def test_prediction_bands_update_matches_fast_kalman_prediction_statistics():
    close = np.asarray([100.0, 100.4, 101.2, 101.0, 102.5, 103.0], dtype=np.float64)
    tuning = KalmanPredictionBands.tune(close)
    indicator = KalmanPredictionBands(tuning, multiplier=1.5)

    actual = [indicator.update(value) for value in close]
    expected_middle, expected_upper, expected_lower = _fast_kalman_prediction_bands(close, tuning, multiplier=1.5)
    np.testing.assert_allclose([item.middle for item in actual], expected_middle, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose([item.upper for item in actual], expected_upper, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose([item.lower for item in actual], expected_lower, rtol=1e-12, atol=1e-12)


def test_prediction_bands_batch_replay_scalar_and_splat_paths():
    rng = np.random.default_rng(97531)
    close = 95.0 + np.cumsum(rng.normal(0.04, 0.7, 512))
    tuning = KalmanPredictionBands.tune(close)

    batch = KalmanPredictionBands(tuning).batch(close)
    replay = KalmanPredictionBands(tuning).replay_update_outputs(close)
    expected_middle, expected_upper, expected_lower = _fast_kalman_prediction_bands(close, tuning)
    np.testing.assert_allclose(batch.middle, expected_middle, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batch.upper, expected_upper, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batch.lower, expected_lower, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(replay.middle, expected_middle, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(replay.upper, expected_upper, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(replay.lower, expected_lower, rtol=1e-12, atol=1e-12)

    via_args = KalmanPredictionBands(*tuning).batch(close)
    np.testing.assert_allclose(via_args.middle, expected_middle, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(via_args.upper, expected_upper, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(via_args.lower, expected_lower, rtol=1e-12, atol=1e-12)

    middle_indicator = KalmanPredictionBands(tuning)
    upper_indicator = KalmanPredictionBands(tuning)
    lower_indicator = KalmanPredictionBands(tuning)
    result = KalmanPredictionBands(tuning).update(float(close[0]))
    assert middle_indicator.update_middle(float(close[0])) == pytest.approx(result.middle)
    assert middle_indicator.last_middle() == pytest.approx(result.middle)
    assert upper_indicator.update_upper(float(close[0])) == pytest.approx(result.upper)
    assert upper_indicator.last_upper() == pytest.approx(result.upper)
    assert lower_indicator.update_lower(float(close[0])) == pytest.approx(result.lower)
    assert lower_indicator.last_lower() == pytest.approx(result.lower)

    records = [{"close": float(value)} for value in close]
    from_records = KalmanPredictionBands(tuning).batch(records)
    np.testing.assert_allclose(from_records.middle, expected_middle, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(from_records.upper, expected_upper, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(from_records.lower, expected_lower, rtol=1e-12, atol=1e-12)

    pandas = pytest.importorskip("pandas")
    from_table = KalmanPredictionBands(tuning).batch(pandas.DataFrame({"close": close}))
    np.testing.assert_allclose(from_table.middle, expected_middle, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(from_table.upper, expected_upper, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(from_table.lower, expected_lower, rtol=1e-12, atol=1e-12)


def test_prediction_bands_tune_accepts_records_pandas_and_is_immutable():
    close = np.asarray([10.0, 10.4, 10.3, 11.0, 11.6], dtype=np.float64)
    expected = KalmanPredictionBands.tune(close)

    records = [{"close": float(value)} for value in close]
    from_records = KalmanPredictionBands.tune(records)
    assert from_records.measurement_variance == pytest.approx(expected.measurement_variance)

    pandas = pytest.importorskip("pandas")
    from_table = KalmanPredictionBands.tune(pandas.DataFrame({"close": close}))
    assert from_table.process_velocity_variance == pytest.approx(expected.process_velocity_variance)

    assert isinstance(from_table, rtta.KalmanPredictionBandsTuning)
    with pytest.raises(AttributeError):
        from_table.measurement_variance = 1.0


def test_prediction_bands_advance_replay_and_last_follow_update_state():
    close = np.asarray([30.0, 30.1, 30.7, 30.6, 31.2], dtype=np.float64)
    update_indicator = KalmanPredictionBands()
    advance_indicator = KalmanPredictionBands()

    for value in close[:-1]:
        expected = update_indicator.update(value)
        assert advance_indicator.advance(value) is None
        assert advance_indicator.last().middle == pytest.approx(expected.middle)
        assert advance_indicator.last().upper == pytest.approx(expected.upper)
        assert advance_indicator.last().lower == pytest.approx(expected.lower)

    expected = update_indicator.update(close[-1])
    actual = advance_indicator.update(close[-1])
    assert actual.middle == pytest.approx(expected.middle)
    assert actual.upper == pytest.approx(expected.upper)
    assert actual.lower == pytest.approx(expected.lower)
    assert isinstance(KalmanPredictionBands().replay_update(close), float)
    assert isinstance(KalmanPredictionBands().replay_advance(close), float)


def test_velocity_oscillator_update_matches_fast_kalman_velocity_state():
    close = np.asarray([100.0, 100.4, 101.2, 101.0, 102.5, 103.0], dtype=np.float64)
    tuning = KalmanVelocityOscillator.tune(close)
    indicator = KalmanVelocityOscillator(tuning)

    actual = np.asarray([indicator.update(value) for value in close])
    expected = _fast_kalman_velocity(close, tuning)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_velocity_oscillator_batch_splat_and_512_sample_paths():
    rng = np.random.default_rng(24680)
    close = 70.0 + np.cumsum(rng.normal(0.02, 0.6, 512))
    tuning = KalmanVelocityOscillator.tune(close)

    via_tuning = KalmanVelocityOscillator(tuning).batch(close)
    via_args = KalmanVelocityOscillator(*tuning).batch(close)
    expected = _fast_kalman_velocity(close, tuning)
    np.testing.assert_allclose(via_tuning, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(via_args, expected, rtol=1e-12, atol=1e-12)

    records = [{"close": float(value)} for value in close]
    from_records = KalmanVelocityOscillator(tuning).batch(records)
    np.testing.assert_allclose(from_records, expected, rtol=1e-12, atol=1e-12)

    pandas = pytest.importorskip("pandas")
    from_table = KalmanVelocityOscillator(tuning).batch(pandas.DataFrame({"close": close}))
    np.testing.assert_allclose(from_table, expected, rtol=1e-12, atol=1e-12)


def test_velocity_oscillator_tune_accepts_records_pandas_and_is_immutable():
    close = np.asarray([10.0, 10.4, 10.3, 11.0, 11.6], dtype=np.float64)
    expected = KalmanVelocityOscillator.tune(close)

    records = [{"close": float(value)} for value in close]
    from_records = KalmanVelocityOscillator.tune(records)
    assert from_records.measurement_variance == pytest.approx(expected.measurement_variance)

    pandas = pytest.importorskip("pandas")
    from_table = KalmanVelocityOscillator.tune(pandas.DataFrame({"close": close}))
    assert from_table.process_velocity_variance == pytest.approx(expected.process_velocity_variance)

    assert isinstance(from_table, rtta.KalmanVelocityOscillatorTuning)
    with pytest.raises(AttributeError):
        from_table.measurement_variance = 1.0


def test_velocity_oscillator_advance_replay_and_last_follow_update_state():
    close = np.asarray([30.0, 30.1, 30.7, 30.6, 31.2], dtype=np.float64)
    update_indicator = KalmanVelocityOscillator()
    advance_indicator = KalmanVelocityOscillator()

    for value in close[:-1]:
        expected = update_indicator.update(value)
        assert advance_indicator.advance(value) is None
        assert advance_indicator.last() == pytest.approx(expected)

    assert advance_indicator.update(close[-1]) == pytest.approx(update_indicator.update(close[-1]))
    assert isinstance(KalmanVelocityOscillator().replay_update(close), float)
    assert isinstance(KalmanVelocityOscillator().replay_advance(close), float)


def test_local_linear_trend_update_matches_fast_kalman_constant_velocity_filter():
    close = np.asarray([100.0, 100.4, 101.2, 101.0, 102.5, 103.0], dtype=np.float64)
    tuning = KalmanLocalLinearTrend.tune(close)
    indicator = KalmanLocalLinearTrend(tuning)

    actual = [indicator.update(value) for value in close]
    expected_level, expected_trend = _fast_kalman_level_trend(close, tuning)
    np.testing.assert_allclose([item.level for item in actual], expected_level, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose([item.trend for item in actual], expected_trend, rtol=1e-12, atol=1e-12)


def test_local_linear_trend_batch_replay_scalar_and_splat_paths():
    rng = np.random.default_rng(12345)
    close = 50.0 + np.cumsum(rng.normal(0.05, 0.5, 512))
    tuning = KalmanLocalLinearTrend.tune(close)

    batch = KalmanLocalLinearTrend(tuning).batch(close)
    replay = KalmanLocalLinearTrend(tuning).replay_update_outputs(close)
    np.testing.assert_allclose(batch.level, replay.level, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batch.trend, replay.trend, rtol=1e-12, atol=1e-12)

    via_args = KalmanLocalLinearTrend(*tuning).batch(close)
    np.testing.assert_allclose(batch.level, via_args.level, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batch.trend, via_args.trend, rtol=1e-12, atol=1e-12)

    level_indicator = KalmanLocalLinearTrend(tuning)
    trend_indicator = KalmanLocalLinearTrend(tuning)
    result = KalmanLocalLinearTrend(tuning).update(float(close[0]))
    assert level_indicator.update_level(float(close[0])) == pytest.approx(result.level)
    assert level_indicator.last_level() == pytest.approx(result.level)
    assert trend_indicator.update_trend(float(close[0])) == pytest.approx(result.trend)
    assert trend_indicator.last_trend() == pytest.approx(result.trend)


def test_local_linear_trend_tune_accepts_records_pandas_and_is_immutable():
    close = np.asarray([10.0, 10.4, 10.3, 11.0, 11.6], dtype=np.float64)
    expected = KalmanLocalLinearTrend.tune(close)

    records = [{"close": float(value)} for value in close]
    from_records = KalmanLocalLinearTrend.tune(records)
    assert from_records.observation_variance == pytest.approx(expected.observation_variance)

    pandas = pytest.importorskip("pandas")
    from_table = KalmanLocalLinearTrend.tune(pandas.DataFrame({"close": close}))
    assert from_table.process_trend_variance == pytest.approx(expected.process_trend_variance)

    assert isinstance(from_table, rtta.KalmanLocalLinearTrendTuning)
    with pytest.raises(AttributeError):
        from_table.observation_variance = 1.0
