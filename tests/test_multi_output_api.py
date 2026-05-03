import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rtta
from benchmarks.benchmark_indicators import INDICATORS, generate_market_data


MULTI_OUTPUTS = {
    "Aroon": ("down", "up"),
    "AlphaBetaGammaTrackingFilter": ("price", "velocity", "acceleration", "residual"),
    "BollingerBands": ("middle", "upper", "lower"),
    "DonchianChannel": ("upper", "lower", "middle", "width", "percent"),
    "EaseOfMovement": ("ease_of_movement", "sma"),
    "ElderRayIndex": ("bull_power", "bear_power"),
    "FastStochastic": ("fastk", "fastd"),
    "FibonacciRetracementLevels": ("level0", "level236", "level382", "level500", "level618", "level100"),
    "GaussianProcessRegressionBands": ("middle", "upper", "lower"),
    "HeikinAshiTransform": ("open", "high", "low", "close"),
    "HighLow": ("min", "max"),
    "HighLowIndex": ("min_index", "max_index"),
    "Ichimoku": ("conversion", "base", "span_a", "span_b"),
    "InteractingMultipleModelFilter": ("value", "velocity", "low_vol_probability", "high_vol_probability", "trend_probability", "chop_probability"),
    "KSTOscillator": ("kst", "signal", "difference"),
    "KalmanExtremumTrend": ("trend", "oscillator", "signal"),
    "KalmanHedgeRatio": ("hedge_ratio", "intercept", "spread"),
    "KalmanLocalLinearTrend": ("level", "trend"),
    "KalmanPredictionBands": ("middle", "upper", "lower"),
    "KalmanRegressionChannel": ("slope", "intercept", "middle", "upper", "lower", "spread"),
    "KalmanTrendSignal": ("trend", "signal"),
    "KeltnerChannel": ("middle", "upper", "lower"),
    "KeltnerChannelOriginal": ("middle", "upper", "lower"),
    "PercentagePrice": ("ppo", "signal", "histogram"),
    "PercentageVolume": ("pvo", "signal", "histogram"),
    "RelativeVigorIndex": ("rvi", "signal"),
    "RenkoBrickGenerator": ("brick_open", "brick_close", "direction", "bricks", "reversal"),
    "SavitzkyGolayFilter": ("smooth", "first_derivative", "second_derivative"),
    "KlingerVolumeOscillator": ("kvo", "signal", "histogram"),
    "MesaAdaptiveMovingAverage": ("mama", "fama"),
    "NadarayaWatsonEnvelope": ("middle", "upper", "lower"),
    "ParticleFilterTrend": ("trend", "velocity", "signal", "effective_sample_size"),
    "Stochastic": ("slowk", "slowd"),
    "SuperTrend": ("value", "direction", "upper", "lower"),
    "TwoFactorKalmanTrendFilter": ("short_trend", "long_trend", "value"),
    "VolumeProfile": ("point_of_control", "value_area_high", "value_area_low"),
    "Vortex": ("positive", "negative", "difference"),
    "ZigZagSwingDetector": ("value", "direction", "pivot", "pivot_index"),
}


SPECS = {spec.name: spec for spec in INDICATORS}


def _same_scalar(left, right):
    if math.isnan(left) or math.isnan(right):
        assert math.isnan(left) and math.isnan(right)
    else:
        assert left == pytest.approx(right, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("name,fields", sorted(MULTI_OUTPUTS.items()))
def test_multi_output_scalar_update_and_last_accessors(name, fields):
    spec = SPECS[name]
    data = generate_market_data(128, 20240501)
    indicator_cls = getattr(rtta, name)
    result_indicator = indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)
    scalar_indicators = {
        field: indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)
        for field in fields
    }
    inputs = [data.lists[column] for column in spec.update_inputs]

    for index in range(96):
        args = [values[index] for values in inputs]
        result = result_indicator.update(*args)
        for field in fields:
            update_field = getattr(scalar_indicators[field], f"update_{field}")
            last_field = getattr(scalar_indicators[field], f"last_{field}")
            scalar = update_field(*args)
            _same_scalar(scalar, float(getattr(result, field)))
            _same_scalar(last_field(), scalar)


@pytest.mark.parametrize("name,fields", sorted(MULTI_OUTPUTS.items()))
def test_multi_output_replay_update_outputs_match_batch(name, fields):
    spec = SPECS[name]
    data = generate_market_data(512, 20240502)
    indicator_cls = getattr(rtta, name)
    batch_indicator = indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)
    replay_indicator = indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)
    arrays = [data.arrays[column] for column in spec.update_inputs]

    batch = batch_indicator.batch(*arrays)
    replay = replay_indicator.replay_update_outputs(*arrays)

    for field in fields:
        assert hasattr(replay_indicator, f"update_{field}")
        assert hasattr(replay_indicator, f"last_{field}")
        assert np.allclose(
            getattr(replay, field),
            getattr(batch, field),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
