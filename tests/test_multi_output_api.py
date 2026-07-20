import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rtta
from benchmarks.benchmark_indicators import INDICATORS, generate_market_data


MULTI_OUTPUTS = {
    "FlowPressureCapacitySignal": (
        "signal", "score", "fair_value", "microprice", "raw_queue_imbalance",
        "queue_imbalance", "flow_imbalance", "pressure", "replenishment",
        "fragility", "spread_bps",
    ),
    "SqrtImpactFlowSignal": (
        "signal", "score", "impact", "residual", "continuation", "reversion",
        "participation", "flow", "volatility", "vwap_gap",
    ),
    "CDLPatternPack": (
        "doji", "hammer", "hanging_man", "inverted_hammer", "shooting_star",
        "engulfing", "harami", "piercing", "dark_cloud_cover",
        "morning_star", "evening_star", "three_white_soldiers", "three_black_crows",
        "marubozu", "spinning_top",
    ),
    "GuppyMMARibbon": ("s3", "s5", "s8", "s10", "s12", "s15", "l30", "l35", "l40", "l45", "l50", "l60", "short_average", "long_average", "spread"),
    "FourierResidueIdentity": (
        "rho", "rho_sign", "rho_magnitude", "z_rho", "z_sign",
        "directional_share", "elliptical_ratio", "variance_ratio",
        "variance_ratio_sign", "variance_ratio_magnitude", "z_variance_ratio",
        "persistence", "signal", "score", "magnitude_forecast",
    ),
    "FibonacciPivotPoints": ("pp", "r1", "r2", "r3", "s1", "s2", "s3"),
    "ElderThermometer": ("ratio", "hot", "range"),
    "RainbowMovingAverage": ("outer", "highest", "lowest", "mid", "width"),
    "RainbowOscillator": ("value", "position", "width"),
    "ProjectionOscillator": ("value", "signal", "upper", "lower"),
    "MessageEventOrderFlowImbalance": ("ofi", "event", "signed_size"),
    "HawkesIntensity": ("intensity", "excitation", "baseline"),
    "ConformalBands": ("middle", "upper", "lower", "radius"),
    "AndrewsPitchfork": ("median", "upper", "lower", "pivot", "direction"),
    "VolumeRunBarGenerator": ("bar_open", "bar_close", "bar_high", "bar_low", "bar_volume", "direction", "complete", "bars"),
    "KalmanInnovationResidualFOCuS": ("signal", "score", "residual"),
    "KalmanInnovationResidualBOCPD": ("signal", "score", "residual"),
    "DollarRunBarGenerator": ("bar_open", "bar_close", "bar_high", "bar_low", "bar_volume", "direction", "complete", "bars"),
    "WoodiePivotPoints": ("pp", "r1", "r2", "r3", "s1", "s2", "s3"),
    "RunBarGenerator": ("bar_open", "bar_close", "bar_high", "bar_low", "bar_volume", "direction", "complete", "bars"),
    "ResidualBOCPD": ("signal", "probability"),
    "CrossAssetOrderFlowImbalance": ("beta", "impact", "residual", "peer_ofi"),
    "CamarillaPivotPoints": ("pp", "r1", "r2", "r3", "s1", "s2", "s3"),
    "IntegratedOrderFlowImbalance": ("ofi", "weight_l1"),
    "MultiLevelOrderFlowImbalance": ("total", "mean", "l1", "l2", "l3", "l4", "l5"),
    "VolumeBarGenerator": ("bar_open", "bar_close", "bar_high", "bar_low", "bar_volume", "direction", "complete", "bars"),
    "ResidualFOCuS": ("signal", "statistic"),
    "ImbalanceBarGenerator": ("bar_open", "bar_close", "bar_high", "bar_low", "bar_volume", "direction", "complete", "bars"),
    "FOCuS": ("signal", "statistic"),
    "DollarBarGenerator": ("bar_open", "bar_close", "bar_high", "bar_low", "bar_volume", "direction", "complete", "bars"),
    "DirectionalChangeDetector": ("event", "overshoot", "extremum", "direction"),
    "DecomposedOrderFlowImbalance": ("add", "cancel", "trade", "total"),
    "PointAndFigure": ("box_price", "direction", "boxes", "reversal"),
    "KagiChart": ("line", "direction", "reversal"),
    "GuppyMultipleMovingAverage": ("short_average", "long_average", "spread"),
    "EhlersRoofingFilter": ("roof", "highpass"),
    "EhlersInstantaneousTrendline": ("trendline", "trigger"),
    "EhlersDecycler": ("decycle", "oscillator"),
    "EhlersCyberCycle": ("cycle", "trigger"),
    "EhlersCenterOfGravity": ("cg", "lag"),
    "Aroon": ("down", "up"),
    "AlphaBetaGammaTrackingFilter": ("price", "velocity", "acceleration", "residual"),
    "BollingerBands": ("middle", "upper", "lower"),
    "AccelerationBands": ("middle", "upper", "lower"),
    "Alligator": ("jaw", "teeth", "lips"),
    "ChandelierExit": ("long_exit", "short_exit"),
    "GatorOscillator": ("upper", "lower"),
    "HilbertPhasor": ("inphase", "quadrature"),
    "HilbertSineWave": ("sine", "lead_sine"),
    "SqueezeMomentum": ("on", "momentum"),
    "StochasticMomentumIndex": ("smi", "signal"),
    "WaveTrend": ("wt1", "wt2"),
    "MovingAverageEnvelope": ("middle", "upper", "lower"),
    "MACD": ("macd", "signal", "histogram"),
    "MACDExt": ("macd", "signal", "histogram"),
    "MACDFix": ("macd", "signal", "histogram"),
    "PivotPoints": ("pp", "r1", "r2", "r3", "s1", "s2", "s3"),
    "RandomWalkIndex": ("high", "low"),
    "RelativeVolatilityIndex": ("rvi", "signal"),
    "WilliamsFractals": ("up", "down"),
    "DonchianChannel": ("upper", "lower", "middle", "width", "percent"),
    "EaseOfMovement": ("ease_of_movement", "sma"),
    "ElderRayIndex": ("bull_power", "bear_power"),
    "FastStochastic": ("fastk", "fastd"),
    "FibonacciRetracementLevels": ("level0", "level236", "level382", "level500", "level618", "level100"),
    "GaussianProcessRegressionBands": ("middle", "upper", "lower"),
    "HeikinAshiTransform": ("open", "high", "low", "close"),
    "HighLow": ("min", "max"),
    "HighLowIndex": ("min_index", "max_index"),
    "Ichimoku": ("conversion", "base", "span_a", "span_b", "lagging_span", "span_a_displaced", "span_b_displaced"),
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
    "ClosePressureReversalSignal": (
        "bar_number",
        "rod_return",
        "frozen_rod_return",
        "loser_z",
        "winner_z",
        "range_z",
        "volume_shock",
        "transaction_shock",
        "vwap_gap",
        "pressure_score",
        "prediction",
        "radius",
        "score",
        "signal",
        "target_fraction",
        "max_trade_dollars",
        "realized_error",
        "entry_window",
        "exit_window",
        "frozen",
        "news_guard",
    ),
    "PercentagePrice": ("ppo", "signal", "histogram"),
    "PercentageVolume": ("pvo", "signal", "histogram"),
    "RelativeVigorIndex": ("rvi", "signal"),
    "RenkoBrickGenerator": ("brick_open", "brick_close", "direction", "bricks", "reversal"),
    "SavitzkyGolayFilter": ("smooth", "first_derivative", "second_derivative"),
    "KlingerVolumeOscillator": ("kvo", "signal", "histogram"),
    "MatchedFlowConformalSignal": (
        "prediction",
        "radius",
        "score",
        "signal",
        "target_fraction",
        "alpha_flow",
        "participation",
        "flow_score",
        "momentum",
        "volatility",
        "vwap_gap",
        "rel_dollar_volume",
        "max_trade_dollars",
        "realized_error",
    ),
    "MesaAdaptiveMovingAverage": ("mama", "fama"),
    "NadarayaWatsonEnvelope": ("middle", "upper", "lower"),
    "ParticleFilterTrend": ("trend", "velocity", "signal", "effective_sample_size"),
    "SpreadFeatures": ("quoted_spread", "effective_spread", "realized_spread"),
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
    # Depth-book indicators take (levels,) arrays per tick and do not expose update_<field>.
    if getattr(spec, "depth_book", False):
        return
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
        if not getattr(spec, "depth_book", False):
            assert hasattr(replay_indicator, f"update_{field}")
            assert hasattr(replay_indicator, f"last_{field}")
        assert np.allclose(
            getattr(replay, field),
            getattr(batch, field),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
