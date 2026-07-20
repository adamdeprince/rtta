import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.benchmark_indicators import INDICATORS, generate_market_data, make_rtta_batch_runner, make_rtta_incremental_runner


DATA = generate_market_data(512, 42)


def _series():
    pandas = pytest.importorskip("pandas")
    return {
        name: pandas.Series(values, copy=False)
        for name, values in DATA.arrays.items()
        if values.ndim == 1
    }


def _incremental(indicator_cls, input_names, kwargs=None, attr=None):
    indicator = indicator_cls(**(kwargs or {}))
    inputs = [DATA.lists[name] for name in input_names]
    output = []
    for values in zip(*inputs):
        result = indicator.update(*values)
        output.append(getattr(result, attr) if attr else result)
    return np.asarray(output, dtype=np.float64)


def _assert_tail_close(actual, expected, start=100, rtol=1e-6, atol=1e-6):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    mask = np.isfinite(actual[start:]) & np.isfinite(expected[start:])
    assert mask.any()
    np.testing.assert_allclose(actual[start:][mask], expected[start:][mask], rtol=rtol, atol=atol)


@pytest.mark.parametrize("spec", INDICATORS, ids=lambda spec: spec.name)
def test_every_indicator_accepts_realistic_512_sample_incremental_sequence(spec):
    import rtta

    make_rtta_incremental_runner(rtta, spec, DATA)()


@pytest.mark.parametrize("spec", INDICATORS, ids=lambda spec: spec.name)
def test_every_batch_indicator_accepts_realistic_512_sample_sequence(spec):
    import rtta

    batch_runner = make_rtta_batch_runner(rtta, spec, DATA)
    assert batch_runner is not None
    runner, _ = batch_runner
    assert runner() is not None


def test_talib_equivalent_indicators_match_on_realistic_512_sample_sequence():
    talib = pytest.importorskip("talib")
    import rtta

    close = DATA.arrays["close"]
    high = DATA.arrays["high"]
    low = DATA.arrays["low"]
    open_ = DATA.arrays["open"]
    volume = DATA.arrays["volume"]
    real1 = DATA.arrays["real1"]

    cases = [
        ("SMA", _incremental(rtta.SMA, ("close",), {"window": 30, "fillna": False}), talib.SMA(close, timeperiod=30), 30),
        ("EMA", _incremental(rtta.EMA, ("close",), {"window": 30, "fillna": False}), talib.EMA(close, timeperiod=30), 150),
        ("MACD", _incremental(rtta.MACD, ("close",), {"fillna": False}, attr="signal"), talib.MACD(close)[1], 150),
        ("MACD_line", _incremental(rtta.MACD, ("close",), {"fillna": False}, attr="macd"), talib.MACD(close)[0], 150),
        ("ROC", _incremental(rtta.ROC, ("close",), {"window": 10, "fillna": False}), talib.ROC(close, timeperiod=10), 20),
        ("ATR", _incremental(rtta.ATR, ("close", "high", "low")), talib.ATR(high, low, close, timeperiod=14), 200),
        ("ATRP", _incremental(rtta.ATRP, ("close", "high", "low")), talib.NATR(high, low, close, timeperiod=14) / 100.0, 200),
        ("NormalizedATR", _incremental(rtta.NormalizedATR, ("close", "high", "low")), talib.NATR(high, low, close, timeperiod=14), 200),
        ("OBV", _incremental(rtta.OnBalanceVolume, ("close", "volume")), talib.OBV(close, volume), 1),
        ("TrueRange", _incremental(rtta.TrueRange, ("close", "high", "low")), talib.TRANGE(high, low, close), 1),
        ("TypicalPrice", _incremental(rtta.TypicalPrice, ("close", "high", "low")), talib.TYPPRICE(high, low, close), 0),
        ("WeightedClosePrice", _incremental(rtta.WeightedClosePrice, ("close", "high", "low")), talib.WCLPRICE(high, low, close), 0),
        ("AveragePrice", _incremental(rtta.AveragePrice, ("open", "high", "low", "close")), talib.AVGPRICE(open_, high, low, close), 0),
        ("MedianPrice", _incremental(rtta.MedianPrice, ("high", "low")), talib.MEDPRICE(high, low), 0),
        ("BalanceOfPower", _incremental(rtta.BalanceOfPower, ("open", "high", "low", "close")), talib.BOP(open_, high, low, close), 0),
        ("Momentum", _incremental(rtta.Momentum, ("close",)), talib.MOM(close, timeperiod=10), 20),
        ("RateOfChangePercentage", _incremental(rtta.RateOfChangePercentage, ("close",)), talib.ROCP(close, timeperiod=10), 20),
        ("RateOfChangeRatio", _incremental(rtta.RateOfChangeRatio, ("close",)), talib.ROCR(close, timeperiod=10), 20),
        ("RateOfChangeRatio100", _incremental(rtta.RateOfChangeRatio100, ("close",)), talib.ROCR100(close, timeperiod=10), 20),
        ("Low", _incremental(rtta.Low, ("close",), {"window": 30}), talib.MIN(close, timeperiod=30), 30),
        ("High", _incremental(rtta.High, ("close",), {"window": 30}), talib.MAX(close, timeperiod=30), 30),
        ("LowIndex", _incremental(rtta.LowIndex, ("close",), {"window": 30}), talib.MININDEX(close, timeperiod=30), 30),
        ("HighIndex", _incremental(rtta.HighIndex, ("close",), {"window": 30}), talib.MAXINDEX(close, timeperiod=30), 30),
        ("Summation", _incremental(rtta.Summation, ("close",), {"window": 30}), talib.SUM(close, timeperiod=30), 30),
        ("StdDev", _incremental(rtta.StdDev, ("close",), {"window": 5}), talib.STDDEV(close, timeperiod=5, nbdev=1), 5),
        ("Variance", _incremental(rtta.Variance, ("close",), {"window": 5}), talib.VAR(close, timeperiod=5, nbdev=1), 5),
        ("WilliamsR", _incremental(rtta.WilliamsR, ("close", "high", "low")), talib.WILLR(high, low, close, timeperiod=14), 20),
        ("ADX", _incremental(rtta.AverageDirectionalMovementIndex, ("close", "high", "low")), talib.ADX(high, low, close, timeperiod=14), 200),
        ("DMI", _incremental(rtta.DirectionalMovementIndex, ("close", "high", "low")), talib.DX(high, low, close, timeperiod=14), 200),
        ("PlusDI", _incremental(rtta.PlusDirectionalIndicator, ("close", "high", "low")), talib.PLUS_DI(high, low, close, timeperiod=14), 200),
        ("MinusDI", _incremental(rtta.MinusDirectionalIndicator, ("close", "high", "low")), talib.MINUS_DI(high, low, close, timeperiod=14), 200),
        ("MFI", _incremental(rtta.MoneyFlowIndex, ("close", "high", "low", "volume")), talib.MFI(high, low, close, volume, timeperiod=14), 20),
        ("CCI", _incremental(rtta.CommodityChannelIndex, ("close", "high", "low")), talib.CCI(high, low, close, timeperiod=14), 20),
        ("Correlation", _incremental(rtta.Correlation, ("close", "real1")), talib.CORREL(close, real1, timeperiod=30), 40),
        ("LinearRegression", _incremental(rtta.LinearRegression, ("close",)), talib.LINEARREG(close, timeperiod=14), 20),
        ("LinearRegressionSlope", _incremental(rtta.LinearRegressionSlope, ("close",)), talib.LINEARREG_SLOPE(close, timeperiod=14), 20),
        ("LinearRegressionIntercept", _incremental(rtta.LinearRegressionIntercept, ("close",)), talib.LINEARREG_INTERCEPT(close, timeperiod=14), 20),
        ("LinearRegressionAngle", _incremental(rtta.LinearRegressionAngle, ("close",)), talib.LINEARREG_ANGLE(close, timeperiod=14), 20),
        ("TimeSeriesForecast", _incremental(rtta.TimeSeriesForecast, ("close",)), talib.TSF(close, timeperiod=14), 20),
        ("MidPoint", _incremental(rtta.MidPoint, ("close",)), talib.MIDPOINT(close, timeperiod=14), 20),
        ("MidPrice", _incremental(rtta.MidPrice, ("high", "low")), talib.MIDPRICE(high, low, timeperiod=14), 20),
        ("AccumulationDistribution", _incremental(rtta.AccumulationDistribution, ("close", "high", "low", "volume")), talib.AD(high, low, close, volume), 0),
        ("ChaikinOscillator", _incremental(rtta.ChaikinOscillator, ("close", "high", "low", "volume")), talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10), 50),
        ("FastStochastic.fastk", _incremental(rtta.FastStochastic, ("close", "high", "low"), attr="fastk"), talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3)[0], 20),
        ("FastStochastic.fastd", _incremental(rtta.FastStochastic, ("close", "high", "low"), attr="fastd"), talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3)[1], 20),
        ("Stochastic.slowk", _incremental(rtta.Stochastic, ("close", "high", "low"), attr="slowk"), talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)[0], 20),
        ("Stochastic.slowd", _incremental(rtta.Stochastic, ("close", "high", "low"), attr="slowd"), talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)[1], 20),
        ("WeightedMovingAverage", _incremental(rtta.WeightedMovingAverage, ("close",), {"window": 30}), talib.WMA(close, timeperiod=30), 30),
        ("TriangularMovingAverage", _incremental(rtta.TriangularMovingAverage, ("close",), {"window": 30}), talib.TRIMA(close, timeperiod=30), 50),
        ("DoubleEMA", _incremental(rtta.DoubleEMA, ("close",)), talib.DEMA(close, timeperiod=30), 200),
        ("TripleEMA", _incremental(rtta.TripleEMA, ("close",)), talib.TEMA(close, timeperiod=30), 300),
        ("T3MovingAverage", _incremental(rtta.T3MovingAverage, ("close",)), talib.T3(close, timeperiod=5, vfactor=0.7), 100),
    ]

    for name, actual, expected, start in cases:
        try:
            _assert_tail_close(actual, expected, start=start, rtol=1e-4, atol=1e-4)
        except AssertionError as exc:
            raise AssertionError(name) from exc


def test_ta_equivalent_indicators_match_on_realistic_512_sample_sequence():
    pytest.importorskip("ta")
    import rtta
    from ta import momentum, others, trend, volatility, volume

    series = _series()
    close = DATA.arrays["close"]
    high = DATA.arrays["high"]
    low = DATA.arrays["low"]
    kst = trend.KSTIndicator(series["close"], fillna=True)
    vortex = trend.VortexIndicator(series["high"], series["low"], series["close"], fillna=True)
    ichimoku = trend.IchimokuIndicator(series["high"], series["low"], fillna=True)

    cases = [
        ("AbsolutePriceOscillator", _incremental(rtta.AbsolutePriceOscillator, ("close",)), trend.macd(series["close"], fillna=True), 100, 1e-12, 1e-12),
        ("AwesomeOscillator", _incremental(rtta.AwesomeOscillator, ("high", "low")), momentum.awesome_oscillator(series["high"], series["low"], window1=5, window2=34, fillna=True), 100, 1e-12, 1e-12),
        ("Kama", _incremental(rtta.Kama, ("close",)), momentum.kama(series["close"], fillna=True), 100, 3e-3, 3e-3),
        ("TSI", _incremental(rtta.TSI, ("close",)), momentum.tsi(series["close"], fillna=True), 300, 1e-5, 1e-5),
        ("UltimateOscillator", _incremental(rtta.UltimateOscillator, ("close", "high", "low")), momentum.ultimate_oscillator(series["high"], series["low"], series["close"], fillna=True), 100, 1e-12, 1e-12),
        ("MassIndex", _incremental(rtta.MassIndex, ("high", "low")), trend.mass_index(series["high"], series["low"], fillna=True), 100, 1e-12, 1e-12),
        ("DetrendedPriceOscillator", _incremental(rtta.DetrendedPriceOscillator, ("close",)), trend.dpo(series["close"], fillna=True), 100, 1e-12, 1e-12),
        ("ChaikinMoneyFlow", _incremental(rtta.ChaikinMoneyFlow, ("close", "high", "low", "volume")), volume.chaikin_money_flow(series["high"], series["low"], series["close"], series["volume"], fillna=True), 100, 1e-12, 1e-12),
        ("ForceIndex", _incremental(rtta.ForceIndex, ("close", "volume")), volume.force_index(series["close"], series["volume"], fillna=True), 200, 1e-2, 1e-2),
        ("EaseOfMovement", _incremental(rtta.EaseOfMovement, ("high", "low", "volume"), attr="ease_of_movement"), volume.ease_of_movement(series["high"], series["low"], series["volume"], fillna=True), 100, 1e-12, 1e-12),
        ("NegativeVolumeIndex", _incremental(rtta.NegativeVolumeIndex, ("close", "volume")), volume.negative_volume_index(series["close"], series["volume"], fillna=True), 100, 1e-12, 1e-12),
        ("VolumePriceTrend", _incremental(rtta.VolumePriceTrend, ("close", "volume")), volume.volume_price_trend(series["close"], series["volume"], fillna=True), 100, 1e-8, 1e-8),
        ("VolumeWeightedAveragePrice", _incremental(rtta.VolumeWeightedAveragePrice, ("close", "high", "low", "volume")), volume.volume_weighted_average_price(series["high"], series["low"], series["close"], series["volume"], fillna=True), 100, 1e-12, 1e-12),
        ("Donchian.upper", _incremental(rtta.DonchianChannel, ("close", "high", "low"), attr="upper"), volatility.donchian_channel_hband(series["high"], series["low"], series["close"], fillna=True), 100, 1e-12, 1e-12),
        ("UlcerIndex", _incremental(rtta.UlcerIndex, ("close",)), volatility.ulcer_index(series["close"], fillna=True), 100, 1e-12, 1e-12),
        ("DailyReturn", _incremental(rtta.DailyReturn, ("close",)), others.daily_return(series["close"], fillna=True), 1, 1e-12, 1e-12),
        ("DailyLogReturn", _incremental(rtta.DailyLogReturn, ("close",)), others.daily_log_return(series["close"], fillna=True), 1, 1e-12, 1e-12),
        ("CumulativeReturn", _incremental(rtta.CumulativeReturn, ("close",)), others.cumulative_return(series["close"], fillna=True), 1, 1e-12, 1e-12),
        ("KST.kst", _incremental(rtta.KSTOscillator, ("close",), attr="kst"), kst.kst(), 100, 1e-12, 1e-12),
        ("KST.signal", _incremental(rtta.KSTOscillator, ("close",), attr="signal"), kst.kst_sig(), 100, 1e-12, 1e-12),
        ("KST.difference", _incremental(rtta.KSTOscillator, ("close",), attr="difference"), kst.kst_diff(), 100, 1e-12, 1e-12),
        ("Vortex.positive", _incremental(rtta.Vortex, ("close", "high", "low"), attr="positive"), vortex.vortex_indicator_pos(), 100, 1e-12, 1e-12),
        ("Vortex.negative", _incremental(rtta.Vortex, ("close", "high", "low"), attr="negative"), vortex.vortex_indicator_neg(), 100, 1e-12, 1e-12),
        ("Vortex.difference", _incremental(rtta.Vortex, ("close", "high", "low"), attr="difference"), vortex.vortex_indicator_diff(), 100, 1e-12, 1e-12),
        ("Ichimoku.conversion", _incremental(rtta.Ichimoku, ("high", "low", "close"), attr="conversion"), ichimoku.ichimoku_conversion_line(), 100, 1e-12, 1e-12),
        ("Ichimoku.base", _incremental(rtta.Ichimoku, ("high", "low", "close"), attr="base"), ichimoku.ichimoku_base_line(), 100, 1e-12, 1e-12),
        ("Ichimoku.span_a", _incremental(rtta.Ichimoku, ("high", "low", "close"), attr="span_a"), ichimoku.ichimoku_a(), 100, 1e-12, 1e-12),
        ("Ichimoku.span_b", _incremental(rtta.Ichimoku, ("high", "low", "close"), attr="span_b"), ichimoku.ichimoku_b(), 100, 1e-12, 1e-12),
        # STC accumulates floating-point drift vs ta; allow ~1e-10 abs on the warm tail.
        ("SchaffTrendCycle", _incremental(rtta.SchaffTrendCycle, ("close",)), trend.STCIndicator(series["close"], fillna=True).stc(), 200, 1e-10, 1e-10),
        ("PercentagePrice", _incremental(rtta.PercentagePrice, ("close",), {"fillna": True}, "ppo"), momentum.ppo(series["close"], fillna=True), 100, 1e-12, 1e-12),
        ("PercentageVolume", _incremental(rtta.PercentageVolume, ("volume",), attr="pvo"), momentum.pvo(series["volume"], fillna=True), 100, 1e-12, 1e-12),
    ]

    for name, actual, expected, start, rtol, atol in cases:
        try:
            _assert_tail_close(actual, expected, start=start, rtol=rtol, atol=atol)
        except AssertionError as exc:
            raise AssertionError(name) from exc
