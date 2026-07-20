import math
from unittest import TestCase, main

import numpy as np

import rtta


def _smma_reference(values, window):
    out = []
    total = 0.0
    smma = 0.0
    for i, value in enumerate(values, start=1):
        if i < window:
            total += value
            out.append(total / i)
        elif i == window:
            total += value
            smma = total / window
            out.append(smma)
        else:
            smma = (smma * (window - 1) + value) / window
            out.append(smma)
    return out


def _zlema_reference(values, window):
    lag = max((window - 1) // 2, 0)
    alpha = 2.0 / (window + 1.0)
    ema = 0.0
    first = True
    capacity = max(lag, 1)
    history = [0.0] * capacity
    history_index = 0
    history_count = 0
    out = []
    for value in values:
        delayed = value
        if lag > 0:
            if history_count < capacity:
                history[history_count] = value
                history_count += 1
                delayed = history[0]
            else:
                delayed = history[history_index]
                history[history_index] = value
                history_index = (history_index + 1) % capacity
        de_lagged = 2.0 * value - delayed
        if first:
            ema = de_lagged
            first = False
        ema = alpha * de_lagged + (1.0 - alpha) * ema
        out.append(ema)
    return out


def _alma_reference(values, window=9, offset=0.85, sigma=6.0):
    m = offset * (window - 1)
    s = window / sigma
    weights = np.exp(-0.5 * ((np.arange(window) - m) / s) ** 2)
    weight_sum = weights.sum()
    out = []
    for i in range(len(values)):
        start = max(0, i + 1 - window)
        chunk = values[start : i + 1]
        w = weights[window - len(chunk) :]
        out.append(float(np.dot(chunk, w) / w.sum()))
    return out


def _mcginley_reference(values, window):
    out = []
    md = None
    for value in values:
        if md is None:
            md = value
        else:
            ratio = value / md if md != 0.0 else 1.0
            denom = window * (ratio ** 4)
            md = md + (value - md) / denom if denom != 0.0 else value
        out.append(md)
    return out


class ClassicalCompletenessTest(TestCase):
    def test_smoothed_moving_average(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        expected = _smma_reference(values, window=3)
        indicator = rtta.SmoothedMovingAverage(window=3, fillna=True)
        got = [indicator.update(v) for v in values]
        np.testing.assert_allclose(got, expected)
        np.testing.assert_allclose(rtta.SmoothedMovingAverage(window=3).batch(np.array(values)), expected)

    def test_smoothed_moving_average_fillna_false(self):
        indicator = rtta.SmoothedMovingAverage(window=3, fillna=False)
        self.assertTrue(math.isnan(indicator.update(1.0)))
        self.assertTrue(math.isnan(indicator.update(2.0)))
        self.assertAlmostEqual(indicator.update(3.0), 2.0)
        self.assertAlmostEqual(indicator.update(6.0), (2.0 * 2 + 6.0) / 3.0)

    def test_zero_lag_ema_matches_reference(self):
        values = np.array([10.0, 11.0, 12.5, 12.0, 13.0, 14.5, 14.0, 15.0])
        expected = _zlema_reference(values.tolist(), window=4)
        indicator = rtta.ZeroLagEMA(window=4, fillna=True)
        got = [indicator.update(float(v)) for v in values]
        batch = rtta.ZeroLagEMA(window=4, fillna=True).batch(values)
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(batch, expected, rtol=1e-9, atol=1e-9)

    def test_alma_matches_reference(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        expected = _alma_reference(values, window=5, offset=0.85, sigma=6.0)
        indicator = rtta.ArnaudLegouxMovingAverage(window=5, offset=0.85, sigma=6.0, fillna=True)
        got = [indicator.update(float(v)) for v in values]
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            rtta.ArnaudLegouxMovingAverage(window=5, offset=0.85, sigma=6.0).batch(values),
            expected,
            rtol=1e-9,
            atol=1e-9,
        )

    def test_mcginley_dynamic(self):
        values = [100.0, 101.0, 102.5, 101.5, 103.0]
        expected = _mcginley_reference(values, window=4)
        indicator = rtta.McGinleyDynamic(window=4, fillna=True)
        got = [indicator.update(v) for v in values]
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)

    def test_bollinger_percent_b_and_bandwidth(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        window = 4
        bb = rtta.BollingerBands(window=window, fillna=True)
        pb = rtta.BollingerPercentB(window=window, num_std=2.0, fillna=True)
        bw = rtta.BollingerBandwidth(window=window, num_std=2.0, fillna=True)
        for value in values:
            bands = bb.update(float(value))
            percent_b = pb.update(float(value))
            bandwidth = bw.update(float(value))
            if math.isnan(bands.middle):
                self.assertTrue(math.isnan(percent_b))
                self.assertTrue(math.isnan(bandwidth))
                continue
            width = bands.upper - bands.lower
            if width == 0.0:
                expected_pb = 0.0  # safe_divide fallback when bands collapse
            else:
                expected_pb = (value - bands.lower) / width
            expected_bw = width / bands.middle if bands.middle != 0.0 else 0.0
            self.assertAlmostEqual(percent_b, expected_pb, places=12)
            self.assertAlmostEqual(bandwidth, expected_bw, places=12)

    def test_moving_average_envelope(self):
        values = [10.0, 11.0, 12.0, 13.0, 14.0]
        indicator = rtta.MovingAverageEnvelope(window=3, percent=0.1, fillna=True)
        for value in values:
            out = indicator.update(value)
        # last SMA of [12,13,14] = 13
        self.assertAlmostEqual(out.middle, 13.0)
        self.assertAlmostEqual(out.upper, 13.0 * 1.1)
        self.assertAlmostEqual(out.lower, 13.0 * 0.9)
        self.assertAlmostEqual(indicator.last_middle(), 13.0)

    def test_positive_volume_index(self):
        close = np.array([10.0, 11.0, 10.5, 12.0])
        volume = np.array([100.0, 120.0, 110.0, 150.0])
        indicator = rtta.PositiveVolumeIndex()
        got = [indicator.update(c, v) for c, v in zip(close, volume)]
        # start 1000; vol up 100->120: 1000*(1+0.1)=1100; vol down: hold; vol up: 1100*(1+12/10.5-1)
        self.assertAlmostEqual(got[0], 1000.0)
        self.assertAlmostEqual(got[1], 1100.0)
        self.assertAlmostEqual(got[2], 1100.0)
        self.assertAlmostEqual(got[3], 1100.0 * (12.0 / 10.5))
        np.testing.assert_allclose(rtta.PositiveVolumeIndex().batch(close, volume), got)

    def test_volume_oscillator(self):
        volume = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        indicator = rtta.VolumeOscillator(short_window=2, long_window=4, fillna=True)
        got = [indicator.update(float(v)) for v in volume]
        short = (50.0 + 60.0) / 2.0
        long = (30.0 + 40.0 + 50.0 + 60.0) / 4.0
        self.assertAlmostEqual(got[-1], 100.0 * (short - long) / long)

    def test_efficiency_ratio_bounds(self):
        # Perfect uptrend: ER should be 1 after window fills
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        indicator = rtta.EfficiencyRatio(window=3, fillna=True)
        results = [indicator.update(v) for v in values]
        self.assertAlmostEqual(results[-1], 1.0, places=12)

        # Pure oscillation with zero net move over window may be near 0
        chop = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        chop_er = [rtta.EfficiencyRatio(window=4, fillna=True).update(v) for v in chop]
        self.assertLess(chop_er[-1], 0.5)

    def test_historical_volatility_constant_price(self):
        values = np.full(30, 100.0)
        hv = [rtta.HistoricalVolatility(window=10, periods_per_year=252.0, fillna=True).update(float(v)) for v in values]
        self.assertAlmostEqual(hv[-1], 0.0, places=12)

    def test_historical_volatility_positive_for_moves(self):
        rng = np.random.default_rng(0)
        values = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=50)))
        hv = rtta.HistoricalVolatility(window=10, periods_per_year=252.0, fillna=True).batch(values)
        self.assertTrue(np.all(hv[10:] > 0.0))

    def test_chaikin_volatility_batch_matches_update(self):
        high = np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0])
        low = high - 1.0
        indicator = rtta.ChaikinVolatility(ema_window=3, roc_window=3, fillna=True)
        incremental = [indicator.update(h, l) for h, l in zip(high, low)]
        batch = rtta.ChaikinVolatility(ema_window=3, roc_window=3, fillna=True).batch(high, low)
        np.testing.assert_allclose(batch, incremental)


if __name__ == "__main__":
    main()
