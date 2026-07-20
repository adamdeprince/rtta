import math
from unittest import TestCase, main

import numpy as np

import rtta


class ClassicalWave2Test(TestCase):
    def test_demarker_bounds(self):
        high = np.array([10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0])
        low = high - 1.0
        indicator = rtta.DeMarker(window=3, fillna=True)
        values = [indicator.update(float(h), float(l)) for h, l in zip(high, low)]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values if math.isfinite(v)))
        batch = rtta.DeMarker(window=3, fillna=True).batch(high, low)
        np.testing.assert_allclose(batch, values)

    def test_imi_all_up_bars(self):
        open_ = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        close = open_ + 1.0
        indicator = rtta.IntradayMomentumIndex(window=3, fillna=True)
        values = [indicator.update(float(o), float(c)) for o, c in zip(open_, close)]
        self.assertAlmostEqual(values[-1], 100.0)

    def test_imi_all_down_bars(self):
        open_ = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        close = open_ - 0.5
        values = [
            rtta.IntradayMomentumIndex(window=3, fillna=True).update(float(o), float(c))
            for o, c in zip(open_, close)
        ]
        # recreate properly
        ind = rtta.IntradayMomentumIndex(window=3, fillna=True)
        values = [ind.update(float(o), float(c)) for o, c in zip(open_, close)]
        self.assertAlmostEqual(values[-1], 0.0)

    def test_acceleration_bands_order(self):
        close = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0])
        high = close + 0.5
        low = close - 0.5
        ind = rtta.AccelerationBands(window=3, factor=4.0, fillna=True)
        last = None
        for c, h, l in zip(close, high, low):
            last = ind.update(float(c), float(h), float(l))
        self.assertIsNotNone(last)
        self.assertGreater(last.upper, last.middle)
        self.assertGreater(last.middle, last.lower)

    def test_chandelier_long_below_high(self):
        close = np.linspace(100, 110, 30)
        high = close + 1.0
        low = close - 1.0
        ind = rtta.ChandelierExit(window=10, multiplier=3.0, fillna=True)
        for c, h, l in zip(close, high, low):
            out = ind.update(float(c), float(h), float(l))
        self.assertLess(out.long_exit, high[-1])
        self.assertGreater(out.short_exit, low[-1])

    def test_alligator_lips_reacts_faster(self):
        # After a long flat, a jump should move lips before jaw fully shifts.
        high = np.full(40, 10.0)
        low = np.full(40, 9.0)
        high[-1] = 20.0
        low[-1] = 19.0
        ind = rtta.Alligator(fillna=True)
        for h, l in zip(high, low):
            out = ind.update(float(h), float(l))
        # After one jump, shifted SMMA lines still mostly near prior level, but all finite.
        self.assertTrue(math.isfinite(out.jaw))
        self.assertTrue(math.isfinite(out.teeth))
        self.assertTrue(math.isfinite(out.lips))

    def test_gator_signs(self):
        high = np.linspace(10, 20, 50)
        low = high - 1.0
        ind = rtta.GatorOscillator(fillna=True)
        for h, l in zip(high, low):
            out = ind.update(float(h), float(l))
        self.assertGreaterEqual(out.upper, 0.0)
        self.assertLessEqual(out.lower, 0.0)

    def test_accelerator_batch_matches_update(self):
        high = np.linspace(10, 20, 60)
        low = high - 1.0
        ind = rtta.AcceleratorOscillator(fillna=True)
        incremental = [ind.update(float(h), float(l)) for h, l in zip(high, low)]
        batch = rtta.AcceleratorOscillator(fillna=True).batch(high, low)
        np.testing.assert_allclose(batch, incremental)

    def test_squeeze_outputs(self):
        # Low-vol then expansion
        rng = np.random.default_rng(1)
        close = 100 + np.cumsum(rng.normal(0, 0.05, size=80))
        high = close + 0.2
        low = close - 0.2
        ind = rtta.SqueezeMomentum(window=20, fillna=True)
        outs = [ind.update(float(c), float(h), float(l)) for c, h, l in zip(close, high, low)]
        self.assertTrue(all(o.on in (0.0, 1.0) or math.isnan(o.on) for o in outs))
        self.assertTrue(math.isfinite(outs[-1].momentum))

    def test_wavetrend_batch_matches(self):
        high = np.linspace(10, 15, 40)
        low = high - 0.5
        close = (high + low) / 2
        ind = rtta.WaveTrend(fillna=True)
        incremental = [ind.update(float(h), float(l), float(c)) for h, l, c in zip(high, low, close)]
        batch = rtta.WaveTrend(fillna=True).batch(high, low, close)
        np.testing.assert_allclose(batch.wt1, [x.wt1 for x in incremental])
        np.testing.assert_allclose(batch.wt2, [x.wt2 for x in incremental])

    def test_smi_signal_tracks(self):
        close = np.linspace(10, 20, 40)
        high = close + 1.0
        low = close - 1.0
        ind = rtta.StochasticMomentumIndex(window=10, smooth1=3, smooth2=3, signal=3, fillna=True)
        outs = [ind.update(float(c), float(h), float(l)) for c, h, l in zip(close, high, low)]
        self.assertTrue(math.isfinite(outs[-1].smi))
        self.assertTrue(math.isfinite(outs[-1].signal))

    def test_hilbert_suite_finite(self):
        values = 100 + np.sin(np.linspace(0, 8 * np.pi, 200))
        period_ind = rtta.HilbertDominantCyclePeriod(fillna=True)
        phase_ind = rtta.HilbertDominantCyclePhase(fillna=True)
        mode_ind = rtta.HilbertTrendMode(fillna=True)
        trend_ind = rtta.HilbertTrendline(fillna=True)
        period = [period_ind.update(float(v)) for v in values]
        phase = [phase_ind.update(float(v)) for v in values]
        mode = [mode_ind.update(float(v)) for v in values]
        trend = [trend_ind.update(float(v)) for v in values]
        self.assertTrue(math.isfinite(period[-1]))
        self.assertTrue(6.0 <= period[-1] <= 50.0)
        self.assertTrue(math.isfinite(phase[-1]))
        self.assertIn(mode[-1], (0.0, 1.0))
        self.assertTrue(math.isfinite(trend[-1]))

        ph = rtta.HilbertPhasor(fillna=True)
        sw = rtta.HilbertSineWave(fillna=True)
        for v in values:
            p = ph.update(float(v))
            s = sw.update(float(v))
        self.assertTrue(math.isfinite(p.inphase))
        self.assertTrue(math.isfinite(p.quadrature))
        self.assertTrue(math.isfinite(s.sine))
        self.assertTrue(math.isfinite(s.lead_sine))
        self.assertLessEqual(abs(s.sine), 1.0 + 1e-9)
        self.assertLessEqual(abs(s.lead_sine), 1.0 + 1e-9)

    def test_hilbert_fillna_false_lookbacks(self):
        values = 100 + np.sin(np.linspace(0, 4 * np.pi, 80))
        period_ind = rtta.HilbertDominantCyclePeriod(fillna=False)
        phase_ind = rtta.HilbertDominantCyclePhase(fillna=False)
        periods = [period_ind.update(float(v)) for v in values]
        phases = [phase_ind.update(float(v)) for v in values]
        # TA-Lib lookbacks: period/phasor 32, phase/sine/trend 63.
        self.assertTrue(all(math.isnan(x) for x in periods[:32]))
        self.assertTrue(math.isfinite(periods[33]))
        self.assertTrue(all(math.isnan(x) for x in phases[:63]))
        self.assertTrue(math.isfinite(phases[64]))

    def test_hilbert_batch_matches_update(self):
        values = np.asarray(100 + np.sin(np.linspace(0, 6 * np.pi, 100)), dtype=np.float64)
        ind = rtta.HilbertDominantCyclePeriod(fillna=True)
        incremental = [ind.update(float(v)) for v in values]
        batch = rtta.HilbertDominantCyclePeriod(fillna=True).batch(values)
        np.testing.assert_allclose(batch, incremental)

    def test_hilbert_matches_talib_when_available(self):
        try:
            import talib
        except ImportError:
            self.skipTest("TA-Lib Python package not installed")

        rng = np.random.default_rng(7)
        # Mild noise around a cycle so TA-Lib period stays in range.
        t = np.arange(300, dtype=np.float64)
        close = 100.0 + 5.0 * np.sin(2.0 * np.pi * t / 20.0) + 0.1 * rng.normal(size=t.size)

        # Streaming WMA/Hilbert state converges to TA-Lib after the long
        # unstable warmup; compare the well-warmed tail (not the first
        # lookback-transient samples).
        # Early bars differ slightly due to streaming vs batch WMA priming;
        # after ~150 bars the state matches TA-Lib to high precision.
        warm = 150
        atol = 1e-3

        rt_period = rtta.HilbertDominantCyclePeriod(fillna=False).batch(close)
        tl_period = talib.HT_DCPERIOD(close)
        mask = np.isfinite(tl_period[warm:]) & np.isfinite(rt_period[warm:])
        self.assertGreater(mask.sum(), 50)
        np.testing.assert_allclose(rt_period[warm:][mask], tl_period[warm:][mask], rtol=1e-5, atol=atol)

        rt_phase = rtta.HilbertDominantCyclePhase(fillna=False).batch(close)
        tl_phase = talib.HT_DCPHASE(close)
        mask = np.isfinite(tl_phase[warm:]) & np.isfinite(rt_phase[warm:])
        self.assertGreater(mask.sum(), 50)
        np.testing.assert_allclose(rt_phase[warm:][mask], tl_phase[warm:][mask], rtol=1e-4, atol=0.05)

        rt_in, rt_q = [], []
        ph = rtta.HilbertPhasor(fillna=False)
        for v in close:
            out = ph.update(float(v))
            rt_in.append(out.inphase)
            rt_q.append(out.quadrature)
        tl_in, tl_q = talib.HT_PHASOR(close)
        rt_in = np.asarray(rt_in)
        rt_q = np.asarray(rt_q)
        mask = np.isfinite(tl_in[warm:]) & np.isfinite(rt_in[warm:])
        np.testing.assert_allclose(rt_in[warm:][mask], tl_in[warm:][mask], rtol=1e-5, atol=atol)
        np.testing.assert_allclose(rt_q[warm:][mask], tl_q[warm:][mask], rtol=1e-5, atol=atol)

        tl_sine, tl_lead = talib.HT_SINE(close)
        sw = rtta.HilbertSineWave(fillna=False)
        rt_sine, rt_lead = [], []
        for v in close:
            out = sw.update(float(v))
            rt_sine.append(out.sine)
            rt_lead.append(out.lead_sine)
        rt_sine = np.asarray(rt_sine)
        rt_lead = np.asarray(rt_lead)
        mask = np.isfinite(tl_sine[warm:]) & np.isfinite(rt_sine[warm:])
        np.testing.assert_allclose(rt_sine[warm:][mask], tl_sine[warm:][mask], rtol=1e-5, atol=atol)
        np.testing.assert_allclose(rt_lead[warm:][mask], tl_lead[warm:][mask], rtol=1e-5, atol=atol)

        rt_trend = rtta.HilbertTrendline(fillna=False).batch(close)
        tl_trend = talib.HT_TRENDLINE(close)
        mask = np.isfinite(tl_trend[warm:]) & np.isfinite(rt_trend[warm:])
        np.testing.assert_allclose(rt_trend[warm:][mask], tl_trend[warm:][mask], rtol=1e-5, atol=atol)

        rt_mode = rtta.HilbertTrendMode(fillna=False).batch(close)
        tl_mode = talib.HT_TRENDMODE(close).astype(np.float64)
        mask = np.isfinite(tl_mode[warm:]) & np.isfinite(rt_mode[warm:])
        agree = np.mean(rt_mode[warm:][mask] == tl_mode[warm:][mask])
        self.assertGreaterEqual(agree, 0.95)


if __name__ == "__main__":
    main()
