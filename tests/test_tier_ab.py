import math
from unittest import TestCase, main

import numpy as np

import rtta


class TierABTest(TestCase):
    def test_macd_multi_output(self):
        values = np.linspace(10, 20, 80)
        ind = rtta.MACD(fillna=True)
        outs = [ind.update(float(v)) for v in values]
        self.assertTrue(math.isfinite(outs[-1].macd))
        self.assertTrue(math.isfinite(outs[-1].signal))
        self.assertAlmostEqual(outs[-1].histogram, outs[-1].macd - outs[-1].signal, places=12)
        batch = rtta.MACD(fillna=True).batch(values)
        np.testing.assert_allclose(batch.macd, [o.macd for o in outs])
        np.testing.assert_allclose(batch.signal, [o.signal for o in outs])

    def test_macd_ext_ema_matches_macd(self):
        values = np.linspace(10, 30, 100)
        a = rtta.MACD(fillna=True)
        b = rtta.MACDExt(fast_ma_type=1, slow_ma_type=1, signal_ma_type=1, fillna=True)
        for v in values:
            x = a.update(float(v))
            y = b.update(float(v))
        self.assertAlmostEqual(x.macd, y.macd, places=10)
        self.assertAlmostEqual(x.signal, y.signal, places=10)

    def test_qstick_and_bias_and_psy(self):
        open_ = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        close = np.array([1.5, 1.5, 3.5, 3.5, 5.5, 6.5])
        qs = rtta.QStick(window=3, fillna=True)
        outs = [qs.update(float(o), float(c)) for o, c in zip(open_, close)]
        # diffs: 0.5, -0.5, 0.5, -0.5, 0.5, 0.5 -> last3 mean = (-0.5+0.5+0.5)/3
        self.assertAlmostEqual(outs[-1], 1.0 / 6.0)

        bias = rtta.Bias(window=3, fillna=True)
        bvals = [bias.update(float(c)) for c in close]
        self.assertTrue(math.isfinite(bvals[-1]))

        psy = rtta.PsychologicalLine(window=3, fillna=True)
        pvals = [psy.update(float(c)) for c in close]
        self.assertTrue(0.0 <= pvals[-1] <= 100.0)

    def test_pivot_points_previous_bar(self):
        ind = rtta.PivotPoints(fillna=True)
        # first bar seeds previous
        ind.update(12.0, 10.0, 11.0)
        out = ind.update(13.0, 11.0, 12.0)
        # pivots from first bar H12 L10 C11
        pp = (12 + 10 + 11) / 3.0
        self.assertAlmostEqual(out.pp, pp)
        self.assertAlmostEqual(out.r1, 2 * pp - 10)
        self.assertAlmostEqual(out.s1, 2 * pp - 12)

    def test_rvi_and_ifrsi(self):
        close = 100 + np.sin(np.linspace(0, 6 * np.pi, 80))
        rvi = rtta.RelativeVolatilityIndex(fillna=True)
        outs = [rvi.update(float(v)) for v in close]
        self.assertTrue(0.0 <= outs[-1].rvi <= 100.0)

        ifr = rtta.InverseFisherRSI(fillna=True)
        vals = [ifr.update(float(v)) for v in close]
        self.assertTrue(-1.01 <= vals[-1] <= 1.01)

    def test_williams_and_fractals(self):
        high = np.array([1, 2, 5, 2, 1, 2, 6, 2, 1], dtype=float)
        low = high - 0.5
        close = high - 0.2
        fr = rtta.WilliamsFractals(fillna=True)
        outs = [fr.update(float(h), float(l)) for h, l in zip(high, low)]
        # mid of first five is high[2]=5 should become up fractal when bar 4 arrives
        self.assertEqual(outs[4].up, 5.0)

        wad = rtta.WilliamsAD()
        vals = [wad.update(float(h), float(l), float(c)) for h, l, c in zip(high, low, close)]
        self.assertTrue(math.isfinite(vals[-1]))

    def test_swing_asi_mfi_ii_tmf(self):
        open_ = np.linspace(10, 12, 20)
        high = open_ + 1
        low = open_ - 1
        close = open_ + 0.2
        volume = np.full(20, 1000.0)
        si = rtta.SwingIndex(limit=1.0)
        asi = rtta.AccumulativeSwingIndex(limit=1.0)
        svals = []
        avals = []
        for o, h, l, c in zip(open_, high, low, close):
            svals.append(si.update(float(o), float(h), float(l), float(c)))
            avals.append(asi.update(float(o), float(h), float(l), float(c)))
        self.assertAlmostEqual(avals[-1], sum(svals), places=10)

        mfi = rtta.MarketFacilitationIndex()
        self.assertAlmostEqual(mfi.update(12.0, 10.0, 100.0), 0.02)

        ii = rtta.IntradayIntensity(window=5, fillna=True)
        tmf = rtta.TwiggsMoneyFlow(window=5, fillna=True)
        for h, l, c, v in zip(high, low, close, volume):
            ii_v = ii.update(float(h), float(l), float(c), float(v))
            tmf_v = tmf.update(float(h), float(l), float(c), float(v))
        self.assertTrue(math.isfinite(ii_v))
        self.assertTrue(math.isfinite(tmf_v))

    def test_vhf_rwi_pgo_tii_crs(self):
        close = np.linspace(10, 20, 40)
        high = close + 0.5
        low = close - 0.5
        vhf = [rtta.VerticalHorizontalFilter(window=10, fillna=True).update(float(c)) for c in close]
        # recreate properly
        # Strictly monotonic closes: (HCP-LCP) / path length == 1.
        ind = rtta.VerticalHorizontalFilter(window=10, fillna=True)
        vhf = [ind.update(float(c)) for c in close]
        self.assertAlmostEqual(vhf[-1], 1.0, places=8)

        rwi = rtta.RandomWalkIndex(window=10, fillna=True)
        outs = [rwi.update(float(c), float(h), float(l)) for c, h, l in zip(close, high, low)]
        self.assertTrue(math.isfinite(outs[-1].high))
        self.assertTrue(math.isfinite(outs[-1].low))

        pgo = rtta.PrettyGoodOscillator(window=10, fillna=True)
        pvals = [pgo.update(float(c), float(h), float(l)) for c, h, l in zip(close, high, low)]
        self.assertTrue(math.isfinite(pvals[-1]))

        tii = rtta.TrendIntensityIndex(window=10, fillna=True)
        tvals = [tii.update(float(c)) for c in close]
        self.assertTrue(0.0 <= tvals[-1] <= 100.0)

        crs = rtta.ComparativeRelativeStrength()
        self.assertAlmostEqual(crs.update(10.0, 5.0), 2.0)


if __name__ == "__main__":
    main()
