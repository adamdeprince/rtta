import math
from unittest import TestCase, main

import numpy as np

import rtta


class TierCDTest(TestCase):
    def test_ehlers_super_smoother_finite(self):
        x = 100 + np.sin(np.linspace(0, 8 * np.pi, 80))
        ind = rtta.EhlersSuperSmoother(period=10, fillna=True)
        outs = [ind.update(float(v)) for v in x]
        self.assertTrue(math.isfinite(outs[-1]))
        batch = rtta.EhlersSuperSmoother(period=10, fillna=True).batch(x)
        np.testing.assert_allclose(batch, outs)

    def test_ehlers_pack_multi_output(self):
        x = 50 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, 120))
        roof = rtta.EhlersRoofingFilter(fillna=True)
        cy = rtta.EhlersCyberCycle(fillna=True)
        cg = rtta.EhlersCenterOfGravity(fillna=True)
        it = rtta.EhlersInstantaneousTrendline(fillna=True)
        de = rtta.EhlersDecycler(fillna=True)
        for v in x:
            r = roof.update(float(v))
            c = cy.update(float(v))
            g = cg.update(float(v))
            t = it.update(float(v))
            d = de.update(float(v))
        self.assertTrue(math.isfinite(r.roof))
        self.assertTrue(math.isfinite(c.cycle))
        self.assertTrue(math.isfinite(g.cg))
        self.assertTrue(math.isfinite(t.trendline))
        self.assertTrue(math.isfinite(d.decycle))
        self.assertAlmostEqual(d.oscillator, float(x[-1]) - d.decycle, places=10)

    def test_sar_ext_defaults(self):
        high = np.array([10, 11, 12, 11.5, 13, 12.5, 11, 10.5, 12], dtype=float)
        low = high - 1.0
        b = rtta.ParabolicSARExtended()
        for h, l in zip(high, low):
            vb = b.update(float(h), float(l))
        self.assertTrue(math.isfinite(vb))

    def test_kagi_and_pnf(self):
        prices = np.array([10, 11, 12, 11.5, 10.2, 9.5, 10.8, 12.5], dtype=float)
        kagi = rtta.KagiChart(reversal=1.0)
        outs = [kagi.update(float(p)) for p in prices]
        self.assertTrue(any(o.reversal == 1.0 for o in outs))
        self.assertIn(outs[-1].direction, (-1.0, 1.0))

        pnf = rtta.PointAndFigure(box_size=1.0, reversal_boxes=3)
        pouts = [pnf.update(float(p)) for p in prices]
        self.assertTrue(math.isfinite(pouts[-1].box_price))
        batch = rtta.KagiChart(reversal=1.0).batch(prices)
        np.testing.assert_allclose(batch.line, [o.line for o in outs])

    def test_guppy_median_geo(self):
        x = np.linspace(10, 20, 80)
        g = rtta.GuppyMultipleMovingAverage(fillna=True)
        gout = [g.update(float(v)) for v in x][-1]
        self.assertTrue(math.isfinite(gout.short_average))
        self.assertTrue(math.isfinite(gout.long_average))
        self.assertAlmostEqual(gout.spread, gout.short_average - gout.long_average, places=12)

        med = rtta.RollingMedian(window=5, fillna=True)
        series = [1, 3, 2, 5, 4, 6]
        mvals = [med.update(float(v)) for v in series]
        self.assertAlmostEqual(mvals[-1], 4.0)

        geo = rtta.GeometricMovingAverage(window=3, fillna=True)
        vals = [geo.update(float(v)) for v in [1.0, 2.0, 4.0]]
        self.assertAlmostEqual(vals[-1], 2.0, places=12)


if __name__ == "__main__":
    main()
