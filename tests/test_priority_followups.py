import math
from unittest import TestCase, main

import numpy as np

import rtta


class PriorityFollowupsTest(TestCase):
    def test_cross_asset_ofi(self):
        rng = np.random.default_rng(0)
        peer = rng.normal(0, 1, 100)
        own = 0.4 * peer + rng.normal(0, 0.2, 100)
        ind = rtta.CrossAssetOrderFlowImbalance(window=30, fillna=True)
        outs = [ind.update(float(r), float(p)) for r, p in zip(own, peer)]
        self.assertTrue(math.isfinite(outs[-1].beta))
        self.assertAlmostEqual(outs[-1].residual, outs[-1].beta * 0 + (own[-1] - outs[-1].impact), places=10)
        self.assertAlmostEqual(outs[-1].impact, outs[-1].beta * peer[-1], places=12)
        batch = rtta.CrossAssetOrderFlowImbalance(window=30, fillna=True).batch(own, peer)
        np.testing.assert_allclose(batch.beta, [o.beta for o in outs])

    def test_residual_bocpd(self):
        rng = np.random.default_rng(1)
        x = np.concatenate([rng.normal(0, 0.1, 80), rng.normal(8, 0.1, 80)])
        ind = rtta.ResidualBOCPD(max_run_length=64, hazard=0.2, threshold=0.15)
        outs = [ind.update(float(v)) for v in x]
        self.assertTrue(all(o.signal in (0.0, 1.0) for o in outs))
        self.assertTrue(all(0.0 <= o.probability <= 1.0 for o in outs))
        # With hazard mass on run-length 0, low threshold should fire at least once.
        self.assertTrue(any(o.signal == 1.0 for o in outs))
        batch = rtta.ResidualBOCPD(max_run_length=64, hazard=0.2, threshold=0.15).batch(x)
        np.testing.assert_allclose(batch.signal, [o.signal for o in outs])
        np.testing.assert_allclose(batch.probability, [o.probability for o in outs])

    def test_run_bar_generator(self):
        # Strictly alternating would not complete; use a mono run
        closes = np.array([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0])
        ind = rtta.RunBarGenerator(threshold=5)
        outs = [ind.update(float(c)) for c in closes]
        self.assertTrue(any(o.complete == 1.0 for o in outs))
        batch = rtta.RunBarGenerator(threshold=5).batch(closes)
        np.testing.assert_allclose(batch.complete, [o.complete for o in outs])
        # volume overload
        vol = np.ones_like(closes)
        vouts = [rtta.RunBarGenerator(threshold=5).update(float(c), float(v)) for c, v in zip(closes, vol)]
        # single instance for state
        ind2 = rtta.RunBarGenerator(threshold=5)
        vouts = [ind2.update(float(c), float(v)) for c, v in zip(closes, vol)]
        self.assertTrue(any(o.complete == 1.0 for o in vouts))

    def test_woodie_and_camarilla(self):
        w = rtta.WoodiePivotPoints(fillna=True)
        c = rtta.CamarillaPivotPoints(fillna=True)
        w.update(12.0, 10.0, 11.0)
        c.update(12.0, 10.0, 11.0)
        wo = w.update(13.0, 11.0, 12.0)
        co = c.update(13.0, 11.0, 12.0)
        # Woodie PP = (H+L+2C)/4 from prev
        self.assertAlmostEqual(wo.pp, (12 + 10 + 2 * 11) / 4.0)
        self.assertAlmostEqual(wo.r1, 2 * wo.pp - 10)
        # Camarilla R1 from prev
        rng = 12 - 10
        self.assertAlmostEqual(co.r1, 11 + rng * (1.1 / 12.0))
        self.assertAlmostEqual(co.s1, 11 - rng * (1.1 / 12.0))

    def test_ichimoku_displaced_spans(self):
        high = np.arange(10.0, 40.0)
        low = high - 1.0
        close = high - 0.2
        ind = rtta.Ichimoku(window1=2, window2=3, window3=4, fillna=True)
        outs = [ind.update(float(h), float(l), float(c)) for h, l, c in zip(high, low, close)]
        # After enough bars, displaced should equal span from 3 bars ago
        self.assertAlmostEqual(outs[10].span_a_displaced, outs[7].span_a, places=10)
        self.assertAlmostEqual(outs[10].span_b_displaced, outs[7].span_b, places=10)
        batch = rtta.Ichimoku(window1=2, window2=3, window3=4, fillna=True).batch(high, low, close)
        np.testing.assert_allclose(batch.span_a_displaced, [o.span_a_displaced for o in outs])

    def test_mavp(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        periods = np.array([2.0, 2.0, 3.0, 3.0, 3.0, 2.0])
        ind = rtta.MovingAverageVariablePeriod(max_period=5, min_period=2, fillna=True)
        outs = [ind.update(float(v), float(p)) for v, p in zip(values, periods)]
        # last with period 2: mean of 5,6 = 5.5
        self.assertAlmostEqual(outs[-1], 5.5)
        batch = rtta.MovingAverageVariablePeriod(max_period=5, min_period=2, fillna=True).batch(values, periods)
        np.testing.assert_allclose(batch, outs)


if __name__ == "__main__":
    main()
