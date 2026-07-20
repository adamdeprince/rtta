import math
from unittest import TestCase, main

import numpy as np

import rtta


class RetailOneOffsTest(TestCase):
    def test_rainbow_ma_and_osc(self):
        x = np.linspace(10.0, 20.0, 80)
        ma = rtta.RainbowMovingAverage(period=2, layers=5, fillna=True)
        outs = [ma.update(float(v)) for v in x]
        last = outs[-1]
        self.assertTrue(math.isfinite(last.outer))
        self.assertGreaterEqual(last.highest, last.lowest)
        self.assertAlmostEqual(last.mid, 0.5 * (last.highest + last.lowest), places=12)
        self.assertAlmostEqual(last.width, last.highest - last.lowest, places=12)
        batch = rtta.RainbowMovingAverage(period=2, layers=5, fillna=True).batch(x)
        np.testing.assert_allclose(batch.outer, [o.outer for o in outs])

        osc = rtta.RainbowOscillator(period=2, layers=5, fillna=True)
        oouts = [osc.update(float(v)) for v in x]
        self.assertTrue(math.isfinite(oouts[-1].value))
        self.assertAlmostEqual(
            oouts[-1].value, 100.0 * oouts[-1].width / x[-1], places=10
        )
        ob = rtta.RainbowOscillator(period=2, layers=5, fillna=True).batch(x)
        np.testing.assert_allclose(ob.position, [o.position for o in oouts], equal_nan=True)

    def test_chande_forecast_oscillator(self):
        x = np.linspace(50.0, 60.0, 40)
        ind = rtta.ChandeForecastOscillator(window=10, fillna=True)
        outs = [ind.update(float(v)) for v in x]
        self.assertTrue(math.isfinite(outs[-1]))
        batch = rtta.ChandeForecastOscillator(window=10, fillna=True).batch(x)
        np.testing.assert_allclose(batch, outs, equal_nan=True)
        # On a straight line TSF is near close so CFO ~ 0
        self.assertLess(abs(outs[-1]), 5.0)

    def test_ravi(self):
        x = np.concatenate([np.full(40, 10.0), np.linspace(10.0, 20.0, 40)])
        ind = rtta.RangeActionVerificationIndex(short_window=5, long_window=20, fillna=True)
        outs = [ind.update(float(v)) for v in x]
        self.assertTrue(math.isfinite(outs[-1]))
        self.assertGreater(outs[-1], 0.0)
        batch = rtta.RangeActionVerificationIndex(short_window=5, long_window=20, fillna=True).batch(x)
        np.testing.assert_allclose(batch, outs, equal_nan=True)

    def test_bulls_bears_power(self):
        close = np.array([10.0, 11.0, 12.0, 11.5, 13.0])
        high = close + 0.5
        low = close - 0.5
        bulls = rtta.BullsPower(window=3, fillna=True)
        bears = rtta.BearsPower(window=3, fillna=True)
        bouts = [bulls.update(float(h), float(c)) for h, c in zip(high, close)]
        eouts = [bears.update(float(l), float(c)) for l, c in zip(low, close)]
        self.assertTrue(all(math.isfinite(v) for v in bouts))
        self.assertTrue(all(math.isfinite(v) for v in eouts))
        # Matches Elder Ray components (same EMA seed path)
        elder = rtta.ElderRayIndex(window=3, fillna=True)
        er = [elder.update(float(c), float(h), float(l)) for c, h, l in zip(close, high, low)]
        np.testing.assert_allclose(bouts, [o.bull_power for o in er], atol=1e-12)
        np.testing.assert_allclose(eouts, [o.bear_power for o in er], atol=1e-12)
        # high above close => bulls > bears on same bar
        self.assertGreater(bouts[-1], eouts[-1])

    def test_projection_oscillator(self):
        t = np.linspace(0, 4 * np.pi, 100)
        close = 100 + 5 * np.sin(t)
        high = close + 1.0
        low = close - 1.0
        ind = rtta.ProjectionOscillator(window=14, signal_window=3, fillna=True)
        outs = [ind.update(float(h), float(l), float(c)) for h, l, c in zip(high, low, close)]
        self.assertTrue(math.isfinite(outs[-1].value))
        self.assertTrue(math.isfinite(outs[-1].signal))
        batch = rtta.ProjectionOscillator(window=14, signal_window=3, fillna=True).batch(high, low, close)
        np.testing.assert_allclose(batch.value, [o.value for o in outs], equal_nan=True)

    def test_inertia(self):
        rng = np.random.default_rng(0)
        close = 100 + np.cumsum(rng.normal(0, 0.5, 120))
        ind = rtta.Inertia(std_window=5, smooth_window=8, reg_window=10, fillna=True)
        outs = [ind.update(float(c)) for c in close]
        self.assertTrue(math.isfinite(outs[-1]))
        batch = rtta.Inertia(std_window=5, smooth_window=8, reg_window=10, fillna=True).batch(close)
        np.testing.assert_allclose(batch, outs, equal_nan=True)


class ResearchDepthTest(TestCase):
    def test_message_event_ofi(self):
        # bid add +10, ask add +10 (negative), bid cancel -5, buy trade +20
        et = np.array([1.0, 1.0, 2.0, 3.0])
        side = np.array([1.0, -1.0, 1.0, 1.0])
        size = np.array([10.0, 10.0, 5.0, 20.0])
        ind = rtta.MessageEventOrderFlowImbalance(window=10, fillna=True)
        outs = [ind.update(float(e), float(s), float(z)) for e, s, z in zip(et, side, size)]
        # contribs: +10, -10, -5, +20 => cumulative 15
        self.assertAlmostEqual(outs[-1].ofi, 15.0)
        self.assertAlmostEqual(outs[0].event, 10.0)
        self.assertAlmostEqual(outs[1].event, -10.0)
        self.assertAlmostEqual(outs[2].event, -5.0)
        batch = rtta.MessageEventOrderFlowImbalance(window=10, fillna=True).batch(et, side, size)
        np.testing.assert_allclose(batch.ofi, [o.ofi for o in outs])

    def test_hawkes_intensity(self):
        times = np.array([0.0, 1.0, 2.0, 3.0])
        jumps = np.array([1.0, 1.0, 1.0, 1.0])
        ind = rtta.HawkesIntensity(mu=1.0, alpha=0.5, beta=1.0)
        outs = [ind.update(float(t), float(j)) for t, j in zip(times, jumps)]
        # first: intensity = 1 + 0.5
        self.assertAlmostEqual(outs[0].intensity, 1.5)
        self.assertAlmostEqual(outs[0].baseline, 1.0)
        # intensity stays above baseline with ongoing events
        self.assertGreater(outs[-1].intensity, 1.0)
        batch = rtta.HawkesIntensity(mu=1.0, alpha=0.5, beta=1.0).batch(times, jumps)
        np.testing.assert_allclose(batch.intensity, [o.intensity for o in outs])
        # unit-jump convenience batch
        batch1 = rtta.HawkesIntensity(mu=1.0, alpha=0.5, beta=1.0).batch(times)
        np.testing.assert_allclose(batch1.intensity, batch.intensity)

    def test_weighted_multi_peer_ofi(self):
        rng = np.random.default_rng(3)
        n, peers = 60, 3
        peer = rng.normal(0, 1, (n, peers))
        weights = np.array([0.5, 0.3, 0.2])
        wmat = np.tile(weights, (n, 1))
        peer_mean = (peer * weights).sum(axis=1) / weights.sum()
        own = 0.6 * peer_mean + rng.normal(0, 0.15, n)
        ind = rtta.WeightedMultiPeerOrderFlowImbalance(window=20, fillna=True)
        outs = [
            ind.update(
                float(own[i]),
                np.ascontiguousarray(peer[i]),
                np.ascontiguousarray(weights),
            )
            for i in range(n)
        ]
        self.assertTrue(math.isfinite(outs[-1].beta))
        self.assertAlmostEqual(outs[-1].impact, outs[-1].beta * outs[-1].peer_mean, places=12)
        batch = rtta.WeightedMultiPeerOrderFlowImbalance(window=20, fillna=True).batch(
            np.ascontiguousarray(own),
            np.ascontiguousarray(peer),
            np.ascontiguousarray(wmat),
        )
        np.testing.assert_allclose(batch.peer_mean, [o.peer_mean for o in outs])
        # equal weights should match MultiPeer peer_mean
        eq = np.ones(peers) / peers
        mp = rtta.MultiPeerOrderFlowImbalance(window=20, fillna=True)
        wmp = rtta.WeightedMultiPeerOrderFlowImbalance(window=20, fillna=True)
        for i in range(n):
            a = mp.update(float(own[i]), np.ascontiguousarray(peer[i]))
            b = wmp.update(float(own[i]), np.ascontiguousarray(peer[i]), np.ascontiguousarray(eq))
            self.assertAlmostEqual(a.peer_mean, b.peer_mean, places=12)

    def test_conformal_bands(self):
        rng = np.random.default_rng(4)
        x = 100 + np.cumsum(rng.normal(0, 1.0, 100))
        ind = rtta.ConformalBands(window=20, alpha=0.1, fillna=True)
        outs = [ind.update(float(v)) for v in x]
        last = outs[-1]
        self.assertTrue(math.isfinite(last.middle))
        self.assertGreaterEqual(last.upper, last.lower)
        self.assertAlmostEqual(last.upper, last.middle + last.radius, places=12)
        self.assertAlmostEqual(last.lower, last.middle - last.radius, places=12)
        batch = rtta.ConformalBands(window=20, alpha=0.1, fillna=True).batch(x)
        np.testing.assert_allclose(batch.radius, [o.radius for o in outs], equal_nan=True)


if __name__ == "__main__":
    main()
