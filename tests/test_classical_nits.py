import math
from unittest import TestCase, main

import numpy as np

import rtta


class ClassicalNitsTest(TestCase):
    def test_fibonacci_pivots(self):
        ind = rtta.FibonacciPivotPoints(fillna=True)
        ind.update(12.0, 10.0, 11.0)
        out = ind.update(13.0, 11.0, 12.0)
        pp = (12 + 10 + 11) / 3.0
        rng = 12 - 10
        self.assertAlmostEqual(out.pp, pp)
        self.assertAlmostEqual(out.r1, pp + 0.382 * rng)
        self.assertAlmostEqual(out.s2, pp - 0.618 * rng)
        batch = rtta.FibonacciPivotPoints(fillna=True).batch(
            np.array([12.0, 13.0]), np.array([10.0, 11.0]), np.array([11.0, 12.0])
        )
        self.assertAlmostEqual(float(batch.pp[1]), pp)

    def test_guppy_ribbon(self):
        x = np.linspace(10, 20, 80)
        ind = rtta.GuppyMMARibbon(fillna=True)
        outs = [ind.update(float(v)) for v in x]
        last = outs[-1]
        self.assertTrue(math.isfinite(last.s3))
        self.assertTrue(math.isfinite(last.l60))
        self.assertAlmostEqual(last.short_average, (last.s3 + last.s5 + last.s8 + last.s10 + last.s12 + last.s15) / 6.0, places=10)
        self.assertAlmostEqual(last.spread, last.short_average - last.long_average, places=12)
        batch = rtta.GuppyMMARibbon(fillna=True).batch(x)
        np.testing.assert_allclose(batch.s15, [o.s15 for o in outs])
        # Group averages should match compressed Guppy on same path
        g = rtta.GuppyMultipleMovingAverage(fillna=True)
        gouts = [g.update(float(v)) for v in x]
        self.assertAlmostEqual(last.short_average, gouts[-1].short_average, places=8)
        self.assertAlmostEqual(last.long_average, gouts[-1].long_average, places=8)

    def test_andrews_pitchfork_finite(self):
        # Sinusoid to force swings
        t = np.linspace(0, 8 * np.pi, 200)
        close = 100 + 5 * np.sin(t)
        high = close + 0.5
        low = close - 0.5
        ind = rtta.AndrewsPitchfork(percent_change=0.02, fillna=True)
        outs = [ind.update(float(h), float(l), float(c)) for h, l, c in zip(high, low, close)]
        self.assertTrue(any(o.pivot == 1.0 for o in outs))
        self.assertTrue(math.isfinite(outs[-1].median))
        batch = rtta.AndrewsPitchfork(percent_change=0.02, fillna=True).batch(high, low, close)
        np.testing.assert_allclose(batch.median, [o.median for o in outs], equal_nan=True)

    def test_elder_thermometer(self):
        ind = rtta.ElderThermometer(fillna=True)
        a = ind.update(12.0, 10.0)  # range 2
        b = ind.update(15.0, 10.0)  # range 5
        self.assertAlmostEqual(a.ratio, 1.0)
        self.assertAlmostEqual(b.ratio, 5.0 / 2.0)
        self.assertEqual(b.hot, 1.0)
        c = ind.update(11.0, 10.0)  # range 1
        self.assertEqual(c.hot, 0.0)
        batch = rtta.ElderThermometer(fillna=True).batch(
            np.array([12.0, 15.0, 11.0]), np.array([10.0, 10.0, 10.0])
        )
        np.testing.assert_allclose(batch.hot, [0.0, 1.0, 0.0])


if __name__ == "__main__":
    main()
