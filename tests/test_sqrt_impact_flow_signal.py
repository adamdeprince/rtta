import math
from unittest import TestCase, main

import numpy as np

import rtta


class SqrtImpactFlowSignalTest(TestCase):
    def test_basic_path_finite(self):
        rng = np.random.default_rng(0)
        close = 100 + np.cumsum(rng.normal(0, 0.3, 200))
        volume = rng.integers(1_000, 50_000, 200).astype(float)
        ind = rtta.SqrtImpactFlowSignal(fillna=True)
        outs = [ind.update(float(c), float(v)) for c, v in zip(close, volume)]
        self.assertTrue(all(o.signal in (-1.0, 0.0, 1.0) for o in outs))
        self.assertTrue(math.isfinite(outs[-1].score))
        self.assertGreaterEqual(outs[-1].impact, 0.0)
        self.assertGreater(outs[-1].volatility, 0.0)

    def test_batch_matches_update(self):
        rng = np.random.default_rng(1)
        close = 50 + np.cumsum(rng.normal(0, 0.2, 120))
        volume = rng.integers(500, 20_000, 120).astype(float)
        ind = rtta.SqrtImpactFlowSignal(fillna=True)
        outs = [ind.update(float(c), float(v)) for c, v in zip(close, volume)]
        batch = rtta.SqrtImpactFlowSignal(fillna=True).batch(close, volume)
        np.testing.assert_allclose(batch.score, [o.score for o in outs], equal_nan=True)
        np.testing.assert_allclose(batch.signal, [o.signal for o in outs], equal_nan=True)
        np.testing.assert_allclose(batch.impact, [o.impact for o in outs], equal_nan=True)

    def test_high_volume_small_move_boosts_continuation(self):
        # Flat prices then huge volume with tiny drift up
        close = np.concatenate([np.full(40, 100.0), np.array([100.01, 100.02, 100.03])])
        volume = np.concatenate([np.full(40, 1_000.0), np.array([500_000.0, 500_000.0, 500_000.0])])
        ind = rtta.SqrtImpactFlowSignal(
            impact_coefficient=1.0,
            continuation_weight=1.0,
            reversion_weight=0.0,
            vwap_weight=0.0,
            fillna=True,
        )
        outs = [ind.update(float(c), float(v)) for c, v in zip(close, volume)]
        # After large participation, continuation should be meaningful and non-negative for uptick
        self.assertGreater(outs[-1].participation, 1.0)
        self.assertGreaterEqual(outs[-1].continuation, 0.0)

    def test_signed_dollar_and_vwap_path(self):
        close = np.linspace(10, 12, 50)
        volume = np.full(50, 10_000.0)
        # Force sell flow even if price drifts up
        signed = np.full(50, -1.0e6)
        vwap = close * 1.001
        ind = rtta.SqrtImpactFlowSignal(fillna=True)
        outs = [
            ind.update(float(c), float(v), float(s), float(w))
            for c, v, s, w in zip(close, volume, signed, vwap)
        ]
        self.assertTrue(any(o.flow < 0 for o in outs[5:]))
        batch = rtta.SqrtImpactFlowSignal(fillna=True).batch(close, volume, signed, vwap)
        np.testing.assert_allclose(batch.vwap_gap, [o.vwap_gap for o in outs], equal_nan=True)

    def test_ohlcv_overload(self):
        o = np.array([10.0, 10.5, 11.0])
        h = o + 0.5
        l = o - 0.5
        c = o + 0.1
        v = np.array([1000.0, 2000.0, 1500.0])
        ind = rtta.SqrtImpactFlowSignal(fillna=True)
        a = [ind.update(float(x), float(vv)) for x, vv in zip(c, v)]
        ind2 = rtta.SqrtImpactFlowSignal(fillna=True)
        b = [ind2.update(float(oo), float(hh), float(ll), float(cc), float(vv))
             for oo, hh, ll, cc, vv in zip(o, h, l, c, v)]
        np.testing.assert_allclose([x.score for x in a], [x.score for x in b], equal_nan=True)


if __name__ == "__main__":
    main()
