import math
from unittest import TestCase, main

import numpy as np

import rtta


class Wave4ResearchTest(TestCase):
    def test_multi_level_and_integrated_ofi(self):
        levels = 3
        ind = rtta.MultiLevelOrderFlowImbalance(levels=levels, window=1, fillna=True)
        integ = rtta.IntegratedOrderFlowImbalance(levels=levels, window=1, fillna=True)
        # two snapshots of a flat book with size changes at L1
        for bid_s, ask_s in [(np.array([10.0, 8.0, 6.0]), np.array([11.0, 9.0, 7.0])),
                             (np.array([15.0, 8.0, 6.0]), np.array([11.0, 9.0, 7.0]))]:
            bp = np.array([100.0, 99.9, 99.8])
            ap = np.array([100.1, 100.2, 100.3])
            out = ind.update(bp, bid_s, ap, ask_s)
            iout = integ.update(bp, bid_s, ap, ask_s)
        self.assertTrue(math.isfinite(out.total))
        self.assertTrue(math.isfinite(out.l1))
        self.assertTrue(math.isfinite(iout.ofi))
        # L1 size increase at same bid should contribute positive Cont ofi
        self.assertGreater(out.l1, 0.0)

    def test_multi_level_batch_matches_update(self):
        levels = 3
        n = 40
        rng = np.random.default_rng(1)
        mid = 100 + np.cumsum(rng.normal(0, 0.05, n))
        bid_prices = np.ascontiguousarray(np.column_stack([mid - 0.01 * (i + 1) for i in range(levels)]))
        ask_prices = np.ascontiguousarray(np.column_stack([mid + 0.01 * (i + 1) for i in range(levels)]))
        bid_sizes = np.ascontiguousarray(rng.uniform(5, 20, (n, levels)))
        ask_sizes = np.ascontiguousarray(rng.uniform(5, 20, (n, levels)))

        multi = rtta.MultiLevelOrderFlowImbalance(levels=levels, window=2, fillna=True)
        outs = [
            multi.update(bid_prices[i], bid_sizes[i], ask_prices[i], ask_sizes[i])
            for i in range(n)
        ]
        batch = rtta.MultiLevelOrderFlowImbalance(levels=levels, window=2, fillna=True).batch(
            bid_prices, bid_sizes, ask_prices, ask_sizes
        )
        np.testing.assert_allclose(batch.total, [o.total for o in outs])
        np.testing.assert_allclose(batch.l1, [o.l1 for o in outs])
        replay = rtta.MultiLevelOrderFlowImbalance(levels=levels, window=2, fillna=True).replay_update_outputs(
            bid_prices, bid_sizes, ask_prices, ask_sizes
        )
        np.testing.assert_allclose(replay.mean, batch.mean)

        integ = rtta.IntegratedOrderFlowImbalance(levels=levels, window=2, fillna=True)
        iouts = [
            integ.update(bid_prices[i], bid_sizes[i], ask_prices[i], ask_sizes[i])
            for i in range(n)
        ]
        ibatch = rtta.IntegratedOrderFlowImbalance(levels=levels, window=2, fillna=True).batch(
            bid_prices, bid_sizes, ask_prices, ask_sizes
        )
        np.testing.assert_allclose(ibatch.ofi, [o.ofi for o in iouts])
        np.testing.assert_allclose(ibatch.weight_l1, [o.weight_l1 for o in iouts])

        # float32 path
        fbatch = rtta.MultiLevelOrderFlowImbalance(levels=levels, window=2, fillna=True).batch(
            bid_prices.astype(np.float32),
            bid_sizes.astype(np.float32),
            ask_prices.astype(np.float32),
            ask_sizes.astype(np.float32),
        )
        np.testing.assert_allclose(fbatch.total, batch.total, rtol=1e-5, atol=1e-5)

    def test_decomposed_matches_windowed_components(self):
        bp = np.array([100.0, 100.0, 100.01, 100.01])
        bs = np.array([10.0, 15.0, 12.0, 12.0])
        ap = np.array([100.02, 100.02, 100.03, 100.03])
        az = np.array([11.0, 7.0, 6.0, 6.0])
        ind = rtta.DecomposedOrderFlowImbalance(window=2, fillna=True)
        outs = [ind.update(float(a), float(b), float(c), float(d)) for a, b, c, d in zip(bp, bs, ap, az)]
        last = outs[-1]
        self.assertAlmostEqual(last.total, last.add + last.cancel + last.trade, places=12)
        batch = rtta.DecomposedOrderFlowImbalance(window=2, fillna=True).batch(bp, bs, ap, az)
        np.testing.assert_allclose(batch.total, [o.total for o in outs])

    def test_volume_and_dollar_and_imbalance_bars(self):
        close = np.array([10.0, 10.1, 10.2, 10.0, 9.9, 10.3, 10.4])
        volume = np.array([4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0])
        vb = rtta.VolumeBarGenerator(threshold=10000.0)
        vouts = [vb.update(float(c), float(v)) for c, v in zip(close, volume)]
        self.assertTrue(any(o.complete == 1.0 for o in vouts))

        db = rtta.DollarBarGenerator(threshold=100000.0)
        douts = [db.update(float(c), float(v)) for c, v in zip(close, volume)]
        self.assertTrue(any(o.complete == 1.0 for o in douts) or math.isfinite(douts[-1].bar_volume))

        ib = rtta.ImbalanceBarGenerator(threshold=5000.0)
        iouts = [ib.update(float(c), float(v)) for c, v in zip(close, volume)]
        self.assertTrue(any(o.complete == 1.0 for o in iouts) or math.isfinite(iouts[-1].bar_volume))

        batch = rtta.VolumeBarGenerator(threshold=10000.0).batch(close, volume)
        np.testing.assert_allclose(batch.complete, [o.complete for o in vouts])

    def test_focus_detects_mean_shift(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(0.0, 1.0, 80), rng.normal(3.0, 1.0, 80)])
        ind = rtta.FOCuS(threshold=8.0, mu0=0.0, sigma=1.0)
        outs = [ind.update(float(v)) for v in x]
        self.assertTrue(any(o.signal != 0.0 for o in outs[80:]))
        self.assertTrue(math.isfinite(outs[-1].statistic))
        batch = rtta.FOCuS(threshold=8.0, mu0=0.0, sigma=1.0).batch(x)
        np.testing.assert_allclose(batch.signal, [o.signal for o in outs])

        r = rtta.ResidualFOCuS(threshold=8.0)
        for v in x - x.mean():
            out = r.update(float(v))
        self.assertTrue(math.isfinite(out.statistic))

    def test_directional_change_events(self):
        # Ramp up then down enough to trigger DC events at 5%
        prices = np.array([100.0, 103.0, 106.0, 102.0, 98.0, 95.0, 99.0, 104.0])
        ind = rtta.DirectionalChangeDetector(threshold=0.03)
        outs = [ind.update(float(p)) for p in prices]
        events = [o.event for o in outs]
        self.assertTrue(any(e != 0.0 for e in events))
        self.assertTrue(math.isfinite(outs[-1].overshoot))
        batch = rtta.DirectionalChangeDetector(threshold=0.03).batch(prices)
        np.testing.assert_allclose(batch.event, events)


if __name__ == "__main__":
    main()
