import math
from unittest import TestCase, main

import numpy as np

import rtta


class ResearchExtensionsTest(TestCase):
    def test_kalman_innovation_residual_focus(self):
        rng = np.random.default_rng(0)
        close = 100 + np.cumsum(rng.normal(0, 0.5, 120))
        close = np.concatenate([close, close[-1] + np.cumsum(rng.normal(2.0, 0.5, 80))])
        ind = rtta.KalmanInnovationResidualFOCuS(focus_threshold=5.0, fillna=True)
        outs = [ind.update(float(c)) for c in close]
        self.assertTrue(all(math.isfinite(o.residual) or math.isnan(o.residual) for o in outs))
        self.assertTrue(any(o.signal != 0.0 for o in outs) or math.isfinite(outs[-1].score))
        batch = rtta.KalmanInnovationResidualFOCuS(focus_threshold=5.0, fillna=True).batch(close)
        np.testing.assert_allclose(batch.signal, [o.signal for o in outs], equal_nan=True)
        np.testing.assert_allclose(batch.residual, [o.residual for o in outs], equal_nan=True)

    def test_kalman_innovation_residual_bocpd(self):
        rng = np.random.default_rng(1)
        close = 50 + np.cumsum(rng.normal(0, 0.3, 100))
        close = np.concatenate([close, close[-1] + np.cumsum(rng.normal(1.5, 0.3, 100))])
        ind = rtta.KalmanInnovationResidualBOCPD(hazard=0.2, threshold=0.15, fillna=True)
        outs = [ind.update(float(c)) for c in close]
        self.assertTrue(all(o.signal in (0.0, 1.0) for o in outs))
        self.assertTrue(all(0.0 <= o.score <= 1.0 or math.isnan(o.score) for o in outs))
        batch = rtta.KalmanInnovationResidualBOCPD(hazard=0.2, threshold=0.15, fillna=True).batch(close)
        np.testing.assert_allclose(batch.signal, [o.signal for o in outs], equal_nan=True)

    def test_multi_peer_ofi(self):
        rng = np.random.default_rng(2)
        n, peers = 80, 3
        peer = rng.normal(0, 1, (n, peers))
        peer_mean = peer.mean(axis=1)
        own = 0.5 * peer_mean + rng.normal(0, 0.2, n)
        ind = rtta.MultiPeerOrderFlowImbalance(window=20, fillna=True)
        outs = [ind.update(float(own[i]), np.ascontiguousarray(peer[i])) for i in range(n)]
        self.assertTrue(math.isfinite(outs[-1].beta))
        self.assertAlmostEqual(outs[-1].impact, outs[-1].beta * outs[-1].peer_mean, places=12)
        batch = rtta.MultiPeerOrderFlowImbalance(window=20, fillna=True).batch(
            np.ascontiguousarray(own), np.ascontiguousarray(peer)
        )
        np.testing.assert_allclose(batch.beta, [o.beta for o in outs])
        np.testing.assert_allclose(batch.peer_mean, [o.peer_mean for o in outs])
        replay = rtta.MultiPeerOrderFlowImbalance(window=20, fillna=True).replay_update_outputs(
            np.ascontiguousarray(own), np.ascontiguousarray(peer)
        )
        np.testing.assert_allclose(replay.impact, batch.impact)

    def test_volume_and_dollar_run_bars(self):
        # Strong up-run with volume
        close = np.linspace(10, 20, 30)
        volume = np.full(30, 1000.0)
        v = rtta.VolumeRunBarGenerator(threshold=5000.0)
        vouts = [v.update(float(c), float(vol)) for c, vol in zip(close, volume)]
        self.assertTrue(any(o.complete == 1.0 for o in vouts))
        vb = rtta.VolumeRunBarGenerator(threshold=5000.0).batch(close, volume)
        np.testing.assert_allclose(vb.complete, [o.complete for o in vouts])

        d = rtta.DollarRunBarGenerator(threshold=50000.0)
        douts = [d.update(float(c), float(vol)) for c, vol in zip(close, volume)]
        self.assertTrue(any(o.complete == 1.0 for o in douts) or math.isfinite(douts[-1].bar_volume))
        db = rtta.DollarRunBarGenerator(threshold=50000.0).batch(close, volume)
        np.testing.assert_allclose(db.bar_volume, [o.bar_volume for o in douts])


if __name__ == "__main__":
    main()
