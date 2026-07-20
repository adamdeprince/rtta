import math
from unittest import TestCase, main

import numpy as np

import rtta


def _ohlc(rows):
    o, h, l, c = zip(*rows)
    return (
        np.asarray(o, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
        np.asarray(l, dtype=np.float64),
        np.asarray(c, dtype=np.float64),
    )


class CDLPatternsTest(TestCase):
    def test_doji_and_marubozu(self):
        # perfect doji
        doji = rtta.CDLDoji(fillna=True)
        self.assertEqual(doji.update(10.0, 12.0, 8.0, 10.0), 100.0)
        # marubozu-like bull bar
        maru = rtta.CDLMarubozu(fillna=True)
        self.assertEqual(maru.update(10.0, 15.0, 10.0, 15.0), 100.0)
        self.assertEqual(maru.update(15.0, 15.0, 10.0, 10.0), -100.0)

    def test_engulfing(self):
        # bear then bull engulfing
        eng = rtta.CDLEngulfing(fillna=True)
        self.assertEqual(eng.update(12.0, 12.5, 10.0, 10.5), 0.0)  # bear
        self.assertEqual(eng.update(10.0, 13.5, 9.5, 13.0), 100.0)  # bull engulfs

    def test_hammer_shape_in_downtrend(self):
        # declining closes then hammer
        ham = rtta.CDLHammer(fillna=True)
        rows = [
            (20.0, 21.0, 19.0, 19.5),
            (19.5, 20.0, 18.0, 18.2),
            (18.2, 18.5, 15.0, 18.0),  # long lower shadow
        ]
        outs = [ham.update(*r) for r in rows]
        self.assertEqual(outs[-1], 100.0)

    def test_morning_star(self):
        ms = rtta.CDLMorningStar(fillna=True)
        # long bear, small body, long bull recovering midpoint
        rows = [
            (20.0, 20.5, 15.0, 15.2),  # long bear
            (15.0, 15.5, 14.5, 15.1),  # small
            (15.2, 19.0, 15.0, 18.5),  # bull
        ]
        outs = [ms.update(*r) for r in rows]
        self.assertEqual(outs[-1], 100.0)

    def test_batch_matches_update(self):
        rng = np.random.default_rng(0)
        n = 80
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        open_ = close + rng.normal(0, 0.2, n)
        high = np.maximum(open_, close) + rng.uniform(0.1, 0.8, n)
        low = np.minimum(open_, close) - rng.uniform(0.1, 0.8, n)
        for name in ("CDLDoji", "CDLEngulfing", "CDLHarami", "CDL3WhiteSoldiers"):
            cls = getattr(rtta, name)
            ind = cls(fillna=True)
            outs = [ind.update(float(o), float(h), float(l), float(c)) for o, h, l, c in zip(open_, high, low, close)]
            batch = cls(fillna=True).batch(open_, high, low, close)
            np.testing.assert_allclose(batch, outs, equal_nan=True)

    def test_pattern_pack(self):
        pack = rtta.CDLPatternPack(fillna=True)
        # doji bar
        out = pack.update(10.0, 12.0, 8.0, 10.0)
        self.assertEqual(out.doji, 100.0)
        o, h, l, c = _ohlc(
            [
                (12.0, 12.5, 10.0, 10.5),
                (10.0, 13.5, 9.5, 13.0),
            ]
        )
        batch = rtta.CDLPatternPack(fillna=True).batch(o, h, l, c)
        self.assertEqual(float(batch.engulfing[-1]), 100.0)

    def test_outputs_are_ta_lib_style(self):
        for name in ("CDLDoji", "CDLEngulfing", "CDLMorningStar", "CDLSpinningTop"):
            cls = getattr(rtta, name)
            ind = cls(fillna=True)
            v = ind.update(10.0, 11.0, 9.0, 10.5)
            self.assertTrue(v in (0.0, 100.0, -100.0) or math.isnan(v))


if __name__ == "__main__":
    main()
