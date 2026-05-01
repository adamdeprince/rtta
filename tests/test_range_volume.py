import math
from unittest import TestCase, main

from rtta.indicator import (
    ChandeMomentumOscillator,
    CommodityChannelIndex,
    MoneyFlowIndex,
    NormalizedATR,
    TrueRange,
    UltimateOscillator,
)


class RangeVolumeTest(TestCase):
    def test_true_range(self):
        trange = TrueRange()
        self.assertEqual(trange.update(10, 12, 9), 3)
        self.assertEqual(trange.update(11, 15, 10), 5)

    def test_normalized_atr(self):
        natr = NormalizedATR(window=2)
        self.assertAlmostEqual(natr.update(10, 12, 9), 30)

    def test_money_flow_index(self):
        mfi = MoneyFlowIndex(window=2)
        self.assertEqual(mfi.update(10, 11, 9, 100), 50)
        self.assertEqual(mfi.update(11, 12, 10, 100), 100)

    def test_commodity_channel_index(self):
        cci = CommodityChannelIndex(window=3)
        self.assertEqual(cci.update(10, 11, 9), 0)

    def test_chande_momentum_oscillator(self):
        cmo = ChandeMomentumOscillator(window=2)
        cmo.update(10)
        cmo.update(12)
        self.assertEqual(cmo.update(14), 100)

    def test_ultimate_oscillator(self):
        oscillator = UltimateOscillator()
        self.assertTrue(math.isfinite(oscillator.update(10, 11, 9)))


if __name__ == "__main__":
    main()
