import math
from unittest import TestCase, main

from rtta.indicator import AccumulationDistribution, ChaikinOscillator, OnBalanceVolume


class AccumulationDistributionTest(TestCase):
    def test(self):
        ad = AccumulationDistribution()
        self.assertEqual(ad.update(close=8, high=10, low=5, volume=100), 20)
        self.assertEqual(ad.update(close=5, high=10, low=5, volume=50), -30)

    def test_oscillator(self):
        oscillator = ChaikinOscillator()
        self.assertTrue(math.isfinite(oscillator.update(8, 10, 5, 100)))


class OnBalanceVolumeTest(TestCase):
    def test(self):
        obv = OnBalanceVolume()
        self.assertEqual(obv.update(10, 100), 100)
        self.assertEqual(obv.update(11, 50), 150)
        self.assertEqual(obv.update(9, 20), 130)


if __name__ == "__main__":
    main()
