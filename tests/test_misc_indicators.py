import math
from unittest import TestCase, main

import rtta
from rtta.indicator import (
    AbsolutePriceOscillator,
    MACDFix,
    ParabolicSAR,
    PercentagePrice,
)


class MiscIndicatorTest(TestCase):
    def test_package_exports(self):
        self.assertIs(rtta.PercentagePrice, PercentagePrice)
        self.assertFalse(hasattr(rtta, "PPO"))
        self.assertFalse(hasattr(rtta, "BBANDS"))

    def test(self):
        self.assertTrue(math.isfinite(AbsolutePriceOscillator().update(10)))
        macd_fix = MACDFix(fillna=False).update(10)
        self.assertTrue(math.isnan(macd_fix.macd))
        self.assertTrue(math.isnan(macd_fix.signal))
        self.assertTrue(math.isnan(macd_fix.histogram))
        self.assertTrue(math.isfinite(PercentagePrice(fillna=True).update(10).ppo))
        self.assertTrue(math.isfinite(ParabolicSAR().update(10, 9)))


if __name__ == "__main__":
    main()
