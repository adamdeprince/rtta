from unittest import TestCase, main

from rtta.indicator import ATR

class ATRTest(TestCase):

    def test(self):
        atr = ATR(window=2)
        self.assertEqual(atr.update(10, 11, 9), 2.0)
        self.assertEqual(atr.update(12, 13, 10), 2.5)
        self.assertEqual(atr.update(11, 12, 10), 2.25)
        self.assertEqual(atr.update(15, 16, 13), 3.625)

if __name__ == "__main__":
    main()
