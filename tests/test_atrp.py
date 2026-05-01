from unittest import TestCase, main

from rtta.indicator import ATRP

class ATRTest(TestCase):

    def test(self):
        atr = ATRP(window=2)
        self.assertEqual(atr.update(10, 11, 9), 0.2)
        self.assertEqual(atr.update(12, 13, 10), 2.5 / 12.0)
        self.assertEqual(atr.update(11, 12, 10), 2.25 / 11.0)
        self.assertEqual(atr.update(15, 16, 13), 3.625 / 15.0)

if __name__ == "__main__":
    main()
