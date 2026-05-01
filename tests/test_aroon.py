from unittest import TestCase, main

from rtta.indicator import Aroon, AroonOscillator


class AroonTest(TestCase):
    def test(self):
        aroon = Aroon(window=2)
        oscillator = AroonOscillator(window=2)

        for high, low in [(3, 1), (4, 2), (5, 3)]:
            aroon_value = aroon.update(high, low)
            oscillator_value = oscillator.update(high, low)

        self.assertEqual(aroon_value.up, 100)
        self.assertEqual(aroon_value.down, 0)
        self.assertEqual(oscillator_value, 100)


if __name__ == "__main__":
    main()
