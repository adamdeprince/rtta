from unittest import TestCase, main

from rtta.indicator import FastStochastic, Stochastic, WilliamsR


class StochasticTest(TestCase):
    def test(self):
        stochastic = Stochastic(fastk=3, slowk=1, slowd=1)
        fast = FastStochastic(fastk=3, fastd=1)

        for close, high, low in [(2, 3, 1), (3, 4, 1), (5, 5, 1)]:
            stochastic_value = stochastic.update(close, high, low)
            fast_value = fast.update(close, high, low)

        self.assertEqual(stochastic_value.slowk, 100)
        self.assertEqual(fast_value.fastk, 100)

    def test_williams_r(self):
        williams = WilliamsR(window=3)
        for close, high, low in [(2, 3, 1), (3, 4, 1), (5, 5, 1)]:
            value = williams.update(close, high, low)
        self.assertEqual(value, 0)


if __name__ == "__main__":
    main()
