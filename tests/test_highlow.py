from unittest import TestCase, main

from rtta.indicator import High, HighIndex, HighLow, HighLowIndex, Low, LowIndex, MidPoint, MidPrice


class HighLowTest(TestCase):
    def test_indexes(self):
        values = [3, 1, 4, 2]
        high = High(window=3)
        low = Low(window=3)
        high_index = HighIndex(window=3)
        low_index = LowIndex(window=3)

        for value in values:
            high_value = high.update(value)
            low_value = low.update(value)
            high_idx = high_index.update(value)
            low_idx = low_index.update(value)

        self.assertEqual(high_value, 4)
        self.assertEqual(low_value, 1)
        self.assertEqual(high_idx, 2)
        self.assertEqual(low_idx, 1)

    def test_pairs(self):
        high_low = HighLow(window=3).update(3)
        high_low_index = HighLowIndex(window=3).update(3)

        self.assertEqual(high_low.min, 3.0)
        self.assertEqual(high_low.max, 3.0)
        self.assertEqual(high_low_index.min_index, 0.0)
        self.assertEqual(high_low_index.max_index, 0.0)

    def test_midpoints(self):
        self.assertEqual(MidPoint(window=2).update(2), 2)
        self.assertEqual(MidPrice(window=2).update(4, 2), 3)


if __name__ == "__main__":
    main()
