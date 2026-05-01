from math import isnan
from unittest import TestCase, main

import numpy as np

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

    def test_midprice_batch_preserves_incremental_state(self):
        highs = [10.0, 11.0, 13.0, 12.0, 14.0, 13.0]
        lows = [8.0, 7.0, 9.0, 6.0, 10.0, 9.0]

        expected = MidPrice(window=3)
        expected_values = [expected.update(high, low) for high, low in zip(highs, lows)]

        actual = MidPrice(window=3)
        self.assertEqual(actual.update(highs[0], lows[0]), expected_values[0])
        batch = actual.batch(
            np.array(highs[1:5], dtype=np.float64),
            np.array(lows[1:5], dtype=np.float64),
        )

        self.assertEqual(batch.tolist(), expected_values[1:5])
        self.assertEqual(actual.update(highs[5], lows[5]), expected_values[5])

    def test_midprice_batch_fillna_false(self):
        midprice = MidPrice(window=3, fillna=False)
        batch = midprice.batch(
            np.array([10.0, 11.0, 13.0], dtype=np.float64),
            np.array([8.0, 7.0, 9.0], dtype=np.float64),
        )

        self.assertTrue(isnan(batch[0]))
        self.assertTrue(isnan(batch[1]))
        self.assertEqual(batch[2], 10.0)


if __name__ == "__main__":
    main()
