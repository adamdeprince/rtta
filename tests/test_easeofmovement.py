from unittest import TestCase, main

import numpy as np

from rtta.indicator import EaseOfMovement


class EaseOfMovementTest(TestCase):
    def test_batch_matches_update(self):
        high = np.array([10.0, 12.0, 11.0, 15.0])
        low = np.array([8.0, 9.0, 10.0, 12.0])
        volume = np.array([100.0, 150.0, 120.0, 130.0])

        indicator = EaseOfMovement(window=2)
        incremental = [indicator.update(h, l, v) for h, l, v in zip(high, low, volume)]
        batch = EaseOfMovement(window=2).batch(high, low, volume)

        np.testing.assert_allclose(batch.ease_of_movement, [x.ease_of_movement for x in incremental])
        np.testing.assert_allclose(batch.sma, [x.sma for x in incremental])


if __name__ == "__main__":
    main()
