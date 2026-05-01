import math
from unittest import TestCase, main

from rtta.indicator import (
    AverageDirectionalMovementIndex,
    AverageDirectionalMovementIndexRating,
    DirectionalMovementIndex,
    MinusDirectionalIndicator,
    MinusDirectionalMovement,
    PlusDirectionalIndicator,
    PlusDirectionalMovement,
)


class DirectionalMovementTest(TestCase):
    def test(self):
        plus_indicator = PlusDirectionalIndicator(window=3)
        minus_indicator = MinusDirectionalIndicator(window=3)
        direction = DirectionalMovementIndex(window=3)
        average = AverageDirectionalMovementIndex(window=3)
        rating = AverageDirectionalMovementIndexRating(window=3)
        plus_movement = PlusDirectionalMovement(window=3)
        minus_movement = MinusDirectionalMovement(window=3)

        for close, high, low in [(10, 11, 9), (11, 13, 10), (12, 15, 11), (13, 16, 12)]:
            plus_value = plus_indicator.update(close, high, low)
            minus_value = minus_indicator.update(close, high, low)
            direction_value = direction.update(close, high, low)
            average_value = average.update(close, high, low)
            rating_value = rating.update(close, high, low)
            plus_movement_value = plus_movement.update(high, low)
            minus_movement_value = minus_movement.update(high, low)

        self.assertGreater(plus_value, minus_value)
        self.assertGreaterEqual(direction_value, 0)
        self.assertTrue(math.isfinite(average_value))
        self.assertTrue(math.isfinite(rating_value))
        self.assertGreater(plus_movement_value, minus_movement_value)


if __name__ == "__main__":
    main()
