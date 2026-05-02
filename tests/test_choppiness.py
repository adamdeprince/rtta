import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rtta
from benchmarks.benchmark_indicators import generate_market_data


def _reference_choppiness(close, high, low, window=14, fillna=True):
    output = []
    true_ranges = []

    for index, (close_value, high_value, low_value) in enumerate(zip(close, high, low)):
        if index == 0:
            tr = high_value - low_value
        else:
            previous_close = close[index - 1]
            tr = max(high_value - low_value, abs(high_value - previous_close), abs(low_value - previous_close))
        true_ranges.append(tr)

        count = min(window, index + 1)
        if not fillna and count < window:
            output.append(math.nan)
            continue

        start = index + 1 - count
        tr_sum = sum(true_ranges[start : index + 1])
        highest = max(high[start : index + 1])
        lowest = min(low[start : index + 1])
        price_range = highest - lowest
        if price_range <= 0.0 or tr_sum <= 0.0:
            output.append(100.0)
            continue

        periods = max(count, 2) if fillna else window
        output.append(100.0 * math.log10(tr_sum / price_range) / math.log10(periods))

    return np.asarray(output, dtype=np.float64)


def test_choppiness_index_matches_reference_512_sample_sequence():
    data = generate_market_data(512, 20260504)
    indicator = rtta.ChoppinessIndex(window=14, fillna=True)
    actual = np.asarray(
        [
            indicator.update(close, high, low)
            for close, high, low in zip(data.lists["close"], data.lists["high"], data.lists["low"])
        ],
        dtype=np.float64,
    )
    expected = _reference_choppiness(data.arrays["close"], data.arrays["high"], data.arrays["low"])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_choppiness_index_fillna_false_and_batch_match_incremental():
    data = generate_market_data(128, 20260505)
    arrays = [data.arrays[name] for name in ("close", "high", "low")]
    expected = _reference_choppiness(*arrays, window=14, fillna=False)
    batch = rtta.ChoppinessIndex(window=14, fillna=False).batch(*arrays)
    np.testing.assert_allclose(batch, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    replay_indicator = rtta.ChoppinessIndex(window=14, fillna=False)
    checksum = replay_indicator.replay_update(*arrays)
    assert isinstance(checksum, float)
    next_value = replay_indicator.update(
        data.lists["close"][-1],
        data.lists["high"][-1],
        data.lists["low"][-1],
    )
    assert isinstance(next_value, float)
