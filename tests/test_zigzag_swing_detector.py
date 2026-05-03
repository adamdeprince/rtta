import numpy as np
import pytest


def _reference_zigzag(close, percent_change=5.0):
    close = np.asarray(close, dtype=np.float64)
    threshold = percent_change / 100.0 if percent_change > 1.0 else percent_change
    value = np.empty_like(close)
    direction = np.empty_like(close)
    pivot = np.empty_like(close)
    pivot_index = np.empty_like(close)

    start = close[0]
    extreme = close[0]
    extreme_index = 0
    last_pivot = close[0]
    last_pivot_index = 0
    swing = 0.0
    for index, price in enumerate(close):
        if index == 0:
            value[index] = price
            direction[index] = 0.0
            pivot[index] = price
            pivot_index[index] = 0.0
            continue

        confirmed_pivot = last_pivot
        confirmed_pivot_index = last_pivot_index
        if swing == 0.0:
            if price >= start * (1.0 + threshold):
                swing = 1.0
                last_pivot = start
                last_pivot_index = 0
                extreme = price
                extreme_index = index
            elif price <= start * (1.0 - threshold):
                swing = -1.0
                last_pivot = start
                last_pivot_index = 0
                extreme = price
                extreme_index = index
            elif abs(price - start) > abs(extreme - start):
                extreme = price
                extreme_index = index
            confirmed_pivot = last_pivot
            confirmed_pivot_index = last_pivot_index
        elif swing > 0.0:
            if price >= extreme:
                extreme = price
                extreme_index = index
            elif price <= extreme * (1.0 - threshold):
                last_pivot = extreme
                last_pivot_index = extreme_index
                swing = -1.0
                extreme = price
                extreme_index = index
                confirmed_pivot = last_pivot
                confirmed_pivot_index = last_pivot_index
        else:
            if price <= extreme:
                extreme = price
                extreme_index = index
            elif price >= extreme * (1.0 + threshold):
                last_pivot = extreme
                last_pivot_index = extreme_index
                swing = 1.0
                extreme = price
                extreme_index = index
                confirmed_pivot = last_pivot
                confirmed_pivot_index = last_pivot_index

        value[index] = extreme
        direction[index] = swing
        pivot[index] = confirmed_pivot
        pivot_index[index] = float(confirmed_pivot_index)

    return value, direction, pivot, pivot_index


def _assert_result_close(result, expected):
    for field, values in zip(("value", "direction", "pivot", "pivot_index"), expected):
        np.testing.assert_allclose(getattr(result, field), values, rtol=0.0, atol=0.0)


def test_update_matches_reference_reversal_logic():
    import rtta

    close = np.asarray([100.0, 103.0, 106.0, 104.0, 99.0, 96.0, 101.0, 102.0], dtype=np.float64)
    expected = _reference_zigzag(close, percent_change=5.0)
    indicator = rtta.ZigZagSwingDetector(percent_change=5.0)

    actual = [indicator.update(float(value)) for value in close]

    for index, result in enumerate(actual):
        assert result.value == expected[0][index]
        assert result.direction == expected[1][index]
        assert result.pivot == expected[2][index]
        assert result.pivot_index == expected[3][index]
    assert indicator.last().value == expected[0][-1]


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(101)
    close = np.ascontiguousarray(100.0 + np.cumsum(rng.normal(0.0, 0.8, 512)), dtype=np.float64)
    expected = _reference_zigzag(close, percent_change=3.0)
    records = [{"close": float(value)} for value in close]
    table = pandas.DataFrame({"close": close}, copy=False)
    close32 = np.ascontiguousarray(close.astype(np.float32))

    _assert_result_close(rtta.ZigZagSwingDetector(percent_change=3.0).batch(close), expected)
    _assert_result_close(rtta.ZigZagSwingDetector(percent_change=3.0).batch(records), expected)
    _assert_result_close(rtta.ZigZagSwingDetector(percent_change=3.0).batch(table), expected)
    _assert_result_close(
        rtta.ZigZagSwingDetector(percent_change=3.0).batch(close32),
        _reference_zigzag(close32, percent_change=3.0),
    )


def test_advance_last_scalar_accessors_and_replay_outputs():
    import rtta

    close = np.ascontiguousarray([100.0, 106.0, 99.0, 95.0, 101.0], dtype=np.float64)
    update_indicator = rtta.ZigZagSwingDetector()
    update_results = [update_indicator.update(float(value)) for value in close]
    advance_indicator = rtta.ZigZagSwingDetector()
    for value, result in zip(close, update_results):
        assert advance_indicator.advance(float(value)) is None
        assert advance_indicator.last().value == result.value
        assert advance_indicator.last_value() == result.value
        assert advance_indicator.last_pivot_index() == result.pivot_index

    replay = rtta.ZigZagSwingDetector().replay_update_outputs(close)
    np.testing.assert_allclose(replay.value, [result.value for result in update_results], rtol=0.0, atol=0.0)
    checksum = sum(result.value + result.direction + result.pivot + result.pivot_index for result in update_results)
    assert rtta.ZigZagSwingDetector().replay_update(close) == pytest.approx(checksum)
    assert rtta.ZigZagSwingDetector().replay_advance(close) == pytest.approx(checksum)
