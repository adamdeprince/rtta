import numpy as np
import pytest


def _reference_renko(close, brick_size=1.0):
    close = np.asarray(close, dtype=np.float64)
    brick_open = np.empty_like(close)
    brick_close = np.empty_like(close)
    direction = np.empty_like(close)
    bricks = np.empty_like(close)
    reversal = np.empty_like(close)

    current = close[0]
    trend = 0.0
    for index, price in enumerate(close):
        if index == 0:
            brick_open[index] = price
            brick_close[index] = price
            direction[index] = 0.0
            bricks[index] = 0.0
            reversal[index] = 0.0
            continue

        start = current
        previous_trend = trend
        signed = 0.0
        while price >= current + brick_size:
            current += brick_size
            trend = 1.0
            signed += 1.0
        while price <= current - brick_size:
            current -= brick_size
            trend = -1.0
            signed -= 1.0

        brick_open[index] = start
        brick_close[index] = current
        direction[index] = trend
        bricks[index] = signed
        reversal[index] = 1.0 if signed != 0.0 and previous_trend != 0.0 and trend != previous_trend else 0.0

    return brick_open, brick_close, direction, bricks, reversal


def _assert_result_close(result, expected):
    for field, values in zip(("brick_open", "brick_close", "direction", "bricks", "reversal"), expected):
        np.testing.assert_allclose(getattr(result, field), values, rtol=0.0, atol=0.0)


def test_update_matches_reference_brick_generation():
    import rtta

    close = np.asarray([100.0, 100.4, 101.2, 103.8, 101.1, 99.0, 100.2], dtype=np.float64)
    expected = _reference_renko(close, brick_size=1.0)
    indicator = rtta.RenkoBrickGenerator(brick_size=1.0)

    actual = [indicator.update(float(value)) for value in close]

    for index, result in enumerate(actual):
        assert result.brick_open == expected[0][index]
        assert result.brick_close == expected[1][index]
        assert result.direction == expected[2][index]
        assert result.bricks == expected[3][index]
        assert result.reversal == expected[4][index]
    assert indicator.last().brick_close == expected[1][-1]


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(202)
    close = np.ascontiguousarray(100.0 + np.cumsum(rng.normal(0.0, 0.8, 512)), dtype=np.float64)
    expected = _reference_renko(close, brick_size=0.75)
    records = [{"close": float(value)} for value in close]
    table = pandas.DataFrame({"close": close}, copy=False)
    close32 = np.ascontiguousarray(close.astype(np.float32))

    _assert_result_close(rtta.RenkoBrickGenerator(brick_size=0.75).batch(close), expected)
    _assert_result_close(rtta.RenkoBrickGenerator(brick_size=0.75).batch(records), expected)
    _assert_result_close(rtta.RenkoBrickGenerator(brick_size=0.75).batch(table), expected)
    _assert_result_close(
        rtta.RenkoBrickGenerator(brick_size=0.75).batch(close32),
        _reference_renko(close32, brick_size=0.75),
    )


def test_advance_last_scalar_accessors_and_replay_outputs():
    import rtta

    close = np.ascontiguousarray([100.0, 101.2, 103.0, 100.0, 98.0], dtype=np.float64)
    update_indicator = rtta.RenkoBrickGenerator()
    update_results = [update_indicator.update(float(value)) for value in close]
    advance_indicator = rtta.RenkoBrickGenerator()
    for value, result in zip(close, update_results):
        assert advance_indicator.advance(float(value)) is None
        assert advance_indicator.last().brick_close == result.brick_close
        assert advance_indicator.last_brick_close() == result.brick_close
        assert advance_indicator.last_bricks() == result.bricks

    replay = rtta.RenkoBrickGenerator().replay_update_outputs(close)
    np.testing.assert_allclose(replay.brick_close, [result.brick_close for result in update_results], rtol=0.0, atol=0.0)
    checksum = sum(result.brick_open + result.brick_close + result.direction + result.bricks + result.reversal for result in update_results)
    assert rtta.RenkoBrickGenerator().replay_update(close) == pytest.approx(checksum)
    assert rtta.RenkoBrickGenerator().replay_advance(close) == pytest.approx(checksum)
