import numpy as np
import pytest


def _reference_heikin_ashi(open_, high, low, close):
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    out_open = np.empty_like(close)
    out_high = np.empty_like(close)
    out_low = np.empty_like(close)
    out_close = np.empty_like(close)

    previous_open = 0.0
    previous_close = 0.0
    for index, values in enumerate(zip(open_, high, low, close)):
        o, h, l, c = values
        ha_close = 0.25 * (o + h + l + c)
        ha_open = 0.5 * (o + c) if index == 0 else 0.5 * (previous_open + previous_close)
        out_open[index] = ha_open
        out_high[index] = max(h, ha_open, ha_close)
        out_low[index] = min(l, ha_open, ha_close)
        out_close[index] = ha_close
        previous_open = ha_open
        previous_close = ha_close
    return out_open, out_high, out_low, out_close


def _assert_result_close(result, expected):
    for field, values in zip(("open", "high", "low", "close"), expected):
        np.testing.assert_allclose(getattr(result, field), values, rtol=1e-12, atol=1e-12)


def test_update_matches_reference_heikin_ashi_formula():
    import rtta

    open_ = np.asarray([10.0, 11.0, 12.0, 11.5], dtype=np.float64)
    high = np.asarray([12.0, 13.0, 12.5, 12.0], dtype=np.float64)
    low = np.asarray([9.0, 10.5, 11.0, 10.8], dtype=np.float64)
    close = np.asarray([11.0, 12.0, 11.8, 11.0], dtype=np.float64)
    expected = _reference_heikin_ashi(open_, high, low, close)
    indicator = rtta.HeikinAshiTransform()

    actual = [indicator.update(float(o), float(h), float(l), float(c)) for o, h, l, c in zip(open_, high, low, close)]

    for index, result in enumerate(actual):
        assert result.open == pytest.approx(expected[0][index])
        assert result.high == pytest.approx(expected[1][index])
        assert result.low == pytest.approx(expected[2][index])
        assert result.close == pytest.approx(expected[3][index])
    assert indicator.last().close == pytest.approx(expected[3][-1])


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(303)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.45, 512))
    open_ = close + rng.normal(0.0, 0.08, 512)
    spread = rng.uniform(0.02, 0.8, 512)
    high = np.ascontiguousarray(np.maximum(open_, close) + spread, dtype=np.float64)
    low = np.ascontiguousarray(np.minimum(open_, close) - spread, dtype=np.float64)
    arrays = {
        "open": np.ascontiguousarray(open_, dtype=np.float64),
        "high": high,
        "low": low,
        "close": np.ascontiguousarray(close, dtype=np.float64),
    }
    expected = _reference_heikin_ashi(arrays["open"], arrays["high"], arrays["low"], arrays["close"])
    records = [{name: float(values[i]) for name, values in arrays.items()} for i in range(len(close))]
    table = pandas.DataFrame(arrays, copy=False)
    arrays32 = {name: np.ascontiguousarray(values.astype(np.float32)) for name, values in arrays.items()}

    _assert_result_close(rtta.HeikinAshiTransform().batch(arrays["open"], arrays["high"], arrays["low"], arrays["close"]), expected)
    _assert_result_close(rtta.HeikinAshiTransform().batch(records), expected)
    _assert_result_close(rtta.HeikinAshiTransform().batch(table), expected)
    _assert_result_close(
        rtta.HeikinAshiTransform().batch(arrays32["open"], arrays32["high"], arrays32["low"], arrays32["close"]),
        _reference_heikin_ashi(arrays32["open"], arrays32["high"], arrays32["low"], arrays32["close"]),
    )


def test_advance_last_scalar_accessors_and_replay_outputs():
    import rtta

    open_ = np.ascontiguousarray([10.0, 11.0, 12.0], dtype=np.float64)
    high = np.ascontiguousarray([12.0, 13.0, 12.5], dtype=np.float64)
    low = np.ascontiguousarray([9.0, 10.5, 11.0], dtype=np.float64)
    close = np.ascontiguousarray([11.0, 12.0, 11.8], dtype=np.float64)
    update_indicator = rtta.HeikinAshiTransform()
    update_results = [update_indicator.update(float(o), float(h), float(l), float(c)) for o, h, l, c in zip(open_, high, low, close)]
    advance_indicator = rtta.HeikinAshiTransform()
    for args, result in zip(zip(open_, high, low, close), update_results):
        assert advance_indicator.advance(*(float(value) for value in args)) is None
        assert advance_indicator.last().close == pytest.approx(result.close)
        assert advance_indicator.last_open() == pytest.approx(result.open)
        assert advance_indicator.last_close() == pytest.approx(result.close)

    replay = rtta.HeikinAshiTransform().replay_update_outputs(open_, high, low, close)
    np.testing.assert_allclose(replay.close, [result.close for result in update_results], rtol=1e-12, atol=1e-12)
    checksum = sum(result.open + result.high + result.low + result.close for result in update_results)
    assert rtta.HeikinAshiTransform().replay_update(open_, high, low, close) == pytest.approx(checksum)
    assert rtta.HeikinAshiTransform().replay_advance(open_, high, low, close) == pytest.approx(checksum)
