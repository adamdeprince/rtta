import numpy as np
import pytest


def _reference_anchored_vwap(close, high, low, volume, anchor):
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    output = np.empty_like(close)
    cumulative_price_volume = 0.0
    cumulative_volume = 0.0

    for index, values in enumerate(zip(close, high, low, volume, anchor)):
        c, h, l, v, a = values
        if a != 0.0 or cumulative_volume == 0.0:
            cumulative_price_volume = 0.0
            cumulative_volume = 0.0
        typical_price = (h + l + c) / 3.0
        cumulative_price_volume += typical_price * v
        cumulative_volume += v
        output[index] = 0.0 if cumulative_volume == 0.0 else cumulative_price_volume / cumulative_volume

    return output


def test_update_matches_anchored_vwap_formula():
    import rtta

    close = np.asarray([10.0, 11.0, 12.0, 10.0, 9.0], dtype=np.float64)
    high = np.asarray([10.5, 11.4, 12.7, 10.3, 9.4], dtype=np.float64)
    low = np.asarray([9.5, 10.6, 11.8, 9.7, 8.8], dtype=np.float64)
    volume = np.asarray([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
    anchor = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    expected = _reference_anchored_vwap(close, high, low, volume, anchor)

    indicator = rtta.AnchoredVWAP()
    actual = [
        indicator.update(float(c), float(h), float(l), float(v), float(a))
        for c, h, l, v, a in zip(close, high, low, volume, anchor)
    ]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert indicator.last() == pytest.approx(expected[-1])


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(404)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.45, 512))
    open_ = close + rng.normal(0.0, 0.08, 512)
    spread = rng.uniform(0.02, 0.8, 512)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(1_000, 100_000, 512).astype(np.float64)
    anchor = np.zeros(512, dtype=np.float64)
    anchor[[0, 101, 307]] = 1.0
    arrays = {
        "close": np.ascontiguousarray(close, dtype=np.float64),
        "high": np.ascontiguousarray(high, dtype=np.float64),
        "low": np.ascontiguousarray(low, dtype=np.float64),
        "volume": np.ascontiguousarray(volume, dtype=np.float64),
        "anchor": np.ascontiguousarray(anchor, dtype=np.float64),
    }
    expected = _reference_anchored_vwap(
        arrays["close"],
        arrays["high"],
        arrays["low"],
        arrays["volume"],
        arrays["anchor"],
    )
    records = [{name: float(values[i]) for name, values in arrays.items()} for i in range(len(close))]
    table = pandas.DataFrame(arrays, copy=False)
    arrays32 = {name: np.ascontiguousarray(values.astype(np.float32)) for name, values in arrays.items()}

    np.testing.assert_allclose(
        rtta.AnchoredVWAP().batch(
            arrays["close"],
            arrays["high"],
            arrays["low"],
            arrays["volume"],
            arrays["anchor"],
        ),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(rtta.AnchoredVWAP().batch(records), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rtta.AnchoredVWAP().batch(table), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        rtta.AnchoredVWAP().batch(
            arrays32["close"],
            arrays32["high"],
            arrays32["low"],
            arrays32["volume"],
            arrays32["anchor"],
        ),
        _reference_anchored_vwap(
            arrays32["close"],
            arrays32["high"],
            arrays32["low"],
            arrays32["volume"],
            arrays32["anchor"],
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_advance_and_replay_paths_match_update_checksum():
    import rtta

    close = np.ascontiguousarray([10.0, 11.0, 12.0, 10.0], dtype=np.float64)
    high = np.ascontiguousarray([10.5, 11.4, 12.7, 10.3], dtype=np.float64)
    low = np.ascontiguousarray([9.5, 10.6, 11.8, 9.7], dtype=np.float64)
    volume = np.ascontiguousarray([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
    anchor = np.ascontiguousarray([1.0, 0.0, 0.0, 1.0], dtype=np.float64)

    update_indicator = rtta.AnchoredVWAP()
    update_results = [
        update_indicator.update(float(c), float(h), float(l), float(v), float(a))
        for c, h, l, v, a in zip(close, high, low, volume, anchor)
    ]
    advance_indicator = rtta.AnchoredVWAP()
    for args, result in zip(zip(close, high, low, volume, anchor), update_results):
        assert advance_indicator.advance(*(float(value) for value in args)) is None
        assert advance_indicator.last() == pytest.approx(result)

    checksum = sum(update_results)
    assert rtta.AnchoredVWAP().replay_update(close, high, low, volume, anchor) == pytest.approx(checksum)
    assert rtta.AnchoredVWAP().replay_advance(close, high, low, volume, anchor) == pytest.approx(checksum)
