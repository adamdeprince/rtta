import math

import numpy as np
import pytest


def _normal_cdf(value):
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _reference_vpin(close, volume, bucket_volume=1_000_000.0, buckets=50, price_change_window=50, fillna=True):
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    output = np.empty_like(close)
    price_changes = []
    imbalances = []
    partial_buy = 0.0
    partial_sell = 0.0
    partial_volume = 0.0
    previous_close = 0.0
    has_previous = False

    for index, (price, vol) in enumerate(zip(close, volume)):
        price_change = price - previous_close if has_previous else 0.0
        price_changes.append(price_change)
        if len(price_changes) > price_change_window:
            price_changes.pop(0)

        if len(price_changes) < 2:
            sigma = 0.0
        else:
            sigma = float(np.std(price_changes, ddof=1))

        if sigma == 0.0:
            buy_fraction = 1.0 if price_change > 0.0 else 0.0 if price_change < 0.0 else 0.5
        else:
            buy_fraction = min(max(_normal_cdf(price_change / sigma), 0.0), 1.0)

        remaining_buy = max(float(vol), 0.0) * buy_fraction
        remaining_sell = max(float(vol), 0.0) * (1.0 - buy_fraction)
        remaining_volume = remaining_buy + remaining_sell
        while remaining_volume > 0.0:
            take = min(bucket_volume - partial_volume, remaining_volume)
            fraction = take / remaining_volume
            partial_buy += remaining_buy * fraction
            partial_sell += remaining_sell * fraction
            partial_volume += take
            remaining_buy -= remaining_buy * fraction
            remaining_sell -= remaining_sell * fraction
            remaining_volume -= take
            if partial_volume >= bucket_volume - 1e-12:
                imbalances.append(abs(partial_buy - partial_sell))
                if len(imbalances) > buckets:
                    imbalances.pop(0)
                partial_buy = 0.0
                partial_sell = 0.0
                partial_volume = 0.0

        if not imbalances:
            output[index] = 0.0 if fillna else np.nan
        elif not fillna and len(imbalances) < buckets:
            output[index] = np.nan
        else:
            output[index] = sum(imbalances) / (bucket_volume * len(imbalances))

        previous_close = price
        has_previous = True

    return output


def test_update_matches_reference_bulk_volume_classification():
    import rtta

    close = np.asarray([10.0, 10.2, 10.1, 10.4, 10.0, 10.3], dtype=np.float64)
    volume = np.asarray([60.0, 80.0, 120.0, 50.0, 90.0, 110.0], dtype=np.float64)
    expected = _reference_vpin(close, volume, bucket_volume=100.0, buckets=3, price_change_window=3)

    indicator = rtta.VPIN(bucket_volume=100.0, buckets=3, price_change_window=3)
    actual = [indicator.update(float(c), float(v)) for c, v in zip(close, volume)]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert indicator.last() == pytest.approx(expected[-1])


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(606)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.45, 512))
    volume = rng.integers(1_000, 100_000, 512).astype(np.float64)
    arrays = {
        "close": np.ascontiguousarray(close, dtype=np.float64),
        "volume": np.ascontiguousarray(volume, dtype=np.float64),
    }
    expected = _reference_vpin(arrays["close"], arrays["volume"], bucket_volume=500_000.0, buckets=10, price_change_window=16)
    records = [{name: float(values[i]) for name, values in arrays.items()} for i in range(len(close))]
    table = pandas.DataFrame(arrays, copy=False)
    arrays32 = {name: np.ascontiguousarray(values.astype(np.float32)) for name, values in arrays.items()}

    np.testing.assert_allclose(
        rtta.VPIN(bucket_volume=500_000.0, buckets=10, price_change_window=16).batch(arrays["close"], arrays["volume"]),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rtta.VPIN(bucket_volume=500_000.0, buckets=10, price_change_window=16).batch(records),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rtta.VPIN(bucket_volume=500_000.0, buckets=10, price_change_window=16).batch(table),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rtta.VPIN(bucket_volume=500_000.0, buckets=10, price_change_window=16).batch(arrays32["close"], arrays32["volume"]),
        _reference_vpin(arrays32["close"], arrays32["volume"], bucket_volume=500_000.0, buckets=10, price_change_window=16),
        rtol=1e-5,
        atol=1e-5,
    )


def test_advance_replay_and_fillna_validation():
    import rtta

    close = np.ascontiguousarray([10.0, 10.2, 10.1, 10.4, 10.0], dtype=np.float64)
    volume = np.ascontiguousarray([60.0, 80.0, 120.0, 50.0, 90.0], dtype=np.float64)
    update_indicator = rtta.VPIN(bucket_volume=100.0, buckets=3, price_change_window=3)
    update_results = [update_indicator.update(float(c), float(v)) for c, v in zip(close, volume)]
    advance_indicator = rtta.VPIN(bucket_volume=100.0, buckets=3, price_change_window=3)
    for args, result in zip(zip(close, volume), update_results):
        assert advance_indicator.advance(*(float(value) for value in args)) is None
        assert advance_indicator.last() == pytest.approx(result)

    checksum = sum(update_results)
    assert rtta.VPIN(bucket_volume=100.0, buckets=3, price_change_window=3).replay_update(close, volume) == pytest.approx(checksum)
    assert rtta.VPIN(bucket_volume=100.0, buckets=3, price_change_window=3).replay_advance(close, volume) == pytest.approx(checksum)

    out = rtta.VPIN(bucket_volume=100.0, buckets=3, price_change_window=3, fillna=False).batch(close, volume)
    assert np.isnan(out[:2]).all()
    assert np.isfinite(out[-1])

    with pytest.raises(ValueError):
        rtta.VPIN(bucket_volume=0.0)
    with pytest.raises(ValueError):
        rtta.VPIN(buckets=0)
    with pytest.raises(ValueError):
        rtta.VPIN(price_change_window=0)
