import numpy as np
import pytest


def _profile_from_window(price, volume, bins, value_area_percent):
    price = np.asarray(price, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    min_price = float(np.min(price))
    max_price = float(np.max(price))
    total_volume = float(np.sum(volume))
    if total_volume == 0.0 or min_price == max_price:
        return min_price, max_price, min_price

    width = (max_price - min_price) / float(bins)
    histogram = np.zeros(bins, dtype=np.float64)
    for p, v in zip(price, volume):
        bin_index = int((p - min_price) / width)
        if bin_index >= bins:
            bin_index = bins - 1
        histogram[bin_index] += v

    point_of_control_bin = int(np.argmax(histogram))
    low_bin = point_of_control_bin
    high_bin = point_of_control_bin
    cumulative_volume = histogram[point_of_control_bin]
    target_volume = total_volume * value_area_percent / 100.0
    while cumulative_volume < target_volume and (low_bin > 0 or high_bin + 1 < bins):
        left_volume = histogram[low_bin - 1] if low_bin > 0 else -1.0
        right_volume = histogram[high_bin + 1] if high_bin + 1 < bins else -1.0
        if right_volume > left_volume:
            high_bin += 1
            cumulative_volume += right_volume
        else:
            low_bin -= 1
            cumulative_volume += left_volume

    point_of_control = min_price + (point_of_control_bin + 0.5) * width
    value_area_low = min_price + low_bin * width
    value_area_high = min_price + (high_bin + 1) * width
    return point_of_control, value_area_high, value_area_low


def _reference_volume_profile(close, volume, window=128, bins=24, value_area_percent=70.0, fillna=True):
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    point_of_control = np.empty_like(close)
    value_area_high = np.empty_like(close)
    value_area_low = np.empty_like(close)

    for index in range(len(close)):
        start = max(0, index + 1 - window)
        if not fillna and index + 1 < window:
            point_of_control[index] = np.nan
            value_area_high[index] = np.nan
            value_area_low[index] = np.nan
            continue
        poc, vah, val = _profile_from_window(close[start : index + 1], volume[start : index + 1], bins, value_area_percent)
        point_of_control[index] = poc
        value_area_high[index] = vah
        value_area_low[index] = val

    return point_of_control, value_area_high, value_area_low


def _assert_result_close(result, expected, *, rtol=1e-12, atol=1e-12):
    for field, values in zip(("point_of_control", "value_area_high", "value_area_low"), expected):
        np.testing.assert_allclose(getattr(result, field), values, rtol=rtol, atol=atol, equal_nan=True)


def test_update_matches_volume_profile_reference():
    import rtta

    close = np.asarray([10.0, 10.5, 11.0, 12.0, 11.8, 10.8], dtype=np.float64)
    volume = np.asarray([100.0, 300.0, 200.0, 500.0, 100.0, 400.0], dtype=np.float64)
    expected = _reference_volume_profile(close, volume, window=4, bins=4, value_area_percent=70.0)

    indicator = rtta.VolumeProfile(window=4, bins=4, value_area_percent=70.0)
    actual = [indicator.update(float(c), float(v)) for c, v in zip(close, volume)]

    for index, result in enumerate(actual):
        assert result.point_of_control == pytest.approx(expected[0][index])
        assert result.value_area_high == pytest.approx(expected[1][index])
        assert result.value_area_low == pytest.approx(expected[2][index])
    assert indicator.last().point_of_control == pytest.approx(expected[0][-1])


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(505)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.45, 512))
    volume = rng.integers(1_000, 100_000, 512).astype(np.float64)
    arrays = {
        "close": np.ascontiguousarray(close, dtype=np.float64),
        "volume": np.ascontiguousarray(volume, dtype=np.float64),
    }
    expected = _reference_volume_profile(arrays["close"], arrays["volume"], window=64, bins=16)
    records = [{name: float(values[i]) for name, values in arrays.items()} for i in range(len(close))]
    table = pandas.DataFrame(arrays, copy=False)
    arrays32 = {name: np.ascontiguousarray(values.astype(np.float32)) for name, values in arrays.items()}

    _assert_result_close(rtta.VolumeProfile(window=64, bins=16).batch(arrays["close"], arrays["volume"]), expected)
    _assert_result_close(rtta.VolumeProfile(window=64, bins=16).batch(records), expected)
    _assert_result_close(rtta.VolumeProfile(window=64, bins=16).batch(table), expected)
    _assert_result_close(
        rtta.VolumeProfile(window=64, bins=16).batch(arrays32["close"], arrays32["volume"]),
        _reference_volume_profile(arrays32["close"], arrays32["volume"], window=64, bins=16),
        rtol=1e-5,
        atol=1e-5,
    )


def test_advance_last_scalar_accessors_and_replay_outputs():
    import rtta

    close = np.ascontiguousarray([10.0, 10.5, 11.0, 12.0, 11.8], dtype=np.float64)
    volume = np.ascontiguousarray([100.0, 300.0, 200.0, 500.0, 100.0], dtype=np.float64)
    update_indicator = rtta.VolumeProfile(window=4, bins=4)
    update_results = [update_indicator.update(float(c), float(v)) for c, v in zip(close, volume)]
    advance_indicator = rtta.VolumeProfile(window=4, bins=4)

    for args, result in zip(zip(close, volume), update_results):
        assert advance_indicator.advance(*(float(value) for value in args)) is None
        assert advance_indicator.last().point_of_control == pytest.approx(result.point_of_control)
        assert advance_indicator.last_point_of_control() == pytest.approx(result.point_of_control)
        assert advance_indicator.last_value_area_high() == pytest.approx(result.value_area_high)
        assert advance_indicator.last_value_area_low() == pytest.approx(result.value_area_low)

    replay = rtta.VolumeProfile(window=4, bins=4).replay_update_outputs(close, volume)
    np.testing.assert_allclose(
        replay.point_of_control,
        [result.point_of_control for result in update_results],
        rtol=1e-12,
        atol=1e-12,
    )
    checksum = sum(
        result.point_of_control + result.value_area_high + result.value_area_low
        for result in update_results
    )
    assert rtta.VolumeProfile(window=4, bins=4).replay_update(close, volume) == pytest.approx(checksum)
    assert rtta.VolumeProfile(window=4, bins=4).replay_advance(close, volume) == pytest.approx(checksum)


def test_fillna_false_and_constructor_validation():
    import rtta

    out = rtta.VolumeProfile(window=3, bins=4, fillna=False).batch(
        np.ascontiguousarray([10.0, 11.0, 12.0], dtype=np.float64),
        np.ascontiguousarray([100.0, 100.0, 100.0], dtype=np.float64),
    )
    assert np.isnan(out.point_of_control[:2]).all()
    assert np.isfinite(out.point_of_control[2])

    with pytest.raises(ValueError):
        rtta.VolumeProfile(window=0)
    with pytest.raises(ValueError):
        rtta.VolumeProfile(bins=0)
    with pytest.raises(ValueError):
        rtta.VolumeProfile(value_area_percent=0.0)
