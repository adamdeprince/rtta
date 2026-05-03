import numpy as np
import pytest


def _reference_amihud(close, volume, window=30, scale=1_000_000.0, fillna=True):
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    output = np.empty_like(close)
    ratios = []
    previous_close = 0.0
    has_previous = False

    for index, (price, vol) in enumerate(zip(close, volume)):
        price_return = (price - previous_close) / previous_close if has_previous and previous_close != 0.0 else 0.0
        dollar_volume = abs(price * vol)
        ratio = 0.0 if dollar_volume == 0.0 else abs(price_return) / dollar_volume * scale
        ratios.append(ratio)
        if len(ratios) > window:
            ratios.pop(0)
        output[index] = np.nan if not fillna and len(ratios) < window else np.mean(ratios)
        previous_close = price
        has_previous = True

    return output


def test_update_matches_rolling_amihud_formula():
    import rtta

    close = np.asarray([10.0, 10.2, 10.1, 10.5, 10.3, 10.8], dtype=np.float64)
    volume = np.asarray([10_000.0, 12_000.0, 11_000.0, 15_000.0, 14_000.0, 13_000.0], dtype=np.float64)
    expected = _reference_amihud(close, volume, window=3)

    indicator = rtta.AmihudIlliquidity(window=3)
    actual = [indicator.update(float(c), float(v)) for c, v in zip(close, volume)]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert indicator.last() == pytest.approx(expected[-1])


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(808)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.45, 512))
    volume = rng.integers(1_000, 100_000, 512).astype(np.float64)
    arrays = {
        "close": np.ascontiguousarray(close, dtype=np.float64),
        "volume": np.ascontiguousarray(volume, dtype=np.float64),
    }
    expected = _reference_amihud(arrays["close"], arrays["volume"], window=32)
    records = [{name: float(values[i]) for name, values in arrays.items()} for i in range(len(close))]
    table = pandas.DataFrame(arrays, copy=False)
    arrays32 = {name: np.ascontiguousarray(values.astype(np.float32)) for name, values in arrays.items()}

    np.testing.assert_allclose(rtta.AmihudIlliquidity(window=32).batch(arrays["close"], arrays["volume"]), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rtta.AmihudIlliquidity(window=32).batch(records), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rtta.AmihudIlliquidity(window=32).batch(table), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        rtta.AmihudIlliquidity(window=32).batch(arrays32["close"], arrays32["volume"]),
        _reference_amihud(arrays32["close"], arrays32["volume"], window=32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_advance_replay_and_fillna():
    import rtta

    close = np.ascontiguousarray([10.0, 10.2, 10.1, 10.5, 10.3], dtype=np.float64)
    volume = np.ascontiguousarray([10_000.0, 12_000.0, 11_000.0, 15_000.0, 14_000.0], dtype=np.float64)
    update_indicator = rtta.AmihudIlliquidity(window=3)
    update_results = [update_indicator.update(float(c), float(v)) for c, v in zip(close, volume)]
    advance_indicator = rtta.AmihudIlliquidity(window=3)
    for args, result in zip(zip(close, volume), update_results):
        assert advance_indicator.advance(*(float(value) for value in args)) is None
        assert advance_indicator.last() == pytest.approx(result)

    checksum = sum(update_results)
    assert rtta.AmihudIlliquidity(window=3).replay_update(close, volume) == pytest.approx(checksum)
    assert rtta.AmihudIlliquidity(window=3).replay_advance(close, volume) == pytest.approx(checksum)

    out = rtta.AmihudIlliquidity(window=3, fillna=False).batch(close, volume)
    assert np.isnan(out[:2]).all()
    assert np.isfinite(out[-1])
