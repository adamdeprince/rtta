import numpy as np
import pytest


def _reference_kyle_lambda(close, signed_dollar_volume, window=30, scale=1_000_000.0, fillna=True):
    close = np.asarray(close, dtype=np.float64)
    signed_dollar_volume = np.asarray(signed_dollar_volume, dtype=np.float64)
    output = np.empty_like(close)
    returns = []
    flow = []
    previous_close = 0.0
    has_previous = False

    for index, (price, signed_flow) in enumerate(zip(close, signed_dollar_volume)):
        price_return = (price - previous_close) / previous_close if has_previous and previous_close != 0.0 else 0.0
        signed_sqrt_flow = np.sign(signed_flow) * np.sqrt(abs(signed_flow))
        returns.append(price_return)
        flow.append(signed_sqrt_flow)
        if len(returns) > window:
            returns.pop(0)
            flow.pop(0)

        if not fillna and len(returns) < window:
            output[index] = np.nan
        else:
            x = np.asarray(returns, dtype=np.float64)
            y = np.asarray(flow, dtype=np.float64)
            covariance = len(x) * np.sum(x * y) - np.sum(x) * np.sum(y)
            variance = len(y) * np.sum(y * y) - np.sum(y) * np.sum(y)
            output[index] = 0.0 if variance == 0.0 else covariance / variance * scale

        previous_close = price
        has_previous = True

    return output


def test_update_matches_rolling_price_impact_regression():
    import rtta

    close = np.asarray([10.0, 10.2, 10.1, 10.5, 10.3, 10.8], dtype=np.float64)
    signed_dollar_volume = np.asarray([0.0, 50_000.0, -30_000.0, 80_000.0, -20_000.0, 90_000.0], dtype=np.float64)
    expected = _reference_kyle_lambda(close, signed_dollar_volume, window=4, scale=1_000_000.0)

    indicator = rtta.KyleLambda(window=4)
    actual = [indicator.update(float(c), float(v)) for c, v in zip(close, signed_dollar_volume)]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert indicator.last() == pytest.approx(expected[-1])


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(707)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.45, 512))
    volume = rng.integers(1_000, 100_000, 512).astype(np.float64)
    direction = np.sign(np.diff(close, prepend=close[0]))
    direction[direction == 0.0] = 1.0
    signed_dollar_volume = direction * close * volume
    arrays = {
        "close": np.ascontiguousarray(close, dtype=np.float64),
        "signed_dollar_volume": np.ascontiguousarray(signed_dollar_volume, dtype=np.float64),
    }
    expected = _reference_kyle_lambda(arrays["close"], arrays["signed_dollar_volume"], window=32)
    records = [{name: float(values[i]) for name, values in arrays.items()} for i in range(len(close))]
    table = pandas.DataFrame(arrays, copy=False)
    arrays32 = {name: np.ascontiguousarray(values.astype(np.float32)) for name, values in arrays.items()}

    np.testing.assert_allclose(rtta.KyleLambda(window=32).batch(arrays["close"], arrays["signed_dollar_volume"]), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rtta.KyleLambda(window=32).batch(records), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rtta.KyleLambda(window=32).batch(table), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        rtta.KyleLambda(window=32).batch(arrays32["close"], arrays32["signed_dollar_volume"]),
        _reference_kyle_lambda(arrays32["close"], arrays32["signed_dollar_volume"], window=32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_advance_replay_and_fillna():
    import rtta

    close = np.ascontiguousarray([10.0, 10.2, 10.1, 10.5, 10.3], dtype=np.float64)
    signed_dollar_volume = np.ascontiguousarray([0.0, 50_000.0, -30_000.0, 80_000.0, -20_000.0], dtype=np.float64)
    update_indicator = rtta.KyleLambda(window=4)
    update_results = [update_indicator.update(float(c), float(v)) for c, v in zip(close, signed_dollar_volume)]
    advance_indicator = rtta.KyleLambda(window=4)
    for args, result in zip(zip(close, signed_dollar_volume), update_results):
        assert advance_indicator.advance(*(float(value) for value in args)) is None
        assert advance_indicator.last() == pytest.approx(result)

    checksum = sum(update_results)
    assert rtta.KyleLambda(window=4).replay_update(close, signed_dollar_volume) == pytest.approx(checksum)
    assert rtta.KyleLambda(window=4).replay_advance(close, signed_dollar_volume) == pytest.approx(checksum)

    out = rtta.KyleLambda(window=4, fillna=False).batch(close, signed_dollar_volume)
    assert np.isnan(out[:3]).all()
    assert np.isfinite(out[-1])
