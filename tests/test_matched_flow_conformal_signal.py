import numpy as np
import pytest


FIELDS = (
    "prediction",
    "radius",
    "score",
    "signal",
    "target_fraction",
    "alpha_flow",
    "participation",
    "flow_score",
    "momentum",
    "volatility",
    "vwap_gap",
    "rel_dollar_volume",
    "max_trade_dollars",
    "realized_error",
)


def _market_bars(size=512, seed=1010):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.25, size))
    open_ = np.concatenate(([close[0]], close[:-1])) + rng.normal(0.0, 0.03, size)
    spread = rng.uniform(0.02, 0.35, size)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(20_000, 250_000, size).astype(np.float64)
    return {
        "open": np.ascontiguousarray(open_, dtype=np.float64),
        "high": np.ascontiguousarray(high, dtype=np.float64),
        "low": np.ascontiguousarray(low, dtype=np.float64),
        "close": np.ascontiguousarray(close, dtype=np.float64),
        "volume": np.ascontiguousarray(volume, dtype=np.float64),
    }


def _assert_batch_close(result, expected, *, rtol=1e-12, atol=1e-12):
    for field in FIELDS:
        np.testing.assert_allclose(
            getattr(result, field),
            getattr(expected, field),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


def test_update_optional_inputs_and_session_reset():
    import rtta

    bars = _market_bars(80)
    indicator = rtta.MatchedFlowConformalSignal(
        horizon_bars=4,
        calibration_window=16,
        calibration_quantile=0.80,
        fillna=True,
    )

    first = indicator.update(
        float(bars["open"][0]),
        float(bars["high"][0]),
        float(bars["low"][0]),
        float(bars["close"][0]),
        float(bars["volume"][0]),
        normal_dollar_volume=2_000_000.0,
        market_cap=10_000_000_000.0,
        reset_session=True,
    )
    assert first.max_trade_dollars == pytest.approx(40_000.0)
    assert np.isfinite(first.radius)

    last = first
    for index in range(1, 80):
        last = indicator.update(
            float(bars["open"][index]),
            float(bars["high"][index]),
            float(bars["low"][index]),
            float(bars["close"][index]),
            float(bars["volume"][index]),
        )

    assert np.isfinite(last.prediction)
    assert np.isfinite(last.radius)
    assert np.isfinite(last.score)
    assert last.signal in (-1.0, 0.0, 1.0)
    assert abs(last.target_fraction) <= 0.05
    assert np.isfinite(last.realized_error)

    before_reset_gap = indicator.last().vwap_gap
    reset_result = indicator.update(
        float(bars["open"][0]),
        float(bars["high"][0]),
        float(bars["low"][0]),
        float(bars["close"][0]),
        float(bars["volume"][0]),
        reset_session=True,
    )
    assert before_reset_gap != pytest.approx(reset_result.vwap_gap)
    assert reset_result.vwap_gap == pytest.approx(0.0)


def test_fillna_false_early_outputs_are_nan_and_scalar_accessors_match():
    import rtta

    bars = _market_bars(24)
    indicator = rtta.MatchedFlowConformalSignal(horizon_bars=4, calibration_window=16, fillna=False)
    scalar_indicator = rtta.MatchedFlowConformalSignal(horizon_bars=4, calibration_window=16, fillna=False)

    result = indicator.update(
        float(bars["open"][0]),
        float(bars["high"][0]),
        float(bars["low"][0]),
        float(bars["close"][0]),
        float(bars["volume"][0]),
    )
    assert np.isnan(result.prediction)
    assert np.isnan(result.radius)
    assert np.isnan(result.score)
    assert result.signal == 0.0

    scalar = scalar_indicator.update_prediction(
        float(bars["open"][0]),
        float(bars["high"][0]),
        float(bars["low"][0]),
        float(bars["close"][0]),
        float(bars["volume"][0]),
    )
    assert np.isnan(scalar)
    assert np.isnan(scalar_indicator.last_prediction())


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    bars = _market_bars(512)
    records = [{name: float(values[i]) for name, values in bars.items()} for i in range(512)]
    table = pandas.DataFrame(bars, copy=False)
    bars32 = {name: np.ascontiguousarray(values.astype(np.float32)) for name, values in bars.items()}

    expected = rtta.MatchedFlowConformalSignal(fillna=True).batch(
        bars["open"],
        bars["high"],
        bars["low"],
        bars["close"],
        bars["volume"],
    )
    _assert_batch_close(rtta.MatchedFlowConformalSignal(fillna=True).batch(records), expected)
    _assert_batch_close(rtta.MatchedFlowConformalSignal(fillna=True).batch(table), expected)

    expected32 = rtta.MatchedFlowConformalSignal(fillna=True).batch(
        bars32["open"],
        bars32["high"],
        bars32["low"],
        bars32["close"],
        bars32["volume"],
    )
    _assert_batch_close(
        rtta.MatchedFlowConformalSignal(fillna=True).batch(
            bars32["open"],
            bars32["high"],
            bars32["low"],
            bars32["close"],
            bars32["volume"],
        ),
        expected32,
        rtol=1e-5,
        atol=1e-5,
    )


def test_advance_replay_and_replay_update_outputs():
    import rtta

    bars = _market_bars(96)
    update_indicator = rtta.MatchedFlowConformalSignal(horizon_bars=4, calibration_window=16, fillna=True)
    advance_indicator = rtta.MatchedFlowConformalSignal(horizon_bars=4, calibration_window=16, fillna=True)
    update_results = []
    for args in zip(bars["open"], bars["high"], bars["low"], bars["close"], bars["volume"]):
        update_result = update_indicator.update(*(float(value) for value in args))
        update_results.append(update_result)
        assert advance_indicator.advance(*(float(value) for value in args)) is None
        assert advance_indicator.last_signal() == pytest.approx(update_result.signal)

    checksum = sum(
        sum(float(getattr(result, field)) for field in FIELDS if np.isfinite(float(getattr(result, field))))
        for result in update_results
    )
    args = (bars["open"], bars["high"], bars["low"], bars["close"], bars["volume"])
    assert rtta.MatchedFlowConformalSignal(horizon_bars=4, calibration_window=16, fillna=True).replay_update(*args) == pytest.approx(checksum)
    assert rtta.MatchedFlowConformalSignal(horizon_bars=4, calibration_window=16, fillna=True).replay_advance(*args) == pytest.approx(checksum)

    batch = rtta.MatchedFlowConformalSignal(horizon_bars=4, calibration_window=16, fillna=True).batch(*args)
    replay = rtta.MatchedFlowConformalSignal(horizon_bars=4, calibration_window=16, fillna=True).replay_update_outputs(*args)
    _assert_batch_close(replay, batch)
