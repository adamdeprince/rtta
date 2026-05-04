import numpy as np
import pytest


FIELDS = (
    "bar_number",
    "rod_return",
    "frozen_rod_return",
    "loser_z",
    "winner_z",
    "range_z",
    "volume_shock",
    "transaction_shock",
    "vwap_gap",
    "pressure_score",
    "prediction",
    "radius",
    "score",
    "signal",
    "target_fraction",
    "max_trade_dollars",
    "realized_error",
    "entry_window",
    "exit_window",
    "frozen",
    "news_guard",
)


def _market_bars(size=512, seed=4404):
    rng = np.random.default_rng(seed)
    close = 80.0 + np.cumsum(rng.normal(0.0, 0.18, size))
    open_ = np.concatenate(([close[0]], close[:-1])) + rng.normal(0.0, 0.04, size)
    spread = rng.uniform(0.02, 0.28, size)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(30_000, 400_000, size).astype(np.float64)
    return {
        "open": np.ascontiguousarray(open_, dtype=np.float64),
        "high": np.ascontiguousarray(high, dtype=np.float64),
        "low": np.ascontiguousarray(low, dtype=np.float64),
        "close": np.ascontiguousarray(close, dtype=np.float64),
        "volume": np.ascontiguousarray(volume, dtype=np.float64),
    }


def _assert_result_close(actual, expected, *, rtol=1e-12, atol=1e-12):
    for field in FIELDS:
        left = float(getattr(actual, field))
        right = float(getattr(expected, field))
        if np.isnan(left) or np.isnan(right):
            assert np.isnan(left) and np.isnan(right)
        else:
            assert left == pytest.approx(right, rel=rtol, abs=atol)


def _assert_batch_close(actual, expected, *, rtol=1e-12, atol=1e-12):
    for field in FIELDS:
        np.testing.assert_allclose(
            getattr(actual, field),
            getattr(expected, field),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


def test_optional_update_session_reset_and_scalar_accessors():
    import rtta

    bars = _market_bars(96)
    indicator = rtta.ClosePressureReversalSignal(
        cutoff_after_bars=8,
        entry_start_after_bars=10,
        entry_end_after_bars=14,
        exit_after_bars=18,
        calibration_window=16,
        fillna=True,
    )
    scalar = rtta.ClosePressureReversalSignal(
        cutoff_after_bars=8,
        entry_start_after_bars=10,
        entry_end_after_bars=14,
        exit_after_bars=18,
        calibration_window=16,
        fillna=True,
    )

    first = indicator.update(
        float(bars["open"][0]),
        float(bars["high"][0]),
        float(bars["low"][0]),
        float(bars["close"][0]),
        float(bars["volume"][0]),
        vwap=float((bars["high"][0] + bars["low"][0] + bars["close"][0]) / 3.0),
        transactions=1000.0,
        previous_session_close=float(bars["close"][0] * 1.01),
        normal_dollar_volume=2_000_000.0,
        normal_transactions=900.0,
        reset_session=True,
    )
    assert first.bar_number == 1.0
    assert first.rod_return < 0.0
    assert first.max_trade_dollars == pytest.approx(40_000.0)

    signal_value = scalar.update_signal(
        float(bars["open"][0]),
        float(bars["high"][0]),
        float(bars["low"][0]),
        float(bars["close"][0]),
        float(bars["volume"][0]),
    )
    assert scalar.last_signal() == pytest.approx(signal_value)

    for index in range(1, 40):
        out = indicator.update(
            float(bars["open"][index]),
            float(bars["high"][index]),
            float(bars["low"][index]),
            float(bars["close"][index]),
            float(bars["volume"][index]),
        )

    assert out.bar_number == 40.0
    assert out.frozen
    assert out.exit_window
    assert np.isfinite(out.prediction)
    assert np.isfinite(out.radius)

    indicator.reset_session(previous_session_close=float(bars["close"][39]))
    reset_out = indicator.update(
        float(bars["open"][40]),
        float(bars["high"][40]),
        float(bars["low"][40]),
        float(bars["close"][40]),
        float(bars["volume"][40]),
    )
    assert reset_out.bar_number == 1.0


def test_fillna_false_early_outputs_are_nan():
    import rtta

    bars = _market_bars(8)
    result = rtta.ClosePressureReversalSignal(fillna=False).update(
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


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    bars = _market_bars(512)
    records = [{name: float(values[i]) for name, values in bars.items()} for i in range(512)]
    table = pandas.DataFrame(bars, copy=False)
    bars32 = {name: np.ascontiguousarray(values.astype(np.float32)) for name, values in bars.items()}

    kwargs = dict(
        cutoff_after_bars=8,
        entry_start_after_bars=10,
        entry_end_after_bars=14,
        exit_after_bars=18,
        calibration_window=16,
        fillna=True,
    )
    expected = rtta.ClosePressureReversalSignal(**kwargs).batch(
        bars["open"],
        bars["high"],
        bars["low"],
        bars["close"],
        bars["volume"],
    )
    _assert_batch_close(rtta.ClosePressureReversalSignal(**kwargs).batch(records), expected)
    _assert_batch_close(rtta.ClosePressureReversalSignal(**kwargs).batch(table), expected)

    expected32 = rtta.ClosePressureReversalSignal(**kwargs).batch(
        bars32["open"],
        bars32["high"],
        bars32["low"],
        bars32["close"],
        bars32["volume"],
    )
    _assert_batch_close(
        rtta.ClosePressureReversalSignal(**kwargs).batch(
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

    bars = _market_bars(128)
    kwargs = dict(
        cutoff_after_bars=8,
        entry_start_after_bars=10,
        entry_end_after_bars=14,
        exit_after_bars=18,
        calibration_window=16,
        fillna=True,
    )
    update_indicator = rtta.ClosePressureReversalSignal(**kwargs)
    advance_indicator = rtta.ClosePressureReversalSignal(**kwargs)
    update_results = []
    for args in zip(bars["open"], bars["high"], bars["low"], bars["close"], bars["volume"]):
        result = update_indicator.update(*(float(value) for value in args))
        update_results.append(result)
        assert advance_indicator.advance(*(float(value) for value in args)) is None
        _assert_result_close(advance_indicator.last(), result)

    checksum = sum(
        sum(float(getattr(result, field)) for field in FIELDS if np.isfinite(float(getattr(result, field))))
        for result in update_results
    )
    args = (bars["open"], bars["high"], bars["low"], bars["close"], bars["volume"])
    assert rtta.ClosePressureReversalSignal(**kwargs).replay_update(*args) == pytest.approx(checksum)
    assert rtta.ClosePressureReversalSignal(**kwargs).replay_advance(*args) == pytest.approx(checksum)

    batch = rtta.ClosePressureReversalSignal(**kwargs).batch(*args)
    replay = rtta.ClosePressureReversalSignal(**kwargs).replay_update_outputs(*args)
    _assert_batch_close(replay, batch)
