import numpy as np
import pytest


def _reference_spread_features(trade_price, bid_price, ask_price, realized_horizon=5, fillna=True):
    trade_price = np.asarray(trade_price, dtype=np.float64)
    bid_price = np.asarray(bid_price, dtype=np.float64)
    ask_price = np.asarray(ask_price, dtype=np.float64)
    quoted_spread = np.empty_like(trade_price)
    effective_spread = np.empty_like(trade_price)
    realized_spread = np.empty_like(trade_price)
    pending_trade_price = []
    pending_side = []
    previous_trade_price = 0.0
    has_previous_trade = False

    for index, (trade, bid, ask) in enumerate(zip(trade_price, bid_price, ask_price)):
        midpoint = 0.5 * (bid + ask)
        if trade > midpoint:
            side = 1.0
        elif trade < midpoint:
            side = -1.0
        elif has_previous_trade and trade > previous_trade_price:
            side = 1.0
        elif has_previous_trade and trade < previous_trade_price:
            side = -1.0
        else:
            side = 0.0

        quoted_spread[index] = ask - bid
        effective_spread[index] = 2.0 * abs(trade - midpoint)
        if realized_horizon == 0:
            realized_spread[index] = 2.0 * side * (trade - midpoint)
        elif len(pending_trade_price) < realized_horizon:
            realized_spread[index] = 0.0 if fillna else np.nan
        else:
            realized_spread[index] = 2.0 * pending_side[0] * (pending_trade_price[0] - midpoint)

        if realized_horizon > 0:
            pending_trade_price.append(trade)
            pending_side.append(side)
            if len(pending_trade_price) > realized_horizon:
                pending_trade_price.pop(0)
                pending_side.pop(0)
        previous_trade_price = trade
        has_previous_trade = True

    return quoted_spread, effective_spread, realized_spread


def _assert_result_close(result, expected, *, rtol=1e-12, atol=1e-12):
    for field, values in zip(("quoted_spread", "effective_spread", "realized_spread"), expected):
        np.testing.assert_allclose(getattr(result, field), values, rtol=rtol, atol=atol, equal_nan=True)


def test_update_matches_spread_feature_reference():
    import rtta

    bid = np.asarray([9.95, 10.05, 9.95, 10.25, 10.15], dtype=np.float64)
    ask = bid + 0.10
    trade = np.asarray([10.05, 10.05, 9.95, 10.35, 10.15], dtype=np.float64)
    expected = _reference_spread_features(trade, bid, ask, realized_horizon=2)

    indicator = rtta.SpreadFeatures(realized_horizon=2)
    actual = [indicator.update(float(t), float(b), float(a)) for t, b, a in zip(trade, bid, ask)]

    for index, result in enumerate(actual):
        assert result.quoted_spread == pytest.approx(expected[0][index])
        assert result.effective_spread == pytest.approx(expected[1][index])
        assert result.realized_spread == pytest.approx(expected[2][index])
    assert indicator.last().realized_spread == pytest.approx(expected[2][-1])


def test_batch_records_pandas_and_float32_match_on_realistic_512_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(909)
    midpoint = 100.0 + np.cumsum(rng.normal(0.0, 0.45, 512))
    spread = rng.uniform(0.01, 0.05, 512)
    direction = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=512)
    arrays = {
        "trade_price": np.ascontiguousarray(midpoint + 0.5 * direction * spread, dtype=np.float64),
        "bid_price": np.ascontiguousarray(midpoint - 0.5 * spread, dtype=np.float64),
        "ask_price": np.ascontiguousarray(midpoint + 0.5 * spread, dtype=np.float64),
    }
    expected = _reference_spread_features(
        arrays["trade_price"],
        arrays["bid_price"],
        arrays["ask_price"],
        realized_horizon=4,
    )
    records = [{name: float(values[i]) for name, values in arrays.items()} for i in range(len(midpoint))]
    table = pandas.DataFrame(arrays, copy=False)
    arrays32 = {name: np.ascontiguousarray(values.astype(np.float32)) for name, values in arrays.items()}

    _assert_result_close(rtta.SpreadFeatures(realized_horizon=4).batch(arrays["trade_price"], arrays["bid_price"], arrays["ask_price"]), expected)
    _assert_result_close(rtta.SpreadFeatures(realized_horizon=4).batch(records), expected)
    _assert_result_close(rtta.SpreadFeatures(realized_horizon=4).batch(table), expected)
    _assert_result_close(
        rtta.SpreadFeatures(realized_horizon=4).batch(arrays32["trade_price"], arrays32["bid_price"], arrays32["ask_price"]),
        _reference_spread_features(
            arrays32["trade_price"],
            arrays32["bid_price"],
            arrays32["ask_price"],
            realized_horizon=4,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_advance_last_scalar_accessors_replay_and_fillna():
    import rtta

    bid = np.ascontiguousarray([9.95, 10.05, 9.95, 10.25], dtype=np.float64)
    ask = np.ascontiguousarray(bid + 0.10, dtype=np.float64)
    trade = np.ascontiguousarray([10.05, 10.05, 9.95, 10.35], dtype=np.float64)
    update_indicator = rtta.SpreadFeatures(realized_horizon=2)
    update_results = [update_indicator.update(float(t), float(b), float(a)) for t, b, a in zip(trade, bid, ask)]
    advance_indicator = rtta.SpreadFeatures(realized_horizon=2)

    for args, result in zip(zip(trade, bid, ask), update_results):
        assert advance_indicator.advance(*(float(value) for value in args)) is None
        assert advance_indicator.last().effective_spread == pytest.approx(result.effective_spread)
        assert advance_indicator.last_quoted_spread() == pytest.approx(result.quoted_spread)
        assert advance_indicator.last_effective_spread() == pytest.approx(result.effective_spread)
        assert advance_indicator.last_realized_spread() == pytest.approx(result.realized_spread)

    replay = rtta.SpreadFeatures(realized_horizon=2).replay_update_outputs(trade, bid, ask)
    np.testing.assert_allclose(replay.realized_spread, [result.realized_spread for result in update_results], rtol=1e-12, atol=1e-12)
    checksum = sum(result.quoted_spread + result.effective_spread + result.realized_spread for result in update_results)
    assert rtta.SpreadFeatures(realized_horizon=2).replay_update(trade, bid, ask) == pytest.approx(checksum)
    assert rtta.SpreadFeatures(realized_horizon=2).replay_advance(trade, bid, ask) == pytest.approx(checksum)

    out = rtta.SpreadFeatures(realized_horizon=2, fillna=False).batch(trade, bid, ask)
    assert np.isnan(out.realized_spread[:2]).all()
    assert np.isfinite(out.realized_spread[-1])

    with pytest.raises(ValueError):
        rtta.SpreadFeatures(realized_horizon=-1)
