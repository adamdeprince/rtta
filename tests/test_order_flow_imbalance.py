import math

import numpy as np
import pytest


def _event_ofi(bid_price, bid_size, ask_price, ask_size):
    bid_price = np.asarray(bid_price, dtype=np.float64)
    bid_size = np.asarray(bid_size, dtype=np.float64)
    ask_price = np.asarray(ask_price, dtype=np.float64)
    ask_size = np.asarray(ask_size, dtype=np.float64)
    output = np.zeros_like(bid_price, dtype=np.float64)
    for i in range(1, len(output)):
        event = 0.0
        if bid_price[i] >= bid_price[i - 1]:
            event += bid_size[i]
        if bid_price[i] <= bid_price[i - 1]:
            event -= bid_size[i - 1]
        if ask_price[i] <= ask_price[i - 1]:
            event -= ask_size[i]
        if ask_price[i] >= ask_price[i - 1]:
            event += ask_size[i - 1]
        output[i] = event
    return output


def _rolling_sum(values, window, fillna=True):
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    for i in range(len(values)):
        output[i] = values[max(0, i + 1 - window) : i + 1].sum()
        if not fillna and i + 1 < window:
            output[i] = np.nan
    return output


def test_update_matches_cont_order_flow_imbalance_formula():
    import rtta

    bid_price = np.asarray([100.00, 100.00, 100.01, 99.99, 99.99, 100.02], dtype=np.float64)
    bid_size = np.asarray([10.0, 15.0, 12.0, 9.0, 8.0, 16.0], dtype=np.float64)
    ask_price = np.asarray([100.02, 100.02, 100.03, 100.01, 100.01, 100.04], dtype=np.float64)
    ask_size = np.asarray([11.0, 7.0, 6.0, 13.0, 12.0, 14.0], dtype=np.float64)
    expected = _rolling_sum(_event_ofi(bid_price, bid_size, ask_price, ask_size), 3)

    indicator = rtta.OrderFlowImbalance(window=3)
    actual = [
        indicator.update(float(bp), float(bs), float(ap), float(az))
        for bp, bs, ap, az in zip(bid_price, bid_size, ask_price, ask_size)
    ]

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert indicator.last() == expected[-1]


def test_fillna_false_waits_for_the_rolling_window():
    import rtta

    bid_price = np.asarray([100.00, 100.01, 100.01, 100.02], dtype=np.float64)
    bid_size = np.asarray([10.0, 11.0, 9.0, 12.0], dtype=np.float64)
    ask_price = np.asarray([100.03, 100.04, 100.02, 100.05], dtype=np.float64)
    ask_size = np.asarray([13.0, 14.0, 10.0, 15.0], dtype=np.float64)
    expected = _rolling_sum(_event_ofi(bid_price, bid_size, ask_price, ask_size), 3, fillna=False)

    actual = rtta.OrderFlowImbalance(window=3, fillna=False).batch(bid_price, bid_size, ask_price, ask_size)

    assert math.isnan(actual[0])
    assert math.isnan(actual[1])
    np.testing.assert_allclose(actual[2:], expected[2:], rtol=0.0, atol=0.0)


def test_batch_records_pandas_and_float32_match_on_realistic_512_quote_sequence():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(123)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.04, 512))
    spread = rng.uniform(0.01, 0.04, 512)
    arrays = {
        "bid_price": np.ascontiguousarray(close - 0.5 * spread, dtype=np.float64),
        "bid_size": np.ascontiguousarray(rng.integers(100, 10_000, 512).astype(np.float64)),
        "ask_price": np.ascontiguousarray(close + 0.5 * spread, dtype=np.float64),
        "ask_size": np.ascontiguousarray(rng.integers(100, 10_000, 512).astype(np.float64)),
    }
    expected = _rolling_sum(
        _event_ofi(arrays["bid_price"], arrays["bid_size"], arrays["ask_price"], arrays["ask_size"]),
        20,
    )

    batch = rtta.OrderFlowImbalance(window=20).batch(
        arrays["bid_price"],
        arrays["bid_size"],
        arrays["ask_price"],
        arrays["ask_size"],
    )
    records = [
        {name: float(values[i]) for name, values in arrays.items()}
        for i in range(len(arrays["bid_price"]))
    ]
    table = pandas.DataFrame(arrays, copy=False)
    arrays32 = {name: np.ascontiguousarray(values.astype(np.float32)) for name, values in arrays.items()}

    np.testing.assert_allclose(batch, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(rtta.OrderFlowImbalance(window=20).batch(records), expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(rtta.OrderFlowImbalance(window=20).batch(table), expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        rtta.OrderFlowImbalance(window=20).batch(
            arrays32["bid_price"],
            arrays32["bid_size"],
            arrays32["ask_price"],
            arrays32["ask_size"],
        ),
        _rolling_sum(_event_ofi(arrays32["bid_price"], arrays32["bid_size"], arrays32["ask_price"], arrays32["ask_size"]), 20),
        rtol=0.0,
        atol=0.0,
    )


def test_advance_last_and_replay_paths_match_update_checksum():
    import rtta

    bid_price = np.ascontiguousarray([100.0, 100.01, 100.00, 100.02], dtype=np.float64)
    bid_size = np.ascontiguousarray([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    ask_price = np.ascontiguousarray([100.03, 100.02, 100.04, 100.01], dtype=np.float64)
    ask_size = np.ascontiguousarray([14.0, 15.0, 16.0, 17.0], dtype=np.float64)

    update_indicator = rtta.OrderFlowImbalance(window=2)
    expected = [
        update_indicator.update(float(bp), float(bs), float(ap), float(az))
        for bp, bs, ap, az in zip(bid_price, bid_size, ask_price, ask_size)
    ]

    advance_indicator = rtta.OrderFlowImbalance(window=2)
    for bp, bs, ap, az, value in zip(bid_price, bid_size, ask_price, ask_size, expected):
        assert advance_indicator.advance(float(bp), float(bs), float(ap), float(az)) is None
        assert advance_indicator.last() == value

    assert rtta.OrderFlowImbalance(window=2).replay_update(bid_price, bid_size, ask_price, ask_size) == sum(expected)
    assert rtta.OrderFlowImbalance(window=2).replay_advance(bid_price, bid_size, ask_price, ask_size) == sum(expected)
