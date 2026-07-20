"""FlowPressureCapacitySignal event-time microstructure tests."""

from __future__ import annotations

import math

import numpy as np
import pytest


FIELDS = (
    "signal",
    "score",
    "fair_value",
    "microprice",
    "raw_queue_imbalance",
    "queue_imbalance",
    "flow_imbalance",
    "pressure",
    "replenishment",
    "fragility",
    "spread_bps",
)


def _assert_result_close(actual, expected, *, rel=1e-11, abs_=1e-12):
    for field in FIELDS:
        left = float(getattr(actual, field))
        right = float(getattr(expected, field))
        if math.isnan(left) or math.isnan(right):
            assert math.isnan(left) and math.isnan(right), field
        else:
            assert left == pytest.approx(right, rel=rel, abs=abs_), field


def test_first_quote_reports_weighted_microprice_and_queue_state():
    import rtta

    out = rtta.FlowPressureCapacitySignal(warmup=1).update(
        100.00, 300.0, 100.02, 100.0, 0.0
    )

    assert out.raw_queue_imbalance == pytest.approx(0.5)
    assert out.queue_imbalance == pytest.approx(0.5)
    assert out.microprice == pytest.approx(100.015)
    assert 100.00 <= out.fair_value <= 100.02
    assert out.flow_imbalance == 0.0
    assert out.spread_bps == pytest.approx(0.02 / 100.01 * 10_000.0)


def test_replenishment_separates_absorbed_flow_from_queue_depletion():
    import rtta

    absorbed = rtta.FlowPressureCapacitySignal(warmup=1)
    absorbed.update(100.00, 100.0, 100.01, 100.0, 0.0)
    absorbed_out = absorbed.update(100.00, 100.0, 100.01, 100.0, 100.0, 0.0)

    depleted = rtta.FlowPressureCapacitySignal(warmup=1)
    depleted.update(100.00, 100.0, 100.01, 100.0, 0.0)
    depleted_out = depleted.update(100.00, 100.0, 100.02, 100.0, 100.0, 0.0)

    # The same aggressive buy flow is almost neutral when the ask fully refills,
    # but strongly bullish when the old ask level disappears.
    assert absorbed_out.replenishment < -0.45
    assert abs(absorbed_out.score) < 0.05
    assert absorbed_out.signal == 0.0
    assert depleted_out.replenishment == pytest.approx(0.0)
    assert depleted_out.fragility > 0.45
    assert depleted_out.score > 0.55
    assert depleted_out.signal == 1.0


def test_sell_side_depletion_is_the_exact_directional_mirror():
    import rtta

    buy = rtta.FlowPressureCapacitySignal(warmup=1)
    sell = rtta.FlowPressureCapacitySignal(warmup=1)
    buy.update(100.00, 100.0, 100.01, 100.0, 0.0, 0.0)
    sell.update(100.00, 100.0, 100.01, 100.0, 0.0, 0.0)

    bullish = buy.update(100.00, 100.0, 100.02, 100.0, 100.0, 0.0)
    bearish = sell.update(99.99, 100.0, 100.01, 100.0, 0.0, 100.0)

    assert bearish.score == pytest.approx(-bullish.score)
    assert bearish.pressure == pytest.approx(-bullish.pressure)
    assert bearish.replenishment == pytest.approx(-bullish.replenishment)
    assert bearish.fragility == pytest.approx(-bullish.fragility)
    assert bearish.signal == -bullish.signal


def test_bid_ask_reflection_symmetry_holds_over_a_path():
    import rtta

    center_twice = 200.10
    path = (
        (100.00, 200.0, 100.02, 100.0, 0.0, 0.0),
        (100.00, 150.0, 100.02, 80.0, 30.0, 10.0),
        (100.01, 220.0, 100.02, 50.0, 50.0, 5.0),
        (100.01, 200.0, 100.03, 90.0, 80.0, 20.0),
    )
    direct = rtta.FlowPressureCapacitySignal(warmup=1)
    reflected = rtta.FlowPressureCapacitySignal(warmup=1)

    for bid, bid_size, ask, ask_size, buys, sells in path:
        left = direct.update(bid, bid_size, ask, ask_size, buys, sells)
        right = reflected.update(
            center_twice - ask,
            ask_size,
            center_twice - bid,
            bid_size,
            sells,
            buys,
        )
        assert right.signal == -left.signal
        for field in (
            "score",
            "raw_queue_imbalance",
            "queue_imbalance",
            "flow_imbalance",
            "pressure",
            "replenishment",
            "fragility",
        ):
            assert getattr(right, field) == pytest.approx(-getattr(left, field)), field
        assert right.microprice == pytest.approx(center_twice - left.microprice)
        assert right.fair_value == pytest.approx(center_twice - left.fair_value)
        reflected_mid = center_twice - 0.5 * (bid + ask)
        assert right.spread_bps == pytest.approx((ask - bid) / reflected_mid * 10_000.0)


def test_compact_signed_flow_matches_detailed_one_sided_flow():
    import rtta

    compact = rtta.FlowPressureCapacitySignal(warmup=1)
    detailed = rtta.FlowPressureCapacitySignal(warmup=1)
    path = (
        (100.00, 100.0, 100.01, 100.0, 0.0),
        (100.00, 100.0, 100.02, 100.0, 50.0),
        (100.00, 80.0, 100.02, 120.0, -25.0),
        (100.01, 140.0, 100.02, 90.0, 10.0),
    )
    for bid, bid_size, ask, ask_size, signed in path:
        left = compact.update(bid, bid_size, ask, ask_size, signed)
        right = detailed.update(
            bid, bid_size, ask, ask_size, max(signed, 0.0), max(-signed, 0.0)
        )
        _assert_result_close(left, right)


def test_event_time_filter_damps_a_one_update_queue_flip():
    import rtta

    indicator = rtta.FlowPressureCapacitySignal(
        half_life_updates=16.0, warmup=1, fragility_weight=0.0
    )
    indicator.update(100.00, 900.0, 100.01, 100.0, 0.0)
    out = indicator.update(100.00, 100.0, 100.01, 900.0, 0.0)

    assert out.raw_queue_imbalance == pytest.approx(-0.8)
    assert out.queue_imbalance > 0.65
    assert abs(out.queue_imbalance) < 0.8


def test_flow_pressure_decays_without_new_aggressive_trades():
    import rtta

    indicator = rtta.FlowPressureCapacitySignal(
        half_life_updates=2.0,
        replenishment_weight=0.0,
        fragility_weight=0.0,
        warmup=1,
    )
    indicator.update(100.00, 100.0, 100.01, 100.0, 0.0)
    first = indicator.update(100.00, 100.0, 100.01, 100.0, 100.0, 0.0)
    later = None
    for _ in range(12):
        later = indicator.update(100.00, 100.0, 100.01, 100.0, 0.0, 0.0)

    assert later is not None
    assert first.pressure > 0.3
    assert 0.0 < later.pressure < first.pressure * 0.1


def test_batch_and_streaming_match_for_compact_and_detailed_paths():
    import rtta

    bid = np.array([100.00, 100.00, 100.00, 100.01, 100.01], dtype=np.float64)
    bid_size = np.array([100, 100, 80, 150, 120], dtype=np.float64)
    ask = np.array([100.01, 100.02, 100.02, 100.02, 100.03], dtype=np.float64)
    ask_size = np.array([100, 100, 120, 90, 100], dtype=np.float64)
    signed = np.array([0, 50, -25, 20, 40], dtype=np.float64)

    compact_batch = rtta.FlowPressureCapacitySignal(warmup=1).batch(
        bid, bid_size, ask, ask_size, signed
    )
    compact_stream = rtta.FlowPressureCapacitySignal(warmup=1)
    compact_results = [
        compact_stream.update(*map(float, values))
        for values in zip(bid, bid_size, ask, ask_size, signed)
    ]
    for field in FIELDS:
        assert np.asarray(getattr(compact_batch, field)) == pytest.approx(
            [getattr(result, field) for result in compact_results]
        )

    buys = np.maximum(signed, 0.0) + np.array([0, 0, 5, 3, 0], dtype=np.float64)
    sells = np.maximum(-signed, 0.0) + np.array([0, 0, 5, 3, 0], dtype=np.float64)
    detailed_batch = rtta.FlowPressureCapacitySignal(warmup=1).batch(
        bid, bid_size, ask, ask_size, buys, sells
    )
    detailed_stream = rtta.FlowPressureCapacitySignal(warmup=1)
    detailed_results = [
        detailed_stream.update(*map(float, values))
        for values in zip(bid, bid_size, ask, ask_size, buys, sells)
    ]
    for field in FIELDS:
        assert np.asarray(getattr(detailed_batch, field)) == pytest.approx(
            [getattr(result, field) for result in detailed_results]
        )


def test_float32_batch_is_supported_and_close_to_float64():
    import rtta

    arrays64 = (
        np.array([100.00, 100.00, 100.01], dtype=np.float64),
        np.array([100, 90, 140], dtype=np.float64),
        np.array([100.01, 100.02, 100.02], dtype=np.float64),
        np.array([100, 110, 80], dtype=np.float64),
        np.array([0, 50, -20], dtype=np.float64),
    )
    result64 = rtta.FlowPressureCapacitySignal(warmup=1).batch(*arrays64)
    result32 = rtta.FlowPressureCapacitySignal(warmup=1).batch(
        *(values.astype(np.float32) for values in arrays64)
    )
    for field in FIELDS:
        np.testing.assert_allclose(
            getattr(result32, field), getattr(result64, field), rtol=1e-3, atol=1e-3
        )


def test_advance_last_reset_and_warmup_contract():
    import rtta

    kwargs = dict(warmup=3, fillna=False)
    updated = rtta.FlowPressureCapacitySignal(**kwargs)
    advanced = rtta.FlowPressureCapacitySignal(**kwargs)
    path = (
        (100.00, 100.0, 100.01, 100.0, 0.0),
        (100.00, 100.0, 100.02, 100.0, 50.0),
        (100.01, 150.0, 100.02, 80.0, 25.0),
    )
    for index, values in enumerate(path):
        result = updated.update(*values)
        assert advanced.advance(*values) is None
        _assert_result_close(advanced.last(), result)
        if index < 2:
            assert all(math.isnan(getattr(result, field)) for field in FIELDS)

    advanced.reset()
    fresh = rtta.FlowPressureCapacitySignal(**kwargs)
    _assert_result_close(advanced.update(*path[0]), fresh.update(*path[0]))


def test_invalid_quotes_do_not_contaminate_the_next_valid_transition():
    import rtta

    indicator = rtta.FlowPressureCapacitySignal(warmup=1, fillna=False)
    assert math.isnan(indicator.update(100.01, 100, 100.00, 100, 0).score)
    first_valid = indicator.update(100.00, 100, 100.01, 100, 0)
    fresh = rtta.FlowPressureCapacitySignal(warmup=1, fillna=False).update(
        100.00, 100, 100.01, 100, 0
    )
    _assert_result_close(first_valid, fresh)


def test_batch_rejects_mismatched_lengths():
    import rtta

    two = np.ones(2, dtype=np.float64)
    three = np.ones(3, dtype=np.float64)
    with pytest.raises(ValueError, match="same length"):
        rtta.FlowPressureCapacitySignal().batch(two, two, two + 1.0, two, three)
