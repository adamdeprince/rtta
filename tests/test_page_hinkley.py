import math

import numpy as np
import pytest


def _page_hinkley(close, threshold=1.0, delta=0.0):
    mean = 0.0
    positive_sum = 0.0
    positive_min = 0.0
    negative_sum = 0.0
    negative_min = 0.0
    count = 0
    output = []

    for value in close:
        value = float(value)
        if count == 0:
            mean = value
            positive_sum = 0.0
            positive_min = 0.0
            negative_sum = 0.0
            negative_min = 0.0
            count = 1
            output.append(0.0)
            continue

        count += 1
        mean += (value - mean) / count

        positive_sum += value - mean - delta
        positive_min = min(positive_min, positive_sum)
        positive_score = positive_sum - positive_min

        negative_sum += mean - value - delta
        negative_min = min(negative_min, negative_sum)
        negative_score = negative_sum - negative_min

        if positive_score > threshold and positive_score >= negative_score:
            output.append(1.0)
            mean = value
            positive_sum = 0.0
            positive_min = 0.0
            negative_sum = 0.0
            negative_min = 0.0
            count = 1
        elif negative_score > threshold:
            output.append(-1.0)
            mean = value
            positive_sum = 0.0
            positive_min = 0.0
            negative_sum = 0.0
            negative_min = 0.0
            count = 1
        else:
            output.append(0.0)

    return np.asarray(output, dtype=np.float64)


def test_update_matches_causal_page_hinkley_filter_and_last():
    import rtta

    close = np.asarray([
        100.0,
        100.05,
        99.95,
        100.1,
        101.2,
        101.15,
        101.3,
        100.0,
        99.7,
        100.2,
    ])
    expected = _page_hinkley(close, threshold=0.75, delta=0.02)

    indicator = rtta.PageHinkley(threshold=0.75, delta=0.02)
    actual = []
    for value, expected_value in zip(close, expected):
        actual_value = indicator.update(float(value))
        actual.append(actual_value)
        assert indicator.last() == expected_value

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert np.any(expected == 1.0)
    assert np.any(expected == -1.0)


def test_batch_records_pandas_and_float32_match_reference():
    import rtta

    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(20260529)
    close = np.ascontiguousarray(
        100.0 + np.cumsum(rng.normal(0.0, 0.35, 512)),
        dtype=np.float64,
    )
    close32 = np.ascontiguousarray(close.astype(np.float32))
    records = [{"close": float(value)} for value in close]
    table = pandas.DataFrame({"close": close}, copy=False)

    np.testing.assert_allclose(
        rtta.PageHinkley(threshold=1.2, delta=0.02).batch(close),
        _page_hinkley(close, threshold=1.2, delta=0.02),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rtta.PageHinkley(threshold=1.2, delta=0.02).batch(records),
        _page_hinkley(close, threshold=1.2, delta=0.02),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rtta.PageHinkley(threshold=1.2, delta=0.02).batch(table),
        _page_hinkley(close, threshold=1.2, delta=0.02),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rtta.PageHinkley(threshold=1.2, delta=0.02).batch(close32),
        _page_hinkley(close32, threshold=1.2, delta=0.02),
        rtol=0.0,
        atol=0.0,
    )


def test_advance_last_and_replay_paths_match_update_checksum():
    import rtta

    close = np.ascontiguousarray(
        [100.0, 100.1, 100.2, 101.1, 101.2, 100.2, 99.8],
        dtype=np.float64,
    )
    expected = rtta.PageHinkley(threshold=0.65, delta=0.01).batch(close)

    advance_indicator = rtta.PageHinkley(threshold=0.65, delta=0.01)
    for value, expected_value in zip(close, expected):
        assert advance_indicator.advance(float(value)) is None
        assert advance_indicator.last() == expected_value

    assert rtta.PageHinkley(threshold=0.65, delta=0.01).replay_update(close) == pytest.approx(
        float(expected.sum())
    )
    assert rtta.PageHinkley(threshold=0.65, delta=0.01).replay_advance(close) == pytest.approx(
        float(expected.sum())
    )


def test_invalid_parameters_are_rejected():
    import rtta

    with pytest.raises(ValueError):
        rtta.PageHinkley(threshold=0.0)
    with pytest.raises(ValueError):
        rtta.PageHinkley(threshold=math.inf)
    with pytest.raises(ValueError):
        rtta.PageHinkley(delta=-0.1)
    with pytest.raises(ValueError):
        rtta.PageHinkley(delta=math.nan)
