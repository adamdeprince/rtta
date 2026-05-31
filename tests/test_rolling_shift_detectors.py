import math

import numpy as np
import pytest


class _TwoWindowStats:
    def __init__(self, window):
        self.window = window
        self.reference = []
        self.recent = []

    def push(self, value):
        if len(self.recent) == self.window:
            moving = self.recent.pop(0)
            if len(self.reference) == self.window:
                self.reference.pop(0)
            self.reference.append(moving)
        self.recent.append(float(value))

    def ready(self):
        return len(self.reference) == self.window and len(self.recent) == self.window

    @staticmethod
    def mean(values):
        return sum(values) / len(values)

    @staticmethod
    def variance(values):
        mean = _TwoWindowStats.mean(values)
        value = sum((x - mean) ** 2 for x in values) / len(values)
        return 0.0 if value < 0.0 and value > -1.0e-12 else value


class _TwoWindowPairStats:
    def __init__(self, window):
        self.window = window
        self.reference = []
        self.recent = []

    def push(self, x, y):
        if len(self.recent) == self.window:
            moving = self.recent.pop(0)
            if len(self.reference) == self.window:
                self.reference.pop(0)
            self.reference.append(moving)
        self.recent.append((float(x), float(y)))

    def ready(self):
        return len(self.reference) == self.window and len(self.recent) == self.window

    @staticmethod
    def correlation(values):
        x = np.asarray([value[0] for value in values], dtype=np.float64)
        y = np.asarray([value[1] for value in values], dtype=np.float64)
        numerator = len(values) * float(np.dot(x, y)) - float(x.sum() * y.sum())
        denominator = math.sqrt(
            (len(values) * float(np.dot(x, x)) - float(x.sum() * x.sum())) *
            (len(values) * float(np.dot(y, y)) - float(y.sum() * y.sum()))
        )
        return 0.0 if denominator == 0.0 else numerator / denominator

    @staticmethod
    def beta(values):
        x = np.asarray([value[0] for value in values], dtype=np.float64)
        y = np.asarray([value[1] for value in values], dtype=np.float64)
        covariance = len(values) * float(np.dot(x, y)) - float(x.sum() * y.sum())
        variance_y = len(values) * float(np.dot(y, y)) - float(y.sum() * y.sum())
        return 0.0 if variance_y == 0.0 else covariance / variance_y


def _rolling_mean_shift(close, window=4, threshold=2.0, variance_floor=1.0e-4):
    stats = _TwoWindowStats(window)
    output = []
    for value in close:
        stats.push(value)
        if not stats.ready():
            output.append(0.0)
            continue
        reference_variance = stats.variance(stats.reference)
        recent_variance = stats.variance(stats.recent)
        denominator = math.sqrt(reference_variance / window + recent_variance / window + variance_floor)
        z_score = (stats.mean(stats.recent) - stats.mean(stats.reference)) / denominator
        output.append(1.0 if z_score > threshold else -1.0 if z_score < -threshold else 0.0)
    return np.asarray(output, dtype=np.float64)


def _rolling_variance_shift(close, window=4, threshold=1.0, variance_floor=1.0e-12):
    stats = _TwoWindowStats(window)
    output = []
    for value in close:
        stats.push(value)
        if not stats.ready():
            output.append(0.0)
            continue
        log_ratio = math.log((stats.variance(stats.recent) + variance_floor) / (stats.variance(stats.reference) + variance_floor))
        output.append(1.0 if log_ratio > threshold else -1.0 if log_ratio < -threshold else 0.0)
    return np.asarray(output, dtype=np.float64)


def _rolling_mean_variance_shift(close, window=4, threshold=2.0, variance_weight=1.0, variance_floor=1.0e-4):
    stats = _TwoWindowStats(window)
    output = []
    for value in close:
        stats.push(value)
        if not stats.ready():
            output.append(0.0)
            continue
        reference_variance = stats.variance(stats.reference)
        recent_variance = stats.variance(stats.recent)
        mean_denominator = math.sqrt(reference_variance / window + recent_variance / window + variance_floor)
        mean_score = (stats.mean(stats.recent) - stats.mean(stats.reference)) / mean_denominator
        variance_score = math.log((recent_variance + variance_floor) / (reference_variance + variance_floor))
        weighted_variance_score = math.sqrt(variance_weight) * variance_score
        score = math.sqrt(mean_score * mean_score + weighted_variance_score * weighted_variance_score)
        if score <= threshold:
            output.append(0.0)
        else:
            direction = mean_score if abs(mean_score) >= abs(weighted_variance_score) else weighted_variance_score
            output.append(1.0 if direction >= 0.0 else -1.0)
    return np.asarray(output, dtype=np.float64)


def _rolling_correlation_shift(real0, real1, window=4, threshold=0.5):
    stats = _TwoWindowPairStats(window)
    output = []
    for x, y in zip(real0, real1):
        stats.push(x, y)
        if not stats.ready():
            output.append(0.0)
            continue
        difference = stats.correlation(stats.recent) - stats.correlation(stats.reference)
        output.append(1.0 if difference > threshold else -1.0 if difference < -threshold else 0.0)
    return np.asarray(output, dtype=np.float64)


def _rolling_beta_shift(real0, real1, window=4, threshold=0.5):
    stats = _TwoWindowPairStats(window)
    output = []
    for x, y in zip(real0, real1):
        stats.push(x, y)
        if not stats.ready():
            output.append(0.0)
            continue
        difference = stats.beta(stats.recent) - stats.beta(stats.reference)
        output.append(1.0 if difference > threshold else -1.0 if difference < -threshold else 0.0)
    return np.asarray(output, dtype=np.float64)


def _rolling_spread_liquidity_shift(bid_price, bid_size, ask_price, ask_size, window=4, threshold=1.0e-5, depth_floor=1.0e-12):
    stress = []
    for bid, bid_depth, ask, ask_depth in zip(bid_price, bid_size, ask_price, ask_size):
        quoted_spread = max(float(ask) - float(bid), 0.0)
        depth = max(float(bid_depth), 0.0) + max(float(ask_depth), 0.0)
        stress.append(quoted_spread / max(depth, depth_floor))

    stats = _TwoWindowStats(window)
    output = []
    for value in stress:
        stats.push(value)
        if not stats.ready():
            output.append(0.0)
            continue
        difference = stats.mean(stats.recent) - stats.mean(stats.reference)
        output.append(1.0 if difference > threshold else -1.0 if difference < -threshold else 0.0)
    return np.asarray(output, dtype=np.float64)


def _threshold_regime(values, upper_entry=1.0, upper_exit=0.5, lower_entry=-1.0, lower_exit=-0.5):
    state = 0.0
    output = []
    for value in values:
        if state > 0.0:
            if value <= lower_entry:
                state = -1.0
            elif value <= upper_exit:
                state = 0.0
        elif state < 0.0:
            if value >= upper_entry:
                state = 1.0
            elif value >= lower_exit:
                state = 0.0
        elif value >= upper_entry:
            state = 1.0
        elif value <= lower_entry:
            state = -1.0
        output.append(state)
    return np.asarray(output, dtype=np.float64)


def _market():
    close = np.ascontiguousarray([
        0.0, 0.1, -0.1, 0.0,
        3.0, 3.1, 2.9, 3.0,
        -3.0, -3.1, -2.9, -3.0,
    ])
    variance_close = np.ascontiguousarray([
        0.0, 0.1, -0.1, 0.0,
        2.0, -2.0, 2.0, -2.0,
        0.0, 0.1, -0.1, 0.0,
    ])
    corr_x = np.ascontiguousarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float64)
    corr_y = np.ascontiguousarray([1, 2, 3, 4, -5, -6, -7, -8, 9, 10, 11, 12], dtype=np.float64)
    beta_y = np.ascontiguousarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float64)
    beta_x = np.ascontiguousarray([1, 2, 3, 4, 10, 12, 14, 16, -9, -10, -11, -12], dtype=np.float64)
    bid_price = np.ascontiguousarray([100.0] * 12)
    ask_price = np.ascontiguousarray([100.01] * 4 + [100.05] * 4 + [100.01] * 4)
    bid_size = np.ascontiguousarray([10_000.0] * 4 + [100.0] * 4 + [10_000.0] * 4)
    ask_size = np.ascontiguousarray([10_000.0] * 4 + [100.0] * 4 + [10_000.0] * 4)
    regime = np.ascontiguousarray([0.0, 1.2, 0.8, 0.4, -1.2, -0.8, -0.4, 1.2])
    return close, variance_close, corr_x, corr_y, beta_x, beta_y, bid_price, bid_size, ask_price, ask_size, regime


CASES = (
    ("RollingMeanShiftDetector", ("close",), {"window": 4, "threshold": 2.0, "variance_floor": 1.0e-4}, _rolling_mean_shift),
    ("RollingVarianceShiftDetector", ("variance_close",), {"window": 4, "threshold": 1.0}, _rolling_variance_shift),
    ("RollingMeanVarianceShiftDetector", ("close",), {"window": 4, "threshold": 2.0, "variance_floor": 1.0e-4}, _rolling_mean_variance_shift),
    ("RollingCorrelationShiftDetector", ("corr_x", "corr_y"), {"window": 4, "threshold": 0.5}, _rolling_correlation_shift),
    ("RollingBetaShiftDetector", ("beta_x", "beta_y"), {"window": 4, "threshold": 0.5}, _rolling_beta_shift),
    ("RollingSpreadLiquidityShiftDetector", ("bid_price", "bid_size", "ask_price", "ask_size"), {"window": 4, "threshold": 1.0e-5}, _rolling_spread_liquidity_shift),
    ("ThresholdRegimeDetector", ("regime",), {}, _threshold_regime),
)


def _arrays():
    names = (
        "close", "variance_close", "corr_x", "corr_y", "beta_x", "beta_y",
        "bid_price", "bid_size", "ask_price", "ask_size", "regime",
    )
    return dict(zip(names, _market()))


def _records(arrays, fields):
    size = len(arrays[fields[0]])
    records = []
    for index in range(size):
        record = {field: float(arrays[field][index]) for field in fields}
        if "close" not in record and "variance_close" in record:
            record["close"] = record["variance_close"]
        if "real0" not in record and fields[0] in ("corr_x", "beta_x"):
            record["real0"] = record[fields[0]]
            record["real1"] = record[fields[1]]
        if "value" not in record and "regime" in record:
            record["value"] = record["regime"]
        records.append(record)
    return records


def _batch_args(arrays, fields, dtype=np.float64):
    return [np.ascontiguousarray(arrays[field].astype(dtype)) for field in fields]


@pytest.mark.parametrize("name,fields,kwargs,reference", CASES)
def test_update_matches_reference_and_last(name, fields, kwargs, reference):
    import rtta

    arrays = _arrays()
    indicator = getattr(rtta, name)(**kwargs)
    expected = reference(*[arrays[field] for field in fields], **kwargs)
    actual = []
    for args, expected_value in zip(zip(*[arrays[field] for field in fields]), expected):
        actual_value = indicator.update(*[float(value) for value in args])
        actual.append(actual_value)
        assert indicator.last() == expected_value

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    if name != "ThresholdRegimeDetector":
        assert np.any(expected == 1.0)
        assert np.any(expected == -1.0)


@pytest.mark.parametrize("name,fields,kwargs,reference", CASES)
def test_array_record_table_and_float32_batches_match_reference(name, fields, kwargs, reference):
    import rtta

    pandas = pytest.importorskip("pandas")
    arrays = _arrays()
    cls = getattr(rtta, name)
    expected = reference(*[arrays[field] for field in fields], **kwargs)

    np.testing.assert_allclose(cls(**kwargs).batch(*_batch_args(arrays, fields)), expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(cls(**kwargs).batch(_records(arrays, fields)), expected, rtol=0.0, atol=0.0)

    table = pandas.DataFrame(_records(arrays, fields), copy=False)
    np.testing.assert_allclose(cls(**kwargs).batch(table), expected, rtol=0.0, atol=0.0)

    args32 = _batch_args(arrays, fields, np.float32)
    expected32 = reference(*args32, **kwargs)
    np.testing.assert_allclose(cls(**kwargs).batch(*args32), expected32, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("name,fields,kwargs,reference", CASES)
def test_advance_last_and_replay_paths_match_update_checksum(name, fields, kwargs, reference):
    import rtta

    arrays = _arrays()
    cls = getattr(rtta, name)
    expected = cls(**kwargs).batch(*_batch_args(arrays, fields))

    advance_indicator = cls(**kwargs)
    for args, expected_value in zip(zip(*[arrays[field] for field in fields]), expected):
        assert advance_indicator.advance(*[float(value) for value in args]) is None
        assert advance_indicator.last() == expected_value

    batch_args = _batch_args(arrays, fields)
    assert cls(**kwargs).replay_update(*batch_args) == pytest.approx(float(expected.sum()))
    assert cls(**kwargs).replay_advance(*batch_args) == pytest.approx(float(expected.sum()))


def test_invalid_parameters_are_rejected():
    import rtta

    with pytest.raises(ValueError):
        rtta.RollingMeanShiftDetector(window=0)
    with pytest.raises(ValueError):
        rtta.RollingVarianceShiftDetector(threshold=0.0)
    with pytest.raises(ValueError):
        rtta.RollingMeanVarianceShiftDetector(variance_weight=-1.0)
    with pytest.raises(ValueError):
        rtta.RollingCorrelationShiftDetector(threshold=math.inf)
    with pytest.raises(ValueError):
        rtta.RollingBetaShiftDetector(window=-1)
    with pytest.raises(ValueError):
        rtta.RollingSpreadLiquidityShiftDetector(depth_floor=0.0)
    with pytest.raises(ValueError):
        rtta.ThresholdRegimeDetector(upper_entry=1.0, upper_exit=2.0, lower_entry=-1.0, lower_exit=-0.5)
