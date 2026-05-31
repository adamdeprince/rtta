import math

import numpy as np
import pytest


def _safe_divide(numerator, denominator):
    return 0.0 if denominator == 0.0 else numerator / denominator


def _true_range(close, high, low, previous_close):
    return max(high - low, abs(high - previous_close), abs(low - previous_close))


def _regime_hysteresis(values, upper_entry, upper_exit, lower_entry, lower_exit):
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


def _upper_hysteresis(values, entry, exit):
    state = 0.0
    output = []
    for value in values:
        if state > 0.0:
            if value <= exit:
                state = 0.0
        elif value >= entry:
            state = 1.0
        output.append(state)
    return np.asarray(output, dtype=np.float64)


def _regime_step(state, value, upper_entry, upper_exit, lower_entry, lower_exit):
    if state > 0.0:
        if value <= lower_entry:
            return -1.0
        if value <= upper_exit:
            return 0.0
    elif state < 0.0:
        if value >= upper_entry:
            return 1.0
        if value >= lower_exit:
            return 0.0
    elif value >= upper_entry:
        return 1.0
    elif value <= lower_entry:
        return -1.0
    return state


def _volatility_regime(close, alpha=0.5, high_entry=1.0, high_exit=0.4, low_entry=0.05, low_exit=0.1):
    previous_close = 0.0
    variance = 0.0
    has_previous = False
    state = 0.0
    output = []
    for value in close:
        value = float(value)
        if not has_previous:
            previous_close = value
            has_previous = True
            output.append(0.0)
            continue
        change = value - previous_close
        previous_close = value
        variance = (1.0 - alpha) * (variance + alpha * change * change)
        state = _regime_step(state, math.sqrt(variance), high_entry, high_exit, low_entry, low_exit)
        output.append(state)
    return np.asarray(output, dtype=np.float64)


def _atr_regime(close, high, low, window=4, high_entry=1.0, high_exit=0.5, low_entry=0.15, low_exit=0.25):
    previous_close = 0.0
    first = True
    count = 0
    tr_sum = 0.0
    atr = 0.0
    state = 0.0
    output = []
    for close_value, high_value, low_value in zip(close, high, low):
        close_value = float(close_value)
        high_value = float(high_value)
        low_value = float(low_value)
        tr = high_value - low_value if first else _true_range(close_value, high_value, low_value, previous_close)
        previous_close = close_value
        first = False
        count += 1

        if count <= window:
            tr_sum += tr
            atr = tr_sum / count
            if count < window:
                output.append(0.0)
                continue
        else:
            atr = (atr * (window - 1.0) + tr) / window

        if math.isfinite(atr):
            state = _regime_step(state, atr, high_entry, high_exit, low_entry, low_exit)
            output.append(state)
        else:
            output.append(0.0)
    return np.asarray(output, dtype=np.float64)


def _realized_variance_regime(close, window=4, high_entry=1.0, high_exit=0.5, low_entry=0.01, low_exit=0.05):
    values = []
    previous_close = 0.0
    has_previous = False
    state = 0.0
    output = []
    for close_value in close:
        close_value = float(close_value)
        change = close_value - previous_close if has_previous else 0.0
        previous_close = close_value
        has_previous = True
        if len(values) == window:
            values.pop(0)
        values.append(change * change)
        if len(values) == window:
            state = _regime_step(state, sum(values) / window, high_entry, high_exit, low_entry, low_exit)
            output.append(state)
        else:
            output.append(0.0)
    return np.asarray(output, dtype=np.float64)


def _trend_chop_regime(close, high, low, window=4, trend_entry=0.6, trend_exit=0.45, chop_entry=0.25, chop_exit=0.35):
    ranges = []
    closes = []
    previous_close = 0.0
    first = True
    state = 0.0
    output = []
    for close_value, high_value, low_value in zip(close, high, low):
        close_value = float(close_value)
        high_value = float(high_value)
        low_value = float(low_value)
        range_value = high_value - low_value if first else _true_range(close_value, high_value, low_value, previous_close)
        if len(ranges) == window:
            ranges.pop(0)
        ranges.append(max(range_value, 0.0))
        if len(closes) == window + 1:
            closes.pop(0)
        closes.append(close_value)
        previous_close = close_value
        first = False
        if len(ranges) == window and len(closes) == window + 1:
            state = _regime_step(
                state,
                _safe_divide(abs(close_value - closes[0]), sum(ranges)),
                trend_entry,
                trend_exit,
                chop_entry,
                chop_exit,
            )
            output.append(state)
        else:
            output.append(0.0)
    return np.asarray(output, dtype=np.float64)


def _liquidity_regime(
    close,
    volume,
    alpha=1.0,
    illiquid_entry=1.0e-6,
    illiquid_exit=1.0e-7,
    liquid_entry=1.0e-10,
    liquid_exit=5.0e-10,
    dollar_floor=1.0e-12,
):
    previous_close = 0.0
    ewma = 0.0
    has_previous = False
    metrics = []
    for close_value, volume_value in zip(close, volume):
        close_value = float(close_value)
        volume_value = float(volume_value)
        dollars = max(abs(close_value) * max(volume_value, 0.0), dollar_floor)
        price_return = abs(_safe_divide(close_value - previous_close, previous_close)) if has_previous else 0.0
        illiquidity = price_return / dollars
        ewma = alpha * illiquidity + (1.0 - alpha) * ewma if has_previous else illiquidity
        previous_close = close_value
        has_previous = True
        metrics.append(ewma)
    return _regime_hysteresis(metrics, illiquid_entry, illiquid_exit, liquid_entry, liquid_exit)


def _spread_regime(bid_price, ask_price, wide_entry=0.001, wide_exit=0.0005, tight_entry=0.0001, tight_exit=0.0002, mid_floor=1.0e-12):
    metrics = []
    for bid, ask in zip(bid_price, ask_price):
        mid = 0.5 * (float(bid) + float(ask))
        metrics.append(max(float(ask) - float(bid), 0.0) / max(abs(mid), mid_floor))
    return _regime_hysteresis(metrics, wide_entry, wide_exit, tight_entry, tight_exit)


def _relative_ewma_regime(values, alpha, high_entry, high_exit, low_entry, low_exit, value_floor):
    ewma = 0.0
    initialized = False
    metrics = []
    for value in values:
        positive = max(float(value), 0.0)
        if not initialized:
            ewma = positive
            initialized = True
            metrics.append((low_exit + high_exit) * 0.5)
            continue
        ratio = positive / max(ewma, value_floor)
        metrics.append(ratio)
        ewma = alpha * positive + (1.0 - alpha) * ewma
    return _regime_hysteresis(metrics, high_entry, high_exit, low_entry, low_exit)


def _volume_regime(volume, alpha=1.0, high_entry=3.0, high_exit=1.5, low_entry=0.4, low_exit=0.8, volume_floor=1.0e-12):
    return _relative_ewma_regime(volume, alpha, high_entry, high_exit, low_entry, low_exit, volume_floor)


def _trade_intensity_regime(transactions, alpha=1.0, high_entry=3.0, high_exit=1.5, low_entry=0.4, low_exit=0.8, intensity_floor=1.0e-12):
    return _relative_ewma_regime(transactions, alpha, high_entry, high_exit, low_entry, low_exit, intensity_floor)


def _order_flow_imbalance_regime(
    bid_price,
    bid_size,
    ask_price,
    ask_size,
    alpha=1.0,
    buy_entry=0.5,
    buy_exit=0.2,
    sell_entry=-0.5,
    sell_exit=-0.2,
    depth_floor=1.0e-12,
):
    previous = None
    ewma = 0.0
    metrics = []
    for bid, bid_depth, ask, ask_depth in zip(bid_price, bid_size, ask_price, ask_size):
        bid = float(bid)
        bid_depth = float(bid_depth)
        ask = float(ask)
        ask_depth = float(ask_depth)
        event = 0.0
        if previous is not None:
            prev_bid, prev_bid_depth, prev_ask, prev_ask_depth = previous
            if bid >= prev_bid:
                event += bid_depth
            if bid <= prev_bid:
                event -= prev_bid_depth
            if ask <= prev_ask:
                event -= ask_depth
            if ask >= prev_ask:
                event += prev_ask_depth
        depth = max(bid_depth, 0.0) + max(ask_depth, 0.0)
        normalized = event / max(depth, depth_floor)
        ewma = alpha * normalized + (1.0 - alpha) * ewma if previous is not None else 0.0
        previous = (bid, bid_depth, ask, ask_depth)
        metrics.append(ewma)
    return _regime_hysteresis(metrics, buy_entry, buy_exit, sell_entry, sell_exit)


def _rolling_correlation_regime(real0, real1, window=4, high_entry=0.8, high_exit=0.5, low_entry=-0.8, low_exit=-0.5):
    pairs = []
    metrics = []
    for x, y in zip(real0, real1):
        if len(pairs) == window:
            pairs.pop(0)
        pairs.append((float(x), float(y)))
        if len(pairs) < window:
            metrics.append(0.0)
            continue
        xs = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        ys = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        n = float(window)
        numerator = n * float(np.dot(xs, ys)) - float(xs.sum() * ys.sum())
        denominator = math.sqrt((n * float(np.dot(xs, xs)) - float(xs.sum() * xs.sum())) * (n * float(np.dot(ys, ys)) - float(ys.sum() * ys.sum())))
        metrics.append(_safe_divide(numerator, denominator))
    return _regime_hysteresis(metrics, high_entry, high_exit, low_entry, low_exit)


def _rolling_beta_regime(real0, real1, window=4, high_entry=1.5, high_exit=1.1, low_entry=-0.5, low_exit=0.0):
    pairs = []
    metrics = []
    for x, y in zip(real0, real1):
        if len(pairs) == window:
            pairs.pop(0)
        pairs.append((float(x), float(y)))
        if len(pairs) < window:
            metrics.append(0.0)
            continue
        xs = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        ys = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        n = float(window)
        covariance = n * float(np.dot(xs, ys)) - float(xs.sum() * ys.sum())
        variance_y = n * float(np.dot(ys, ys)) - float(ys.sum() * ys.sum())
        metrics.append(_safe_divide(covariance, variance_y))
    return _regime_hysteresis(metrics, high_entry, high_exit, low_entry, low_exit)


def _ewma_residual_metrics(real0, real1, alpha, min_variance, absolute):
    mean_x = 0.0
    mean_y = 0.0
    cov_xy = 0.0
    var_y = 0.0
    residual_mean = 0.0
    residual_variance = 0.0
    count = 0
    residual_count = 0
    metrics = []

    for x, y in zip(real0, real1):
        x = float(x)
        y = float(y)
        ready = count >= 3 and residual_count >= 2
        if ready:
            beta = _safe_divide(cov_xy, var_y)
            intercept = mean_x - beta * mean_y
            residual = x - (beta * y + intercept)
            z = (residual - residual_mean) / math.sqrt(max(residual_variance, min_variance))
            metrics.append(abs(z) if absolute else z)
        else:
            metrics.append(0.0)

        if count == 0:
            mean_x = x
            mean_y = y
            count = 1
            continue

        beta = _safe_divide(cov_xy, var_y)
        intercept = mean_x - beta * mean_y
        residual = x - (beta * y + intercept)
        if residual_count == 0:
            residual_mean = residual
            residual_variance = 0.0
            residual_count = 1
        else:
            residual_delta = residual - residual_mean
            residual_mean += alpha * residual_delta
            residual_variance = (1.0 - alpha) * (residual_variance + alpha * residual_delta * residual_delta)
            residual_count += 1

        dx = x - mean_x
        dy = y - mean_y
        mean_x += alpha * dx
        mean_y += alpha * dy
        cov_xy = (1.0 - alpha) * (cov_xy + alpha * dx * dy)
        var_y = (1.0 - alpha) * (var_y + alpha * dy * dy)
        count += 1
    return metrics


def _pairs_spread_regime(real0, real1, alpha=0.4, z_entry=1.0, z_exit=0.25, min_variance=1.0e-4):
    metrics = _ewma_residual_metrics(real0, real1, alpha, min_variance, absolute=False)
    return _regime_hysteresis(metrics, z_entry, z_exit, -z_entry, -z_exit)


def _cointegration_breakdown(real0, real1, alpha=0.4, z_entry=1.0, z_exit=0.25, min_variance=1.0e-4):
    metrics = _ewma_residual_metrics(real0, real1, alpha, min_variance, absolute=True)
    return _upper_hysteresis(metrics, z_entry, z_exit)


def _execution_cost_regime(trade_price, bid_price, ask_price, high_entry=0.001, high_exit=0.0005, low_entry=0.0001, low_exit=0.0002, mid_floor=1.0e-12):
    metrics = []
    for trade, bid, ask in zip(trade_price, bid_price, ask_price):
        mid = 0.5 * (float(bid) + float(ask))
        metrics.append(abs(float(trade) - mid) / max(abs(mid), mid_floor))
    return _regime_hysteresis(metrics, high_entry, high_exit, low_entry, low_exit)


def _arrays():
    close = np.ascontiguousarray([
        100.0, 100.01, 100.0, 100.02, 100.01,
        103.0, 106.0, 109.0, 109.05, 109.0, 109.02, 109.01,
    ])
    high = np.ascontiguousarray(close + np.asarray([0.05] * 5 + [2.0, 2.0, 2.0] + [0.05] * 4))
    low = np.ascontiguousarray(close - np.asarray([0.05] * 5 + [2.0, 2.0, 2.0] + [0.05] * 4))
    trend_close = np.ascontiguousarray([
        100.0, 100.1, 99.9, 100.05, 99.95, 100.0,
        101.0, 102.0, 103.0, 104.0, 105.0,
        105.1, 104.9, 105.05, 104.95, 105.0,
    ])
    trend_high = np.ascontiguousarray(trend_close + 0.05)
    trend_low = np.ascontiguousarray(trend_close - 0.05)
    liquidity_close = np.ascontiguousarray([100.0, 100.01, 100.02, 102.0, 104.0, 104.01, 104.02, 104.03, 104.04])
    volume = np.ascontiguousarray([1_000_000.0, 1_000_000.0, 1_000_000.0, 1.0, 1.0, 1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0])
    bid_price = np.ascontiguousarray([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    ask_price = np.ascontiguousarray([100.005, 100.005, 100.2, 100.2, 100.005, 100.005])
    relative_values = np.ascontiguousarray([100.0, 100.0, 20.0, 20.0, 100.0, 500.0, 500.0, 100.0, 30.0, 100.0])
    transactions = np.ascontiguousarray([50.0, 50.0, 10.0, 10.0, 50.0, 250.0, 250.0, 50.0, 15.0, 50.0])
    ofi_bid = np.ascontiguousarray([100.0, 101.0, 102.0, 102.0, 101.0, 100.0, 99.0, 99.0, 100.0])
    ofi_ask = np.ascontiguousarray([101.0, 102.0, 103.0, 103.0, 102.0, 101.0, 100.0, 100.0, 101.0])
    ofi_bid_size = np.ascontiguousarray([100.0] * len(ofi_bid))
    ofi_ask_size = np.ascontiguousarray([100.0] * len(ofi_bid))
    corr_x = np.ascontiguousarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float64)
    corr_y = np.ascontiguousarray([1, 2, 3, 4, -5, -6, -7, -8, 9, 10, 11, 12], dtype=np.float64)
    beta_y = np.ascontiguousarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float64)
    beta_x = np.ascontiguousarray([1, 2, 3, 4, 10, 12, 14, 16, -9, -10, -11, -12], dtype=np.float64)
    pair_y = np.ascontiguousarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.float64)
    residual = np.ascontiguousarray([0, 0, 0, 0, 0.1, 3.0, 3.0, -3.0, -3.0, 0.0, 0.0, 2.5, -2.5, 0.0, 0.0], dtype=np.float64)
    pair_x = np.ascontiguousarray(2.0 * pair_y + residual)
    trade_price = np.ascontiguousarray([100.0, 100.005, 100.25, 100.25, 100.005, 100.0])

    return {
        "close": close,
        "high": high,
        "low": low,
        "trend_close": trend_close,
        "trend_high": trend_high,
        "trend_low": trend_low,
        "liquidity_close": liquidity_close,
        "volume": volume,
        "bid_price": bid_price,
        "ask_price": ask_price,
        "relative_values": relative_values,
        "transactions": transactions,
        "ofi_bid": ofi_bid,
        "ofi_bid_size": ofi_bid_size,
        "ofi_ask": ofi_ask,
        "ofi_ask_size": ofi_ask_size,
        "corr_x": corr_x,
        "corr_y": corr_y,
        "beta_x": beta_x,
        "beta_y": beta_y,
        "pair_x": pair_x,
        "pair_y": pair_y,
        "trade_price": trade_price,
    }


CASES = (
    ("VolatilityRegimeDetector", ("close",), {"alpha": 0.5, "high_entry": 1.0, "high_exit": 0.4, "low_entry": 0.05, "low_exit": 0.1}, _volatility_regime),
    ("ATRRegimeDetector", ("close", "high", "low"), {"window": 4, "high_entry": 1.0, "high_exit": 0.5, "low_entry": 0.15, "low_exit": 0.25}, _atr_regime),
    ("RealizedVarianceRegimeDetector", ("close",), {"window": 4, "high_entry": 1.0, "high_exit": 0.5, "low_entry": 0.01, "low_exit": 0.05}, _realized_variance_regime),
    ("TrendChopRegimeDetector", ("trend_close", "trend_high", "trend_low"), {"window": 4, "trend_entry": 0.6, "trend_exit": 0.45, "chop_entry": 0.25, "chop_exit": 0.35}, _trend_chop_regime),
    ("LiquidityRegimeDetector", ("liquidity_close", "volume"), {"alpha": 1.0, "illiquid_entry": 1.0e-6, "illiquid_exit": 1.0e-7, "liquid_entry": 1.0e-10, "liquid_exit": 5.0e-10}, _liquidity_regime),
    ("SpreadRegimeDetector", ("bid_price", "ask_price"), {"wide_entry": 0.001, "wide_exit": 0.0005, "tight_entry": 0.0001, "tight_exit": 0.0002}, _spread_regime),
    ("VolumeRegimeDetector", ("relative_values",), {"alpha": 1.0, "high_entry": 3.0, "high_exit": 1.5, "low_entry": 0.4, "low_exit": 0.8}, _volume_regime),
    ("TradeIntensityRegimeDetector", ("transactions",), {"alpha": 1.0, "high_entry": 3.0, "high_exit": 1.5, "low_entry": 0.4, "low_exit": 0.8}, _trade_intensity_regime),
    ("OrderFlowImbalanceRegimeDetector", ("ofi_bid", "ofi_bid_size", "ofi_ask", "ofi_ask_size"), {"alpha": 1.0, "buy_entry": 0.5, "buy_exit": 0.2, "sell_entry": -0.5, "sell_exit": -0.2}, _order_flow_imbalance_regime),
    ("CorrelationRegimeDetector", ("corr_x", "corr_y"), {"window": 4, "high_entry": 0.8, "high_exit": 0.5, "low_entry": -0.8, "low_exit": -0.5}, _rolling_correlation_regime),
    ("BetaRegimeDetector", ("beta_x", "beta_y"), {"window": 4, "high_entry": 1.5, "high_exit": 1.1, "low_entry": -0.5, "low_exit": 0.0}, _rolling_beta_regime),
    ("PairsSpreadRegimeDetector", ("pair_x", "pair_y"), {"alpha": 0.4, "z_entry": 1.0, "z_exit": 0.25, "min_variance": 1.0e-4}, _pairs_spread_regime),
    ("CointegrationBreakdownMonitor", ("pair_x", "pair_y"), {"alpha": 0.4, "z_entry": 1.0, "z_exit": 0.25, "min_variance": 1.0e-4}, _cointegration_breakdown),
    ("ExecutionCostSlippageRegimeDetector", ("trade_price", "bid_price", "ask_price"), {"high_entry": 0.001, "high_exit": 0.0005, "low_entry": 0.0001, "low_exit": 0.0002}, _execution_cost_regime),
)


FIELD_ALIASES = {
    "trend_close": "close",
    "trend_high": "high",
    "trend_low": "low",
    "liquidity_close": "close",
    "relative_values": "volume",
    "ofi_bid": "bid_price",
    "ofi_bid_size": "bid_size",
    "ofi_ask": "ask_price",
    "ofi_ask_size": "ask_size",
    "corr_x": "real0",
    "corr_y": "real1",
    "beta_x": "real0",
    "beta_y": "real1",
    "pair_x": "real0",
    "pair_y": "real1",
}


def _records(arrays, fields):
    size = len(arrays[fields[0]])
    records = []
    for index in range(size):
        record = {}
        for field in fields:
            record[FIELD_ALIASES.get(field, field)] = float(arrays[field][index])
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
    assert np.any(expected != 0.0)


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
        rtta.VolatilityRegimeDetector(alpha=0.0)
    with pytest.raises(ValueError):
        rtta.ATRRegimeDetector(window=0)
    with pytest.raises(ValueError):
        rtta.RealizedVarianceRegimeDetector(window=-1)
    with pytest.raises(ValueError):
        rtta.TrendChopRegimeDetector(trend_entry=0.5, trend_exit=0.7, chop_entry=0.2, chop_exit=0.3)
    with pytest.raises(ValueError):
        rtta.LiquidityRegimeDetector(dollar_floor=0.0)
    with pytest.raises(ValueError):
        rtta.SpreadRegimeDetector(wide_entry=0.001, wide_exit=0.002, tight_entry=0.0001, tight_exit=0.0002)
    with pytest.raises(ValueError):
        rtta.VolumeRegimeDetector(volume_floor=0.0)
    with pytest.raises(ValueError):
        rtta.TradeIntensityRegimeDetector(intensity_floor=0.0)
    with pytest.raises(ValueError):
        rtta.OrderFlowImbalanceRegimeDetector(buy_entry=0.2, buy_exit=0.5, sell_entry=-0.5, sell_exit=-0.2)
    with pytest.raises(ValueError):
        rtta.CorrelationRegimeDetector(window=0)
    with pytest.raises(ValueError):
        rtta.BetaRegimeDetector(window=0)
    with pytest.raises(ValueError):
        rtta.PairsSpreadRegimeDetector(z_entry=1.0, z_exit=1.0)
    with pytest.raises(ValueError):
        rtta.CointegrationBreakdownMonitor(alpha=math.nan)
    with pytest.raises(ValueError):
        rtta.ExecutionCostSlippageRegimeDetector(high_entry=0.001, high_exit=0.002, low_entry=0.0001, low_exit=0.0002)
