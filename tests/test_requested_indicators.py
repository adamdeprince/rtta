import math

import numpy as np

import rtta


def _ema(values, window):
    alpha = 2.0 / (window + 1.0)
    out = []
    last = 0.0
    for index, value in enumerate(values):
        if index == 0:
            last = value
        last = alpha * value + (1.0 - alpha) * last
        out.append(last)
    return np.asarray(out)


def _wma(values, window):
    history = []
    out = []
    for value in values:
        history.append(value)
        history = history[-window:]
        weights = np.arange(1, len(history) + 1, dtype=np.float64)
        out.append(float(np.dot(history, weights) / weights.sum()))
    return np.asarray(out)


def _rsi(values, window):
    out = []
    prev = 0.0
    high = 0.0
    low = 0.0
    for counter, value in enumerate(values):
        retval = math.nan
        if counter == 0:
            prev = value
            retval = 50.0
        elif counter <= window:
            if value < prev:
                low = (low * (counter - 1) + prev - value) / counter
            elif value > prev:
                high = (high * (counter - 1) + value - prev) / counter
            if low == 0.0:
                retval = 100.0
            else:
                retval = 100.0 - (100.0 / (1.0 + (high / window) / (low / window)))
        else:
            if value < prev:
                low = (low * (window - 1) + prev - value) / window
            elif low == 0.0:
                retval = 100.0
            elif value > prev:
                high = (high * (window - 1) + value - prev) / window
            if math.isnan(retval):
                retval = 100.0 - (100.0 / (1.0 + (high / window) / (low / window)))
        prev = value
        out.append(retval)
    return np.asarray(out)


def _connors_rsi(close, rsi_window=3, streak_rsi_window=2, rank_window=100):
    streaks = []
    changes = []
    streak = 0.0
    prev = close[0]
    for index, value in enumerate(close):
        change = 0.0 if index == 0 else value - prev
        if index > 0:
            if change > 0.0:
                streak = streak + 1.0 if streak > 0.0 else 1.0
            elif change < 0.0:
                streak = streak - 1.0 if streak < 0.0 else -1.0
            else:
                streak = 0.0
        streaks.append(streak)
        changes.append(change)
        prev = value

    ranks = []
    history = []
    for change in changes:
        history.append(change)
        history = history[-rank_window:]
        ranks.append(100.0 * sum(old < change for old in history) / len(history))

    return (_rsi(close, rsi_window) + _rsi(np.asarray(streaks), streak_rsi_window) + np.asarray(ranks)) / 3.0


def _weighted_recent(values):
    weights = (1.0, 2.0, 2.0, 1.0)
    recent = list(reversed(values[-4:]))
    denom = sum(weights[: len(recent)])
    return sum(weight * value for weight, value in zip(weights, recent)) / denom


def _relative_vigor(open_, high, low, close, window=10):
    numerators = []
    denominators = []
    rvi = []
    signal = []
    close_open = []
    high_low = []
    rvi_history = []
    for o, h, l, c in zip(open_, high, low, close):
        close_open.append(c - o)
        high_low.append(h - l)
        numerators.append(_weighted_recent(close_open))
        denominators.append(_weighted_recent(high_low))
        num = sum(numerators[-window:])
        den = sum(denominators[-window:])
        value = 0.0 if den == 0.0 else num / den
        rvi.append(value)
        rvi_history.append(value)
        signal.append(_weighted_recent(rvi_history))
    return np.asarray(rvi), np.asarray(signal)


def _klinger(close, high, low, volume, fast=34, slow=55, signal_window=13):
    previous_hlc = 0.0
    previous_dm = 0.0
    previous_trend = 1.0
    cumulative = 0.0
    vf = []
    for index, (c, h, l, v) in enumerate(zip(close, high, low, volume)):
        hlc = h + l + c
        dm = h - l
        trend = previous_trend if index == 0 else (1.0 if hlc > previous_hlc else -1.0)
        cumulative = dm if index == 0 else (cumulative + dm if trend == previous_trend else previous_dm + dm)
        temp = 0.0 if cumulative == 0.0 else abs(2.0 * (dm / cumulative - 1.0))
        vf.append(v * temp * trend * 100.0)
        previous_hlc = hlc
        previous_dm = dm
        previous_trend = trend
    kvo = _ema(vf, fast) - _ema(vf, slow)
    signal = _ema(kvo, signal_window)
    return kvo, signal, kvo - signal


def _coppock(close, wma_window=10, long_roc=14, short_roc=11):
    values = []
    for index, value in enumerate(close):
        long_prev = close[index - long_roc] if index >= long_roc else 0.0
        short_prev = close[index - short_roc] if index >= short_roc else 0.0
        long_value = 0.0 if long_prev == 0.0 else (value - long_prev) / long_prev * 100.0
        short_value = 0.0 if short_prev == 0.0 else (value - short_prev) / short_prev * 100.0
        values.append(long_value + short_value)
    return _wma(values, wma_window)


def _fisher(high, low, window=10):
    value = 0.0
    fisher = 0.0
    out = []
    for index, (h, l) in enumerate(zip(high, low)):
        start = max(0, index + 1 - window)
        highest = high[start : index + 1].max()
        lowest = low[start : index + 1].min()
        price = (h + l) * 0.5
        normalized = 0.0 if highest == lowest else 2.0 * (price - lowest) / (highest - lowest) - 1.0
        normalized = min(max(normalized, -0.999), 0.999)
        value = min(max(0.33 * normalized + 0.67 * value, -0.999), 0.999)
        fisher = 0.5 * math.log((1.0 + value) / (1.0 - value)) + 0.5 * fisher
        out.append(fisher)
    return np.asarray(out)


def _frama(close, window=16):
    half = window // 2
    out = []
    value = close[0]
    for index, price in enumerate(close):
        if index + 1 >= window:
            segment = close[index + 1 - window : index + 1]
            n1 = (segment[:half].max() - segment[:half].min()) / half
            n2 = (segment[half:].max() - segment[half:].min()) / (window - half)
            n3 = (segment.max() - segment.min()) / window
            alpha = math.exp(-4.6 * ((math.log(n1 + n2) - math.log(n3)) / math.log(2.0) - 1.0)) if n1 > 0.0 and n2 > 0.0 and n3 > 0.0 else 1.0
            alpha = min(max(alpha, 0.01), 1.0)
        else:
            alpha = 1.0
        value = alpha * price + (1.0 - alpha) * value
        out.append(value)
    return np.asarray(out)


def _market():
    rng = np.random.default_rng(20260503)
    close = 100.0 + np.cumsum(rng.normal(0.02, 0.7, 512))
    open_ = close + rng.normal(0.0, 0.15, 512)
    spread = rng.uniform(0.1, 1.2, 512)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(10_000, 300_000, 512).astype(np.float64)
    real0 = close
    real1 = 0.8 * close + 5.0 + rng.normal(0.0, 0.4, 512)
    return open_, high, low, close, volume, real0, real1


def test_classic_requested_indicators_match_reference_formulas():
    open_, high, low, close, volume, _, _ = _market()

    np.testing.assert_allclose(rtta.ConnorsRSI().batch(close), _connors_rsi(close), rtol=1e-12, atol=1e-12)

    rvi, rvi_signal = _relative_vigor(open_, high, low, close)
    actual_rvi = rtta.RelativeVigorIndex().batch(open_, high, low, close)
    np.testing.assert_allclose(actual_rvi.rvi, rvi, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_rvi.signal, rvi_signal, rtol=1e-12, atol=1e-12)

    kvo, signal, histogram = _klinger(close, high, low, volume)
    actual_kvo = rtta.KlingerVolumeOscillator().batch(close, high, low, volume)
    np.testing.assert_allclose(actual_kvo.kvo, kvo, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_kvo.signal, signal, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_kvo.histogram, histogram, rtol=1e-12, atol=1e-12)

    elder = rtta.ElderRayIndex().batch(close, high, low)
    ema = _ema(close, 13)
    np.testing.assert_allclose(elder.bull_power, high - ema, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(elder.bear_power, low - ema, rtol=1e-12, atol=1e-12)

    np.testing.assert_allclose(rtta.CoppockCurve().batch(close), _coppock(close), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rtta.FisherTransform().batch(high, low), _fisher(high, low), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rtta.FractalAdaptiveMovingAverage().batch(close), _frama(close), rtol=1e-12, atol=1e-12)


def test_adaptive_and_kalman_requested_indicators_have_consistent_batch_and_incremental_paths():
    _, high, low, close, _, real0, real1 = _market()
    specs = [
        (rtta.MesaAdaptiveMovingAverage, (close,), ("mama", "fama")),
        (rtta.EhlersOptimalTrackingFilter, (high, low), None),
        (rtta.KalmanHedgeRatio, (real0, real1), ("hedge_ratio", "intercept", "spread")),
        (rtta.KalmanRegressionChannel, (real0, real1), ("slope", "intercept", "middle", "upper", "lower", "spread")),
        (rtta.TwoFactorKalmanTrendFilter, (close,), ("short_trend", "long_trend", "value")),
        (rtta.KalmanExtremumTrend, (close, high, low), ("trend", "oscillator", "signal")),
    ]

    for cls, arrays, fields in specs:
        batch_indicator = cls()
        update_indicator = cls()
        batch = batch_indicator.batch(*arrays)
        updates = [update_indicator.update(*[array[i] for array in arrays]) for i in range(len(close))]
        if fields is None:
            np.testing.assert_allclose(batch, np.asarray(updates), rtol=1e-12, atol=1e-12, equal_nan=True)
        else:
            for field in fields:
                np.testing.assert_allclose(
                    getattr(batch, field),
                    np.asarray([getattr(item, field) for item in updates]),
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )
