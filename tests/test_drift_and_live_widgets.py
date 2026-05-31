import numpy as np
import pytest


def _arrays():
    n = 96
    base = np.arange(n, dtype=np.float64)
    value = np.ascontiguousarray(np.r_[np.zeros(32), np.ones(32) * 4.0, np.linspace(4.0, -2.0, 32)])
    close = np.ascontiguousarray(100.0 + np.cumsum(np.r_[np.full(24, 0.02), np.full(24, 1.0), np.full(24, 0.01), np.full(24, -0.4)]))
    error = np.ascontiguousarray(np.r_[np.zeros(40), np.ones(30), np.zeros(26)])
    residual = np.ascontiguousarray(np.r_[np.full(32, 0.05), np.full(16, 4.0), np.full(24, -4.0), np.full(24, 0.0)])
    actual = np.ascontiguousarray(close)
    prediction = np.ascontiguousarray(actual - residual)
    hit = np.ascontiguousarray(1.0 - error)
    probability = np.ascontiguousarray(np.r_[np.full(40, 0.9), np.full(30, 0.1), np.full(26, 0.7)])
    outcome = np.ascontiguousarray(np.r_[np.ones(40), np.ones(30), np.zeros(26)])
    feature = np.ascontiguousarray(value + np.sin(base * 0.2) * 0.01)
    quote_messages = np.ascontiguousarray(np.r_[np.full(32, 100.0), np.full(24, 10_000.0), np.full(40, 150.0)])
    trades = np.ascontiguousarray(np.r_[np.full(32, 50.0), np.full(24, 5.0), np.full(40, 50.0)])
    session_progress = np.ascontiguousarray(np.linspace(0.0, 1.0, n))
    auction_signal = np.ascontiguousarray(np.r_[np.ones(12), np.zeros(72), np.ones(12)])
    bid_price = np.ascontiguousarray(100.0 + np.zeros(n))
    ask_price = np.ascontiguousarray(100.01 + np.r_[np.full(32, 0.0), np.full(24, 0.8), np.full(40, 0.0)])
    trade_price = np.ascontiguousarray(np.where((base.astype(int) % 2) == 0, bid_price, ask_price))
    bid_size = np.ascontiguousarray(np.r_[np.full(32, 10_000.0), np.full(24, 20.0), np.full(40, 10_000.0)])
    ask_size = np.ascontiguousarray(np.r_[np.full(32, 10_000.0), np.full(24, 20.0), np.full(40, 10_000.0)])
    volume = np.ascontiguousarray(np.r_[np.full(32, 20_000.0), np.full(24, 10.0), np.full(40, 20_000.0)])
    real0 = np.ascontiguousarray(np.sin(base * 0.2))
    real1 = np.ascontiguousarray(np.r_[real0[:48], -real0[48:]])
    return {
        "value": value,
        "close": close,
        "error": error,
        "residual": residual,
        "prediction": prediction,
        "actual": actual,
        "hit": hit,
        "probability": probability,
        "outcome": outcome,
        "feature": feature,
        "quote_messages": quote_messages,
        "trades": trades,
        "session_progress": session_progress,
        "auction_signal": auction_signal,
        "trade_price": trade_price,
        "bid_price": bid_price,
        "ask_price": ask_price,
        "volume": volume,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "real0": real0,
        "real1": real1,
    }


CASES = (
    ("ADWIN", ("value",), {"max_window": 32, "min_window": 8, "delta": 0.2}),
    ("DDM", ("error",), {"min_samples": 10}),
    ("EDDM", ("error",), {"min_errors": 5}),
    ("HDDM", ("error",), {"min_samples": 10}),
    ("KSWIN", ("value",), {"window": 32, "stat_window": 8, "alpha": 0.2}),
    ("ResidualDriftDetector", ("residual",), {"alpha": 0.4, "z_entry": 1.0, "z_exit": 0.25, "min_variance": 1.0e-4}),
    ("PredictionErrorDriftDetector", ("prediction", "actual"), {"alpha": 0.4, "z_entry": 1.0, "z_exit": 0.25, "min_variance": 1.0e-4}),
    ("HitRateDriftDetector", ("hit",), {"alpha": 0.4, "miss_entry": 0.4, "miss_exit": 0.2}),
    ("CalibrationDriftDetector", ("probability", "outcome"), {"alpha": 0.4, "error_entry": 0.5, "error_exit": 0.25}),
    ("FeatureDistributionDriftDetector", ("feature",), {"max_window": 32, "min_window": 8, "delta": 0.2}),
    ("OnlineHMMRegimeFilter", ("value",), {"states": 2, "stay_probability": 0.9, "alpha": 0.2}),
    ("StickyHMMRegimeFilter", ("value",), {"states": 2, "stay_probability": 0.97, "alpha": 0.2}),
    ("OnlineMarkovSwitchingVolatilityFilter", ("close",), {"stay_probability": 0.9, "alpha": 0.2}),
    ("BoundedBOCPD", ("value",), {"max_run_length": 32, "hazard": 0.1, "threshold": 0.2, "min_variance": 1.0e-4}),
    ("OnlineGaussianMixtureRegimeFilter", ("value",), {"components": 2, "alpha": 0.2}),
    ("HiddenSemiMarkovRegimeFilter", ("value",), {"states": 2, "stay_probability": 0.9, "expected_duration": 8.0, "duration_strength": 0.5, "alpha": 0.2}),
    ("VolatilityBreakoutDetector", ("close",), {"alpha": 0.4, "z_entry": 1.0, "z_exit": 0.25}),
    ("VolatilityCompressionExpansionDetector", ("close",), {"short_alpha": 0.5, "long_alpha": 0.05, "expansion_entry": 1.2, "expansion_exit": 1.0, "compression_entry": 0.7, "compression_exit": 0.9}),
    ("MicrostructureNoiseRegimeDetector", ("trade_price", "bid_price", "ask_price"), {"alpha": 0.4, "noise_entry": 0.4, "noise_exit": 0.2}),
    ("BidAskBounceRegimeDetector", ("trade_price", "bid_price", "ask_price"), {"alpha": 0.4, "bounce_entry": 0.4, "bounce_exit": 0.2}),
    ("QuoteMessageRateRegimeDetector", ("quote_messages",), {"alpha": 0.4, "high_entry": 3.0, "high_exit": 1.5, "low_entry": 0.4, "low_exit": 0.8}),
    ("QuoteStuffingDetector", ("quote_messages", "trades"), {"alpha": 0.4, "ratio_entry": 100.0, "ratio_exit": 50.0}),
    ("LeadLagRegimeDetector", ("real0", "real1"), {"alpha": 0.4, "lead_entry": 0.2, "lead_exit": 0.05}),
    ("LiquidityDroughtDetector", ("volume", "bid_size", "ask_size"), {"alpha": 0.4, "drought_entry": 0.3, "drought_exit": 0.7}),
    ("SpreadExplosionDetector", ("bid_price", "ask_price"), {"alpha": 0.4, "ratio_entry": 5.0, "ratio_exit": 2.0}),
    ("MarketOpenCloseTransitionDetector", ("session_progress",), {"open_entry": 0.1, "open_exit": 0.15, "close_entry": 0.9, "close_exit": 0.85}),
    ("AuctionContinuousMarketTransitionDetector", ("auction_signal",), {"auction_entry": 0.5, "auction_exit": 0.2}),
    ("CrossAssetCorrelationBreakDetector", ("real0", "real1"), {"short_window": 8, "long_window": 24, "break_entry": 0.5, "break_exit": 0.25}),
)


PROBABILITY_CLASSES = {
    "OnlineHMMRegimeFilter",
    "StickyHMMRegimeFilter",
    "OnlineMarkovSwitchingVolatilityFilter",
    "BoundedBOCPD",
    "OnlineGaussianMixtureRegimeFilter",
    "HiddenSemiMarkovRegimeFilter",
}


def _batch_args(arrays, fields, dtype=np.float64):
    return [np.ascontiguousarray(arrays[field].astype(dtype)) for field in fields]


def _records(arrays, fields):
    size = len(arrays[fields[0]])
    return [
        {field: float(arrays[field][index]) for field in fields}
        for index in range(size)
    ]


def _update_output(cls, kwargs, arrays, fields):
    indicator = cls(**kwargs)
    output = []
    for args in zip(*[arrays[field] for field in fields]):
        output.append(indicator.update(*[float(value) for value in args]))
        assert indicator.last() == output[-1]
    return np.asarray(output, dtype=np.float64)


@pytest.mark.parametrize("name,fields,kwargs", CASES)
def test_update_batch_advance_and_replay_paths_are_consistent(name, fields, kwargs):
    import rtta

    arrays = _arrays()
    cls = getattr(rtta, name)
    expected = _update_output(cls, kwargs, arrays, fields)
    np.testing.assert_allclose(cls(**kwargs).batch(*_batch_args(arrays, fields)), expected, rtol=0.0, atol=0.0)

    advance_indicator = cls(**kwargs)
    for args, expected_value in zip(zip(*[arrays[field] for field in fields]), expected):
        assert advance_indicator.advance(*[float(value) for value in args]) is None
        assert advance_indicator.last() == expected_value

    batch_args = _batch_args(arrays, fields)
    assert cls(**kwargs).replay_update(*batch_args) == pytest.approx(float(expected.sum()))
    assert cls(**kwargs).replay_advance(*batch_args) == pytest.approx(float(expected.sum()))

    if name in PROBABILITY_CLASSES:
        probability = cls(**kwargs).last_probability()
        assert 0.0 <= probability <= 1.0


@pytest.mark.parametrize("name,fields,kwargs", CASES)
def test_record_table_and_float32_batches_match_update(name, fields, kwargs):
    import rtta

    pandas = pytest.importorskip("pandas")
    arrays = _arrays()
    cls = getattr(rtta, name)
    expected = _update_output(cls, kwargs, arrays, fields)

    np.testing.assert_allclose(cls(**kwargs).batch(_records(arrays, fields)), expected, rtol=0.0, atol=0.0)

    table = pandas.DataFrame(_records(arrays, fields), copy=False)
    np.testing.assert_allclose(cls(**kwargs).batch(table), expected, rtol=0.0, atol=0.0)

    args32 = _batch_args(arrays, fields, np.float32)
    arrays32 = {field: arg for field, arg in zip(fields, args32)}
    expected32 = _update_output(cls, kwargs, arrays32, fields)
    np.testing.assert_allclose(cls(**kwargs).batch(*args32), expected32, rtol=0.0, atol=0.0)


def test_invalid_parameters_are_rejected():
    import rtta

    with pytest.raises(ValueError):
        rtta.ADWIN(max_window=8, min_window=8)
    with pytest.raises(ValueError):
        rtta.DDM(warning_level=3.0, drift_level=2.0)
    with pytest.raises(ValueError):
        rtta.EDDM(warning_ratio=0.8, drift_ratio=0.9)
    with pytest.raises(ValueError):
        rtta.HDDM(warning_delta=0.001, drift_delta=0.01)
    with pytest.raises(ValueError):
        rtta.KSWIN(window=8, stat_window=8)
    with pytest.raises(ValueError):
        rtta.OnlineHMMRegimeFilter(states=1)
    with pytest.raises(ValueError):
        rtta.BoundedBOCPD(hazard=1.0)
    with pytest.raises(ValueError):
        rtta.OnlineGaussianMixtureRegimeFilter(components=1)
    with pytest.raises(ValueError):
        rtta.VolatilityCompressionExpansionDetector(expansion_entry=1.0, expansion_exit=1.1, compression_entry=0.5, compression_exit=0.8)
    with pytest.raises(ValueError):
        rtta.QuoteStuffingDetector(ratio_entry=10.0, ratio_exit=20.0)
    with pytest.raises(ValueError):
        rtta.MarketOpenCloseTransitionDetector(open_entry=0.2, open_exit=0.1)
    with pytest.raises(ValueError):
        rtta.CrossAssetCorrelationBreakDetector(short_window=10, long_window=5)
