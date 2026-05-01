from dataclasses import dataclass

import numpy as np

from rtta.indicator import EMA, ROC, SMA, Kama, PercentagePrice, Summation


@dataclass(frozen=True)
class Record:
    input: float
    close: float


def _data():
    rng = np.random.default_rng(42)
    return 100.0 + np.cumsum(rng.normal(0.0, 0.45, 512))


def _records(values):
    return [Record(input=float(value), close=float(value)) for value in values]


def test_single_input_record_batches_match_array_batches():
    data = _data()
    records = _records(data)

    cases = [
        (EMA(window=30, fillna=False), EMA(window=30, fillna=False), "input"),
        (SMA(window=30, fillna=False), SMA(window=30, fillna=False), "input"),
        (Summation(window=30), Summation(window=30), "input"),
        (Kama(), Kama(), "input"),
        (ROC(window=10, fillna=False), ROC(window=10, fillna=False), "close"),
    ]

    for array_indicator, record_indicator, _ in cases:
        np.testing.assert_allclose(
            record_indicator.batch(records),
            array_indicator.batch(data),
            equal_nan=True,
        )


def test_percentage_price_record_batches_and_ppo_fast_path_match_array_batch():
    data = _data()
    records = _records(data)

    full = PercentagePrice(fillna=True).batch(data)
    records_full = PercentagePrice(fillna=True).batch(records)
    ppo_only = PercentagePrice(fillna=True).batch_ppo(data)
    records_ppo_only = PercentagePrice(fillna=True).batch_ppo(records)

    np.testing.assert_allclose(records_full.ppo, full.ppo, equal_nan=True)
    np.testing.assert_allclose(records_full.signal, full.signal, equal_nan=True)
    np.testing.assert_allclose(records_full.histogram, full.histogram, equal_nan=True)
    np.testing.assert_allclose(ppo_only, full.ppo, equal_nan=True)
    np.testing.assert_allclose(records_ppo_only, full.ppo, equal_nan=True)
