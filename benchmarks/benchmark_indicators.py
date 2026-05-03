#!/usr/bin/env python3
"""Benchmark RTTA indicators and optional comparison libraries.

Markdown output has separate batch and update tables. Batch timings compare
RTTA batch calls to third-party batch calls. RTTA update timings are reported
separately because they intentionally make one Python/C++ call per sample.

Optional comparison libraries are intentionally not project dependencies:

    python -m pip install ta==0.11.0 TA-Lib

Run after installing RTTA into the active environment, for example:

    python -m pip install --no-build-isolation -e .
    python benchmarks/benchmark_indicators.py --samples 200000
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


Number = float | int
ExternalRunner = Callable[["MarketData"], Any]


BLACKHOLE: Any = None


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    update_inputs: tuple[str, ...]
    ctor_args: tuple[Any, ...] = ()
    ctor_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_inputs: tuple[str, ...] | None = None
    batch_method: str = "batch"


@dataclass(frozen=True)
class MarketRecord:
    open: float
    high: float
    low: float
    close: float
    volume: float
    value: float
    input: float
    x: float
    real0: float
    real1: float
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float

    def __getitem__(self, key: str) -> float:
        return getattr(self, key)


@dataclass
class MarketData:
    arrays: dict[str, np.ndarray]
    lists: dict[str, list[float]]
    series: dict[str, Any]
    table: Any | None
    records: list[MarketRecord]
    record_dicts: list[dict[str, float]]


@dataclass
class BenchResult:
    indicator: str
    rtta_update_ns: float | None = None
    rtta_batch_ns: float | None = None
    rtta_table_batch_ns: float | None = None
    rtta_record_batch_ns: float | None = None
    rtta_record_batch_input: str | None = None
    talib_batch_ns: float | None = None
    ta_batch_ns: float | None = None


INDICATORS: tuple[IndicatorSpec, ...] = (
    IndicatorSpec("ATR", ("close", "high", "low")),
    IndicatorSpec("ATRP", ("close", "high", "low")),
    IndicatorSpec("EMA", ("value",), ctor_kwargs={"window": 30.0}, batch_inputs=("input",)),
    IndicatorSpec("EWMA", ("value",), ctor_kwargs={"span": 30.0}),
    IndicatorSpec("MACD", ("value",), batch_inputs=("input",)),
    IndicatorSpec("ROC", ("close",), ctor_kwargs={"window": 10}, batch_inputs=("close",)),
    IndicatorSpec("RSI", ("value",)),
    IndicatorSpec("SMA", ("value",), ctor_kwargs={"window": 30}, batch_inputs=("input",)),
    IndicatorSpec("TSI", ("x",)),
    IndicatorSpec("AbsolutePriceOscillator", ("close",)),
    IndicatorSpec("AccumulationDistribution", ("close", "high", "low", "volume")),
    IndicatorSpec("AlphaBetaGammaTrackingFilter", ("close",), batch_inputs=("close",)),
    IndicatorSpec("Aroon", ("high", "low")),
    IndicatorSpec("AroonOscillator", ("high", "low")),
    IndicatorSpec("AverageDirectionalMovementIndex", ("close", "high", "low")),
    IndicatorSpec("AverageDirectionalMovementIndexRating", ("close", "high", "low")),
    IndicatorSpec("AveragePrice", ("open", "high", "low", "close")),
    IndicatorSpec("AwesomeOscillator", ("high", "low"), batch_inputs=("high", "low")),
    IndicatorSpec("BalanceOfPower", ("open", "high", "low", "close")),
    IndicatorSpec("Beta", ("real0", "real1")),
    IndicatorSpec("BollingerBands", ("value",)),
    IndicatorSpec("ChaikinMoneyFlow", ("close", "high", "low", "volume"), batch_inputs=("close", "high", "low", "volume")),
    IndicatorSpec("ChaikinOscillator", ("close", "high", "low", "volume")),
    IndicatorSpec("ChandeMomentumOscillator", ("close",)),
    IndicatorSpec("ChoppinessIndex", ("close", "high", "low"), batch_inputs=("close", "high", "low")),
    IndicatorSpec("ConnorsRSI", ("close",), batch_inputs=("close",)),
    IndicatorSpec("CommodityChannelIndex", ("close", "high", "low")),
    IndicatorSpec("CoppockCurve", ("close",), batch_inputs=("close",)),
    IndicatorSpec("Correlation", ("real0", "real1")),
    IndicatorSpec("CumulativeReturn", ("close",), batch_inputs=("close",)),
    IndicatorSpec("DailyLogReturn", ("close",), batch_inputs=("close",)),
    IndicatorSpec("DailyReturn", ("close",), batch_inputs=("close",)),
    IndicatorSpec("Delay", ("value",)),
    IndicatorSpec("DetrendedPriceOscillator", ("close",), batch_inputs=("close",)),
    IndicatorSpec("DirectionalMovementIndex", ("close", "high", "low")),
    IndicatorSpec("DoubleEMA", ("value",)),
    IndicatorSpec("DonchianChannel", ("close", "high", "low"), batch_inputs=("close", "high", "low")),
    IndicatorSpec("EhlersOptimalTrackingFilter", ("high", "low"), batch_inputs=("high", "low")),
    IndicatorSpec("ElderRayIndex", ("close", "high", "low"), batch_inputs=("close", "high", "low")),
    IndicatorSpec("EaseOfMovement", ("high", "low", "volume"), batch_inputs=("high", "low", "volume")),
    IndicatorSpec("FastStochastic", ("close", "high", "low")),
    IndicatorSpec("FibonacciRetracementLevels", ("high", "low"), batch_inputs=("high", "low")),
    IndicatorSpec("FisherTransform", ("high", "low"), batch_inputs=("high", "low")),
    IndicatorSpec("ForceIndex", ("close", "volume"), batch_inputs=("close", "volume")),
    IndicatorSpec("FractalAdaptiveMovingAverage", ("value",), ctor_kwargs={"window": 16}, batch_inputs=("input",)),
    IndicatorSpec("GaussianProcessRegressionBands", ("close",), ctor_kwargs={"window": 16}, batch_inputs=("close",)),
    IndicatorSpec("High", ("value",), ctor_kwargs={"window": 30}),
    IndicatorSpec("HighIndex", ("value",)),
    IndicatorSpec("HighLow", ("value",)),
    IndicatorSpec("HighLowIndex", ("value",)),
    IndicatorSpec("HullMovingAverage", ("value",), ctor_kwargs={"window": 30}, batch_inputs=("input",)),
    IndicatorSpec("Ichimoku", ("high", "low"), batch_inputs=("high", "low")),
    IndicatorSpec("InteractingMultipleModelFilter", ("close",), batch_inputs=("close",)),
    IndicatorSpec("KSTOscillator", ("close",), batch_inputs=("close",)),
    IndicatorSpec("KalmanExtremumTrend", ("close", "high", "low"), batch_inputs=("close", "high", "low")),
    IndicatorSpec("KalmanHedgeRatio", ("real0", "real1"), batch_inputs=("real0", "real1")),
    IndicatorSpec("KalmanInnovationZScore", ("close",), batch_inputs=("close",)),
    IndicatorSpec("KalmanLocalLinearTrend", ("close",), batch_inputs=("close",)),
    IndicatorSpec("KalmanMovingAverage", ("close",), batch_inputs=("close",)),
    IndicatorSpec("KalmanPredictionBands", ("close",), batch_inputs=("close",)),
    IndicatorSpec("KalmanRegressionChannel", ("real0", "real1"), batch_inputs=("real0", "real1")),
    IndicatorSpec("KalmanTrendSignal", ("close",), batch_inputs=("close",)),
    IndicatorSpec("KalmanVelocityOscillator", ("close",), batch_inputs=("close",)),
    IndicatorSpec("Kama", ("close",), batch_inputs=("input",)),
    IndicatorSpec("KeltnerChannel", ("close", "high", "low")),
    IndicatorSpec("KeltnerChannelOriginal", ("close", "high", "low")),
    IndicatorSpec("KlingerVolumeOscillator", ("close", "high", "low", "volume"), batch_inputs=("close", "high", "low", "volume")),
    IndicatorSpec("LinearRegression", ("value",)),
    IndicatorSpec("LinearRegressionAngle", ("value",)),
    IndicatorSpec("LinearRegressionIntercept", ("value",)),
    IndicatorSpec("LinearRegressionSlope", ("value",)),
    IndicatorSpec("Low", ("value",), ctor_kwargs={"window": 30}),
    IndicatorSpec("LowIndex", ("value",)),
    IndicatorSpec("MACDFix", ("close",)),
    IndicatorSpec("MassIndex", ("high", "low")),
    IndicatorSpec("MedianPrice", ("high", "low")),
    IndicatorSpec("MesaAdaptiveMovingAverage", ("value",), batch_inputs=("input",)),
    IndicatorSpec("MidPoint", ("value",)),
    IndicatorSpec("MidPrice", ("high", "low")),
    IndicatorSpec("MinusDirectionalIndicator", ("close", "high", "low")),
    IndicatorSpec("MinusDirectionalMovement", ("high", "low")),
    IndicatorSpec("Momentum", ("close",)),
    IndicatorSpec("MoneyFlowIndex", ("close", "high", "low", "volume")),
    IndicatorSpec("NadarayaWatsonEnvelope", ("close",), ctor_kwargs={"window": 32}, batch_inputs=("close",)),
    IndicatorSpec("NegativeVolumeIndex", ("close", "volume"), batch_inputs=("close", "volume")),
    IndicatorSpec("NormalizedATR", ("close", "high", "low")),
    IndicatorSpec("OnBalanceVolume", ("close", "volume")),
    IndicatorSpec("OrderFlowImbalance", ("bid_price", "bid_size", "ask_price", "ask_size"), batch_inputs=("bid_price", "bid_size", "ask_price", "ask_size")),
    IndicatorSpec("ParticleFilterTrend", ("close",), ctor_kwargs={"particles": 64}, batch_inputs=("close",)),
    IndicatorSpec("ParabolicSAR", ("high", "low")),
    IndicatorSpec("PercentagePrice", ("close",), batch_inputs=("close",), batch_method="batch_ppo"),
    IndicatorSpec("PercentageVolume", ("volume",), batch_inputs=("volume",)),
    IndicatorSpec("PlusDirectionalIndicator", ("close", "high", "low")),
    IndicatorSpec("PlusDirectionalMovement", ("high", "low")),
    IndicatorSpec("RateOfChangePercentage", ("close",)),
    IndicatorSpec("RateOfChangeRatio", ("close",)),
    IndicatorSpec("RateOfChangeRatio100", ("close",)),
    IndicatorSpec("RenkoBrickGenerator", ("close",), batch_inputs=("close",)),
    IndicatorSpec("RelativeVigorIndex", ("open", "high", "low", "close"), batch_inputs=("open", "high", "low", "close")),
    IndicatorSpec("SavitzkyGolayFilter", ("close",), batch_inputs=("close",)),
    IndicatorSpec("SchaffTrendCycle", ("close",), batch_inputs=("close",)),
    IndicatorSpec("StdDev", ("value",), ctor_kwargs={"window": 5}),
    IndicatorSpec("StochRSI", ("value",)),
    IndicatorSpec("Stochastic", ("close", "high", "low")),
    IndicatorSpec("SuperTrend", ("close", "high", "low"), batch_inputs=("close", "high", "low")),
    IndicatorSpec("Summation", ("value",), ctor_kwargs={"window": 30}, batch_inputs=("input",)),
    IndicatorSpec("T3MovingAverage", ("value",)),
    IndicatorSpec("TimeSeriesForecast", ("value",)),
    IndicatorSpec("TwoFactorKalmanTrendFilter", ("close",), batch_inputs=("close",)),
    IndicatorSpec("TrueRange", ("close", "high", "low")),
    IndicatorSpec("TriangularMovingAverage", ("value",)),
    IndicatorSpec("TripleEMA", ("value",)),
    IndicatorSpec("Trix", ("value",)),
    IndicatorSpec("TypicalPrice", ("close", "high", "low")),
    IndicatorSpec("UltimateOscillator", ("close", "high", "low")),
    IndicatorSpec("UlcerIndex", ("close",), batch_inputs=("close",)),
    IndicatorSpec("Variance", ("value",)),
    IndicatorSpec("VariableIndexDynamicAverage", ("close",), batch_inputs=("close",)),
    IndicatorSpec("VolumePriceTrend", ("close", "volume"), batch_inputs=("close", "volume")),
    IndicatorSpec("VolumeWeightedAveragePrice", ("close", "high", "low", "volume"), batch_inputs=("close", "high", "low", "volume")),
    IndicatorSpec("VolumeWeightedMovingAverage", ("close", "volume"), batch_inputs=("close", "volume")),
    IndicatorSpec("Vortex", ("close", "high", "low"), batch_inputs=("close", "high", "low")),
    IndicatorSpec("WeightedClosePrice", ("close", "high", "low")),
    IndicatorSpec("WeightedMovingAverage", ("value",)),
    IndicatorSpec("WilliamsR", ("close", "high", "low")),
    IndicatorSpec("ZigZagSwingDetector", ("close",), batch_inputs=("close",)),
)


def generate_market_data(samples: int, seed: int) -> MarketData:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.45, samples))
    open_ = close + rng.normal(0.0, 0.08, samples)
    spread = rng.uniform(0.02, 0.8, samples)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(1_000, 100_000, samples).astype(np.float64)
    real0 = close
    real1 = close * 0.73 + rng.normal(0.0, 0.25, samples)
    quote_spread = rng.uniform(0.01, 0.05, samples)
    bid_price = close - 0.5 * quote_spread
    ask_price = close + 0.5 * quote_spread
    bid_size = rng.integers(100, 10_000, samples).astype(np.float64)
    ask_size = rng.integers(100, 10_000, samples).astype(np.float64)

    arrays = {
        "open": open_.astype(np.float64),
        "high": high.astype(np.float64),
        "low": low.astype(np.float64),
        "close": close.astype(np.float64),
        "volume": volume,
        "value": close.astype(np.float64),
        "input": close.astype(np.float64),
        "x": close.astype(np.float64),
        "real0": real0.astype(np.float64),
        "real1": real1.astype(np.float64),
        "bid_price": bid_price.astype(np.float64),
        "bid_size": bid_size,
        "ask_price": ask_price.astype(np.float64),
        "ask_size": ask_size,
    }

    lists = {name: values.tolist() for name, values in arrays.items()}
    series: dict[str, Any] = {}
    table: Any | None = None

    try:
        pandas = importlib.import_module("pandas")
    except ImportError:
        pass
    else:
        series = {name: pandas.Series(values, copy=False) for name, values in arrays.items()}
        table = pandas.DataFrame(arrays, copy=False)

    records = [
        MarketRecord(
            open=float(arrays["open"][i]),
            high=float(arrays["high"][i]),
            low=float(arrays["low"][i]),
            close=float(arrays["close"][i]),
            volume=float(arrays["volume"][i]),
            value=float(arrays["value"][i]),
            input=float(arrays["input"][i]),
            x=float(arrays["x"][i]),
            real0=float(arrays["real0"][i]),
            real1=float(arrays["real1"][i]),
            bid_price=float(arrays["bid_price"][i]),
            bid_size=float(arrays["bid_size"][i]),
            ask_price=float(arrays["ask_price"][i]),
            ask_size=float(arrays["ask_size"][i]),
        )
        for i in range(samples)
    ]

    record_dicts = [
        {
            "open": record.open,
            "high": record.high,
            "low": record.low,
            "close": record.close,
            "volume": record.volume,
            "value": record.value,
            "input": record.input,
            "x": record.x,
            "real0": record.real0,
            "real1": record.real1,
            "bid_price": record.bid_price,
            "bid_size": record.bid_size,
            "ask_price": record.ask_price,
            "ask_size": record.ask_size,
        }
        for record in records
    ]

    return MarketData(arrays=arrays, lists=lists, series=series, table=table, records=records, record_dicts=record_dicts)


def make_rtta_incremental_runner(rtta: Any, spec: IndicatorSpec, data: MarketData) -> Callable[[], None]:
    indicator_cls = getattr(rtta, spec.name)
    inputs = [data.lists[name] for name in spec.update_inputs]

    def make_indicator() -> Any:
        return indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)

    if len(inputs) == 1:
        a0 = inputs[0]

        def run() -> None:
            indicator = make_indicator()
            update = indicator.update
            for x0 in a0:
                update(x0)

        return run

    if len(inputs) == 2:
        a0, a1 = inputs

        def run() -> None:
            indicator = make_indicator()
            update = indicator.update
            for x0, x1 in zip(a0, a1):
                update(x0, x1)

        return run

    if len(inputs) == 3:
        a0, a1, a2 = inputs

        def run() -> None:
            indicator = make_indicator()
            update = indicator.update
            for x0, x1, x2 in zip(a0, a1, a2):
                update(x0, x1, x2)

        return run

    if len(inputs) == 4:
        a0, a1, a2, a3 = inputs

        def run() -> None:
            indicator = make_indicator()
            update = indicator.update
            for x0, x1, x2, x3 in zip(a0, a1, a2, a3):
                update(x0, x1, x2, x3)

        return run

    raise ValueError(f"{spec.name} has unsupported arity {len(inputs)}")


def batch_inputs_for(spec: IndicatorSpec) -> tuple[str, ...]:
    return spec.batch_inputs or spec.update_inputs


def make_rtta_array_batch_runner(rtta: Any, spec: IndicatorSpec, data: MarketData) -> Callable[[], Any] | None:
    indicator_cls = getattr(rtta, spec.name)
    arrays = [data.arrays[name] for name in batch_inputs_for(spec)]

    def array_run() -> Any:
        indicator = indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)
        batch = getattr(indicator, spec.batch_method)
        return batch(*arrays)

    return array_run


def make_rtta_table_batch_runner(rtta: Any, spec: IndicatorSpec, data: MarketData) -> Callable[[], Any] | None:
    if data.table is None:
        return None

    indicator_cls = getattr(rtta, spec.name)

    def table_run() -> Any:
        indicator = indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)
        batch = getattr(indicator, spec.batch_method)
        return batch(data.table)

    try:
        table_run()
    except TypeError:
        return None
    return table_run


def make_rtta_record_batch_runner(rtta: Any, spec: IndicatorSpec, data: MarketData) -> tuple[Callable[[], Any], str] | None:
    indicator_cls = getattr(rtta, spec.name)
    batch_inputs = batch_inputs_for(spec)

    def attr_record_run() -> Any:
        indicator = indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)
        batch = getattr(indicator, spec.batch_method)
        return batch(data.records)

    def dict_record_run() -> Any:
        indicator = indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)
        batch = getattr(indicator, spec.batch_method)
        return batch(data.record_dicts)

    tuple_records = [tuple(record[name] for name in batch_inputs) for record in data.records]

    def tuple_record_run() -> Any:
        indicator = indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)
        batch = getattr(indicator, spec.batch_method)
        return batch(tuple_records)

    for label, runner in (
        ("records:attributes", attr_record_run),
        ("records:dicts", dict_record_run),
        ("records:tuples", tuple_record_run),
    ):
        try:
            runner()
        except TypeError:
            continue
        return runner, label

    return None


def make_rtta_batch_runner(rtta: Any, spec: IndicatorSpec, data: MarketData) -> tuple[Callable[[], Any], str] | None:
    record_runner = make_rtta_record_batch_runner(rtta, spec, data)
    if record_runner is not None:
        return record_runner

    array_runner = make_rtta_array_batch_runner(rtta, spec, data)
    if array_runner is not None:
        return array_runner, "arrays"

    return None


def talib_benchmarks(talib: Any) -> dict[str, ExternalRunner]:
    return {
        "ATR": lambda d: talib.ATR(d.arrays["high"], d.arrays["low"], d.arrays["close"], timeperiod=14),
        "ATRP": lambda d: talib.NATR(d.arrays["high"], d.arrays["low"], d.arrays["close"], timeperiod=14),
        "EMA": lambda d: talib.EMA(d.arrays["close"], timeperiod=30),
        "MACD": lambda d: talib.MACD(d.arrays["close"], fastperiod=12, slowperiod=26, signalperiod=9)[1],
        "ROC": lambda d: talib.ROC(d.arrays["close"], timeperiod=10),
        "RSI": lambda d: talib.RSI(d.arrays["close"], timeperiod=14),
        "SMA": lambda d: talib.SMA(d.arrays["close"], timeperiod=30),
        "AbsolutePriceOscillator": lambda d: talib.APO(d.arrays["close"], fastperiod=12, slowperiod=26),
        "AccumulationDistribution": lambda d: talib.AD(d.arrays["high"], d.arrays["low"], d.arrays["close"], d.arrays["volume"]),
        "Aroon": lambda d: talib.AROON(d.arrays["high"], d.arrays["low"], timeperiod=14),
        "AroonOscillator": lambda d: talib.AROONOSC(d.arrays["high"], d.arrays["low"], timeperiod=14),
        "AverageDirectionalMovementIndex": lambda d: talib.ADX(d.arrays["high"], d.arrays["low"], d.arrays["close"], timeperiod=14),
        "AverageDirectionalMovementIndexRating": lambda d: talib.ADXR(d.arrays["high"], d.arrays["low"], d.arrays["close"], timeperiod=14),
        "AveragePrice": lambda d: talib.AVGPRICE(d.arrays["open"], d.arrays["high"], d.arrays["low"], d.arrays["close"]),
        "BalanceOfPower": lambda d: talib.BOP(d.arrays["open"], d.arrays["high"], d.arrays["low"], d.arrays["close"]),
        "Beta": lambda d: talib.BETA(d.arrays["real0"], d.arrays["real1"], timeperiod=5),
        "BollingerBands": lambda d: talib.BBANDS(d.arrays["close"], timeperiod=20, nbdevup=2, nbdevdn=2),
        "ChaikinOscillator": lambda d: talib.ADOSC(d.arrays["high"], d.arrays["low"], d.arrays["close"], d.arrays["volume"], fastperiod=3, slowperiod=10),
        "ChandeMomentumOscillator": lambda d: talib.CMO(d.arrays["close"], timeperiod=14),
        "CommodityChannelIndex": lambda d: talib.CCI(d.arrays["high"], d.arrays["low"], d.arrays["close"], timeperiod=14),
        "Correlation": lambda d: talib.CORREL(d.arrays["real0"], d.arrays["real1"], timeperiod=30),
        "DirectionalMovementIndex": lambda d: talib.DX(d.arrays["high"], d.arrays["low"], d.arrays["close"], timeperiod=14),
        "DoubleEMA": lambda d: talib.DEMA(d.arrays["close"], timeperiod=30),
        "FastStochastic": lambda d: talib.STOCHF(d.arrays["high"], d.arrays["low"], d.arrays["close"], fastk_period=5, fastd_period=3),
        "High": lambda d: talib.MAX(d.arrays["close"], timeperiod=30),
        "HighIndex": lambda d: talib.MAXINDEX(d.arrays["close"], timeperiod=30),
        "HighLow": lambda d: talib.MINMAX(d.arrays["close"], timeperiod=30),
        "HighLowIndex": lambda d: talib.MINMAXINDEX(d.arrays["close"], timeperiod=30),
        "Kama": lambda d: talib.KAMA(d.arrays["close"], timeperiod=10),
        "LinearRegression": lambda d: talib.LINEARREG(d.arrays["close"], timeperiod=14),
        "LinearRegressionAngle": lambda d: talib.LINEARREG_ANGLE(d.arrays["close"], timeperiod=14),
        "LinearRegressionIntercept": lambda d: talib.LINEARREG_INTERCEPT(d.arrays["close"], timeperiod=14),
        "LinearRegressionSlope": lambda d: talib.LINEARREG_SLOPE(d.arrays["close"], timeperiod=14),
        "Low": lambda d: talib.MIN(d.arrays["close"], timeperiod=30),
        "LowIndex": lambda d: talib.MININDEX(d.arrays["close"], timeperiod=30),
        "MACDFix": lambda d: talib.MACDFIX(d.arrays["close"], signalperiod=9)[1],
        "MedianPrice": lambda d: talib.MEDPRICE(d.arrays["high"], d.arrays["low"]),
        "MidPoint": lambda d: talib.MIDPOINT(d.arrays["close"], timeperiod=14),
        "MidPrice": lambda d: talib.MIDPRICE(d.arrays["high"], d.arrays["low"], timeperiod=14),
        "MinusDirectionalIndicator": lambda d: talib.MINUS_DI(d.arrays["high"], d.arrays["low"], d.arrays["close"], timeperiod=14),
        "MinusDirectionalMovement": lambda d: talib.MINUS_DM(d.arrays["high"], d.arrays["low"], timeperiod=14),
        "Momentum": lambda d: talib.MOM(d.arrays["close"], timeperiod=10),
        "MoneyFlowIndex": lambda d: talib.MFI(d.arrays["high"], d.arrays["low"], d.arrays["close"], d.arrays["volume"], timeperiod=14),
        "NormalizedATR": lambda d: talib.NATR(d.arrays["high"], d.arrays["low"], d.arrays["close"], timeperiod=14),
        "OnBalanceVolume": lambda d: talib.OBV(d.arrays["close"], d.arrays["volume"]),
        "ParabolicSAR": lambda d: talib.SAR(d.arrays["high"], d.arrays["low"], acceleration=0.02, maximum=0.2),
        "PercentagePrice": lambda d: talib.PPO(d.arrays["close"], fastperiod=12, slowperiod=26),
        "PlusDirectionalIndicator": lambda d: talib.PLUS_DI(d.arrays["high"], d.arrays["low"], d.arrays["close"], timeperiod=14),
        "PlusDirectionalMovement": lambda d: talib.PLUS_DM(d.arrays["high"], d.arrays["low"], timeperiod=14),
        "RateOfChangePercentage": lambda d: talib.ROCP(d.arrays["close"], timeperiod=10),
        "RateOfChangeRatio": lambda d: talib.ROCR(d.arrays["close"], timeperiod=10),
        "RateOfChangeRatio100": lambda d: talib.ROCR100(d.arrays["close"], timeperiod=10),
        "StdDev": lambda d: talib.STDDEV(d.arrays["close"], timeperiod=5, nbdev=1),
        "StochRSI": lambda d: talib.STOCHRSI(d.arrays["close"], timeperiod=14, fastk_period=5, fastd_period=3),
        "Stochastic": lambda d: talib.STOCH(d.arrays["high"], d.arrays["low"], d.arrays["close"], fastk_period=5, slowk_period=3, slowd_period=3),
        "Summation": lambda d: talib.SUM(d.arrays["close"], timeperiod=30),
        "T3MovingAverage": lambda d: talib.T3(d.arrays["close"], timeperiod=5, vfactor=0.7),
        "TimeSeriesForecast": lambda d: talib.TSF(d.arrays["close"], timeperiod=14),
        "TrueRange": lambda d: talib.TRANGE(d.arrays["high"], d.arrays["low"], d.arrays["close"]),
        "TriangularMovingAverage": lambda d: talib.TRIMA(d.arrays["close"], timeperiod=30),
        "TripleEMA": lambda d: talib.TEMA(d.arrays["close"], timeperiod=30),
        "Trix": lambda d: talib.TRIX(d.arrays["close"], timeperiod=30),
        "TypicalPrice": lambda d: talib.TYPPRICE(d.arrays["high"], d.arrays["low"], d.arrays["close"]),
        "UltimateOscillator": lambda d: talib.ULTOSC(d.arrays["high"], d.arrays["low"], d.arrays["close"], timeperiod1=7, timeperiod2=14, timeperiod3=28),
        "Variance": lambda d: talib.VAR(d.arrays["close"], timeperiod=5, nbdev=1),
        "WeightedClosePrice": lambda d: talib.WCLPRICE(d.arrays["high"], d.arrays["low"], d.arrays["close"]),
        "WeightedMovingAverage": lambda d: talib.WMA(d.arrays["close"], timeperiod=30),
        "WilliamsR": lambda d: talib.WILLR(d.arrays["high"], d.arrays["low"], d.arrays["close"], timeperiod=14),
    }


def ta_benchmarks(ta_pkg: Any) -> dict[str, ExternalRunner]:
    momentum = importlib.import_module("ta.momentum")
    others = importlib.import_module("ta.others")
    trend = importlib.import_module("ta.trend")
    volatility = importlib.import_module("ta.volatility")
    volume = importlib.import_module("ta.volume")

    def aroon_pair(d: MarketData) -> tuple[Any, Any]:
        indicator = trend.AroonIndicator(d.series["high"], d.series["low"], window=14, fillna=True)
        return indicator.aroon_up(), indicator.aroon_down()

    def bollinger(d: MarketData) -> tuple[Any, Any, Any, Any, Any]:
        indicator = volatility.BollingerBands(d.series["close"], window=20, fillna=True)
        return (
            indicator.bollinger_hband(),
            indicator.bollinger_lband(),
            indicator.bollinger_mavg(),
            indicator.bollinger_pband(),
            indicator.bollinger_wband(),
        )

    def donchian(d: MarketData) -> tuple[Any, Any, Any, Any, Any]:
        indicator = volatility.DonchianChannel(d.series["high"], d.series["low"], d.series["close"], window=20, fillna=True)
        return (
            indicator.donchian_channel_hband(),
            indicator.donchian_channel_lband(),
            indicator.donchian_channel_mband(),
            indicator.donchian_channel_pband(),
            indicator.donchian_channel_wband(),
        )

    def ease_of_movement(d: MarketData) -> tuple[Any, Any]:
        indicator = volume.EaseOfMovementIndicator(d.series["high"], d.series["low"], d.series["volume"], window=14, fillna=True)
        return indicator.ease_of_movement(), indicator.sma_ease_of_movement()

    def ichimoku(d: MarketData) -> tuple[Any, Any, Any, Any]:
        indicator = trend.IchimokuIndicator(d.series["high"], d.series["low"], window1=9, window2=26, window3=52, visual=False, fillna=True)
        return (
            indicator.ichimoku_conversion_line(),
            indicator.ichimoku_base_line(),
            indicator.ichimoku_a(),
            indicator.ichimoku_b(),
        )

    def keltner(d: MarketData) -> tuple[Any, Any, Any, Any, Any]:
        indicator = volatility.KeltnerChannel(
            d.series["high"],
            d.series["low"],
            d.series["close"],
            window=20,
            window_atr=20,
            original_version=False,
            multiplier=2,
            fillna=True,
        )
        return (
            indicator.keltner_channel_hband(),
            indicator.keltner_channel_lband(),
            indicator.keltner_channel_mband(),
            indicator.keltner_channel_pband(),
            indicator.keltner_channel_wband(),
        )

    def keltner_original(d: MarketData) -> tuple[Any, Any, Any, Any, Any]:
        indicator = volatility.KeltnerChannel(
            d.series["high"],
            d.series["low"],
            d.series["close"],
            window=20,
            original_version=True,
            fillna=True,
        )
        return (
            indicator.keltner_channel_hband(),
            indicator.keltner_channel_lband(),
            indicator.keltner_channel_mband(),
            indicator.keltner_channel_pband(),
            indicator.keltner_channel_wband(),
        )

    def kst(d: MarketData) -> tuple[Any, Any, Any]:
        indicator = trend.KSTIndicator(d.series["close"], fillna=True)
        return indicator.kst(), indicator.kst_sig(), indicator.kst_diff()

    def macd(d: MarketData) -> tuple[Any, Any, Any]:
        indicator = trend.MACD(d.series["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        return indicator.macd(), indicator.macd_signal(), indicator.macd_diff()

    def ppo(d: MarketData) -> Any:
        indicator = momentum.PercentagePriceOscillator(d.series["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        return indicator.ppo()

    def pvo(d: MarketData) -> tuple[Any, Any, Any]:
        indicator = momentum.PercentageVolumeOscillator(d.series["volume"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        return indicator.pvo(), indicator.pvo_signal(), indicator.pvo_hist()

    def psar(d: MarketData) -> tuple[Any, Any, Any]:
        indicator = trend.PSARIndicator(d.series["high"], d.series["low"], d.series["close"], step=0.02, max_step=0.2, fillna=True)
        return indicator.psar(), indicator.psar_up(), indicator.psar_down()

    def stoch(d: MarketData) -> tuple[Any, Any]:
        indicator = momentum.StochasticOscillator(d.series["high"], d.series["low"], d.series["close"], window=5, smooth_window=3, fillna=True)
        return indicator.stoch(), indicator.stoch_signal()

    def stochrsi(d: MarketData) -> tuple[Any, Any, Any]:
        indicator = momentum.StochRSIIndicator(d.series["close"], window=14, smooth1=3, smooth2=3, fillna=True)
        return indicator.stochrsi(), indicator.stochrsi_k(), indicator.stochrsi_d()

    def vortex(d: MarketData) -> tuple[Any, Any, Any]:
        indicator = trend.VortexIndicator(d.series["high"], d.series["low"], d.series["close"], window=14, fillna=True)
        return indicator.vortex_indicator_pos(), indicator.vortex_indicator_neg(), indicator.vortex_indicator_diff()

    return {
        "ATR": lambda d: volatility.average_true_range(d.series["high"], d.series["low"], d.series["close"], window=14, fillna=True),
        "EMA": lambda d: trend.ema_indicator(d.series["close"], window=30, fillna=True),
        "MACD": macd,
        "ROC": lambda d: momentum.roc(d.series["close"], window=10, fillna=True),
        "RSI": lambda d: momentum.rsi(d.series["close"], window=14, fillna=True),
        "SMA": lambda d: trend.sma_indicator(d.series["close"], window=30, fillna=True),
        "TSI": lambda d: momentum.tsi(d.series["close"], window_slow=25, window_fast=13, fillna=True),
        "AccumulationDistribution": lambda d: volume.acc_dist_index(d.series["high"], d.series["low"], d.series["close"], d.series["volume"], fillna=True),
        "Aroon": aroon_pair,
        "AroonOscillator": lambda d: trend.AroonIndicator(d.series["high"], d.series["low"], window=14, fillna=True).aroon_indicator(),
        "AverageDirectionalMovementIndex": lambda d: trend.adx(d.series["high"], d.series["low"], d.series["close"], window=14, fillna=True),
        "AwesomeOscillator": lambda d: momentum.awesome_oscillator(d.series["high"], d.series["low"], window1=5, window2=34, fillna=True),
        "BollingerBands": bollinger,
        "ChaikinMoneyFlow": lambda d: volume.chaikin_money_flow(d.series["high"], d.series["low"], d.series["close"], d.series["volume"], window=20, fillna=True),
        "CommodityChannelIndex": lambda d: trend.cci(d.series["high"], d.series["low"], d.series["close"], window=14, fillna=True),
        "CumulativeReturn": lambda d: others.cumulative_return(d.series["close"], fillna=True),
        "DailyLogReturn": lambda d: others.daily_log_return(d.series["close"], fillna=True),
        "DailyReturn": lambda d: others.daily_return(d.series["close"], fillna=True),
        "DetrendedPriceOscillator": lambda d: trend.dpo(d.series["close"], window=20, fillna=True),
        "DonchianChannel": donchian,
        "EaseOfMovement": ease_of_movement,
        "ForceIndex": lambda d: volume.force_index(d.series["close"], d.series["volume"], window=13, fillna=True),
        "Ichimoku": ichimoku,
        "Kama": lambda d: momentum.kama(d.series["close"], window=10, pow1=2, pow2=30, fillna=True),
        "KeltnerChannel": keltner,
        "KeltnerChannelOriginal": keltner_original,
        "KSTOscillator": kst,
        "MACDFix": lambda d: trend.MACD(d.series["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True).macd_signal(),
        "MassIndex": lambda d: trend.mass_index(d.series["high"], d.series["low"], window_fast=9, window_slow=25, fillna=True),
        "MinusDirectionalIndicator": lambda d: trend.adx_neg(d.series["high"], d.series["low"], d.series["close"], window=14, fillna=True),
        "MoneyFlowIndex": lambda d: volume.money_flow_index(d.series["high"], d.series["low"], d.series["close"], d.series["volume"], window=14, fillna=True),
        "NegativeVolumeIndex": lambda d: volume.negative_volume_index(d.series["close"], d.series["volume"], fillna=True),
        "OnBalanceVolume": lambda d: volume.on_balance_volume(d.series["close"], d.series["volume"], fillna=True),
        "ParabolicSAR": psar,
        "PercentagePrice": ppo,
        "PercentageVolume": pvo,
        "PlusDirectionalIndicator": lambda d: trend.adx_pos(d.series["high"], d.series["low"], d.series["close"], window=14, fillna=True),
        "SchaffTrendCycle": lambda d: trend.STCIndicator(d.series["close"], fillna=True).stc(),
        "StochRSI": stochrsi,
        "Stochastic": stoch,
        "Trix": lambda d: trend.trix(d.series["close"], window=30, fillna=True),
        "UltimateOscillator": lambda d: momentum.ultimate_oscillator(d.series["high"], d.series["low"], d.series["close"], window1=7, window2=14, window3=28, fillna=True),
        "UlcerIndex": lambda d: volatility.ulcer_index(d.series["close"], window=14, fillna=True),
        "VolumePriceTrend": lambda d: volume.volume_price_trend(d.series["close"], d.series["volume"], fillna=True),
        "VolumeWeightedAveragePrice": lambda d: volume.volume_weighted_average_price(d.series["high"], d.series["low"], d.series["close"], d.series["volume"], window=14, fillna=True),
        "Vortex": vortex,
        "WeightedMovingAverage": lambda d: trend.wma_indicator(d.series["close"], window=30, fillna=True),
        "WilliamsR": lambda d: momentum.williams_r(d.series["high"], d.series["low"], d.series["close"], lbp=14, fillna=True),
    }


def import_optional(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def benchmark_runner(
    runner: Callable[[], Any],
    samples: int,
    repeat: int,
    warmup: int,
) -> float:
    global BLACKHOLE

    for _ in range(warmup):
        BLACKHOLE = runner()

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        best = math.inf
        for _ in range(repeat):
            start = time.perf_counter_ns()
            BLACKHOLE = runner()
            elapsed = time.perf_counter_ns() - start
            best = min(best, elapsed)
    finally:
        if was_enabled:
            gc.enable()

    return best / samples


def run_benchmarks(args: argparse.Namespace) -> tuple[list[BenchResult], dict[str, str]]:
    try:
        rtta = importlib.import_module("rtta")
    except ImportError as exc:
        raise SystemExit(
            "Could not import rtta. Install the local package first, for example:\n"
            "  python -m pip install --no-build-isolation -e ."
        ) from exc

    data = generate_market_data(args.samples, args.seed)
    talib = import_optional("talib")
    ta_pkg = import_optional("ta")
    talib_map = talib_benchmarks(talib) if talib is not None else {}
    ta_map = ta_benchmarks(ta_pkg) if ta_pkg is not None and data.series else {}

    rows: list[BenchResult] = []
    for spec in INDICATORS:
        row = BenchResult(indicator=spec.name)

        if args.library in {"all", "rtta"}:
            row.rtta_update_ns = benchmark_runner(
                make_rtta_incremental_runner(rtta, spec, data),
                args.samples,
                args.repeat,
                args.warmup,
            )

            batch_runner = make_rtta_array_batch_runner(rtta, spec, data)
            if batch_runner is not None:
                row.rtta_batch_ns = benchmark_runner(batch_runner, args.samples, args.repeat, args.warmup)

            table_batch_runner = make_rtta_table_batch_runner(rtta, spec, data)
            if table_batch_runner is not None:
                row.rtta_table_batch_ns = benchmark_runner(table_batch_runner, args.samples, args.repeat, args.warmup)

            record_batch_runner_info = make_rtta_record_batch_runner(rtta, spec, data)
            if record_batch_runner_info is not None:
                record_batch_runner, row.rtta_record_batch_input = record_batch_runner_info
                row.rtta_record_batch_ns = benchmark_runner(record_batch_runner, args.samples, args.repeat, args.warmup)

        if args.library in {"all", "talib"} and spec.name in talib_map:
            row.talib_batch_ns = benchmark_runner(
                lambda runner=talib_map[spec.name]: runner(data),
                args.samples,
                args.repeat,
                args.warmup,
            )

        if args.library in {"all", "ta"} and spec.name in ta_map:
            row.ta_batch_ns = benchmark_runner(
                lambda runner=ta_map[spec.name]: runner(data),
                args.samples,
                args.repeat,
                args.warmup,
            )

        if not args.hide_unavailable or any(
            value is not None
            for value in (
                row.rtta_update_ns,
                row.rtta_batch_ns,
                row.rtta_table_batch_ns,
                row.rtta_record_batch_ns,
                row.rtta_record_batch_input,
                row.talib_batch_ns,
                row.ta_batch_ns,
            )
        ):
            rows.append(row)

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rtta": getattr(rtta, "__version__", "installed"),
        "ta-lib": getattr(talib, "__version__", "not installed") if talib is not None else "not installed",
        "ta": getattr(ta_pkg, "__version__", "installed") if ta_pkg is not None else "not installed",
        "platform": platform.platform(),
    }
    return rows, versions


def format_ns(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 10:
        return f"{value:.2f}"
    if value < 100:
        return f"{value:.1f}"
    return f"{value:.0f}"


def write_markdown(rows: list[BenchResult], versions: dict[str, str], args: argparse.Namespace, output: Path | None) -> None:
    lines = [
        f"<!-- Generated by benchmarks/benchmark_indicators.py with samples={args.samples}, repeat={args.repeat}, warmup={args.warmup}. -->",
        f"<!-- Python {versions['python']}; NumPy {versions['numpy']}; TA-Lib {versions['ta-lib']}; ta {versions['ta']}; {versions['platform']} -->",
        "",
    ]

    batch_rows = [
        row
        for row in rows
        if any(value is not None for value in (row.rtta_batch_ns, row.rtta_table_batch_ns, row.rtta_record_batch_ns, row.talib_batch_ns, row.ta_batch_ns))
    ]
    if batch_rows:
        lines.extend(
            [
                "## Batch",
                "",
                "| Indicator | RTTA array batch ns/sample | RTTA table batch ns/sample | RTTA record batch ns/sample | RTTA record input | TA-Lib batch ns/sample | ta batch ns/sample |",
                "|---|---:|---:|---:|---|---:|---:|",
            ]
        )
        for row in batch_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.indicator,
                        format_ns(row.rtta_batch_ns),
                        format_ns(row.rtta_table_batch_ns),
                        format_ns(row.rtta_record_batch_ns),
                        row.rtta_record_batch_input or "n/a",
                        format_ns(row.talib_batch_ns),
                        format_ns(row.ta_batch_ns),
                    ]
                )
                + " |"
            )
        lines.append("")

    update_rows = [row for row in rows if row.rtta_update_ns is not None]
    if update_rows:
        lines.extend(
            [
                "## Updates",
                "",
                "| Indicator | RTTA update ns/sample |",
                "|---|---:|",
            ]
        )
        for row in update_rows:
            lines.append(f"| {row.indicator} | {format_ns(row.rtta_update_ns)} |")

    text = "\n".join(lines) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.write_text(text)


def write_csv(rows: list[BenchResult], output: Path | None) -> None:
    fieldnames = [
        "indicator",
        "rtta_update_ns_per_sample",
        "rtta_array_batch_ns_per_sample",
        "rtta_table_batch_ns_per_sample",
        "rtta_record_batch_ns_per_sample",
        "rtta_record_batch_input",
        "talib_batch_ns_per_sample",
        "ta_batch_ns_per_sample",
    ]
    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        write_rows(writer, rows)
    else:
        with output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            write_rows(writer, rows)


def write_rows(writer: csv.DictWriter[str], rows: list[BenchResult]) -> None:
    for row in rows:
        writer.writerow(
            {
                "indicator": row.indicator,
                "rtta_update_ns_per_sample": row.rtta_update_ns,
                "rtta_array_batch_ns_per_sample": row.rtta_batch_ns,
                "rtta_table_batch_ns_per_sample": row.rtta_table_batch_ns,
                "rtta_record_batch_ns_per_sample": row.rtta_record_batch_ns,
                "rtta_record_batch_input": row.rtta_record_batch_input,
                "talib_batch_ns_per_sample": row.talib_batch_ns,
                "ta_batch_ns_per_sample": row.ta_batch_ns,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RTTA indicators in nanoseconds per input sample.")
    parser.add_argument("--samples", type=int, default=200_000, help="Number of generated OHLCV samples.")
    parser.add_argument("--repeat", type=int, default=5, help="Benchmark repeats; the best run is reported.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generated market data.")
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown", help="Output format.")
    parser.add_argument("--output", type=Path, help="Output file. Defaults to stdout.")
    parser.add_argument("--library", choices=("all", "rtta", "talib", "ta"), default="all", help="Limit benchmarked libraries.")
    parser.add_argument("--hide-unavailable", action="store_true", help="Drop rows with no measured values for the selected libraries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, versions = run_benchmarks(args)
    if args.format == "csv":
        write_csv(rows, args.output)
    else:
        write_markdown(rows, versions, args, args.output)


if __name__ == "__main__":
    main()
