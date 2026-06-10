# Apple M4 Max Benchmarks

These results were collected on 2026-06-09. Public documentation identifies benchmark systems by CPU type rather than hostname.

## System

- CPU: **Apple M4 Max**
- Runtime processor string: `arm`
- Architecture: `arm64`
- Platform: `macOS-15.3.1-arm64-arm-64bit-Mach-O`
- Python: `3.14.5`
- NumPy: `2.4.6`
- RTTA: `0.2.1`
- Samples: `50000`
- Repeats: `5`
- Warmup repeats: `1`
- Seed: `42`

Run command:

```bash
python benchmarks/benchmark_readme.py --samples 50000 --repeat 5 --warmup 1 --output <benchmark-output.md>
```

## Latency Snapshot

- Benchmarked algorithms in registry: **188**.
- Algorithms shown: **188**.
- Median `advance(...)` latency, update state only: **37.9 ns/update**.
- Median `update(...)` latency, update state and return a value/result: **43.2 ns/update**.

| Algorithm | Inputs | update only: `advance(...)` ns/update | update + return: `update(...)` ns/update |
|---|---:|---:|---:|
| [ATR](../algorithms/atr.md) | 3 | 38.8 | 42.9 |
| [ATRP](../algorithms/atrp.md) | 3 | 39.0 | 43.6 |
| [ATRRegimeDetector](../algorithms/atr-regime-detector.md) | 3 | 41.2 | 44.6 |
| [ADWIN](../algorithms/adwin.md) | 1 | 333 | 349 |
| [EMA](../algorithms/ema.md) | 1 | 26.6 | 31.4 |
| [EWMA](../algorithms/ewma.md) | 1 | 25.1 | 29.4 |
| [EWMAZScoreShiftDetector](../algorithms/ewmaz-score-shift-detector.md) | 1 | 30.0 | 33.6 |
| [MACD](../algorithms/macd.md) | 1 | 29.4 | 33.0 |
| [ROC](../algorithms/roc.md) | 1 | 27.0 | 30.8 |
| [RSI](../algorithms/rsi.md) | 1 | 29.7 | 34.4 |
| [SMA](../algorithms/sma.md) | 1 | 26.8 | 30.9 |
| [TSI](../algorithms/tsi.md) | 1 | 32.1 | 35.6 |
| [AbsolutePriceOscillator](../algorithms/absolute-price-oscillator.md) | 1 | 27.9 | 31.9 |
| [AccumulationDistribution](../algorithms/accumulation-distribution.md) | 4 | 42.5 | 47.1 |
| [AlphaBetaGammaTrackingFilter](../algorithms/alpha-beta-gamma-tracking-filter.md) | 1 | 27.3 | 50.5 |
| [AmihudIlliquidity](../algorithms/amihud-illiquidity.md) | 2 | 34.3 | 39.7 |
| [AnchoredVWAP](../algorithms/anchored-vwap.md) | 5 | 50.4 | 54.9 |
| [Aroon](../algorithms/aroon.md) | 2 | 53.9 | 75.8 |
| [AroonOscillator](../algorithms/aroon-oscillator.md) | 2 | 53.2 | 57.8 |
| [AverageDirectionalMovementIndex](../algorithms/average-directional-movement-index.md) | 3 | 44.8 | 49.8 |
| [AverageDirectionalMovementIndexRating](../algorithms/average-directional-movement-index-rating.md) | 3 | 47.1 | 52.3 |
| [AveragePrice](../algorithms/average-price.md) | 4 | 41.8 | 45.7 |
| [AuctionContinuousMarketTransitionDetector](../algorithms/auction-continuous-market-transition-detector.md) | 1 | 26.1 | 29.7 |
| [AwesomeOscillator](../algorithms/awesome-oscillator.md) | 2 | 39.1 | 40.7 |
| [BalanceOfPower](../algorithms/balance-of-power.md) | 4 | 43.4 | 47.3 |
| [Beta](../algorithms/beta.md) | 2 | 34.6 | 40.3 |
| [BetaRegimeDetector](../algorithms/beta-regime-detector.md) | 2 | 37.8 | 42.5 |
| [BidAskBounceRegimeDetector](../algorithms/bid-ask-bounce-regime-detector.md) | 3 | 40.9 | 44.1 |
| [BollingerBands](../algorithms/bollinger-bands.md) | 1 | 28.7 | 53.0 |
| [BoundedBOCPD](../algorithms/bounded-bocpd.md) | 1 | 785 | 789 |
| [CalibrationDriftDetector](../algorithms/calibration-drift-detector.md) | 2 | 32.7 | 36.8 |
| [ChaikinMoneyFlow](../algorithms/chaikin-money-flow.md) | 4 | 80.4 | 83.2 |
| [ChaikinOscillator](../algorithms/chaikin-oscillator.md) | 4 | 43.9 | 48.2 |
| [ChandeMomentumOscillator](../algorithms/chande-momentum-oscillator.md) | 1 | 27.3 | 30.3 |
| [ChoppinessIndex](../algorithms/choppiness-index.md) | 3 | 65.8 | 71.3 |
| [ClosePressureReversalSignal](../algorithms/close-pressure-reversal-signal.md) | 5 | 81.9 | 104 |
| [CointegrationBreakdownMonitor](../algorithms/cointegration-breakdown-monitor.md) | 2 | 34.2 | 37.5 |
| [ConnorsRSI](../algorithms/connors-rsi.md) | 1 | 76.6 | 84.1 |
| [CommodityChannelIndex](../algorithms/commodity-channel-index.md) | 3 | 64.4 | 70.0 |
| [CoppockCurve](../algorithms/coppock-curve.md) | 1 | 28.5 | 32.0 |
| [Correlation](../algorithms/correlation.md) | 2 | 36.6 | 41.7 |
| [CorrelationRegimeDetector](../algorithms/correlation-regime-detector.md) | 2 | 38.0 | 42.5 |
| [CrossAssetCorrelationBreakDetector](../algorithms/cross-asset-correlation-break-detector.md) | 2 | 40.5 | 45.3 |
| [CumulativeReturn](../algorithms/cumulative-return.md) | 1 | 24.6 | 28.9 |
| [CUSUM](../algorithms/cusum.md) | 1 | 27.9 | 32.0 |
| [DDM](../algorithms/ddm.md) | 1 | 26.5 | 32.0 |
| [DailyLogReturn](../algorithms/daily-log-return.md) | 1 | 25.2 | 32.6 |
| [DailyReturn](../algorithms/daily-return.md) | 1 | 24.9 | 29.2 |
| [Delay](../algorithms/delay.md) | 1 | 25.9 | 29.2 |
| [DetrendedPriceOscillator](../algorithms/detrended-price-oscillator.md) | 1 | 28.5 | 31.3 |
| [DirectionalMovementIndex](../algorithms/directional-movement-index.md) | 3 | 42.3 | 46.7 |
| [DoubleEMA](../algorithms/double-ema.md) | 1 | 28.7 | 33.3 |
| [DonchianChannel](../algorithms/donchian-channel.md) | 3 | 60.9 | 81.8 |
| [EDDM](../algorithms/eddm.md) | 1 | 27.4 | 31.0 |
| [EhlersOptimalTrackingFilter](../algorithms/ehlers-optimal-tracking-filter.md) | 2 | 35.8 | 39.2 |
| [ElderRayIndex](../algorithms/elder-ray-index.md) | 3 | 39.0 | 58.9 |
| [EaseOfMovement](../algorithms/ease-of-movement.md) | 3 | 61.0 | 82.1 |
| [ExecutionCostSlippageRegimeDetector](../algorithms/execution-cost-slippage-regime-detector.md) | 3 | 39.3 | 43.2 |
| [FastStochastic](../algorithms/fast-stochastic.md) | 3 | 63.6 | 84.2 |
| [FeatureDistributionDriftDetector](../algorithms/feature-distribution-drift-detector.md) | 1 | 340 | 338 |
| [FibonacciRetracementLevels](../algorithms/fibonacci-retracement-levels.md) | 2 | 53.4 | 77.2 |
| [FisherTransform](../algorithms/fisher-transform.md) | 2 | 65.1 | 70.5 |
| [ForceIndex](../algorithms/force-index.md) | 2 | 33.5 | 38.6 |
| [FractalAdaptiveMovingAverage](../algorithms/fractal-adaptive-moving-average.md) | 1 | 65.9 | 72.5 |
| [GaussianProcessRegressionBands](../algorithms/gaussian-process-regression-bands.md) | 1 | 2307 | 2325 |
| [High](../algorithms/high.md) | 1 | 37.2 | 41.3 |
| [HighIndex](../algorithms/high-index.md) | 1 | 36.5 | 41.0 |
| [HighLow](../algorithms/high-low.md) | 1 | 42.9 | 66.0 |
| [HighLowIndex](../algorithms/high-low-index.md) | 1 | 43.3 | 68.1 |
| [HDDM](../algorithms/hddm.md) | 1 | 30.4 | 34.4 |
| [HeikinAshiTransform](../algorithms/heikin-ashi-transform.md) | 4 | 43.1 | 66.5 |
| [HiddenSemiMarkovRegimeFilter](../algorithms/hidden-semi-markov-regime-filter.md) | 1 | 63.1 | 68.8 |
| [HitRateDriftDetector](../algorithms/hit-rate-drift-detector.md) | 1 | 26.5 | 30.4 |
| [HullMovingAverage](../algorithms/hull-moving-average.md) | 1 | 30.1 | 34.0 |
| [Ichimoku](../algorithms/ichimoku.md) | 2 | 73.5 | 93.8 |
| [IntradayClockEchoSignal](../algorithms/intraday-clock-echo-signal.md) | 5 | 112 | 138 |
| [InteractingMultipleModelFilter](../algorithms/interacting-multiple-model-filter.md) | 1 | 114 | 140 |
| [KSTOscillator](../algorithms/kst-oscillator.md) | 1 | 36.2 | 59.6 |
| [KalmanExtremumTrend](../algorithms/kalman-extremum-trend.md) | 3 | 82.6 | 106 |
| [KalmanHedgeRatio](../algorithms/kalman-hedge-ratio.md) | 2 | 65.0 | 87.8 |
| [KalmanInnovationZScore](../algorithms/kalman-innovation-z-score.md) | 1 | 57.5 | 60.8 |
| [KalmanLocalLinearTrend](../algorithms/kalman-local-linear-trend.md) | 1 | 57.0 | 77.3 |
| [KalmanMovingAverage](../algorithms/kalman-moving-average.md) | 1 | 57.0 | 60.7 |
| [KalmanPredictionBands](../algorithms/kalman-prediction-bands.md) | 1 | 57.3 | 77.5 |
| [KalmanRegressionChannel](../algorithms/kalman-regression-channel.md) | 2 | 67.3 | 85.9 |
| [KalmanTrendSignal](../algorithms/kalman-trend-signal.md) | 1 | 57.3 | 77.6 |
| [KalmanVelocityOscillator](../algorithms/kalman-velocity-oscillator.md) | 1 | 55.7 | 59.6 |
| [Kama](../algorithms/kama.md) | 1 | 26.8 | 31.4 |
| [KeltnerChannel](../algorithms/keltner-channel.md) | 3 | 40.0 | 62.3 |
| [KeltnerChannelOriginal](../algorithms/keltner-channel-original.md) | 3 | 41.7 | 64.6 |
| [KlingerVolumeOscillator](../algorithms/klinger-volume-oscillator.md) | 4 | 49.0 | 71.4 |
| [KSWIN](../algorithms/kswin.md) | 1 | 1491 | 1497 |
| [KyleLambda](../algorithms/kyle-lambda.md) | 2 | 35.8 | 39.9 |
| [LeadLagRegimeDetector](../algorithms/lead-lag-regime-detector.md) | 2 | 35.6 | 40.0 |
| [LiquidityDroughtDetector](../algorithms/liquidity-drought-detector.md) | 3 | 43.8 | 49.4 |
| [LiquidityRegimeDetector](../algorithms/liquidity-regime-detector.md) | 2 | 34.9 | 39.8 |
| [LinearRegression](../algorithms/linear-regression.md) | 1 | 32.0 | 39.7 |
| [LinearRegressionAngle](../algorithms/linear-regression-angle.md) | 1 | 34.0 | 41.9 |
| [LinearRegressionIntercept](../algorithms/linear-regression-intercept.md) | 1 | 33.5 | 39.7 |
| [LinearRegressionSlope](../algorithms/linear-regression-slope.md) | 1 | 32.8 | 39.7 |
| [Low](../algorithms/low.md) | 1 | 37.4 | 40.8 |
| [LowIndex](../algorithms/low-index.md) | 1 | 37.5 | 40.8 |
| [MACDFix](../algorithms/macd-fix.md) | 1 | 29.7 | 33.5 |
| [MassIndex](../algorithms/mass-index.md) | 2 | 38.0 | 40.9 |
| [MarketOpenCloseTransitionDetector](../algorithms/market-open-close-transition-detector.md) | 1 | 26.2 | 30.4 |
| [MatchedFlowConformalSignal](../algorithms/matched-flow-conformal-signal.md) | 5 | 1278 | 1299 |
| [MedianPrice](../algorithms/median-price.md) | 2 | 31.4 | 36.2 |
| [MesaAdaptiveMovingAverage](../algorithms/mesa-adaptive-moving-average.md) | 1 | 73.5 | 99.4 |
| [MicrostructureNoiseRegimeDetector](../algorithms/microstructure-noise-regime-detector.md) | 3 | 39.6 | 43.8 |
| [MidPoint](../algorithms/mid-point.md) | 1 | 43.9 | 48.1 |
| [MidPrice](../algorithms/mid-price.md) | 2 | 55.3 | 59.5 |
| [MinusDirectionalIndicator](../algorithms/minus-directional-indicator.md) | 3 | 44.6 | 49.1 |
| [MinusDirectionalMovement](../algorithms/minus-directional-movement.md) | 2 | 33.4 | 34.9 |
| [Momentum](../algorithms/momentum.md) | 1 | 26.9 | 30.0 |
| [MoneyFlowIndex](../algorithms/money-flow-index.md) | 4 | 51.7 | 56.0 |
| [NadarayaWatsonEnvelope](../algorithms/nadaraya-watson-envelope.md) | 1 | 90.2 | 114 |
| [NegativeVolumeIndex](../algorithms/negative-volume-index.md) | 2 | 37.8 | 38.0 |
| [NormalizedATR](../algorithms/normalized-atr.md) | 3 | 39.0 | 42.3 |
| [OnBalanceVolume](../algorithms/on-balance-volume.md) | 2 | 36.4 | 41.0 |
| [OnlineGaussianMixtureRegimeFilter](../algorithms/online-gaussian-mixture-regime-filter.md) | 1 | 52.8 | 56.8 |
| [OnlineHMMRegimeFilter](../algorithms/online-hmm-regime-filter.md) | 1 | 63.8 | 63.8 |
| [OnlineMarkovSwitchingVolatilityFilter](../algorithms/online-markov-switching-volatility-filter.md) | 1 | 67.1 | 72.2 |
| [OrderFlowImbalance](../algorithms/order-flow-imbalance.md) | 4 | 47.9 | 52.2 |
| [OrderFlowImbalanceRegimeDetector](../algorithms/order-flow-imbalance-regime-detector.md) | 4 | 49.6 | 53.6 |
| [PageHinkley](../algorithms/page-hinkley.md) | 1 | 29.3 | 33.2 |
| [PairsSpreadRegimeDetector](../algorithms/pairs-spread-regime-detector.md) | 2 | 35.6 | 39.0 |
| [ParticleFilterTrend](../algorithms/particle-filter-trend.md) | 1 | 2353 | 2376 |
| [ParabolicSAR](../algorithms/parabolic-sar.md) | 2 | 34.3 | 38.2 |
| [PercentagePrice](../algorithms/percentage-price.md) | 1 | 31.4 | 53.2 |
| [PercentageVolume](../algorithms/percentage-volume.md) | 1 | 30.5 | 52.4 |
| [PlusDirectionalIndicator](../algorithms/plus-directional-indicator.md) | 3 | 42.2 | 47.2 |
| [PlusDirectionalMovement](../algorithms/plus-directional-movement.md) | 2 | 34.0 | 37.3 |
| [PredictionErrorDriftDetector](../algorithms/prediction-error-drift-detector.md) | 2 | 36.0 | 38.1 |
| [QuoteMessageRateRegimeDetector](../algorithms/quote-message-rate-regime-detector.md) | 1 | 31.3 | 34.2 |
| [QuoteStuffingDetector](../algorithms/quote-stuffing-detector.md) | 2 | 34.6 | 37.9 |
| [RateOfChangePercentage](../algorithms/rate-of-change-percentage.md) | 1 | 27.1 | 31.2 |
| [RateOfChangeRatio](../algorithms/rate-of-change-ratio.md) | 1 | 27.4 | 30.9 |
| [RateOfChangeRatio100](../algorithms/rate-of-change-ratio-100.md) | 1 | 27.3 | 31.4 |
| [RenkoBrickGenerator](../algorithms/renko-brick-generator.md) | 1 | 26.6 | 49.4 |
| [ResidualDriftDetector](../algorithms/residual-drift-detector.md) | 1 | 27.5 | 31.4 |
| [RelativeVigorIndex](../algorithms/relative-vigor-index.md) | 4 | 63.0 | 84.2 |
| [RealizedVarianceRegimeDetector](../algorithms/realized-variance-regime-detector.md) | 1 | 26.6 | 30.6 |
| [RollingBetaShiftDetector](../algorithms/rolling-beta-shift-detector.md) | 2 | 39.5 | 42.7 |
| [RollingCorrelationShiftDetector](../algorithms/rolling-correlation-shift-detector.md) | 2 | 40.0 | 44.1 |
| [RollingMeanShiftDetector](../algorithms/rolling-mean-shift-detector.md) | 1 | 39.0 | 40.2 |
| [RollingMeanVarianceShiftDetector](../algorithms/rolling-mean-variance-shift-detector.md) | 1 | 47.4 | 54.6 |
| [RollingSpreadLiquidityShiftDetector](../algorithms/rolling-spread-liquidity-shift-detector.md) | 4 | 47.7 | 52.2 |
| [RollingVarianceShiftDetector](../algorithms/rolling-variance-shift-detector.md) | 1 | 39.5 | 46.5 |
| [SavitzkyGolayFilter](../algorithms/savitzky-golay-filter.md) | 1 | 34.2 | 56.7 |
| [SchaffTrendCycle](../algorithms/schaff-trend-cycle.md) | 1 | 54.6 | 56.8 |
| [SpreadFeatures](../algorithms/spread-features.md) | 3 | 46.5 | 69.0 |
| [SpreadExplosionDetector](../algorithms/spread-explosion-detector.md) | 2 | 33.5 | 37.4 |
| [SpreadRegimeDetector](../algorithms/spread-regime-detector.md) | 2 | 36.1 | 40.6 |
| [StdDev](../algorithms/std-dev.md) | 1 | 27.3 | 31.4 |
| [StickyHMMRegimeFilter](../algorithms/sticky-hmm-regime-filter.md) | 1 | 62.6 | 69.0 |
| [StochRSI](../algorithms/stoch-rsi.md) | 1 | 56.6 | 61.1 |
| [Stochastic](../algorithms/stochastic.md) | 3 | 68.3 | 90.6 |
| [SuperTrend](../algorithms/super-trend.md) | 3 | 46.4 | 70.0 |
| [Summation](../algorithms/summation.md) | 1 | 26.1 | 29.7 |
| [T3MovingAverage](../algorithms/t-3-moving-average.md) | 1 | 32.8 | 36.5 |
| [ThresholdRegimeDetector](../algorithms/threshold-regime-detector.md) | 1 | 25.4 | 29.3 |
| [TimeSeriesForecast](../algorithms/time-series-forecast.md) | 1 | 33.7 | 42.6 |
| [TradeIntensityRegimeDetector](../algorithms/trade-intensity-regime-detector.md) | 1 | 30.9 | 34.7 |
| [TrendChopRegimeDetector](../algorithms/trend-chop-regime-detector.md) | 3 | 41.0 | 44.1 |
| [TwoFactorKalmanTrendFilter](../algorithms/two-factor-kalman-trend-filter.md) | 1 | 57.1 | 77.3 |
| [TrueRange](../algorithms/true-range.md) | 3 | 38.9 | 43.3 |
| [TriangularMovingAverage](../algorithms/triangular-moving-average.md) | 1 | 29.3 | 32.4 |
| [TripleEMA](../algorithms/triple-ema.md) | 1 | 30.2 | 34.6 |
| [Trix](../algorithms/trix.md) | 1 | 31.0 | 34.6 |
| [TypicalPrice](../algorithms/typical-price.md) | 3 | 37.1 | 41.6 |
| [UltimateOscillator](../algorithms/ultimate-oscillator.md) | 3 | 43.1 | 46.7 |
| [UlcerIndex](../algorithms/ulcer-index.md) | 1 | 37.1 | 41.1 |
| [VPIN](../algorithms/vpin.md) | 2 | 69.2 | 73.3 |
| [Variance](../algorithms/variance.md) | 1 | 26.3 | 31.1 |
| [VariableIndexDynamicAverage](../algorithms/variable-index-dynamic-average.md) | 1 | 28.6 | 32.4 |
| [VolatilityBreakoutDetector](../algorithms/volatility-breakout-detector.md) | 1 | 27.5 | 30.2 |
| [VolatilityCompressionExpansionDetector](../algorithms/volatility-compression-expansion-detector.md) | 1 | 27.5 | 31.6 |
| [VolatilityRegimeDetector](../algorithms/volatility-regime-detector.md) | 1 | 26.9 | 30.4 |
| [VolumeProfile](../algorithms/volume-profile.md) | 2 | 370 | 399 |
| [VolumePriceTrend](../algorithms/volume-price-trend.md) | 2 | 35.2 | 37.6 |
| [VolumeRegimeDetector](../algorithms/volume-regime-detector.md) | 1 | 31.3 | 34.6 |
| [VolumeWeightedAveragePrice](../algorithms/volume-weighted-average-price.md) | 4 | 71.6 | 75.3 |
| [VolumeWeightedMovingAverage](../algorithms/volume-weighted-moving-average.md) | 2 | 33.1 | 37.8 |
| [Vortex](../algorithms/vortex.md) | 3 | 38.8 | 64.7 |
| [WeightedClosePrice](../algorithms/weighted-close-price.md) | 3 | 37.8 | 42.8 |
| [WeightedMovingAverage](../algorithms/weighted-moving-average.md) | 1 | 26.2 | 29.9 |
| [WilliamsR](../algorithms/williams-r.md) | 3 | 61.6 | 65.3 |
| [ZigZagSwingDetector](../algorithms/zig-zag-swing-detector.md) | 1 | 28.2 | 50.9 |
