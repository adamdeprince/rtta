# Intel Xeon 6975P-C Benchmarks

These results were collected on 2026-07-19. Public documentation identifies benchmark systems by CPU type rather than hostname.

## System

- CPU: **Intel Xeon 6975P-C**
- Runtime processor string: `Intel(R) Xeon(R) 6975P-C`
- Architecture: `x86_64`
- Platform: `Linux-7.0.0-1004-aws-x86_64-with-glibc2.43`
- Python: `3.14.4`
- NumPy: `2.5.1`
- RTTA: `0.2.2`
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
- Median `advance(...)` latency, update state only: **35.9 ns/update**.
- Median `update(...)` latency, update state and return a value/result: **42.0 ns/update**.

| Algorithm | Inputs | update only: `advance(...)` ns/update | update + return: `update(...)` ns/update |
|---|---:|---:|---:|
| [ATR](../algorithms/atr.zh-CN.md) | 3 | 37.3 | 41.9 |
| [ATRP](../algorithms/atrp.zh-CN.md) | 3 | 37.9 | 41.5 |
| [ATRRegimeDetector](../algorithms/atr-regime-detector.zh-CN.md) | 3 | 39.4 | 43.0 |
| [ADWIN](../algorithms/adwin.zh-CN.md) | 1 | 423 | 430 |
| [EMA](../algorithms/ema.zh-CN.md) | 1 | 20.6 | 24.7 |
| [EWMA](../algorithms/ewma.zh-CN.md) | 1 | 20.0 | 24.5 |
| [EWMAZScoreShiftDetector](../algorithms/ewmaz-score-shift-detector.zh-CN.md) | 1 | 27.3 | 31.8 |
| [MACD](../algorithms/macd.zh-CN.md) | 1 | 22.4 | 26.5 |
| [ROC](../algorithms/roc.zh-CN.md) | 1 | 20.1 | 24.9 |
| [RSI](../algorithms/rsi.zh-CN.md) | 1 | 27.6 | 33.4 |
| [SMA](../algorithms/sma.zh-CN.md) | 1 | 21.4 | 25.3 |
| [TSI](../algorithms/tsi.zh-CN.md) | 1 | 23.9 | 28.7 |
| [AbsolutePriceOscillator](../algorithms/absolute-price-oscillator.zh-CN.md) | 1 | 21.0 | 25.1 |
| [AccumulationDistribution](../algorithms/accumulation-distribution.zh-CN.md) | 4 | 41.6 | 46.3 |
| [AlphaBetaGammaTrackingFilter](../algorithms/alpha-beta-gamma-tracking-filter.zh-CN.md) | 1 | 22.4 | 56.3 |
| [AmihudIlliquidity](../algorithms/amihud-illiquidity.zh-CN.md) | 2 | 30.6 | 34.8 |
| [AnchoredVWAP](../algorithms/anchored-vwap.zh-CN.md) | 5 | 48.0 | 53.8 |
| [Aroon](../algorithms/aroon.zh-CN.md) | 2 | 53.5 | 81.6 |
| [AroonOscillator](../algorithms/aroon-oscillator.zh-CN.md) | 2 | 53.7 | 59.2 |
| [AverageDirectionalMovementIndex](../algorithms/average-directional-movement-index.zh-CN.md) | 3 | 53.6 | 57.6 |
| [AverageDirectionalMovementIndexRating](../algorithms/average-directional-movement-index-rating.zh-CN.md) | 3 | 54.0 | 58.9 |
| [AveragePrice](../algorithms/average-price.zh-CN.md) | 4 | 40.4 | 46.3 |
| [AuctionContinuousMarketTransitionDetector](../algorithms/auction-continuous-market-transition-detector.zh-CN.md) | 1 | 19.9 | 24.7 |
| [AwesomeOscillator](../algorithms/awesome-oscillator.zh-CN.md) | 2 | 32.6 | 37.5 |
| [BalanceOfPower](../algorithms/balance-of-power.zh-CN.md) | 4 | 40.7 | 45.5 |
| [Beta](../algorithms/beta.zh-CN.md) | 2 | 31.4 | 36.0 |
| [BetaRegimeDetector](../algorithms/beta-regime-detector.zh-CN.md) | 2 | 33.1 | 36.9 |
| [BidAskBounceRegimeDetector](../algorithms/bid-ask-bounce-regime-detector.zh-CN.md) | 3 | 45.0 | 49.2 |
| [BollingerBands](../algorithms/bollinger-bands.zh-CN.md) | 1 | 22.2 | 59.9 |
| [BoundedBOCPD](../algorithms/bounded-bocpd.zh-CN.md) | 1 | 1126 | 1130 |
| [CalibrationDriftDetector](../algorithms/calibration-drift-detector.zh-CN.md) | 2 | 29.7 | 33.8 |
| [ChaikinMoneyFlow](../algorithms/chaikin-money-flow.zh-CN.md) | 4 | 43.8 | 48.8 |
| [ChaikinOscillator](../algorithms/chaikin-oscillator.zh-CN.md) | 4 | 42.5 | 47.7 |
| [ChandeMomentumOscillator](../algorithms/chande-momentum-oscillator.zh-CN.md) | 1 | 28.1 | 33.1 |
| [ChoppinessIndex](../algorithms/choppiness-index.zh-CN.md) | 3 | 75.6 | 80.2 |
| [ClosePressureReversalSignal](../algorithms/close-pressure-reversal-signal.zh-CN.md) | 5 | 113 | 141 |
| [CointegrationBreakdownMonitor](../algorithms/cointegration-breakdown-monitor.zh-CN.md) | 2 | 33.3 | 36.8 |
| [ConnorsRSI](../algorithms/connors-rsi.zh-CN.md) | 1 | 50.3 | 58.5 |
| [CommodityChannelIndex](../algorithms/commodity-channel-index.zh-CN.md) | 3 | 49.6 | 57.2 |
| [CoppockCurve](../algorithms/coppock-curve.zh-CN.md) | 1 | 24.0 | 28.2 |
| [Correlation](../algorithms/correlation.zh-CN.md) | 2 | 32.1 | 38.2 |
| [CorrelationRegimeDetector](../algorithms/correlation-regime-detector.zh-CN.md) | 2 | 36.0 | 38.8 |
| [CrossAssetCorrelationBreakDetector](../algorithms/cross-asset-correlation-break-detector.zh-CN.md) | 2 | 41.0 | 45.6 |
| [CumulativeReturn](../algorithms/cumulative-return.zh-CN.md) | 1 | 19.6 | 24.2 |
| [CUSUM](../algorithms/cusum.zh-CN.md) | 1 | 23.5 | 28.3 |
| [DDM](../algorithms/ddm.zh-CN.md) | 1 | 25.0 | 29.3 |
| [DailyLogReturn](../algorithms/daily-log-return.zh-CN.md) | 1 | 20.0 | 29.3 |
| [DailyReturn](../algorithms/daily-return.zh-CN.md) | 1 | 19.5 | 24.4 |
| [Delay](../algorithms/delay.zh-CN.md) | 1 | 21.1 | 24.5 |
| [DetrendedPriceOscillator](../algorithms/detrended-price-oscillator.zh-CN.md) | 1 | 22.3 | 26.5 |
| [DirectionalMovementIndex](../algorithms/directional-movement-index.zh-CN.md) | 3 | 48.9 | 54.6 |
| [DoubleEMA](../algorithms/double-ema.zh-CN.md) | 1 | 21.0 | 25.5 |
| [DonchianChannel](../algorithms/donchian-channel.zh-CN.md) | 3 | 61.3 | 93.0 |
| [EDDM](../algorithms/eddm.zh-CN.md) | 1 | 24.5 | 28.9 |
| [EhlersOptimalTrackingFilter](../algorithms/ehlers-optimal-tracking-filter.zh-CN.md) | 2 | 34.4 | 37.2 |
| [ElderRayIndex](../algorithms/elder-ray-index.zh-CN.md) | 3 | 36.4 | 66.6 |
| [EaseOfMovement](../algorithms/ease-of-movement.zh-CN.md) | 3 | 37.6 | 65.6 |
| [ExecutionCostSlippageRegimeDetector](../algorithms/execution-cost-slippage-regime-detector.zh-CN.md) | 3 | 38.6 | 43.0 |
| [FastStochastic](../algorithms/fast-stochastic.zh-CN.md) | 3 | 64.5 | 91.6 |
| [FeatureDistributionDriftDetector](../algorithms/feature-distribution-drift-detector.zh-CN.md) | 1 | 425 | 431 |
| [FibonacciRetracementLevels](../algorithms/fibonacci-retracement-levels.zh-CN.md) | 2 | 52.7 | 80.2 |
| [FisherTransform](../algorithms/fisher-transform.zh-CN.md) | 2 | 66.6 | 72.0 |
| [ForceIndex](../algorithms/force-index.zh-CN.md) | 2 | 29.1 | 33.1 |
| [FractalAdaptiveMovingAverage](../algorithms/fractal-adaptive-moving-average.zh-CN.md) | 1 | 65.1 | 71.1 |
| [GaussianProcessRegressionBands](../algorithms/gaussian-process-regression-bands.zh-CN.md) | 1 | 784 | 823 |
| [High](../algorithms/high.zh-CN.md) | 1 | 32.6 | 38.3 |
| [HighIndex](../algorithms/high-index.zh-CN.md) | 1 | 32.8 | 38.8 |
| [HighLow](../algorithms/high-low.zh-CN.md) | 1 | 40.4 | 75.3 |
| [HighLowIndex](../algorithms/high-low-index.zh-CN.md) | 1 | 40.0 | 73.6 |
| [HDDM](../algorithms/hddm.zh-CN.md) | 1 | 33.7 | 38.9 |
| [HeikinAshiTransform](../algorithms/heikin-ashi-transform.zh-CN.md) | 4 | 42.4 | 71.9 |
| [HiddenSemiMarkovRegimeFilter](../algorithms/hidden-semi-markov-regime-filter.zh-CN.md) | 1 | 83.9 | 87.7 |
| [HitRateDriftDetector](../algorithms/hit-rate-drift-detector.zh-CN.md) | 1 | 20.9 | 25.0 |
| [HullMovingAverage](../algorithms/hull-moving-average.zh-CN.md) | 1 | 25.5 | 29.5 |
| [Ichimoku](../algorithms/ichimoku.zh-CN.md) | 2 | 75.5 | 110 |
| [IntradayClockEchoSignal](../algorithms/intraday-clock-echo-signal.zh-CN.md) | 5 | 159 | 183 |
| [InteractingMultipleModelFilter](../algorithms/interacting-multiple-model-filter.zh-CN.md) | 1 | 163 | 192 |
| [KSTOscillator](../algorithms/kst-oscillator.zh-CN.md) | 1 | 31.6 | 68.4 |
| [KalmanExtremumTrend](../algorithms/kalman-extremum-trend.zh-CN.md) | 3 | 109 | 135 |
| [KalmanHedgeRatio](../algorithms/kalman-hedge-ratio.zh-CN.md) | 2 | 97.7 | 133 |
| [KalmanInnovationZScore](../algorithms/kalman-innovation-z-score.zh-CN.md) | 1 | 90.7 | 93.0 |
| [KalmanLocalLinearTrend](../algorithms/kalman-local-linear-trend.zh-CN.md) | 1 | 85.0 | 121 |
| [KalmanMovingAverage](../algorithms/kalman-moving-average.zh-CN.md) | 1 | 89.0 | 92.4 |
| [KalmanPredictionBands](../algorithms/kalman-prediction-bands.zh-CN.md) | 1 | 92.4 | 127 |
| [KalmanRegressionChannel](../algorithms/kalman-regression-channel.zh-CN.md) | 2 | 98.9 | 121 |
| [KalmanTrendSignal](../algorithms/kalman-trend-signal.zh-CN.md) | 1 | 84.2 | 120 |
| [KalmanVelocityOscillator](../algorithms/kalman-velocity-oscillator.zh-CN.md) | 1 | 88.6 | 92.8 |
| [Kama](../algorithms/kama.zh-CN.md) | 1 | 22.7 | 27.2 |
| [KeltnerChannel](../algorithms/keltner-channel.zh-CN.md) | 3 | 36.6 | 64.1 |
| [KeltnerChannelOriginal](../algorithms/keltner-channel-original.zh-CN.md) | 3 | 39.7 | 65.9 |
| [KlingerVolumeOscillator](../algorithms/klinger-volume-oscillator.zh-CN.md) | 4 | 53.1 | 79.2 |
| [KSWIN](../algorithms/kswin.zh-CN.md) | 1 | 2101 | 2107 |
| [KyleLambda](../algorithms/kyle-lambda.zh-CN.md) | 2 | 38.9 | 42.0 |
| [LeadLagRegimeDetector](../algorithms/lead-lag-regime-detector.zh-CN.md) | 2 | 31.1 | 35.1 |
| [LiquidityDroughtDetector](../algorithms/liquidity-drought-detector.zh-CN.md) | 3 | 39.8 | 43.3 |
| [LiquidityRegimeDetector](../algorithms/liquidity-regime-detector.zh-CN.md) | 2 | 30.4 | 34.2 |
| [LinearRegression](../algorithms/linear-regression.zh-CN.md) | 1 | 33.8 | 26.6 |
| [LinearRegressionAngle](../algorithms/linear-regression-angle.zh-CN.md) | 1 | 34.0 | 37.2 |
| [LinearRegressionIntercept](../algorithms/linear-regression-intercept.zh-CN.md) | 1 | 33.7 | 26.6 |
| [LinearRegressionSlope](../algorithms/linear-regression-slope.zh-CN.md) | 1 | 34.0 | 26.3 |
| [Low](../algorithms/low.zh-CN.md) | 1 | 33.5 | 38.1 |
| [LowIndex](../algorithms/low-index.zh-CN.md) | 1 | 32.8 | 38.8 |
| [MACDFix](../algorithms/macd-fix.zh-CN.md) | 1 | 22.5 | 26.5 |
| [MassIndex](../algorithms/mass-index.zh-CN.md) | 2 | 31.5 | 35.4 |
| [MarketOpenCloseTransitionDetector](../algorithms/market-open-close-transition-detector.zh-CN.md) | 1 | 20.7 | 24.8 |
| [MatchedFlowConformalSignal](../algorithms/matched-flow-conformal-signal.zh-CN.md) | 5 | 1153 | 1196 |
| [MedianPrice](../algorithms/median-price.zh-CN.md) | 2 | 28.2 | 32.3 |
| [MesaAdaptiveMovingAverage](../algorithms/mesa-adaptive-moving-average.zh-CN.md) | 1 | 74.4 | 113 |
| [MicrostructureNoiseRegimeDetector](../algorithms/microstructure-noise-regime-detector.zh-CN.md) | 3 | 37.1 | 41.9 |
| [MidPoint](../algorithms/mid-point.zh-CN.md) | 1 | 41.0 | 45.6 |
| [MidPrice](../algorithms/mid-price.zh-CN.md) | 2 | 54.7 | 59.7 |
| [MinusDirectionalIndicator](../algorithms/minus-directional-indicator.zh-CN.md) | 3 | 45.0 | 49.2 |
| [MinusDirectionalMovement](../algorithms/minus-directional-movement.zh-CN.md) | 2 | 39.3 | 40.6 |
| [Momentum](../algorithms/momentum.zh-CN.md) | 1 | 20.5 | 25.1 |
| [MoneyFlowIndex](../algorithms/money-flow-index.zh-CN.md) | 4 | 53.9 | 56.6 |
| [NadarayaWatsonEnvelope](../algorithms/nadaraya-watson-envelope.zh-CN.md) | 1 | 70.8 | 106 |
| [NegativeVolumeIndex](../algorithms/negative-volume-index.zh-CN.md) | 2 | 38.7 | 37.7 |
| [NormalizedATR](../algorithms/normalized-atr.zh-CN.md) | 3 | 38.8 | 41.8 |
| [OnBalanceVolume](../algorithms/on-balance-volume.zh-CN.md) | 2 | 35.4 | 39.4 |
| [OnlineGaussianMixtureRegimeFilter](../algorithms/online-gaussian-mixture-regime-filter.zh-CN.md) | 1 | 133 | 137 |
| [OnlineHMMRegimeFilter](../algorithms/online-hmm-regime-filter.zh-CN.md) | 1 | 73.9 | 78.1 |
| [OnlineMarkovSwitchingVolatilityFilter](../algorithms/online-markov-switching-volatility-filter.zh-CN.md) | 1 | 79.1 | 83.6 |
| [OrderFlowImbalance](../algorithms/order-flow-imbalance.zh-CN.md) | 4 | 49.2 | 54.4 |
| [OrderFlowImbalanceRegimeDetector](../algorithms/order-flow-imbalance-regime-detector.zh-CN.md) | 4 | 51.1 | 55.0 |
| [PageHinkley](../algorithms/page-hinkley.zh-CN.md) | 1 | 25.2 | 30.0 |
| [PairsSpreadRegimeDetector](../algorithms/pairs-spread-regime-detector.zh-CN.md) | 2 | 33.6 | 37.3 |
| [ParticleFilterTrend](../algorithms/particle-filter-trend.zh-CN.md) | 1 | 3544 | 3590 |
| [ParabolicSAR](../algorithms/parabolic-sar.zh-CN.md) | 2 | 29.6 | 33.8 |
| [PercentagePrice](../algorithms/percentage-price.zh-CN.md) | 1 | 21.8 | 54.0 |
| [PercentageVolume](../algorithms/percentage-volume.zh-CN.md) | 1 | 21.9 | 55.0 |
| [PlusDirectionalIndicator](../algorithms/plus-directional-indicator.zh-CN.md) | 3 | 45.3 | 49.7 |
| [PlusDirectionalMovement](../algorithms/plus-directional-movement.zh-CN.md) | 2 | 37.8 | 40.3 |
| [PredictionErrorDriftDetector](../algorithms/prediction-error-drift-detector.zh-CN.md) | 2 | 30.4 | 34.3 |
| [QuoteMessageRateRegimeDetector](../algorithms/quote-message-rate-regime-detector.zh-CN.md) | 1 | 26.3 | 30.2 |
| [QuoteStuffingDetector](../algorithms/quote-stuffing-detector.zh-CN.md) | 2 | 29.8 | 34.1 |
| [RateOfChangePercentage](../algorithms/rate-of-change-percentage.zh-CN.md) | 1 | 20.6 | 25.2 |
| [RateOfChangeRatio](../algorithms/rate-of-change-ratio.zh-CN.md) | 1 | 20.9 | 25.1 |
| [RateOfChangeRatio100](../algorithms/rate-of-change-ratio-100.zh-CN.md) | 1 | 20.8 | 25.0 |
| [RenkoBrickGenerator](../algorithms/renko-brick-generator.zh-CN.md) | 1 | 23.3 | 48.7 |
| [ResidualDriftDetector](../algorithms/residual-drift-detector.zh-CN.md) | 1 | 21.6 | 26.2 |
| [RelativeVigorIndex](../algorithms/relative-vigor-index.zh-CN.md) | 4 | 54.2 | 82.8 |
| [RealizedVarianceRegimeDetector](../algorithms/realized-variance-regime-detector.zh-CN.md) | 1 | 22.9 | 27.1 |
| [RollingBetaShiftDetector](../algorithms/rolling-beta-shift-detector.zh-CN.md) | 2 | 38.8 | 42.9 |
| [RollingCorrelationShiftDetector](../algorithms/rolling-correlation-shift-detector.zh-CN.md) | 2 | 42.2 | 47.5 |
| [RollingMeanShiftDetector](../algorithms/rolling-mean-shift-detector.zh-CN.md) | 1 | 31.8 | 41.8 |
| [RollingMeanVarianceShiftDetector](../algorithms/rolling-mean-variance-shift-detector.zh-CN.md) | 1 | 44.7 | 52.9 |
| [RollingSpreadLiquidityShiftDetector](../algorithms/rolling-spread-liquidity-shift-detector.zh-CN.md) | 4 | 47.1 | 51.0 |
| [RollingVarianceShiftDetector](../algorithms/rolling-variance-shift-detector.zh-CN.md) | 1 | 35.8 | 44.4 |
| [SavitzkyGolayFilter](../algorithms/savitzky-golay-filter.zh-CN.md) | 1 | 34.3 | 65.1 |
| [SchaffTrendCycle](../algorithms/schaff-trend-cycle.zh-CN.md) | 1 | 45.4 | 50.5 |
| [SpreadFeatures](../algorithms/spread-features.zh-CN.md) | 3 | 45.6 | 71.5 |
| [SpreadExplosionDetector](../algorithms/spread-explosion-detector.zh-CN.md) | 2 | 29.5 | 33.5 |
| [SpreadRegimeDetector](../algorithms/spread-regime-detector.zh-CN.md) | 2 | 32.4 | 37.4 |
| [StdDev](../algorithms/std-dev.zh-CN.md) | 1 | 22.2 | 26.2 |
| [StickyHMMRegimeFilter](../algorithms/sticky-hmm-regime-filter.zh-CN.md) | 1 | 74.2 | 78.3 |
| [StochRSI](../algorithms/stoch-rsi.zh-CN.md) | 1 | 57.2 | 63.3 |
| [Stochastic](../algorithms/stochastic.zh-CN.md) | 3 | 68.2 | 95.6 |
| [SuperTrend](../algorithms/super-trend.zh-CN.md) | 3 | 46.4 | 81.4 |
| [Summation](../algorithms/summation.zh-CN.md) | 1 | 20.9 | 25.0 |
| [T3MovingAverage](../algorithms/t-3-moving-average.zh-CN.md) | 1 | 25.1 | 29.1 |
| [ThresholdRegimeDetector](../algorithms/threshold-regime-detector.zh-CN.md) | 1 | 20.3 | 24.6 |
| [TimeSeriesForecast](../algorithms/time-series-forecast.zh-CN.md) | 1 | 33.7 | 26.5 |
| [TradeIntensityRegimeDetector](../algorithms/trade-intensity-regime-detector.zh-CN.md) | 1 | 28.0 | 32.4 |
| [TrendChopRegimeDetector](../algorithms/trend-chop-regime-detector.zh-CN.md) | 3 | 37.9 | 42.7 |
| [TwoFactorKalmanTrendFilter](../algorithms/two-factor-kalman-trend-filter.zh-CN.md) | 1 | 88.2 | 120 |
| [TrueRange](../algorithms/true-range.zh-CN.md) | 3 | 36.6 | 41.0 |
| [TriangularMovingAverage](../algorithms/triangular-moving-average.zh-CN.md) | 1 | 23.2 | 27.8 |
| [TripleEMA](../algorithms/triple-ema.zh-CN.md) | 1 | 21.5 | 26.8 |
| [Trix](../algorithms/trix.zh-CN.md) | 1 | 22.8 | 27.5 |
| [TypicalPrice](../algorithms/typical-price.zh-CN.md) | 3 | 35.2 | 40.6 |
| [UltimateOscillator](../algorithms/ultimate-oscillator.zh-CN.md) | 3 | 43.3 | 49.5 |
| [UlcerIndex](../algorithms/ulcer-index.zh-CN.md) | 1 | 34.4 | 39.7 |
| [VPIN](../algorithms/vpin.zh-CN.md) | 2 | 76.4 | 81.8 |
| [Variance](../algorithms/variance.zh-CN.md) | 1 | 21.1 | 25.7 |
| [VariableIndexDynamicAverage](../algorithms/variable-index-dynamic-average.zh-CN.md) | 1 | 29.6 | 34.0 |
| [VolatilityBreakoutDetector](../algorithms/volatility-breakout-detector.zh-CN.md) | 1 | 22.1 | 26.6 |
| [VolatilityCompressionExpansionDetector](../algorithms/volatility-compression-expansion-detector.zh-CN.md) | 1 | 23.9 | 28.9 |
| [VolatilityRegimeDetector](../algorithms/volatility-regime-detector.zh-CN.md) | 1 | 21.5 | 25.5 |
| [VolumeProfile](../algorithms/volume-profile.zh-CN.md) | 2 | 157 | 193 |
| [VolumePriceTrend](../algorithms/volume-price-trend.zh-CN.md) | 2 | 29.3 | 33.1 |
| [VolumeRegimeDetector](../algorithms/volume-regime-detector.zh-CN.md) | 1 | 28.0 | 32.0 |
| [VolumeWeightedAveragePrice](../algorithms/volume-weighted-average-price.zh-CN.md) | 4 | 43.2 | 47.2 |
| [VolumeWeightedMovingAverage](../algorithms/volume-weighted-moving-average.zh-CN.md) | 2 | 30.3 | 35.0 |
| [Vortex](../algorithms/vortex.zh-CN.md) | 3 | 39.6 | 67.6 |
| [WeightedClosePrice](../algorithms/weighted-close-price.zh-CN.md) | 3 | 35.9 | 40.5 |
| [WeightedMovingAverage](../algorithms/weighted-moving-average.zh-CN.md) | 1 | 21.6 | 25.5 |
| [WilliamsR](../algorithms/williams-r.zh-CN.md) | 3 | 60.1 | 65.5 |
| [ZigZagSwingDetector](../algorithms/zig-zag-swing-detector.zh-CN.md) | 1 | 23.0 | 55.6 |

