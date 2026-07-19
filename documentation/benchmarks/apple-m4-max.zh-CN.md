# Apple M4 Max 基准测试

以下结果采集于 2026-07-19。公开文档以 CPU 类型而不是主机名标识基准测试系统。

## 系统

- CPU： **Apple M4 Max**
- 运行时处理器字符串： `arm`
- 架构： `arm64`
- 平台： `macOS-26.5.1-arm64-arm-64bit-Mach-O`
- Python： `3.14.5`
- NumPy： `2.5.1`
- RTTA： `0.2.2`
- 样本数： `50000`
- 重复次数： `5`
- 预热重复次数： `1`
- 随机种子： `42`

运行命令：

```bash
python benchmarks/benchmark_readme.py --samples 50000 --repeat 5 --warmup 1 --output <benchmark-output.md>
```

## 延迟快照

- 注册表中的基准算法数: **188**.
- 展示的算法数: **188**.
- 中位 `advance(...)` 延迟（仅更新状态）: **28.5 ns/update**.
- 中位 `update(...)` 延迟（更新状态并返回值/结果）: **35.6 ns/update**.

| 算法 | 输入数 | 仅更新：`advance(...)` ns/update | 更新并返回：`update(...)` ns/update |
|---|---:|---:|---:|
| [ATR](../algorithms/atr.zh-CN.md) | 3 | 31.2 | 34.9 |
| [ATRP](../algorithms/atrp.zh-CN.md) | 3 | 29.9 | 36.0 |
| [ATRRegimeDetector](../algorithms/atr-regime-detector.zh-CN.md) | 3 | 32.6 | 35.9 |
| [ADWIN](../algorithms/adwin.zh-CN.md) | 1 | 284 | 301 |
| [EMA](../algorithms/ema.zh-CN.md) | 1 | 17.4 | 23.7 |
| [EWMA](../algorithms/ewma.zh-CN.md) | 1 | 18.0 | 22.4 |
| [EWMAZScoreShiftDetector](../algorithms/ewmaz-score-shift-detector.zh-CN.md) | 1 | 22.2 | 27.1 |
| [MACD](../algorithms/macd.zh-CN.md) | 1 | 23.1 | 24.6 |
| [ROC](../algorithms/roc.zh-CN.md) | 1 | 18.6 | 24.1 |
| [RSI](../algorithms/rsi.zh-CN.md) | 1 | 21.4 | 27.6 |
| [SMA](../algorithms/sma.zh-CN.md) | 1 | 19.1 | 23.3 |
| [TSI](../algorithms/tsi.zh-CN.md) | 1 | 22.1 | 25.9 |
| [AbsolutePriceOscillator](../algorithms/absolute-price-oscillator.zh-CN.md) | 1 | 18.4 | 24.4 |
| [AccumulationDistribution](../algorithms/accumulation-distribution.zh-CN.md) | 4 | 33.2 | 38.5 |
| [AlphaBetaGammaTrackingFilter](../algorithms/alpha-beta-gamma-tracking-filter.zh-CN.md) | 1 | 19.4 | 42.6 |
| [AmihudIlliquidity](../algorithms/amihud-illiquidity.zh-CN.md) | 2 | 26.1 | 33.2 |
| [AnchoredVWAP](../algorithms/anchored-vwap.zh-CN.md) | 5 | 41.4 | 43.3 |
| [Aroon](../algorithms/aroon.zh-CN.md) | 2 | 44.9 | 70.3 |
| [AroonOscillator](../algorithms/aroon-oscillator.zh-CN.md) | 2 | 44.7 | 49.5 |
| [AverageDirectionalMovementIndex](../algorithms/average-directional-movement-index.zh-CN.md) | 3 | 41.1 | 41.7 |
| [AverageDirectionalMovementIndexRating](../algorithms/average-directional-movement-index-rating.zh-CN.md) | 3 | 40.9 | 42.7 |
| [AveragePrice](../algorithms/average-price.zh-CN.md) | 4 | 32.7 | 37.4 |
| [AuctionContinuousMarketTransitionDetector](../algorithms/auction-continuous-market-transition-detector.zh-CN.md) | 1 | 18.2 | 24.4 |
| [AwesomeOscillator](../algorithms/awesome-oscillator.zh-CN.md) | 2 | 27.9 | 32.6 |
| [BalanceOfPower](../algorithms/balance-of-power.zh-CN.md) | 4 | 32.4 | 38.4 |
| [Beta](../algorithms/beta.zh-CN.md) | 2 | 26.2 | 33.7 |
| [BetaRegimeDetector](../algorithms/beta-regime-detector.zh-CN.md) | 2 | 26.8 | 35.2 |
| [BidAskBounceRegimeDetector](../algorithms/bid-ask-bounce-regime-detector.zh-CN.md) | 3 | 32.0 | 36.0 |
| [BollingerBands](../algorithms/bollinger-bands.zh-CN.md) | 1 | 19.2 | 45.9 |
| [BoundedBOCPD](../algorithms/bounded-bocpd.zh-CN.md) | 1 | 665 | 665 |
| [CalibrationDriftDetector](../algorithms/calibration-drift-detector.zh-CN.md) | 2 | 23.7 | 28.8 |
| [ChaikinMoneyFlow](../algorithms/chaikin-money-flow.zh-CN.md) | 4 | 34.9 | 40.8 |
| [ChaikinOscillator](../algorithms/chaikin-oscillator.zh-CN.md) | 4 | 35.1 | 40.3 |
| [ChandeMomentumOscillator](../algorithms/chande-momentum-oscillator.zh-CN.md) | 1 | 19.2 | 23.7 |
| [ChoppinessIndex](../algorithms/choppiness-index.zh-CN.md) | 3 | 55.3 | 64.1 |
| [ClosePressureReversalSignal](../algorithms/close-pressure-reversal-signal.zh-CN.md) | 5 | 81.5 | 105 |
| [CointegrationBreakdownMonitor](../algorithms/cointegration-breakdown-monitor.zh-CN.md) | 2 | 26.5 | 30.9 |
| [ConnorsRSI](../algorithms/connors-rsi.zh-CN.md) | 1 | 30.9 | 36.7 |
| [CommodityChannelIndex](../algorithms/commodity-channel-index.zh-CN.md) | 3 | 31.6 | 41.5 |
| [CoppockCurve](../algorithms/coppock-curve.zh-CN.md) | 1 | 19.3 | 24.7 |
| [Correlation](../algorithms/correlation.zh-CN.md) | 2 | 24.8 | 32.2 |
| [CorrelationRegimeDetector](../algorithms/correlation-regime-detector.zh-CN.md) | 2 | 28.4 | 35.7 |
| [CrossAssetCorrelationBreakDetector](../algorithms/cross-asset-correlation-break-detector.zh-CN.md) | 2 | 32.1 | 37.9 |
| [CumulativeReturn](../algorithms/cumulative-return.zh-CN.md) | 1 | 17.4 | 22.5 |
| [CUSUM](../algorithms/cusum.zh-CN.md) | 1 | 20.1 | 24.8 |
| [DDM](../algorithms/ddm.zh-CN.md) | 1 | 18.6 | 23.3 |
| [DailyLogReturn](../algorithms/daily-log-return.zh-CN.md) | 1 | 16.0 | 26.3 |
| [DailyReturn](../algorithms/daily-return.zh-CN.md) | 1 | 17.4 | 22.6 |
| [Delay](../algorithms/delay.zh-CN.md) | 1 | 18.0 | 22.3 |
| [DetrendedPriceOscillator](../algorithms/detrended-price-oscillator.zh-CN.md) | 1 | 19.4 | 23.9 |
| [DirectionalMovementIndex](../algorithms/directional-movement-index.zh-CN.md) | 3 | 35.9 | 39.3 |
| [DoubleEMA](../algorithms/double-ema.zh-CN.md) | 1 | 19.0 | 24.1 |
| [DonchianChannel](../algorithms/donchian-channel.zh-CN.md) | 3 | 51.0 | 76.5 |
| [EDDM](../algorithms/eddm.zh-CN.md) | 1 | 20.3 | 24.9 |
| [EhlersOptimalTrackingFilter](../algorithms/ehlers-optimal-tracking-filter.zh-CN.md) | 2 | 27.8 | 35.5 |
| [ElderRayIndex](../algorithms/elder-ray-index.zh-CN.md) | 3 | 30.4 | 53.6 |
| [EaseOfMovement](../algorithms/ease-of-movement.zh-CN.md) | 3 | 32.4 | 53.1 |
| [ExecutionCostSlippageRegimeDetector](../algorithms/execution-cost-slippage-regime-detector.zh-CN.md) | 3 | 31.1 | 35.9 |
| [FastStochastic](../algorithms/fast-stochastic.zh-CN.md) | 3 | 54.5 | 74.2 |
| [FeatureDistributionDriftDetector](../algorithms/feature-distribution-drift-detector.zh-CN.md) | 1 | 285 | 297 |
| [FibonacciRetracementLevels](../algorithms/fibonacci-retracement-levels.zh-CN.md) | 2 | 44.4 | 70.9 |
| [FisherTransform](../algorithms/fisher-transform.zh-CN.md) | 2 | 56.5 | 63.8 |
| [ForceIndex](../algorithms/force-index.zh-CN.md) | 2 | 23.4 | 30.0 |
| [FractalAdaptiveMovingAverage](../algorithms/fractal-adaptive-moving-average.zh-CN.md) | 1 | 45.5 | 48.0 |
| [GaussianProcessRegressionBands](../algorithms/gaussian-process-regression-bands.zh-CN.md) | 1 | 480 | 502 |
| [High](../algorithms/high.zh-CN.md) | 1 | 28.0 | 34.3 |
| [HighIndex](../algorithms/high-index.zh-CN.md) | 1 | 27.8 | 34.2 |
| [HighLow](../algorithms/high-low.zh-CN.md) | 1 | 34.8 | 56.6 |
| [HighLowIndex](../algorithms/high-low-index.zh-CN.md) | 1 | 33.2 | 59.8 |
| [HDDM](../algorithms/hddm.zh-CN.md) | 1 | 22.7 | 27.2 |
| [HeikinAshiTransform](../algorithms/heikin-ashi-transform.zh-CN.md) | 4 | 33.3 | 62.0 |
| [HiddenSemiMarkovRegimeFilter](../algorithms/hidden-semi-markov-regime-filter.zh-CN.md) | 1 | 58.2 | 65.6 |
| [HitRateDriftDetector](../algorithms/hit-rate-drift-detector.zh-CN.md) | 1 | 18.5 | 22.7 |
| [HullMovingAverage](../algorithms/hull-moving-average.zh-CN.md) | 1 | 20.8 | 25.7 |
| [Ichimoku](../algorithms/ichimoku.zh-CN.md) | 2 | 62.1 | 86.2 |
| [IntradayClockEchoSignal](../algorithms/intraday-clock-echo-signal.zh-CN.md) | 5 | 108 | 133 |
| [InteractingMultipleModelFilter](../algorithms/interacting-multiple-model-filter.zh-CN.md) | 1 | 103 | 137 |
| [KSTOscillator](../algorithms/kst-oscillator.zh-CN.md) | 1 | 26.2 | 52.4 |
| [KalmanExtremumTrend](../algorithms/kalman-extremum-trend.zh-CN.md) | 3 | 65.5 | 89.4 |
| [KalmanHedgeRatio](../algorithms/kalman-hedge-ratio.zh-CN.md) | 2 | 53.6 | 66.9 |
| [KalmanInnovationZScore](../algorithms/kalman-innovation-z-score.zh-CN.md) | 1 | 37.6 | 50.8 |
| [KalmanLocalLinearTrend](../algorithms/kalman-local-linear-trend.zh-CN.md) | 1 | 37.7 | 64.2 |
| [KalmanMovingAverage](../algorithms/kalman-moving-average.zh-CN.md) | 1 | 38.3 | 50.9 |
| [KalmanPredictionBands](../algorithms/kalman-prediction-bands.zh-CN.md) | 1 | 39.6 | 60.3 |
| [KalmanRegressionChannel](../algorithms/kalman-regression-channel.zh-CN.md) | 2 | 48.4 | 69.3 |
| [KalmanTrendSignal](../algorithms/kalman-trend-signal.zh-CN.md) | 1 | 36.8 | 60.0 |
| [KalmanVelocityOscillator](../algorithms/kalman-velocity-oscillator.zh-CN.md) | 1 | 37.6 | 48.7 |
| [Kama](../algorithms/kama.zh-CN.md) | 1 | 20.1 | 25.1 |
| [KeltnerChannel](../algorithms/keltner-channel.zh-CN.md) | 3 | 31.3 | 55.4 |
| [KeltnerChannelOriginal](../algorithms/keltner-channel-original.zh-CN.md) | 3 | 34.0 | 57.7 |
| [KlingerVolumeOscillator](../algorithms/klinger-volume-oscillator.zh-CN.md) | 4 | 40.0 | 66.7 |
| [KSWIN](../algorithms/kswin.zh-CN.md) | 1 | 1237 | 1252 |
| [KyleLambda](../algorithms/kyle-lambda.zh-CN.md) | 2 | 26.7 | 32.1 |
| [LeadLagRegimeDetector](../algorithms/lead-lag-regime-detector.zh-CN.md) | 2 | 27.0 | 31.3 |
| [LiquidityDroughtDetector](../algorithms/liquidity-drought-detector.zh-CN.md) | 3 | 31.6 | 36.3 |
| [LiquidityRegimeDetector](../algorithms/liquidity-regime-detector.zh-CN.md) | 2 | 25.5 | 30.9 |
| [LinearRegression](../algorithms/linear-regression.zh-CN.md) | 1 | 24.2 | 30.4 |
| [LinearRegressionAngle](../algorithms/linear-regression-angle.zh-CN.md) | 1 | 25.1 | 29.9 |
| [LinearRegressionIntercept](../algorithms/linear-regression-intercept.zh-CN.md) | 1 | 24.4 | 29.6 |
| [LinearRegressionSlope](../algorithms/linear-regression-slope.zh-CN.md) | 1 | 23.9 | 29.3 |
| [Low](../algorithms/low.zh-CN.md) | 1 | 27.9 | 33.9 |
| [LowIndex](../algorithms/low-index.zh-CN.md) | 1 | 28.4 | 33.9 |
| [MACDFix](../algorithms/macd-fix.zh-CN.md) | 1 | 20.1 | 24.7 |
| [MassIndex](../algorithms/mass-index.zh-CN.md) | 2 | 26.9 | 31.8 |
| [MarketOpenCloseTransitionDetector](../algorithms/market-open-close-transition-detector.zh-CN.md) | 1 | 17.8 | 24.4 |
| [MatchedFlowConformalSignal](../algorithms/matched-flow-conformal-signal.zh-CN.md) | 5 | 891 | 926 |
| [MedianPrice](../algorithms/median-price.zh-CN.md) | 2 | 25.0 | 29.0 |
| [MesaAdaptiveMovingAverage](../algorithms/mesa-adaptive-moving-average.zh-CN.md) | 1 | 52.8 | 80.3 |
| [MicrostructureNoiseRegimeDetector](../algorithms/microstructure-noise-regime-detector.zh-CN.md) | 3 | 31.9 | 35.4 |
| [MidPoint](../algorithms/mid-point.zh-CN.md) | 1 | 34.9 | 41.3 |
| [MidPrice](../algorithms/mid-price.zh-CN.md) | 2 | 45.6 | 51.7 |
| [MinusDirectionalIndicator](../algorithms/minus-directional-indicator.zh-CN.md) | 3 | 35.8 | 39.9 |
| [MinusDirectionalMovement](../algorithms/minus-directional-movement.zh-CN.md) | 2 | 25.4 | 30.0 |
| [Momentum](../algorithms/momentum.zh-CN.md) | 1 | 17.6 | 22.6 |
| [MoneyFlowIndex](../algorithms/money-flow-index.zh-CN.md) | 4 | 41.4 | 47.0 |
| [NadarayaWatsonEnvelope](../algorithms/nadaraya-watson-envelope.zh-CN.md) | 1 | 46.6 | 79.4 |
| [NegativeVolumeIndex](../algorithms/negative-volume-index.zh-CN.md) | 2 | 27.9 | 32.9 |
| [NormalizedATR](../algorithms/normalized-atr.zh-CN.md) | 3 | 30.5 | 36.0 |
| [OnBalanceVolume](../algorithms/on-balance-volume.zh-CN.md) | 2 | 29.6 | 34.1 |
| [OnlineGaussianMixtureRegimeFilter](../algorithms/online-gaussian-mixture-regime-filter.zh-CN.md) | 1 | 51.5 | 51.8 |
| [OnlineHMMRegimeFilter](../algorithms/online-hmm-regime-filter.zh-CN.md) | 1 | 60.0 | 66.2 |
| [OnlineMarkovSwitchingVolatilityFilter](../algorithms/online-markov-switching-volatility-filter.zh-CN.md) | 1 | 62.8 | 68.9 |
| [OrderFlowImbalance](../algorithms/order-flow-imbalance.zh-CN.md) | 4 | 39.3 | 44.3 |
| [OrderFlowImbalanceRegimeDetector](../algorithms/order-flow-imbalance-regime-detector.zh-CN.md) | 4 | 40.5 | 45.3 |
| [PageHinkley](../algorithms/page-hinkley.zh-CN.md) | 1 | 21.2 | 26.5 |
| [PairsSpreadRegimeDetector](../algorithms/pairs-spread-regime-detector.zh-CN.md) | 2 | 26.5 | 30.5 |
| [ParticleFilterTrend](../algorithms/particle-filter-trend.zh-CN.md) | 1 | 2139 | 2180 |
| [ParabolicSAR](../algorithms/parabolic-sar.zh-CN.md) | 2 | 26.1 | 30.3 |
| [PercentagePrice](../algorithms/percentage-price.zh-CN.md) | 1 | 20.2 | 44.4 |
| [PercentageVolume](../algorithms/percentage-volume.zh-CN.md) | 1 | 19.3 | 43.1 |
| [PlusDirectionalIndicator](../algorithms/plus-directional-indicator.zh-CN.md) | 3 | 36.0 | 38.2 |
| [PlusDirectionalMovement](../algorithms/plus-directional-movement.zh-CN.md) | 2 | 24.7 | 30.4 |
| [PredictionErrorDriftDetector](../algorithms/prediction-error-drift-detector.zh-CN.md) | 2 | 25.4 | 30.1 |
| [QuoteMessageRateRegimeDetector](../algorithms/quote-message-rate-regime-detector.zh-CN.md) | 1 | 20.8 | 27.4 |
| [QuoteStuffingDetector](../algorithms/quote-stuffing-detector.zh-CN.md) | 2 | 24.9 | 29.9 |
| [RateOfChangePercentage](../algorithms/rate-of-change-percentage.zh-CN.md) | 1 | 17.7 | 23.2 |
| [RateOfChangeRatio](../algorithms/rate-of-change-ratio.zh-CN.md) | 1 | 17.7 | 22.0 |
| [RateOfChangeRatio100](../algorithms/rate-of-change-ratio-100.zh-CN.md) | 1 | 17.5 | 23.0 |
| [RenkoBrickGenerator](../algorithms/renko-brick-generator.zh-CN.md) | 1 | 19.0 | 44.2 |
| [ResidualDriftDetector](../algorithms/residual-drift-detector.zh-CN.md) | 1 | 19.0 | 24.0 |
| [RelativeVigorIndex](../algorithms/relative-vigor-index.zh-CN.md) | 4 | 45.0 | 71.5 |
| [RealizedVarianceRegimeDetector](../algorithms/realized-variance-regime-detector.zh-CN.md) | 1 | 19.0 | 24.2 |
| [RollingBetaShiftDetector](../algorithms/rolling-beta-shift-detector.zh-CN.md) | 2 | 29.7 | 35.1 |
| [RollingCorrelationShiftDetector](../algorithms/rolling-correlation-shift-detector.zh-CN.md) | 2 | 33.1 | 40.5 |
| [RollingMeanShiftDetector](../algorithms/rolling-mean-shift-detector.zh-CN.md) | 1 | 24.3 | 29.8 |
| [RollingMeanVarianceShiftDetector](../algorithms/rolling-mean-variance-shift-detector.zh-CN.md) | 1 | 34.1 | 41.4 |
| [RollingSpreadLiquidityShiftDetector](../algorithms/rolling-spread-liquidity-shift-detector.zh-CN.md) | 4 | 38.4 | 42.5 |
| [RollingVarianceShiftDetector](../algorithms/rolling-variance-shift-detector.zh-CN.md) | 1 | 28.7 | 33.6 |
| [SavitzkyGolayFilter](../algorithms/savitzky-golay-filter.zh-CN.md) | 1 | 25.5 | 50.8 |
| [SchaffTrendCycle](../algorithms/schaff-trend-cycle.zh-CN.md) | 1 | 38.4 | 45.0 |
| [SpreadFeatures](../algorithms/spread-features.zh-CN.md) | 3 | 37.4 | 60.8 |
| [SpreadExplosionDetector](../algorithms/spread-explosion-detector.zh-CN.md) | 2 | 23.7 | 29.4 |
| [SpreadRegimeDetector](../algorithms/spread-regime-detector.zh-CN.md) | 2 | 26.6 | 31.5 |
| [StdDev](../algorithms/std-dev.zh-CN.md) | 1 | 17.7 | 24.3 |
| [StickyHMMRegimeFilter](../algorithms/sticky-hmm-regime-filter.zh-CN.md) | 1 | 57.8 | 62.2 |
| [StochRSI](../algorithms/stoch-rsi.zh-CN.md) | 1 | 46.0 | 52.7 |
| [Stochastic](../algorithms/stochastic.zh-CN.md) | 3 | 57.6 | 80.6 |
| [SuperTrend](../algorithms/super-trend.zh-CN.md) | 3 | 37.7 | 62.2 |
| [Summation](../algorithms/summation.zh-CN.md) | 1 | 17.7 | 23.4 |
| [T3MovingAverage](../algorithms/t-3-moving-average.zh-CN.md) | 1 | 23.3 | 28.1 |
| [ThresholdRegimeDetector](../algorithms/threshold-regime-detector.zh-CN.md) | 1 | 17.7 | 22.4 |
| [TimeSeriesForecast](../algorithms/time-series-forecast.zh-CN.md) | 1 | 23.8 | 29.8 |
| [TradeIntensityRegimeDetector](../algorithms/trade-intensity-regime-detector.zh-CN.md) | 1 | 22.9 | 27.4 |
| [TrendChopRegimeDetector](../algorithms/trend-chop-regime-detector.zh-CN.md) | 3 | 32.4 | 36.1 |
| [TwoFactorKalmanTrendFilter](../algorithms/two-factor-kalman-trend-filter.zh-CN.md) | 1 | 39.4 | 61.2 |
| [TrueRange](../algorithms/true-range.zh-CN.md) | 3 | 28.8 | 34.7 |
| [TriangularMovingAverage](../algorithms/triangular-moving-average.zh-CN.md) | 1 | 20.9 | 24.7 |
| [TripleEMA](../algorithms/triple-ema.zh-CN.md) | 1 | 20.4 | 24.9 |
| [Trix](../algorithms/trix.zh-CN.md) | 1 | 20.8 | 25.6 |
| [TypicalPrice](../algorithms/typical-price.zh-CN.md) | 3 | 29.9 | 34.9 |
| [UltimateOscillator](../algorithms/ultimate-oscillator.zh-CN.md) | 3 | 36.0 | 40.6 |
| [UlcerIndex](../algorithms/ulcer-index.zh-CN.md) | 1 | 29.4 | 34.2 |
| [VPIN](../algorithms/vpin.zh-CN.md) | 2 | 58.6 | 67.4 |
| [Variance](../algorithms/variance.zh-CN.md) | 1 | 18.5 | 23.7 |
| [VariableIndexDynamicAverage](../algorithms/variable-index-dynamic-average.zh-CN.md) | 1 | 19.4 | 25.5 |
| [VolatilityBreakoutDetector](../algorithms/volatility-breakout-detector.zh-CN.md) | 1 | 19.0 | 23.5 |
| [VolatilityCompressionExpansionDetector](../algorithms/volatility-compression-expansion-detector.zh-CN.md) | 1 | 19.2 | 24.5 |
| [VolatilityRegimeDetector](../algorithms/volatility-regime-detector.zh-CN.md) | 1 | 18.3 | 23.7 |
| [VolumeProfile](../algorithms/volume-profile.zh-CN.md) | 2 | 184 | 210 |
| [VolumePriceTrend](../algorithms/volume-price-trend.zh-CN.md) | 2 | 24.3 | 30.0 |
| [VolumeRegimeDetector](../algorithms/volume-regime-detector.zh-CN.md) | 1 | 20.2 | 27.3 |
| [VolumeWeightedAveragePrice](../algorithms/volume-weighted-average-price.zh-CN.md) | 4 | 34.7 | 39.1 |
| [VolumeWeightedMovingAverage](../algorithms/volume-weighted-moving-average.zh-CN.md) | 2 | 25.1 | 30.7 |
| [Vortex](../algorithms/vortex.zh-CN.md) | 3 | 32.6 | 58.1 |
| [WeightedClosePrice](../algorithms/weighted-close-price.zh-CN.md) | 3 | 29.7 | 35.1 |
| [WeightedMovingAverage](../algorithms/weighted-moving-average.zh-CN.md) | 1 | 18.2 | 23.5 |
| [WilliamsR](../algorithms/williams-r.zh-CN.md) | 3 | 49.8 | 56.4 |
| [ZigZagSwingDetector](../algorithms/zig-zag-swing-detector.zh-CN.md) | 1 | 19.9 | 45.0 |

