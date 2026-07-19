# Intel Xeon 6975P-C 基准测试

以下结果采集于 2026-06-09。公开文档以 CPU 类型而不是主机名标识基准测试系统。

## 系统

- CPU：**Intel Xeon 6975P-C**
- 运行时处理器字符串：`Intel(R) Xeon(R) 6975P-C`
- 架构：`x86_64`
- 平台：`Linux-7.0.0-1004-aws-x86_64-with-glibc2.43`
- Python：`3.14.4`
- NumPy：`2.4.6`
- RTTA：`0.2.1`
- 样本数：`50000`
- 重复次数：`5`
- 预热重复次数：`1`
- 随机种子：`42`

运行命令：

```bash
python benchmarks/benchmark_readme.py --samples 50000 --repeat 5 --warmup 1 --output <benchmark-output.md>
```

## 延迟概览

- 注册表中参与基准测试的算法：**188**。
- 本页列出的算法：**188**。
- 仅更新状态的 `advance(...)` 延迟中位数：**51.6 ns/update**。
- 更新状态并返回数值/结果的 `update(...)` 延迟中位数：**59.4 ns/update**。

| 算法 | 输入数 | 仅更新：`advance(...)` ns/update | 更新并返回：`update(...)` ns/update |
|---|---:|---:|---:|
| [ATR](../algorithms/atr.zh-CN.md) | 3 | 58.7 | 61.7 |
| [ATRP](../algorithms/atrp.zh-CN.md) | 3 | 59.2 | 61.4 |
| [ATRRegimeDetector](../algorithms/atr-regime-detector.zh-CN.md) | 3 | 60.7 | 63.5 |
| [ADWIN](../algorithms/adwin.zh-CN.md) | 1 | 955 | 956 |
| [EMA](../algorithms/ema.zh-CN.md) | 1 | 31.4 | 34.1 |
| [EWMA](../algorithms/ewma.zh-CN.md) | 1 | 29.7 | 33.9 |
| [EWMAZScoreShiftDetector](../algorithms/ewmaz-score-shift-detector.zh-CN.md) | 1 | 36.7 | 40.8 |
| [MACD](../algorithms/macd.zh-CN.md) | 1 | 35.3 | 39.3 |
| [ROC](../algorithms/roc.zh-CN.md) | 1 | 31.0 | 34.5 |
| [RSI](../algorithms/rsi.zh-CN.md) | 1 | 39.5 | 43.0 |
| [SMA](../algorithms/sma.zh-CN.md) | 1 | 31.1 | 34.0 |
| [TSI](../algorithms/tsi.zh-CN.md) | 1 | 39.0 | 42.5 |
| [AbsolutePriceOscillator](../algorithms/absolute-price-oscillator.zh-CN.md) | 1 | 34.0 | 36.1 |
| [AccumulationDistribution](../algorithms/accumulation-distribution.zh-CN.md) | 4 | 56.6 | 60.1 |
| [AlphaBetaGammaTrackingFilter](../algorithms/alpha-beta-gamma-tracking-filter.zh-CN.md) | 1 | 31.7 | 69.0 |
| [AmihudIlliquidity](../algorithms/amihud-illiquidity.zh-CN.md) | 2 | 44.1 | 46.1 |
| [AnchoredVWAP](../algorithms/anchored-vwap.zh-CN.md) | 5 | 65.0 | 68.2 |
| [Aroon](../algorithms/aroon.zh-CN.md) | 2 | 79.2 | 113 |
| [AroonOscillator](../algorithms/aroon-oscillator.zh-CN.md) | 2 | 79.2 | 84.1 |
| [AverageDirectionalMovementIndex](../algorithms/average-directional-movement-index.zh-CN.md) | 3 | 73.0 | 74.4 |
| [AverageDirectionalMovementIndexRating](../algorithms/average-directional-movement-index-rating.zh-CN.md) | 3 | 76.5 | 77.9 |
| [AveragePrice](../algorithms/average-price.zh-CN.md) | 4 | 55.0 | 59.0 |
| [AuctionContinuousMarketTransitionDetector](../algorithms/auction-continuous-market-transition-detector.zh-CN.md) | 1 | 30.2 | 34.4 |
| [AwesomeOscillator](../algorithms/awesome-oscillator.zh-CN.md) | 2 | 45.8 | 48.9 |
| [BalanceOfPower](../algorithms/balance-of-power.zh-CN.md) | 4 | 55.8 | 59.9 |
| [Beta](../algorithms/beta.zh-CN.md) | 2 | 51.2 | 53.0 |
| [BetaRegimeDetector](../algorithms/beta-regime-detector.zh-CN.md) | 2 | 53.0 | 55.9 |
| [BidAskBounceRegimeDetector](../algorithms/bid-ask-bounce-regime-detector.zh-CN.md) | 3 | 60.4 | 63.5 |
| [BollingerBands](../algorithms/bollinger-bands.zh-CN.md) | 1 | 38.3 | 77.8 |
| [BoundedBOCPD](../algorithms/bounded-bocpd.zh-CN.md) | 1 | 1592 | 1600 |
| [CalibrationDriftDetector](../algorithms/calibration-drift-detector.zh-CN.md) | 2 | 42.3 | 44.7 |
| [ChaikinMoneyFlow](../algorithms/chaikin-money-flow.zh-CN.md) | 4 | 119 | 121 |
| [ChaikinOscillator](../algorithms/chaikin-oscillator.zh-CN.md) | 4 | 60.2 | 64.8 |
| [ChandeMomentumOscillator](../algorithms/chande-momentum-oscillator.zh-CN.md) | 1 | 47.5 | 50.9 |
| [ChoppinessIndex](../algorithms/choppiness-index.zh-CN.md) | 3 | 120 | 122 |
| [ClosePressureReversalSignal](../algorithms/close-pressure-reversal-signal.zh-CN.md) | 5 | 155 | 186 |
| [CointegrationBreakdownMonitor](../algorithms/cointegration-breakdown-monitor.zh-CN.md) | 2 | 46.3 | 48.8 |
| [ConnorsRSI](../algorithms/connors-rsi.zh-CN.md) | 1 | 551 | 555 |
| [CommodityChannelIndex](../algorithms/commodity-channel-index.zh-CN.md) | 3 | 113 | 114 |
| [CoppockCurve](../algorithms/coppock-curve.zh-CN.md) | 1 | 39.8 | 43.3 |
| [Correlation](../algorithms/correlation.zh-CN.md) | 2 | 51.2 | 55.2 |
| [CorrelationRegimeDetector](../algorithms/correlation-regime-detector.zh-CN.md) | 2 | 55.0 | 58.1 |
| [CrossAssetCorrelationBreakDetector](../algorithms/cross-asset-correlation-break-detector.zh-CN.md) | 2 | 71.1 | 74.1 |
| [CumulativeReturn](../algorithms/cumulative-return.zh-CN.md) | 1 | 29.5 | 33.5 |
| [CUSUM](../algorithms/cusum.zh-CN.md) | 1 | 33.7 | 36.3 |
| [DDM](../algorithms/ddm.zh-CN.md) | 1 | 37.2 | 38.5 |
| [DailyLogReturn](../algorithms/daily-log-return.zh-CN.md) | 1 | 34.3 | 38.8 |
| [DailyReturn](../algorithms/daily-return.zh-CN.md) | 1 | 29.7 | 33.6 |
| [Delay](../algorithms/delay.zh-CN.md) | 1 | 29.9 | 33.8 |
| [DetrendedPriceOscillator](../algorithms/detrended-price-oscillator.zh-CN.md) | 1 | 34.1 | 38.7 |
| [DirectionalMovementIndex](../algorithms/directional-movement-index.zh-CN.md) | 3 | 69.5 | 72.1 |
| [DoubleEMA](../algorithms/double-ema.zh-CN.md) | 1 | 33.5 | 38.1 |
| [DonchianChannel](../algorithms/donchian-channel.zh-CN.md) | 3 | 89.5 | 132 |
| [EDDM](../algorithms/eddm.zh-CN.md) | 1 | 34.2 | 37.4 |
| [EhlersOptimalTrackingFilter](../algorithms/ehlers-optimal-tracking-filter.zh-CN.md) | 2 | 47.5 | 48.4 |
| [ElderRayIndex](../algorithms/elder-ray-index.zh-CN.md) | 3 | 52.7 | 74.7 |
| [EaseOfMovement](../algorithms/ease-of-movement.zh-CN.md) | 3 | 82.4 | 106 |
| [ExecutionCostSlippageRegimeDetector](../algorithms/execution-cost-slippage-regime-detector.zh-CN.md) | 3 | 51.8 | 56.4 |
| [FastStochastic](../algorithms/fast-stochastic.zh-CN.md) | 3 | 92.9 | 119 |
| [FeatureDistributionDriftDetector](../algorithms/feature-distribution-drift-detector.zh-CN.md) | 1 | 958 | 959 |
| [FibonacciRetracementLevels](../algorithms/fibonacci-retracement-levels.zh-CN.md) | 2 | 77.4 | 107 |
| [FisherTransform](../algorithms/fisher-transform.zh-CN.md) | 2 | 84.8 | 87.3 |
| [ForceIndex](../algorithms/force-index.zh-CN.md) | 2 | 40.0 | 43.5 |
| [FractalAdaptiveMovingAverage](../algorithms/fractal-adaptive-moving-average.zh-CN.md) | 1 | 145 | 147 |
| [GaussianProcessRegressionBands](../algorithms/gaussian-process-regression-bands.zh-CN.md) | 1 | 6040 | 6094 |
| [High](../algorithms/high.zh-CN.md) | 1 | 47.4 | 50.2 |
| [HighIndex](../algorithms/high-index.zh-CN.md) | 1 | 47.4 | 52.3 |
| [HighLow](../algorithms/high-low.zh-CN.md) | 1 | 61.2 | 95.0 |
| [HighLowIndex](../algorithms/high-low-index.zh-CN.md) | 1 | 61.3 | 94.7 |
| [HDDM](../algorithms/hddm.zh-CN.md) | 1 | 43.9 | 48.8 |
| [HeikinAshiTransform](../algorithms/heikin-ashi-transform.zh-CN.md) | 4 | 55.7 | 83.6 |
| [HiddenSemiMarkovRegimeFilter](../algorithms/hidden-semi-markov-regime-filter.zh-CN.md) | 1 | 77.9 | 81.1 |
| [HitRateDriftDetector](../algorithms/hit-rate-drift-detector.zh-CN.md) | 1 | 30.7 | 34.3 |
| [HullMovingAverage](../algorithms/hull-moving-average.zh-CN.md) | 1 | 47.0 | 50.8 |
| [Ichimoku](../algorithms/ichimoku.zh-CN.md) | 2 | 125 | 160 |
| [IntradayClockEchoSignal](../algorithms/intraday-clock-echo-signal.zh-CN.md) | 5 | 304 | 335 |
| [InteractingMultipleModelFilter](../algorithms/interacting-multiple-model-filter.zh-CN.md) | 1 | 194 | 221 |
| [KSTOscillator](../algorithms/kst-oscillator.zh-CN.md) | 1 | 46.8 | 84.2 |
| [KalmanExtremumTrend](../algorithms/kalman-extremum-trend.zh-CN.md) | 3 | 232 | 270 |
| [KalmanHedgeRatio](../algorithms/kalman-hedge-ratio.zh-CN.md) | 2 | 186 | 222 |
| [KalmanInnovationZScore](../algorithms/kalman-innovation-z-score.zh-CN.md) | 1 | 175 | 178 |
| [KalmanLocalLinearTrend](../algorithms/kalman-local-linear-trend.zh-CN.md) | 1 | 173 | 209 |
| [KalmanMovingAverage](../algorithms/kalman-moving-average.zh-CN.md) | 1 | 174 | 177 |
| [KalmanPredictionBands](../algorithms/kalman-prediction-bands.zh-CN.md) | 1 | 176 | 211 |
| [KalmanRegressionChannel](../algorithms/kalman-regression-channel.zh-CN.md) | 2 | 188 | 222 |
| [KalmanTrendSignal](../algorithms/kalman-trend-signal.zh-CN.md) | 1 | 177 | 212 |
| [KalmanVelocityOscillator](../algorithms/kalman-velocity-oscillator.zh-CN.md) | 1 | 172 | 176 |
| [Kama](../algorithms/kama.zh-CN.md) | 1 | 31.1 | 35.5 |
| [KeltnerChannel](../algorithms/keltner-channel.zh-CN.md) | 3 | 60.9 | 90.5 |
| [KeltnerChannelOriginal](../algorithms/keltner-channel-original.zh-CN.md) | 3 | 53.9 | 88.7 |
| [KlingerVolumeOscillator](../algorithms/klinger-volume-oscillator.zh-CN.md) | 4 | 75.3 | 95.1 |
| [KSWIN](../algorithms/kswin.zh-CN.md) | 1 | 2586 | 2594 |
| [KyleLambda](../algorithms/kyle-lambda.zh-CN.md) | 2 | 60.1 | 62.4 |
| [LeadLagRegimeDetector](../algorithms/lead-lag-regime-detector.zh-CN.md) | 2 | 43.1 | 47.2 |
| [LiquidityDroughtDetector](../algorithms/liquidity-drought-detector.zh-CN.md) | 3 | 52.1 | 55.5 |
| [LiquidityRegimeDetector](../algorithms/liquidity-regime-detector.zh-CN.md) | 2 | 42.2 | 44.6 |
| [LinearRegression](../algorithms/linear-regression.zh-CN.md) | 1 | 51.1 | 54.3 |
| [LinearRegressionAngle](../algorithms/linear-regression-angle.zh-CN.md) | 1 | 51.5 | 54.9 |
| [LinearRegressionIntercept](../algorithms/linear-regression-intercept.zh-CN.md) | 1 | 51.0 | 54.5 |
| [LinearRegressionSlope](../algorithms/linear-regression-slope.zh-CN.md) | 1 | 50.9 | 54.3 |
| [Low](../algorithms/low.zh-CN.md) | 1 | 47.2 | 50.7 |
| [LowIndex](../algorithms/low-index.zh-CN.md) | 1 | 48.0 | 52.1 |
| [MACDFix](../algorithms/macd-fix.zh-CN.md) | 1 | 34.8 | 38.7 |
| [MassIndex](../algorithms/mass-index.zh-CN.md) | 2 | 46.3 | 48.8 |
| [MarketOpenCloseTransitionDetector](../algorithms/market-open-close-transition-detector.zh-CN.md) | 1 | 30.8 | 34.6 |
| [MatchedFlowConformalSignal](../algorithms/matched-flow-conformal-signal.zh-CN.md) | 5 | 1820 | 1858 |
| [MedianPrice](../algorithms/median-price.zh-CN.md) | 2 | 39.3 | 43.4 |
| [MesaAdaptiveMovingAverage](../algorithms/mesa-adaptive-moving-average.zh-CN.md) | 1 | 139 | 183 |
| [MicrostructureNoiseRegimeDetector](../algorithms/microstructure-noise-regime-detector.zh-CN.md) | 3 | 49.9 | 53.6 |
| [MidPoint](../algorithms/mid-point.zh-CN.md) | 1 | 66.5 | 68.7 |
| [MidPrice](../algorithms/mid-price.zh-CN.md) | 2 | 81.4 | 84.8 |
| [MinusDirectionalIndicator](../algorithms/minus-directional-indicator.zh-CN.md) | 3 | 70.4 | 72.0 |
| [MinusDirectionalMovement](../algorithms/minus-directional-movement.zh-CN.md) | 2 | 50.4 | 52.4 |
| [Momentum](../algorithms/momentum.zh-CN.md) | 1 | 34.1 | 39.2 |
| [MoneyFlowIndex](../algorithms/money-flow-index.zh-CN.md) | 4 | 76.9 | 78.7 |
| [NadarayaWatsonEnvelope](../algorithms/nadaraya-watson-envelope.zh-CN.md) | 1 | 209 | 247 |
| [NegativeVolumeIndex](../algorithms/negative-volume-index.zh-CN.md) | 2 | 47.0 | 49.4 |
| [NormalizedATR](../algorithms/normalized-atr.zh-CN.md) | 3 | 58.8 | 62.5 |
| [OnBalanceVolume](../algorithms/on-balance-volume.zh-CN.md) | 2 | 45.9 | 49.1 |
| [OnlineGaussianMixtureRegimeFilter](../algorithms/online-gaussian-mixture-regime-filter.zh-CN.md) | 1 | 133 | 136 |
| [OnlineHMMRegimeFilter](../algorithms/online-hmm-regime-filter.zh-CN.md) | 1 | 77.7 | 81.9 |
| [OnlineMarkovSwitchingVolatilityFilter](../algorithms/online-markov-switching-volatility-filter.zh-CN.md) | 1 | 83.2 | 87.1 |
| [OrderFlowImbalance](../algorithms/order-flow-imbalance.zh-CN.md) | 4 | 65.8 | 71.3 |
| [OrderFlowImbalanceRegimeDetector](../algorithms/order-flow-imbalance-regime-detector.zh-CN.md) | 4 | 64.0 | 68.4 |
| [PageHinkley](../algorithms/page-hinkley.zh-CN.md) | 1 | 35.1 | 38.5 |
| [PairsSpreadRegimeDetector](../algorithms/pairs-spread-regime-detector.zh-CN.md) | 2 | 45.6 | 49.5 |
| [ParticleFilterTrend](../algorithms/particle-filter-trend.zh-CN.md) | 1 | 5019 | 5064 |
| [ParabolicSAR](../algorithms/parabolic-sar.zh-CN.md) | 2 | 47.7 | 51.1 |
| [PercentagePrice](../algorithms/percentage-price.zh-CN.md) | 1 | 34.2 | 68.9 |
| [PercentageVolume](../algorithms/percentage-volume.zh-CN.md) | 1 | 35.2 | 69.3 |
| [PlusDirectionalIndicator](../algorithms/plus-directional-indicator.zh-CN.md) | 3 | 70.4 | 73.0 |
| [PlusDirectionalMovement](../algorithms/plus-directional-movement.zh-CN.md) | 2 | 49.5 | 51.7 |
| [PredictionErrorDriftDetector](../algorithms/prediction-error-drift-detector.zh-CN.md) | 2 | 45.2 | 47.8 |
| [QuoteMessageRateRegimeDetector](../algorithms/quote-message-rate-regime-detector.zh-CN.md) | 1 | 35.4 | 40.0 |
| [QuoteStuffingDetector](../algorithms/quote-stuffing-detector.zh-CN.md) | 2 | 42.9 | 45.9 |
| [RateOfChangePercentage](../algorithms/rate-of-change-percentage.zh-CN.md) | 1 | 32.1 | 36.5 |
| [RateOfChangeRatio](../algorithms/rate-of-change-ratio.zh-CN.md) | 1 | 32.0 | 36.7 |
| [RateOfChangeRatio100](../algorithms/rate-of-change-ratio-100.zh-CN.md) | 1 | 33.0 | 37.3 |
| [RenkoBrickGenerator](../algorithms/renko-brick-generator.zh-CN.md) | 1 | 34.4 | 68.0 |
| [ResidualDriftDetector](../algorithms/residual-drift-detector.zh-CN.md) | 1 | 34.8 | 38.2 |
| [RelativeVigorIndex](../algorithms/relative-vigor-index.zh-CN.md) | 4 | 109 | 133 |
| [RealizedVarianceRegimeDetector](../algorithms/realized-variance-regime-detector.zh-CN.md) | 1 | 33.6 | 37.4 |
| [RollingBetaShiftDetector](../algorithms/rolling-beta-shift-detector.zh-CN.md) | 2 | 70.9 | 74.0 |
| [RollingCorrelationShiftDetector](../algorithms/rolling-correlation-shift-detector.zh-CN.md) | 2 | 73.2 | 77.7 |
| [RollingMeanShiftDetector](../algorithms/rolling-mean-shift-detector.zh-CN.md) | 1 | 49.2 | 53.1 |
| [RollingMeanVarianceShiftDetector](../algorithms/rolling-mean-variance-shift-detector.zh-CN.md) | 1 | 61.3 | 64.9 |
| [RollingSpreadLiquidityShiftDetector](../algorithms/rolling-spread-liquidity-shift-detector.zh-CN.md) | 4 | 71.4 | 77.2 |
| [RollingVarianceShiftDetector](../algorithms/rolling-variance-shift-detector.zh-CN.md) | 1 | 54.5 | 55.1 |
| [SavitzkyGolayFilter](../algorithms/savitzky-golay-filter.zh-CN.md) | 1 | 86.4 | 123 |
| [SchaffTrendCycle](../algorithms/schaff-trend-cycle.zh-CN.md) | 1 | 84.6 | 87.9 |
| [SpreadFeatures](../algorithms/spread-features.zh-CN.md) | 3 | 68.8 | 94.9 |
| [SpreadExplosionDetector](../algorithms/spread-explosion-detector.zh-CN.md) | 2 | 41.6 | 45.4 |
| [SpreadRegimeDetector](../algorithms/spread-regime-detector.zh-CN.md) | 2 | 45.8 | 48.4 |
| [StdDev](../algorithms/std-dev.zh-CN.md) | 1 | 35.8 | 39.6 |
| [StickyHMMRegimeFilter](../algorithms/sticky-hmm-regime-filter.zh-CN.md) | 1 | 77.5 | 82.9 |
| [StochRSI](../algorithms/stoch-rsi.zh-CN.md) | 1 | 79.0 | 82.3 |
| [Stochastic](../algorithms/stochastic.zh-CN.md) | 3 | 93.6 | 119 |
| [SuperTrend](../algorithms/super-trend.zh-CN.md) | 3 | 65.6 | 100 |
| [Summation](../algorithms/summation.zh-CN.md) | 1 | 30.2 | 33.9 |
| [T3MovingAverage](../algorithms/t-3-moving-average.zh-CN.md) | 1 | 42.4 | 45.4 |
| [ThresholdRegimeDetector](../algorithms/threshold-regime-detector.zh-CN.md) | 1 | 29.6 | 33.3 |
| [TimeSeriesForecast](../algorithms/time-series-forecast.zh-CN.md) | 1 | 51.1 | 54.4 |
| [TradeIntensityRegimeDetector](../algorithms/trade-intensity-regime-detector.zh-CN.md) | 1 | 38.0 | 41.3 |
| [TrendChopRegimeDetector](../algorithms/trend-chop-regime-detector.zh-CN.md) | 3 | 68.2 | 71.8 |
| [TwoFactorKalmanTrendFilter](../algorithms/two-factor-kalman-trend-filter.zh-CN.md) | 1 | 174 | 209 |
| [TrueRange](../algorithms/true-range.zh-CN.md) | 3 | 58.2 | 61.5 |
| [TriangularMovingAverage](../algorithms/triangular-moving-average.zh-CN.md) | 1 | 34.5 | 38.6 |
| [TripleEMA](../algorithms/triple-ema.zh-CN.md) | 1 | 34.5 | 38.6 |
| [Trix](../algorithms/trix.zh-CN.md) | 1 | 36.3 | 40.7 |
| [TypicalPrice](../algorithms/typical-price.zh-CN.md) | 3 | 49.8 | 52.6 |
| [UltimateOscillator](../algorithms/ultimate-oscillator.zh-CN.md) | 3 | 83.2 | 88.5 |
| [UlcerIndex](../algorithms/ulcer-index.zh-CN.md) | 1 | 54.2 | 57.7 |
| [VPIN](../algorithms/vpin.zh-CN.md) | 2 | 91.4 | 94.9 |
| [Variance](../algorithms/variance.zh-CN.md) | 1 | 35.1 | 39.1 |
| [VariableIndexDynamicAverage](../algorithms/variable-index-dynamic-average.zh-CN.md) | 1 | 47.5 | 50.7 |
| [VolatilityBreakoutDetector](../algorithms/volatility-breakout-detector.zh-CN.md) | 1 | 35.0 | 38.6 |
| [VolatilityCompressionExpansionDetector](../algorithms/volatility-compression-expansion-detector.zh-CN.md) | 1 | 36.1 | 40.2 |
| [VolatilityRegimeDetector](../algorithms/volatility-regime-detector.zh-CN.md) | 1 | 33.5 | 36.7 |
| [VolumeProfile](../algorithms/volume-profile.zh-CN.md) | 2 | 432 | 466 |
| [VolumePriceTrend](../algorithms/volume-price-trend.zh-CN.md) | 2 | 40.4 | 44.3 |
| [VolumeRegimeDetector](../algorithms/volume-regime-detector.zh-CN.md) | 1 | 37.1 | 41.0 |
| [VolumeWeightedAveragePrice](../algorithms/volume-weighted-average-price.zh-CN.md) | 4 | 107 | 108 |
| [VolumeWeightedMovingAverage](../algorithms/volume-weighted-moving-average.zh-CN.md) | 2 | 43.8 | 46.9 |
| [Vortex](../algorithms/vortex.zh-CN.md) | 3 | 64.7 | 97.2 |
| [WeightedClosePrice](../algorithms/weighted-close-price.zh-CN.md) | 3 | 48.4 | 51.9 |
| [WeightedMovingAverage](../algorithms/weighted-moving-average.zh-CN.md) | 1 | 33.4 | 38.5 |
| [WilliamsR](../algorithms/williams-r.zh-CN.md) | 3 | 86.4 | 90.0 |
| [ZigZagSwingDetector](../algorithms/zig-zag-swing-detector.zh-CN.md) | 1 | 32.8 | 67.6 |
