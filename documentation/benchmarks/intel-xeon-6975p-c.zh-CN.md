# Intel Xeon 6975P-C 基准测试

以下结果采集于 2026-07-19。公开文档以 CPU 类型而不是主机名标识基准测试系统。

## 系统

- CPU： **Intel Xeon 6975P-C**
- 运行时处理器字符串： `Intel(R) Xeon(R) 6975P-C`
- 架构： `x86_64`
- 平台： `Linux-7.0.0-1004-aws-x86_64-with-glibc2.43`
- Python： `3.14.4`
- NumPy： `2.5.1`
- RTTA： `0.2.3`
- 样本数： `50000`
- 重复次数： `5`
- 预热重复次数： `1`
- 随机种子： `42`

运行命令：

```bash
python benchmarks/benchmark_readme.py --samples 50000 --repeat 5 --warmup 1 --output <benchmark-output.md>
```

## 延迟快照

- 注册表中的基准算法数: **247**.
- 展示的算法数: **247**.
- 中位 `advance(...)` 延迟（仅更新状态）: **39.2 ns/update**.
- 中位 `update(...)` 延迟（更新状态并返回值/结果）: **52.2 ns/update**.

| 算法 | 输入数 | 仅更新：`advance(...)` ns/update | 更新并返回：`update(...)` ns/update |
|---|---:|---:|---:|
| [AbsolutePriceOscillator](../algorithms/absolute-price-oscillator.zh-CN.md) | 1 | 23.7 | 28.4 |
| [AccelerationBands](../algorithms/acceleration-bands.zh-CN.md) | 3 | 48.3 | 73.4 |
| [AcceleratorOscillator](../algorithms/accelerator-oscillator.zh-CN.md) | 2 | 39.2 | 44.8 |
| [AccumulationDistribution](../algorithms/accumulation-distribution.zh-CN.md) | 4 | 53.1 | 59.3 |
| [AccumulativeSwingIndex](../algorithms/accumulative-swing-index.zh-CN.md) | 4 | 66.3 | 65.1 |
| [ADWIN](../algorithms/adwin.zh-CN.md) | 1 | 426 | 432 |
| [Alligator](../algorithms/alligator.zh-CN.md) | 2 | 39.2 | 72.1 |
| [AlphaBetaGammaTrackingFilter](../algorithms/alpha-beta-gamma-tracking-filter.zh-CN.md) | 1 | 24.7 | 56.2 |
| [AmihudIlliquidity](../algorithms/amihud-illiquidity.zh-CN.md) | 2 | 36.4 | 40.9 |
| [AnchoredVWAP](../algorithms/anchored-vwap.zh-CN.md) | 5 | 63.1 | 67.7 |
| [ArnaudLegouxMovingAverage](../algorithms/arnaud-legoux-moving-average.zh-CN.md) | 1 | 27.5 | 31.5 |
| [Aroon](../algorithms/aroon.zh-CN.md) | 2 | 59.3 | 84.1 |
| [AroonOscillator](../algorithms/aroon-oscillator.zh-CN.md) | 2 | 59.1 | 64.8 |
| [ATR](../algorithms/atr.zh-CN.md) | 3 | 46.6 | 51.5 |
| [ATRP](../algorithms/atrp.zh-CN.md) | 3 | 47.9 | 51.1 |
| [ATRRegimeDetector](../algorithms/atr-regime-detector.zh-CN.md) | 3 | 49.1 | 51.8 |
| [AuctionContinuousMarketTransitionDetector](../algorithms/auction-continuous-market-transition-detector.zh-CN.md) | 1 | 23.4 | 28.7 |
| [AverageDirectionalMovementIndex](../algorithms/average-directional-movement-index.zh-CN.md) | 3 | 56.0 | 60.5 |
| [AverageDirectionalMovementIndexRating](../algorithms/average-directional-movement-index-rating.zh-CN.md) | 3 | 58.3 | 61.1 |
| [AveragePrice](../algorithms/average-price.zh-CN.md) | 4 | 52.4 | 57.6 |
| [AwesomeOscillator](../algorithms/awesome-oscillator.zh-CN.md) | 2 | 38.9 | 43.5 |
| [BalanceOfPower](../algorithms/balance-of-power.zh-CN.md) | 4 | 52.2 | 58.9 |
| [Beta](../algorithms/beta.zh-CN.md) | 2 | 35.8 | 41.2 |
| [BetaRegimeDetector](../algorithms/beta-regime-detector.zh-CN.md) | 2 | 37.6 | 42.6 |
| [Bias](../algorithms/bias.zh-CN.md) | 1 | 23.9 | 29.2 |
| [BidAskBounceRegimeDetector](../algorithms/bid-ask-bounce-regime-detector.zh-CN.md) | 3 | 52.7 | 57.8 |
| [BollingerBands](../algorithms/bollinger-bands.zh-CN.md) | 1 | 24.0 | 59.5 |
| [BollingerBandwidth](../algorithms/bollinger-bandwidth.zh-CN.md) | 1 | 25.3 | 29.9 |
| [BollingerPercentB](../algorithms/bollinger-percent-b.zh-CN.md) | 1 | 25.1 | 29.7 |
| [BoundedBOCPD](../algorithms/bounded-bocpd.zh-CN.md) | 1 | 1130 | 1133 |
| [CalibrationDriftDetector](../algorithms/calibration-drift-detector.zh-CN.md) | 2 | 35.9 | 39.8 |
| [ChaikinMoneyFlow](../algorithms/chaikin-money-flow.zh-CN.md) | 4 | 55.8 | 60.8 |
| [ChaikinOscillator](../algorithms/chaikin-oscillator.zh-CN.md) | 4 | 56.6 | 59.7 |
| [ChaikinVolatility](../algorithms/chaikin-volatility.zh-CN.md) | 2 | 36.9 | 42.5 |
| [ChandelierExit](../algorithms/chandelier-exit.zh-CN.md) | 3 | 69.1 | 97.8 |
| [ChandeMomentumOscillator](../algorithms/chande-momentum-oscillator.zh-CN.md) | 1 | 30.0 | 35.0 |
| [ChoppinessIndex](../algorithms/choppiness-index.zh-CN.md) | 3 | 80.5 | 85.3 |
| [ClosePressureReversalSignal](../algorithms/close-pressure-reversal-signal.zh-CN.md) | 5 | 137 | 167 |
| [CointegrationBreakdownMonitor](../algorithms/cointegration-breakdown-monitor.zh-CN.md) | 2 | 38.5 | 42.9 |
| [CommodityChannelIndex](../algorithms/commodity-channel-index.zh-CN.md) | 3 | 52.1 | 56.1 |
| [ComparativeRelativeStrength](../algorithms/comparative-relative-strength.zh-CN.md) | 2 | 34.6 | 40.5 |
| [ConnorsRSI](../algorithms/connors-rsi.zh-CN.md) | 1 | 51.8 | 59.8 |
| [CoppockCurve](../algorithms/coppock-curve.zh-CN.md) | 1 | 27.1 | 31.3 |
| [Correlation](../algorithms/correlation.zh-CN.md) | 2 | 37.6 | 42.5 |
| [CorrelationRegimeDetector](../algorithms/correlation-regime-detector.zh-CN.md) | 2 | 39.4 | 41.9 |
| [CrossAssetCorrelationBreakDetector](../algorithms/cross-asset-correlation-break-detector.zh-CN.md) | 2 | 42.8 | 47.6 |
| [CumulativeReturn](../algorithms/cumulative-return.zh-CN.md) | 1 | 22.5 | 27.3 |
| [CUSUM](../algorithms/cusum.zh-CN.md) | 1 | 24.8 | 31.3 |
| [DailyLogReturn](../algorithms/daily-log-return.zh-CN.md) | 1 | 22.6 | 31.8 |
| [DailyReturn](../algorithms/daily-return.zh-CN.md) | 1 | 22.9 | 27.4 |
| [DDM](../algorithms/ddm.zh-CN.md) | 1 | 25.6 | 31.3 |
| [Delay](../algorithms/delay.zh-CN.md) | 1 | 22.2 | 27.9 |
| [DeMarker](../algorithms/de-marker.zh-CN.md) | 2 | 37.1 | 42.3 |
| [DetrendedPriceOscillator](../algorithms/detrended-price-oscillator.zh-CN.md) | 1 | 24.2 | 29.6 |
| [DirectionalMovementIndex](../algorithms/directional-movement-index.zh-CN.md) | 3 | 56.4 | 59.4 |
| [DonchianChannel](../algorithms/donchian-channel.zh-CN.md) | 3 | 69.8 | 99.4 |
| [DoubleEMA](../algorithms/double-ema.zh-CN.md) | 1 | 23.0 | 28.7 |
| [EaseOfMovement](../algorithms/ease-of-movement.zh-CN.md) | 3 | 48.9 | 70.4 |
| [EDDM](../algorithms/eddm.zh-CN.md) | 1 | 26.4 | 30.9 |
| [EfficiencyRatio](../algorithms/efficiency-ratio.zh-CN.md) | 1 | 23.5 | 29.3 |
| [EhlersCenterOfGravity](../algorithms/ehlers-center-of-gravity.zh-CN.md) | 1 | 29.5 | 62.5 |
| [EhlersCyberCycle](../algorithms/ehlers-cyber-cycle.zh-CN.md) | 1 | 25.2 | 56.6 |
| [EhlersDecycler](../algorithms/ehlers-decycler.zh-CN.md) | 1 | 23.5 | 55.6 |
| [EhlersInstantaneousTrendline](../algorithms/ehlers-instantaneous-trendline.zh-CN.md) | 1 | 23.0 | 56.1 |
| [EhlersOptimalTrackingFilter](../algorithms/ehlers-optimal-tracking-filter.zh-CN.md) | 2 | 36.8 | 40.8 |
| [EhlersRoofingFilter](../algorithms/ehlers-roofing-filter.zh-CN.md) | 1 | 23.6 | 57.1 |
| [EhlersSuperSmoother](../algorithms/ehlers-super-smoother.zh-CN.md) | 1 | 22.3 | 28.5 |
| [ElderRayIndex](../algorithms/elder-ray-index.zh-CN.md) | 3 | 45.7 | 68.7 |
| [EMA](../algorithms/ema.zh-CN.md) | 1 | 22.5 | 28.0 |
| [EWMA](../algorithms/ewma.zh-CN.md) | 1 | 22.0 | 27.6 |
| [EWMAZScoreShiftDetector](../algorithms/ewmaz-score-shift-detector.zh-CN.md) | 1 | 29.9 | 35.1 |
| [ExecutionCostSlippageRegimeDetector](../algorithms/execution-cost-slippage-regime-detector.zh-CN.md) | 3 | 47.3 | 52.2 |
| [FastStochastic](../algorithms/fast-stochastic.zh-CN.md) | 3 | 72.8 | 99.6 |
| [FeatureDistributionDriftDetector](../algorithms/feature-distribution-drift-detector.zh-CN.md) | 1 | 426 | 432 |
| [FibonacciRetracementLevels](../algorithms/fibonacci-retracement-levels.zh-CN.md) | 2 | 57.5 | 84.0 |
| [FisherTransform](../algorithms/fisher-transform.zh-CN.md) | 2 | 68.2 | 72.2 |
| [ForceIndex](../algorithms/force-index.zh-CN.md) | 2 | 35.9 | 40.4 |
| [FractalAdaptiveMovingAverage](../algorithms/fractal-adaptive-moving-average.zh-CN.md) | 1 | 67.3 | 71.9 |
| [GatorOscillator](../algorithms/gator-oscillator.zh-CN.md) | 2 | 40.2 | 66.8 |
| [GaussianProcessRegressionBands](../algorithms/gaussian-process-regression-bands.zh-CN.md) | 1 | 785 | 825 |
| [GeometricMovingAverage](../algorithms/geometric-moving-average.zh-CN.md) | 1 | 27.2 | 37.6 |
| [GuppyMultipleMovingAverage](../algorithms/guppy-multiple-moving-average.zh-CN.md) | 1 | 29.9 | 61.6 |
| [HDDM](../algorithms/hddm.zh-CN.md) | 1 | 35.7 | 40.1 |
| [HeikinAshiTransform](../algorithms/heikin-ashi-transform.zh-CN.md) | 4 | 54.6 | 80.0 |
| [HiddenSemiMarkovRegimeFilter](../algorithms/hidden-semi-markov-regime-filter.zh-CN.md) | 1 | 82.9 | 87.7 |
| [High](../algorithms/high.zh-CN.md) | 1 | 35.3 | 40.9 |
| [HighIndex](../algorithms/high-index.zh-CN.md) | 1 | 35.1 | 40.4 |
| [HighLow](../algorithms/high-low.zh-CN.md) | 1 | 42.7 | 74.1 |
| [HighLowIndex](../algorithms/high-low-index.zh-CN.md) | 1 | 42.3 | 74.5 |
| [HilbertDominantCyclePeriod](../algorithms/hilbert-dominant-cycle-period.zh-CN.md) | 1 | 426 | 428 |
| [HilbertDominantCyclePhase](../algorithms/hilbert-dominant-cycle-phase.zh-CN.md) | 1 | 426 | 427 |
| [HilbertPhasor](../algorithms/hilbert-phasor.zh-CN.md) | 1 | 426 | 457 |
| [HilbertSineWave](../algorithms/hilbert-sine-wave.zh-CN.md) | 1 | 425 | 458 |
| [HilbertTrendline](../algorithms/hilbert-trendline.zh-CN.md) | 1 | 426 | 428 |
| [HilbertTrendMode](../algorithms/hilbert-trend-mode.zh-CN.md) | 1 | 426 | 428 |
| [HistoricalVolatility](../algorithms/historical-volatility.zh-CN.md) | 1 | 28.6 | 36.0 |
| [HitRateDriftDetector](../algorithms/hit-rate-drift-detector.zh-CN.md) | 1 | 22.8 | 28.2 |
| [HullMovingAverage](../algorithms/hull-moving-average.zh-CN.md) | 1 | 28.8 | 31.9 |
| [Ichimoku](../algorithms/ichimoku.zh-CN.md) | 3 | 91.7 | 116 |
| [InteractingMultipleModelFilter](../algorithms/interacting-multiple-model-filter.zh-CN.md) | 1 | 167 | 195 |
| [IntradayClockEchoSignal](../algorithms/intraday-clock-echo-signal.zh-CN.md) | 5 | 184 | 212 |
| [IntradayIntensity](../algorithms/intraday-intensity.zh-CN.md) | 4 | 56.1 | 60.3 |
| [IntradayMomentumIndex](../algorithms/intraday-momentum-index.zh-CN.md) | 2 | 44.1 | 47.8 |
| [InverseFisherRSI](../algorithms/inverse-fisher-rsi.zh-CN.md) | 1 | 33.2 | 54.6 |
| [KagiChart](../algorithms/kagi-chart.zh-CN.md) | 1 | 28.5 | 60.1 |
| [KalmanExtremumTrend](../algorithms/kalman-extremum-trend.zh-CN.md) | 3 | 112 | 143 |
| [KalmanHedgeRatio](../algorithms/kalman-hedge-ratio.zh-CN.md) | 2 | 96.0 | 131 |
| [KalmanInnovationZScore](../algorithms/kalman-innovation-z-score.zh-CN.md) | 1 | 87.8 | 91.0 |
| [KalmanLocalLinearTrend](../algorithms/kalman-local-linear-trend.zh-CN.md) | 1 | 82.4 | 121 |
| [KalmanMovingAverage](../algorithms/kalman-moving-average.zh-CN.md) | 1 | 82.2 | 85.8 |
| [KalmanPredictionBands](../algorithms/kalman-prediction-bands.zh-CN.md) | 1 | 99.6 | 124 |
| [KalmanRegressionChannel](../algorithms/kalman-regression-channel.zh-CN.md) | 2 | 99.4 | 124 |
| [KalmanTrendSignal](../algorithms/kalman-trend-signal.zh-CN.md) | 1 | 89.3 | 118 |
| [KalmanVelocityOscillator](../algorithms/kalman-velocity-oscillator.zh-CN.md) | 1 | 81.8 | 85.9 |
| [Kama](../algorithms/kama.zh-CN.md) | 1 | 24.2 | 29.1 |
| [KeltnerChannel](../algorithms/keltner-channel.zh-CN.md) | 3 | 46.7 | 73.9 |
| [KeltnerChannelOriginal](../algorithms/keltner-channel-original.zh-CN.md) | 3 | 49.0 | 76.6 |
| [KlingerVolumeOscillator](../algorithms/klinger-volume-oscillator.zh-CN.md) | 4 | 64.3 | 87.3 |
| [KSTOscillator](../algorithms/kst-oscillator.zh-CN.md) | 1 | 32.6 | 72.3 |
| [KSWIN](../algorithms/kswin.zh-CN.md) | 1 | 2109 | 2114 |
| [KyleLambda](../algorithms/kyle-lambda.zh-CN.md) | 2 | 42.7 | 47.3 |
| [LeadLagRegimeDetector](../algorithms/lead-lag-regime-detector.zh-CN.md) | 2 | 37.8 | 42.4 |
| [LinearRegression](../algorithms/linear-regression.zh-CN.md) | 1 | 35.7 | 29.0 |
| [LinearRegressionAngle](../algorithms/linear-regression-angle.zh-CN.md) | 1 | 35.6 | 39.8 |
| [LinearRegressionIntercept](../algorithms/linear-regression-intercept.zh-CN.md) | 1 | 35.5 | 29.5 |
| [LinearRegressionSlope](../algorithms/linear-regression-slope.zh-CN.md) | 1 | 35.3 | 29.1 |
| [LiquidityDroughtDetector](../algorithms/liquidity-drought-detector.zh-CN.md) | 3 | 47.5 | 52.5 |
| [LiquidityRegimeDetector](../algorithms/liquidity-regime-detector.zh-CN.md) | 2 | 36.0 | 41.5 |
| [Low](../algorithms/low.zh-CN.md) | 1 | 34.8 | 39.9 |
| [LowIndex](../algorithms/low-index.zh-CN.md) | 1 | 34.3 | 40.5 |
| [MACD](../algorithms/macd.zh-CN.md) | 1 | 25.0 | 56.3 |
| [MACDExt](../algorithms/macd-ext.zh-CN.md) | 1 | 26.3 | 56.3 |
| [MACDFix](../algorithms/macd-fix.zh-CN.md) | 1 | 24.5 | 56.0 |
| [MarketFacilitationIndex](../algorithms/market-facilitation-index.zh-CN.md) | 3 | 44.2 | 50.0 |
| [MarketOpenCloseTransitionDetector](../algorithms/market-open-close-transition-detector.zh-CN.md) | 1 | 22.7 | 29.1 |
| [MassIndex](../algorithms/mass-index.zh-CN.md) | 2 | 36.8 | 42.6 |
| [MatchedFlowConformalSignal](../algorithms/matched-flow-conformal-signal.zh-CN.md) | 5 | 1177 | 1204 |
| [McGinleyDynamic](../algorithms/mc-ginley-dynamic.zh-CN.md) | 1 | 33.8 | 42.8 |
| [MedianPrice](../algorithms/median-price.zh-CN.md) | 2 | 33.9 | 39.5 |
| [MesaAdaptiveMovingAverage](../algorithms/mesa-adaptive-moving-average.zh-CN.md) | 1 | 76.0 | 114 |
| [MicrostructureNoiseRegimeDetector](../algorithms/microstructure-noise-regime-detector.zh-CN.md) | 3 | 47.4 | 51.7 |
| [MidPoint](../algorithms/mid-point.zh-CN.md) | 1 | 43.3 | 47.0 |
| [MidPrice](../algorithms/mid-price.zh-CN.md) | 2 | 59.5 | 65.3 |
| [MinusDirectionalIndicator](../algorithms/minus-directional-indicator.zh-CN.md) | 3 | 53.7 | 58.4 |
| [MinusDirectionalMovement](../algorithms/minus-directional-movement.zh-CN.md) | 2 | 41.9 | 47.5 |
| [Momentum](../algorithms/momentum.zh-CN.md) | 1 | 23.0 | 28.5 |
| [MoneyFlowIndex](../algorithms/money-flow-index.zh-CN.md) | 4 | 64.2 | 72.3 |
| [MovingAverageEnvelope](../algorithms/moving-average-envelope.zh-CN.md) | 1 | 25.0 | 57.8 |
| [NadarayaWatsonEnvelope](../algorithms/nadaraya-watson-envelope.zh-CN.md) | 1 | 72.0 | 108 |
| [NegativeVolumeIndex](../algorithms/negative-volume-index.zh-CN.md) | 2 | 40.4 | 43.4 |
| [NormalizedATR](../algorithms/normalized-atr.zh-CN.md) | 3 | 46.2 | 51.0 |
| [OnBalanceVolume](../algorithms/on-balance-volume.zh-CN.md) | 2 | 39.6 | 44.4 |
| [OnlineGaussianMixtureRegimeFilter](../algorithms/online-gaussian-mixture-regime-filter.zh-CN.md) | 1 | 133 | 138 |
| [OnlineHMMRegimeFilter](../algorithms/online-hmm-regime-filter.zh-CN.md) | 1 | 78.2 | 81.4 |
| [OnlineMarkovSwitchingVolatilityFilter](../algorithms/online-markov-switching-volatility-filter.zh-CN.md) | 1 | 82.5 | 85.4 |
| [OrderFlowImbalance](../algorithms/order-flow-imbalance.zh-CN.md) | 4 | 63.4 | 69.1 |
| [OrderFlowImbalanceRegimeDetector](../algorithms/order-flow-imbalance-regime-detector.zh-CN.md) | 4 | 62.2 | 67.4 |
| [PageHinkley](../algorithms/page-hinkley.zh-CN.md) | 1 | 27.9 | 32.4 |
| [PairsSpreadRegimeDetector](../algorithms/pairs-spread-regime-detector.zh-CN.md) | 2 | 37.6 | 42.5 |
| [ParabolicSAR](../algorithms/parabolic-sar.zh-CN.md) | 2 | 36.7 | 41.6 |
| [ParabolicSARExtended](../algorithms/parabolic-sar-extended.zh-CN.md) | 2 | 36.9 | 40.8 |
| [ParticleFilterTrend](../algorithms/particle-filter-trend.zh-CN.md) | 1 | 3528 | 3573 |
| [PercentagePrice](../algorithms/percentage-price.zh-CN.md) | 1 | 25.5 | 55.8 |
| [PercentageVolume](../algorithms/percentage-volume.zh-CN.md) | 1 | 24.6 | 56.3 |
| [PivotPoints](../algorithms/pivot-points.zh-CN.md) | 3 | 45.4 | 72.2 |
| [PlusDirectionalIndicator](../algorithms/plus-directional-indicator.zh-CN.md) | 3 | 58.1 | 58.9 |
| [PlusDirectionalMovement](../algorithms/plus-directional-movement.zh-CN.md) | 2 | 41.5 | 46.5 |
| [PointAndFigure](../algorithms/point-and-figure.zh-CN.md) | 1 | 24.8 | 60.2 |
| [PositiveVolumeIndex](../algorithms/positive-volume-index.zh-CN.md) | 2 | 41.7 | 44.2 |
| [PredictionErrorDriftDetector](../algorithms/prediction-error-drift-detector.zh-CN.md) | 2 | 37.2 | 42.1 |
| [PrettyGoodOscillator](../algorithms/pretty-good-oscillator.zh-CN.md) | 3 | 47.8 | 52.0 |
| [PsychologicalLine](../algorithms/psychological-line.zh-CN.md) | 1 | 29.0 | 34.4 |
| [QStick](../algorithms/q-stick.zh-CN.md) | 2 | 35.3 | 40.3 |
| [QuoteMessageRateRegimeDetector](../algorithms/quote-message-rate-regime-detector.zh-CN.md) | 1 | 28.4 | 32.6 |
| [QuoteStuffingDetector](../algorithms/quote-stuffing-detector.zh-CN.md) | 2 | 36.9 | 41.5 |
| [RandomWalkIndex](../algorithms/random-walk-index.zh-CN.md) | 3 | 69.4 | 94.7 |
| [RateOfChangePercentage](../algorithms/rate-of-change-percentage.zh-CN.md) | 1 | 22.9 | 28.8 |
| [RateOfChangeRatio](../algorithms/rate-of-change-ratio.zh-CN.md) | 1 | 22.3 | 27.7 |
| [RateOfChangeRatio100](../algorithms/rate-of-change-ratio-100.zh-CN.md) | 1 | 22.2 | 28.2 |
| [RealizedVarianceRegimeDetector](../algorithms/realized-variance-regime-detector.zh-CN.md) | 1 | 25.5 | 30.4 |
| [RelativeVigorIndex](../algorithms/relative-vigor-index.zh-CN.md) | 4 | 63.0 | 92.6 |
| [RelativeVolatilityIndex](../algorithms/relative-volatility-index.zh-CN.md) | 1 | 32.9 | 68.4 |
| [RenkoBrickGenerator](../algorithms/renko-brick-generator.zh-CN.md) | 1 | 24.9 | 54.1 |
| [ResidualDriftDetector](../algorithms/residual-drift-detector.zh-CN.md) | 1 | 24.8 | 28.7 |
| [ROC](../algorithms/roc.zh-CN.md) | 1 | 22.2 | 28.3 |
| [RollingBetaShiftDetector](../algorithms/rolling-beta-shift-detector.zh-CN.md) | 2 | 42.9 | 47.4 |
| [RollingCorrelationShiftDetector](../algorithms/rolling-correlation-shift-detector.zh-CN.md) | 2 | 45.3 | 49.7 |
| [RollingMeanShiftDetector](../algorithms/rolling-mean-shift-detector.zh-CN.md) | 1 | 34.3 | 43.1 |
| [RollingMeanVarianceShiftDetector](../algorithms/rolling-mean-variance-shift-detector.zh-CN.md) | 1 | 46.3 | 54.1 |
| [RollingMedian](../algorithms/rolling-median.zh-CN.md) | 1 | 193 | 199 |
| [RollingSpreadLiquidityShiftDetector](../algorithms/rolling-spread-liquidity-shift-detector.zh-CN.md) | 4 | 58.7 | 62.3 |
| [RollingVarianceShiftDetector](../algorithms/rolling-variance-shift-detector.zh-CN.md) | 1 | 37.0 | 45.8 |
| [RSI](../algorithms/rsi.zh-CN.md) | 1 | 29.6 | 34.9 |
| [SavitzkyGolayFilter](../algorithms/savitzky-golay-filter.zh-CN.md) | 1 | 38.4 | 67.8 |
| [SchaffTrendCycle](../algorithms/schaff-trend-cycle.zh-CN.md) | 1 | 47.7 | 53.5 |
| [SMA](../algorithms/sma.zh-CN.md) | 1 | 22.9 | 28.3 |
| [SmoothedMovingAverage](../algorithms/smoothed-moving-average.zh-CN.md) | 1 | 22.8 | 28.7 |
| [SpreadExplosionDetector](../algorithms/spread-explosion-detector.zh-CN.md) | 2 | 36.5 | 40.5 |
| [SpreadFeatures](../algorithms/spread-features.zh-CN.md) | 3 | 54.4 | 76.9 |
| [SpreadRegimeDetector](../algorithms/spread-regime-detector.zh-CN.md) | 2 | 38.4 | 43.3 |
| [SqueezeMomentum](../algorithms/squeeze-momentum.zh-CN.md) | 3 | 98.2 | 127 |
| [StdDev](../algorithms/std-dev.zh-CN.md) | 1 | 23.9 | 28.6 |
| [StickyHMMRegimeFilter](../algorithms/sticky-hmm-regime-filter.zh-CN.md) | 1 | 77.8 | 79.2 |
| [Stochastic](../algorithms/stochastic.zh-CN.md) | 3 | 74.4 | 101 |
| [StochasticMomentumIndex](../algorithms/stochastic-momentum-index.zh-CN.md) | 3 | 74.7 | 100 |
| [StochRSI](../algorithms/stoch-rsi.zh-CN.md) | 1 | 59.7 | 66.0 |
| [Summation](../algorithms/summation.zh-CN.md) | 1 | 23.4 | 28.4 |
| [SuperTrend](../algorithms/super-trend.zh-CN.md) | 3 | 55.4 | 86.4 |
| [SwingIndex](../algorithms/swing-index.zh-CN.md) | 4 | 69.5 | 68.7 |
| [T3MovingAverage](../algorithms/t-3-moving-average.zh-CN.md) | 1 | 26.6 | 31.5 |
| [ThresholdRegimeDetector](../algorithms/threshold-regime-detector.zh-CN.md) | 1 | 22.5 | 28.2 |
| [TimeSeriesForecast](../algorithms/time-series-forecast.zh-CN.md) | 1 | 35.5 | 29.0 |
| [TradeIntensityRegimeDetector](../algorithms/trade-intensity-regime-detector.zh-CN.md) | 1 | 29.4 | 34.8 |
| [TrendChopRegimeDetector](../algorithms/trend-chop-regime-detector.zh-CN.md) | 3 | 48.3 | 52.8 |
| [TrendIntensityIndex](../algorithms/trend-intensity-index.zh-CN.md) | 1 | 26.7 | 31.4 |
| [TriangularMovingAverage](../algorithms/triangular-moving-average.zh-CN.md) | 1 | 25.7 | 30.3 |
| [TripleEMA](../algorithms/triple-ema.zh-CN.md) | 1 | 24.1 | 30.0 |
| [Trix](../algorithms/trix.zh-CN.md) | 1 | 25.5 | 30.3 |
| [TrueRange](../algorithms/true-range.zh-CN.md) | 3 | 45.0 | 51.5 |
| [TSI](../algorithms/tsi.zh-CN.md) | 1 | 26.2 | 31.3 |
| [TwiggsMoneyFlow](../algorithms/twiggs-money-flow.zh-CN.md) | 4 | 55.7 | 61.5 |
| [TwoFactorKalmanTrendFilter](../algorithms/two-factor-kalman-trend-filter.zh-CN.md) | 1 | 90.0 | 123 |
| [TypicalPrice](../algorithms/typical-price.zh-CN.md) | 3 | 44.5 | 50.4 |
| [UlcerIndex](../algorithms/ulcer-index.zh-CN.md) | 1 | 36.4 | 41.1 |
| [UltimateOscillator](../algorithms/ultimate-oscillator.zh-CN.md) | 3 | 51.5 | 57.9 |
| [VariableIndexDynamicAverage](../algorithms/variable-index-dynamic-average.zh-CN.md) | 1 | 32.1 | 36.3 |
| [Variance](../algorithms/variance.zh-CN.md) | 1 | 22.5 | 28.9 |
| [VerticalHorizontalFilter](../algorithms/vertical-horizontal-filter.zh-CN.md) | 1 | 49.3 | 54.9 |
| [VolatilityBreakoutDetector](../algorithms/volatility-breakout-detector.zh-CN.md) | 1 | 23.7 | 29.3 |
| [VolatilityCompressionExpansionDetector](../algorithms/volatility-compression-expansion-detector.zh-CN.md) | 1 | 26.4 | 31.0 |
| [VolatilityRegimeDetector](../algorithms/volatility-regime-detector.zh-CN.md) | 1 | 23.9 | 29.5 |
| [VolumeOscillator](../algorithms/volume-oscillator.zh-CN.md) | 1 | 24.5 | 29.9 |
| [VolumePriceTrend](../algorithms/volume-price-trend.zh-CN.md) | 2 | 36.0 | 40.2 |
| [VolumeProfile](../algorithms/volume-profile.zh-CN.md) | 2 | 163 | 198 |
| [VolumeRegimeDetector](../algorithms/volume-regime-detector.zh-CN.md) | 1 | 30.0 | 34.0 |
| [VolumeWeightedAveragePrice](../algorithms/volume-weighted-average-price.zh-CN.md) | 4 | 56.9 | 60.3 |
| [VolumeWeightedMovingAverage](../algorithms/volume-weighted-moving-average.zh-CN.md) | 2 | 35.5 | 42.0 |
| [Vortex](../algorithms/vortex.zh-CN.md) | 3 | 47.8 | 74.8 |
| [VPIN](../algorithms/vpin.zh-CN.md) | 2 | 79.8 | 83.7 |
| [WaveTrend](../algorithms/wave-trend.zh-CN.md) | 3 | 51.7 | 82.5 |
| [WeightedClosePrice](../algorithms/weighted-close-price.zh-CN.md) | 3 | 44.9 | 50.4 |
| [WeightedMovingAverage](../algorithms/weighted-moving-average.zh-CN.md) | 1 | 23.1 | 28.5 |
| [WilliamsAD](../algorithms/williams-ad.zh-CN.md) | 3 | 50.6 | 54.6 |
| [WilliamsFractals](../algorithms/williams-fractals.zh-CN.md) | 2 | 47.8 | 72.3 |
| [WilliamsR](../algorithms/williams-r.zh-CN.md) | 3 | 69.9 | 73.7 |
| [ZeroLagEMA](../algorithms/zero-lag-ema.zh-CN.md) | 1 | 24.8 | 28.7 |
| [ZigZagSwingDetector](../algorithms/zig-zag-swing-detector.zh-CN.md) | 1 | 24.9 | 58.3 |
