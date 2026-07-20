# Loongson-3A6000 基准测试

以下结果采集于 2026-07-19。公开文档以 CPU 类型而不是主机名标识基准测试系统。

## 系统

- CPU： **Loongson-3A6000**
- 架构： `loongarch64`
- 平台： `Linux-5.4.18-167-generic-loongarch64-with-glibc2.28`
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
- 中位 `advance(...)` 延迟（仅更新状态）: **104 ns/update**.
- 中位 `update(...)` 延迟（更新状态并返回值/结果）: **139 ns/update**.

| 算法 | 输入数 | 仅更新：`advance(...)` ns/update | 更新并返回：`update(...)` ns/update |
|---|---:|---:|---:|
| [AbsolutePriceOscillator](../algorithms/absolute-price-oscillator.zh-CN.md) | 1 | 55.4 | 76.1 |
| [AccelerationBands](../algorithms/acceleration-bands.zh-CN.md) | 3 | 140 | 228 |
| [AcceleratorOscillator](../algorithms/accelerator-oscillator.zh-CN.md) | 2 | 124 | 145 |
| [AccumulationDistribution](../algorithms/accumulation-distribution.zh-CN.md) | 4 | 118 | 141 |
| [AccumulativeSwingIndex](../algorithms/accumulative-swing-index.zh-CN.md) | 4 | 132 | 156 |
| [ADWIN](../algorithms/adwin.zh-CN.md) | 1 | 1049 | 1065 |
| [Alligator](../algorithms/alligator.zh-CN.md) | 2 | 94.3 | 179 |
| [AlphaBetaGammaTrackingFilter](../algorithms/alpha-beta-gamma-tracking-filter.zh-CN.md) | 1 | 69.9 | 144 |
| [AmihudIlliquidity](../algorithms/amihud-illiquidity.zh-CN.md) | 2 | 107 | 129 |
| [AnchoredVWAP](../algorithms/anchored-vwap.zh-CN.md) | 5 | 148 | 177 |
| [ArnaudLegouxMovingAverage](../algorithms/arnaud-legoux-moving-average.zh-CN.md) | 1 | 52.6 | 93.4 |
| [Aroon](../algorithms/aroon.zh-CN.md) | 2 | 123 | 206 |
| [AroonOscillator](../algorithms/aroon-oscillator.zh-CN.md) | 2 | 123 | 149 |
| [ATR](../algorithms/atr.zh-CN.md) | 3 | 104 | 127 |
| [ATRP](../algorithms/atrp.zh-CN.md) | 3 | 105 | 138 |
| [ATRRegimeDetector](../algorithms/atr-regime-detector.zh-CN.md) | 3 | 109 | 128 |
| [AuctionContinuousMarketTransitionDetector](../algorithms/auction-continuous-market-transition-detector.zh-CN.md) | 1 | 52.0 | 72.9 |
| [AverageDirectionalMovementIndex](../algorithms/average-directional-movement-index.zh-CN.md) | 3 | 161 | 183 |
| [AverageDirectionalMovementIndexRating](../algorithms/average-directional-movement-index-rating.zh-CN.md) | 3 | 165 | 190 |
| [AveragePrice](../algorithms/average-price.zh-CN.md) | 4 | 113 | 138 |
| [AwesomeOscillator](../algorithms/awesome-oscillator.zh-CN.md) | 2 | 102 | 126 |
| [BalanceOfPower](../algorithms/balance-of-power.zh-CN.md) | 4 | 113 | 139 |
| [Beta](../algorithms/beta.zh-CN.md) | 2 | 89.2 | 120 |
| [BetaRegimeDetector](../algorithms/beta-regime-detector.zh-CN.md) | 2 | 96.1 | 120 |
| [Bias](../algorithms/bias.zh-CN.md) | 1 | 62.6 | 97.8 |
| [BidAskBounceRegimeDetector](../algorithms/bid-ask-bounce-regime-detector.zh-CN.md) | 3 | 111 | 138 |
| [BollingerBands](../algorithms/bollinger-bands.zh-CN.md) | 1 | 84.1 | 163 |
| [BollingerBandwidth](../algorithms/bollinger-bandwidth.zh-CN.md) | 1 | 93.8 | 115 |
| [BollingerPercentB](../algorithms/bollinger-percent-b.zh-CN.md) | 1 | 94.7 | 117 |
| [BoundedBOCPD](../algorithms/bounded-bocpd.zh-CN.md) | 1 | 5375 | 5395 |
| [CalibrationDriftDetector](../algorithms/calibration-drift-detector.zh-CN.md) | 2 | 89.1 | 111 |
| [ChaikinMoneyFlow](../algorithms/chaikin-money-flow.zh-CN.md) | 4 | 132 | 156 |
| [ChaikinOscillator](../algorithms/chaikin-oscillator.zh-CN.md) | 4 | 123 | 146 |
| [ChaikinVolatility](../algorithms/chaikin-volatility.zh-CN.md) | 2 | 87.6 | 112 |
| [ChandelierExit](../algorithms/chandelier-exit.zh-CN.md) | 3 | 142 | 233 |
| [ChandeMomentumOscillator](../algorithms/chande-momentum-oscillator.zh-CN.md) | 1 | 64.2 | 96.1 |
| [ChoppinessIndex](../algorithms/choppiness-index.zh-CN.md) | 3 | 237 | 262 |
| [ClosePressureReversalSignal](../algorithms/close-pressure-reversal-signal.zh-CN.md) | 5 | 342 | 487 |
| [CointegrationBreakdownMonitor](../algorithms/cointegration-breakdown-monitor.zh-CN.md) | 2 | 113 | 143 |
| [CommodityChannelIndex](../algorithms/commodity-channel-index.zh-CN.md) | 3 | 152 | 177 |
| [ComparativeRelativeStrength](../algorithms/comparative-relative-strength.zh-CN.md) | 2 | 78.5 | 102 |
| [ConnorsRSI](../algorithms/connors-rsi.zh-CN.md) | 1 | 214 | 234 |
| [CoppockCurve](../algorithms/coppock-curve.zh-CN.md) | 1 | 90.6 | 107 |
| [Correlation](../algorithms/correlation.zh-CN.md) | 2 | 95.3 | 140 |
| [CorrelationRegimeDetector](../algorithms/correlation-regime-detector.zh-CN.md) | 2 | 123 | 140 |
| [CrossAssetCorrelationBreakDetector](../algorithms/cross-asset-correlation-break-detector.zh-CN.md) | 2 | 126 | 149 |
| [CumulativeReturn](../algorithms/cumulative-return.zh-CN.md) | 1 | 51.1 | 77.5 |
| [CUSUM](../algorithms/cusum.zh-CN.md) | 1 | 56.9 | 77.6 |
| [DailyLogReturn](../algorithms/daily-log-return.zh-CN.md) | 1 | 52.5 | 117 |
| [DailyReturn](../algorithms/daily-return.zh-CN.md) | 1 | 52.2 | 77.3 |
| [DDM](../algorithms/ddm.zh-CN.md) | 1 | 93.1 | 113 |
| [Delay](../algorithms/delay.zh-CN.md) | 1 | 53.2 | 72.9 |
| [DeMarker](../algorithms/de-marker.zh-CN.md) | 2 | 94.0 | 118 |
| [DetrendedPriceOscillator](../algorithms/detrended-price-oscillator.zh-CN.md) | 1 | 64.0 | 83.7 |
| [DirectionalMovementIndex](../algorithms/directional-movement-index.zh-CN.md) | 3 | 148 | 172 |
| [DonchianChannel](../algorithms/donchian-channel.zh-CN.md) | 3 | 151 | 256 |
| [DoubleEMA](../algorithms/double-ema.zh-CN.md) | 1 | 55.3 | 77.1 |
| [EaseOfMovement](../algorithms/ease-of-movement.zh-CN.md) | 3 | 117 | 206 |
| [EDDM](../algorithms/eddm.zh-CN.md) | 1 | 65.7 | 85.8 |
| [EfficiencyRatio](../algorithms/efficiency-ratio.zh-CN.md) | 1 | 55.1 | 83.6 |
| [EhlersCenterOfGravity](../algorithms/ehlers-center-of-gravity.zh-CN.md) | 1 | 80.7 | 165 |
| [EhlersCyberCycle](../algorithms/ehlers-cyber-cycle.zh-CN.md) | 1 | 68.9 | 144 |
| [EhlersDecycler](../algorithms/ehlers-decycler.zh-CN.md) | 1 | 52.7 | 128 |
| [EhlersInstantaneousTrendline](../algorithms/ehlers-instantaneous-trendline.zh-CN.md) | 1 | 55.0 | 129 |
| [EhlersOptimalTrackingFilter](../algorithms/ehlers-optimal-tracking-filter.zh-CN.md) | 2 | 124 | 145 |
| [EhlersRoofingFilter](../algorithms/ehlers-roofing-filter.zh-CN.md) | 1 | 60.6 | 135 |
| [EhlersSuperSmoother](../algorithms/ehlers-super-smoother.zh-CN.md) | 1 | 54.0 | 75.6 |
| [ElderRayIndex](../algorithms/elder-ray-index.zh-CN.md) | 3 | 101 | 187 |
| [EMA](../algorithms/ema.zh-CN.md) | 1 | 53.1 | 74.0 |
| [EWMA](../algorithms/ewma.zh-CN.md) | 1 | 52.0 | 72.2 |
| [EWMAZScoreShiftDetector](../algorithms/ewmaz-score-shift-detector.zh-CN.md) | 1 | 73.0 | 93.1 |
| [ExecutionCostSlippageRegimeDetector](../algorithms/execution-cost-slippage-regime-detector.zh-CN.md) | 3 | 106 | 127 |
| [FastStochastic](../algorithms/fast-stochastic.zh-CN.md) | 3 | 156 | 247 |
| [FeatureDistributionDriftDetector](../algorithms/feature-distribution-drift-detector.zh-CN.md) | 1 | 1051 | 1070 |
| [FibonacciRetracementLevels](../algorithms/fibonacci-retracement-levels.zh-CN.md) | 2 | 116 | 207 |
| [FisherTransform](../algorithms/fisher-transform.zh-CN.md) | 2 | 191 | 215 |
| [ForceIndex](../algorithms/force-index.zh-CN.md) | 2 | 82.7 | 103 |
| [FractalAdaptiveMovingAverage](../algorithms/fractal-adaptive-moving-average.zh-CN.md) | 1 | 221 | 235 |
| [GatorOscillator](../algorithms/gator-oscillator.zh-CN.md) | 2 | 95.9 | 180 |
| [GaussianProcessRegressionBands](../algorithms/gaussian-process-regression-bands.zh-CN.md) | 1 | 1912 | 1997 |
| [GeometricMovingAverage](../algorithms/geometric-moving-average.zh-CN.md) | 1 | 105 | 157 |
| [GuppyMultipleMovingAverage](../algorithms/guppy-multiple-moving-average.zh-CN.md) | 1 | 77.7 | 175 |
| [HDDM](../algorithms/hddm.zh-CN.md) | 1 | 151 | 167 |
| [HeikinAshiTransform](../algorithms/heikin-ashi-transform.zh-CN.md) | 4 | 119 | 207 |
| [HiddenSemiMarkovRegimeFilter](../algorithms/hidden-semi-markov-regime-filter.zh-CN.md) | 1 | 216 | 241 |
| [High](../algorithms/high.zh-CN.md) | 1 | 70.8 | 93.1 |
| [HighIndex](../algorithms/high-index.zh-CN.md) | 1 | 70.7 | 93.7 |
| [HighLow](../algorithms/high-low.zh-CN.md) | 1 | 85.5 | 161 |
| [HighLowIndex](../algorithms/high-low-index.zh-CN.md) | 1 | 82.8 | 162 |
| [HilbertDominantCyclePeriod](../algorithms/hilbert-dominant-cycle-period.zh-CN.md) | 1 | 1242 | 1261 |
| [HilbertDominantCyclePhase](../algorithms/hilbert-dominant-cycle-phase.zh-CN.md) | 1 | 1241 | 1260 |
| [HilbertPhasor](../algorithms/hilbert-phasor.zh-CN.md) | 1 | 1243 | 1343 |
| [HilbertSineWave](../algorithms/hilbert-sine-wave.zh-CN.md) | 1 | 1242 | 1362 |
| [HilbertTrendline](../algorithms/hilbert-trendline.zh-CN.md) | 1 | 1242 | 1261 |
| [HilbertTrendMode](../algorithms/hilbert-trend-mode.zh-CN.md) | 1 | 1241 | 1263 |
| [HistoricalVolatility](../algorithms/historical-volatility.zh-CN.md) | 1 | 112 | 132 |
| [HitRateDriftDetector](../algorithms/hit-rate-drift-detector.zh-CN.md) | 1 | 56.8 | 77.0 |
| [HullMovingAverage](../algorithms/hull-moving-average.zh-CN.md) | 1 | 77.6 | 109 |
| [Ichimoku](../algorithms/ichimoku.zh-CN.md) | 3 | 180 | 302 |
| [InteractingMultipleModelFilter](../algorithms/interacting-multiple-model-filter.zh-CN.md) | 1 | 521 | 614 |
| [IntradayClockEchoSignal](../algorithms/intraday-clock-echo-signal.zh-CN.md) | 5 | 548 | 760 |
| [IntradayIntensity](../algorithms/intraday-intensity.zh-CN.md) | 4 | 131 | 157 |
| [IntradayMomentumIndex](../algorithms/intraday-momentum-index.zh-CN.md) | 2 | 98.4 | 123 |
| [InverseFisherRSI](../algorithms/inverse-fisher-rsi.zh-CN.md) | 1 | 127 | 192 |
| [KagiChart](../algorithms/kagi-chart.zh-CN.md) | 1 | 58.9 | 140 |
| [KalmanExtremumTrend](../algorithms/kalman-extremum-trend.zh-CN.md) | 3 | 246 | 347 |
| [KalmanHedgeRatio](../algorithms/kalman-hedge-ratio.zh-CN.md) | 2 | 179 | 260 |
| [KalmanInnovationZScore](../algorithms/kalman-innovation-z-score.zh-CN.md) | 1 | 161 | 182 |
| [KalmanLocalLinearTrend](../algorithms/kalman-local-linear-trend.zh-CN.md) | 1 | 145 | 223 |
| [KalmanMovingAverage](../algorithms/kalman-moving-average.zh-CN.md) | 1 | 144 | 165 |
| [KalmanPredictionBands](../algorithms/kalman-prediction-bands.zh-CN.md) | 1 | 160 | 241 |
| [KalmanRegressionChannel](../algorithms/kalman-regression-channel.zh-CN.md) | 2 | 185 | 276 |
| [KalmanTrendSignal](../algorithms/kalman-trend-signal.zh-CN.md) | 1 | 149 | 241 |
| [KalmanVelocityOscillator](../algorithms/kalman-velocity-oscillator.zh-CN.md) | 1 | 144 | 165 |
| [Kama](../algorithms/kama.zh-CN.md) | 1 | 76.1 | 93.9 |
| [KeltnerChannel](../algorithms/keltner-channel.zh-CN.md) | 3 | 108 | 198 |
| [KeltnerChannelOriginal](../algorithms/keltner-channel-original.zh-CN.md) | 3 | 126 | 218 |
| [KlingerVolumeOscillator](../algorithms/klinger-volume-oscillator.zh-CN.md) | 4 | 149 | 235 |
| [KSTOscillator](../algorithms/kst-oscillator.zh-CN.md) | 1 | 116 | 201 |
| [KSWIN](../algorithms/kswin.zh-CN.md) | 1 | 3141 | 3169 |
| [KyleLambda](../algorithms/kyle-lambda.zh-CN.md) | 2 | 114 | 136 |
| [LeadLagRegimeDetector](../algorithms/lead-lag-regime-detector.zh-CN.md) | 2 | 93.1 | 113 |
| [LinearRegression](../algorithms/linear-regression.zh-CN.md) | 1 | 120 | 96.7 |
| [LinearRegressionAngle](../algorithms/linear-regression-angle.zh-CN.md) | 1 | 120 | 138 |
| [LinearRegressionIntercept](../algorithms/linear-regression-intercept.zh-CN.md) | 1 | 120 | 95.2 |
| [LinearRegressionSlope](../algorithms/linear-regression-slope.zh-CN.md) | 1 | 120 | 88.5 |
| [LiquidityDroughtDetector](../algorithms/liquidity-drought-detector.zh-CN.md) | 3 | 108 | 126 |
| [LiquidityRegimeDetector](../algorithms/liquidity-regime-detector.zh-CN.md) | 2 | 95.8 | 118 |
| [Low](../algorithms/low.zh-CN.md) | 1 | 71.3 | 92.9 |
| [LowIndex](../algorithms/low-index.zh-CN.md) | 1 | 71.2 | 93.5 |
| [MACD](../algorithms/macd.zh-CN.md) | 1 | 58.1 | 136 |
| [MACDExt](../algorithms/macd-ext.zh-CN.md) | 1 | 58.7 | 145 |
| [MACDFix](../algorithms/macd-fix.zh-CN.md) | 1 | 58.1 | 136 |
| [MarketFacilitationIndex](../algorithms/market-facilitation-index.zh-CN.md) | 3 | 97.9 | 121 |
| [MarketOpenCloseTransitionDetector](../algorithms/market-open-close-transition-detector.zh-CN.md) | 1 | 52.2 | 72.7 |
| [MassIndex](../algorithms/mass-index.zh-CN.md) | 2 | 94.6 | 117 |
| [MatchedFlowConformalSignal](../algorithms/matched-flow-conformal-signal.zh-CN.md) | 5 | 2265 | 2446 |
| [McGinleyDynamic](../algorithms/mc-ginley-dynamic.zh-CN.md) | 1 | 174 | 193 |
| [MedianPrice](../algorithms/median-price.zh-CN.md) | 2 | 78.6 | 102 |
| [MesaAdaptiveMovingAverage](../algorithms/mesa-adaptive-moving-average.zh-CN.md) | 1 | 246 | 323 |
| [MicrostructureNoiseRegimeDetector](../algorithms/microstructure-noise-regime-detector.zh-CN.md) | 3 | 103 | 124 |
| [MidPoint](../algorithms/mid-point.zh-CN.md) | 1 | 85.7 | 105 |
| [MidPrice](../algorithms/mid-price.zh-CN.md) | 2 | 117 | 141 |
| [MinusDirectionalIndicator](../algorithms/minus-directional-indicator.zh-CN.md) | 3 | 124 | 148 |
| [MinusDirectionalMovement](../algorithms/minus-directional-movement.zh-CN.md) | 2 | 91.0 | 115 |
| [Momentum](../algorithms/momentum.zh-CN.md) | 1 | 53.0 | 74.5 |
| [MoneyFlowIndex](../algorithms/money-flow-index.zh-CN.md) | 4 | 152 | 176 |
| [MovingAverageEnvelope](../algorithms/moving-average-envelope.zh-CN.md) | 1 | 65.9 | 145 |
| [NadarayaWatsonEnvelope](../algorithms/nadaraya-watson-envelope.zh-CN.md) | 1 | 202 | 282 |
| [NegativeVolumeIndex](../algorithms/negative-volume-index.zh-CN.md) | 2 | 86.8 | 109 |
| [NormalizedATR](../algorithms/normalized-atr.zh-CN.md) | 3 | 105 | 140 |
| [OnBalanceVolume](../algorithms/on-balance-volume.zh-CN.md) | 2 | 86.9 | 108 |
| [OnlineGaussianMixtureRegimeFilter](../algorithms/online-gaussian-mixture-regime-filter.zh-CN.md) | 1 | 180 | 206 |
| [OnlineHMMRegimeFilter](../algorithms/online-hmm-regime-filter.zh-CN.md) | 1 | 213 | 240 |
| [OnlineMarkovSwitchingVolatilityFilter](../algorithms/online-markov-switching-volatility-filter.zh-CN.md) | 1 | 229 | 253 |
| [OrderFlowImbalance](../algorithms/order-flow-imbalance.zh-CN.md) | 4 | 125 | 149 |
| [OrderFlowImbalanceRegimeDetector](../algorithms/order-flow-imbalance-regime-detector.zh-CN.md) | 4 | 128 | 150 |
| [PageHinkley](../algorithms/page-hinkley.zh-CN.md) | 1 | 68.8 | 87.3 |
| [PairsSpreadRegimeDetector](../algorithms/pairs-spread-regime-detector.zh-CN.md) | 2 | 104 | 130 |
| [ParabolicSAR](../algorithms/parabolic-sar.zh-CN.md) | 2 | 82.2 | 104 |
| [ParabolicSARExtended](../algorithms/parabolic-sar-extended.zh-CN.md) | 2 | 81.7 | 104 |
| [ParticleFilterTrend](../algorithms/particle-filter-trend.zh-CN.md) | 1 | 11862 | 12045 |
| [PercentagePrice](../algorithms/percentage-price.zh-CN.md) | 1 | 69.9 | 147 |
| [PercentageVolume](../algorithms/percentage-volume.zh-CN.md) | 1 | 69.9 | 149 |
| [PivotPoints](../algorithms/pivot-points.zh-CN.md) | 3 | 105 | 195 |
| [PlusDirectionalIndicator](../algorithms/plus-directional-indicator.zh-CN.md) | 3 | 125 | 148 |
| [PlusDirectionalMovement](../algorithms/plus-directional-movement.zh-CN.md) | 2 | 91.2 | 114 |
| [PointAndFigure](../algorithms/point-and-figure.zh-CN.md) | 1 | 57.1 | 150 |
| [PositiveVolumeIndex](../algorithms/positive-volume-index.zh-CN.md) | 2 | 86.8 | 109 |
| [PredictionErrorDriftDetector](../algorithms/prediction-error-drift-detector.zh-CN.md) | 2 | 97.4 | 119 |
| [PrettyGoodOscillator](../algorithms/pretty-good-oscillator.zh-CN.md) | 3 | 117 | 144 |
| [PsychologicalLine](../algorithms/psychological-line.zh-CN.md) | 1 | 61.2 | 87.9 |
| [QStick](../algorithms/q-stick.zh-CN.md) | 2 | 89.6 | 111 |
| [QuoteMessageRateRegimeDetector](../algorithms/quote-message-rate-regime-detector.zh-CN.md) | 1 | 62.0 | 82.2 |
| [QuoteStuffingDetector](../algorithms/quote-stuffing-detector.zh-CN.md) | 2 | 87.3 | 108 |
| [RandomWalkIndex](../algorithms/random-walk-index.zh-CN.md) | 3 | 156 | 252 |
| [RateOfChangePercentage](../algorithms/rate-of-change-percentage.zh-CN.md) | 1 | 52.6 | 77.1 |
| [RateOfChangeRatio](../algorithms/rate-of-change-ratio.zh-CN.md) | 1 | 52.7 | 75.5 |
| [RateOfChangeRatio100](../algorithms/rate-of-change-ratio-100.zh-CN.md) | 1 | 52.7 | 77.8 |
| [RealizedVarianceRegimeDetector](../algorithms/realized-variance-regime-detector.zh-CN.md) | 1 | 63.3 | 81.7 |
| [RelativeVigorIndex](../algorithms/relative-vigor-index.zh-CN.md) | 4 | 166 | 274 |
| [RelativeVolatilityIndex](../algorithms/relative-volatility-index.zh-CN.md) | 1 | 103 | 186 |
| [RenkoBrickGenerator](../algorithms/renko-brick-generator.zh-CN.md) | 1 | 55.5 | 135 |
| [ResidualDriftDetector](../algorithms/residual-drift-detector.zh-CN.md) | 1 | 71.7 | 91.7 |
| [ROC](../algorithms/roc.zh-CN.md) | 1 | 52.4 | 78.6 |
| [RollingBetaShiftDetector](../algorithms/rolling-beta-shift-detector.zh-CN.md) | 2 | 112 | 138 |
| [RollingCorrelationShiftDetector](../algorithms/rolling-correlation-shift-detector.zh-CN.md) | 2 | 130 | 152 |
| [RollingMeanShiftDetector](../algorithms/rolling-mean-shift-detector.zh-CN.md) | 1 | 128 | 150 |
| [RollingMeanVarianceShiftDetector](../algorithms/rolling-mean-variance-shift-detector.zh-CN.md) | 1 | 180 | 199 |
| [RollingMedian](../algorithms/rolling-median.zh-CN.md) | 1 | 319 | 339 |
| [RollingSpreadLiquidityShiftDetector](../algorithms/rolling-spread-liquidity-shift-detector.zh-CN.md) | 4 | 138 | 161 |
| [RollingVarianceShiftDetector](../algorithms/rolling-variance-shift-detector.zh-CN.md) | 1 | 145 | 166 |
| [RSI](../algorithms/rsi.zh-CN.md) | 1 | 101 | 121 |
| [SavitzkyGolayFilter](../algorithms/savitzky-golay-filter.zh-CN.md) | 1 | 85.4 | 172 |
| [SchaffTrendCycle](../algorithms/schaff-trend-cycle.zh-CN.md) | 1 | 106 | 128 |
| [SMA](../algorithms/sma.zh-CN.md) | 1 | 62.6 | 82.5 |
| [SmoothedMovingAverage](../algorithms/smoothed-moving-average.zh-CN.md) | 1 | 58.8 | 78.0 |
| [SpreadExplosionDetector](../algorithms/spread-explosion-detector.zh-CN.md) | 2 | 83.8 | 104 |
| [SpreadFeatures](../algorithms/spread-features.zh-CN.md) | 3 | 123 | 211 |
| [SpreadRegimeDetector](../algorithms/spread-regime-detector.zh-CN.md) | 2 | 89.7 | 110 |
| [SqueezeMomentum](../algorithms/squeeze-momentum.zh-CN.md) | 3 | 237 | 347 |
| [StdDev](../algorithms/std-dev.zh-CN.md) | 1 | 61.2 | 99.8 |
| [StickyHMMRegimeFilter](../algorithms/sticky-hmm-regime-filter.zh-CN.md) | 1 | 212 | 237 |
| [Stochastic](../algorithms/stochastic.zh-CN.md) | 3 | 176 | 269 |
| [StochasticMomentumIndex](../algorithms/stochastic-momentum-index.zh-CN.md) | 3 | 152 | 257 |
| [StochRSI](../algorithms/stoch-rsi.zh-CN.md) | 1 | 136 | 165 |
| [Summation](../algorithms/summation.zh-CN.md) | 1 | 52.6 | 74.7 |
| [SuperTrend](../algorithms/super-trend.zh-CN.md) | 3 | 120 | 218 |
| [SwingIndex](../algorithms/swing-index.zh-CN.md) | 4 | 128 | 155 |
| [T3MovingAverage](../algorithms/t-3-moving-average.zh-CN.md) | 1 | 73.0 | 92.5 |
| [ThresholdRegimeDetector](../algorithms/threshold-regime-detector.zh-CN.md) | 1 | 52.4 | 72.4 |
| [TimeSeriesForecast](../algorithms/time-series-forecast.zh-CN.md) | 1 | 120 | 96.6 |
| [TradeIntensityRegimeDetector](../algorithms/trade-intensity-regime-detector.zh-CN.md) | 1 | 64.5 | 85.2 |
| [TrendChopRegimeDetector](../algorithms/trend-chop-regime-detector.zh-CN.md) | 3 | 112 | 133 |
| [TrendIntensityIndex](../algorithms/trend-intensity-index.zh-CN.md) | 1 | 83.3 | 103 |
| [TriangularMovingAverage](../algorithms/triangular-moving-average.zh-CN.md) | 1 | 82.6 | 101 |
| [TripleEMA](../algorithms/triple-ema.zh-CN.md) | 1 | 57.2 | 80.2 |
| [Trix](../algorithms/trix.zh-CN.md) | 1 | 69.1 | 89.8 |
| [TrueRange](../algorithms/true-range.zh-CN.md) | 3 | 97.1 | 123 |
| [TSI](../algorithms/tsi.zh-CN.md) | 1 | 66.9 | 89.1 |
| [TwiggsMoneyFlow](../algorithms/twiggs-money-flow.zh-CN.md) | 4 | 130 | 156 |
| [TwoFactorKalmanTrendFilter](../algorithms/two-factor-kalman-trend-filter.zh-CN.md) | 1 | 148 | 225 |
| [TypicalPrice](../algorithms/typical-price.zh-CN.md) | 3 | 97.8 | 123 |
| [UlcerIndex](../algorithms/ulcer-index.zh-CN.md) | 1 | 105 | 126 |
| [UltimateOscillator](../algorithms/ultimate-oscillator.zh-CN.md) | 3 | 133 | 159 |
| [VariableIndexDynamicAverage](../algorithms/variable-index-dynamic-average.zh-CN.md) | 1 | 84.3 | 103 |
| [Variance](../algorithms/variance.zh-CN.md) | 1 | 54.9 | 80.0 |
| [VerticalHorizontalFilter](../algorithms/vertical-horizontal-filter.zh-CN.md) | 1 | 113 | 141 |
| [VolatilityBreakoutDetector](../algorithms/volatility-breakout-detector.zh-CN.md) | 1 | 71.9 | 91.8 |
| [VolatilityCompressionExpansionDetector](../algorithms/volatility-compression-expansion-detector.zh-CN.md) | 1 | 97.2 | 117 |
| [VolatilityRegimeDetector](../algorithms/volatility-regime-detector.zh-CN.md) | 1 | 71.6 | 90.5 |
| [VolumeOscillator](../algorithms/volume-oscillator.zh-CN.md) | 1 | 67.4 | 99.5 |
| [VolumePriceTrend](../algorithms/volume-price-trend.zh-CN.md) | 2 | 83.5 | 107 |
| [VolumeProfile](../algorithms/volume-profile.zh-CN.md) | 2 | 336 | 464 |
| [VolumeRegimeDetector](../algorithms/volume-regime-detector.zh-CN.md) | 1 | 64.1 | 84.6 |
| [VolumeWeightedAveragePrice](../algorithms/volume-weighted-average-price.zh-CN.md) | 4 | 123 | 150 |
| [VolumeWeightedMovingAverage](../algorithms/volume-weighted-moving-average.zh-CN.md) | 2 | 85.0 | 115 |
| [Vortex](../algorithms/vortex.zh-CN.md) | 3 | 116 | 206 |
| [VPIN](../algorithms/vpin.zh-CN.md) | 2 | 198 | 220 |
| [WaveTrend](../algorithms/wave-trend.zh-CN.md) | 3 | 149 | 235 |
| [WeightedClosePrice](../algorithms/weighted-close-price.zh-CN.md) | 3 | 98.2 | 122 |
| [WeightedMovingAverage](../algorithms/weighted-moving-average.zh-CN.md) | 1 | 53.0 | 83.6 |
| [WilliamsAD](../algorithms/williams-ad.zh-CN.md) | 3 | 115 | 129 |
| [WilliamsFractals](../algorithms/williams-fractals.zh-CN.md) | 2 | 93.5 | 189 |
| [WilliamsR](../algorithms/williams-r.zh-CN.md) | 3 | 140 | 170 |
| [ZeroLagEMA](../algorithms/zero-lag-ema.zh-CN.md) | 1 | 56.6 | 76.2 |
| [ZigZagSwingDetector](../algorithms/zig-zag-swing-detector.zh-CN.md) | 1 | 56.2 | 135 |
