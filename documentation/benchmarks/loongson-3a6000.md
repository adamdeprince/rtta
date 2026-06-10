# Loongson-3A6000 Benchmarks

These results were collected on 2026-06-09. Public documentation identifies benchmark systems by CPU type rather than hostname.

## System

- CPU: **Loongson-3A6000**
- Architecture: `loongarch64`
- Platform: `Linux-5.4.18-110-generic-loongarch64-with-glibc2.28`
- Python: `3.14.4`
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
- Median `advance(...)` latency, update state only: **121 ns/update**.
- Median `update(...)` latency, update state and return a value/result: **152 ns/update**.

| Algorithm | Inputs | update only: `advance(...)` ns/update | update + return: `update(...)` ns/update |
|---|---:|---:|---:|
| [ATR](../algorithms/atr.md) | 3 | 119 | 142 |
| [ATRP](../algorithms/atrp.md) | 3 | 126 | 151 |
| [ATRRegimeDetector](../algorithms/atr-regime-detector.md) | 3 | 123 | 148 |
| [ADWIN](../algorithms/adwin.md) | 1 | 1637 | 1657 |
| [EMA](../algorithms/ema.md) | 1 | 66.1 | 84.7 |
| [EWMA](../algorithms/ewma.md) | 1 | 63.7 | 80.7 |
| [EWMAZScoreShiftDetector](../algorithms/ewmaz-score-shift-detector.md) | 1 | 83.7 | 102 |
| [MACD](../algorithms/macd.md) | 1 | 75.2 | 92.2 |
| [ROC](../algorithms/roc.md) | 1 | 67.0 | 86.8 |
| [RSI](../algorithms/rsi.md) | 1 | 112 | 131 |
| [SMA](../algorithms/sma.md) | 1 | 71.8 | 90.5 |
| [TSI](../algorithms/tsi.md) | 1 | 81.2 | 97.6 |
| [AbsolutePriceOscillator](../algorithms/absolute-price-oscillator.md) | 1 | 70.1 | 88.8 |
| [AccumulationDistribution](../algorithms/accumulation-distribution.md) | 4 | 124 | 147 |
| [AlphaBetaGammaTrackingFilter](../algorithms/alpha-beta-gamma-tracking-filter.md) | 1 | 82.4 | 154 |
| [AmihudIlliquidity](../algorithms/amihud-illiquidity.md) | 2 | 120 | 142 |
| [AnchoredVWAP](../algorithms/anchored-vwap.md) | 5 | 152 | 178 |
| [Aroon](../algorithms/aroon.md) | 2 | 150 | 267 |
| [AroonOscillator](../algorithms/aroon-oscillator.md) | 2 | 150 | 179 |
| [AverageDirectionalMovementIndex](../algorithms/average-directional-movement-index.md) | 3 | 179 | 201 |
| [AverageDirectionalMovementIndexRating](../algorithms/average-directional-movement-index-rating.md) | 3 | 182 | 205 |
| [AveragePrice](../algorithms/average-price.md) | 4 | 121 | 144 |
| [AuctionContinuousMarketTransitionDetector](../algorithms/auction-continuous-market-transition-detector.md) | 1 | 65.0 | 81.8 |
| [AwesomeOscillator](../algorithms/awesome-oscillator.md) | 2 | 103 | 125 |
| [BalanceOfPower](../algorithms/balance-of-power.md) | 4 | 121 | 145 |
| [Beta](../algorithms/beta.md) | 2 | 109 | 133 |
| [BetaRegimeDetector](../algorithms/beta-regime-detector.md) | 2 | 113 | 139 |
| [BidAskBounceRegimeDetector](../algorithms/bid-ask-bounce-regime-detector.md) | 3 | 127 | 148 |
| [BollingerBands](../algorithms/bollinger-bands.md) | 1 | 112 | 195 |
| [BoundedBOCPD](../algorithms/bounded-bocpd.md) | 1 | 6545 | 6580 |
| [CalibrationDriftDetector](../algorithms/calibration-drift-detector.md) | 2 | 99.2 | 119 |
| [ChaikinMoneyFlow](../algorithms/chaikin-money-flow.md) | 4 | 222 | 248 |
| [ChaikinOscillator](../algorithms/chaikin-oscillator.md) | 4 | 132 | 155 |
| [ChandeMomentumOscillator](../algorithms/chande-momentum-oscillator.md) | 1 | 97.1 | 117 |
| [ChoppinessIndex](../algorithms/choppiness-index.md) | 3 | 272 | 297 |
| [ClosePressureReversalSignal](../algorithms/close-pressure-reversal-signal.md) | 5 | 343 | 486 |
| [CointegrationBreakdownMonitor](../algorithms/cointegration-breakdown-monitor.md) | 2 | 112 | 132 |
| [ConnorsRSI](../algorithms/connors-rsi.md) | 1 | 877 | 895 |
| [CommodityChannelIndex](../algorithms/commodity-channel-index.md) | 3 | 208 | 232 |
| [CoppockCurve](../algorithms/coppock-curve.md) | 1 | 94.9 | 114 |
| [Correlation](../algorithms/correlation.md) | 2 | 130 | 152 |
| [CorrelationRegimeDetector](../algorithms/correlation-regime-detector.md) | 2 | 134 | 154 |
| [CrossAssetCorrelationBreakDetector](../algorithms/cross-asset-correlation-break-detector.md) | 2 | 158 | 178 |
| [CumulativeReturn](../algorithms/cumulative-return.md) | 1 | 64.5 | 83.5 |
| [CUSUM](../algorithms/cusum.md) | 1 | 70.9 | 87.3 |
| [DDM](../algorithms/ddm.md) | 1 | 103 | 121 |
| [DailyLogReturn](../algorithms/daily-log-return.md) | 1 | 105 | 125 |
| [DailyReturn](../algorithms/daily-return.md) | 1 | 65.4 | 84.5 |
| [Delay](../algorithms/delay.md) | 1 | 63.6 | 80.3 |
| [DetrendedPriceOscillator](../algorithms/detrended-price-oscillator.md) | 1 | 73.2 | 93.2 |
| [DirectionalMovementIndex](../algorithms/directional-movement-index.md) | 3 | 162 | 186 |
| [DoubleEMA](../algorithms/double-ema.md) | 1 | 71.7 | 87.7 |
| [DonchianChannel](../algorithms/donchian-channel.md) | 3 | 179 | 298 |
| [EDDM](../algorithms/eddm.md) | 1 | 76.2 | 93.7 |
| [EhlersOptimalTrackingFilter](../algorithms/ehlers-optimal-tracking-filter.md) | 2 | 132 | 152 |
| [ElderRayIndex](../algorithms/elder-ray-index.md) | 3 | 111 | 195 |
| [EaseOfMovement](../algorithms/ease-of-movement.md) | 3 | 168 | 260 |
| [ExecutionCostSlippageRegimeDetector](../algorithms/execution-cost-slippage-regime-detector.md) | 3 | 114 | 135 |
| [FastStochastic](../algorithms/fast-stochastic.md) | 3 | 186 | 298 |
| [FeatureDistributionDriftDetector](../algorithms/feature-distribution-drift-detector.md) | 1 | 1641 | 1662 |
| [FibonacciRetracementLevels](../algorithms/fibonacci-retracement-levels.md) | 2 | 144 | 243 |
| [FisherTransform](../algorithms/fisher-transform.md) | 2 | 218 | 240 |
| [ForceIndex](../algorithms/force-index.md) | 2 | 90.3 | 112 |
| [FractalAdaptiveMovingAverage](../algorithms/fractal-adaptive-moving-average.md) | 1 | 335 | 356 |
| [GaussianProcessRegressionBands](../algorithms/gaussian-process-regression-bands.md) | 1 | 14599 | 14741 |
| [High](../algorithms/high.md) | 1 | 89.7 | 108 |
| [HighIndex](../algorithms/high-index.md) | 1 | 91.1 | 109 |
| [HighLow](../algorithms/high-low.md) | 1 | 111 | 193 |
| [HighLowIndex](../algorithms/high-low-index.md) | 1 | 112 | 194 |
| [HDDM](../algorithms/hddm.md) | 1 | 157 | 177 |
| [HeikinAshiTransform](../algorithms/heikin-ashi-transform.md) | 4 | 128 | 211 |
| [HiddenSemiMarkovRegimeFilter](../algorithms/hidden-semi-markov-regime-filter.md) | 1 | 264 | 287 |
| [HitRateDriftDetector](../algorithms/hit-rate-drift-detector.md) | 1 | 70.8 | 87.3 |
| [HullMovingAverage](../algorithms/hull-moving-average.md) | 1 | 104 | 124 |
| [Ichimoku](../algorithms/ichimoku.md) | 2 | 229 | 353 |
| [IntradayClockEchoSignal](../algorithms/intraday-clock-echo-signal.md) | 5 | 548 | 678 |
| [InteractingMultipleModelFilter](../algorithms/interacting-multiple-model-filter.md) | 1 | 594 | 692 |
| [KSTOscillator](../algorithms/kst-oscillator.md) | 1 | 127 | 202 |
| [KalmanExtremumTrend](../algorithms/kalman-extremum-trend.md) | 3 | 504 | 619 |
| [KalmanHedgeRatio](../algorithms/kalman-hedge-ratio.md) | 2 | 422 | 541 |
| [KalmanInnovationZScore](../algorithms/kalman-innovation-z-score.md) | 1 | 382 | 402 |
| [KalmanLocalLinearTrend](../algorithms/kalman-local-linear-trend.md) | 1 | 376 | 494 |
| [KalmanMovingAverage](../algorithms/kalman-moving-average.md) | 1 | 378 | 396 |
| [KalmanPredictionBands](../algorithms/kalman-prediction-bands.md) | 1 | 388 | 493 |
| [KalmanRegressionChannel](../algorithms/kalman-regression-channel.md) | 2 | 435 | 553 |
| [KalmanTrendSignal](../algorithms/kalman-trend-signal.md) | 1 | 381 | 497 |
| [KalmanVelocityOscillator](../algorithms/kalman-velocity-oscillator.md) | 1 | 377 | 396 |
| [Kama](../algorithms/kama.md) | 1 | 80.2 | 99.7 |
| [KeltnerChannel](../algorithms/keltner-channel.md) | 3 | 123 | 213 |
| [KeltnerChannelOriginal](../algorithms/keltner-channel-original.md) | 3 | 140 | 222 |
| [KlingerVolumeOscillator](../algorithms/klinger-volume-oscillator.md) | 4 | 155 | 241 |
| [KSWIN](../algorithms/kswin.md) | 1 | 3689 | 3715 |
| [KyleLambda](../algorithms/kyle-lambda.md) | 2 | 121 | 143 |
| [LeadLagRegimeDetector](../algorithms/lead-lag-regime-detector.md) | 2 | 98.0 | 119 |
| [LiquidityDroughtDetector](../algorithms/liquidity-drought-detector.md) | 3 | 114 | 134 |
| [LiquidityRegimeDetector](../algorithms/liquidity-regime-detector.md) | 2 | 104 | 124 |
| [LinearRegression](../algorithms/linear-regression.md) | 1 | 147 | 168 |
| [LinearRegressionAngle](../algorithms/linear-regression-angle.md) | 1 | 148 | 171 |
| [LinearRegressionIntercept](../algorithms/linear-regression-intercept.md) | 1 | 147 | 168 |
| [LinearRegressionSlope](../algorithms/linear-regression-slope.md) | 1 | 147 | 168 |
| [Low](../algorithms/low.md) | 1 | 90.7 | 109 |
| [LowIndex](../algorithms/low-index.md) | 1 | 90.3 | 109 |
| [MACDFix](../algorithms/macd-fix.md) | 1 | 75.7 | 92.6 |
| [MassIndex](../algorithms/mass-index.md) | 2 | 101 | 123 |
| [MarketOpenCloseTransitionDetector](../algorithms/market-open-close-transition-detector.md) | 1 | 65.5 | 81.1 |
| [MatchedFlowConformalSignal](../algorithms/matched-flow-conformal-signal.md) | 5 | 4377 | 4527 |
| [MedianPrice](../algorithms/median-price.md) | 2 | 85.0 | 108 |
| [MesaAdaptiveMovingAverage](../algorithms/mesa-adaptive-moving-average.md) | 1 | 293 | 393 |
| [MicrostructureNoiseRegimeDetector](../algorithms/microstructure-noise-regime-detector.md) | 3 | 111 | 132 |
| [MidPoint](../algorithms/mid-point.md) | 1 | 119 | 140 |
| [MidPrice](../algorithms/mid-price.md) | 2 | 155 | 179 |
| [MinusDirectionalIndicator](../algorithms/minus-directional-indicator.md) | 3 | 161 | 181 |
| [MinusDirectionalMovement](../algorithms/minus-directional-movement.md) | 2 | 100 | 121 |
| [Momentum](../algorithms/momentum.md) | 1 | 71.7 | 87.0 |
| [MoneyFlowIndex](../algorithms/money-flow-index.md) | 4 | 173 | 197 |
| [NadarayaWatsonEnvelope](../algorithms/nadaraya-watson-envelope.md) | 1 | 387 | 471 |
| [NegativeVolumeIndex](../algorithms/negative-volume-index.md) | 2 | 95.1 | 118 |
| [NormalizedATR](../algorithms/normalized-atr.md) | 3 | 130 | 154 |
| [OnBalanceVolume](../algorithms/on-balance-volume.md) | 2 | 96.0 | 117 |
| [OnlineGaussianMixtureRegimeFilter](../algorithms/online-gaussian-mixture-regime-filter.md) | 1 | 213 | 235 |
| [OnlineHMMRegimeFilter](../algorithms/online-hmm-regime-filter.md) | 1 | 262 | 285 |
| [OnlineMarkovSwitchingVolatilityFilter](../algorithms/online-markov-switching-volatility-filter.md) | 1 | 281 | 303 |
| [OrderFlowImbalance](../algorithms/order-flow-imbalance.md) | 4 | 134 | 155 |
| [OrderFlowImbalanceRegimeDetector](../algorithms/order-flow-imbalance-regime-detector.md) | 4 | 137 | 159 |
| [PageHinkley](../algorithms/page-hinkley.md) | 1 | 78.2 | 96.6 |
| [PairsSpreadRegimeDetector](../algorithms/pairs-spread-regime-detector.md) | 2 | 112 | 134 |
| [ParticleFilterTrend](../algorithms/particle-filter-trend.md) | 1 | 13061 | 13157 |
| [ParabolicSAR](../algorithms/parabolic-sar.md) | 2 | 98.8 | 120 |
| [PercentagePrice](../algorithms/percentage-price.md) | 1 | 78.1 | 159 |
| [PercentageVolume](../algorithms/percentage-volume.md) | 1 | 78.3 | 160 |
| [PlusDirectionalIndicator](../algorithms/plus-directional-indicator.md) | 3 | 161 | 181 |
| [PlusDirectionalMovement](../algorithms/plus-directional-movement.md) | 2 | 100 | 121 |
| [PredictionErrorDriftDetector](../algorithms/prediction-error-drift-detector.md) | 2 | 108 | 128 |
| [QuoteMessageRateRegimeDetector](../algorithms/quote-message-rate-regime-detector.md) | 1 | 74.9 | 91.8 |
| [QuoteStuffingDetector](../algorithms/quote-stuffing-detector.md) | 2 | 92.8 | 114 |
| [RateOfChangePercentage](../algorithms/rate-of-change-percentage.md) | 1 | 65.9 | 85.2 |
| [RateOfChangeRatio](../algorithms/rate-of-change-ratio.md) | 1 | 66.4 | 84.2 |
| [RateOfChangeRatio100](../algorithms/rate-of-change-ratio-100.md) | 1 | 68.0 | 86.3 |
| [RenkoBrickGenerator](../algorithms/renko-brick-generator.md) | 1 | 71.1 | 147 |
| [ResidualDriftDetector](../algorithms/residual-drift-detector.md) | 1 | 81.4 | 101 |
| [RelativeVigorIndex](../algorithms/relative-vigor-index.md) | 4 | 218 | 336 |
| [RealizedVarianceRegimeDetector](../algorithms/realized-variance-regime-detector.md) | 1 | 77.1 | 97.5 |
| [RollingBetaShiftDetector](../algorithms/rolling-beta-shift-detector.md) | 2 | 139 | 162 |
| [RollingCorrelationShiftDetector](../algorithms/rolling-correlation-shift-detector.md) | 2 | 155 | 177 |
| [RollingMeanShiftDetector](../algorithms/rolling-mean-shift-detector.md) | 1 | 154 | 172 |
| [RollingMeanVarianceShiftDetector](../algorithms/rolling-mean-variance-shift-detector.md) | 1 | 196 | 216 |
| [RollingSpreadLiquidityShiftDetector](../algorithms/rolling-spread-liquidity-shift-detector.md) | 4 | 155 | 178 |
| [RollingVarianceShiftDetector](../algorithms/rolling-variance-shift-detector.md) | 1 | 173 | 189 |
| [SavitzkyGolayFilter](../algorithms/savitzky-golay-filter.md) | 1 | 164 | 242 |
| [SchaffTrendCycle](../algorithms/schaff-trend-cycle.md) | 1 | 168 | 186 |
| [SpreadFeatures](../algorithms/spread-features.md) | 3 | 133 | 216 |
| [SpreadExplosionDetector](../algorithms/spread-explosion-detector.md) | 2 | 92.4 | 112 |
| [SpreadRegimeDetector](../algorithms/spread-regime-detector.md) | 2 | 97.4 | 119 |
| [StdDev](../algorithms/std-dev.md) | 1 | 104 | 126 |
| [StickyHMMRegimeFilter](../algorithms/sticky-hmm-regime-filter.md) | 1 | 262 | 285 |
| [StochRSI](../algorithms/stoch-rsi.md) | 1 | 167 | 188 |
| [Stochastic](../algorithms/stochastic.md) | 3 | 203 | 320 |
| [SuperTrend](../algorithms/super-trend.md) | 3 | 136 | 228 |
| [Summation](../algorithms/summation.md) | 1 | 65.3 | 82.8 |
| [T3MovingAverage](../algorithms/t-3-moving-average.md) | 1 | 86.9 | 102 |
| [ThresholdRegimeDetector](../algorithms/threshold-regime-detector.md) | 1 | 64.6 | 80.8 |
| [TimeSeriesForecast](../algorithms/time-series-forecast.md) | 1 | 146 | 168 |
| [TradeIntensityRegimeDetector](../algorithms/trade-intensity-regime-detector.md) | 1 | 77.5 | 94.7 |
| [TrendChopRegimeDetector](../algorithms/trend-chop-regime-detector.md) | 3 | 137 | 163 |
| [TwoFactorKalmanTrendFilter](../algorithms/two-factor-kalman-trend-filter.md) | 1 | 377 | 494 |
| [TrueRange](../algorithms/true-range.md) | 3 | 117 | 137 |
| [TriangularMovingAverage](../algorithms/triangular-moving-average.md) | 1 | 89.7 | 108 |
| [TripleEMA](../algorithms/triple-ema.md) | 1 | 75.5 | 91.1 |
| [Trix](../algorithms/trix.md) | 1 | 77.4 | 94.9 |
| [TypicalPrice](../algorithms/typical-price.md) | 3 | 105 | 129 |
| [UltimateOscillator](../algorithms/ultimate-oscillator.md) | 3 | 180 | 205 |
| [UlcerIndex](../algorithms/ulcer-index.md) | 1 | 121 | 143 |
| [VPIN](../algorithms/vpin.md) | 2 | 218 | 239 |
| [Variance](../algorithms/variance.md) | 1 | 87.6 | 107 |
| [VariableIndexDynamicAverage](../algorithms/variable-index-dynamic-average.md) | 1 | 106 | 124 |
| [VolatilityBreakoutDetector](../algorithms/volatility-breakout-detector.md) | 1 | 84.4 | 105 |
| [VolatilityCompressionExpansionDetector](../algorithms/volatility-compression-expansion-detector.md) | 1 | 123 | 139 |
| [VolatilityRegimeDetector](../algorithms/volatility-regime-detector.md) | 1 | 80.8 | 97.4 |
| [VolumeProfile](../algorithms/volume-profile.md) | 2 | 895 | 988 |
| [VolumePriceTrend](../algorithms/volume-price-trend.md) | 2 | 92.2 | 115 |
| [VolumeRegimeDetector](../algorithms/volume-regime-detector.md) | 1 | 77.4 | 94.4 |
| [VolumeWeightedAveragePrice](../algorithms/volume-weighted-average-price.md) | 4 | 207 | 232 |
| [VolumeWeightedMovingAverage](../algorithms/volume-weighted-moving-average.md) | 2 | 99.8 | 125 |
| [Vortex](../algorithms/vortex.md) | 3 | 134 | 228 |
| [WeightedClosePrice](../algorithms/weighted-close-price.md) | 3 | 105 | 128 |
| [WeightedMovingAverage](../algorithms/weighted-moving-average.md) | 1 | 73.0 | 93.9 |
| [WilliamsR](../algorithms/williams-r.md) | 3 | 165 | 190 |
| [ZigZagSwingDetector](../algorithms/zig-zag-swing-detector.md) | 1 | 72.2 | 151 |
