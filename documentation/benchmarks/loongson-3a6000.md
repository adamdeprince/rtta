# Loongson-3A6000 Benchmarks

These results were collected on 2026-07-19. Public documentation identifies benchmark systems by CPU type rather than hostname.

## System

- CPU: **Loongson-3A6000**
- Architecture: `loongarch64`
- Platform: `Linux-5.4.18-167-generic-loongarch64-with-glibc2.28`
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
- Median `advance(...)` latency, update state only: **101 ns/update**.
- Median `update(...)` latency, update state and return a value/result: **129 ns/update**.

| Algorithm | Inputs | update only: `advance(...)` ns/update | update + return: `update(...)` ns/update |
|---|---:|---:|---:|
| [ATR](../algorithms/atr.zh-CN.md) | 3 | 98.0 | 120 |
| [ATRP](../algorithms/atrp.zh-CN.md) | 3 | 98.2 | 131 |
| [ATRRegimeDetector](../algorithms/atr-regime-detector.zh-CN.md) | 3 | 100 | 123 |
| [ADWIN](../algorithms/adwin.zh-CN.md) | 1 | 1044 | 1063 |
| [EMA](../algorithms/ema.zh-CN.md) | 1 | 52.9 | 70.2 |
| [EWMA](../algorithms/ewma.zh-CN.md) | 1 | 51.6 | 71.3 |
| [EWMAZScoreShiftDetector](../algorithms/ewmaz-score-shift-detector.zh-CN.md) | 1 | 72.9 | 92.3 |
| [MACD](../algorithms/macd.zh-CN.md) | 1 | 56.9 | 78.2 |
| [ROC](../algorithms/roc.zh-CN.md) | 1 | 51.9 | 77.8 |
| [RSI](../algorithms/rsi.zh-CN.md) | 1 | 101 | 121 |
| [SMA](../algorithms/sma.zh-CN.md) | 1 | 62.0 | 82.0 |
| [TSI](../algorithms/tsi.zh-CN.md) | 1 | 67.2 | 88.2 |
| [AbsolutePriceOscillator](../algorithms/absolute-price-oscillator.zh-CN.md) | 1 | 54.4 | 74.8 |
| [AccumulationDistribution](../algorithms/accumulation-distribution.zh-CN.md) | 4 | 109 | 134 |
| [AlphaBetaGammaTrackingFilter](../algorithms/alpha-beta-gamma-tracking-filter.zh-CN.md) | 1 | 70.7 | 140 |
| [AmihudIlliquidity](../algorithms/amihud-illiquidity.zh-CN.md) | 2 | 106 | 127 |
| [AnchoredVWAP](../algorithms/anchored-vwap.zh-CN.md) | 5 | 140 | 157 |
| [Aroon](../algorithms/aroon.zh-CN.md) | 2 | 119 | 195 |
| [AroonOscillator](../algorithms/aroon-oscillator.zh-CN.md) | 2 | 120 | 143 |
| [AverageDirectionalMovementIndex](../algorithms/average-directional-movement-index.zh-CN.md) | 3 | 153 | 176 |
| [AverageDirectionalMovementIndexRating](../algorithms/average-directional-movement-index-rating.zh-CN.md) | 3 | 158 | 180 |
| [AveragePrice](../algorithms/average-price.zh-CN.md) | 4 | 103 | 125 |
| [AuctionContinuousMarketTransitionDetector](../algorithms/auction-continuous-market-transition-detector.zh-CN.md) | 1 | 51.9 | 70.3 |
| [AwesomeOscillator](../algorithms/awesome-oscillator.zh-CN.md) | 2 | 98.7 | 121 |
| [BalanceOfPower](../algorithms/balance-of-power.zh-CN.md) | 4 | 103 | 127 |
| [Beta](../algorithms/beta.zh-CN.md) | 2 | 84.2 | 114 |
| [BetaRegimeDetector](../algorithms/beta-regime-detector.zh-CN.md) | 2 | 92.7 | 115 |
| [BidAskBounceRegimeDetector](../algorithms/bid-ask-bounce-regime-detector.zh-CN.md) | 3 | 104 | 131 |
| [BollingerBands](../algorithms/bollinger-bands.zh-CN.md) | 1 | 84.1 | 159 |
| [BoundedBOCPD](../algorithms/bounded-bocpd.zh-CN.md) | 1 | 5365 | 5388 |
| [CalibrationDriftDetector](../algorithms/calibration-drift-detector.zh-CN.md) | 2 | 85.9 | 106 |
| [ChaikinMoneyFlow](../algorithms/chaikin-money-flow.zh-CN.md) | 4 | 125 | 148 |
| [ChaikinOscillator](../algorithms/chaikin-oscillator.zh-CN.md) | 4 | 116 | 139 |
| [ChandeMomentumOscillator](../algorithms/chande-momentum-oscillator.zh-CN.md) | 1 | 64.2 | 95.8 |
| [ChoppinessIndex](../algorithms/choppiness-index.zh-CN.md) | 3 | 231 | 256 |
| [ClosePressureReversalSignal](../algorithms/close-pressure-reversal-signal.zh-CN.md) | 5 | 300 | 447 |
| [CointegrationBreakdownMonitor](../algorithms/cointegration-breakdown-monitor.zh-CN.md) | 2 | 110 | 127 |
| [ConnorsRSI](../algorithms/connors-rsi.zh-CN.md) | 1 | 215 | 234 |
| [CommodityChannelIndex](../algorithms/commodity-channel-index.zh-CN.md) | 3 | 147 | 170 |
| [CoppockCurve](../algorithms/coppock-curve.zh-CN.md) | 1 | 81.9 | 101 |
| [Correlation](../algorithms/correlation.zh-CN.md) | 2 | 90.4 | 135 |
| [CorrelationRegimeDetector](../algorithms/correlation-regime-detector.zh-CN.md) | 2 | 118 | 135 |
| [CrossAssetCorrelationBreakDetector](../algorithms/cross-asset-correlation-break-detector.zh-CN.md) | 2 | 123 | 144 |
| [CumulativeReturn](../algorithms/cumulative-return.zh-CN.md) | 1 | 50.8 | 76.8 |
| [CUSUM](../algorithms/cusum.zh-CN.md) | 1 | 56.3 | 77.0 |
| [DDM](../algorithms/ddm.zh-CN.md) | 1 | 93.6 | 113 |
| [DailyLogReturn](../algorithms/daily-log-return.zh-CN.md) | 1 | 51.3 | 114 |
| [DailyReturn](../algorithms/daily-return.zh-CN.md) | 1 | 50.6 | 76.9 |
| [Delay](../algorithms/delay.zh-CN.md) | 1 | 52.3 | 71.5 |
| [DetrendedPriceOscillator](../algorithms/detrended-price-oscillator.zh-CN.md) | 1 | 63.6 | 83.0 |
| [DirectionalMovementIndex](../algorithms/directional-movement-index.zh-CN.md) | 3 | 141 | 164 |
| [DoubleEMA](../algorithms/double-ema.zh-CN.md) | 1 | 55.0 | 74.7 |
| [DonchianChannel](../algorithms/donchian-channel.zh-CN.md) | 3 | 144 | 246 |
| [EDDM](../algorithms/eddm.zh-CN.md) | 1 | 64.3 | 82.5 |
| [EhlersOptimalTrackingFilter](../algorithms/ehlers-optimal-tracking-filter.zh-CN.md) | 2 | 120 | 141 |
| [ElderRayIndex](../algorithms/elder-ray-index.zh-CN.md) | 3 | 94.5 | 177 |
| [EaseOfMovement](../algorithms/ease-of-movement.zh-CN.md) | 3 | 114 | 191 |
| [ExecutionCostSlippageRegimeDetector](../algorithms/execution-cost-slippage-regime-detector.zh-CN.md) | 3 | 99.0 | 120 |
| [FastStochastic](../algorithms/fast-stochastic.zh-CN.md) | 3 | 150 | 230 |
| [FeatureDistributionDriftDetector](../algorithms/feature-distribution-drift-detector.zh-CN.md) | 1 | 1050 | 1066 |
| [FibonacciRetracementLevels](../algorithms/fibonacci-retracement-levels.zh-CN.md) | 2 | 112 | 198 |
| [FisherTransform](../algorithms/fisher-transform.zh-CN.md) | 2 | 188 | 210 |
| [ForceIndex](../algorithms/force-index.zh-CN.md) | 2 | 78.2 | 97.0 |
| [FractalAdaptiveMovingAverage](../algorithms/fractal-adaptive-moving-average.zh-CN.md) | 1 | 216 | 235 |
| [GaussianProcessRegressionBands](../algorithms/gaussian-process-regression-bands.zh-CN.md) | 1 | 1914 | 1989 |
| [High](../algorithms/high.zh-CN.md) | 1 | 70.7 | 92.2 |
| [HighIndex](../algorithms/high-index.zh-CN.md) | 1 | 70.4 | 93.1 |
| [HighLow](../algorithms/high-low.zh-CN.md) | 1 | 83.8 | 156 |
| [HighLowIndex](../algorithms/high-low-index.zh-CN.md) | 1 | 82.4 | 164 |
| [HDDM](../algorithms/hddm.zh-CN.md) | 1 | 153 | 171 |
| [HeikinAshiTransform](../algorithms/heikin-ashi-transform.zh-CN.md) | 4 | 110 | 187 |
| [HiddenSemiMarkovRegimeFilter](../algorithms/hidden-semi-markov-regime-filter.zh-CN.md) | 1 | 213 | 235 |
| [HitRateDriftDetector](../algorithms/hit-rate-drift-detector.zh-CN.md) | 1 | 56.4 | 76.6 |
| [HullMovingAverage](../algorithms/hull-moving-average.zh-CN.md) | 1 | 89.9 | 107 |
| [Ichimoku](../algorithms/ichimoku.zh-CN.md) | 2 | 156 | 246 |
| [IntradayClockEchoSignal](../algorithms/intraday-clock-echo-signal.zh-CN.md) | 5 | 506 | 633 |
| [InteractingMultipleModelFilter](../algorithms/interacting-multiple-model-filter.zh-CN.md) | 1 | 515 | 603 |
| [KSTOscillator](../algorithms/kst-oscillator.zh-CN.md) | 1 | 116 | 191 |
| [KalmanExtremumTrend](../algorithms/kalman-extremum-trend.zh-CN.md) | 3 | 233 | 327 |
| [KalmanHedgeRatio](../algorithms/kalman-hedge-ratio.zh-CN.md) | 2 | 173 | 248 |
| [KalmanInnovationZScore](../algorithms/kalman-innovation-z-score.zh-CN.md) | 1 | 165 | 184 |
| [KalmanLocalLinearTrend](../algorithms/kalman-local-linear-trend.zh-CN.md) | 1 | 148 | 218 |
| [KalmanMovingAverage](../algorithms/kalman-moving-average.zh-CN.md) | 1 | 148 | 168 |
| [KalmanPredictionBands](../algorithms/kalman-prediction-bands.zh-CN.md) | 1 | 158 | 230 |
| [KalmanRegressionChannel](../algorithms/kalman-regression-channel.zh-CN.md) | 2 | 184 | 260 |
| [KalmanTrendSignal](../algorithms/kalman-trend-signal.zh-CN.md) | 1 | 148 | 224 |
| [KalmanVelocityOscillator](../algorithms/kalman-velocity-oscillator.zh-CN.md) | 1 | 148 | 166 |
| [Kama](../algorithms/kama.zh-CN.md) | 1 | 74.3 | 93.9 |
| [KeltnerChannel](../algorithms/keltner-channel.zh-CN.md) | 3 | 102 | 181 |
| [KeltnerChannelOriginal](../algorithms/keltner-channel-original.zh-CN.md) | 3 | 123 | 205 |
| [KlingerVolumeOscillator](../algorithms/klinger-volume-oscillator.zh-CN.md) | 4 | 142 | 217 |
| [KSWIN](../algorithms/kswin.zh-CN.md) | 1 | 3144 | 3166 |
| [KyleLambda](../algorithms/kyle-lambda.zh-CN.md) | 2 | 111 | 132 |
| [LeadLagRegimeDetector](../algorithms/lead-lag-regime-detector.zh-CN.md) | 2 | 89.4 | 110 |
| [LiquidityDroughtDetector](../algorithms/liquidity-drought-detector.zh-CN.md) | 3 | 98.4 | 122 |
| [LiquidityRegimeDetector](../algorithms/liquidity-regime-detector.zh-CN.md) | 2 | 95.1 | 116 |
| [LinearRegression](../algorithms/linear-regression.zh-CN.md) | 1 | 121 | 94.3 |
| [LinearRegressionAngle](../algorithms/linear-regression-angle.zh-CN.md) | 1 | 121 | 137 |
| [LinearRegressionIntercept](../algorithms/linear-regression-intercept.zh-CN.md) | 1 | 121 | 92.9 |
| [LinearRegressionSlope](../algorithms/linear-regression-slope.zh-CN.md) | 1 | 121 | 86.8 |
| [Low](../algorithms/low.zh-CN.md) | 1 | 70.9 | 92.7 |
| [LowIndex](../algorithms/low-index.zh-CN.md) | 1 | 70.4 | 93.2 |
| [MACDFix](../algorithms/macd-fix.zh-CN.md) | 1 | 57.0 | 79.8 |
| [MassIndex](../algorithms/mass-index.zh-CN.md) | 2 | 91.6 | 114 |
| [MarketOpenCloseTransitionDetector](../algorithms/market-open-close-transition-detector.zh-CN.md) | 1 | 51.9 | 72.2 |
| [MatchedFlowConformalSignal](../algorithms/matched-flow-conformal-signal.zh-CN.md) | 5 | 2225 | 2377 |
| [MedianPrice](../algorithms/median-price.zh-CN.md) | 2 | 74.6 | 95.2 |
| [MesaAdaptiveMovingAverage](../algorithms/mesa-adaptive-moving-average.zh-CN.md) | 1 | 247 | 320 |
| [MicrostructureNoiseRegimeDetector](../algorithms/microstructure-noise-regime-detector.zh-CN.md) | 3 | 95.8 | 118 |
| [MidPoint](../algorithms/mid-point.zh-CN.md) | 1 | 84.5 | 105 |
| [MidPrice](../algorithms/mid-price.zh-CN.md) | 2 | 113 | 134 |
| [MinusDirectionalIndicator](../algorithms/minus-directional-indicator.zh-CN.md) | 3 | 120 | 142 |
| [MinusDirectionalMovement](../algorithms/minus-directional-movement.zh-CN.md) | 2 | 87.7 | 110 |
| [Momentum](../algorithms/momentum.zh-CN.md) | 1 | 52.4 | 71.3 |
| [MoneyFlowIndex](../algorithms/money-flow-index.zh-CN.md) | 4 | 146 | 169 |
| [NadarayaWatsonEnvelope](../algorithms/nadaraya-watson-envelope.zh-CN.md) | 1 | 202 | 288 |
| [NegativeVolumeIndex](../algorithms/negative-volume-index.zh-CN.md) | 2 | 83.6 | 104 |
| [NormalizedATR](../algorithms/normalized-atr.zh-CN.md) | 3 | 98.3 | 134 |
| [OnBalanceVolume](../algorithms/on-balance-volume.zh-CN.md) | 2 | 84.1 | 104 |
| [OnlineGaussianMixtureRegimeFilter](../algorithms/online-gaussian-mixture-regime-filter.zh-CN.md) | 1 | 177 | 200 |
| [OnlineHMMRegimeFilter](../algorithms/online-hmm-regime-filter.zh-CN.md) | 1 | 209 | 232 |
| [OnlineMarkovSwitchingVolatilityFilter](../algorithms/online-markov-switching-volatility-filter.zh-CN.md) | 1 | 226 | 247 |
| [OrderFlowImbalance](../algorithms/order-flow-imbalance.zh-CN.md) | 4 | 116 | 137 |
| [OrderFlowImbalanceRegimeDetector](../algorithms/order-flow-imbalance-regime-detector.zh-CN.md) | 4 | 119 | 140 |
| [PageHinkley](../algorithms/page-hinkley.zh-CN.md) | 1 | 68.3 | 88.2 |
| [PairsSpreadRegimeDetector](../algorithms/pairs-spread-regime-detector.zh-CN.md) | 2 | 107 | 123 |
| [ParticleFilterTrend](../algorithms/particle-filter-trend.zh-CN.md) | 1 | 11908 | 12042 |
| [ParabolicSAR](../algorithms/parabolic-sar.zh-CN.md) | 2 | 77.9 | 97.3 |
| [PercentagePrice](../algorithms/percentage-price.zh-CN.md) | 1 | 69.3 | 143 |
| [PercentageVolume](../algorithms/percentage-volume.zh-CN.md) | 1 | 68.9 | 145 |
| [PlusDirectionalIndicator](../algorithms/plus-directional-indicator.zh-CN.md) | 3 | 119 | 142 |
| [PlusDirectionalMovement](../algorithms/plus-directional-movement.zh-CN.md) | 2 | 87.7 | 110 |
| [PredictionErrorDriftDetector](../algorithms/prediction-error-drift-detector.zh-CN.md) | 2 | 93.3 | 114 |
| [QuoteMessageRateRegimeDetector](../algorithms/quote-message-rate-regime-detector.zh-CN.md) | 1 | 61.2 | 81.2 |
| [QuoteStuffingDetector](../algorithms/quote-stuffing-detector.zh-CN.md) | 2 | 83.3 | 104 |
| [RateOfChangePercentage](../algorithms/rate-of-change-percentage.zh-CN.md) | 1 | 52.6 | 75.1 |
| [RateOfChangeRatio](../algorithms/rate-of-change-ratio.zh-CN.md) | 1 | 52.7 | 74.3 |
| [RateOfChangeRatio100](../algorithms/rate-of-change-ratio-100.zh-CN.md) | 1 | 52.7 | 76.0 |
| [RenkoBrickGenerator](../algorithms/renko-brick-generator.zh-CN.md) | 1 | 55.3 | 131 |
| [ResidualDriftDetector](../algorithms/residual-drift-detector.zh-CN.md) | 1 | 71.8 | 91.1 |
| [RelativeVigorIndex](../algorithms/relative-vigor-index.zh-CN.md) | 4 | 157 | 234 |
| [RealizedVarianceRegimeDetector](../algorithms/realized-variance-regime-detector.zh-CN.md) | 1 | 63.3 | 81.4 |
| [RollingBetaShiftDetector](../algorithms/rolling-beta-shift-detector.zh-CN.md) | 2 | 108 | 130 |
| [RollingCorrelationShiftDetector](../algorithms/rolling-correlation-shift-detector.zh-CN.md) | 2 | 126 | 148 |
| [RollingMeanShiftDetector](../algorithms/rolling-mean-shift-detector.zh-CN.md) | 1 | 129 | 149 |
| [RollingMeanVarianceShiftDetector](../algorithms/rolling-mean-variance-shift-detector.zh-CN.md) | 1 | 181 | 199 |
| [RollingSpreadLiquidityShiftDetector](../algorithms/rolling-spread-liquidity-shift-detector.zh-CN.md) | 4 | 130 | 150 |
| [RollingVarianceShiftDetector](../algorithms/rolling-variance-shift-detector.zh-CN.md) | 1 | 146 | 166 |
| [SavitzkyGolayFilter](../algorithms/savitzky-golay-filter.zh-CN.md) | 1 | 81.3 | 166 |
| [SchaffTrendCycle](../algorithms/schaff-trend-cycle.zh-CN.md) | 1 | 107 | 126 |
| [SpreadFeatures](../algorithms/spread-features.zh-CN.md) | 3 | 108 | 182 |
| [SpreadExplosionDetector](../algorithms/spread-explosion-detector.zh-CN.md) | 2 | 79.6 | 100.0 |
| [SpreadRegimeDetector](../algorithms/spread-regime-detector.zh-CN.md) | 2 | 86.5 | 107 |
| [StdDev](../algorithms/std-dev.zh-CN.md) | 1 | 78.0 | 101 |
| [StickyHMMRegimeFilter](../algorithms/sticky-hmm-regime-filter.zh-CN.md) | 1 | 208 | 232 |
| [StochRSI](../algorithms/stoch-rsi.zh-CN.md) | 1 | 136 | 165 |
| [Stochastic](../algorithms/stochastic.zh-CN.md) | 3 | 169 | 256 |
| [SuperTrend](../algorithms/super-trend.zh-CN.md) | 3 | 113 | 198 |
| [Summation](../algorithms/summation.zh-CN.md) | 1 | 52.7 | 73.1 |
| [T3MovingAverage](../algorithms/t-3-moving-average.zh-CN.md) | 1 | 72.8 | 92.8 |
| [ThresholdRegimeDetector](../algorithms/threshold-regime-detector.zh-CN.md) | 1 | 50.1 | 71.2 |
| [TimeSeriesForecast](../algorithms/time-series-forecast.zh-CN.md) | 1 | 121 | 94.4 |
| [TradeIntensityRegimeDetector](../algorithms/trade-intensity-regime-detector.zh-CN.md) | 1 | 63.9 | 84.2 |
| [TrendChopRegimeDetector](../algorithms/trend-chop-regime-detector.zh-CN.md) | 3 | 105 | 126 |
| [TwoFactorKalmanTrendFilter](../algorithms/two-factor-kalman-trend-filter.zh-CN.md) | 1 | 148 | 218 |
| [TrueRange](../algorithms/true-range.zh-CN.md) | 3 | 90.5 | 112 |
| [TriangularMovingAverage](../algorithms/triangular-moving-average.zh-CN.md) | 1 | 82.5 | 101 |
| [TripleEMA](../algorithms/triple-ema.zh-CN.md) | 1 | 56.7 | 78.3 |
| [Trix](../algorithms/trix.zh-CN.md) | 1 | 68.8 | 88.8 |
| [TypicalPrice](../algorithms/typical-price.zh-CN.md) | 3 | 90.2 | 116 |
| [UltimateOscillator](../algorithms/ultimate-oscillator.zh-CN.md) | 3 | 128 | 155 |
| [UlcerIndex](../algorithms/ulcer-index.zh-CN.md) | 1 | 105 | 126 |
| [VPIN](../algorithms/vpin.zh-CN.md) | 2 | 195 | 216 |
| [Variance](../algorithms/variance.zh-CN.md) | 1 | 54.3 | 79.1 |
| [VariableIndexDynamicAverage](../algorithms/variable-index-dynamic-average.zh-CN.md) | 1 | 83.8 | 103 |
| [VolatilityBreakoutDetector](../algorithms/volatility-breakout-detector.zh-CN.md) | 1 | 72.6 | 92.1 |
| [VolatilityCompressionExpansionDetector](../algorithms/volatility-compression-expansion-detector.zh-CN.md) | 1 | 96.2 | 120 |
| [VolatilityRegimeDetector](../algorithms/volatility-regime-detector.zh-CN.md) | 1 | 71.4 | 89.5 |
| [VolumeProfile](../algorithms/volume-profile.zh-CN.md) | 2 | 332 | 421 |
| [VolumePriceTrend](../algorithms/volume-price-trend.zh-CN.md) | 2 | 82.6 | 104 |
| [VolumeRegimeDetector](../algorithms/volume-regime-detector.zh-CN.md) | 1 | 64.1 | 83.7 |
| [VolumeWeightedAveragePrice](../algorithms/volume-weighted-average-price.zh-CN.md) | 4 | 116 | 139 |
| [VolumeWeightedMovingAverage](../algorithms/volume-weighted-moving-average.zh-CN.md) | 2 | 81.0 | 109 |
| [Vortex](../algorithms/vortex.zh-CN.md) | 3 | 109 | 219 |
| [WeightedClosePrice](../algorithms/weighted-close-price.zh-CN.md) | 3 | 90.2 | 111 |
| [WeightedMovingAverage](../algorithms/weighted-moving-average.zh-CN.md) | 1 | 53.2 | 81.8 |
| [WilliamsR](../algorithms/williams-r.zh-CN.md) | 3 | 126 | 153 |
| [ZigZagSwingDetector](../algorithms/zig-zag-swing-detector.zh-CN.md) | 1 | 57.5 | 134 |

