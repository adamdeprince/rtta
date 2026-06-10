# Intel Xeon 6975P-C Benchmarks

These results were collected on 2026-06-09. Public documentation identifies benchmark systems by CPU type rather than hostname.

## System

- CPU: **Intel Xeon 6975P-C**
- Runtime processor string: `Intel(R) Xeon(R) 6975P-C`
- Architecture: `x86_64`
- Platform: `Linux-7.0.0-1004-aws-x86_64-with-glibc2.43`
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
- Median `advance(...)` latency, update state only: **51.6 ns/update**.
- Median `update(...)` latency, update state and return a value/result: **59.4 ns/update**.

| Algorithm | Inputs | update only: `advance(...)` ns/update | update + return: `update(...)` ns/update |
|---|---:|---:|---:|
| [ATR](../algorithms/atr.md) | 3 | 58.7 | 61.7 |
| [ATRP](../algorithms/atrp.md) | 3 | 59.2 | 61.4 |
| [ATRRegimeDetector](../algorithms/atr-regime-detector.md) | 3 | 60.7 | 63.5 |
| [ADWIN](../algorithms/adwin.md) | 1 | 955 | 956 |
| [EMA](../algorithms/ema.md) | 1 | 31.4 | 34.1 |
| [EWMA](../algorithms/ewma.md) | 1 | 29.7 | 33.9 |
| [EWMAZScoreShiftDetector](../algorithms/ewmaz-score-shift-detector.md) | 1 | 36.7 | 40.8 |
| [MACD](../algorithms/macd.md) | 1 | 35.3 | 39.3 |
| [ROC](../algorithms/roc.md) | 1 | 31.0 | 34.5 |
| [RSI](../algorithms/rsi.md) | 1 | 39.5 | 43.0 |
| [SMA](../algorithms/sma.md) | 1 | 31.1 | 34.0 |
| [TSI](../algorithms/tsi.md) | 1 | 39.0 | 42.5 |
| [AbsolutePriceOscillator](../algorithms/absolute-price-oscillator.md) | 1 | 34.0 | 36.1 |
| [AccumulationDistribution](../algorithms/accumulation-distribution.md) | 4 | 56.6 | 60.1 |
| [AlphaBetaGammaTrackingFilter](../algorithms/alpha-beta-gamma-tracking-filter.md) | 1 | 31.7 | 69.0 |
| [AmihudIlliquidity](../algorithms/amihud-illiquidity.md) | 2 | 44.1 | 46.1 |
| [AnchoredVWAP](../algorithms/anchored-vwap.md) | 5 | 65.0 | 68.2 |
| [Aroon](../algorithms/aroon.md) | 2 | 79.2 | 113 |
| [AroonOscillator](../algorithms/aroon-oscillator.md) | 2 | 79.2 | 84.1 |
| [AverageDirectionalMovementIndex](../algorithms/average-directional-movement-index.md) | 3 | 73.0 | 74.4 |
| [AverageDirectionalMovementIndexRating](../algorithms/average-directional-movement-index-rating.md) | 3 | 76.5 | 77.9 |
| [AveragePrice](../algorithms/average-price.md) | 4 | 55.0 | 59.0 |
| [AuctionContinuousMarketTransitionDetector](../algorithms/auction-continuous-market-transition-detector.md) | 1 | 30.2 | 34.4 |
| [AwesomeOscillator](../algorithms/awesome-oscillator.md) | 2 | 45.8 | 48.9 |
| [BalanceOfPower](../algorithms/balance-of-power.md) | 4 | 55.8 | 59.9 |
| [Beta](../algorithms/beta.md) | 2 | 51.2 | 53.0 |
| [BetaRegimeDetector](../algorithms/beta-regime-detector.md) | 2 | 53.0 | 55.9 |
| [BidAskBounceRegimeDetector](../algorithms/bid-ask-bounce-regime-detector.md) | 3 | 60.4 | 63.5 |
| [BollingerBands](../algorithms/bollinger-bands.md) | 1 | 38.3 | 77.8 |
| [BoundedBOCPD](../algorithms/bounded-bocpd.md) | 1 | 1592 | 1600 |
| [CalibrationDriftDetector](../algorithms/calibration-drift-detector.md) | 2 | 42.3 | 44.7 |
| [ChaikinMoneyFlow](../algorithms/chaikin-money-flow.md) | 4 | 119 | 121 |
| [ChaikinOscillator](../algorithms/chaikin-oscillator.md) | 4 | 60.2 | 64.8 |
| [ChandeMomentumOscillator](../algorithms/chande-momentum-oscillator.md) | 1 | 47.5 | 50.9 |
| [ChoppinessIndex](../algorithms/choppiness-index.md) | 3 | 120 | 122 |
| [ClosePressureReversalSignal](../algorithms/close-pressure-reversal-signal.md) | 5 | 155 | 186 |
| [CointegrationBreakdownMonitor](../algorithms/cointegration-breakdown-monitor.md) | 2 | 46.3 | 48.8 |
| [ConnorsRSI](../algorithms/connors-rsi.md) | 1 | 551 | 555 |
| [CommodityChannelIndex](../algorithms/commodity-channel-index.md) | 3 | 113 | 114 |
| [CoppockCurve](../algorithms/coppock-curve.md) | 1 | 39.8 | 43.3 |
| [Correlation](../algorithms/correlation.md) | 2 | 51.2 | 55.2 |
| [CorrelationRegimeDetector](../algorithms/correlation-regime-detector.md) | 2 | 55.0 | 58.1 |
| [CrossAssetCorrelationBreakDetector](../algorithms/cross-asset-correlation-break-detector.md) | 2 | 71.1 | 74.1 |
| [CumulativeReturn](../algorithms/cumulative-return.md) | 1 | 29.5 | 33.5 |
| [CUSUM](../algorithms/cusum.md) | 1 | 33.7 | 36.3 |
| [DDM](../algorithms/ddm.md) | 1 | 37.2 | 38.5 |
| [DailyLogReturn](../algorithms/daily-log-return.md) | 1 | 34.3 | 38.8 |
| [DailyReturn](../algorithms/daily-return.md) | 1 | 29.7 | 33.6 |
| [Delay](../algorithms/delay.md) | 1 | 29.9 | 33.8 |
| [DetrendedPriceOscillator](../algorithms/detrended-price-oscillator.md) | 1 | 34.1 | 38.7 |
| [DirectionalMovementIndex](../algorithms/directional-movement-index.md) | 3 | 69.5 | 72.1 |
| [DoubleEMA](../algorithms/double-ema.md) | 1 | 33.5 | 38.1 |
| [DonchianChannel](../algorithms/donchian-channel.md) | 3 | 89.5 | 132 |
| [EDDM](../algorithms/eddm.md) | 1 | 34.2 | 37.4 |
| [EhlersOptimalTrackingFilter](../algorithms/ehlers-optimal-tracking-filter.md) | 2 | 47.5 | 48.4 |
| [ElderRayIndex](../algorithms/elder-ray-index.md) | 3 | 52.7 | 74.7 |
| [EaseOfMovement](../algorithms/ease-of-movement.md) | 3 | 82.4 | 106 |
| [ExecutionCostSlippageRegimeDetector](../algorithms/execution-cost-slippage-regime-detector.md) | 3 | 51.8 | 56.4 |
| [FastStochastic](../algorithms/fast-stochastic.md) | 3 | 92.9 | 119 |
| [FeatureDistributionDriftDetector](../algorithms/feature-distribution-drift-detector.md) | 1 | 958 | 959 |
| [FibonacciRetracementLevels](../algorithms/fibonacci-retracement-levels.md) | 2 | 77.4 | 107 |
| [FisherTransform](../algorithms/fisher-transform.md) | 2 | 84.8 | 87.3 |
| [ForceIndex](../algorithms/force-index.md) | 2 | 40.0 | 43.5 |
| [FractalAdaptiveMovingAverage](../algorithms/fractal-adaptive-moving-average.md) | 1 | 145 | 147 |
| [GaussianProcessRegressionBands](../algorithms/gaussian-process-regression-bands.md) | 1 | 6040 | 6094 |
| [High](../algorithms/high.md) | 1 | 47.4 | 50.2 |
| [HighIndex](../algorithms/high-index.md) | 1 | 47.4 | 52.3 |
| [HighLow](../algorithms/high-low.md) | 1 | 61.2 | 95.0 |
| [HighLowIndex](../algorithms/high-low-index.md) | 1 | 61.3 | 94.7 |
| [HDDM](../algorithms/hddm.md) | 1 | 43.9 | 48.8 |
| [HeikinAshiTransform](../algorithms/heikin-ashi-transform.md) | 4 | 55.7 | 83.6 |
| [HiddenSemiMarkovRegimeFilter](../algorithms/hidden-semi-markov-regime-filter.md) | 1 | 77.9 | 81.1 |
| [HitRateDriftDetector](../algorithms/hit-rate-drift-detector.md) | 1 | 30.7 | 34.3 |
| [HullMovingAverage](../algorithms/hull-moving-average.md) | 1 | 47.0 | 50.8 |
| [Ichimoku](../algorithms/ichimoku.md) | 2 | 125 | 160 |
| [IntradayClockEchoSignal](../algorithms/intraday-clock-echo-signal.md) | 5 | 304 | 335 |
| [InteractingMultipleModelFilter](../algorithms/interacting-multiple-model-filter.md) | 1 | 194 | 221 |
| [KSTOscillator](../algorithms/kst-oscillator.md) | 1 | 46.8 | 84.2 |
| [KalmanExtremumTrend](../algorithms/kalman-extremum-trend.md) | 3 | 232 | 270 |
| [KalmanHedgeRatio](../algorithms/kalman-hedge-ratio.md) | 2 | 186 | 222 |
| [KalmanInnovationZScore](../algorithms/kalman-innovation-z-score.md) | 1 | 175 | 178 |
| [KalmanLocalLinearTrend](../algorithms/kalman-local-linear-trend.md) | 1 | 173 | 209 |
| [KalmanMovingAverage](../algorithms/kalman-moving-average.md) | 1 | 174 | 177 |
| [KalmanPredictionBands](../algorithms/kalman-prediction-bands.md) | 1 | 176 | 211 |
| [KalmanRegressionChannel](../algorithms/kalman-regression-channel.md) | 2 | 188 | 222 |
| [KalmanTrendSignal](../algorithms/kalman-trend-signal.md) | 1 | 177 | 212 |
| [KalmanVelocityOscillator](../algorithms/kalman-velocity-oscillator.md) | 1 | 172 | 176 |
| [Kama](../algorithms/kama.md) | 1 | 31.1 | 35.5 |
| [KeltnerChannel](../algorithms/keltner-channel.md) | 3 | 60.9 | 90.5 |
| [KeltnerChannelOriginal](../algorithms/keltner-channel-original.md) | 3 | 53.9 | 88.7 |
| [KlingerVolumeOscillator](../algorithms/klinger-volume-oscillator.md) | 4 | 75.3 | 95.1 |
| [KSWIN](../algorithms/kswin.md) | 1 | 2586 | 2594 |
| [KyleLambda](../algorithms/kyle-lambda.md) | 2 | 60.1 | 62.4 |
| [LeadLagRegimeDetector](../algorithms/lead-lag-regime-detector.md) | 2 | 43.1 | 47.2 |
| [LiquidityDroughtDetector](../algorithms/liquidity-drought-detector.md) | 3 | 52.1 | 55.5 |
| [LiquidityRegimeDetector](../algorithms/liquidity-regime-detector.md) | 2 | 42.2 | 44.6 |
| [LinearRegression](../algorithms/linear-regression.md) | 1 | 51.1 | 54.3 |
| [LinearRegressionAngle](../algorithms/linear-regression-angle.md) | 1 | 51.5 | 54.9 |
| [LinearRegressionIntercept](../algorithms/linear-regression-intercept.md) | 1 | 51.0 | 54.5 |
| [LinearRegressionSlope](../algorithms/linear-regression-slope.md) | 1 | 50.9 | 54.3 |
| [Low](../algorithms/low.md) | 1 | 47.2 | 50.7 |
| [LowIndex](../algorithms/low-index.md) | 1 | 48.0 | 52.1 |
| [MACDFix](../algorithms/macd-fix.md) | 1 | 34.8 | 38.7 |
| [MassIndex](../algorithms/mass-index.md) | 2 | 46.3 | 48.8 |
| [MarketOpenCloseTransitionDetector](../algorithms/market-open-close-transition-detector.md) | 1 | 30.8 | 34.6 |
| [MatchedFlowConformalSignal](../algorithms/matched-flow-conformal-signal.md) | 5 | 1820 | 1858 |
| [MedianPrice](../algorithms/median-price.md) | 2 | 39.3 | 43.4 |
| [MesaAdaptiveMovingAverage](../algorithms/mesa-adaptive-moving-average.md) | 1 | 139 | 183 |
| [MicrostructureNoiseRegimeDetector](../algorithms/microstructure-noise-regime-detector.md) | 3 | 49.9 | 53.6 |
| [MidPoint](../algorithms/mid-point.md) | 1 | 66.5 | 68.7 |
| [MidPrice](../algorithms/mid-price.md) | 2 | 81.4 | 84.8 |
| [MinusDirectionalIndicator](../algorithms/minus-directional-indicator.md) | 3 | 70.4 | 72.0 |
| [MinusDirectionalMovement](../algorithms/minus-directional-movement.md) | 2 | 50.4 | 52.4 |
| [Momentum](../algorithms/momentum.md) | 1 | 34.1 | 39.2 |
| [MoneyFlowIndex](../algorithms/money-flow-index.md) | 4 | 76.9 | 78.7 |
| [NadarayaWatsonEnvelope](../algorithms/nadaraya-watson-envelope.md) | 1 | 209 | 247 |
| [NegativeVolumeIndex](../algorithms/negative-volume-index.md) | 2 | 47.0 | 49.4 |
| [NormalizedATR](../algorithms/normalized-atr.md) | 3 | 58.8 | 62.5 |
| [OnBalanceVolume](../algorithms/on-balance-volume.md) | 2 | 45.9 | 49.1 |
| [OnlineGaussianMixtureRegimeFilter](../algorithms/online-gaussian-mixture-regime-filter.md) | 1 | 133 | 136 |
| [OnlineHMMRegimeFilter](../algorithms/online-hmm-regime-filter.md) | 1 | 77.7 | 81.9 |
| [OnlineMarkovSwitchingVolatilityFilter](../algorithms/online-markov-switching-volatility-filter.md) | 1 | 83.2 | 87.1 |
| [OrderFlowImbalance](../algorithms/order-flow-imbalance.md) | 4 | 65.8 | 71.3 |
| [OrderFlowImbalanceRegimeDetector](../algorithms/order-flow-imbalance-regime-detector.md) | 4 | 64.0 | 68.4 |
| [PageHinkley](../algorithms/page-hinkley.md) | 1 | 35.1 | 38.5 |
| [PairsSpreadRegimeDetector](../algorithms/pairs-spread-regime-detector.md) | 2 | 45.6 | 49.5 |
| [ParticleFilterTrend](../algorithms/particle-filter-trend.md) | 1 | 5019 | 5064 |
| [ParabolicSAR](../algorithms/parabolic-sar.md) | 2 | 47.7 | 51.1 |
| [PercentagePrice](../algorithms/percentage-price.md) | 1 | 34.2 | 68.9 |
| [PercentageVolume](../algorithms/percentage-volume.md) | 1 | 35.2 | 69.3 |
| [PlusDirectionalIndicator](../algorithms/plus-directional-indicator.md) | 3 | 70.4 | 73.0 |
| [PlusDirectionalMovement](../algorithms/plus-directional-movement.md) | 2 | 49.5 | 51.7 |
| [PredictionErrorDriftDetector](../algorithms/prediction-error-drift-detector.md) | 2 | 45.2 | 47.8 |
| [QuoteMessageRateRegimeDetector](../algorithms/quote-message-rate-regime-detector.md) | 1 | 35.4 | 40.0 |
| [QuoteStuffingDetector](../algorithms/quote-stuffing-detector.md) | 2 | 42.9 | 45.9 |
| [RateOfChangePercentage](../algorithms/rate-of-change-percentage.md) | 1 | 32.1 | 36.5 |
| [RateOfChangeRatio](../algorithms/rate-of-change-ratio.md) | 1 | 32.0 | 36.7 |
| [RateOfChangeRatio100](../algorithms/rate-of-change-ratio-100.md) | 1 | 33.0 | 37.3 |
| [RenkoBrickGenerator](../algorithms/renko-brick-generator.md) | 1 | 34.4 | 68.0 |
| [ResidualDriftDetector](../algorithms/residual-drift-detector.md) | 1 | 34.8 | 38.2 |
| [RelativeVigorIndex](../algorithms/relative-vigor-index.md) | 4 | 109 | 133 |
| [RealizedVarianceRegimeDetector](../algorithms/realized-variance-regime-detector.md) | 1 | 33.6 | 37.4 |
| [RollingBetaShiftDetector](../algorithms/rolling-beta-shift-detector.md) | 2 | 70.9 | 74.0 |
| [RollingCorrelationShiftDetector](../algorithms/rolling-correlation-shift-detector.md) | 2 | 73.2 | 77.7 |
| [RollingMeanShiftDetector](../algorithms/rolling-mean-shift-detector.md) | 1 | 49.2 | 53.1 |
| [RollingMeanVarianceShiftDetector](../algorithms/rolling-mean-variance-shift-detector.md) | 1 | 61.3 | 64.9 |
| [RollingSpreadLiquidityShiftDetector](../algorithms/rolling-spread-liquidity-shift-detector.md) | 4 | 71.4 | 77.2 |
| [RollingVarianceShiftDetector](../algorithms/rolling-variance-shift-detector.md) | 1 | 54.5 | 55.1 |
| [SavitzkyGolayFilter](../algorithms/savitzky-golay-filter.md) | 1 | 86.4 | 123 |
| [SchaffTrendCycle](../algorithms/schaff-trend-cycle.md) | 1 | 84.6 | 87.9 |
| [SpreadFeatures](../algorithms/spread-features.md) | 3 | 68.8 | 94.9 |
| [SpreadExplosionDetector](../algorithms/spread-explosion-detector.md) | 2 | 41.6 | 45.4 |
| [SpreadRegimeDetector](../algorithms/spread-regime-detector.md) | 2 | 45.8 | 48.4 |
| [StdDev](../algorithms/std-dev.md) | 1 | 35.8 | 39.6 |
| [StickyHMMRegimeFilter](../algorithms/sticky-hmm-regime-filter.md) | 1 | 77.5 | 82.9 |
| [StochRSI](../algorithms/stoch-rsi.md) | 1 | 79.0 | 82.3 |
| [Stochastic](../algorithms/stochastic.md) | 3 | 93.6 | 119 |
| [SuperTrend](../algorithms/super-trend.md) | 3 | 65.6 | 100 |
| [Summation](../algorithms/summation.md) | 1 | 30.2 | 33.9 |
| [T3MovingAverage](../algorithms/t-3-moving-average.md) | 1 | 42.4 | 45.4 |
| [ThresholdRegimeDetector](../algorithms/threshold-regime-detector.md) | 1 | 29.6 | 33.3 |
| [TimeSeriesForecast](../algorithms/time-series-forecast.md) | 1 | 51.1 | 54.4 |
| [TradeIntensityRegimeDetector](../algorithms/trade-intensity-regime-detector.md) | 1 | 38.0 | 41.3 |
| [TrendChopRegimeDetector](../algorithms/trend-chop-regime-detector.md) | 3 | 68.2 | 71.8 |
| [TwoFactorKalmanTrendFilter](../algorithms/two-factor-kalman-trend-filter.md) | 1 | 174 | 209 |
| [TrueRange](../algorithms/true-range.md) | 3 | 58.2 | 61.5 |
| [TriangularMovingAverage](../algorithms/triangular-moving-average.md) | 1 | 34.5 | 38.6 |
| [TripleEMA](../algorithms/triple-ema.md) | 1 | 34.5 | 38.6 |
| [Trix](../algorithms/trix.md) | 1 | 36.3 | 40.7 |
| [TypicalPrice](../algorithms/typical-price.md) | 3 | 49.8 | 52.6 |
| [UltimateOscillator](../algorithms/ultimate-oscillator.md) | 3 | 83.2 | 88.5 |
| [UlcerIndex](../algorithms/ulcer-index.md) | 1 | 54.2 | 57.7 |
| [VPIN](../algorithms/vpin.md) | 2 | 91.4 | 94.9 |
| [Variance](../algorithms/variance.md) | 1 | 35.1 | 39.1 |
| [VariableIndexDynamicAverage](../algorithms/variable-index-dynamic-average.md) | 1 | 47.5 | 50.7 |
| [VolatilityBreakoutDetector](../algorithms/volatility-breakout-detector.md) | 1 | 35.0 | 38.6 |
| [VolatilityCompressionExpansionDetector](../algorithms/volatility-compression-expansion-detector.md) | 1 | 36.1 | 40.2 |
| [VolatilityRegimeDetector](../algorithms/volatility-regime-detector.md) | 1 | 33.5 | 36.7 |
| [VolumeProfile](../algorithms/volume-profile.md) | 2 | 432 | 466 |
| [VolumePriceTrend](../algorithms/volume-price-trend.md) | 2 | 40.4 | 44.3 |
| [VolumeRegimeDetector](../algorithms/volume-regime-detector.md) | 1 | 37.1 | 41.0 |
| [VolumeWeightedAveragePrice](../algorithms/volume-weighted-average-price.md) | 4 | 107 | 108 |
| [VolumeWeightedMovingAverage](../algorithms/volume-weighted-moving-average.md) | 2 | 43.8 | 46.9 |
| [Vortex](../algorithms/vortex.md) | 3 | 64.7 | 97.2 |
| [WeightedClosePrice](../algorithms/weighted-close-price.md) | 3 | 48.4 | 51.9 |
| [WeightedMovingAverage](../algorithms/weighted-moving-average.md) | 1 | 33.4 | 38.5 |
| [WilliamsR](../algorithms/williams-r.md) | 3 | 86.4 | 90.0 |
| [ZigZagSwingDetector](../algorithms/zig-zag-swing-detector.md) | 1 | 32.8 | 67.6 |
