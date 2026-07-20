# Intel Xeon 6975P-C Benchmarks

These results were collected on 2026-07-19. Public documentation identifies benchmark systems by CPU type rather than hostname.

## System

- CPU: **Intel Xeon 6975P-C**
- Runtime processor string: `Intel(R) Xeon(R) 6975P-C`
- Architecture: `x86_64`
- Platform: `Linux-7.0.0-1004-aws-x86_64-with-glibc2.43`
- Python: `3.14.4`
- NumPy: `2.5.1`
- RTTA: `0.2.3`
- Samples: `50000`
- Repeats: `5`
- Warmup repeats: `1`
- Seed: `42`

Run command:

```bash
python benchmarks/benchmark_readme.py --samples 50000 --repeat 5 --warmup 1 --output <benchmark-output.md>
```

## Latency Snapshot

- Benchmarked algorithms in registry: **247**.
- Algorithms shown: **247**.
- Median `advance(...)` latency, update state only: **39.2 ns/update**.
- Median `update(...)` latency, update state and return a value/result: **52.2 ns/update**.

| Algorithm | Inputs | update only: `advance(...)` ns/update | update + return: `update(...)` ns/update |
|---|---:|---:|---:|
| [AbsolutePriceOscillator](../algorithms/absolute-price-oscillator.md) | 1 | 23.7 | 28.4 |
| [AccelerationBands](../algorithms/acceleration-bands.md) | 3 | 48.3 | 73.4 |
| [AcceleratorOscillator](../algorithms/accelerator-oscillator.md) | 2 | 39.2 | 44.8 |
| [AccumulationDistribution](../algorithms/accumulation-distribution.md) | 4 | 53.1 | 59.3 |
| [AccumulativeSwingIndex](../algorithms/accumulative-swing-index.md) | 4 | 66.3 | 65.1 |
| [ADWIN](../algorithms/adwin.md) | 1 | 426 | 432 |
| [Alligator](../algorithms/alligator.md) | 2 | 39.2 | 72.1 |
| [AlphaBetaGammaTrackingFilter](../algorithms/alpha-beta-gamma-tracking-filter.md) | 1 | 24.7 | 56.2 |
| [AmihudIlliquidity](../algorithms/amihud-illiquidity.md) | 2 | 36.4 | 40.9 |
| [AnchoredVWAP](../algorithms/anchored-vwap.md) | 5 | 63.1 | 67.7 |
| [ArnaudLegouxMovingAverage](../algorithms/arnaud-legoux-moving-average.md) | 1 | 27.5 | 31.5 |
| [Aroon](../algorithms/aroon.md) | 2 | 59.3 | 84.1 |
| [AroonOscillator](../algorithms/aroon-oscillator.md) | 2 | 59.1 | 64.8 |
| [ATR](../algorithms/atr.md) | 3 | 46.6 | 51.5 |
| [ATRP](../algorithms/atrp.md) | 3 | 47.9 | 51.1 |
| [ATRRegimeDetector](../algorithms/atr-regime-detector.md) | 3 | 49.1 | 51.8 |
| [AuctionContinuousMarketTransitionDetector](../algorithms/auction-continuous-market-transition-detector.md) | 1 | 23.4 | 28.7 |
| [AverageDirectionalMovementIndex](../algorithms/average-directional-movement-index.md) | 3 | 56.0 | 60.5 |
| [AverageDirectionalMovementIndexRating](../algorithms/average-directional-movement-index-rating.md) | 3 | 58.3 | 61.1 |
| [AveragePrice](../algorithms/average-price.md) | 4 | 52.4 | 57.6 |
| [AwesomeOscillator](../algorithms/awesome-oscillator.md) | 2 | 38.9 | 43.5 |
| [BalanceOfPower](../algorithms/balance-of-power.md) | 4 | 52.2 | 58.9 |
| [Beta](../algorithms/beta.md) | 2 | 35.8 | 41.2 |
| [BetaRegimeDetector](../algorithms/beta-regime-detector.md) | 2 | 37.6 | 42.6 |
| [Bias](../algorithms/bias.md) | 1 | 23.9 | 29.2 |
| [BidAskBounceRegimeDetector](../algorithms/bid-ask-bounce-regime-detector.md) | 3 | 52.7 | 57.8 |
| [BollingerBands](../algorithms/bollinger-bands.md) | 1 | 24.0 | 59.5 |
| [BollingerBandwidth](../algorithms/bollinger-bandwidth.md) | 1 | 25.3 | 29.9 |
| [BollingerPercentB](../algorithms/bollinger-percent-b.md) | 1 | 25.1 | 29.7 |
| [BoundedBOCPD](../algorithms/bounded-bocpd.md) | 1 | 1130 | 1133 |
| [CalibrationDriftDetector](../algorithms/calibration-drift-detector.md) | 2 | 35.9 | 39.8 |
| [ChaikinMoneyFlow](../algorithms/chaikin-money-flow.md) | 4 | 55.8 | 60.8 |
| [ChaikinOscillator](../algorithms/chaikin-oscillator.md) | 4 | 56.6 | 59.7 |
| [ChaikinVolatility](../algorithms/chaikin-volatility.md) | 2 | 36.9 | 42.5 |
| [ChandelierExit](../algorithms/chandelier-exit.md) | 3 | 69.1 | 97.8 |
| [ChandeMomentumOscillator](../algorithms/chande-momentum-oscillator.md) | 1 | 30.0 | 35.0 |
| [ChoppinessIndex](../algorithms/choppiness-index.md) | 3 | 80.5 | 85.3 |
| [ClosePressureReversalSignal](../algorithms/close-pressure-reversal-signal.md) | 5 | 137 | 167 |
| [CointegrationBreakdownMonitor](../algorithms/cointegration-breakdown-monitor.md) | 2 | 38.5 | 42.9 |
| [CommodityChannelIndex](../algorithms/commodity-channel-index.md) | 3 | 52.1 | 56.1 |
| [ComparativeRelativeStrength](../algorithms/comparative-relative-strength.md) | 2 | 34.6 | 40.5 |
| [ConnorsRSI](../algorithms/connors-rsi.md) | 1 | 51.8 | 59.8 |
| [CoppockCurve](../algorithms/coppock-curve.md) | 1 | 27.1 | 31.3 |
| [Correlation](../algorithms/correlation.md) | 2 | 37.6 | 42.5 |
| [CorrelationRegimeDetector](../algorithms/correlation-regime-detector.md) | 2 | 39.4 | 41.9 |
| [CrossAssetCorrelationBreakDetector](../algorithms/cross-asset-correlation-break-detector.md) | 2 | 42.8 | 47.6 |
| [CumulativeReturn](../algorithms/cumulative-return.md) | 1 | 22.5 | 27.3 |
| [CUSUM](../algorithms/cusum.md) | 1 | 24.8 | 31.3 |
| [DailyLogReturn](../algorithms/daily-log-return.md) | 1 | 22.6 | 31.8 |
| [DailyReturn](../algorithms/daily-return.md) | 1 | 22.9 | 27.4 |
| [DDM](../algorithms/ddm.md) | 1 | 25.6 | 31.3 |
| [Delay](../algorithms/delay.md) | 1 | 22.2 | 27.9 |
| [DeMarker](../algorithms/de-marker.md) | 2 | 37.1 | 42.3 |
| [DetrendedPriceOscillator](../algorithms/detrended-price-oscillator.md) | 1 | 24.2 | 29.6 |
| [DirectionalMovementIndex](../algorithms/directional-movement-index.md) | 3 | 56.4 | 59.4 |
| [DonchianChannel](../algorithms/donchian-channel.md) | 3 | 69.8 | 99.4 |
| [DoubleEMA](../algorithms/double-ema.md) | 1 | 23.0 | 28.7 |
| [EaseOfMovement](../algorithms/ease-of-movement.md) | 3 | 48.9 | 70.4 |
| [EDDM](../algorithms/eddm.md) | 1 | 26.4 | 30.9 |
| [EfficiencyRatio](../algorithms/efficiency-ratio.md) | 1 | 23.5 | 29.3 |
| [EhlersCenterOfGravity](../algorithms/ehlers-center-of-gravity.md) | 1 | 29.5 | 62.5 |
| [EhlersCyberCycle](../algorithms/ehlers-cyber-cycle.md) | 1 | 25.2 | 56.6 |
| [EhlersDecycler](../algorithms/ehlers-decycler.md) | 1 | 23.5 | 55.6 |
| [EhlersInstantaneousTrendline](../algorithms/ehlers-instantaneous-trendline.md) | 1 | 23.0 | 56.1 |
| [EhlersOptimalTrackingFilter](../algorithms/ehlers-optimal-tracking-filter.md) | 2 | 36.8 | 40.8 |
| [EhlersRoofingFilter](../algorithms/ehlers-roofing-filter.md) | 1 | 23.6 | 57.1 |
| [EhlersSuperSmoother](../algorithms/ehlers-super-smoother.md) | 1 | 22.3 | 28.5 |
| [ElderRayIndex](../algorithms/elder-ray-index.md) | 3 | 45.7 | 68.7 |
| [EMA](../algorithms/ema.md) | 1 | 22.5 | 28.0 |
| [EWMA](../algorithms/ewma.md) | 1 | 22.0 | 27.6 |
| [EWMAZScoreShiftDetector](../algorithms/ewmaz-score-shift-detector.md) | 1 | 29.9 | 35.1 |
| [ExecutionCostSlippageRegimeDetector](../algorithms/execution-cost-slippage-regime-detector.md) | 3 | 47.3 | 52.2 |
| [FastStochastic](../algorithms/fast-stochastic.md) | 3 | 72.8 | 99.6 |
| [FeatureDistributionDriftDetector](../algorithms/feature-distribution-drift-detector.md) | 1 | 426 | 432 |
| [FibonacciRetracementLevels](../algorithms/fibonacci-retracement-levels.md) | 2 | 57.5 | 84.0 |
| [FisherTransform](../algorithms/fisher-transform.md) | 2 | 68.2 | 72.2 |
| [ForceIndex](../algorithms/force-index.md) | 2 | 35.9 | 40.4 |
| [FractalAdaptiveMovingAverage](../algorithms/fractal-adaptive-moving-average.md) | 1 | 67.3 | 71.9 |
| [GatorOscillator](../algorithms/gator-oscillator.md) | 2 | 40.2 | 66.8 |
| [GaussianProcessRegressionBands](../algorithms/gaussian-process-regression-bands.md) | 1 | 785 | 825 |
| [GeometricMovingAverage](../algorithms/geometric-moving-average.md) | 1 | 27.2 | 37.6 |
| [GuppyMultipleMovingAverage](../algorithms/guppy-multiple-moving-average.md) | 1 | 29.9 | 61.6 |
| [HDDM](../algorithms/hddm.md) | 1 | 35.7 | 40.1 |
| [HeikinAshiTransform](../algorithms/heikin-ashi-transform.md) | 4 | 54.6 | 80.0 |
| [HiddenSemiMarkovRegimeFilter](../algorithms/hidden-semi-markov-regime-filter.md) | 1 | 82.9 | 87.7 |
| [High](../algorithms/high.md) | 1 | 35.3 | 40.9 |
| [HighIndex](../algorithms/high-index.md) | 1 | 35.1 | 40.4 |
| [HighLow](../algorithms/high-low.md) | 1 | 42.7 | 74.1 |
| [HighLowIndex](../algorithms/high-low-index.md) | 1 | 42.3 | 74.5 |
| [HilbertDominantCyclePeriod](../algorithms/hilbert-dominant-cycle-period.md) | 1 | 426 | 428 |
| [HilbertDominantCyclePhase](../algorithms/hilbert-dominant-cycle-phase.md) | 1 | 426 | 427 |
| [HilbertPhasor](../algorithms/hilbert-phasor.md) | 1 | 426 | 457 |
| [HilbertSineWave](../algorithms/hilbert-sine-wave.md) | 1 | 425 | 458 |
| [HilbertTrendline](../algorithms/hilbert-trendline.md) | 1 | 426 | 428 |
| [HilbertTrendMode](../algorithms/hilbert-trend-mode.md) | 1 | 426 | 428 |
| [HistoricalVolatility](../algorithms/historical-volatility.md) | 1 | 28.6 | 36.0 |
| [HitRateDriftDetector](../algorithms/hit-rate-drift-detector.md) | 1 | 22.8 | 28.2 |
| [HullMovingAverage](../algorithms/hull-moving-average.md) | 1 | 28.8 | 31.9 |
| [Ichimoku](../algorithms/ichimoku.md) | 3 | 91.7 | 116 |
| [InteractingMultipleModelFilter](../algorithms/interacting-multiple-model-filter.md) | 1 | 167 | 195 |
| [IntradayClockEchoSignal](../algorithms/intraday-clock-echo-signal.md) | 5 | 184 | 212 |
| [IntradayIntensity](../algorithms/intraday-intensity.md) | 4 | 56.1 | 60.3 |
| [IntradayMomentumIndex](../algorithms/intraday-momentum-index.md) | 2 | 44.1 | 47.8 |
| [InverseFisherRSI](../algorithms/inverse-fisher-rsi.md) | 1 | 33.2 | 54.6 |
| [KagiChart](../algorithms/kagi-chart.md) | 1 | 28.5 | 60.1 |
| [KalmanExtremumTrend](../algorithms/kalman-extremum-trend.md) | 3 | 112 | 143 |
| [KalmanHedgeRatio](../algorithms/kalman-hedge-ratio.md) | 2 | 96.0 | 131 |
| [KalmanInnovationZScore](../algorithms/kalman-innovation-z-score.md) | 1 | 87.8 | 91.0 |
| [KalmanLocalLinearTrend](../algorithms/kalman-local-linear-trend.md) | 1 | 82.4 | 121 |
| [KalmanMovingAverage](../algorithms/kalman-moving-average.md) | 1 | 82.2 | 85.8 |
| [KalmanPredictionBands](../algorithms/kalman-prediction-bands.md) | 1 | 99.6 | 124 |
| [KalmanRegressionChannel](../algorithms/kalman-regression-channel.md) | 2 | 99.4 | 124 |
| [KalmanTrendSignal](../algorithms/kalman-trend-signal.md) | 1 | 89.3 | 118 |
| [KalmanVelocityOscillator](../algorithms/kalman-velocity-oscillator.md) | 1 | 81.8 | 85.9 |
| [Kama](../algorithms/kama.md) | 1 | 24.2 | 29.1 |
| [KeltnerChannel](../algorithms/keltner-channel.md) | 3 | 46.7 | 73.9 |
| [KeltnerChannelOriginal](../algorithms/keltner-channel-original.md) | 3 | 49.0 | 76.6 |
| [KlingerVolumeOscillator](../algorithms/klinger-volume-oscillator.md) | 4 | 64.3 | 87.3 |
| [KSTOscillator](../algorithms/kst-oscillator.md) | 1 | 32.6 | 72.3 |
| [KSWIN](../algorithms/kswin.md) | 1 | 2109 | 2114 |
| [KyleLambda](../algorithms/kyle-lambda.md) | 2 | 42.7 | 47.3 |
| [LeadLagRegimeDetector](../algorithms/lead-lag-regime-detector.md) | 2 | 37.8 | 42.4 |
| [LinearRegression](../algorithms/linear-regression.md) | 1 | 35.7 | 29.0 |
| [LinearRegressionAngle](../algorithms/linear-regression-angle.md) | 1 | 35.6 | 39.8 |
| [LinearRegressionIntercept](../algorithms/linear-regression-intercept.md) | 1 | 35.5 | 29.5 |
| [LinearRegressionSlope](../algorithms/linear-regression-slope.md) | 1 | 35.3 | 29.1 |
| [LiquidityDroughtDetector](../algorithms/liquidity-drought-detector.md) | 3 | 47.5 | 52.5 |
| [LiquidityRegimeDetector](../algorithms/liquidity-regime-detector.md) | 2 | 36.0 | 41.5 |
| [Low](../algorithms/low.md) | 1 | 34.8 | 39.9 |
| [LowIndex](../algorithms/low-index.md) | 1 | 34.3 | 40.5 |
| [MACD](../algorithms/macd.md) | 1 | 25.0 | 56.3 |
| [MACDExt](../algorithms/macd-ext.md) | 1 | 26.3 | 56.3 |
| [MACDFix](../algorithms/macd-fix.md) | 1 | 24.5 | 56.0 |
| [MarketFacilitationIndex](../algorithms/market-facilitation-index.md) | 3 | 44.2 | 50.0 |
| [MarketOpenCloseTransitionDetector](../algorithms/market-open-close-transition-detector.md) | 1 | 22.7 | 29.1 |
| [MassIndex](../algorithms/mass-index.md) | 2 | 36.8 | 42.6 |
| [MatchedFlowConformalSignal](../algorithms/matched-flow-conformal-signal.md) | 5 | 1177 | 1204 |
| [McGinleyDynamic](../algorithms/mc-ginley-dynamic.md) | 1 | 33.8 | 42.8 |
| [MedianPrice](../algorithms/median-price.md) | 2 | 33.9 | 39.5 |
| [MesaAdaptiveMovingAverage](../algorithms/mesa-adaptive-moving-average.md) | 1 | 76.0 | 114 |
| [MicrostructureNoiseRegimeDetector](../algorithms/microstructure-noise-regime-detector.md) | 3 | 47.4 | 51.7 |
| [MidPoint](../algorithms/mid-point.md) | 1 | 43.3 | 47.0 |
| [MidPrice](../algorithms/mid-price.md) | 2 | 59.5 | 65.3 |
| [MinusDirectionalIndicator](../algorithms/minus-directional-indicator.md) | 3 | 53.7 | 58.4 |
| [MinusDirectionalMovement](../algorithms/minus-directional-movement.md) | 2 | 41.9 | 47.5 |
| [Momentum](../algorithms/momentum.md) | 1 | 23.0 | 28.5 |
| [MoneyFlowIndex](../algorithms/money-flow-index.md) | 4 | 64.2 | 72.3 |
| [MovingAverageEnvelope](../algorithms/moving-average-envelope.md) | 1 | 25.0 | 57.8 |
| [NadarayaWatsonEnvelope](../algorithms/nadaraya-watson-envelope.md) | 1 | 72.0 | 108 |
| [NegativeVolumeIndex](../algorithms/negative-volume-index.md) | 2 | 40.4 | 43.4 |
| [NormalizedATR](../algorithms/normalized-atr.md) | 3 | 46.2 | 51.0 |
| [OnBalanceVolume](../algorithms/on-balance-volume.md) | 2 | 39.6 | 44.4 |
| [OnlineGaussianMixtureRegimeFilter](../algorithms/online-gaussian-mixture-regime-filter.md) | 1 | 133 | 138 |
| [OnlineHMMRegimeFilter](../algorithms/online-hmm-regime-filter.md) | 1 | 78.2 | 81.4 |
| [OnlineMarkovSwitchingVolatilityFilter](../algorithms/online-markov-switching-volatility-filter.md) | 1 | 82.5 | 85.4 |
| [OrderFlowImbalance](../algorithms/order-flow-imbalance.md) | 4 | 63.4 | 69.1 |
| [OrderFlowImbalanceRegimeDetector](../algorithms/order-flow-imbalance-regime-detector.md) | 4 | 62.2 | 67.4 |
| [PageHinkley](../algorithms/page-hinkley.md) | 1 | 27.9 | 32.4 |
| [PairsSpreadRegimeDetector](../algorithms/pairs-spread-regime-detector.md) | 2 | 37.6 | 42.5 |
| [ParabolicSAR](../algorithms/parabolic-sar.md) | 2 | 36.7 | 41.6 |
| [ParabolicSARExtended](../algorithms/parabolic-sar-extended.md) | 2 | 36.9 | 40.8 |
| [ParticleFilterTrend](../algorithms/particle-filter-trend.md) | 1 | 3528 | 3573 |
| [PercentagePrice](../algorithms/percentage-price.md) | 1 | 25.5 | 55.8 |
| [PercentageVolume](../algorithms/percentage-volume.md) | 1 | 24.6 | 56.3 |
| [PivotPoints](../algorithms/pivot-points.md) | 3 | 45.4 | 72.2 |
| [PlusDirectionalIndicator](../algorithms/plus-directional-indicator.md) | 3 | 58.1 | 58.9 |
| [PlusDirectionalMovement](../algorithms/plus-directional-movement.md) | 2 | 41.5 | 46.5 |
| [PointAndFigure](../algorithms/point-and-figure.md) | 1 | 24.8 | 60.2 |
| [PositiveVolumeIndex](../algorithms/positive-volume-index.md) | 2 | 41.7 | 44.2 |
| [PredictionErrorDriftDetector](../algorithms/prediction-error-drift-detector.md) | 2 | 37.2 | 42.1 |
| [PrettyGoodOscillator](../algorithms/pretty-good-oscillator.md) | 3 | 47.8 | 52.0 |
| [PsychologicalLine](../algorithms/psychological-line.md) | 1 | 29.0 | 34.4 |
| [QStick](../algorithms/q-stick.md) | 2 | 35.3 | 40.3 |
| [QuoteMessageRateRegimeDetector](../algorithms/quote-message-rate-regime-detector.md) | 1 | 28.4 | 32.6 |
| [QuoteStuffingDetector](../algorithms/quote-stuffing-detector.md) | 2 | 36.9 | 41.5 |
| [RandomWalkIndex](../algorithms/random-walk-index.md) | 3 | 69.4 | 94.7 |
| [RateOfChangePercentage](../algorithms/rate-of-change-percentage.md) | 1 | 22.9 | 28.8 |
| [RateOfChangeRatio](../algorithms/rate-of-change-ratio.md) | 1 | 22.3 | 27.7 |
| [RateOfChangeRatio100](../algorithms/rate-of-change-ratio-100.md) | 1 | 22.2 | 28.2 |
| [RealizedVarianceRegimeDetector](../algorithms/realized-variance-regime-detector.md) | 1 | 25.5 | 30.4 |
| [RelativeVigorIndex](../algorithms/relative-vigor-index.md) | 4 | 63.0 | 92.6 |
| [RelativeVolatilityIndex](../algorithms/relative-volatility-index.md) | 1 | 32.9 | 68.4 |
| [RenkoBrickGenerator](../algorithms/renko-brick-generator.md) | 1 | 24.9 | 54.1 |
| [ResidualDriftDetector](../algorithms/residual-drift-detector.md) | 1 | 24.8 | 28.7 |
| [ROC](../algorithms/roc.md) | 1 | 22.2 | 28.3 |
| [RollingBetaShiftDetector](../algorithms/rolling-beta-shift-detector.md) | 2 | 42.9 | 47.4 |
| [RollingCorrelationShiftDetector](../algorithms/rolling-correlation-shift-detector.md) | 2 | 45.3 | 49.7 |
| [RollingMeanShiftDetector](../algorithms/rolling-mean-shift-detector.md) | 1 | 34.3 | 43.1 |
| [RollingMeanVarianceShiftDetector](../algorithms/rolling-mean-variance-shift-detector.md) | 1 | 46.3 | 54.1 |
| [RollingMedian](../algorithms/rolling-median.md) | 1 | 193 | 199 |
| [RollingSpreadLiquidityShiftDetector](../algorithms/rolling-spread-liquidity-shift-detector.md) | 4 | 58.7 | 62.3 |
| [RollingVarianceShiftDetector](../algorithms/rolling-variance-shift-detector.md) | 1 | 37.0 | 45.8 |
| [RSI](../algorithms/rsi.md) | 1 | 29.6 | 34.9 |
| [SavitzkyGolayFilter](../algorithms/savitzky-golay-filter.md) | 1 | 38.4 | 67.8 |
| [SchaffTrendCycle](../algorithms/schaff-trend-cycle.md) | 1 | 47.7 | 53.5 |
| [SMA](../algorithms/sma.md) | 1 | 22.9 | 28.3 |
| [SmoothedMovingAverage](../algorithms/smoothed-moving-average.md) | 1 | 22.8 | 28.7 |
| [SpreadExplosionDetector](../algorithms/spread-explosion-detector.md) | 2 | 36.5 | 40.5 |
| [SpreadFeatures](../algorithms/spread-features.md) | 3 | 54.4 | 76.9 |
| [SpreadRegimeDetector](../algorithms/spread-regime-detector.md) | 2 | 38.4 | 43.3 |
| [SqueezeMomentum](../algorithms/squeeze-momentum.md) | 3 | 98.2 | 127 |
| [StdDev](../algorithms/std-dev.md) | 1 | 23.9 | 28.6 |
| [StickyHMMRegimeFilter](../algorithms/sticky-hmm-regime-filter.md) | 1 | 77.8 | 79.2 |
| [Stochastic](../algorithms/stochastic.md) | 3 | 74.4 | 101 |
| [StochasticMomentumIndex](../algorithms/stochastic-momentum-index.md) | 3 | 74.7 | 100 |
| [StochRSI](../algorithms/stoch-rsi.md) | 1 | 59.7 | 66.0 |
| [Summation](../algorithms/summation.md) | 1 | 23.4 | 28.4 |
| [SuperTrend](../algorithms/super-trend.md) | 3 | 55.4 | 86.4 |
| [SwingIndex](../algorithms/swing-index.md) | 4 | 69.5 | 68.7 |
| [T3MovingAverage](../algorithms/t-3-moving-average.md) | 1 | 26.6 | 31.5 |
| [ThresholdRegimeDetector](../algorithms/threshold-regime-detector.md) | 1 | 22.5 | 28.2 |
| [TimeSeriesForecast](../algorithms/time-series-forecast.md) | 1 | 35.5 | 29.0 |
| [TradeIntensityRegimeDetector](../algorithms/trade-intensity-regime-detector.md) | 1 | 29.4 | 34.8 |
| [TrendChopRegimeDetector](../algorithms/trend-chop-regime-detector.md) | 3 | 48.3 | 52.8 |
| [TrendIntensityIndex](../algorithms/trend-intensity-index.md) | 1 | 26.7 | 31.4 |
| [TriangularMovingAverage](../algorithms/triangular-moving-average.md) | 1 | 25.7 | 30.3 |
| [TripleEMA](../algorithms/triple-ema.md) | 1 | 24.1 | 30.0 |
| [Trix](../algorithms/trix.md) | 1 | 25.5 | 30.3 |
| [TrueRange](../algorithms/true-range.md) | 3 | 45.0 | 51.5 |
| [TSI](../algorithms/tsi.md) | 1 | 26.2 | 31.3 |
| [TwiggsMoneyFlow](../algorithms/twiggs-money-flow.md) | 4 | 55.7 | 61.5 |
| [TwoFactorKalmanTrendFilter](../algorithms/two-factor-kalman-trend-filter.md) | 1 | 90.0 | 123 |
| [TypicalPrice](../algorithms/typical-price.md) | 3 | 44.5 | 50.4 |
| [UlcerIndex](../algorithms/ulcer-index.md) | 1 | 36.4 | 41.1 |
| [UltimateOscillator](../algorithms/ultimate-oscillator.md) | 3 | 51.5 | 57.9 |
| [VariableIndexDynamicAverage](../algorithms/variable-index-dynamic-average.md) | 1 | 32.1 | 36.3 |
| [Variance](../algorithms/variance.md) | 1 | 22.5 | 28.9 |
| [VerticalHorizontalFilter](../algorithms/vertical-horizontal-filter.md) | 1 | 49.3 | 54.9 |
| [VolatilityBreakoutDetector](../algorithms/volatility-breakout-detector.md) | 1 | 23.7 | 29.3 |
| [VolatilityCompressionExpansionDetector](../algorithms/volatility-compression-expansion-detector.md) | 1 | 26.4 | 31.0 |
| [VolatilityRegimeDetector](../algorithms/volatility-regime-detector.md) | 1 | 23.9 | 29.5 |
| [VolumeOscillator](../algorithms/volume-oscillator.md) | 1 | 24.5 | 29.9 |
| [VolumePriceTrend](../algorithms/volume-price-trend.md) | 2 | 36.0 | 40.2 |
| [VolumeProfile](../algorithms/volume-profile.md) | 2 | 163 | 198 |
| [VolumeRegimeDetector](../algorithms/volume-regime-detector.md) | 1 | 30.0 | 34.0 |
| [VolumeWeightedAveragePrice](../algorithms/volume-weighted-average-price.md) | 4 | 56.9 | 60.3 |
| [VolumeWeightedMovingAverage](../algorithms/volume-weighted-moving-average.md) | 2 | 35.5 | 42.0 |
| [Vortex](../algorithms/vortex.md) | 3 | 47.8 | 74.8 |
| [VPIN](../algorithms/vpin.md) | 2 | 79.8 | 83.7 |
| [WaveTrend](../algorithms/wave-trend.md) | 3 | 51.7 | 82.5 |
| [WeightedClosePrice](../algorithms/weighted-close-price.md) | 3 | 44.9 | 50.4 |
| [WeightedMovingAverage](../algorithms/weighted-moving-average.md) | 1 | 23.1 | 28.5 |
| [WilliamsAD](../algorithms/williams-ad.md) | 3 | 50.6 | 54.6 |
| [WilliamsFractals](../algorithms/williams-fractals.md) | 2 | 47.8 | 72.3 |
| [WilliamsR](../algorithms/williams-r.md) | 3 | 69.9 | 73.7 |
| [ZeroLagEMA](../algorithms/zero-lag-ema.md) | 1 | 24.8 | 28.7 |
| [ZigZagSwingDetector](../algorithms/zig-zag-swing-detector.md) | 1 | 24.9 | 58.3 |

