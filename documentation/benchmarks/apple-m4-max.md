# Apple M4 Max Benchmarks

These results were collected on 2026-07-19. Public documentation identifies benchmark systems by CPU type rather than hostname.

## System

- CPU: **Apple M4 Max**
- Runtime processor string: `arm`
- Architecture: `arm64`
- Platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`
- Python: `3.14.5`
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
- Median `advance(...)` latency, update state only: **29.9 ns/update**.
- Median `update(...)` latency, update state and return a value/result: **41.1 ns/update**.

| Algorithm | Inputs | update only: `advance(...)` ns/update | update + return: `update(...)` ns/update |
|---|---:|---:|---:|
| [AbsolutePriceOscillator](../algorithms/absolute-price-oscillator.zh-CN.md) | 1 | 18.3 | 23.5 |
| [AccelerationBands](../algorithms/acceleration-bands.md) | 3 | 39.6 | 63.4 |
| [AcceleratorOscillator](../algorithms/accelerator-oscillator.md) | 2 | 33.6 | 41.7 |
| [AccumulationDistribution](../algorithms/accumulation-distribution.zh-CN.md) | 4 | 39.6 | 44.4 |
| [AccumulativeSwingIndex](../algorithms/accumulative-swing-index.md) | 4 | 43.4 | 44.6 |
| [ADWIN](../algorithms/adwin.zh-CN.md) | 1 | 259 | 262 |
| [Alligator](../algorithms/alligator.md) | 2 | 29.9 | 52.6 |
| [AlphaBetaGammaTrackingFilter](../algorithms/alpha-beta-gamma-tracking-filter.zh-CN.md) | 1 | 19.4 | 40.0 |
| [AmihudIlliquidity](../algorithms/amihud-illiquidity.zh-CN.md) | 2 | 29.6 | 32.2 |
| [AnchoredVWAP](../algorithms/anchored-vwap.zh-CN.md) | 5 | 49.5 | 51.5 |
| [ArnaudLegouxMovingAverage](../algorithms/arnaud-legoux-moving-average.md) | 1 | 17.7 | 24.8 |
| [Aroon](../algorithms/aroon.zh-CN.md) | 2 | 45.1 | 67.3 |
| [AroonOscillator](../algorithms/aroon-oscillator.zh-CN.md) | 2 | 44.7 | 49.9 |
| [ATR](../algorithms/atr.zh-CN.md) | 3 | 34.5 | 39.3 |
| [ATRP](../algorithms/atrp.zh-CN.md) | 3 | 35.5 | 42.0 |
| [ATRRegimeDetector](../algorithms/atr-regime-detector.zh-CN.md) | 3 | 38.0 | 42.4 |
| [AuctionContinuousMarketTransitionDetector](../algorithms/auction-continuous-market-transition-detector.zh-CN.md) | 1 | 18.4 | 23.2 |
| [AverageDirectionalMovementIndex](../algorithms/average-directional-movement-index.zh-CN.md) | 3 | 42.0 | 48.2 |
| [AverageDirectionalMovementIndexRating](../algorithms/average-directional-movement-index-rating.zh-CN.md) | 3 | 43.5 | 55.1 |
| [AveragePrice](../algorithms/average-price.zh-CN.md) | 4 | 39.4 | 42.9 |
| [AwesomeOscillator](../algorithms/awesome-oscillator.zh-CN.md) | 2 | 29.3 | 37.0 |
| [BalanceOfPower](../algorithms/balance-of-power.zh-CN.md) | 4 | 35.8 | 40.9 |
| [Beta](../algorithms/beta.zh-CN.md) | 2 | 27.1 | 36.3 |
| [BetaRegimeDetector](../algorithms/beta-regime-detector.zh-CN.md) | 2 | 28.0 | 40.3 |
| [Bias](../algorithms/bias.md) | 1 | 19.0 | 23.3 |
| [BidAskBounceRegimeDetector](../algorithms/bid-ask-bounce-regime-detector.zh-CN.md) | 3 | 36.2 | 40.4 |
| [BollingerBands](../algorithms/bollinger-bands.zh-CN.md) | 1 | 19.5 | 44.1 |
| [BollingerBandwidth](../algorithms/bollinger-bandwidth.md) | 1 | 17.4 | 24.3 |
| [BollingerPercentB](../algorithms/bollinger-percent-b.md) | 1 | 19.4 | 25.0 |
| [BoundedBOCPD](../algorithms/bounded-bocpd.zh-CN.md) | 1 | 639 | 648 |
| [CalibrationDriftDetector](../algorithms/calibration-drift-detector.zh-CN.md) | 2 | 28.8 | 32.8 |
| [ChaikinMoneyFlow](../algorithms/chaikin-money-flow.zh-CN.md) | 4 | 40.7 | 47.5 |
| [ChaikinOscillator](../algorithms/chaikin-oscillator.zh-CN.md) | 4 | 42.2 | 49.7 |
| [ChaikinVolatility](../algorithms/chaikin-volatility.md) | 2 | 28.5 | 34.2 |
| [ChandelierExit](../algorithms/chandelier-exit.md) | 3 | 54.8 | 76.3 |
| [ChandeMomentumOscillator](../algorithms/chande-momentum-oscillator.zh-CN.md) | 1 | 19.1 | 23.3 |
| [ChoppinessIndex](../algorithms/choppiness-index.zh-CN.md) | 3 | 54.1 | 68.3 |
| [ClosePressureReversalSignal](../algorithms/close-pressure-reversal-signal.zh-CN.md) | 5 | 89.3 | 117 |
| [CointegrationBreakdownMonitor](../algorithms/cointegration-breakdown-monitor.zh-CN.md) | 2 | 28.2 | 35.2 |
| [CommodityChannelIndex](../algorithms/commodity-channel-index.zh-CN.md) | 3 | 35.4 | 47.1 |
| [ComparativeRelativeStrength](../algorithms/comparative-relative-strength.md) | 2 | 26.2 | 31.7 |
| [ConnorsRSI](../algorithms/connors-rsi.zh-CN.md) | 1 | 32.8 | 38.4 |
| [CoppockCurve](../algorithms/coppock-curve.zh-CN.md) | 1 | 21.1 | 22.7 |
| [Correlation](../algorithms/correlation.zh-CN.md) | 2 | 28.4 | 38.9 |
| [CorrelationRegimeDetector](../algorithms/correlation-regime-detector.zh-CN.md) | 2 | 32.4 | 41.9 |
| [CrossAssetCorrelationBreakDetector](../algorithms/cross-asset-correlation-break-detector.zh-CN.md) | 2 | 35.1 | 45.9 |
| [CumulativeReturn](../algorithms/cumulative-return.zh-CN.md) | 1 | 18.4 | 21.3 |
| [CUSUM](../algorithms/cusum.zh-CN.md) | 1 | 19.7 | 24.5 |
| [DailyLogReturn](../algorithms/daily-log-return.zh-CN.md) | 1 | 17.9 | 26.5 |
| [DailyReturn](../algorithms/daily-return.zh-CN.md) | 1 | 17.9 | 22.3 |
| [DDM](../algorithms/ddm.zh-CN.md) | 1 | 18.7 | 22.9 |
| [Delay](../algorithms/delay.zh-CN.md) | 1 | 17.0 | 21.3 |
| [DeMarker](../algorithms/de-marker.md) | 2 | 30.0 | 32.7 |
| [DetrendedPriceOscillator](../algorithms/detrended-price-oscillator.zh-CN.md) | 1 | 20.0 | 24.4 |
| [DirectionalMovementIndex](../algorithms/directional-movement-index.zh-CN.md) | 3 | 38.8 | 47.7 |
| [DonchianChannel](../algorithms/donchian-channel.zh-CN.md) | 3 | 51.0 | 78.5 |
| [DoubleEMA](../algorithms/double-ema.zh-CN.md) | 1 | 19.3 | 24.8 |
| [EaseOfMovement](../algorithms/ease-of-movement.zh-CN.md) | 3 | 37.0 | 60.2 |
| [EDDM](../algorithms/eddm.zh-CN.md) | 1 | 19.7 | 24.6 |
| [EfficiencyRatio](../algorithms/efficiency-ratio.md) | 1 | 18.4 | 22.7 |
| [EhlersCenterOfGravity](../algorithms/ehlers-center-of-gravity.md) | 1 | 22.0 | 45.4 |
| [EhlersCyberCycle](../algorithms/ehlers-cyber-cycle.md) | 1 | 19.0 | 44.2 |
| [EhlersDecycler](../algorithms/ehlers-decycler.md) | 1 | 18.4 | 42.2 |
| [EhlersInstantaneousTrendline](../algorithms/ehlers-instantaneous-trendline.md) | 1 | 17.8 | 40.1 |
| [EhlersOptimalTrackingFilter](../algorithms/ehlers-optimal-tracking-filter.zh-CN.md) | 2 | 31.8 | 36.0 |
| [EhlersRoofingFilter](../algorithms/ehlers-roofing-filter.md) | 1 | 19.1 | 44.3 |
| [EhlersSuperSmoother](../algorithms/ehlers-super-smoother.md) | 1 | 18.4 | 23.5 |
| [ElderRayIndex](../algorithms/elder-ray-index.zh-CN.md) | 3 | 35.3 | 60.7 |
| [EMA](../algorithms/ema.zh-CN.md) | 1 | 18.3 | 22.9 |
| [EWMA](../algorithms/ewma.zh-CN.md) | 1 | 17.4 | 23.2 |
| [EWMAZScoreShiftDetector](../algorithms/ewmaz-score-shift-detector.zh-CN.md) | 1 | 22.0 | 26.7 |
| [ExecutionCostSlippageRegimeDetector](../algorithms/execution-cost-slippage-regime-detector.zh-CN.md) | 3 | 36.2 | 38.1 |
| [FastStochastic](../algorithms/fast-stochastic.zh-CN.md) | 3 | 57.9 | 80.9 |
| [FeatureDistributionDriftDetector](../algorithms/feature-distribution-drift-detector.zh-CN.md) | 1 | 262 | 263 |
| [FibonacciRetracementLevels](../algorithms/fibonacci-retracement-levels.zh-CN.md) | 2 | 45.4 | 74.7 |
| [FisherTransform](../algorithms/fisher-transform.zh-CN.md) | 2 | 59.7 | 65.2 |
| [ForceIndex](../algorithms/force-index.zh-CN.md) | 2 | 26.0 | 32.6 |
| [FractalAdaptiveMovingAverage](../algorithms/fractal-adaptive-moving-average.zh-CN.md) | 1 | 43.9 | 50.3 |
| [GatorOscillator](../algorithms/gator-oscillator.md) | 2 | 31.0 | 54.6 |
| [GaussianProcessRegressionBands](../algorithms/gaussian-process-regression-bands.zh-CN.md) | 1 | 462 | 487 |
| [GeometricMovingAverage](../algorithms/geometric-moving-average.md) | 1 | 21.3 | 29.8 |
| [GuppyMultipleMovingAverage](../algorithms/guppy-multiple-moving-average.md) | 1 | 27.2 | 54.7 |
| [HDDM](../algorithms/hddm.zh-CN.md) | 1 | 23.1 | 28.6 |
| [HeikinAshiTransform](../algorithms/heikin-ashi-transform.zh-CN.md) | 4 | 40.2 | 65.0 |
| [HiddenSemiMarkovRegimeFilter](../algorithms/hidden-semi-markov-regime-filter.zh-CN.md) | 1 | 56.5 | 61.3 |
| [High](../algorithms/high.zh-CN.md) | 1 | 27.2 | 32.9 |
| [HighIndex](../algorithms/high-index.zh-CN.md) | 1 | 28.0 | 31.7 |
| [HighLow](../algorithms/high-low.zh-CN.md) | 1 | 34.8 | 55.1 |
| [HighLowIndex](../algorithms/high-low-index.zh-CN.md) | 1 | 33.8 | 58.0 |
| [HilbertDominantCyclePeriod](../algorithms/hilbert-dominant-cycle-period.md) | 1 | 219 | 228 |
| [HilbertDominantCyclePhase](../algorithms/hilbert-dominant-cycle-phase.md) | 1 | 217 | 225 |
| [HilbertPhasor](../algorithms/hilbert-phasor.md) | 1 | 217 | 243 |
| [HilbertSineWave](../algorithms/hilbert-sine-wave.md) | 1 | 217 | 241 |
| [HilbertTrendline](../algorithms/hilbert-trendline.md) | 1 | 218 | 219 |
| [HilbertTrendMode](../algorithms/hilbert-trend-mode.md) | 1 | 230 | 229 |
| [HistoricalVolatility](../algorithms/historical-volatility.md) | 1 | 22.3 | 26.6 |
| [HitRateDriftDetector](../algorithms/hit-rate-drift-detector.zh-CN.md) | 1 | 18.9 | 23.5 |
| [HullMovingAverage](../algorithms/hull-moving-average.zh-CN.md) | 1 | 20.1 | 24.2 |
| [Ichimoku](../algorithms/ichimoku.zh-CN.md) | 3 | 74.0 | 94.8 |
| [InteractingMultipleModelFilter](../algorithms/interacting-multiple-model-filter.zh-CN.md) | 1 | 101 | 124 |
| [IntradayClockEchoSignal](../algorithms/intraday-clock-echo-signal.zh-CN.md) | 5 | 124 | 145 |
| [IntradayIntensity](../algorithms/intraday-intensity.md) | 4 | 41.8 | 43.3 |
| [IntradayMomentumIndex](../algorithms/intraday-momentum-index.md) | 2 | 28.1 | 34.1 |
| [InverseFisherRSI](../algorithms/inverse-fisher-rsi.md) | 1 | 29.9 | 40.2 |
| [KagiChart](../algorithms/kagi-chart.md) | 1 | 25.0 | 51.5 |
| [KalmanExtremumTrend](../algorithms/kalman-extremum-trend.zh-CN.md) | 3 | 67.3 | 99.8 |
| [KalmanHedgeRatio](../algorithms/kalman-hedge-ratio.zh-CN.md) | 2 | 50.3 | 69.9 |
| [KalmanInnovationZScore](../algorithms/kalman-innovation-z-score.zh-CN.md) | 1 | 35.3 | 53.7 |
| [KalmanLocalLinearTrend](../algorithms/kalman-local-linear-trend.zh-CN.md) | 1 | 36.3 | 63.3 |
| [KalmanMovingAverage](../algorithms/kalman-moving-average.zh-CN.md) | 1 | 39.1 | 53.6 |
| [KalmanPredictionBands](../algorithms/kalman-prediction-bands.zh-CN.md) | 1 | 40.4 | 61.5 |
| [KalmanRegressionChannel](../algorithms/kalman-regression-channel.zh-CN.md) | 2 | 57.7 | 76.0 |
| [KalmanTrendSignal](../algorithms/kalman-trend-signal.zh-CN.md) | 1 | 39.1 | 64.2 |
| [KalmanVelocityOscillator](../algorithms/kalman-velocity-oscillator.zh-CN.md) | 1 | 37.3 | 53.7 |
| [Kama](../algorithms/kama.zh-CN.md) | 1 | 21.4 | 26.1 |
| [KeltnerChannel](../algorithms/keltner-channel.zh-CN.md) | 3 | 39.4 | 61.8 |
| [KeltnerChannelOriginal](../algorithms/keltner-channel-original.zh-CN.md) | 3 | 41.0 | 64.8 |
| [KlingerVolumeOscillator](../algorithms/klinger-volume-oscillator.zh-CN.md) | 4 | 49.9 | 75.7 |
| [KSTOscillator](../algorithms/kst-oscillator.zh-CN.md) | 1 | 28.7 | 55.9 |
| [KSWIN](../algorithms/kswin.zh-CN.md) | 1 | 1306 | 1294 |
| [KyleLambda](../algorithms/kyle-lambda.zh-CN.md) | 2 | 30.4 | 36.0 |
| [LeadLagRegimeDetector](../algorithms/lead-lag-regime-detector.zh-CN.md) | 2 | 30.4 | 36.6 |
| [LinearRegression](../algorithms/linear-regression.zh-CN.md) | 1 | 27.3 | 33.6 |
| [LinearRegressionAngle](../algorithms/linear-regression-angle.zh-CN.md) | 1 | 26.8 | 33.3 |
| [LinearRegressionIntercept](../algorithms/linear-regression-intercept.zh-CN.md) | 1 | 24.3 | 31.1 |
| [LinearRegressionSlope](../algorithms/linear-regression-slope.zh-CN.md) | 1 | 25.9 | 33.5 |
| [LiquidityDroughtDetector](../algorithms/liquidity-drought-detector.zh-CN.md) | 3 | 39.2 | 41.6 |
| [LiquidityRegimeDetector](../algorithms/liquidity-regime-detector.zh-CN.md) | 2 | 27.2 | 35.7 |
| [Low](../algorithms/low.zh-CN.md) | 1 | 27.2 | 35.9 |
| [LowIndex](../algorithms/low-index.zh-CN.md) | 1 | 30.4 | 33.1 |
| [MACD](../algorithms/macd.zh-CN.md) | 1 | 20.2 | 47.1 |
| [MACDExt](../algorithms/macd-ext.md) | 1 | 19.9 | 45.1 |
| [MACDFix](../algorithms/macd-fix.zh-CN.md) | 1 | 22.7 | 48.5 |
| [MarketFacilitationIndex](../algorithms/market-facilitation-index.md) | 3 | 35.9 | 41.1 |
| [MarketOpenCloseTransitionDetector](../algorithms/market-open-close-transition-detector.zh-CN.md) | 1 | 20.0 | 22.8 |
| [MassIndex](../algorithms/mass-index.zh-CN.md) | 2 | 29.4 | 37.3 |
| [MatchedFlowConformalSignal](../algorithms/matched-flow-conformal-signal.zh-CN.md) | 5 | 935 | 938 |
| [McGinleyDynamic](../algorithms/mc-ginley-dynamic.md) | 1 | 28.6 | 31.9 |
| [MedianPrice](../algorithms/median-price.zh-CN.md) | 2 | 27.8 | 34.7 |
| [MesaAdaptiveMovingAverage](../algorithms/mesa-adaptive-moving-average.zh-CN.md) | 1 | 54.1 | 80.6 |
| [MicrostructureNoiseRegimeDetector](../algorithms/microstructure-noise-regime-detector.zh-CN.md) | 3 | 36.9 | 39.6 |
| [MidPoint](../algorithms/mid-point.zh-CN.md) | 1 | 34.8 | 40.0 |
| [MidPrice](../algorithms/mid-price.zh-CN.md) | 2 | 47.3 | 53.2 |
| [MinusDirectionalIndicator](../algorithms/minus-directional-indicator.zh-CN.md) | 3 | 40.1 | 47.3 |
| [MinusDirectionalMovement](../algorithms/minus-directional-movement.zh-CN.md) | 2 | 28.3 | 31.5 |
| [Momentum](../algorithms/momentum.zh-CN.md) | 1 | 18.7 | 24.2 |
| [MoneyFlowIndex](../algorithms/money-flow-index.zh-CN.md) | 4 | 46.6 | 54.3 |
| [MovingAverageEnvelope](../algorithms/moving-average-envelope.md) | 1 | 19.6 | 41.8 |
| [NadarayaWatsonEnvelope](../algorithms/nadaraya-watson-envelope.zh-CN.md) | 1 | 44.9 | 77.6 |
| [NegativeVolumeIndex](../algorithms/negative-volume-index.zh-CN.md) | 2 | 28.7 | 34.3 |
| [NormalizedATR](../algorithms/normalized-atr.zh-CN.md) | 3 | 33.1 | 38.3 |
| [OnBalanceVolume](../algorithms/on-balance-volume.zh-CN.md) | 2 | 31.8 | 36.5 |
| [OnlineGaussianMixtureRegimeFilter](../algorithms/online-gaussian-mixture-regime-filter.zh-CN.md) | 1 | 41.6 | 52.4 |
| [OnlineHMMRegimeFilter](../algorithms/online-hmm-regime-filter.zh-CN.md) | 1 | 57.2 | 63.2 |
| [OnlineMarkovSwitchingVolatilityFilter](../algorithms/online-markov-switching-volatility-filter.zh-CN.md) | 1 | 58.9 | 65.1 |
| [OrderFlowImbalance](../algorithms/order-flow-imbalance.zh-CN.md) | 4 | 42.6 | 48.2 |
| [OrderFlowImbalanceRegimeDetector](../algorithms/order-flow-imbalance-regime-detector.zh-CN.md) | 4 | 48.2 | 51.5 |
| [PageHinkley](../algorithms/page-hinkley.zh-CN.md) | 1 | 22.3 | 26.9 |
| [PairsSpreadRegimeDetector](../algorithms/pairs-spread-regime-detector.zh-CN.md) | 2 | 29.9 | 35.0 |
| [ParabolicSAR](../algorithms/parabolic-sar.zh-CN.md) | 2 | 28.8 | 31.1 |
| [ParabolicSARExtended](../algorithms/parabolic-sar-extended.md) | 2 | 26.8 | 31.3 |
| [ParticleFilterTrend](../algorithms/particle-filter-trend.zh-CN.md) | 1 | 2056 | 2086 |
| [PercentagePrice](../algorithms/percentage-price.zh-CN.md) | 1 | 20.8 | 44.0 |
| [PercentageVolume](../algorithms/percentage-volume.zh-CN.md) | 1 | 20.8 | 47.1 |
| [PivotPoints](../algorithms/pivot-points.md) | 3 | 34.5 | 61.8 |
| [PlusDirectionalIndicator](../algorithms/plus-directional-indicator.zh-CN.md) | 3 | 40.7 | 45.3 |
| [PlusDirectionalMovement](../algorithms/plus-directional-movement.zh-CN.md) | 2 | 26.6 | 31.4 |
| [PointAndFigure](../algorithms/point-and-figure.md) | 1 | 19.3 | 45.3 |
| [PositiveVolumeIndex](../algorithms/positive-volume-index.md) | 2 | 29.2 | 34.8 |
| [PredictionErrorDriftDetector](../algorithms/prediction-error-drift-detector.zh-CN.md) | 2 | 29.3 | 33.8 |
| [PrettyGoodOscillator](../algorithms/pretty-good-oscillator.md) | 3 | 37.7 | 43.7 |
| [PsychologicalLine](../algorithms/psychological-line.md) | 1 | 19.6 | 28.4 |
| [QStick](../algorithms/q-stick.md) | 2 | 27.9 | 33.4 |
| [QuoteMessageRateRegimeDetector](../algorithms/quote-message-rate-regime-detector.zh-CN.md) | 1 | 20.8 | 25.3 |
| [QuoteStuffingDetector](../algorithms/quote-stuffing-detector.zh-CN.md) | 2 | 29.0 | 34.5 |
| [RandomWalkIndex](../algorithms/random-walk-index.md) | 3 | 51.2 | 79.6 |
| [RateOfChangePercentage](../algorithms/rate-of-change-percentage.zh-CN.md) | 1 | 18.0 | 22.7 |
| [RateOfChangeRatio](../algorithms/rate-of-change-ratio.zh-CN.md) | 1 | 17.7 | 23.9 |
| [RateOfChangeRatio100](../algorithms/rate-of-change-ratio-100.zh-CN.md) | 1 | 18.2 | 22.1 |
| [RealizedVarianceRegimeDetector](../algorithms/realized-variance-regime-detector.zh-CN.md) | 1 | 19.9 | 23.7 |
| [RelativeVigorIndex](../algorithms/relative-vigor-index.zh-CN.md) | 4 | 46.5 | 74.9 |
| [RelativeVolatilityIndex](../algorithms/relative-volatility-index.md) | 1 | 27.1 | 52.8 |
| [RenkoBrickGenerator](../algorithms/renko-brick-generator.zh-CN.md) | 1 | 19.3 | 46.0 |
| [ResidualDriftDetector](../algorithms/residual-drift-detector.zh-CN.md) | 1 | 19.4 | 24.5 |
| [ROC](../algorithms/roc.zh-CN.md) | 1 | 20.3 | 24.3 |
| [RollingBetaShiftDetector](../algorithms/rolling-beta-shift-detector.zh-CN.md) | 2 | 32.4 | 41.2 |
| [RollingCorrelationShiftDetector](../algorithms/rolling-correlation-shift-detector.zh-CN.md) | 2 | 38.9 | 44.9 |
| [RollingMeanShiftDetector](../algorithms/rolling-mean-shift-detector.zh-CN.md) | 1 | 26.1 | 29.5 |
| [RollingMeanVarianceShiftDetector](../algorithms/rolling-mean-variance-shift-detector.zh-CN.md) | 1 | 34.7 | 41.2 |
| [RollingMedian](../algorithms/rolling-median.md) | 1 | 133 | 127 |
| [RollingSpreadLiquidityShiftDetector](../algorithms/rolling-spread-liquidity-shift-detector.zh-CN.md) | 4 | 42.3 | 55.3 |
| [RollingVarianceShiftDetector](../algorithms/rolling-variance-shift-detector.zh-CN.md) | 1 | 27.7 | 33.9 |
| [RSI](../algorithms/rsi.zh-CN.md) | 1 | 22.4 | 29.2 |
| [SavitzkyGolayFilter](../algorithms/savitzky-golay-filter.zh-CN.md) | 1 | 26.1 | 50.7 |
| [SchaffTrendCycle](../algorithms/schaff-trend-cycle.zh-CN.md) | 1 | 40.8 | 45.8 |
| [SMA](../algorithms/sma.zh-CN.md) | 1 | 19.8 | 23.3 |
| [SmoothedMovingAverage](../algorithms/smoothed-moving-average.md) | 1 | 19.4 | 24.3 |
| [SpreadExplosionDetector](../algorithms/spread-explosion-detector.zh-CN.md) | 2 | 26.6 | 33.0 |
| [SpreadFeatures](../algorithms/spread-features.zh-CN.md) | 3 | 42.5 | 67.5 |
| [SpreadRegimeDetector](../algorithms/spread-regime-detector.zh-CN.md) | 2 | 29.3 | 33.8 |
| [SqueezeMomentum](../algorithms/squeeze-momentum.md) | 3 | 75.3 | 99.6 |
| [StdDev](../algorithms/std-dev.zh-CN.md) | 1 | 19.3 | 23.9 |
| [StickyHMMRegimeFilter](../algorithms/sticky-hmm-regime-filter.zh-CN.md) | 1 | 58.6 | 63.9 |
| [Stochastic](../algorithms/stochastic.zh-CN.md) | 3 | 62.3 | 86.0 |
| [StochasticMomentumIndex](../algorithms/stochastic-momentum-index.md) | 3 | 58.1 | 82.1 |
| [StochRSI](../algorithms/stoch-rsi.zh-CN.md) | 1 | 47.5 | 53.5 |
| [Summation](../algorithms/summation.zh-CN.md) | 1 | 19.0 | 24.1 |
| [SuperTrend](../algorithms/super-trend.zh-CN.md) | 3 | 43.6 | 69.3 |
| [SwingIndex](../algorithms/swing-index.md) | 4 | 40.3 | 47.2 |
| [T3MovingAverage](../algorithms/t-3-moving-average.zh-CN.md) | 1 | 25.9 | 27.9 |
| [ThresholdRegimeDetector](../algorithms/threshold-regime-detector.zh-CN.md) | 1 | 19.1 | 23.9 |
| [TimeSeriesForecast](../algorithms/time-series-forecast.zh-CN.md) | 1 | 26.4 | 31.9 |
| [TradeIntensityRegimeDetector](../algorithms/trade-intensity-regime-detector.zh-CN.md) | 1 | 23.8 | 28.1 |
| [TrendChopRegimeDetector](../algorithms/trend-chop-regime-detector.zh-CN.md) | 3 | 37.6 | 43.4 |
| [TrendIntensityIndex](../algorithms/trend-intensity-index.md) | 1 | 21.0 | 25.7 |
| [TriangularMovingAverage](../algorithms/triangular-moving-average.zh-CN.md) | 1 | 21.9 | 25.8 |
| [TripleEMA](../algorithms/triple-ema.zh-CN.md) | 1 | 22.3 | 26.8 |
| [Trix](../algorithms/trix.zh-CN.md) | 1 | 21.6 | 26.7 |
| [TrueRange](../algorithms/true-range.zh-CN.md) | 3 | 33.3 | 38.5 |
| [TSI](../algorithms/tsi.zh-CN.md) | 1 | 22.7 | 26.8 |
| [TwiggsMoneyFlow](../algorithms/twiggs-money-flow.md) | 4 | 43.1 | 47.9 |
| [TwoFactorKalmanTrendFilter](../algorithms/two-factor-kalman-trend-filter.zh-CN.md) | 1 | 38.4 | 62.5 |
| [TypicalPrice](../algorithms/typical-price.zh-CN.md) | 3 | 35.4 | 40.2 |
| [UlcerIndex](../algorithms/ulcer-index.zh-CN.md) | 1 | 29.6 | 35.0 |
| [UltimateOscillator](../algorithms/ultimate-oscillator.zh-CN.md) | 3 | 37.9 | 49.2 |
| [VariableIndexDynamicAverage](../algorithms/variable-index-dynamic-average.zh-CN.md) | 1 | 19.9 | 24.8 |
| [Variance](../algorithms/variance.zh-CN.md) | 1 | 19.7 | 23.9 |
| [VerticalHorizontalFilter](../algorithms/vertical-horizontal-filter.md) | 1 | 32.4 | 37.3 |
| [VolatilityBreakoutDetector](../algorithms/volatility-breakout-detector.zh-CN.md) | 1 | 20.0 | 24.3 |
| [VolatilityCompressionExpansionDetector](../algorithms/volatility-compression-expansion-detector.zh-CN.md) | 1 | 21.2 | 25.3 |
| [VolatilityRegimeDetector](../algorithms/volatility-regime-detector.zh-CN.md) | 1 | 20.2 | 23.3 |
| [VolumeOscillator](../algorithms/volume-oscillator.md) | 1 | 19.3 | 24.4 |
| [VolumePriceTrend](../algorithms/volume-price-trend.zh-CN.md) | 2 | 26.8 | 32.2 |
| [VolumeProfile](../algorithms/volume-profile.zh-CN.md) | 2 | 190 | 213 |
| [VolumeRegimeDetector](../algorithms/volume-regime-detector.zh-CN.md) | 1 | 23.4 | 27.9 |
| [VolumeWeightedAveragePrice](../algorithms/volume-weighted-average-price.zh-CN.md) | 4 | 40.6 | 44.5 |
| [VolumeWeightedMovingAverage](../algorithms/volume-weighted-moving-average.zh-CN.md) | 2 | 27.5 | 33.0 |
| [Vortex](../algorithms/vortex.zh-CN.md) | 3 | 38.4 | 58.4 |
| [VPIN](../algorithms/vpin.zh-CN.md) | 2 | 62.0 | 68.6 |
| [WaveTrend](../algorithms/wave-trend.md) | 3 | 41.1 | 65.8 |
| [WeightedClosePrice](../algorithms/weighted-close-price.zh-CN.md) | 3 | 33.7 | 35.9 |
| [WeightedMovingAverage](../algorithms/weighted-moving-average.zh-CN.md) | 1 | 19.0 | 22.9 |
| [WilliamsAD](../algorithms/williams-ad.md) | 3 | 37.1 | 40.8 |
| [WilliamsFractals](../algorithms/williams-fractals.md) | 2 | 29.1 | 55.4 |
| [WilliamsR](../algorithms/williams-r.zh-CN.md) | 3 | 53.1 | 59.2 |
| [ZeroLagEMA](../algorithms/zero-lag-ema.md) | 1 | 18.7 | 22.9 |
| [ZigZagSwingDetector](../algorithms/zig-zag-swing-detector.zh-CN.md) | 1 | 20.3 | 44.0 |

