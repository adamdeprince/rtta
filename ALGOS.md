# Algorithms

This file lists the public indicator algorithms exported by `rtta`. Tuning/result helper types are intentionally omitted. Detailed implementation notes and external references live in per-algorithm pages under `documentation/algorithms/`.

| Algorithm | Description |
|---|---|
| [`ATR`](documentation/algorithms/atr.md) | Average True Range volatility over a rolling window. |
| [`ATRP`](documentation/algorithms/atrp.md) | Average True Range expressed as a percentage of price. |
| [`ATRRegimeDetector`](documentation/algorithms/atr-regime-detector.md) | Stateful ATR regime detector with high/low hysteresis bands. |
| [`ADWIN`](documentation/algorithms/adwin.md) | Adaptive-window mean drift detector with bounded history and directed shift output. |
| [`EMA`](documentation/algorithms/ema.md) | Exponential moving average with more weight on recent samples. |
| [`EWMA`](documentation/algorithms/ewma.md) | Exponentially weighted moving average parameterized by alpha/span/com. |
| [`EWMAZScoreShiftDetector`](documentation/algorithms/ewmaz-score-shift-detector.md) | Causal EWMA mean/variance z-score event detector for threshold-sized shifts. |
| [`MACD`](documentation/algorithms/macd.md) | Moving Average Convergence/Divergence oscillator and signal/histogram. |
| [`ROC`](documentation/algorithms/roc.md) | Rate of Change momentum as percentage change over a lookback. |
| [`RSI`](documentation/algorithms/rsi.md) | Relative Strength Index momentum oscillator. |
| [`SMA`](documentation/algorithms/sma.md) | Simple moving average over a rolling window. |
| [`TSI`](documentation/algorithms/tsi.md) | True Strength Index double-smoothed momentum oscillator. |
| [`AbsolutePriceOscillator`](documentation/algorithms/absolute-price-oscillator.md) | Difference between fast and slow moving averages in price units. |
| [`AccumulationDistribution`](documentation/algorithms/accumulation-distribution.md) | Volume-price accumulation/distribution line. |
| [`AlphaBetaGammaTrackingFilter`](documentation/algorithms/alpha-beta-gamma-tracking-filter.md) | Steady-state Kalman-like price, velocity, and acceleration tracker. |
| [`AmihudIlliquidity`](documentation/algorithms/amihud-illiquidity.md) | Rolling average absolute return per dollar of traded volume. |
| [`AnchoredVWAP`](documentation/algorithms/anchored-vwap.md) | VWAP accumulated from arbitrary anchor/reset events rather than a fixed session or rolling window. |
| [`Aroon`](documentation/algorithms/aroon.md) | Aroon Up/Down trend age indicators based on recent highs and lows. |
| [`AroonOscillator`](documentation/algorithms/aroon-oscillator.md) | Difference between Aroon Up and Aroon Down. |
| [`AverageDirectionalMovementIndex`](documentation/algorithms/average-directional-movement-index.md) | ADX trend-strength indicator. |
| [`AverageDirectionalMovementIndexRating`](documentation/algorithms/average-directional-movement-index-rating.md) | ADXR smoothed ADX trend-strength rating. |
| [`AveragePrice`](documentation/algorithms/average-price.md) | Average of open, high, low, and close. |
| [`AuctionContinuousMarketTransitionDetector`](documentation/algorithms/auction-continuous-market-transition-detector.md) | Hysteresis detector for auction-versus-continuous market phase signals. |
| [`AwesomeOscillator`](documentation/algorithms/awesome-oscillator.md) | Difference between short and long median-price moving averages. |
| [`BalanceOfPower`](documentation/algorithms/balance-of-power.md) | Open/high/low/close buying-versus-selling pressure measure. |
| [`Beta`](documentation/algorithms/beta.md) | Rolling beta of one series against another. |
| [`BetaRegimeDetector`](documentation/algorithms/beta-regime-detector.md) | Stateful rolling beta regime detector with upper/lower hysteresis bands. |
| [`BidAskBounceRegimeDetector`](documentation/algorithms/bid-ask-bounce-regime-detector.md) | EWMA bid/ask side alternation detector for quote-bounce regimes. |
| [`BollingerBands`](documentation/algorithms/bollinger-bands.md) | Moving-average envelope based on standard deviations. |
| [`BoundedBOCPD`](documentation/algorithms/bounded-bocpd.md) | Bounded-memory Bayesian online change-point detector with constant hazard. |
| [`CalibrationDriftDetector`](documentation/algorithms/calibration-drift-detector.md) | EWMA probability-calibration error drift detector. |
| [`ChaikinMoneyFlow`](documentation/algorithms/chaikin-money-flow.md) | Volume-weighted money flow over a window. |
| [`ChaikinOscillator`](documentation/algorithms/chaikin-oscillator.md) | MACD-style oscillator of the accumulation/distribution line. |
| [`ChandeMomentumOscillator`](documentation/algorithms/chande-momentum-oscillator.md) | Momentum oscillator using sums of recent gains and losses. |
| [`ChoppinessIndex`](documentation/algorithms/choppiness-index.md) | CHOP range/trend measure based on true range versus high-low range. |
| [`ClosePressureReversalSignal`](documentation/algorithms/close-pressure-reversal-signal.md) | End-of-day cross-sectional reversal signal using rest-of-day return, volume/transaction pressure, VWAP location, and rolling conformal-style error bands. |
| [`CointegrationBreakdownMonitor`](documentation/algorithms/cointegration-breakdown-monitor.md) | Streaming residual-z monitor for pair relationship breakdowns using an EWMA hedge estimate. |
| [`ConnorsRSI`](documentation/algorithms/connors-rsi.md) | Composite oscillator averaging price RSI, streak RSI, and percent-rank of one-period price change. |
| [`CommodityChannelIndex`](documentation/algorithms/commodity-channel-index.md) | CCI deviation of typical price from its moving average. |
| [`CoppockCurve`](documentation/algorithms/coppock-curve.md) | Weighted moving average of long and short rate-of-change values. |
| [`Correlation`](documentation/algorithms/correlation.md) | Rolling Pearson correlation between two series. |
| [`CorrelationRegimeDetector`](documentation/algorithms/correlation-regime-detector.md) | Stateful rolling correlation regime detector with upper/lower hysteresis bands. |
| [`CrossAssetCorrelationBreakDetector`](documentation/algorithms/cross-asset-correlation-break-detector.md) | Short-versus-long rolling correlation break detector for two assets. |
| [`CumulativeReturn`](documentation/algorithms/cumulative-return.md) | Cumulative return from the first close. |
| [`CUSUM`](documentation/algorithms/cusum.md) | Causal cumulative-sum event filter for detecting threshold-sized directional moves. |
| [`DDM`](documentation/algorithms/ddm.md) | Drift Detection Method for Bernoulli prediction-error streams. |
| [`DailyLogReturn`](documentation/algorithms/daily-log-return.md) | Log return between consecutive closes. |
| [`DailyReturn`](documentation/algorithms/daily-return.md) | Percentage return between consecutive closes. |
| [`Delay`](documentation/algorithms/delay.md) | Lagged value from a fixed number of samples ago. |
| [`DetrendedPriceOscillator`](documentation/algorithms/detrended-price-oscillator.md) | DPO cycle indicator comparing price to a displaced average. |
| [`DirectionalMovementIndex`](documentation/algorithms/directional-movement-index.md) | DX directional movement trend-strength indicator. |
| [`DoubleEMA`](documentation/algorithms/double-ema.md) | DEMA lag-reduced moving average. |
| [`DonchianChannel`](documentation/algorithms/donchian-channel.md) | Channel from rolling highest high and lowest low. |
| [`EDDM`](documentation/algorithms/eddm.md) | Early Drift Detection Method using distances between prediction errors. |
| [`EhlersOptimalTrackingFilter`](documentation/algorithms/ehlers-optimal-tracking-filter.md) | Adaptive tracking filter using Ehlers' price uncertainty tracking index. |
| [`ElderRayIndex`](documentation/algorithms/elder-ray-index.md) | Bull and bear power as high/low distance from an EMA of close. |
| [`EaseOfMovement`](documentation/algorithms/ease-of-movement.md) | Volume/range indicator for ease of price movement. |
| [`ExecutionCostSlippageRegimeDetector`](documentation/algorithms/execution-cost-slippage-regime-detector.md) | Stateful relative execution-cost/slippage regime detector from trade price versus quote mid. |
| [`FastStochastic`](documentation/algorithms/fast-stochastic.md) | Fast stochastic %K/%D oscillator. |
| [`FeatureDistributionDriftDetector`](documentation/algorithms/feature-distribution-drift-detector.md) | Bounded ADWIN-style drift detector for a single streaming feature distribution. |
| [`FibonacciRetracementLevels`](documentation/algorithms/fibonacci-retracement-levels.md) | Rolling Fibonacci retracement levels between recent high and low. |
| [`FisherTransform`](documentation/algorithms/fisher-transform.md) | Ehlers transform of normalized recent high/low position into a turning-point oscillator. |
| [`ForceIndex`](documentation/algorithms/force-index.md) | Price-change times volume oscillator. |
| [`FractalAdaptiveMovingAverage`](documentation/algorithms/fractal-adaptive-moving-average.md) | Ehlers FRAMA using fractal dimension to adapt EMA smoothing. |
| [`GaussianProcessRegressionBands`](documentation/algorithms/gaussian-process-regression-bands.md) | Rolling RBF-kernel Gaussian process posterior mean with uncertainty bands. |
| [`High`](documentation/algorithms/high.md) | Rolling highest value. |
| [`HighIndex`](documentation/algorithms/high-index.md) | Offset/index of the rolling highest value. |
| [`HighLow`](documentation/algorithms/high-low.md) | Combined rolling minimum and maximum values. |
| [`HighLowIndex`](documentation/algorithms/high-low-index.md) | Combined offsets/indexes of rolling minimum and maximum values. |
| [`HDDM`](documentation/algorithms/hddm.md) | Hoeffding-bound drift detector for Bernoulli prediction-error streams. |
| [`HeikinAshiTransform`](documentation/algorithms/heikin-ashi-transform.md) | Incremental Heikin-Ashi OHLC transform for smoothing candles. |
| [`HiddenSemiMarkovRegimeFilter`](documentation/algorithms/hidden-semi-markov-regime-filter.md) | Online Gaussian hidden semi-Markov-style regime filter with bounded duration bias. |
| [`HitRateDriftDetector`](documentation/algorithms/hit-rate-drift-detector.md) | EWMA hit-rate degradation detector using miss-rate hysteresis. |
| [`HullMovingAverage`](documentation/algorithms/hull-moving-average.md) | HMA lag-reduced weighted moving average. |
| [`Ichimoku`](documentation/algorithms/ichimoku.md) | Ichimoku conversion, base, and leading span components. |
| [`IntradayClockEchoSignal`](documentation/algorithms/intraday-clock-echo-signal.md) | Same-clock intraday return-periodicity signal trained from prior aggregate-bar day lists. |
| [`InteractingMultipleModelFilter`](documentation/algorithms/interacting-multiple-model-filter.md) | Four-regime IMM Kalman tracker that blends low-volatility, high-volatility, trend, and chop models by online probabilities. |
| [`KSTOscillator`](documentation/algorithms/kst-oscillator.md) | Pring Know Sure Thing smoothed multi-ROC oscillator. |
| [`KalmanExtremumTrend`](documentation/algorithms/kalman-extremum-trend.md) | Kalman trend combined with stochastic-style position inside recent extrema. |
| [`KalmanHedgeRatio`](documentation/algorithms/kalman-hedge-ratio.md) | Online Kalman regression hedge ratio and pair spread. |
| [`KalmanInnovationZScore`](documentation/algorithms/kalman-innovation-z-score.md) | Signed measurement innovation normalized by the predicted innovation standard deviation. |
| [`KalmanLocalLinearTrend`](documentation/algorithms/kalman-local-linear-trend.md) | Kalman local level/trend state-space estimator. |
| [`KalmanMovingAverage`](documentation/algorithms/kalman-moving-average.md) | Kalman price filter using a local linear price/velocity model. |
| [`KalmanPredictionBands`](documentation/algorithms/kalman-prediction-bands.md) | One-step Kalman prediction with upper/lower bands from predicted measurement uncertainty. |
| [`KalmanRegressionChannel`](documentation/algorithms/kalman-regression-channel.md) | Online Kalman regression with prediction channel and spread. |
| [`KalmanTrendSignal`](documentation/algorithms/kalman-trend-signal.md) | Kalman-filtered trend line with buy/sell signal based on price versus filtered trend. |
| [`KalmanVelocityOscillator`](documentation/algorithms/kalman-velocity-oscillator.md) | Zero-centered velocity state from a constant-velocity Kalman price model. |
| [`Kama`](documentation/algorithms/kama.md) | Kaufman Adaptive Moving Average. |
| [`KeltnerChannel`](documentation/algorithms/keltner-channel.md) | EMA/ATR volatility channel. |
| [`KeltnerChannelOriginal`](documentation/algorithms/keltner-channel-original.md) | Original SMA/range Keltner channel variant. |
| [`KlingerVolumeOscillator`](documentation/algorithms/klinger-volume-oscillator.md) | Volume-force oscillator using fast and slow EMAs plus signal line. |
| [`KSWIN`](documentation/algorithms/kswin.md) | Kolmogorov-Smirnov sliding-window drift detector. |
| [`KyleLambda`](documentation/algorithms/kyle-lambda.md) | Rolling price-impact slope of returns against signed square-root dollar volume. |
| [`LeadLagRegimeDetector`](documentation/algorithms/lead-lag-regime-detector.md) | EWMA cross-lag detector for which of two series is leading. |
| [`LiquidityDroughtDetector`](documentation/algorithms/liquidity-drought-detector.md) | Relative volume/depth drought detector using lower-threshold hysteresis. |
| [`LiquidityRegimeDetector`](documentation/algorithms/liquidity-regime-detector.md) | EWMA Amihud-style liquidity regime detector using absolute return per dollar volume. |
| [`LinearRegression`](documentation/algorithms/linear-regression.md) | Rolling least-squares fitted value. |
| [`LinearRegressionAngle`](documentation/algorithms/linear-regression-angle.md) | Angle of the rolling linear-regression slope. |
| [`LinearRegressionIntercept`](documentation/algorithms/linear-regression-intercept.md) | Intercept of the rolling linear-regression fit. |
| [`LinearRegressionSlope`](documentation/algorithms/linear-regression-slope.md) | Slope of the rolling linear-regression fit. |
| [`Low`](documentation/algorithms/low.md) | Rolling lowest value. |
| [`LowIndex`](documentation/algorithms/low-index.md) | Offset/index of the rolling lowest value. |
| [`MACDFix`](documentation/algorithms/macd-fix.md) | MACD with fixed 12/26 moving-average periods. |
| [`MassIndex`](documentation/algorithms/mass-index.md) | Range-expansion reversal indicator. |
| [`MarketOpenCloseTransitionDetector`](documentation/algorithms/market-open-close-transition-detector.md) | Session-progress transition detector for market-open and market-close bands. |
| [`MatchedFlowConformalSignal`](documentation/algorithms/matched-flow-conformal-signal.md) | Intraday OHLCV matched-flow signal with conformal-style rolling error bands and target sizing diagnostics. |
| [`MedianPrice`](documentation/algorithms/median-price.md) | Average of high and low. |
| [`MesaAdaptiveMovingAverage`](documentation/algorithms/mesa-adaptive-moving-average.md) | Ehlers MAMA/FAMA adaptive moving averages driven by dominant cycle phase. |
| [`MicrostructureNoiseRegimeDetector`](documentation/algorithms/microstructure-noise-regime-detector.md) | EWMA trade-versus-mid noise detector normalized by quoted spread. |
| [`MidPoint`](documentation/algorithms/mid-point.md) | Midpoint of rolling high and low values for one series. |
| [`MidPrice`](documentation/algorithms/mid-price.md) | Midpoint of rolling high and low price series. |
| [`MinusDirectionalIndicator`](documentation/algorithms/minus-directional-indicator.md) | Negative directional indicator. |
| [`MinusDirectionalMovement`](documentation/algorithms/minus-directional-movement.md) | Negative directional movement. |
| [`Momentum`](documentation/algorithms/momentum.md) | Difference between current value and a prior value. |
| [`MoneyFlowIndex`](documentation/algorithms/money-flow-index.md) | Volume-weighted RSI-like money flow oscillator. |
| [`NadarayaWatsonEnvelope`](documentation/algorithms/nadaraya-watson-envelope.md) | Gaussian-kernel Nadaraya-Watson smoother with weighted residual bands. |
| [`NegativeVolumeIndex`](documentation/algorithms/negative-volume-index.md) | Cumulative indicator that changes on lower-volume periods. |
| [`NormalizedATR`](documentation/algorithms/normalized-atr.md) | ATR normalized by close. |
| [`OnBalanceVolume`](documentation/algorithms/on-balance-volume.md) | Cumulative volume added/subtracted by close direction. |
| [`OnlineGaussianMixtureRegimeFilter`](documentation/algorithms/online-gaussian-mixture-regime-filter.md) | Online Gaussian mixture regime filter with bounded component count. |
| [`OnlineHMMRegimeFilter`](documentation/algorithms/online-hmm-regime-filter.md) | Online Gaussian hidden Markov regime filter with fixed transition persistence. |
| [`OnlineMarkovSwitchingVolatilityFilter`](documentation/algorithms/online-markov-switching-volatility-filter.md) | Online two-state Markov-switching volatility filter over close-to-close moves. |
| [`OrderFlowImbalance`](documentation/algorithms/order-flow-imbalance.md) | Quote-level best bid/ask price and size change pressure over a rolling update window. |
| [`OrderFlowImbalanceRegimeDetector`](documentation/algorithms/order-flow-imbalance-regime-detector.md) | EWMA order-flow imbalance regime detector with buy/sell pressure hysteresis. |
| [`PageHinkley`](documentation/algorithms/page-hinkley.md) | Causal Page-Hinkley mean-shift event detector with directed up/down output. |
| [`PairsSpreadRegimeDetector`](documentation/algorithms/pairs-spread-regime-detector.md) | Streaming EWMA hedge-ratio residual z-score detector for pair-spread regimes. |
| [`ParticleFilterTrend`](documentation/algorithms/particle-filter-trend.md) | Deterministic-seed particle trend filter with Laplace measurement likelihood and effective sample size output. |
| [`ParabolicSAR`](documentation/algorithms/parabolic-sar.md) | Parabolic stop-and-reverse trailing trend indicator. |
| [`PercentagePrice`](documentation/algorithms/percentage-price.md) | Percentage Price Oscillator. |
| [`PercentageVolume`](documentation/algorithms/percentage-volume.md) | Percentage Volume Oscillator. |
| [`PlusDirectionalIndicator`](documentation/algorithms/plus-directional-indicator.md) | Positive directional indicator. |
| [`PlusDirectionalMovement`](documentation/algorithms/plus-directional-movement.md) | Positive directional movement. |
| [`PredictionErrorDriftDetector`](documentation/algorithms/prediction-error-drift-detector.md) | EWMA absolute prediction-error drift detector. |
| [`QuoteMessageRateRegimeDetector`](documentation/algorithms/quote-message-rate-regime-detector.md) | Relative EWMA quote-message-rate regime detector. |
| [`QuoteStuffingDetector`](documentation/algorithms/quote-stuffing-detector.md) | EWMA quote-to-trade message ratio detector for quote-stuffing episodes. |
| [`RateOfChangePercentage`](documentation/algorithms/rate-of-change-percentage.md) | Period-over-period rate of change as a fraction. |
| [`RateOfChangeRatio`](documentation/algorithms/rate-of-change-ratio.md) | Rate-of-change ratio against a prior value. |
| [`RateOfChangeRatio100`](documentation/algorithms/rate-of-change-ratio-100.md) | Rate-of-change ratio scaled by 100. |
| [`RenkoBrickGenerator`](documentation/algorithms/renko-brick-generator.md) | Event-driven Renko price transform that emits signed brick counts and current brick state from close updates. |
| [`ResidualDriftDetector`](documentation/algorithms/residual-drift-detector.md) | EWMA residual z-score drift detector with signed hysteresis output. |
| [`RelativeVigorIndex`](documentation/algorithms/relative-vigor-index.md) | Smoothed close-open momentum relative to high-low range with signal line. |
| [`RealizedVarianceRegimeDetector`](documentation/algorithms/realized-variance-regime-detector.md) | Rolling realized-variance regime detector from squared close-to-close changes. |
| [`RollingBetaShiftDetector`](documentation/algorithms/rolling-beta-shift-detector.md) | Causal adjacent-window beta shift detector. |
| [`RollingCorrelationShiftDetector`](documentation/algorithms/rolling-correlation-shift-detector.md) | Causal adjacent-window correlation shift detector. |
| [`RollingMeanShiftDetector`](documentation/algorithms/rolling-mean-shift-detector.md) | Causal adjacent-window mean shift detector using a two-sample z-score. |
| [`RollingMeanVarianceShiftDetector`](documentation/algorithms/rolling-mean-variance-shift-detector.md) | Causal adjacent-window combined mean and variance shift detector. |
| [`RollingSpreadLiquidityShiftDetector`](documentation/algorithms/rolling-spread-liquidity-shift-detector.md) | Causal adjacent-window quote spread/depth liquidity stress shift detector. |
| [`RollingVarianceShiftDetector`](documentation/algorithms/rolling-variance-shift-detector.md) | Causal adjacent-window variance shift detector using log variance ratio. |
| [`SavitzkyGolayFilter`](documentation/algorithms/savitzky-golay-filter.md) | Rolling polynomial least-squares smoother with first and second derivative outputs. |
| [`SchaffTrendCycle`](documentation/algorithms/schaff-trend-cycle.md) | MACD/stochastic cycle oscillator. |
| [`SpreadFeatures`](documentation/algorithms/spread-features.md) | Quoted, effective, and realized spread estimates from trades and contemporaneous quotes. |
| [`SpreadExplosionDetector`](documentation/algorithms/spread-explosion-detector.md) | EWMA relative quoted-spread explosion detector. |
| [`SpreadRegimeDetector`](documentation/algorithms/spread-regime-detector.md) | Stateful quoted-spread regime detector using relative bid/ask spread. |
| [`StdDev`](documentation/algorithms/std-dev.md) | Rolling standard deviation. |
| [`StickyHMMRegimeFilter`](documentation/algorithms/sticky-hmm-regime-filter.md) | Online Gaussian HMM regime filter with high self-transition persistence. |
| [`StochRSI`](documentation/algorithms/stoch-rsi.md) | Stochastic oscillator applied to RSI values. |
| [`Stochastic`](documentation/algorithms/stochastic.md) | Slow stochastic oscillator. |
| [`SuperTrend`](documentation/algorithms/super-trend.md) | ATR-band trend-following indicator. |
| [`Summation`](documentation/algorithms/summation.md) | Rolling sum. |
| [`T3MovingAverage`](documentation/algorithms/t-3-moving-average.md) | Tillson T3 multi-EMA moving average. |
| [`ThresholdRegimeDetector`](documentation/algorithms/threshold-regime-detector.md) | Stateful threshold regime detector with upper/lower hysteresis bands. |
| [`TimeSeriesForecast`](documentation/algorithms/time-series-forecast.md) | Rolling linear-regression time-series forecast. |
| [`TradeIntensityRegimeDetector`](documentation/algorithms/trade-intensity-regime-detector.md) | EWMA relative trade-count intensity regime detector. |
| [`TrendChopRegimeDetector`](documentation/algorithms/trend-chop-regime-detector.md) | Efficiency-ratio trend-versus-chop regime detector using true range. |
| [`TwoFactorKalmanTrendFilter`](documentation/algorithms/two-factor-kalman-trend-filter.md) | Two-state short/long Kalman trend contribution model. |
| [`TrueRange`](documentation/algorithms/true-range.md) | Maximum of high-low and gaps from previous close. |
| [`TriangularMovingAverage`](documentation/algorithms/triangular-moving-average.md) | Double-smoothed triangular moving average. |
| [`TripleEMA`](documentation/algorithms/triple-ema.md) | TEMA lag-reduced moving average. |
| [`Trix`](documentation/algorithms/trix.md) | Triple-smoothed rate-of-change oscillator. |
| [`TypicalPrice`](documentation/algorithms/typical-price.md) | Average of high, low, and close. |
| [`UltimateOscillator`](documentation/algorithms/ultimate-oscillator.md) | Weighted multi-window buying-pressure oscillator. |
| [`UlcerIndex`](documentation/algorithms/ulcer-index.md) | Drawdown-based downside-risk measure. |
| [`VPIN`](documentation/algorithms/vpin.md) | Volume-synchronized probability of informed trading using bulk-volume classification and rolling volume-bucket imbalance. |
| [`Variance`](documentation/algorithms/variance.md) | Rolling variance. |
| [`VariableIndexDynamicAverage`](documentation/algorithms/variable-index-dynamic-average.md) | VIDYA adaptive EMA using absolute CMO as the smoothing factor. |
| [`VolatilityBreakoutDetector`](documentation/algorithms/volatility-breakout-detector.md) | EWMA z-score detector for unusually large close-to-close volatility breakouts. |
| [`VolatilityCompressionExpansionDetector`](documentation/algorithms/volatility-compression-expansion-detector.md) | Short-versus-long EWMA volatility ratio detector for compression and expansion regimes. |
| [`VolatilityRegimeDetector`](documentation/algorithms/volatility-regime-detector.md) | EWMA close-change volatility regime detector with high/low hysteresis bands. |
| [`VolumeProfile`](documentation/algorithms/volume-profile.md) | Rolling volume-by-price histogram that emits point of control and value-area high/low levels. |
| [`VolumePriceTrend`](documentation/algorithms/volume-price-trend.md) | Cumulative volume adjusted by percentage price change. |
| [`VolumeRegimeDetector`](documentation/algorithms/volume-regime-detector.md) | EWMA relative volume regime detector with high/low hysteresis bands. |
| [`VolumeWeightedAveragePrice`](documentation/algorithms/volume-weighted-average-price.md) | VWAP price weighted by traded volume. |
| [`VolumeWeightedMovingAverage`](documentation/algorithms/volume-weighted-moving-average.md) | VWMA rolling close weighted by volume. |
| [`Vortex`](documentation/algorithms/vortex.md) | Positive/negative Vortex trend movement indicator. |
| [`WeightedClosePrice`](documentation/algorithms/weighted-close-price.md) | Weighted close transform using high, low, and close. |
| [`WeightedMovingAverage`](documentation/algorithms/weighted-moving-average.md) | Weighted moving average with larger recent weights. |
| [`WilliamsR`](documentation/algorithms/williams-r.md) | Williams %R overbought/oversold oscillator. |
| [`ZigZagSwingDetector`](documentation/algorithms/zig-zag-swing-detector.md) | Close-based swing detector that filters price moves below a percentage threshold and emits confirmed pivots. |
