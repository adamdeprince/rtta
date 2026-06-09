# Algorithm Documentation

This directory holds detailed Markdown source pages for RTTA's public technical
analysis algorithms. Each page documents the public `update(...)` shape, the
implemented theory of operation, and the C++-derived recurrence used to update
state one sample at a time.

| Algorithm | Summary |
|---|---|
| [`ATR`](atr.md) | Average True Range volatility over a rolling window. |
| [`ATRP`](atrp.md) | Average True Range expressed as a percentage of price. |
| [`ATRRegimeDetector`](atr-regime-detector.md) | Stateful ATR regime detector with high/low hysteresis bands. |
| [`ADWIN`](adwin.md) | Adaptive-window mean drift detector with bounded history and directed shift output. |
| [`EMA`](ema.md) | Exponential moving average with more weight on recent samples. |
| [`EWMA`](ewma.md) | Exponentially weighted moving average parameterized by alpha/span/com. |
| [`EWMAZScoreShiftDetector`](ewmaz-score-shift-detector.md) | Causal EWMA mean/variance z-score event detector for threshold-sized shifts. |
| [`MACD`](macd.md) | Moving Average Convergence/Divergence oscillator and signal/histogram. |
| [`ROC`](roc.md) | Rate of Change momentum as percentage change over a lookback. |
| [`RSI`](rsi.md) | Relative Strength Index momentum oscillator. |
| [`SMA`](sma.md) | Simple moving average over a rolling window. |
| [`TSI`](tsi.md) | True Strength Index double-smoothed momentum oscillator. |
| [`AbsolutePriceOscillator`](absolute-price-oscillator.md) | Difference between fast and slow moving averages in price units. |
| [`AccumulationDistribution`](accumulation-distribution.md) | Volume-price accumulation/distribution line. |
| [`AlphaBetaGammaTrackingFilter`](alpha-beta-gamma-tracking-filter.md) | Steady-state Kalman-like price, velocity, and acceleration tracker. |
| [`AmihudIlliquidity`](amihud-illiquidity.md) | Rolling average absolute return per dollar of traded volume. |
| [`AnchoredVWAP`](anchored-vwap.md) | VWAP accumulated from arbitrary anchor/reset events rather than a fixed session or rolling window. |
| [`Aroon`](aroon.md) | Aroon Up/Down trend age indicators based on recent highs and lows. |
| [`AroonOscillator`](aroon-oscillator.md) | Difference between Aroon Up and Aroon Down. |
| [`AverageDirectionalMovementIndex`](average-directional-movement-index.md) | ADX trend-strength indicator. |
| [`AverageDirectionalMovementIndexRating`](average-directional-movement-index-rating.md) | ADXR smoothed ADX trend-strength rating. |
| [`AveragePrice`](average-price.md) | Average of open, high, low, and close. |
| [`AuctionContinuousMarketTransitionDetector`](auction-continuous-market-transition-detector.md) | Hysteresis detector for auction-versus-continuous market phase signals. |
| [`AwesomeOscillator`](awesome-oscillator.md) | Difference between short and long median-price moving averages. |
| [`BalanceOfPower`](balance-of-power.md) | Open/high/low/close buying-versus-selling pressure measure. |
| [`Beta`](beta.md) | Rolling beta of one series against another. |
| [`BetaRegimeDetector`](beta-regime-detector.md) | Stateful rolling beta regime detector with upper/lower hysteresis bands. |
| [`BidAskBounceRegimeDetector`](bid-ask-bounce-regime-detector.md) | EWMA bid/ask side alternation detector for quote-bounce regimes. |
| [`BollingerBands`](bollinger-bands.md) | Moving-average envelope based on standard deviations. |
| [`BoundedBOCPD`](bounded-bocpd.md) | Bounded-memory Bayesian online change-point detector with constant hazard. |
| [`CalibrationDriftDetector`](calibration-drift-detector.md) | EWMA probability-calibration error drift detector. |
| [`ChaikinMoneyFlow`](chaikin-money-flow.md) | Volume-weighted money flow over a window. |
| [`ChaikinOscillator`](chaikin-oscillator.md) | MACD-style oscillator of the accumulation/distribution line. |
| [`ChandeMomentumOscillator`](chande-momentum-oscillator.md) | Momentum oscillator using sums of recent gains and losses. |
| [`ChoppinessIndex`](choppiness-index.md) | CHOP range/trend measure based on true range versus high-low range. |
| [`ClosePressureReversalSignal`](close-pressure-reversal-signal.md) | End-of-day cross-sectional reversal signal using rest-of-day return, volume/transaction pressure, VWAP location, and rolling conformal-style error bands. |
| [`CointegrationBreakdownMonitor`](cointegration-breakdown-monitor.md) | Streaming residual-z monitor for pair relationship breakdowns using an EWMA hedge estimate. |
| [`ConnorsRSI`](connors-rsi.md) | Composite oscillator averaging price RSI, streak RSI, and percent-rank of one-period price change. |
| [`CommodityChannelIndex`](commodity-channel-index.md) | CCI deviation of typical price from its moving average. |
| [`CoppockCurve`](coppock-curve.md) | Weighted moving average of long and short rate-of-change values. |
| [`Correlation`](correlation.md) | Rolling Pearson correlation between two series. |
| [`CorrelationRegimeDetector`](correlation-regime-detector.md) | Stateful rolling correlation regime detector with upper/lower hysteresis bands. |
| [`CrossAssetCorrelationBreakDetector`](cross-asset-correlation-break-detector.md) | Short-versus-long rolling correlation break detector for two assets. |
| [`CumulativeReturn`](cumulative-return.md) | Cumulative return from the first close. |
| [`CUSUM`](cusum.md) | Causal cumulative-sum event filter for detecting threshold-sized directional moves. |
| [`DDM`](ddm.md) | Drift Detection Method for Bernoulli prediction-error streams. |
| [`DailyLogReturn`](daily-log-return.md) | Log return between consecutive closes. |
| [`DailyReturn`](daily-return.md) | Percentage return between consecutive closes. |
| [`Delay`](delay.md) | Lagged value from a fixed number of samples ago. |
| [`DetrendedPriceOscillator`](detrended-price-oscillator.md) | DPO cycle indicator comparing price to a displaced average. |
| [`DirectionalMovementIndex`](directional-movement-index.md) | DX directional movement trend-strength indicator. |
| [`DoubleEMA`](double-ema.md) | DEMA lag-reduced moving average. |
| [`DonchianChannel`](donchian-channel.md) | Channel from rolling highest high and lowest low. |
| [`EDDM`](eddm.md) | Early Drift Detection Method using distances between prediction errors. |
| [`EhlersOptimalTrackingFilter`](ehlers-optimal-tracking-filter.md) | Adaptive tracking filter using Ehlers' price uncertainty tracking index. |
| [`ElderRayIndex`](elder-ray-index.md) | Bull and bear power as high/low distance from an EMA of close. |
| [`EaseOfMovement`](ease-of-movement.md) | Volume/range indicator for ease of price movement. |
| [`ExecutionCostSlippageRegimeDetector`](execution-cost-slippage-regime-detector.md) | Stateful relative execution-cost/slippage regime detector from trade price versus quote mid. |
| [`FastStochastic`](fast-stochastic.md) | Fast stochastic %K/%D oscillator. |
| [`FeatureDistributionDriftDetector`](feature-distribution-drift-detector.md) | Bounded ADWIN-style drift detector for a single streaming feature distribution. |
| [`FibonacciRetracementLevels`](fibonacci-retracement-levels.md) | Rolling Fibonacci retracement levels between recent high and low. |
| [`FisherTransform`](fisher-transform.md) | Ehlers transform of normalized recent high/low position into a turning-point oscillator. |
| [`ForceIndex`](force-index.md) | Price-change times volume oscillator. |
| [`FractalAdaptiveMovingAverage`](fractal-adaptive-moving-average.md) | Ehlers FRAMA using fractal dimension to adapt EMA smoothing. |
| [`GaussianProcessRegressionBands`](gaussian-process-regression-bands.md) | Rolling RBF-kernel Gaussian process posterior mean with uncertainty bands. |
| [`High`](high.md) | Rolling highest value. |
| [`HighIndex`](high-index.md) | Offset/index of the rolling highest value. |
| [`HighLow`](high-low.md) | Combined rolling minimum and maximum values. |
| [`HighLowIndex`](high-low-index.md) | Combined offsets/indexes of rolling minimum and maximum values. |
| [`HDDM`](hddm.md) | Hoeffding-bound drift detector for Bernoulli prediction-error streams. |
| [`HeikinAshiTransform`](heikin-ashi-transform.md) | Incremental Heikin-Ashi OHLC transform for smoothing candles. |
| [`HiddenSemiMarkovRegimeFilter`](hidden-semi-markov-regime-filter.md) | Online Gaussian hidden semi-Markov-style regime filter with bounded duration bias. |
| [`HitRateDriftDetector`](hit-rate-drift-detector.md) | EWMA hit-rate degradation detector using miss-rate hysteresis. |
| [`HullMovingAverage`](hull-moving-average.md) | HMA lag-reduced weighted moving average. |
| [`Ichimoku`](ichimoku.md) | Ichimoku conversion, base, and leading span components. |
| [`IntradayClockEchoSignal`](intraday-clock-echo-signal.md) | Same-clock intraday return-periodicity signal trained from prior aggregate-bar day lists. |
| [`InteractingMultipleModelFilter`](interacting-multiple-model-filter.md) | Four-regime IMM Kalman tracker that blends low-volatility, high-volatility, trend, and chop models by online probabilities. |
| [`KSTOscillator`](kst-oscillator.md) | Pring Know Sure Thing smoothed multi-ROC oscillator. |
| [`KalmanExtremumTrend`](kalman-extremum-trend.md) | Kalman trend combined with stochastic-style position inside recent extrema. |
| [`KalmanHedgeRatio`](kalman-hedge-ratio.md) | Online Kalman regression hedge ratio and pair spread. |
| [`KalmanInnovationZScore`](kalman-innovation-z-score.md) | Signed measurement innovation normalized by the predicted innovation standard deviation. |
| [`KalmanLocalLinearTrend`](kalman-local-linear-trend.md) | Kalman local level/trend state-space estimator. |
| [`KalmanMovingAverage`](kalman-moving-average.md) | Kalman price filter using a local linear price/velocity model. |
| [`KalmanPredictionBands`](kalman-prediction-bands.md) | One-step Kalman prediction with upper/lower bands from predicted measurement uncertainty. |
| [`KalmanRegressionChannel`](kalman-regression-channel.md) | Online Kalman regression with prediction channel and spread. |
| [`KalmanTrendSignal`](kalman-trend-signal.md) | Kalman-filtered trend line with buy/sell signal based on price versus filtered trend. |
| [`KalmanVelocityOscillator`](kalman-velocity-oscillator.md) | Zero-centered velocity state from a constant-velocity Kalman price model. |
| [`Kama`](kama.md) | Kaufman Adaptive Moving Average. |
| [`KeltnerChannel`](keltner-channel.md) | EMA/ATR volatility channel. |
| [`KeltnerChannelOriginal`](keltner-channel-original.md) | Original SMA/range Keltner channel variant. |
| [`KlingerVolumeOscillator`](klinger-volume-oscillator.md) | Volume-force oscillator using fast and slow EMAs plus signal line. |
| [`KSWIN`](kswin.md) | Kolmogorov-Smirnov sliding-window drift detector. |
| [`KyleLambda`](kyle-lambda.md) | Rolling price-impact slope of returns against signed square-root dollar volume. |
| [`LeadLagRegimeDetector`](lead-lag-regime-detector.md) | EWMA cross-lag detector for which of two series is leading. |
| [`LiquidityDroughtDetector`](liquidity-drought-detector.md) | Relative volume/depth drought detector using lower-threshold hysteresis. |
| [`LiquidityRegimeDetector`](liquidity-regime-detector.md) | EWMA Amihud-style liquidity regime detector using absolute return per dollar volume. |
| [`LinearRegression`](linear-regression.md) | Rolling least-squares fitted value. |
| [`LinearRegressionAngle`](linear-regression-angle.md) | Angle of the rolling linear-regression slope. |
| [`LinearRegressionIntercept`](linear-regression-intercept.md) | Intercept of the rolling linear-regression fit. |
| [`LinearRegressionSlope`](linear-regression-slope.md) | Slope of the rolling linear-regression fit. |
| [`Low`](low.md) | Rolling lowest value. |
| [`LowIndex`](low-index.md) | Offset/index of the rolling lowest value. |
| [`MACDFix`](macd-fix.md) | MACD with fixed 12/26 moving-average periods. |
| [`MassIndex`](mass-index.md) | Range-expansion reversal indicator. |
| [`MarketOpenCloseTransitionDetector`](market-open-close-transition-detector.md) | Session-progress transition detector for market-open and market-close bands. |
| [`MatchedFlowConformalSignal`](matched-flow-conformal-signal.md) | Intraday OHLCV matched-flow signal with conformal-style rolling error bands and target sizing diagnostics. |
| [`MedianPrice`](median-price.md) | Average of high and low. |
| [`MesaAdaptiveMovingAverage`](mesa-adaptive-moving-average.md) | Ehlers MAMA/FAMA adaptive moving averages driven by dominant cycle phase. |
| [`MicrostructureNoiseRegimeDetector`](microstructure-noise-regime-detector.md) | EWMA trade-versus-mid noise detector normalized by quoted spread. |
| [`MidPoint`](mid-point.md) | Midpoint of rolling high and low values for one series. |
| [`MidPrice`](mid-price.md) | Midpoint of rolling high and low price series. |
| [`MinusDirectionalIndicator`](minus-directional-indicator.md) | Negative directional indicator. |
| [`MinusDirectionalMovement`](minus-directional-movement.md) | Negative directional movement. |
| [`Momentum`](momentum.md) | Difference between current value and a prior value. |
| [`MoneyFlowIndex`](money-flow-index.md) | Volume-weighted RSI-like money flow oscillator. |
| [`NadarayaWatsonEnvelope`](nadaraya-watson-envelope.md) | Gaussian-kernel Nadaraya-Watson smoother with weighted residual bands. |
| [`NegativeVolumeIndex`](negative-volume-index.md) | Cumulative indicator that changes on lower-volume periods. |
| [`NormalizedATR`](normalized-atr.md) | ATR normalized by close. |
| [`OnBalanceVolume`](on-balance-volume.md) | Cumulative volume added/subtracted by close direction. |
| [`OnlineGaussianMixtureRegimeFilter`](online-gaussian-mixture-regime-filter.md) | Online Gaussian mixture regime filter with bounded component count. |
| [`OnlineHMMRegimeFilter`](online-hmm-regime-filter.md) | Online Gaussian hidden Markov regime filter with fixed transition persistence. |
| [`OnlineMarkovSwitchingVolatilityFilter`](online-markov-switching-volatility-filter.md) | Online two-state Markov-switching volatility filter over close-to-close moves. |
| [`OrderFlowImbalance`](order-flow-imbalance.md) | Quote-level best bid/ask price and size change pressure over a rolling update window. |
| [`OrderFlowImbalanceRegimeDetector`](order-flow-imbalance-regime-detector.md) | EWMA order-flow imbalance regime detector with buy/sell pressure hysteresis. |
| [`PageHinkley`](page-hinkley.md) | Causal Page-Hinkley mean-shift event detector with directed up/down output. |
| [`PairsSpreadRegimeDetector`](pairs-spread-regime-detector.md) | Streaming EWMA hedge-ratio residual z-score detector for pair-spread regimes. |
| [`ParticleFilterTrend`](particle-filter-trend.md) | Deterministic-seed particle trend filter with Laplace measurement likelihood and effective sample size output. |
| [`ParabolicSAR`](parabolic-sar.md) | Parabolic stop-and-reverse trailing trend indicator. |
| [`PercentagePrice`](percentage-price.md) | Percentage Price Oscillator. |
| [`PercentageVolume`](percentage-volume.md) | Percentage Volume Oscillator. |
| [`PlusDirectionalIndicator`](plus-directional-indicator.md) | Positive directional indicator. |
| [`PlusDirectionalMovement`](plus-directional-movement.md) | Positive directional movement. |
| [`PredictionErrorDriftDetector`](prediction-error-drift-detector.md) | EWMA absolute prediction-error drift detector. |
| [`QuoteMessageRateRegimeDetector`](quote-message-rate-regime-detector.md) | Relative EWMA quote-message-rate regime detector. |
| [`QuoteStuffingDetector`](quote-stuffing-detector.md) | EWMA quote-to-trade message ratio detector for quote-stuffing episodes. |
| [`RateOfChangePercentage`](rate-of-change-percentage.md) | Period-over-period rate of change as a fraction. |
| [`RateOfChangeRatio`](rate-of-change-ratio.md) | Rate-of-change ratio against a prior value. |
| [`RateOfChangeRatio100`](rate-of-change-ratio-100.md) | Rate-of-change ratio scaled by 100. |
| [`RenkoBrickGenerator`](renko-brick-generator.md) | Event-driven Renko price transform that emits signed brick counts and current brick state from close updates. |
| [`ResidualDriftDetector`](residual-drift-detector.md) | EWMA residual z-score drift detector with signed hysteresis output. |
| [`RelativeVigorIndex`](relative-vigor-index.md) | Smoothed close-open momentum relative to high-low range with signal line. |
| [`RealizedVarianceRegimeDetector`](realized-variance-regime-detector.md) | Rolling realized-variance regime detector from squared close-to-close changes. |
| [`RollingBetaShiftDetector`](rolling-beta-shift-detector.md) | Causal adjacent-window beta shift detector. |
| [`RollingCorrelationShiftDetector`](rolling-correlation-shift-detector.md) | Causal adjacent-window correlation shift detector. |
| [`RollingMeanShiftDetector`](rolling-mean-shift-detector.md) | Causal adjacent-window mean shift detector using a two-sample z-score. |
| [`RollingMeanVarianceShiftDetector`](rolling-mean-variance-shift-detector.md) | Causal adjacent-window combined mean and variance shift detector. |
| [`RollingSpreadLiquidityShiftDetector`](rolling-spread-liquidity-shift-detector.md) | Causal adjacent-window quote spread/depth liquidity stress shift detector. |
| [`RollingVarianceShiftDetector`](rolling-variance-shift-detector.md) | Causal adjacent-window variance shift detector using log variance ratio. |
| [`SavitzkyGolayFilter`](savitzky-golay-filter.md) | Rolling polynomial least-squares smoother with first and second derivative outputs. |
| [`SchaffTrendCycle`](schaff-trend-cycle.md) | MACD/stochastic cycle oscillator. |
| [`SpreadFeatures`](spread-features.md) | Quoted, effective, and realized spread estimates from trades and contemporaneous quotes. |
| [`SpreadExplosionDetector`](spread-explosion-detector.md) | EWMA relative quoted-spread explosion detector. |
| [`SpreadRegimeDetector`](spread-regime-detector.md) | Stateful quoted-spread regime detector using relative bid/ask spread. |
| [`StdDev`](std-dev.md) | Rolling standard deviation. |
| [`StickyHMMRegimeFilter`](sticky-hmm-regime-filter.md) | Online Gaussian HMM regime filter with high self-transition persistence. |
| [`StochRSI`](stoch-rsi.md) | Stochastic oscillator applied to RSI values. |
| [`Stochastic`](stochastic.md) | Slow stochastic oscillator. |
| [`SuperTrend`](super-trend.md) | ATR-band trend-following indicator. |
| [`Summation`](summation.md) | Rolling sum. |
| [`T3MovingAverage`](t-3-moving-average.md) | Tillson T3 multi-EMA moving average. |
| [`ThresholdRegimeDetector`](threshold-regime-detector.md) | Stateful threshold regime detector with upper/lower hysteresis bands. |
| [`TimeSeriesForecast`](time-series-forecast.md) | Rolling linear-regression time-series forecast. |
| [`TradeIntensityRegimeDetector`](trade-intensity-regime-detector.md) | EWMA relative trade-count intensity regime detector. |
| [`TrendChopRegimeDetector`](trend-chop-regime-detector.md) | Efficiency-ratio trend-versus-chop regime detector using true range. |
| [`TwoFactorKalmanTrendFilter`](two-factor-kalman-trend-filter.md) | Two-state short/long Kalman trend contribution model. |
| [`TrueRange`](true-range.md) | Maximum of high-low and gaps from previous close. |
| [`TriangularMovingAverage`](triangular-moving-average.md) | Double-smoothed triangular moving average. |
| [`TripleEMA`](triple-ema.md) | TEMA lag-reduced moving average. |
| [`Trix`](trix.md) | Triple-smoothed rate-of-change oscillator. |
| [`TypicalPrice`](typical-price.md) | Average of high, low, and close. |
| [`UltimateOscillator`](ultimate-oscillator.md) | Weighted multi-window buying-pressure oscillator. |
| [`UlcerIndex`](ulcer-index.md) | Drawdown-based downside-risk measure. |
| [`VPIN`](vpin.md) | Volume-synchronized probability of informed trading using bulk-volume classification and rolling volume-bucket imbalance. |
| [`Variance`](variance.md) | Rolling variance. |
| [`VariableIndexDynamicAverage`](variable-index-dynamic-average.md) | VIDYA adaptive EMA using absolute CMO as the smoothing factor. |
| [`VolatilityBreakoutDetector`](volatility-breakout-detector.md) | EWMA z-score detector for unusually large close-to-close volatility breakouts. |
| [`VolatilityCompressionExpansionDetector`](volatility-compression-expansion-detector.md) | Short-versus-long EWMA volatility ratio detector for compression and expansion regimes. |
| [`VolatilityRegimeDetector`](volatility-regime-detector.md) | EWMA close-change volatility regime detector with high/low hysteresis bands. |
| [`VolumeProfile`](volume-profile.md) | Rolling volume-by-price histogram that emits point of control and value-area high/low levels. |
| [`VolumePriceTrend`](volume-price-trend.md) | Cumulative volume adjusted by percentage price change. |
| [`VolumeRegimeDetector`](volume-regime-detector.md) | EWMA relative volume regime detector with high/low hysteresis bands. |
| [`VolumeWeightedAveragePrice`](volume-weighted-average-price.md) | VWAP price weighted by traded volume. |
| [`VolumeWeightedMovingAverage`](volume-weighted-moving-average.md) | VWMA rolling close weighted by volume. |
| [`Vortex`](vortex.md) | Positive/negative Vortex trend movement indicator. |
| [`WeightedClosePrice`](weighted-close-price.md) | Weighted close transform using high, low, and close. |
| [`WeightedMovingAverage`](weighted-moving-average.md) | Weighted moving average with larger recent weights. |
| [`WilliamsR`](williams-r.md) | Williams %R overbought/oversold oscillator. |
| [`ZigZagSwingDetector`](zig-zag-swing-detector.md) | Close-based swing detector that filters price moves below a percentage threshold and emits confirmed pivots. |
