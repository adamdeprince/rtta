# Algorithms

This file lists the public indicator algorithms exported by `rtta`. Tuning/result helper types are intentionally omitted. Documentation links prefer ChartSchool when it has a reasonably direct article; non-ChartSchool links are used only when ChartSchool does not cover that specific algorithm or helper.

| Algorithm | Description | Documentation |
|---|---|---|
| `ATR` | Average True Range volatility over a rolling window. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp |
| `ATRP` | Average True Range expressed as a percentage of price. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp |
| `ATRRegimeDetector` | Stateful ATR regime detector with high/low hysteresis bands. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp |
| `ADWIN` | Adaptive-window mean drift detector with bounded history and directed shift output. | https://en.wikipedia.org/wiki/Concept_drift |
| `EMA` | Exponential moving average with more weight on recent samples. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential |
| `EWMA` | Exponentially weighted moving average parameterized by alpha/span/com. | https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html |
| `EWMAZScoreShiftDetector` | Causal EWMA mean/variance z-score event detector for threshold-sized shifts. | https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html |
| `MACD` | Moving Average Convergence/Divergence oscillator and signal/histogram. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator |
| `ROC` | Rate of Change momentum as percentage change over a lookback. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum |
| `RSI` | Relative Strength Index momentum oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi |
| `SMA` | Simple moving average over a rolling window. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential |
| `TSI` | True Strength Index double-smoothed momentum oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/true-strength-index |
| `AbsolutePriceOscillator` | Difference between fast and slow moving averages in price units. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator |
| `AccumulationDistribution` | Volume-price accumulation/distribution line. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/accumulation-distribution-line |
| `AlphaBetaGammaTrackingFilter` | Steady-state Kalman-like price, velocity, and acceleration tracker. | https://kalmanfilter.net/alphabeta.html |
| `AmihudIlliquidity` | Rolling average absolute return per dollar of traded volume. | https://ba-odegaard.no/teach/notes/liquidity_estimators/amihud_estimator/amihud_lectures.pdf |
| `AnchoredVWAP` | VWAP accumulated from arbitrary anchor/reset events rather than a fixed session or rolling window. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/anchored-vwap |
| `Aroon` | Aroon Up/Down trend age indicators based on recent highs and lows. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/aroon |
| `AroonOscillator` | Difference between Aroon Up and Aroon Down. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/aroon-oscillator |
| `AverageDirectionalMovementIndex` | ADX trend-strength indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `AverageDirectionalMovementIndexRating` | ADXR smoothed ADX trend-strength rating. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `AveragePrice` | Average of open, high, low, and close. | https://tulipindicators.org/avgprice |
| `AuctionContinuousMarketTransitionDetector` | Hysteresis detector for auction-versus-continuous market phase signals. | https://en.wikipedia.org/wiki/Call_market |
| `AwesomeOscillator` | Difference between short and long median-price moving averages. | https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html |
| `BalanceOfPower` | Open/high/low/close buying-versus-selling pressure measure. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/balance-of-power-bop |
| `Beta` | Rolling beta of one series against another. | https://www.investopedia.com/terms/b/beta.asp |
| `BetaRegimeDetector` | Stateful rolling beta regime detector with upper/lower hysteresis bands. | https://www.investopedia.com/terms/b/beta.asp |
| `BidAskBounceRegimeDetector` | EWMA bid/ask side alternation detector for quote-bounce regimes. | https://en.wikipedia.org/wiki/Bid%E2%80%93ask_spread |
| `BollingerBands` | Moving-average envelope based on standard deviations. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/bollinger-bands |
| `BoundedBOCPD` | Bounded-memory Bayesian online change-point detector with constant hazard. | https://arxiv.org/abs/0710.3742 |
| `CalibrationDriftDetector` | EWMA probability-calibration error drift detector. | https://en.wikipedia.org/wiki/Calibration_(statistics) |
| `ChaikinMoneyFlow` | Volume-weighted money flow over a window. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/chaikin-money-flow-cmf |
| `ChaikinOscillator` | MACD-style oscillator of the accumulation/distribution line. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/chaikin-oscillator |
| `ChandeMomentumOscillator` | Momentum oscillator using sums of recent gains and losses. | https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo |
| `ChoppinessIndex` | CHOP range/trend measure based on true range versus high-low range. | https://www.angelone.in/knowledge-center/online-share-trading/choppiness-index-indicator |
| `ClosePressureReversalSignal` | End-of-day cross-sectional reversal signal using rest-of-day return, volume/transaction pressure, VWAP location, and rolling conformal-style error bands. | documentation/close_pressure_reversal_signal.md |
| `CointegrationBreakdownMonitor` | Streaming residual-z monitor for pair relationship breakdowns using an EWMA hedge estimate. | https://en.wikipedia.org/wiki/Cointegration |
| `ConnorsRSI` | Composite oscillator averaging price RSI, streak RSI, and percent-rank of one-period price change. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/connorsrsi |
| `CommodityChannelIndex` | CCI deviation of typical price from its moving average. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/commodity-channel-index-cci |
| `CoppockCurve` | Weighted moving average of long and short rate-of-change values. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/coppock-curve |
| `Correlation` | Rolling Pearson correlation between two series. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/correlation-coefficient |
| `CorrelationRegimeDetector` | Stateful rolling correlation regime detector with upper/lower hysteresis bands. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/correlation-coefficient |
| `CrossAssetCorrelationBreakDetector` | Short-versus-long rolling correlation break detector for two assets. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/correlation-coefficient |
| `CumulativeReturn` | Cumulative return from the first close. | https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html |
| `CUSUM` | Causal cumulative-sum event filter for detecting threshold-sized directional moves. | https://en.wikipedia.org/wiki/CUSUM |
| `DDM` | Drift Detection Method for Bernoulli prediction-error streams. | https://en.wikipedia.org/wiki/Concept_drift |
| `DailyLogReturn` | Log return between consecutive closes. | https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html |
| `DailyReturn` | Percentage return between consecutive closes. | https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html |
| `Delay` | Lagged value from a fixed number of samples ago. | https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html |
| `DetrendedPriceOscillator` | DPO cycle indicator comparing price to a displaced average. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/detrended-price-oscillator-dpo |
| `DirectionalMovementIndex` | DX directional movement trend-strength indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `DoubleEMA` | DEMA lag-reduced moving average. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/double-exponential-moving-average-dema |
| `DonchianChannel` | Channel from rolling highest high and lowest low. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/donchian-channels |
| `EDDM` | Early Drift Detection Method using distances between prediction errors. | https://en.wikipedia.org/wiki/Concept_drift |
| `EhlersOptimalTrackingFilter` | Adaptive tracking filter using Ehlers' price uncertainty tracking index. | https://www.prorealcode.com/prorealtime-indicators/john-ehlers-optimal-tracking-filter/ |
| `ElderRayIndex` | Bull and bear power as high/low distance from an EMA of close. | https://www.investopedia.com/articles/trading/03/022603.asp |
| `EaseOfMovement` | Volume/range indicator for ease of price movement. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ease-of-movement-emv |
| `ExecutionCostSlippageRegimeDetector` | Stateful relative execution-cost/slippage regime detector from trade price versus quote mid. | https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf |
| `FastStochastic` | Fast stochastic %K/%D oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochastic-oscillator-fast-slow-and-full |
| `FeatureDistributionDriftDetector` | Bounded ADWIN-style drift detector for a single streaming feature distribution. | https://en.wikipedia.org/wiki/Concept_drift |
| `FibonacciRetracementLevels` | Rolling Fibonacci retracement levels between recent high and low. | https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/fibonacci-retracement |
| `FisherTransform` | Ehlers transform of normalized recent high/low position into a turning-point oscillator. | https://trendspider.com/learning-center/fisher-transform-a-comprehensive-guide/ |
| `ForceIndex` | Price-change times volume oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/force-index |
| `FractalAdaptiveMovingAverage` | Ehlers FRAMA using fractal dimension to adapt EMA smoothing. | https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/fama |
| `GaussianProcessRegressionBands` | Rolling RBF-kernel Gaussian process posterior mean with uncertainty bands. | https://www.luxalgo.com/library/indicator/machine-learning-gaussian-process-regression/ |
| `High` | Rolling highest value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/highest-high-value |
| `HighIndex` | Offset/index of the rolling highest value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/distance-to-highs |
| `HighLow` | Combined rolling minimum and maximum values. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/high-low-bands |
| `HighLowIndex` | Combined offsets/indexes of rolling minimum and maximum values. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/distance-to-highs |
| `HDDM` | Hoeffding-bound drift detector for Bernoulli prediction-error streams. | https://en.wikipedia.org/wiki/Hoeffding%27s_inequality |
| `HeikinAshiTransform` | Incremental Heikin-Ashi OHLC transform for smoothing candles. | https://www.mql5.com/en/articles/19260 |
| `HiddenSemiMarkovRegimeFilter` | Online Gaussian hidden semi-Markov-style regime filter with bounded duration bias. | https://en.wikipedia.org/wiki/Hidden_semi-Markov_model |
| `HitRateDriftDetector` | EWMA hit-rate degradation detector using miss-rate hysteresis. | https://en.wikipedia.org/wiki/Concept_drift |
| `HullMovingAverage` | HMA lag-reduced weighted moving average. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/hull-moving-average-hma |
| `Ichimoku` | Ichimoku conversion, base, and leading span components. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/ichimoku-cloud |
| `IntradayClockEchoSignal` | Same-clock intraday return-periodicity signal trained from prior aggregate-bar day lists. | documentation/intraday_clock_echo_signal.md |
| `InteractingMultipleModelFilter` | Four-regime IMM Kalman tracker that blends low-volatility, high-volatility, trend, and chop models by online probabilities. | https://www.sciencedirect.com/science/article/abs/pii/S1544612316302215 |
| `KSTOscillator` | Pring Know Sure Thing smoothed multi-ROC oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/pring-s-know-sure-thing-kst |
| `KalmanExtremumTrend` | Kalman trend combined with stochastic-style position inside recent extrema. | https://arxiv.org/pdf/1808.03297 |
| `KalmanHedgeRatio` | Online Kalman regression hedge ratio and pair spread. | https://www.quantstart.com/articles/Dynamic-Hedge-Ratio-Between-ETF-Pairs-Using-the-Kalman-Filter/ |
| `KalmanInnovationZScore` | Signed measurement innovation normalized by the predicted innovation standard deviation. | https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf |
| `KalmanLocalLinearTrend` | Kalman local level/trend state-space estimator. | https://www.statsmodels.org/v0.12.2/examples/notebooks/generated/statespace_local_linear_trend.html |
| `KalmanMovingAverage` | Kalman price filter using a local linear price/velocity model. | https://arxiv.org/pdf/1808.03297 |
| `KalmanPredictionBands` | One-step Kalman prediction with upper/lower bands from predicted measurement uncertainty. | https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf |
| `KalmanRegressionChannel` | Online Kalman regression with prediction channel and spread. | https://www.quantstart.com/articles/Dynamic-Hedge-Ratio-Between-ETF-Pairs-Using-the-Kalman-Filter/ |
| `KalmanTrendSignal` | Kalman-filtered trend line with buy/sell signal based on price versus filtered trend. | https://www.aimspress.com/aimspress-data/dsfe/2024/4/PDF/DSFE-04-04-023.pdf |
| `KalmanVelocityOscillator` | Zero-centered velocity state from a constant-velocity Kalman price model. | https://arxiv.org/pdf/1808.03297 |
| `Kama` | Kaufman Adaptive Moving Average. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama |
| `KeltnerChannel` | EMA/ATR volatility channel. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/keltner-channels |
| `KeltnerChannelOriginal` | Original SMA/range Keltner channel variant. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/keltner-channels |
| `KlingerVolumeOscillator` | Volume-force oscillator using fast and slow EMAs plus signal line. | https://trendspider.com/learning-center/introduction-to-klinger-oscillator/ |
| `KSWIN` | Kolmogorov-Smirnov sliding-window drift detector. | https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test |
| `KyleLambda` | Rolling price-impact slope of returns against signed square-root dollar volume. | https://frds.io/measures/kyle_lambda/ |
| `LeadLagRegimeDetector` | EWMA cross-lag detector for which of two series is leading. | https://en.wikipedia.org/wiki/Cross-correlation |
| `LiquidityDroughtDetector` | Relative volume/depth drought detector using lower-threshold hysteresis. | https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf |
| `LiquidityRegimeDetector` | EWMA Amihud-style liquidity regime detector using absolute return per dollar volume. | https://ba-odegaard.no/teach/notes/liquidity_estimators/amihud_estimator/amihud_lectures.pdf |
| `LinearRegression` | Rolling least-squares fitted value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope |
| `LinearRegressionAngle` | Angle of the rolling linear-regression slope. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope |
| `LinearRegressionIntercept` | Intercept of the rolling linear-regression fit. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/linear-regression-r2 |
| `LinearRegressionSlope` | Slope of the rolling linear-regression fit. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope |
| `Low` | Rolling lowest value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/lowest-low-value |
| `LowIndex` | Offset/index of the rolling lowest value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/distance-to-lows |
| `MACDFix` | MACD with fixed 12/26 moving-average periods. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator |
| `MassIndex` | Range-expansion reversal indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/mass-index |
| `MarketOpenCloseTransitionDetector` | Session-progress transition detector for market-open and market-close bands. | https://en.wikipedia.org/wiki/Trading_day |
| `MatchedFlowConformalSignal` | Intraday OHLCV matched-flow signal with conformal-style rolling error bands and target sizing diagnostics. | documentation/matched_flow_conformal_signal.md |
| `MedianPrice` | Average of high and low. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/median-price |
| `MesaAdaptiveMovingAverage` | Ehlers MAMA/FAMA adaptive moving averages driven by dominant cycle phase. | https://trendspider.com/learning-center/what-is-the-mesa-adaptive-moving-average-mama/ |
| `MicrostructureNoiseRegimeDetector` | EWMA trade-versus-mid noise detector normalized by quoted spread. | https://en.wikipedia.org/wiki/Market_microstructure |
| `MidPoint` | Midpoint of rolling high and low values for one series. | https://vectoralpha.dev/projects/ta/indicators/midpoint/ |
| `MidPrice` | Midpoint of rolling high and low price series. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/high-low-bands |
| `MinusDirectionalIndicator` | Negative directional indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `MinusDirectionalMovement` | Negative directional movement. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `Momentum` | Difference between current value and a prior value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum |
| `MoneyFlowIndex` | Volume-weighted RSI-like money flow oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/money-flow-index-mfi |
| `NadarayaWatsonEnvelope` | Gaussian-kernel Nadaraya-Watson smoother with weighted residual bands. | https://classic.d2l.ai/chapter_attention-mechanisms/nadaraya-watson.html |
| `NegativeVolumeIndex` | Cumulative indicator that changes on lower-volume periods. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/negative-volume-index-nvi |
| `NormalizedATR` | ATR normalized by close. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp |
| `OnBalanceVolume` | Cumulative volume added/subtracted by close direction. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/on-balance-volume-obv |
| `OnlineGaussianMixtureRegimeFilter` | Online Gaussian mixture regime filter with bounded component count. | https://en.wikipedia.org/wiki/Mixture_model |
| `OnlineHMMRegimeFilter` | Online Gaussian hidden Markov regime filter with fixed transition persistence. | https://en.wikipedia.org/wiki/Hidden_Markov_model |
| `OnlineMarkovSwitchingVolatilityFilter` | Online two-state Markov-switching volatility filter over close-to-close moves. | https://en.wikipedia.org/wiki/Markov-switching_model |
| `OrderFlowImbalance` | Quote-level best bid/ask price and size change pressure over a rolling update window. | https://arxiv.org/abs/1011.6402 |
| `OrderFlowImbalanceRegimeDetector` | EWMA order-flow imbalance regime detector with buy/sell pressure hysteresis. | https://arxiv.org/abs/1011.6402 |
| `PageHinkley` | Causal Page-Hinkley mean-shift event detector with directed up/down output. | https://menelaus.readthedocs.io/en/dev/menelaus.change_detection.html |
| `PairsSpreadRegimeDetector` | Streaming EWMA hedge-ratio residual z-score detector for pair-spread regimes. | https://en.wikipedia.org/wiki/Statistical_arbitrage |
| `ParticleFilterTrend` | Deterministic-seed particle trend filter with Laplace measurement likelihood and effective sample size output. | https://alphaarchitect.com/trend-following-filters-part-4/ |
| `ParabolicSAR` | Parabolic stop-and-reverse trailing trend indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/parabolic-sar |
| `PercentagePrice` | Percentage Price Oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/percentage-price-oscillator-ppo |
| `PercentageVolume` | Percentage Volume Oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/percentage-volume-oscillator-pvo |
| `PlusDirectionalIndicator` | Positive directional indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `PlusDirectionalMovement` | Positive directional movement. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `PredictionErrorDriftDetector` | EWMA absolute prediction-error drift detector. | https://en.wikipedia.org/wiki/Concept_drift |
| `QuoteMessageRateRegimeDetector` | Relative EWMA quote-message-rate regime detector. | https://en.wikipedia.org/wiki/Quote_stuffing |
| `QuoteStuffingDetector` | EWMA quote-to-trade message ratio detector for quote-stuffing episodes. | https://en.wikipedia.org/wiki/Quote_stuffing |
| `RateOfChangePercentage` | Period-over-period rate of change as a fraction. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum |
| `RateOfChangeRatio` | Rate-of-change ratio against a prior value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum |
| `RateOfChangeRatio100` | Rate-of-change ratio scaled by 100. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum |
| `RenkoBrickGenerator` | Event-driven Renko price transform that emits signed brick counts and current brick state from close updates. | https://www.tradingview.com/support/solutions/43000502284-understanding-renko-charts/ |
| `ResidualDriftDetector` | EWMA residual z-score drift detector with signed hysteresis output. | https://en.wikipedia.org/wiki/Concept_drift |
| `RelativeVigorIndex` | Smoothed close-open momentum relative to high-low range with signal line. | https://www.investopedia.com/terms/r/relative_vigor_index.asp |
| `RealizedVarianceRegimeDetector` | Rolling realized-variance regime detector from squared close-to-close changes. | https://en.wikipedia.org/wiki/Realized_variance |
| `RollingBetaShiftDetector` | Causal adjacent-window beta shift detector. | https://www.investopedia.com/terms/b/beta.asp |
| `RollingCorrelationShiftDetector` | Causal adjacent-window correlation shift detector. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/correlation-coefficient |
| `RollingMeanShiftDetector` | Causal adjacent-window mean shift detector using a two-sample z-score. | https://en.wikipedia.org/wiki/Student%27s_t-test |
| `RollingMeanVarianceShiftDetector` | Causal adjacent-window combined mean and variance shift detector. | https://en.wikipedia.org/wiki/Change_detection |
| `RollingSpreadLiquidityShiftDetector` | Causal adjacent-window quote spread/depth liquidity stress shift detector. | https://arxiv.org/abs/1011.6402 |
| `RollingVarianceShiftDetector` | Causal adjacent-window variance shift detector using log variance ratio. | https://en.wikipedia.org/wiki/F-test |
| `SavitzkyGolayFilter` | Rolling polynomial least-squares smoother with first and second derivative outputs. | https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter |
| `SchaffTrendCycle` | MACD/stochastic cycle oscillator. | https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html |
| `SpreadFeatures` | Quoted, effective, and realized spread estimates from trades and contemporaneous quotes. | https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf |
| `SpreadExplosionDetector` | EWMA relative quoted-spread explosion detector. | https://en.wikipedia.org/wiki/Bid%E2%80%93ask_spread |
| `SpreadRegimeDetector` | Stateful quoted-spread regime detector using relative bid/ask spread. | https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf |
| `StdDev` | Rolling standard deviation. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility |
| `StickyHMMRegimeFilter` | Online Gaussian HMM regime filter with high self-transition persistence. | https://en.wikipedia.org/wiki/Hidden_Markov_model |
| `StochRSI` | Stochastic oscillator applied to RSI values. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochrsi |
| `Stochastic` | Slow stochastic oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochastic-oscillator-fast-slow-and-full |
| `SuperTrend` | ATR-band trend-following indicator. | https://www.investopedia.com/supertrend-indicator-7976167 |
| `Summation` | Rolling sum. | https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.sum.html |
| `T3MovingAverage` | Tillson T3 multi-EMA moving average. | https://efs.kb.esignal.com/hc/en-us/articles/6362957784603-T3-Average |
| `ThresholdRegimeDetector` | Stateful threshold regime detector with upper/lower hysteresis bands. | https://en.wikipedia.org/wiki/Hysteresis |
| `TimeSeriesForecast` | Rolling linear-regression time-series forecast. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope |
| `TradeIntensityRegimeDetector` | EWMA relative trade-count intensity regime detector. | https://en.wikipedia.org/wiki/Market_microstructure |
| `TrendChopRegimeDetector` | Efficiency-ratio trend-versus-chop regime detector using true range. | https://www.angelone.in/knowledge-center/online-share-trading/choppiness-index-indicator |
| `TwoFactorKalmanTrendFilter` | Two-state short/long Kalman trend contribution model. | https://arxiv.org/pdf/1808.03297 |
| `TrueRange` | Maximum of high-low and gaps from previous close. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/true-range |
| `TriangularMovingAverage` | Double-smoothed triangular moving average. | https://www.marketvolume.com/technicalanalysis/tma.asp |
| `TripleEMA` | TEMA lag-reduced moving average. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/triple-exponential-moving-average-tema |
| `Trix` | Triple-smoothed rate-of-change oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/trix |
| `TypicalPrice` | Average of high, low, and close. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/typical-price |
| `UltimateOscillator` | Weighted multi-window buying-pressure oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ultimate-oscillator |
| `UlcerIndex` | Drawdown-based downside-risk measure. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ulcer-index |
| `VPIN` | Volume-synchronized probability of informed trading using bulk-volume classification and rolling volume-bucket imbalance. | https://www.quantresearch.org/VPIN.pdf |
| `Variance` | Rolling variance. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility |
| `VariableIndexDynamicAverage` | VIDYA adaptive EMA using absolute CMO as the smoothing factor. | https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/vida |
| `VolatilityBreakoutDetector` | EWMA z-score detector for unusually large close-to-close volatility breakouts. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility |
| `VolatilityCompressionExpansionDetector` | Short-versus-long EWMA volatility ratio detector for compression and expansion regimes. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility |
| `VolatilityRegimeDetector` | EWMA close-change volatility regime detector with high/low hysteresis bands. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility |
| `VolumeProfile` | Rolling volume-by-price histogram that emits point of control and value-area high/low levels. | https://www.schwab.com/learn/story/using-volume-profile-indicator |
| `VolumePriceTrend` | Cumulative volume adjusted by percentage price change. | https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html |
| `VolumeRegimeDetector` | EWMA relative volume regime detector with high/low hysteresis bands. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/on-balance-volume-obv |
| `VolumeWeightedAveragePrice` | VWAP price weighted by traded volume. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/volume-weighted-average-price-vwap |
| `VolumeWeightedMovingAverage` | VWMA rolling close weighted by volume. | https://trendspider.com/learning-center/what-is-the-volume-weighted-moving-average-vwma/ |
| `Vortex` | Positive/negative Vortex trend movement indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/vortex-indicator |
| `WeightedClosePrice` | Weighted close transform using high, low, and close. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/weighted-close |
| `WeightedMovingAverage` | Weighted moving average with larger recent weights. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/hull-moving-average-hma |
| `WilliamsR` | Williams %R overbought/oversold oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/williams-r |
| `ZigZagSwingDetector` | Close-based swing detector that filters price moves below a percentage threshold and emits confirmed pivots. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/zigzag |
