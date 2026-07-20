# Algorithm Documentation

This directory holds detailed Markdown source pages for RTTA's public technical
analysis algorithms. Each page documents the public `update(...)` shape, the
implemented theory of operation, and the C++-derived recurrence used to update
state one sample at a time.

| Algorithm | Summary |
|---|---|
| [`AbsolutePriceOscillator`](absolute-price-oscillator.md) | Difference between fast and slow moving averages in price units. |
| [`AccelerationBands`](acceleration-bands.md) | TA-Lib style acceleration bands: SMA of scaled high/low extremes with SMA middle. |
| [`AcceleratorOscillator`](accelerator-oscillator.md) | Bill Williams Accelerator Oscillator: Awesome Oscillator minus its SMA. |
| [`AccumulationDistribution`](accumulation-distribution.zh-CN.md) | Volume-price accumulation/distribution line. |
| [`AccumulativeSwingIndex`](accumulative-swing-index.md) | Cumulative sum of Wilder's Swing Index. |
| [`ADWIN`](adwin.md) | Adaptive-window mean drift detector with bounded history and directed shift output. |
| [`Alligator`](alligator.md) | Bill Williams Alligator jaw/teeth/lips as shifted SMMA of median price. |
| [`AlphaBetaGammaTrackingFilter`](alpha-beta-gamma-tracking-filter.md) | Steady-state Kalman-like price, velocity, and acceleration tracker. |
| [`AmihudIlliquidity`](amihud-illiquidity.zh-CN.md) | Rolling average absolute return per dollar of traded volume. |
| [`AnchoredVWAP`](anchored-vwap.md) | VWAP accumulated from arbitrary anchor/reset events rather than a fixed session or rolling window. |
| [`AndrewsPitchfork`](andrews-pitchfork.md) | Streaming Andrews Pitchfork median and parallel channels from percent swing pivots. |
| [`ArnaudLegouxMovingAverage`](arnaud-legoux-moving-average.md) | Arnaud Legoux moving average with Gaussian weights controlled by offset and sigma. |
| [`Aroon`](aroon.zh-CN.md) | Aroon Up/Down trend age indicators based on recent highs and lows. |
| [`AroonOscillator`](aroon-oscillator.md) | Difference between Aroon Up and Aroon Down. |
| [`ATR`](atr.md) | Average True Range volatility over a rolling window. |
| [`ATRP`](atrp.md) | Average True Range expressed as a percentage of price. |
| [`ATRRegimeDetector`](atr-regime-detector.zh-CN.md) | Stateful ATR regime detector with high/low hysteresis bands. |
| [`AuctionContinuousMarketTransitionDetector`](auction-continuous-market-transition-detector.zh-CN.md) | Hysteresis detector for auction-versus-continuous market phase signals. |
| [`AverageDirectionalMovementIndex`](average-directional-movement-index.md) | ADX trend-strength indicator. |
| [`AverageDirectionalMovementIndexRating`](average-directional-movement-index-rating.zh-CN.md) | ADXR smoothed ADX trend-strength rating. |
| [`AveragePrice`](average-price.md) | Average of open, high, low, and close. |
| [`AwesomeOscillator`](awesome-oscillator.zh-CN.md) | Difference between short and long median-price moving averages. |
| [`BalanceOfPower`](balance-of-power.zh-CN.md) | Open/high/low/close buying-versus-selling pressure measure. |
| [`BearsPower`](bears-power.md) | Elder bears power: low minus EMA of close. |
| [`Beta`](beta.md) | Rolling beta of one series against another. |
| [`BetaRegimeDetector`](beta-regime-detector.md) | Stateful rolling beta regime detector with upper/lower hysteresis bands. |
| [`Bias`](bias.md) | Percent deviation of price from its simple moving average. |
| [`BidAskBounceRegimeDetector`](bid-ask-bounce-regime-detector.md) | EWMA bid/ask side alternation detector for quote-bounce regimes. |
| [`BollingerBands`](bollinger-bands.zh-CN.md) | Moving-average envelope based on standard deviations. |
| [`BollingerBandwidth`](bollinger-bandwidth.md) | Bollinger band width as (upper-lower)/middle for a rolling mean and standard-deviation envelope. |
| [`BollingerPercentB`](bollinger-percent-b.md) | Bollinger %B position of price inside a rolling mean and standard-deviation envelope. |
| [`BoundedBOCPD`](bounded-bocpd.zh-CN.md) | Bounded-memory Bayesian online change-point detector with constant hazard. |
| [`BullsPower`](bulls-power.md) | Elder bulls power: high minus EMA of close. |
| [`CalibrationDriftDetector`](calibration-drift-detector.md) | EWMA probability-calibration error drift detector. |
| [`CamarillaPivotPoints`](camarilla-pivot-points.md) | Camarilla support/resistance pivots from the previous bar HLC. |
| [`ChaikinMoneyFlow`](chaikin-money-flow.zh-CN.md) | Volume-weighted money flow over a window. |
| [`ChaikinOscillator`](chaikin-oscillator.zh-CN.md) | MACD-style oscillator of the accumulation/distribution line. |
| [`ChaikinVolatility`](chaikin-volatility.md) | Percent rate-of-change of an EMA of the high-low range. |
| [`ChandeForecastOscillator`](chande-forecast-oscillator.md) | Chande forecast oscillator: percent distance of close from time-series forecast. |
| [`ChandelierExit`](chandelier-exit.md) | ATR trailing chandelier long and short exits from rolling high/low extremes. |
| [`ChandeMomentumOscillator`](chande-momentum-oscillator.zh-CN.md) | Momentum oscillator using sums of recent gains and losses. |
| [`ChoppinessIndex`](choppiness-index.md) | CHOP range/trend measure based on true range versus high-low range. |
| [`ClosePressureReversalSignal`](close-pressure-reversal-signal.md) | End-of-day cross-sectional reversal signal using rest-of-day return, volume/transaction pressure, VWAP location, and rolling conformal-style error bands. |
| [`CointegrationBreakdownMonitor`](cointegration-breakdown-monitor.md) | Streaming residual-z monitor for pair relationship breakdowns using an EWMA hedge estimate. |
| [`CommodityChannelIndex`](commodity-channel-index.zh-CN.md) | CCI deviation of typical price from its moving average. |
| [`ComparativeRelativeStrength`](comparative-relative-strength.md) | Ratio of two price series (A/B relative strength). |
| [`ConformalBands`](conformal-bands.md) | Streaming split-conformal style bands around an SMA center using rolling absolute residual quantiles. |
| [`ConnorsRSI`](connors-rsi.zh-CN.md) | Composite oscillator averaging price RSI, streak RSI, and percent-rank of one-period price change. |
| [`CoppockCurve`](coppock-curve.zh-CN.md) | Weighted moving average of long and short rate-of-change values. |
| [`Correlation`](correlation.zh-CN.md) | Rolling Pearson correlation between two series. |
| [`CorrelationRegimeDetector`](correlation-regime-detector.md) | Stateful rolling correlation regime detector with upper/lower hysteresis bands. |
| [`CrossAssetCorrelationBreakDetector`](cross-asset-correlation-break-detector.md) | Short-versus-long rolling correlation break detector for two assets. |
| [`CrossAssetOrderFlowImbalance`](cross-asset-order-flow-imbalance.md) | Rolling beta of own return on peer OFI with implied impact and residual. |
| [`CumulativeReturn`](cumulative-return.zh-CN.md) | Cumulative return from the first close. |
| [`CUSUM`](cusum.zh-CN.md) | Causal cumulative-sum event filter for detecting threshold-sized directional moves. |
| [`DailyLogReturn`](daily-log-return.md) | Log return between consecutive closes. |
| [`DailyReturn`](daily-return.md) | Percentage return between consecutive closes. |
| [`DDM`](ddm.md) | Drift Detection Method for Bernoulli prediction-error streams. |
| [`DecomposedOrderFlowImbalance`](decomposed-order-flow-imbalance.md) | Quote-driven add/cancel/trade decomposition of Cont-style order-flow pressure. |
| [`Delay`](delay.zh-CN.md) | Lagged value from a fixed number of samples ago. |
| [`DeMarker`](de-marker.md) | DeMarker oscillator from rolling up-high and down-low pressure. |
| [`DetrendedPriceOscillator`](detrended-price-oscillator.zh-CN.md) | DPO cycle indicator comparing price to a displaced average. |
| [`DirectionalChangeDetector`](directional-change-detector.md) | Directional-change intrinsic-time events with overshoot tracking. |
| [`DirectionalMovementIndex`](directional-movement-index.zh-CN.md) | DX directional movement trend-strength indicator. |
| [`DollarBarGenerator`](dollar-bar-generator.md) | Information-driven dollar bars that close when price×volume accumulates to a threshold. |
| [`DollarRunBarGenerator`](dollar-run-bar-generator.md) | Same-sign dollar run bars that close when cumulative |price|×volume hits a threshold. |
| [`DonchianChannel`](donchian-channel.md) | Channel from rolling highest high and lowest low. |
| [`DoubleEMA`](double-ema.zh-CN.md) | DEMA lag-reduced moving average. |
| [`EaseOfMovement`](ease-of-movement.zh-CN.md) | Volume/range indicator for ease of price movement. |
| [`EDDM`](eddm.zh-CN.md) | Early Drift Detection Method using distances between prediction errors. |
| [`EfficiencyRatio`](efficiency-ratio.md) | Kaufman efficiency ratio of net directional move to path length over a rolling window. |
| [`EhlersCenterOfGravity`](ehlers-center-of-gravity.md) | Ehlers center-of-gravity oscillator of a rolling price window with lag trigger. |
| [`EhlersCyberCycle`](ehlers-cyber-cycle.md) | Ehlers cyber cycle band-pass style cycle oscillator with trigger. |
| [`EhlersDecycler`](ehlers-decycler.md) | Ehlers decycler trend estimate and residual oscillator. |
| [`EhlersInstantaneousTrendline`](ehlers-instantaneous-trendline.md) | Ehlers instantaneous trendline with two-bar trigger. |
| [`EhlersOptimalTrackingFilter`](ehlers-optimal-tracking-filter.zh-CN.md) | Adaptive tracking filter using Ehlers' price uncertainty tracking index. |
| [`EhlersRoofingFilter`](ehlers-roofing-filter.md) | Ehlers roofing filter: high-pass plus Super Smoother low-pass. |
| [`EhlersSuperSmoother`](ehlers-super-smoother.md) | Ehlers two-pole Super Smoother low-pass filter. |
| [`ElderRayIndex`](elder-ray-index.zh-CN.md) | Bull and bear power as high/low distance from an EMA of close. |
| [`ElderThermometer`](elder-thermometer.md) | Elder bar-range thermometer: current range vs previous range (ratio and hot flag). |
| [`EMA`](ema.md) | Exponential moving average with more weight on recent samples. |
| [`EWMA`](ewma.zh-CN.md) | Exponentially weighted moving average parameterized by alpha/span/com. |
| [`EWMAZScoreShiftDetector`](ewmaz-score-shift-detector.zh-CN.md) | Causal EWMA mean/variance z-score event detector for threshold-sized shifts. |
| [`ExecutionCostSlippageRegimeDetector`](execution-cost-slippage-regime-detector.zh-CN.md) | Stateful relative execution-cost/slippage regime detector from trade price versus quote mid. |
| [`FastStochastic`](fast-stochastic.zh-CN.md) | Fast stochastic %K/%D oscillator. |
| [`FeatureDistributionDriftDetector`](feature-distribution-drift-detector.zh-CN.md) | Bounded ADWIN-style drift detector for a single streaming feature distribution. |
| [`FibonacciPivotPoints`](fibonacci-pivot-points.md) | Fibonacci support/resistance pivots from the previous bar HLC. |
| [`FibonacciRetracementLevels`](fibonacci-retracement-levels.zh-CN.md) | Rolling Fibonacci retracement levels between recent high and low. |
| [`FisherTransform`](fisher-transform.md) | Ehlers transform of normalized recent high/low position into a turning-point oscillator. |
| [`FlowPressureCapacitySignal`](flow-pressure-capacity-signal.md) | Event-time aggressive flow versus opposing L1 capacity, corrected for queue replenishment, withdrawal fragility, and transient imbalance. |
| [`FOCuS`](focus.md) | Functional online CUSUM mean changepoint detector with candidate pruning (Romano et al.). |
| [`ForceIndex`](force-index.zh-CN.md) | Price-change times volume oscillator. |
| [`FourierResidueIdentity`](fourier-residue-identity.md) | Fourier-Residue Identity splitting return autocorrelation into testable direction (sign, k=2) and magnitude (k=4) channels, with Fejér variance ratios per channel. |
| [`FractalAdaptiveMovingAverage`](fractal-adaptive-moving-average.zh-CN.md) | Ehlers FRAMA using fractal dimension to adapt EMA smoothing. |
| [`GatorOscillator`](gator-oscillator.md) | Bill Williams Gator Oscillator from Alligator jaw-teeth and teeth-lips gaps. |
| [`GaussianProcessRegressionBands`](gaussian-process-regression-bands.md) | Rolling RBF-kernel Gaussian process posterior mean with uncertainty bands. |
| [`GeometricMovingAverage`](geometric-moving-average.md) | Geometric moving average as exp of SMA of log prices. |
| [`GuppyMMARibbon`](guppy-mma-ribbon.md) | Full Guppy MMA ribbon of six short and six long EMAs plus group averages. |
| [`GuppyMultipleMovingAverage`](guppy-multiple-moving-average.md) | Guppy MMA short and long EMA-group averages and spread. |
| [`HawkesIntensity`](hawkes-intensity.md) | Exponential Hawkes self-exciting intensity process for event times. |
| [`HDDM`](hddm.md) | Hoeffding-bound drift detector for Bernoulli prediction-error streams. |
| [`HeikinAshiTransform`](heikin-ashi-transform.md) | Incremental Heikin-Ashi OHLC transform for smoothing candles. |
| [`HiddenSemiMarkovRegimeFilter`](hidden-semi-markov-regime-filter.md) | Online Gaussian hidden semi-Markov-style regime filter with bounded duration bias. |
| [`High`](high.md) | Rolling highest value. |
| [`HighIndex`](high-index.zh-CN.md) | Offset/index of the rolling highest value. |
| [`HighLow`](high-low.zh-CN.md) | Combined rolling minimum and maximum values. |
| [`HighLowIndex`](high-low-index.zh-CN.md) | Combined offsets/indexes of rolling minimum and maximum values. |
| [`HilbertDominantCyclePeriod`](hilbert-dominant-cycle-period.md) | TA-Lib HT_DCPERIOD-compatible dominant cycle period. |
| [`HilbertDominantCyclePhase`](hilbert-dominant-cycle-phase.md) | TA-Lib HT_DCPHASE-compatible dominant cycle phase in degrees. |
| [`HilbertPhasor`](hilbert-phasor.md) | TA-Lib HT_PHASOR-compatible in-phase and quadrature components. |
| [`HilbertSineWave`](hilbert-sine-wave.md) | TA-Lib HT_SINE-compatible sine and lead-sine waves. |
| [`HilbertTrendline`](hilbert-trendline.md) | TA-Lib HT_TRENDLINE-compatible instantaneous trendline. |
| [`HilbertTrendMode`](hilbert-trend-mode.md) | TA-Lib HT_TRENDMODE-compatible trend-versus-cycle mode flag (1=trend, 0=cycle). |
| [`HistoricalVolatility`](historical-volatility.md) | Annualized rolling standard deviation of log returns. |
| [`HitRateDriftDetector`](hit-rate-drift-detector.md) | EWMA hit-rate degradation detector using miss-rate hysteresis. |
| [`HullMovingAverage`](hull-moving-average.md) | HMA lag-reduced weighted moving average. |
| [`Ichimoku`](ichimoku.zh-CN.md) | Ichimoku conversion, base, leading spans, lagging span, and displaced cloud spans. |
| [`ImbalanceBarGenerator`](imbalance-bar-generator.md) | Volume-imbalance bars that close when |signed volume| hits a threshold (tick-rule signs). |
| [`Inertia`](inertia.md) | Dorsey inertia: linear regression of relative volatility index. |
| [`IntegratedOrderFlowImbalance`](integrated-order-flow-imbalance.md) | Multi-level Cont OFI projected onto an online first principal component. |
| [`InteractingMultipleModelFilter`](interacting-multiple-model-filter.zh-CN.md) | Four-regime IMM Kalman tracker that blends low-volatility, high-volatility, trend, and chop models by online probabilities. |
| [`IntradayClockEchoSignal`](intraday-clock-echo-signal.zh-CN.md) | Same-clock intraday return-periodicity signal trained from prior aggregate-bar day lists. |
| [`IntradayIntensity`](intraday-intensity.md) | Rolling volume-weighted intraday intensity (2C-H-L)/(H-L). |
| [`IntradayMomentumIndex`](intraday-momentum-index.md) | RSI-style oscillator of open-to-close gains versus losses within each bar. |
| [`InverseFisherRSI`](inverse-fisher-rsi.md) | Inverse Fisher transform of RSI for sharper turning points. |
| [`KagiChart`](kagi-chart.md) | Streaming Kagi line, direction, and reversal events. |
| [`KalmanExtremumTrend`](kalman-extremum-trend.md) | Kalman trend combined with stochastic-style position inside recent extrema. |
| [`KalmanHedgeRatio`](kalman-hedge-ratio.zh-CN.md) | Online Kalman regression hedge ratio and pair spread. |
| [`KalmanInnovationResidualBOCPD`](kalman-innovation-residual-bocpd.md) | Kalman innovation z-score residual piped into ResidualBOCPD changepoint detection. |
| [`KalmanInnovationResidualFOCuS`](kalman-innovation-residual-focus.md) | Kalman innovation z-score residual piped into FOCuS changepoint detection. |
| [`KalmanInnovationZScore`](kalman-innovation-z-score.zh-CN.md) | Signed measurement innovation normalized by the predicted innovation standard deviation. |
| [`KalmanLocalLinearTrend`](kalman-local-linear-trend.md) | Kalman local level/trend state-space estimator. |
| [`KalmanMovingAverage`](kalman-moving-average.zh-CN.md) | Kalman price filter using a local linear price/velocity model. |
| [`KalmanPredictionBands`](kalman-prediction-bands.zh-CN.md) | One-step Kalman prediction with upper/lower bands from predicted measurement uncertainty. |
| [`KalmanRegressionChannel`](kalman-regression-channel.md) | Online Kalman regression with prediction channel and spread. |
| [`KalmanTrendSignal`](kalman-trend-signal.md) | Kalman-filtered trend line with buy/sell signal based on price versus filtered trend. |
| [`KalmanVelocityOscillator`](kalman-velocity-oscillator.md) | Zero-centered velocity state from a constant-velocity Kalman price model. |
| [`Kama`](kama.zh-CN.md) | Kaufman Adaptive Moving Average. |
| [`KeltnerChannel`](keltner-channel.md) | EMA/ATR volatility channel. |
| [`KeltnerChannelOriginal`](keltner-channel-original.md) | Original SMA/range Keltner channel variant. |
| [`KlingerVolumeOscillator`](klinger-volume-oscillator.zh-CN.md) | Volume-force oscillator using fast and slow EMAs plus signal line. |
| [`KSTOscillator`](kst-oscillator.zh-CN.md) | Pring Know Sure Thing smoothed multi-ROC oscillator. |
| [`KSWIN`](kswin.zh-CN.md) | Kolmogorov-Smirnov sliding-window drift detector. |
| [`KyleLambda`](kyle-lambda.zh-CN.md) | Rolling price-impact slope of returns against signed square-root dollar volume. |
| [`LeadLagRegimeDetector`](lead-lag-regime-detector.zh-CN.md) | EWMA cross-lag detector for which of two series is leading. |
| [`LinearRegression`](linear-regression.md) | Rolling least-squares fitted value. |
| [`LinearRegressionAngle`](linear-regression-angle.md) | Angle of the rolling linear-regression slope. |
| [`LinearRegressionIntercept`](linear-regression-intercept.md) | Intercept of the rolling linear-regression fit. |
| [`LinearRegressionSlope`](linear-regression-slope.md) | Slope of the rolling linear-regression fit. |
| [`LiquidityDroughtDetector`](liquidity-drought-detector.md) | Relative volume/depth drought detector using lower-threshold hysteresis. |
| [`LiquidityRegimeDetector`](liquidity-regime-detector.md) | EWMA Amihud-style liquidity regime detector using absolute return per dollar volume. |
| [`Low`](low.zh-CN.md) | Rolling lowest value. |
| [`LowIndex`](low-index.zh-CN.md) | Offset/index of the rolling lowest value. |
| [`MACD`](macd.md) | MACD line, signal, and histogram from fast/slow EMA difference. |
| [`MACDExt`](macd-ext.md) | MACD with selectable SMA/EMA types for fast, slow, and signal. |
| [`MACDFix`](macd-fix.md) | Fixed 12/26 MACD with multi-output line, signal, and histogram. |
| [`MarketFacilitationIndex`](market-facilitation-index.md) | Bar range divided by volume (Bill Williams MFI). |
| [`MarketOpenCloseTransitionDetector`](market-open-close-transition-detector.md) | Session-progress transition detector for market-open and market-close bands. |
| [`MassIndex`](mass-index.zh-CN.md) | Range-expansion reversal indicator. |
| [`MatchedFlowConformalSignal`](matched-flow-conformal-signal.md) | Intraday OHLCV matched-flow signal with conformal-style rolling error bands and target sizing diagnostics. |
| [`McGinleyDynamic`](mc-ginley-dynamic.md) | McGinley Dynamic adaptive moving average that speeds up in trends and slows in chop. |
| [`MedianPrice`](median-price.md) | Average of high and low. |
| [`MesaAdaptiveMovingAverage`](mesa-adaptive-moving-average.zh-CN.md) | Ehlers MAMA/FAMA adaptive moving averages driven by dominant cycle phase. |
| [`MessageEventOrderFlowImbalance`](message-event-order-flow-imbalance.md) | Rolling OFI accumulated from discrete LOB/trade message events (add/cancel/trade). |
| [`MicrostructureNoiseRegimeDetector`](microstructure-noise-regime-detector.md) | EWMA trade-versus-mid noise detector normalized by quoted spread. |
| [`MidPoint`](mid-point.md) | Midpoint of rolling high and low values for one series. |
| [`MidPrice`](mid-price.zh-CN.md) | Midpoint of rolling high and low price series. |
| [`MinusDirectionalIndicator`](minus-directional-indicator.zh-CN.md) | Negative directional indicator. |
| [`MinusDirectionalMovement`](minus-directional-movement.md) | Negative directional movement. |
| [`Momentum`](momentum.md) | Difference between current value and a prior value. |
| [`MoneyFlowIndex`](money-flow-index.zh-CN.md) | Volume-weighted RSI-like money flow oscillator. |
| [`MovingAverageEnvelope`](moving-average-envelope.md) | Percentage envelope bands above and below a simple moving average. |
| [`MovingAverageVariablePeriod`](moving-average-variable-period.md) | SMA with a per-bar variable period (TA-Lib MAVP-style). |
| [`MultiLevelOrderFlowImbalance`](multi-level-order-flow-imbalance.md) | Cont-style order-flow imbalance at each book level with sum/mean aggregates. |
| [`MultiPeerOrderFlowImbalance`](multi-peer-order-flow-imbalance.md) | Basket peer OFI (equal-weight mean) with rolling beta impact on own return. |
| [`NadarayaWatsonEnvelope`](nadaraya-watson-envelope.md) | Gaussian-kernel Nadaraya-Watson smoother with weighted residual bands. |
| [`NegativeVolumeIndex`](negative-volume-index.md) | Cumulative indicator that changes on lower-volume periods. |
| [`NormalizedATR`](normalized-atr.md) | ATR normalized by close. |
| [`OnBalanceVolume`](on-balance-volume.md) | Cumulative volume added/subtracted by close direction. |
| [`OnlineGaussianMixtureRegimeFilter`](online-gaussian-mixture-regime-filter.md) | Online Gaussian mixture regime filter with bounded component count. |
| [`OnlineHMMRegimeFilter`](online-hmm-regime-filter.zh-CN.md) | Online Gaussian hidden Markov regime filter with fixed transition persistence. |
| [`OnlineMarkovSwitchingVolatilityFilter`](online-markov-switching-volatility-filter.zh-CN.md) | Online two-state Markov-switching volatility filter over close-to-close moves. |
| [`OrderFlowImbalance`](order-flow-imbalance.zh-CN.md) | Quote-level best bid/ask price and size change pressure over a rolling update window. |
| [`OrderFlowImbalanceRegimeDetector`](order-flow-imbalance-regime-detector.zh-CN.md) | EWMA order-flow imbalance regime detector with buy/sell pressure hysteresis. |
| [`PageHinkley`](page-hinkley.zh-CN.md) | Causal Page-Hinkley mean-shift event detector with directed up/down output. |
| [`PairsSpreadRegimeDetector`](pairs-spread-regime-detector.md) | Streaming EWMA hedge-ratio residual z-score detector for pair-spread regimes. |
| [`ParabolicSAR`](parabolic-sar.md) | Parabolic stop-and-reverse trailing trend indicator. |
| [`ParabolicSARExtended`](parabolic-sar-extended.md) | Extended parabolic SAR with separate long/short AF chains (SAREXT-style). |
| [`ParticleFilterTrend`](particle-filter-trend.zh-CN.md) | Deterministic-seed particle trend filter with Laplace measurement likelihood and effective sample size output. |
| [`PercentagePrice`](percentage-price.md) | Percentage Price Oscillator. |
| [`PercentageVolume`](percentage-volume.md) | Percentage Volume Oscillator. |
| [`PivotPoints`](pivot-points.md) | Classic floor pivot points (PP/R1-R3/S1-S3) from the previous bar. |
| [`PlusDirectionalIndicator`](plus-directional-indicator.md) | Positive directional indicator. |
| [`PlusDirectionalMovement`](plus-directional-movement.md) | Positive directional movement. |
| [`PointAndFigure`](point-and-figure.md) | Streaming point-and-figure box price, direction, and reversals. |
| [`PositiveVolumeIndex`](positive-volume-index.md) | Cumulative indicator that changes on higher-volume periods. |
| [`PredictionErrorDriftDetector`](prediction-error-drift-detector.md) | EWMA absolute prediction-error drift detector. |
| [`PrettyGoodOscillator`](pretty-good-oscillator.md) | Close minus SMA normalized by ATR (PGO). |
| [`ProjectionOscillator`](projection-oscillator.md) | Stochastic-style oscillator of close within linear-regression projection bands of high and low. |
| [`PsychologicalLine`](psychological-line.md) | Percent of up-closes over a rolling window. |
| [`QStick`](q-stick.md) | Simple moving average of close minus open. |
| [`QuoteMessageRateRegimeDetector`](quote-message-rate-regime-detector.zh-CN.md) | Relative EWMA quote-message-rate regime detector. |
| [`QuoteStuffingDetector`](quote-stuffing-detector.zh-CN.md) | EWMA quote-to-trade message ratio detector for quote-stuffing episodes. |
| [`RainbowMovingAverage`](rainbow-moving-average.md) | Mel Widner rainbow: recursive SMA layers with outer/high/low/mid/width. |
| [`RainbowOscillator`](rainbow-oscillator.md) | Rainbow oscillator: percent width and position of recursive SMA layers. |
| [`RandomWalkIndex`](random-walk-index.md) | Random walk index high/low relative to ATR-scaled range. |
| [`RangeActionVerificationIndex`](range-action-verification-index.md) | RAVI: absolute short-vs-long SMA gap as a percent of the long SMA. |
| [`RateOfChangePercentage`](rate-of-change-percentage.md) | Period-over-period rate of change as a fraction. |
| [`RateOfChangeRatio`](rate-of-change-ratio.zh-CN.md) | Rate-of-change ratio against a prior value. |
| [`RateOfChangeRatio100`](rate-of-change-ratio-100.md) | Rate-of-change ratio scaled by 100. |
| [`RealizedVarianceRegimeDetector`](realized-variance-regime-detector.md) | Rolling realized-variance regime detector from squared close-to-close changes. |
| [`RelativeVigorIndex`](relative-vigor-index.md) | Smoothed close-open momentum relative to high-low range with signal line. |
| [`RelativeVolatilityIndex`](relative-volatility-index.md) | RSI-style relative volatility index on rolling close stddev. |
| [`RenkoBrickGenerator`](renko-brick-generator.md) | Event-driven Renko price transform that emits signed brick counts and current brick state from close updates. |
| [`ResidualBOCPD`](residual-bocpd.md) | Bounded BOCPD changepoint detector applied to residual/innovation series. |
| [`ResidualDriftDetector`](residual-drift-detector.md) | EWMA residual z-score drift detector with signed hysteresis output. |
| [`ResidualFOCuS`](residual-focus.md) | FOCuS applied to residual/innovation series for model-based changepoint detection. |
| [`ROC`](roc.md) | Rate of Change momentum as percentage change over a lookback. |
| [`RollingBetaShiftDetector`](rolling-beta-shift-detector.md) | Causal adjacent-window beta shift detector. |
| [`RollingCorrelationShiftDetector`](rolling-correlation-shift-detector.md) | Causal adjacent-window correlation shift detector. |
| [`RollingMeanShiftDetector`](rolling-mean-shift-detector.md) | Causal adjacent-window mean shift detector using a two-sample z-score. |
| [`RollingMeanVarianceShiftDetector`](rolling-mean-variance-shift-detector.zh-CN.md) | Causal adjacent-window combined mean and variance shift detector. |
| [`RollingMedian`](rolling-median.md) | Rolling median of a price window. |
| [`RollingSpreadLiquidityShiftDetector`](rolling-spread-liquidity-shift-detector.zh-CN.md) | Causal adjacent-window quote spread/depth liquidity stress shift detector. |
| [`RollingVarianceShiftDetector`](rolling-variance-shift-detector.md) | Causal adjacent-window variance shift detector using log variance ratio. |
| [`RSI`](rsi.md) | Relative Strength Index momentum oscillator. |
| [`RunBarGenerator`](run-bar-generator.md) | Tick run bars that close after consecutive same-sign ticks reach a threshold. |
| [`SavitzkyGolayFilter`](savitzky-golay-filter.zh-CN.md) | Rolling polynomial least-squares smoother with first and second derivative outputs. |
| [`SchaffTrendCycle`](schaff-trend-cycle.zh-CN.md) | MACD/stochastic cycle oscillator. |
| [`SMA`](sma.zh-CN.md) | Simple moving average over a rolling window. |
| [`SmoothedMovingAverage`](smoothed-moving-average.md) | Wilder/SMMA/RMA smoothed moving average seeded by an initial SMA window. |
| [`SpreadExplosionDetector`](spread-explosion-detector.md) | EWMA relative quoted-spread explosion detector. |
| [`SpreadFeatures`](spread-features.md) | Quoted, effective, and realized spread estimates from trades and contemporaneous quotes. |
| [`SpreadRegimeDetector`](spread-regime-detector.zh-CN.md) | Stateful quoted-spread regime detector using relative bid/ask spread. |
| [`SqueezeMomentum`](squeeze-momentum.md) | TTM-style Bollinger-inside-Keltner squeeze flag with linreg momentum. |
| [`StdDev`](std-dev.zh-CN.md) | Rolling standard deviation. |
| [`StickyHMMRegimeFilter`](sticky-hmm-regime-filter.zh-CN.md) | Online Gaussian HMM regime filter with high self-transition persistence. |
| [`Stochastic`](stochastic.md) | Slow stochastic oscillator. |
| [`StochasticMomentumIndex`](stochastic-momentum-index.md) | Double-smoothed stochastic momentum index with signal line. |
| [`StochRSI`](stoch-rsi.md) | Stochastic oscillator applied to RSI values. |
| [`Summation`](summation.zh-CN.md) | Rolling sum. |
| [`SuperTrend`](super-trend.zh-CN.md) | ATR-band trend-following indicator. |
| [`SwingIndex`](swing-index.md) | Wilder's Swing Index of bar-to-bar price action. |
| [`T3MovingAverage`](t-3-moving-average.zh-CN.md) | Tillson T3 multi-EMA moving average. |
| [`ThresholdRegimeDetector`](threshold-regime-detector.zh-CN.md) | Stateful threshold regime detector with upper/lower hysteresis bands. |
| [`TimeSeriesForecast`](time-series-forecast.zh-CN.md) | Rolling linear-regression time-series forecast. |
| [`TradeIntensityRegimeDetector`](trade-intensity-regime-detector.md) | EWMA relative trade-count intensity regime detector. |
| [`TrendChopRegimeDetector`](trend-chop-regime-detector.md) | Efficiency-ratio trend-versus-chop regime detector using true range. |
| [`TrendIntensityIndex`](trend-intensity-index.md) | Percent of positive deviations from SMA over absolute deviations. |
| [`TriangularMovingAverage`](triangular-moving-average.md) | Double-smoothed triangular moving average. |
| [`TripleEMA`](triple-ema.zh-CN.md) | TEMA lag-reduced moving average. |
| [`Trix`](trix.md) | Triple-smoothed rate-of-change oscillator. |
| [`TrueRange`](true-range.md) | Maximum of high-low and gaps from previous close. |
| [`TSI`](tsi.zh-CN.md) | True Strength Index double-smoothed momentum oscillator. |
| [`TwiggsMoneyFlow`](twiggs-money-flow.md) | Twiggs money flow using true high/low and EMA volume normalization. |
| [`TwoFactorKalmanTrendFilter`](two-factor-kalman-trend-filter.md) | Two-state short/long Kalman trend contribution model. |
| [`TypicalPrice`](typical-price.zh-CN.md) | Average of high, low, and close. |
| [`UlcerIndex`](ulcer-index.md) | Drawdown-based downside-risk measure. |
| [`UltimateOscillator`](ultimate-oscillator.zh-CN.md) | Weighted multi-window buying-pressure oscillator. |
| [`VariableIndexDynamicAverage`](variable-index-dynamic-average.md) | VIDYA adaptive EMA using absolute CMO as the smoothing factor. |
| [`Variance`](variance.zh-CN.md) | Rolling variance. |
| [`VerticalHorizontalFilter`](vertical-horizontal-filter.md) | Trend strength as net move over path length (VHF). |
| [`VolatilityBreakoutDetector`](volatility-breakout-detector.zh-CN.md) | EWMA z-score detector for unusually large close-to-close volatility breakouts. |
| [`VolatilityCompressionExpansionDetector`](volatility-compression-expansion-detector.md) | Short-versus-long EWMA volatility ratio detector for compression and expansion regimes. |
| [`VolatilityRegimeDetector`](volatility-regime-detector.zh-CN.md) | EWMA close-change volatility regime detector with high/low hysteresis bands. |
| [`VolumeBarGenerator`](volume-bar-generator.md) | Information-driven volume bars that close when traded volume hits a threshold. |
| [`VolumeOscillator`](volume-oscillator.md) | Percent difference between short and long simple moving averages of volume. |
| [`VolumePriceTrend`](volume-price-trend.zh-CN.md) | Cumulative volume adjusted by percentage price change. |
| [`VolumeProfile`](volume-profile.zh-CN.md) | Rolling volume-by-price histogram that emits point of control and value-area high/low levels. |
| [`VolumeRegimeDetector`](volume-regime-detector.zh-CN.md) | EWMA relative volume regime detector with high/low hysteresis bands. |
| [`VolumeRunBarGenerator`](volume-run-bar-generator.md) | Same-sign volume run bars that close when cumulative volume hits a threshold. |
| [`VolumeWeightedAveragePrice`](volume-weighted-average-price.md) | VWAP price weighted by traded volume. |
| [`VolumeWeightedMovingAverage`](volume-weighted-moving-average.md) | VWMA rolling close weighted by volume. |
| [`Vortex`](vortex.md) | Positive/negative Vortex trend movement indicator. |
| [`VPIN`](vpin.md) | Volume-synchronized probability of informed trading using bulk-volume classification and rolling volume-bucket imbalance. |
| [`WaveTrend`](wave-trend.md) | LazyBear WaveTrend oscillator (wt1/wt2) on HLC3. |
| [`WeightedClosePrice`](weighted-close-price.zh-CN.md) | Weighted close transform using high, low, and close. |
| [`WeightedMovingAverage`](weighted-moving-average.zh-CN.md) | Weighted moving average with larger recent weights. |
| [`WeightedMultiPeerOrderFlowImbalance`](weighted-multi-peer-order-flow-imbalance.md) | Basket peer OFI with explicit peer weights and rolling beta impact on own return. |
| [`WilliamsAD`](williams-ad.md) | Williams accumulation/distribution cumulative line. |
| [`WilliamsFractals`](williams-fractals.md) | 5-bar Williams up/down fractal pivots with confirmation lag. |
| [`WilliamsR`](williams-r.md) | Williams %R overbought/oversold oscillator. |
| [`WoodiePivotPoints`](woodie-pivot-points.md) | Woodie floor pivots from previous bar H + L + 2C. |
| [`ZeroLagEMA`](zero-lag-ema.md) | Zero-lag exponential moving average using de-lagged price into an EMA. |
| [`ZigZagSwingDetector`](zig-zag-swing-detector.md) | Close-based swing detector that filters price moves below a percentage threshold and emits confirmed pivots. |

## Candlestick (CDL) patterns

Overview: [cdl-patterns.md](cdl-patterns.md).

| Algorithm | Description |
| --- | --- |
| [`CDL3BlackCrows`](cdl-3-black-crows.md) | Three black crows: three declining bearish bodies. |
| [`CDL3Inside`](cdl-3-inside.md) | Three inside up/down: harami plus confirmation bar. |
| [`CDL3Outside`](cdl-3-outside.md) | Three outside up/down: engulfing plus confirmation bar. |
| [`CDL3WhiteSoldiers`](cdl-3-white-soldiers.md) | Three white soldiers: three advancing bullish bodies. |
| [`CDLBeltHold`](cdl-belt-hold.md) | Belt hold: opens at extreme and posts a long body. |
| [`CDLClosingMarubozu`](cdl-closing-marubozu.md) | Closing marubozu: body closes at/near the extreme of the bar. |
| [`CDLCounterAttack`](cdl-counter-attack.md) | Counterattack: opposite long body closing near prior close. |
| [`CDLDarkCloudCover`](cdl-dark-cloud-cover.md) | Dark cloud cover: bearish close through midpoint of prior bull bar. |
| [`CDLDoji`](cdl-doji.md) | Doji candlestick: very small real body relative to range (indecision). |
| [`CDLDojiStar`](cdl-doji-star.md) | Doji star: doji after a long body (reversal risk). |
| [`CDLDragonflyDoji`](cdl-dragonfly-doji.md) | Dragonfly doji: doji with long lower shadow (bullish rejection). |
| [`CDLEngulfing`](cdl-engulfing.md) | Engulfing: current body fully engulfs prior body (reversal). |
| [`CDLEveningDojiStar`](cdl-evening-doji-star.md) | Evening doji star: evening star with doji middle bar. |
| [`CDLEveningStar`](cdl-evening-star.md) | Evening star: three-bar bearish reversal. |
| [`CDLGravestoneDoji`](cdl-gravestone-doji.md) | Gravestone doji: doji with long upper shadow (bearish rejection). |
| [`CDLHammer`](cdl-hammer.md) | Hammer: long lower shadow in a downtrend (bullish). |
| [`CDLHangingMan`](cdl-hanging-man.md) | Hanging man: hammer shape in an uptrend (bearish). |
| [`CDLHarami`](cdl-harami.md) | Harami: small body inside prior body (reversal/pause). |
| [`CDLHaramiCross`](cdl-harami-cross.md) | Harami cross: doji inside prior body. |
| [`CDLHighWave`](cdl-high-wave.md) | High-wave candle: tiny body with very long shadow(s). |
| [`CDLInvertedHammer`](cdl-inverted-hammer.md) | Inverted hammer: long upper shadow in a downtrend (bullish). |
| [`CDLLongLeggedDoji`](cdl-long-legged-doji.md) | Long-legged doji: doji with long upper and lower shadows. |
| [`CDLLongLine`](cdl-long-line.md) | Long line: large real body relative to recent average body. |
| [`CDLMarubozu`](cdl-marubozu.md) | Marubozu: range dominated by real body (strong directional bar). |
| [`CDLMatchingLow`](cdl-matching-low.md) | Matching low: two bear bars with matching closes (support). |
| [`CDLMorningDojiStar`](cdl-morning-doji-star.md) | Morning doji star: morning star with doji middle bar. |
| [`CDLMorningStar`](cdl-morning-star.md) | Morning star: three-bar bullish reversal. |
| [`CDLPatternPack`](cdl-pattern-pack.md) | Multi-output pack of common CDL patterns from one OHLC update. |
| [`CDLPiercing`](cdl-piercing.md) | Piercing line: bullish close through midpoint of prior bear bar. |
| [`CDLShootingStar`](cdl-shooting-star.md) | Shooting star: long upper shadow in an uptrend (bearish). |
| [`CDLShortLine`](cdl-short-line.md) | Short line: small real body relative to recent average body. |
| [`CDLSpinningTop`](cdl-spinning-top.md) | Spinning top: small body with upper and lower shadows (indecision). |
| [`CDLTriStar`](cdl-tri-star.md) | Tri-star: three consecutive dojis (reversal). |
