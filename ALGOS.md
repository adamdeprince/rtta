# Algorithms

This file lists the public indicator algorithms exported by `rtta`. Tuning/result helper types are intentionally omitted. Detailed implementation notes and external references live in per-algorithm pages under `documentation/algorithms/`.

| Algorithm | Description |
|---|---|
| [`AbsolutePriceOscillator`](documentation/algorithms/absolute-price-oscillator.md) | Difference between fast and slow moving averages in price units. |
| [`AccelerationBands`](documentation/algorithms/acceleration-bands.md) | TA-Lib style acceleration bands: SMA of scaled high/low extremes with SMA middle. |
| [`AcceleratorOscillator`](documentation/algorithms/accelerator-oscillator.md) | Bill Williams Accelerator Oscillator: Awesome Oscillator minus its SMA. |
| [`AccumulationDistribution`](documentation/algorithms/accumulation-distribution.zh-CN.md) | Volume-price accumulation/distribution line. |
| [`AccumulativeSwingIndex`](documentation/algorithms/accumulative-swing-index.md) | Cumulative sum of Wilder's Swing Index. |
| [`ADWIN`](documentation/algorithms/adwin.md) | Adaptive-window mean drift detector with bounded history and directed shift output. |
| [`Alligator`](documentation/algorithms/alligator.md) | Bill Williams Alligator jaw/teeth/lips as shifted SMMA of median price. |
| [`AlphaBetaGammaTrackingFilter`](documentation/algorithms/alpha-beta-gamma-tracking-filter.md) | Steady-state Kalman-like price, velocity, and acceleration tracker. |
| [`AmihudIlliquidity`](documentation/algorithms/amihud-illiquidity.zh-CN.md) | Rolling average absolute return per dollar of traded volume. |
| [`AnchoredVWAP`](documentation/algorithms/anchored-vwap.md) | VWAP accumulated from arbitrary anchor/reset events rather than a fixed session or rolling window. |
| [`AndrewsPitchfork`](documentation/algorithms/andrews-pitchfork.md) | Streaming Andrews Pitchfork median and parallel channels from percent swing pivots. |
| [`ArnaudLegouxMovingAverage`](documentation/algorithms/arnaud-legoux-moving-average.md) | Arnaud Legoux moving average with Gaussian weights controlled by offset and sigma. |
| [`Aroon`](documentation/algorithms/aroon.zh-CN.md) | Aroon Up/Down trend age indicators based on recent highs and lows. |
| [`AroonOscillator`](documentation/algorithms/aroon-oscillator.md) | Difference between Aroon Up and Aroon Down. |
| [`ATR`](documentation/algorithms/atr.md) | Average True Range volatility over a rolling window. |
| [`ATRP`](documentation/algorithms/atrp.md) | Average True Range expressed as a percentage of price. |
| [`ATRRegimeDetector`](documentation/algorithms/atr-regime-detector.zh-CN.md) | Stateful ATR regime detector with high/low hysteresis bands. |
| [`AuctionContinuousMarketTransitionDetector`](documentation/algorithms/auction-continuous-market-transition-detector.zh-CN.md) | Hysteresis detector for auction-versus-continuous market phase signals. |
| [`AverageDirectionalMovementIndex`](documentation/algorithms/average-directional-movement-index.md) | ADX trend-strength indicator. |
| [`AverageDirectionalMovementIndexRating`](documentation/algorithms/average-directional-movement-index-rating.zh-CN.md) | ADXR smoothed ADX trend-strength rating. |
| [`AveragePrice`](documentation/algorithms/average-price.md) | Average of open, high, low, and close. |
| [`AwesomeOscillator`](documentation/algorithms/awesome-oscillator.zh-CN.md) | Difference between short and long median-price moving averages. |
| [`BalanceOfPower`](documentation/algorithms/balance-of-power.zh-CN.md) | Open/high/low/close buying-versus-selling pressure measure. |
| [`BearsPower`](documentation/algorithms/bears-power.md) | Elder bears power: low minus EMA of close. |
| [`Beta`](documentation/algorithms/beta.md) | Rolling beta of one series against another. |
| [`BetaRegimeDetector`](documentation/algorithms/beta-regime-detector.md) | Stateful rolling beta regime detector with upper/lower hysteresis bands. |
| [`Bias`](documentation/algorithms/bias.md) | Percent deviation of price from its simple moving average. |
| [`BidAskBounceRegimeDetector`](documentation/algorithms/bid-ask-bounce-regime-detector.md) | EWMA bid/ask side alternation detector for quote-bounce regimes. |
| [`BollingerBands`](documentation/algorithms/bollinger-bands.zh-CN.md) | Moving-average envelope based on standard deviations. |
| [`BollingerBandwidth`](documentation/algorithms/bollinger-bandwidth.md) | Bollinger band width as (upper-lower)/middle for a rolling mean and standard-deviation envelope. |
| [`BollingerPercentB`](documentation/algorithms/bollinger-percent-b.md) | Bollinger %B position of price inside a rolling mean and standard-deviation envelope. |
| [`BoundedBOCPD`](documentation/algorithms/bounded-bocpd.zh-CN.md) | Bounded-memory Bayesian online change-point detector with constant hazard. |
| [`BullsPower`](documentation/algorithms/bulls-power.md) | Elder bulls power: high minus EMA of close. |
| [`CalibrationDriftDetector`](documentation/algorithms/calibration-drift-detector.md) | EWMA probability-calibration error drift detector. |
| [`CamarillaPivotPoints`](documentation/algorithms/camarilla-pivot-points.md) | Camarilla support/resistance pivots from the previous bar HLC. |
| [`CDL3BlackCrows`](documentation/algorithms/cdl-3-black-crows.md) | Three black crows: three declining bearish bodies. |
| [`CDL3Inside`](documentation/algorithms/cdl-3-inside.md) | Three inside up/down: harami plus confirmation bar. |
| [`CDL3Outside`](documentation/algorithms/cdl-3-outside.md) | Three outside up/down: engulfing plus confirmation bar. |
| [`CDL3WhiteSoldiers`](documentation/algorithms/cdl-3-white-soldiers.md) | Three white soldiers: three advancing bullish bodies. |
| [`CDLBeltHold`](documentation/algorithms/cdl-belt-hold.md) | Belt hold: opens at extreme and posts a long body. |
| [`CDLClosingMarubozu`](documentation/algorithms/cdl-closing-marubozu.md) | Closing marubozu: body closes at/near the extreme of the bar. |
| [`CDLCounterAttack`](documentation/algorithms/cdl-counter-attack.md) | Counterattack: opposite long body closing near prior close. |
| [`CDLDarkCloudCover`](documentation/algorithms/cdl-dark-cloud-cover.md) | Dark cloud cover: bearish close through midpoint of prior bull bar. |
| [`CDLDoji`](documentation/algorithms/cdl-doji.md) | Doji candlestick: very small real body relative to range (indecision). |
| [`CDLDojiStar`](documentation/algorithms/cdl-doji-star.md) | Doji star: doji after a long body (reversal risk). |
| [`CDLDragonflyDoji`](documentation/algorithms/cdl-dragonfly-doji.md) | Dragonfly doji: doji with long lower shadow (bullish rejection). |
| [`CDLEngulfing`](documentation/algorithms/cdl-engulfing.md) | Engulfing: current body fully engulfs prior body (reversal). |
| [`CDLEveningDojiStar`](documentation/algorithms/cdl-evening-doji-star.md) | Evening doji star: evening star with doji middle bar. |
| [`CDLEveningStar`](documentation/algorithms/cdl-evening-star.md) | Evening star: three-bar bearish reversal. |
| [`CDLGravestoneDoji`](documentation/algorithms/cdl-gravestone-doji.md) | Gravestone doji: doji with long upper shadow (bearish rejection). |
| [`CDLHammer`](documentation/algorithms/cdl-hammer.md) | Hammer: long lower shadow in a downtrend (bullish). |
| [`CDLHangingMan`](documentation/algorithms/cdl-hanging-man.md) | Hanging man: hammer shape in an uptrend (bearish). |
| [`CDLHarami`](documentation/algorithms/cdl-harami.md) | Harami: small body inside prior body (reversal/pause). |
| [`CDLHaramiCross`](documentation/algorithms/cdl-harami-cross.md) | Harami cross: doji inside prior body. |
| [`CDLHighWave`](documentation/algorithms/cdl-high-wave.md) | High-wave candle: tiny body with very long shadow(s). |
| [`CDLInvertedHammer`](documentation/algorithms/cdl-inverted-hammer.md) | Inverted hammer: long upper shadow in a downtrend (bullish). |
| [`CDLLongLeggedDoji`](documentation/algorithms/cdl-long-legged-doji.md) | Long-legged doji: doji with long upper and lower shadows. |
| [`CDLLongLine`](documentation/algorithms/cdl-long-line.md) | Long line: large real body relative to recent average body. |
| [`CDLMarubozu`](documentation/algorithms/cdl-marubozu.md) | Marubozu: range dominated by real body (strong directional bar). |
| [`CDLMatchingLow`](documentation/algorithms/cdl-matching-low.md) | Matching low: two bear bars with matching closes (support). |
| [`CDLMorningDojiStar`](documentation/algorithms/cdl-morning-doji-star.md) | Morning doji star: morning star with doji middle bar. |
| [`CDLMorningStar`](documentation/algorithms/cdl-morning-star.md) | Morning star: three-bar bullish reversal. |
| [`CDLPatternPack`](documentation/algorithms/cdl-pattern-pack.md) | Multi-output pack of common CDL patterns from one OHLC update. |
| [`CDLPiercing`](documentation/algorithms/cdl-piercing.md) | Piercing line: bullish close through midpoint of prior bear bar. |
| [`CDLShootingStar`](documentation/algorithms/cdl-shooting-star.md) | Shooting star: long upper shadow in an uptrend (bearish). |
| [`CDLShortLine`](documentation/algorithms/cdl-short-line.md) | Short line: small real body relative to recent average body. |
| [`CDLSpinningTop`](documentation/algorithms/cdl-spinning-top.md) | Spinning top: small body with upper and lower shadows (indecision). |
| [`CDLTriStar`](documentation/algorithms/cdl-tri-star.md) | Tri-star: three consecutive dojis (reversal). |
| [`ChaikinMoneyFlow`](documentation/algorithms/chaikin-money-flow.zh-CN.md) | Volume-weighted money flow over a window. |
| [`ChaikinOscillator`](documentation/algorithms/chaikin-oscillator.zh-CN.md) | MACD-style oscillator of the accumulation/distribution line. |
| [`ChaikinVolatility`](documentation/algorithms/chaikin-volatility.md) | Percent rate-of-change of an EMA of the high-low range. |
| [`ChandeForecastOscillator`](documentation/algorithms/chande-forecast-oscillator.md) | Chande forecast oscillator: percent distance of close from time-series forecast. |
| [`ChandelierExit`](documentation/algorithms/chandelier-exit.md) | ATR trailing chandelier long and short exits from rolling high/low extremes. |
| [`ChandeMomentumOscillator`](documentation/algorithms/chande-momentum-oscillator.zh-CN.md) | Momentum oscillator using sums of recent gains and losses. |
| [`ChoppinessIndex`](documentation/algorithms/choppiness-index.md) | CHOP range/trend measure based on true range versus high-low range. |
| [`ClosePressureReversalSignal`](documentation/algorithms/close-pressure-reversal-signal.md) | End-of-day cross-sectional reversal signal using rest-of-day return, volume/transaction pressure, VWAP location, and rolling conformal-style error bands. |
| [`CointegrationBreakdownMonitor`](documentation/algorithms/cointegration-breakdown-monitor.md) | Streaming residual-z monitor for pair relationship breakdowns using an EWMA hedge estimate. |
| [`CommodityChannelIndex`](documentation/algorithms/commodity-channel-index.zh-CN.md) | CCI deviation of typical price from its moving average. |
| [`ComparativeRelativeStrength`](documentation/algorithms/comparative-relative-strength.md) | Ratio of two price series (A/B relative strength). |
| [`ConformalBands`](documentation/algorithms/conformal-bands.md) | Streaming split-conformal style bands around an SMA center using rolling absolute residual quantiles. |
| [`ConnorsRSI`](documentation/algorithms/connors-rsi.zh-CN.md) | Composite oscillator averaging price RSI, streak RSI, and percent-rank of one-period price change. |
| [`CoppockCurve`](documentation/algorithms/coppock-curve.zh-CN.md) | Weighted moving average of long and short rate-of-change values. |
| [`Correlation`](documentation/algorithms/correlation.zh-CN.md) | Rolling Pearson correlation between two series. |
| [`CorrelationRegimeDetector`](documentation/algorithms/correlation-regime-detector.md) | Stateful rolling correlation regime detector with upper/lower hysteresis bands. |
| [`CrossAssetCorrelationBreakDetector`](documentation/algorithms/cross-asset-correlation-break-detector.md) | Short-versus-long rolling correlation break detector for two assets. |
| [`CrossAssetOrderFlowImbalance`](documentation/algorithms/cross-asset-order-flow-imbalance.md) | Rolling beta of own return on peer OFI with implied impact and residual. |
| [`CumulativeReturn`](documentation/algorithms/cumulative-return.zh-CN.md) | Cumulative return from the first close. |
| [`CUSUM`](documentation/algorithms/cusum.zh-CN.md) | Causal cumulative-sum event filter for detecting threshold-sized directional moves. |
| [`DailyLogReturn`](documentation/algorithms/daily-log-return.md) | Log return between consecutive closes. |
| [`DailyReturn`](documentation/algorithms/daily-return.md) | Percentage return between consecutive closes. |
| [`DDM`](documentation/algorithms/ddm.md) | Drift Detection Method for Bernoulli prediction-error streams. |
| [`DecomposedOrderFlowImbalance`](documentation/algorithms/decomposed-order-flow-imbalance.md) | Quote-driven add/cancel/trade decomposition of Cont-style order-flow pressure. |
| [`Delay`](documentation/algorithms/delay.zh-CN.md) | Lagged value from a fixed number of samples ago. |
| [`DeMarker`](documentation/algorithms/de-marker.md) | DeMarker oscillator from rolling up-high and down-low pressure. |
| [`DetrendedPriceOscillator`](documentation/algorithms/detrended-price-oscillator.zh-CN.md) | DPO cycle indicator comparing price to a displaced average. |
| [`DirectionalChangeDetector`](documentation/algorithms/directional-change-detector.md) | Directional-change intrinsic-time events with overshoot tracking. |
| [`DirectionalMovementIndex`](documentation/algorithms/directional-movement-index.zh-CN.md) | DX directional movement trend-strength indicator. |
| [`DollarBarGenerator`](documentation/algorithms/dollar-bar-generator.md) | Information-driven dollar bars that close when price×volume accumulates to a threshold. |
| [`DollarRunBarGenerator`](documentation/algorithms/dollar-run-bar-generator.md) | Same-sign dollar run bars that close when cumulative |price|×volume hits a threshold. |
| [`DonchianChannel`](documentation/algorithms/donchian-channel.md) | Channel from rolling highest high and lowest low. |
| [`DoubleEMA`](documentation/algorithms/double-ema.zh-CN.md) | DEMA lag-reduced moving average. |
| [`EaseOfMovement`](documentation/algorithms/ease-of-movement.zh-CN.md) | Volume/range indicator for ease of price movement. |
| [`EDDM`](documentation/algorithms/eddm.zh-CN.md) | Early Drift Detection Method using distances between prediction errors. |
| [`EfficiencyRatio`](documentation/algorithms/efficiency-ratio.md) | Kaufman efficiency ratio of net directional move to path length over a rolling window. |
| [`EhlersCenterOfGravity`](documentation/algorithms/ehlers-center-of-gravity.md) | Ehlers center-of-gravity oscillator of a rolling price window with lag trigger. |
| [`EhlersCyberCycle`](documentation/algorithms/ehlers-cyber-cycle.md) | Ehlers cyber cycle band-pass style cycle oscillator with trigger. |
| [`EhlersDecycler`](documentation/algorithms/ehlers-decycler.md) | Ehlers decycler trend estimate and residual oscillator. |
| [`EhlersInstantaneousTrendline`](documentation/algorithms/ehlers-instantaneous-trendline.md) | Ehlers instantaneous trendline with two-bar trigger. |
| [`EhlersOptimalTrackingFilter`](documentation/algorithms/ehlers-optimal-tracking-filter.zh-CN.md) | Adaptive tracking filter using Ehlers' price uncertainty tracking index. |
| [`EhlersRoofingFilter`](documentation/algorithms/ehlers-roofing-filter.md) | Ehlers roofing filter: high-pass plus Super Smoother low-pass. |
| [`EhlersSuperSmoother`](documentation/algorithms/ehlers-super-smoother.md) | Ehlers two-pole Super Smoother low-pass filter. |
| [`ElderRayIndex`](documentation/algorithms/elder-ray-index.zh-CN.md) | Bull and bear power as high/low distance from an EMA of close. |
| [`ElderThermometer`](documentation/algorithms/elder-thermometer.md) | Elder bar-range thermometer: current range vs previous range (ratio and hot flag). |
| [`EMA`](documentation/algorithms/ema.md) | Exponential moving average with more weight on recent samples. |
| [`EWMA`](documentation/algorithms/ewma.zh-CN.md) | Exponentially weighted moving average parameterized by alpha/span/com. |
| [`EWMAZScoreShiftDetector`](documentation/algorithms/ewmaz-score-shift-detector.zh-CN.md) | Causal EWMA mean/variance z-score event detector for threshold-sized shifts. |
| [`ExecutionCostSlippageRegimeDetector`](documentation/algorithms/execution-cost-slippage-regime-detector.zh-CN.md) | Stateful relative execution-cost/slippage regime detector from trade price versus quote mid. |
| [`FastStochastic`](documentation/algorithms/fast-stochastic.zh-CN.md) | Fast stochastic %K/%D oscillator. |
| [`FeatureDistributionDriftDetector`](documentation/algorithms/feature-distribution-drift-detector.zh-CN.md) | Bounded ADWIN-style drift detector for a single streaming feature distribution. |
| [`FibonacciPivotPoints`](documentation/algorithms/fibonacci-pivot-points.md) | Fibonacci support/resistance pivots from the previous bar HLC. |
| [`FibonacciRetracementLevels`](documentation/algorithms/fibonacci-retracement-levels.zh-CN.md) | Rolling Fibonacci retracement levels between recent high and low. |
| [`FisherTransform`](documentation/algorithms/fisher-transform.md) | Ehlers transform of normalized recent high/low position into a turning-point oscillator. |
| [`FlowPressureCapacitySignal`](documentation/algorithms/flow-pressure-capacity-signal.md) | Event-time aggressive flow versus opposing L1 capacity, corrected for queue replenishment, withdrawal fragility, and transient imbalance. |
| [`FOCuS`](documentation/algorithms/focus.md) | Functional online CUSUM mean changepoint detector with candidate pruning (Romano et al.). |
| [`ForceIndex`](documentation/algorithms/force-index.zh-CN.md) | Price-change times volume oscillator. |
| [`FourierResidueIdentity`](documentation/algorithms/fourier-residue-identity.md) | Fourier-Residue Identity splitting return autocorrelation into testable direction (sign, k=2) and magnitude (k=4) channels, with Fejér variance ratios per channel. |
| [`FractalAdaptiveMovingAverage`](documentation/algorithms/fractal-adaptive-moving-average.zh-CN.md) | Ehlers FRAMA using fractal dimension to adapt EMA smoothing. |
| [`GatorOscillator`](documentation/algorithms/gator-oscillator.md) | Bill Williams Gator Oscillator from Alligator jaw-teeth and teeth-lips gaps. |
| [`GaussianProcessRegressionBands`](documentation/algorithms/gaussian-process-regression-bands.md) | Rolling RBF-kernel Gaussian process posterior mean with uncertainty bands. |
| [`GeometricMovingAverage`](documentation/algorithms/geometric-moving-average.md) | Geometric moving average as exp of SMA of log prices. |
| [`GuppyMMARibbon`](documentation/algorithms/guppy-mma-ribbon.md) | Full Guppy MMA ribbon of six short and six long EMAs plus group averages. |
| [`GuppyMultipleMovingAverage`](documentation/algorithms/guppy-multiple-moving-average.md) | Guppy MMA short and long EMA-group averages and spread. |
| [`HawkesIntensity`](documentation/algorithms/hawkes-intensity.md) | Exponential Hawkes self-exciting intensity process for event times. |
| [`HDDM`](documentation/algorithms/hddm.md) | Hoeffding-bound drift detector for Bernoulli prediction-error streams. |
| [`HeikinAshiTransform`](documentation/algorithms/heikin-ashi-transform.md) | Incremental Heikin-Ashi OHLC transform for smoothing candles. |
| [`HiddenSemiMarkovRegimeFilter`](documentation/algorithms/hidden-semi-markov-regime-filter.md) | Online Gaussian hidden semi-Markov-style regime filter with bounded duration bias. |
| [`High`](documentation/algorithms/high.md) | Rolling highest value. |
| [`HighIndex`](documentation/algorithms/high-index.zh-CN.md) | Offset/index of the rolling highest value. |
| [`HighLow`](documentation/algorithms/high-low.zh-CN.md) | Combined rolling minimum and maximum values. |
| [`HighLowIndex`](documentation/algorithms/high-low-index.zh-CN.md) | Combined offsets/indexes of rolling minimum and maximum values. |
| [`HilbertDominantCyclePeriod`](documentation/algorithms/hilbert-dominant-cycle-period.md) | TA-Lib HT_DCPERIOD-compatible dominant cycle period. |
| [`HilbertDominantCyclePhase`](documentation/algorithms/hilbert-dominant-cycle-phase.md) | TA-Lib HT_DCPHASE-compatible dominant cycle phase in degrees. |
| [`HilbertPhasor`](documentation/algorithms/hilbert-phasor.md) | TA-Lib HT_PHASOR-compatible in-phase and quadrature components. |
| [`HilbertSineWave`](documentation/algorithms/hilbert-sine-wave.md) | TA-Lib HT_SINE-compatible sine and lead-sine waves. |
| [`HilbertTrendline`](documentation/algorithms/hilbert-trendline.md) | TA-Lib HT_TRENDLINE-compatible instantaneous trendline. |
| [`HilbertTrendMode`](documentation/algorithms/hilbert-trend-mode.md) | TA-Lib HT_TRENDMODE-compatible trend-versus-cycle mode flag (1=trend, 0=cycle). |
| [`HistoricalVolatility`](documentation/algorithms/historical-volatility.md) | Annualized rolling standard deviation of log returns. |
| [`HitRateDriftDetector`](documentation/algorithms/hit-rate-drift-detector.md) | EWMA hit-rate degradation detector using miss-rate hysteresis. |
| [`HullMovingAverage`](documentation/algorithms/hull-moving-average.md) | HMA lag-reduced weighted moving average. |
| [`Ichimoku`](documentation/algorithms/ichimoku.zh-CN.md) | Ichimoku conversion, base, leading spans, lagging span, and displaced cloud spans. |
| [`ImbalanceBarGenerator`](documentation/algorithms/imbalance-bar-generator.md) | Volume-imbalance bars that close when |signed volume| hits a threshold (tick-rule signs). |
| [`Inertia`](documentation/algorithms/inertia.md) | Dorsey inertia: linear regression of relative volatility index. |
| [`IntegratedOrderFlowImbalance`](documentation/algorithms/integrated-order-flow-imbalance.md) | Multi-level Cont OFI projected onto an online first principal component. |
| [`InteractingMultipleModelFilter`](documentation/algorithms/interacting-multiple-model-filter.zh-CN.md) | Four-regime IMM Kalman tracker that blends low-volatility, high-volatility, trend, and chop models by online probabilities. |
| [`IntradayClockEchoSignal`](documentation/algorithms/intraday-clock-echo-signal.zh-CN.md) | Same-clock intraday return-periodicity signal trained from prior aggregate-bar day lists. |
| [`IntradayIntensity`](documentation/algorithms/intraday-intensity.md) | Rolling volume-weighted intraday intensity (2C-H-L)/(H-L). |
| [`IntradayMomentumIndex`](documentation/algorithms/intraday-momentum-index.md) | RSI-style oscillator of open-to-close gains versus losses within each bar. |
| [`InverseFisherRSI`](documentation/algorithms/inverse-fisher-rsi.md) | Inverse Fisher transform of RSI for sharper turning points. |
| [`KagiChart`](documentation/algorithms/kagi-chart.md) | Streaming Kagi line, direction, and reversal events. |
| [`KalmanExtremumTrend`](documentation/algorithms/kalman-extremum-trend.md) | Kalman trend combined with stochastic-style position inside recent extrema. |
| [`KalmanHedgeRatio`](documentation/algorithms/kalman-hedge-ratio.zh-CN.md) | Online Kalman regression hedge ratio and pair spread. |
| [`KalmanInnovationResidualBOCPD`](documentation/algorithms/kalman-innovation-residual-bocpd.md) | Kalman innovation z-score residual piped into ResidualBOCPD changepoint detection. |
| [`KalmanInnovationResidualFOCuS`](documentation/algorithms/kalman-innovation-residual-focus.md) | Kalman innovation z-score residual piped into FOCuS changepoint detection. |
| [`KalmanInnovationZScore`](documentation/algorithms/kalman-innovation-z-score.zh-CN.md) | Signed measurement innovation normalized by the predicted innovation standard deviation. |
| [`KalmanLocalLinearTrend`](documentation/algorithms/kalman-local-linear-trend.md) | Kalman local level/trend state-space estimator. |
| [`KalmanMovingAverage`](documentation/algorithms/kalman-moving-average.zh-CN.md) | Kalman price filter using a local linear price/velocity model. |
| [`KalmanPredictionBands`](documentation/algorithms/kalman-prediction-bands.zh-CN.md) | One-step Kalman prediction with upper/lower bands from predicted measurement uncertainty. |
| [`KalmanRegressionChannel`](documentation/algorithms/kalman-regression-channel.md) | Online Kalman regression with prediction channel and spread. |
| [`KalmanTrendSignal`](documentation/algorithms/kalman-trend-signal.md) | Kalman-filtered trend line with buy/sell signal based on price versus filtered trend. |
| [`KalmanVelocityOscillator`](documentation/algorithms/kalman-velocity-oscillator.md) | Zero-centered velocity state from a constant-velocity Kalman price model. |
| [`Kama`](documentation/algorithms/kama.zh-CN.md) | Kaufman Adaptive Moving Average. |
| [`KeltnerChannel`](documentation/algorithms/keltner-channel.md) | EMA/ATR volatility channel. |
| [`KeltnerChannelOriginal`](documentation/algorithms/keltner-channel-original.md) | Original SMA/range Keltner channel variant. |
| [`KlingerVolumeOscillator`](documentation/algorithms/klinger-volume-oscillator.zh-CN.md) | Volume-force oscillator using fast and slow EMAs plus signal line. |
| [`KSTOscillator`](documentation/algorithms/kst-oscillator.zh-CN.md) | Pring Know Sure Thing smoothed multi-ROC oscillator. |
| [`KSWIN`](documentation/algorithms/kswin.zh-CN.md) | Kolmogorov-Smirnov sliding-window drift detector. |
| [`KyleLambda`](documentation/algorithms/kyle-lambda.zh-CN.md) | Rolling price-impact slope of returns against signed square-root dollar volume. |
| [`LeadLagRegimeDetector`](documentation/algorithms/lead-lag-regime-detector.zh-CN.md) | EWMA cross-lag detector for which of two series is leading. |
| [`LinearRegression`](documentation/algorithms/linear-regression.md) | Rolling least-squares fitted value. |
| [`LinearRegressionAngle`](documentation/algorithms/linear-regression-angle.md) | Angle of the rolling linear-regression slope. |
| [`LinearRegressionIntercept`](documentation/algorithms/linear-regression-intercept.md) | Intercept of the rolling linear-regression fit. |
| [`LinearRegressionSlope`](documentation/algorithms/linear-regression-slope.md) | Slope of the rolling linear-regression fit. |
| [`LiquidityDroughtDetector`](documentation/algorithms/liquidity-drought-detector.md) | Relative volume/depth drought detector using lower-threshold hysteresis. |
| [`LiquidityRegimeDetector`](documentation/algorithms/liquidity-regime-detector.md) | EWMA Amihud-style liquidity regime detector using absolute return per dollar volume. |
| [`Low`](documentation/algorithms/low.zh-CN.md) | Rolling lowest value. |
| [`LowIndex`](documentation/algorithms/low-index.zh-CN.md) | Offset/index of the rolling lowest value. |
| [`MACD`](documentation/algorithms/macd.md) | MACD line, signal, and histogram from fast/slow EMA difference. |
| [`MACDExt`](documentation/algorithms/macd-ext.md) | MACD with selectable SMA/EMA types for fast, slow, and signal. |
| [`MACDFix`](documentation/algorithms/macd-fix.md) | Fixed 12/26 MACD with multi-output line, signal, and histogram. |
| [`MarketFacilitationIndex`](documentation/algorithms/market-facilitation-index.md) | Bar range divided by volume (Bill Williams MFI). |
| [`MarketOpenCloseTransitionDetector`](documentation/algorithms/market-open-close-transition-detector.md) | Session-progress transition detector for market-open and market-close bands. |
| [`MassIndex`](documentation/algorithms/mass-index.zh-CN.md) | Range-expansion reversal indicator. |
| [`MatchedFlowConformalSignal`](documentation/algorithms/matched-flow-conformal-signal.md) | Intraday OHLCV matched-flow signal with conformal-style rolling error bands and target sizing diagnostics. |
| [`McGinleyDynamic`](documentation/algorithms/mc-ginley-dynamic.md) | McGinley Dynamic adaptive moving average that speeds up in trends and slows in chop. |
| [`MedianPrice`](documentation/algorithms/median-price.md) | Average of high and low. |
| [`MesaAdaptiveMovingAverage`](documentation/algorithms/mesa-adaptive-moving-average.zh-CN.md) | Ehlers MAMA/FAMA adaptive moving averages driven by dominant cycle phase. |
| [`MessageEventOrderFlowImbalance`](documentation/algorithms/message-event-order-flow-imbalance.md) | Rolling OFI accumulated from discrete LOB/trade message events (add/cancel/trade). |
| [`MicrostructureNoiseRegimeDetector`](documentation/algorithms/microstructure-noise-regime-detector.md) | EWMA trade-versus-mid noise detector normalized by quoted spread. |
| [`MidPoint`](documentation/algorithms/mid-point.md) | Midpoint of rolling high and low values for one series. |
| [`MidPrice`](documentation/algorithms/mid-price.zh-CN.md) | Midpoint of rolling high and low price series. |
| [`MinusDirectionalIndicator`](documentation/algorithms/minus-directional-indicator.zh-CN.md) | Negative directional indicator. |
| [`MinusDirectionalMovement`](documentation/algorithms/minus-directional-movement.md) | Negative directional movement. |
| [`Momentum`](documentation/algorithms/momentum.md) | Difference between current value and a prior value. |
| [`MoneyFlowIndex`](documentation/algorithms/money-flow-index.zh-CN.md) | Volume-weighted RSI-like money flow oscillator. |
| [`MovingAverageEnvelope`](documentation/algorithms/moving-average-envelope.md) | Percentage envelope bands above and below a simple moving average. |
| [`MovingAverageVariablePeriod`](documentation/algorithms/moving-average-variable-period.md) | SMA with a per-bar variable period (TA-Lib MAVP-style). |
| [`MultiLevelOrderFlowImbalance`](documentation/algorithms/multi-level-order-flow-imbalance.md) | Cont-style order-flow imbalance at each book level with sum/mean aggregates. |
| [`MultiPeerOrderFlowImbalance`](documentation/algorithms/multi-peer-order-flow-imbalance.md) | Basket peer OFI (equal-weight mean) with rolling beta impact on own return. |
| [`NadarayaWatsonEnvelope`](documentation/algorithms/nadaraya-watson-envelope.md) | Gaussian-kernel Nadaraya-Watson smoother with weighted residual bands. |
| [`NegativeVolumeIndex`](documentation/algorithms/negative-volume-index.md) | Cumulative indicator that changes on lower-volume periods. |
| [`NormalizedATR`](documentation/algorithms/normalized-atr.md) | ATR normalized by close. |
| [`OnBalanceVolume`](documentation/algorithms/on-balance-volume.md) | Cumulative volume added/subtracted by close direction. |
| [`OnlineGaussianMixtureRegimeFilter`](documentation/algorithms/online-gaussian-mixture-regime-filter.md) | Online Gaussian mixture regime filter with bounded component count. |
| [`OnlineHMMRegimeFilter`](documentation/algorithms/online-hmm-regime-filter.zh-CN.md) | Online Gaussian hidden Markov regime filter with fixed transition persistence. |
| [`OnlineMarkovSwitchingVolatilityFilter`](documentation/algorithms/online-markov-switching-volatility-filter.zh-CN.md) | Online two-state Markov-switching volatility filter over close-to-close moves. |
| [`OrderFlowImbalance`](documentation/algorithms/order-flow-imbalance.zh-CN.md) | Quote-level best bid/ask price and size change pressure over a rolling update window. |
| [`OrderFlowImbalanceRegimeDetector`](documentation/algorithms/order-flow-imbalance-regime-detector.zh-CN.md) | EWMA order-flow imbalance regime detector with buy/sell pressure hysteresis. |
| [`PageHinkley`](documentation/algorithms/page-hinkley.zh-CN.md) | Causal Page-Hinkley mean-shift event detector with directed up/down output. |
| [`PairsSpreadRegimeDetector`](documentation/algorithms/pairs-spread-regime-detector.md) | Streaming EWMA hedge-ratio residual z-score detector for pair-spread regimes. |
| [`ParabolicSAR`](documentation/algorithms/parabolic-sar.md) | Parabolic stop-and-reverse trailing trend indicator. |
| [`ParabolicSARExtended`](documentation/algorithms/parabolic-sar-extended.md) | Extended parabolic SAR with separate long/short AF chains (SAREXT-style). |
| [`ParticleFilterTrend`](documentation/algorithms/particle-filter-trend.zh-CN.md) | Deterministic-seed particle trend filter with Laplace measurement likelihood and effective sample size output. |
| [`PercentagePrice`](documentation/algorithms/percentage-price.md) | Percentage Price Oscillator. |
| [`PercentageVolume`](documentation/algorithms/percentage-volume.md) | Percentage Volume Oscillator. |
| [`PivotPoints`](documentation/algorithms/pivot-points.md) | Classic floor pivot points (PP/R1-R3/S1-S3) from the previous bar. |
| [`PlusDirectionalIndicator`](documentation/algorithms/plus-directional-indicator.md) | Positive directional indicator. |
| [`PlusDirectionalMovement`](documentation/algorithms/plus-directional-movement.md) | Positive directional movement. |
| [`PointAndFigure`](documentation/algorithms/point-and-figure.md) | Streaming point-and-figure box price, direction, and reversals. |
| [`PositiveVolumeIndex`](documentation/algorithms/positive-volume-index.md) | Cumulative indicator that changes on higher-volume periods. |
| [`PredictionErrorDriftDetector`](documentation/algorithms/prediction-error-drift-detector.md) | EWMA absolute prediction-error drift detector. |
| [`PrettyGoodOscillator`](documentation/algorithms/pretty-good-oscillator.md) | Close minus SMA normalized by ATR (PGO). |
| [`ProjectionOscillator`](documentation/algorithms/projection-oscillator.md) | Stochastic-style oscillator of close within linear-regression projection bands of high and low. |
| [`PsychologicalLine`](documentation/algorithms/psychological-line.md) | Percent of up-closes over a rolling window. |
| [`QStick`](documentation/algorithms/q-stick.md) | Simple moving average of close minus open. |
| [`QuoteMessageRateRegimeDetector`](documentation/algorithms/quote-message-rate-regime-detector.zh-CN.md) | Relative EWMA quote-message-rate regime detector. |
| [`QuoteStuffingDetector`](documentation/algorithms/quote-stuffing-detector.zh-CN.md) | EWMA quote-to-trade message ratio detector for quote-stuffing episodes. |
| [`RainbowMovingAverage`](documentation/algorithms/rainbow-moving-average.md) | Mel Widner rainbow: recursive SMA layers with outer/high/low/mid/width. |
| [`RainbowOscillator`](documentation/algorithms/rainbow-oscillator.md) | Rainbow oscillator: percent width and position of recursive SMA layers. |
| [`RandomWalkIndex`](documentation/algorithms/random-walk-index.md) | Random walk index high/low relative to ATR-scaled range. |
| [`RangeActionVerificationIndex`](documentation/algorithms/range-action-verification-index.md) | RAVI: absolute short-vs-long SMA gap as a percent of the long SMA. |
| [`RateOfChangePercentage`](documentation/algorithms/rate-of-change-percentage.md) | Period-over-period rate of change as a fraction. |
| [`RateOfChangeRatio`](documentation/algorithms/rate-of-change-ratio.zh-CN.md) | Rate-of-change ratio against a prior value. |
| [`RateOfChangeRatio100`](documentation/algorithms/rate-of-change-ratio-100.md) | Rate-of-change ratio scaled by 100. |
| [`RealizedVarianceRegimeDetector`](documentation/algorithms/realized-variance-regime-detector.md) | Rolling realized-variance regime detector from squared close-to-close changes. |
| [`RelativeVigorIndex`](documentation/algorithms/relative-vigor-index.md) | Smoothed close-open momentum relative to high-low range with signal line. |
| [`RelativeVolatilityIndex`](documentation/algorithms/relative-volatility-index.md) | RSI-style relative volatility index on rolling close stddev. |
| [`RenkoBrickGenerator`](documentation/algorithms/renko-brick-generator.md) | Event-driven Renko price transform that emits signed brick counts and current brick state from close updates. |
| [`ResidualBOCPD`](documentation/algorithms/residual-bocpd.md) | Bounded BOCPD changepoint detector applied to residual/innovation series. |
| [`ResidualDriftDetector`](documentation/algorithms/residual-drift-detector.md) | EWMA residual z-score drift detector with signed hysteresis output. |
| [`ResidualFOCuS`](documentation/algorithms/residual-focus.md) | FOCuS applied to residual/innovation series for model-based changepoint detection. |
| [`ROC`](documentation/algorithms/roc.md) | Rate of Change momentum as percentage change over a lookback. |
| [`RollingBetaShiftDetector`](documentation/algorithms/rolling-beta-shift-detector.md) | Causal adjacent-window beta shift detector. |
| [`RollingCorrelationShiftDetector`](documentation/algorithms/rolling-correlation-shift-detector.md) | Causal adjacent-window correlation shift detector. |
| [`RollingMeanShiftDetector`](documentation/algorithms/rolling-mean-shift-detector.md) | Causal adjacent-window mean shift detector using a two-sample z-score. |
| [`RollingMeanVarianceShiftDetector`](documentation/algorithms/rolling-mean-variance-shift-detector.zh-CN.md) | Causal adjacent-window combined mean and variance shift detector. |
| [`RollingMedian`](documentation/algorithms/rolling-median.md) | Rolling median of a price window. |
| [`RollingSpreadLiquidityShiftDetector`](documentation/algorithms/rolling-spread-liquidity-shift-detector.zh-CN.md) | Causal adjacent-window quote spread/depth liquidity stress shift detector. |
| [`RollingVarianceShiftDetector`](documentation/algorithms/rolling-variance-shift-detector.md) | Causal adjacent-window variance shift detector using log variance ratio. |
| [`RSI`](documentation/algorithms/rsi.md) | Relative Strength Index momentum oscillator. |
| [`RunBarGenerator`](documentation/algorithms/run-bar-generator.md) | Tick run bars that close after consecutive same-sign ticks reach a threshold. |
| [`SavitzkyGolayFilter`](documentation/algorithms/savitzky-golay-filter.zh-CN.md) | Rolling polynomial least-squares smoother with first and second derivative outputs. |
| [`SchaffTrendCycle`](documentation/algorithms/schaff-trend-cycle.zh-CN.md) | MACD/stochastic cycle oscillator. |
| [`SMA`](documentation/algorithms/sma.zh-CN.md) | Simple moving average over a rolling window. |
| [`SmoothedMovingAverage`](documentation/algorithms/smoothed-moving-average.md) | Wilder/SMMA/RMA smoothed moving average seeded by an initial SMA window. |
| [`SpreadExplosionDetector`](documentation/algorithms/spread-explosion-detector.md) | EWMA relative quoted-spread explosion detector. |
| [`SpreadFeatures`](documentation/algorithms/spread-features.md) | Quoted, effective, and realized spread estimates from trades and contemporaneous quotes. |
| [`SpreadRegimeDetector`](documentation/algorithms/spread-regime-detector.zh-CN.md) | Stateful quoted-spread regime detector using relative bid/ask spread. |
| [`SqrtImpactFlowSignal`](documentation/algorithms/sqrt-impact-flow-signal.md) | Square-root impact residual flow: continuation of unused impact budget and reversion of overshoot, with optional VWAP alignment (Massive/Polygon). |
| [`SqueezeMomentum`](documentation/algorithms/squeeze-momentum.md) | TTM-style Bollinger-inside-Keltner squeeze flag with linreg momentum. |
| [`StdDev`](documentation/algorithms/std-dev.zh-CN.md) | Rolling standard deviation. |
| [`StickyHMMRegimeFilter`](documentation/algorithms/sticky-hmm-regime-filter.zh-CN.md) | Online Gaussian HMM regime filter with high self-transition persistence. |
| [`Stochastic`](documentation/algorithms/stochastic.md) | Slow stochastic oscillator. |
| [`StochasticMomentumIndex`](documentation/algorithms/stochastic-momentum-index.md) | Double-smoothed stochastic momentum index with signal line. |
| [`StochRSI`](documentation/algorithms/stoch-rsi.md) | Stochastic oscillator applied to RSI values. |
| [`Summation`](documentation/algorithms/summation.zh-CN.md) | Rolling sum. |
| [`SuperTrend`](documentation/algorithms/super-trend.zh-CN.md) | ATR-band trend-following indicator. |
| [`SwingIndex`](documentation/algorithms/swing-index.md) | Wilder's Swing Index of bar-to-bar price action. |
| [`T3MovingAverage`](documentation/algorithms/t-3-moving-average.zh-CN.md) | Tillson T3 multi-EMA moving average. |
| [`ThresholdRegimeDetector`](documentation/algorithms/threshold-regime-detector.zh-CN.md) | Stateful threshold regime detector with upper/lower hysteresis bands. |
| [`TimeSeriesForecast`](documentation/algorithms/time-series-forecast.zh-CN.md) | Rolling linear-regression time-series forecast. |
| [`TradeIntensityRegimeDetector`](documentation/algorithms/trade-intensity-regime-detector.md) | EWMA relative trade-count intensity regime detector. |
| [`TrendChopRegimeDetector`](documentation/algorithms/trend-chop-regime-detector.md) | Efficiency-ratio trend-versus-chop regime detector using true range. |
| [`TrendIntensityIndex`](documentation/algorithms/trend-intensity-index.md) | Percent of positive deviations from SMA over absolute deviations. |
| [`TriangularMovingAverage`](documentation/algorithms/triangular-moving-average.md) | Double-smoothed triangular moving average. |
| [`TripleEMA`](documentation/algorithms/triple-ema.zh-CN.md) | TEMA lag-reduced moving average. |
| [`Trix`](documentation/algorithms/trix.md) | Triple-smoothed rate-of-change oscillator. |
| [`TrueRange`](documentation/algorithms/true-range.md) | Maximum of high-low and gaps from previous close. |
| [`TSI`](documentation/algorithms/tsi.zh-CN.md) | True Strength Index double-smoothed momentum oscillator. |
| [`TwiggsMoneyFlow`](documentation/algorithms/twiggs-money-flow.md) | Twiggs money flow using true high/low and EMA volume normalization. |
| [`TwoFactorKalmanTrendFilter`](documentation/algorithms/two-factor-kalman-trend-filter.md) | Two-state short/long Kalman trend contribution model. |
| [`TypicalPrice`](documentation/algorithms/typical-price.zh-CN.md) | Average of high, low, and close. |
| [`UlcerIndex`](documentation/algorithms/ulcer-index.md) | Drawdown-based downside-risk measure. |
| [`UltimateOscillator`](documentation/algorithms/ultimate-oscillator.zh-CN.md) | Weighted multi-window buying-pressure oscillator. |
| [`VariableIndexDynamicAverage`](documentation/algorithms/variable-index-dynamic-average.md) | VIDYA adaptive EMA using absolute CMO as the smoothing factor. |
| [`Variance`](documentation/algorithms/variance.zh-CN.md) | Rolling variance. |
| [`VerticalHorizontalFilter`](documentation/algorithms/vertical-horizontal-filter.md) | Trend strength as net move over path length (VHF). |
| [`VolatilityBreakoutDetector`](documentation/algorithms/volatility-breakout-detector.zh-CN.md) | EWMA z-score detector for unusually large close-to-close volatility breakouts. |
| [`VolatilityCompressionExpansionDetector`](documentation/algorithms/volatility-compression-expansion-detector.md) | Short-versus-long EWMA volatility ratio detector for compression and expansion regimes. |
| [`VolatilityRegimeDetector`](documentation/algorithms/volatility-regime-detector.zh-CN.md) | EWMA close-change volatility regime detector with high/low hysteresis bands. |
| [`VolumeBarGenerator`](documentation/algorithms/volume-bar-generator.md) | Information-driven volume bars that close when traded volume hits a threshold. |
| [`VolumeOscillator`](documentation/algorithms/volume-oscillator.md) | Percent difference between short and long simple moving averages of volume. |
| [`VolumePriceTrend`](documentation/algorithms/volume-price-trend.zh-CN.md) | Cumulative volume adjusted by percentage price change. |
| [`VolumeProfile`](documentation/algorithms/volume-profile.zh-CN.md) | Rolling volume-by-price histogram that emits point of control and value-area high/low levels. |
| [`VolumeRegimeDetector`](documentation/algorithms/volume-regime-detector.zh-CN.md) | EWMA relative volume regime detector with high/low hysteresis bands. |
| [`VolumeRunBarGenerator`](documentation/algorithms/volume-run-bar-generator.md) | Same-sign volume run bars that close when cumulative volume hits a threshold. |
| [`VolumeWeightedAveragePrice`](documentation/algorithms/volume-weighted-average-price.md) | VWAP price weighted by traded volume. |
| [`VolumeWeightedMovingAverage`](documentation/algorithms/volume-weighted-moving-average.md) | VWMA rolling close weighted by volume. |
| [`Vortex`](documentation/algorithms/vortex.md) | Positive/negative Vortex trend movement indicator. |
| [`VPIN`](documentation/algorithms/vpin.md) | Volume-synchronized probability of informed trading using bulk-volume classification and rolling volume-bucket imbalance. |
| [`WaveTrend`](documentation/algorithms/wave-trend.md) | LazyBear WaveTrend oscillator (wt1/wt2) on HLC3. |
| [`WeightedClosePrice`](documentation/algorithms/weighted-close-price.zh-CN.md) | Weighted close transform using high, low, and close. |
| [`WeightedMovingAverage`](documentation/algorithms/weighted-moving-average.zh-CN.md) | Weighted moving average with larger recent weights. |
| [`WeightedMultiPeerOrderFlowImbalance`](documentation/algorithms/weighted-multi-peer-order-flow-imbalance.md) | Basket peer OFI with explicit peer weights and rolling beta impact on own return. |
| [`WilliamsAD`](documentation/algorithms/williams-ad.md) | Williams accumulation/distribution cumulative line. |
| [`WilliamsFractals`](documentation/algorithms/williams-fractals.md) | 5-bar Williams up/down fractal pivots with confirmation lag. |
| [`WilliamsR`](documentation/algorithms/williams-r.md) | Williams %R overbought/oversold oscillator. |
| [`WoodiePivotPoints`](documentation/algorithms/woodie-pivot-points.md) | Woodie floor pivots from previous bar H + L + 2C. |
| [`ZeroLagEMA`](documentation/algorithms/zero-lag-ema.md) | Zero-lag exponential moving average using de-lagged price into an EMA. |
| [`ZigZagSwingDetector`](documentation/algorithms/zig-zag-swing-detector.md) | Close-based swing detector that filters price moves below a percentage threshold and emits confirmed pivots. |
