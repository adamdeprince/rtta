# Algorithms

This file lists the public indicator algorithms exported by `rtta`. Tuning/result helper types are intentionally omitted. Documentation links prefer ChartSchool when it has a reasonably direct article; non-ChartSchool links are used only when ChartSchool does not cover that specific algorithm or helper.

| Algorithm | Description | Documentation |
|---|---|---|
| `ATR` | Average True Range volatility over a rolling window. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp |
| `ATRP` | Average True Range expressed as a percentage of price. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp |
| `EMA` | Exponential moving average with more weight on recent samples. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential |
| `EWMA` | Exponentially weighted moving average parameterized by alpha/span/com. | https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html |
| `MACD` | Moving Average Convergence/Divergence oscillator and signal/histogram. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator |
| `ROC` | Rate of Change momentum as percentage change over a lookback. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum |
| `RSI` | Relative Strength Index momentum oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi |
| `SMA` | Simple moving average over a rolling window. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential |
| `TSI` | True Strength Index double-smoothed momentum oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/true-strength-index |
| `AbsolutePriceOscillator` | Difference between fast and slow moving averages in price units. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator |
| `AccumulationDistribution` | Volume-price accumulation/distribution line. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/accumulation-distribution-line |
| `Aroon` | Aroon Up/Down trend age indicators based on recent highs and lows. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/aroon |
| `AroonOscillator` | Difference between Aroon Up and Aroon Down. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/aroon-oscillator |
| `AverageDirectionalMovementIndex` | ADX trend-strength indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `AverageDirectionalMovementIndexRating` | ADXR smoothed ADX trend-strength rating. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `AveragePrice` | Average of open, high, low, and close. | https://tulipindicators.org/avgprice |
| `AwesomeOscillator` | Difference between short and long median-price moving averages. | https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html |
| `BalanceOfPower` | Open/high/low/close buying-versus-selling pressure measure. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/balance-of-power-bop |
| `Beta` | Rolling beta of one series against another. | https://www.investopedia.com/terms/b/beta.asp |
| `BollingerBands` | Moving-average envelope based on standard deviations. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/bollinger-bands |
| `ChaikinMoneyFlow` | Volume-weighted money flow over a window. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/chaikin-money-flow-cmf |
| `ChaikinOscillator` | MACD-style oscillator of the accumulation/distribution line. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/chaikin-oscillator |
| `ChandeMomentumOscillator` | Momentum oscillator using sums of recent gains and losses. | https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo |
| `ChoppinessIndex` | CHOP range/trend measure based on true range versus high-low range. | https://www.angelone.in/knowledge-center/online-share-trading/choppiness-index-indicator |
| `CommodityChannelIndex` | CCI deviation of typical price from its moving average. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/commodity-channel-index-cci |
| `Correlation` | Rolling Pearson correlation between two series. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/correlation-coefficient |
| `CumulativeReturn` | Cumulative return from the first close. | https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html |
| `DailyLogReturn` | Log return between consecutive closes. | https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html |
| `DailyReturn` | Percentage return between consecutive closes. | https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html |
| `Delay` | Lagged value from a fixed number of samples ago. | https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html |
| `DetrendedPriceOscillator` | DPO cycle indicator comparing price to a displaced average. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/detrended-price-oscillator-dpo |
| `DirectionalMovementIndex` | DX directional movement trend-strength indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `DoubleEMA` | DEMA lag-reduced moving average. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/double-exponential-moving-average-dema |
| `DonchianChannel` | Channel from rolling highest high and lowest low. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/donchian-channels |
| `EaseOfMovement` | Volume/range indicator for ease of price movement. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ease-of-movement-emv |
| `FastStochastic` | Fast stochastic %K/%D oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochastic-oscillator-fast-slow-and-full |
| `FibonacciRetracementLevels` | Rolling Fibonacci retracement levels between recent high and low. | https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/fibonacci-retracement |
| `ForceIndex` | Price-change times volume oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/force-index |
| `High` | Rolling highest value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/highest-high-value |
| `HighIndex` | Offset/index of the rolling highest value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/distance-to-highs |
| `HighLow` | Combined rolling minimum and maximum values. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/high-low-bands |
| `HighLowIndex` | Combined offsets/indexes of rolling minimum and maximum values. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/distance-to-highs |
| `HullMovingAverage` | HMA lag-reduced weighted moving average. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/hull-moving-average-hma |
| `Ichimoku` | Ichimoku conversion, base, and leading span components. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/ichimoku-cloud |
| `KSTOscillator` | Pring Know Sure Thing smoothed multi-ROC oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/pring-s-know-sure-thing-kst |
| `KalmanInnovationZScore` | Signed measurement innovation normalized by the predicted innovation standard deviation. | https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf |
| `KalmanLocalLinearTrend` | Kalman local level/trend state-space estimator. | https://www.statsmodels.org/v0.12.2/examples/notebooks/generated/statespace_local_linear_trend.html |
| `KalmanMovingAverage` | Kalman price filter using a local linear price/velocity model. | https://arxiv.org/pdf/1808.03297 |
| `KalmanPredictionBands` | One-step Kalman prediction with upper/lower bands from predicted measurement uncertainty. | https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf |
| `KalmanTrendSignal` | Kalman-filtered trend line with buy/sell signal based on price versus filtered trend. | https://www.aimspress.com/aimspress-data/dsfe/2024/4/PDF/DSFE-04-04-023.pdf |
| `KalmanVelocityOscillator` | Zero-centered velocity state from a constant-velocity Kalman price model. | https://arxiv.org/pdf/1808.03297 |
| `Kama` | Kaufman Adaptive Moving Average. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama |
| `KeltnerChannel` | EMA/ATR volatility channel. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/keltner-channels |
| `KeltnerChannelOriginal` | Original SMA/range Keltner channel variant. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/keltner-channels |
| `LinearRegression` | Rolling least-squares fitted value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope |
| `LinearRegressionAngle` | Angle of the rolling linear-regression slope. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope |
| `LinearRegressionIntercept` | Intercept of the rolling linear-regression fit. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/linear-regression-r2 |
| `LinearRegressionSlope` | Slope of the rolling linear-regression fit. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope |
| `Low` | Rolling lowest value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/lowest-low-value |
| `LowIndex` | Offset/index of the rolling lowest value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/distance-to-lows |
| `MACDFix` | MACD with fixed 12/26 moving-average periods. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator |
| `MassIndex` | Range-expansion reversal indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/mass-index |
| `MedianPrice` | Average of high and low. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/median-price |
| `MidPoint` | Midpoint of rolling high and low values for one series. | https://vectoralpha.dev/projects/ta/indicators/midpoint/ |
| `MidPrice` | Midpoint of rolling high and low price series. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/high-low-bands |
| `MinusDirectionalIndicator` | Negative directional indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `MinusDirectionalMovement` | Negative directional movement. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `Momentum` | Difference between current value and a prior value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum |
| `MoneyFlowIndex` | Volume-weighted RSI-like money flow oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/money-flow-index-mfi |
| `NegativeVolumeIndex` | Cumulative indicator that changes on lower-volume periods. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/negative-volume-index-nvi |
| `NormalizedATR` | ATR normalized by close. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp |
| `OnBalanceVolume` | Cumulative volume added/subtracted by close direction. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/on-balance-volume-obv |
| `ParabolicSAR` | Parabolic stop-and-reverse trailing trend indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/parabolic-sar |
| `PercentagePrice` | Percentage Price Oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/percentage-price-oscillator-ppo |
| `PercentageVolume` | Percentage Volume Oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/percentage-volume-oscillator-pvo |
| `PlusDirectionalIndicator` | Positive directional indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `PlusDirectionalMovement` | Positive directional movement. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx |
| `RateOfChangePercentage` | Period-over-period rate of change as a fraction. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum |
| `RateOfChangeRatio` | Rate-of-change ratio against a prior value. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum |
| `RateOfChangeRatio100` | Rate-of-change ratio scaled by 100. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum |
| `SchaffTrendCycle` | MACD/stochastic cycle oscillator. | https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html |
| `StdDev` | Rolling standard deviation. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility |
| `StochRSI` | Stochastic oscillator applied to RSI values. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochrsi |
| `Stochastic` | Slow stochastic oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochastic-oscillator-fast-slow-and-full |
| `SuperTrend` | ATR-band trend-following indicator. | https://www.investopedia.com/supertrend-indicator-7976167 |
| `Summation` | Rolling sum. | https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.sum.html |
| `T3MovingAverage` | Tillson T3 multi-EMA moving average. | https://efs.kb.esignal.com/hc/en-us/articles/6362957784603-T3-Average |
| `TimeSeriesForecast` | Rolling linear-regression time-series forecast. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope |
| `TrueRange` | Maximum of high-low and gaps from previous close. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/true-range |
| `TriangularMovingAverage` | Double-smoothed triangular moving average. | https://www.marketvolume.com/technicalanalysis/tma.asp |
| `TripleEMA` | TEMA lag-reduced moving average. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/triple-exponential-moving-average-tema |
| `Trix` | Triple-smoothed rate-of-change oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/trix |
| `TypicalPrice` | Average of high, low, and close. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/typical-price |
| `UltimateOscillator` | Weighted multi-window buying-pressure oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ultimate-oscillator |
| `UlcerIndex` | Drawdown-based downside-risk measure. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ulcer-index |
| `Variance` | Rolling variance. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility |
| `VariableIndexDynamicAverage` | VIDYA adaptive EMA using absolute CMO as the smoothing factor. | https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/vida |
| `VolumePriceTrend` | Cumulative volume adjusted by percentage price change. | https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html |
| `VolumeWeightedAveragePrice` | VWAP price weighted by traded volume. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/volume-weighted-average-price-vwap |
| `VolumeWeightedMovingAverage` | VWMA rolling close weighted by volume. | https://trendspider.com/learning-center/what-is-the-volume-weighted-moving-average-vwma/ |
| `Vortex` | Positive/negative Vortex trend movement indicator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/vortex-indicator |
| `WeightedClosePrice` | Weighted close transform using high, low, and close. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/weighted-close |
| `WeightedMovingAverage` | Weighted moving average with larger recent weights. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/hull-moving-average-hma |
| `WilliamsR` | Williams %R overbought/oversold oscillator. | https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/williams-r |
