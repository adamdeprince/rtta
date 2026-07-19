# 算法

本文件列出 `rtta` 导出的公共指标算法；调优与结果辅助类型有意省略。每种算法的详细实现说明和外部参考资料位于 `documentation/algorithms/` 下的独立页面中。

| 算法 | 说明 |
|---|---|
| [`ATR`](documentation/algorithms/atr.zh-CN.md) | 在滚动窗口内计算平均真实波幅，用于衡量波动率。 |
| [`ATRP`](documentation/algorithms/atrp.zh-CN.md) | 以价格百分比表示的平均真实波幅。 |
| [`ATRRegimeDetector`](documentation/algorithms/atr-regime-detector.zh-CN.md) | 带高低回滞区间的有状态 ATR 状态检测器。 |
| [`ADWIN`](documentation/algorithms/adwin.zh-CN.md) | 历史长度有界、可输出位移方向的自适应窗口均值漂移检测器。 |
| [`EMA`](documentation/algorithms/ema.zh-CN.md) | 对近期样本赋予更大权重的指数移动平均。 |
| [`EWMA`](documentation/algorithms/ewma.zh-CN.md) | 通过 alpha、span 或 com 参数化的指数加权移动平均。 |
| [`EWMAZScoreShiftDetector`](documentation/algorithms/ewmaz-score-shift-detector.zh-CN.md) | 用于检测超过阈值位移的因果 EWMA 均值/方差 z 分数事件检测器。 |
| [`MACD`](documentation/algorithms/macd.zh-CN.md) | 指数平滑异同移动平均振荡器及其信号线和柱状图。 |
| [`ROC`](documentation/algorithms/roc.zh-CN.md) | 以回看期内百分比变化表示的变动率动量。 |
| [`RSI`](documentation/algorithms/rsi.zh-CN.md) | 相对强弱指数动量振荡器。 |
| [`SMA`](documentation/algorithms/sma.zh-CN.md) | 滚动窗口内的简单移动平均。 |
| [`TSI`](documentation/algorithms/tsi.zh-CN.md) | 经过双重平滑的真实强弱指数动量振荡器。 |
| [`AbsolutePriceOscillator`](documentation/algorithms/absolute-price-oscillator.zh-CN.md) | 以价格单位表示的快慢移动平均之差。 |
| [`AccumulationDistribution`](documentation/algorithms/accumulation-distribution.zh-CN.md) | 结合成交量与价格的累积/派发线。 |
| [`AlphaBetaGammaTrackingFilter`](documentation/algorithms/alpha-beta-gamma-tracking-filter.zh-CN.md) | 类似稳态卡尔曼滤波的价格、速度和加速度跟踪器。 |
| [`AmihudIlliquidity`](documentation/algorithms/amihud-illiquidity.zh-CN.md) | 单位美元成交量绝对收益的滚动平均。 |
| [`AnchoredVWAP`](documentation/algorithms/anchored-vwap.zh-CN.md) | 从任意锚点或重置事件开始累计的 VWAP，而不是固定时段或滚动窗口 VWAP。 |
| [`Aroon`](documentation/algorithms/aroon.zh-CN.md) | 根据近期最高价与最低价距今时间计算的 Aroon Up/Down 趋势指标。 |
| [`AroonOscillator`](documentation/algorithms/aroon-oscillator.zh-CN.md) | Aroon Up 与 Aroon Down 之差。 |
| [`AverageDirectionalMovementIndex`](documentation/algorithms/average-directional-movement-index.zh-CN.md) | 衡量趋势强度的 ADX 指标。 |
| [`AverageDirectionalMovementIndexRating`](documentation/algorithms/average-directional-movement-index-rating.zh-CN.md) | 对 ADX 进一步平滑得到的 ADXR 趋势强度评级。 |
| [`AveragePrice`](documentation/algorithms/average-price.zh-CN.md) | 开盘价、最高价、最低价和收盘价的平均值。 |
| [`AuctionContinuousMarketTransitionDetector`](documentation/algorithms/auction-continuous-market-transition-detector.zh-CN.md) | 用于区分集合竞价与连续交易阶段信号的回滞检测器。 |
| [`AwesomeOscillator`](documentation/algorithms/awesome-oscillator.zh-CN.md) | 短期与长期中间价移动平均之差。 |
| [`BalanceOfPower`](documentation/algorithms/balance-of-power.zh-CN.md) | 根据开、高、低、收衡量买方与卖方压力。 |
| [`Beta`](documentation/algorithms/beta.zh-CN.md) | 一个序列相对另一序列的滚动 beta。 |
| [`BetaRegimeDetector`](documentation/algorithms/beta-regime-detector.zh-CN.md) | 带上下回滞区间的有状态滚动 beta 状态检测器。 |
| [`BidAskBounceRegimeDetector`](documentation/algorithms/bid-ask-bounce-regime-detector.zh-CN.md) | 通过 EWMA 衡量买卖方向交替，用于检测报价反弹状态。 |
| [`BollingerBands`](documentation/algorithms/bollinger-bands.zh-CN.md) | 根据标准差构造的移动平均包络。 |
| [`BoundedBOCPD`](documentation/algorithms/bounded-bocpd.zh-CN.md) | 使用恒定危险率、内存有界的贝叶斯在线变点检测器。 |
| [`CalibrationDriftDetector`](documentation/algorithms/calibration-drift-detector.zh-CN.md) | 基于 EWMA 的概率校准误差漂移检测器。 |
| [`ChaikinMoneyFlow`](documentation/algorithms/chaikin-money-flow.zh-CN.md) | 窗口内按成交量加权的资金流。 |
| [`ChaikinOscillator`](documentation/algorithms/chaikin-oscillator.zh-CN.md) | 对累积/派发线计算的 MACD 式振荡器。 |
| [`ChandeMomentumOscillator`](documentation/algorithms/chande-momentum-oscillator.zh-CN.md) | 根据近期上涨与下跌幅度之和计算的动量振荡器。 |
| [`ChoppinessIndex`](documentation/algorithms/choppiness-index.zh-CN.md) | 根据真实波幅相对高低价区间衡量震荡与趋势的 CHOP 指标。 |
| [`ClosePressureReversalSignal`](documentation/algorithms/close-pressure-reversal-signal.zh-CN.md) | 日终横截面反转信号，综合当日剩余时段收益、成交量/笔数压力、VWAP 位置和类共形滚动误差区间。 |
| [`CointegrationBreakdownMonitor`](documentation/algorithms/cointegration-breakdown-monitor.zh-CN.md) | 使用 EWMA 对冲比率估计，以流式残差 z 分数监控配对关系失效。 |
| [`ConnorsRSI`](documentation/algorithms/connors-rsi.zh-CN.md) | 对价格 RSI、连续涨跌期 RSI 和单期价格变化百分位排名取平均的复合振荡器。 |
| [`CommodityChannelIndex`](documentation/algorithms/commodity-channel-index.zh-CN.md) | 衡量典型价格偏离其移动平均程度的 CCI。 |
| [`CoppockCurve`](documentation/algorithms/coppock-curve.zh-CN.md) | 长短变动率加权移动平均。 |
| [`Correlation`](documentation/algorithms/correlation.zh-CN.md) | 两个序列之间的滚动 Pearson 相关系数。 |
| [`CorrelationRegimeDetector`](documentation/algorithms/correlation-regime-detector.zh-CN.md) | 带上下回滞区间的有状态滚动相关性状态检测器。 |
| [`CrossAssetCorrelationBreakDetector`](documentation/algorithms/cross-asset-correlation-break-detector.zh-CN.md) | 根据两种资产短期与长期滚动相关性之差检测相关性失效。 |
| [`CumulativeReturn`](documentation/algorithms/cumulative-return.zh-CN.md) | 从第一个收盘价开始计算的累计收益。 |
| [`CUSUM`](documentation/algorithms/cusum.zh-CN.md) | 用于检测超过阈值方向性变动的因果累积和事件滤波器。 |
| [`DDM`](documentation/algorithms/ddm.zh-CN.md) | 面向伯努利预测误差流的漂移检测方法。 |
| [`DailyLogReturn`](documentation/algorithms/daily-log-return.zh-CN.md) | 相邻收盘价之间的对数收益。 |
| [`DailyReturn`](documentation/algorithms/daily-return.zh-CN.md) | 相邻收盘价之间的百分比收益。 |
| [`Delay`](documentation/algorithms/delay.zh-CN.md) | 固定样本数之前的滞后值。 |
| [`DetrendedPriceOscillator`](documentation/algorithms/detrended-price-oscillator.zh-CN.md) | 将价格与错位移动平均比较的 DPO 周期指标。 |
| [`DirectionalMovementIndex`](documentation/algorithms/directional-movement-index.zh-CN.md) | 衡量方向运动趋势强度的 DX 指标。 |
| [`DoubleEMA`](documentation/algorithms/double-ema.zh-CN.md) | 减少滞后的 DEMA 移动平均。 |
| [`DonchianChannel`](documentation/algorithms/donchian-channel.zh-CN.md) | 由滚动最高价与最低价构成的通道。 |
| [`EDDM`](documentation/algorithms/eddm.zh-CN.md) | 根据预测误差之间的距离进行早期漂移检测。 |
| [`EhlersOptimalTrackingFilter`](documentation/algorithms/ehlers-optimal-tracking-filter.zh-CN.md) | 使用 Ehlers 价格不确定性跟踪指数的自适应跟踪滤波器。 |
| [`ElderRayIndex`](documentation/algorithms/elder-ray-index.zh-CN.md) | 以最高价/最低价距收盘价 EMA 的距离衡量多头与空头力量。 |
| [`EaseOfMovement`](documentation/algorithms/ease-of-movement.zh-CN.md) | 衡量价格移动难易程度的成交量/区间指标。 |
| [`ExecutionCostSlippageRegimeDetector`](documentation/algorithms/execution-cost-slippage-regime-detector.zh-CN.md) | 根据成交价相对报价中点的偏离检测相对执行成本/滑点状态。 |
| [`FastStochastic`](documentation/algorithms/fast-stochastic.zh-CN.md) | 快速随机指标 %K/%D 振荡器。 |
| [`FeatureDistributionDriftDetector`](documentation/algorithms/feature-distribution-drift-detector.zh-CN.md) | 面向单个流式特征分布、采用有界 ADWIN 风格的漂移检测器。 |
| [`FibonacciRetracementLevels`](documentation/algorithms/fibonacci-retracement-levels.zh-CN.md) | 近期高低点之间的滚动斐波那契回撤位。 |
| [`FisherTransform`](documentation/algorithms/fisher-transform.zh-CN.md) | 将近期高低区间内的归一化位置变换为转折点振荡器的 Ehlers Fisher 变换。 |
| [`ForceIndex`](documentation/algorithms/force-index.zh-CN.md) | 价格变化乘以成交量的振荡器。 |
| [`FractalAdaptiveMovingAverage`](documentation/algorithms/fractal-adaptive-moving-average.zh-CN.md) | 使用分形维数自适应调整 EMA 平滑的 Ehlers FRAMA。 |
| [`GaussianProcessRegressionBands`](documentation/algorithms/gaussian-process-regression-bands.zh-CN.md) | 使用 RBF 核的滚动高斯过程后验均值及不确定性区间。 |
| [`High`](documentation/algorithms/high.zh-CN.md) | 滚动最高值。 |
| [`HighIndex`](documentation/algorithms/high-index.zh-CN.md) | 滚动最高值的偏移量/索引。 |
| [`HighLow`](documentation/algorithms/high-low.zh-CN.md) | 同时输出滚动最小值与最大值。 |
| [`HighLowIndex`](documentation/algorithms/high-low-index.zh-CN.md) | 同时输出滚动最小值与最大值的偏移量/索引。 |
| [`HDDM`](documentation/algorithms/hddm.zh-CN.md) | 面向伯努利预测误差流的 Hoeffding 界漂移检测器。 |
| [`HeikinAshiTransform`](documentation/algorithms/heikin-ashi-transform.zh-CN.md) | 用于平滑 K 线的增量 Heikin-Ashi OHLC 变换。 |
| [`HiddenSemiMarkovRegimeFilter`](documentation/algorithms/hidden-semi-markov-regime-filter.zh-CN.md) | 带有界持续期偏置的在线高斯隐半马尔可夫类状态滤波器。 |
| [`HitRateDriftDetector`](documentation/algorithms/hit-rate-drift-detector.zh-CN.md) | 使用失误率回滞检测命中率下降的 EWMA 检测器。 |
| [`HullMovingAverage`](documentation/algorithms/hull-moving-average.zh-CN.md) | 减少滞后的 HMA 加权移动平均。 |
| [`Ichimoku`](documentation/algorithms/ichimoku.zh-CN.md) | 一目均衡表的转换线、基准线和先行带分量。 |
| [`IntradayClockEchoSignal`](documentation/algorithms/intraday-clock-echo-signal.zh-CN.md) | 从过去多个聚合 K 线交易日学习的日内相同时刻收益周期性信号。 |
| [`InteractingMultipleModelFilter`](documentation/algorithms/interacting-multiple-model-filter.zh-CN.md) | 以在线概率混合低波动、高波动、趋势和震荡模型的四状态 IMM 卡尔曼跟踪器。 |
| [`KSTOscillator`](documentation/algorithms/kst-oscillator.zh-CN.md) | Pring Know Sure Thing 多变动率平滑振荡器。 |
| [`KalmanExtremumTrend`](documentation/algorithms/kalman-extremum-trend.zh-CN.md) | 将卡尔曼趋势与近期极值区间内随机指标式位置结合。 |
| [`KalmanHedgeRatio`](documentation/algorithms/kalman-hedge-ratio.zh-CN.md) | 在线卡尔曼回归对冲比率及配对价差。 |
| [`KalmanInnovationZScore`](documentation/algorithms/kalman-innovation-z-score.zh-CN.md) | 用预测新息标准差归一化的带符号测量新息。 |
| [`KalmanLocalLinearTrend`](documentation/algorithms/kalman-local-linear-trend.zh-CN.md) | 局部水平/趋势状态空间的卡尔曼估计器。 |
| [`KalmanMovingAverage`](documentation/algorithms/kalman-moving-average.zh-CN.md) | 使用局部线性价格/速度模型的卡尔曼价格滤波器。 |
| [`KalmanPredictionBands`](documentation/algorithms/kalman-prediction-bands.zh-CN.md) | 根据预测测量不确定性给出上下区间的一步卡尔曼预测。 |
| [`KalmanRegressionChannel`](documentation/algorithms/kalman-regression-channel.zh-CN.md) | 带预测通道与价差的在线卡尔曼回归。 |
| [`KalmanTrendSignal`](documentation/algorithms/kalman-trend-signal.zh-CN.md) | 卡尔曼滤波趋势线，以及根据价格相对趋势线位置产生的买卖信号。 |
| [`KalmanVelocityOscillator`](documentation/algorithms/kalman-velocity-oscillator.zh-CN.md) | 恒定速度卡尔曼价格模型中以零为中心的速度状态。 |
| [`Kama`](documentation/algorithms/kama.zh-CN.md) | Kaufman 自适应移动平均。 |
| [`KeltnerChannel`](documentation/algorithms/keltner-channel.zh-CN.md) | 基于 EMA 与 ATR 的波动率通道。 |
| [`KeltnerChannelOriginal`](documentation/algorithms/keltner-channel-original.zh-CN.md) | 原始的 SMA/价格区间 Keltner 通道变体。 |
| [`KlingerVolumeOscillator`](documentation/algorithms/klinger-volume-oscillator.zh-CN.md) | 使用快慢 EMA 和信号线的成交量力振荡器。 |
| [`KSWIN`](documentation/algorithms/kswin.zh-CN.md) | Kolmogorov-Smirnov 滑动窗口漂移检测器。 |
| [`KyleLambda`](documentation/algorithms/kyle-lambda.zh-CN.md) | 收益率相对带符号美元成交量平方根的滚动价格冲击斜率。 |
| [`LeadLagRegimeDetector`](documentation/algorithms/lead-lag-regime-detector.zh-CN.md) | 判断两个序列中哪一个领先的 EWMA 交叉滞后检测器。 |
| [`LiquidityDroughtDetector`](documentation/algorithms/liquidity-drought-detector.zh-CN.md) | 使用下阈值回滞检测相对成交量/深度枯竭。 |
| [`LiquidityRegimeDetector`](documentation/algorithms/liquidity-regime-detector.zh-CN.md) | 以单位美元成交量绝对收益衡量流动性状态的 EWMA 检测器。 |
| [`LinearRegression`](documentation/algorithms/linear-regression.zh-CN.md) | 滚动最小二乘拟合值。 |
| [`LinearRegressionAngle`](documentation/algorithms/linear-regression-angle.zh-CN.md) | 滚动线性回归斜率对应的角度。 |
| [`LinearRegressionIntercept`](documentation/algorithms/linear-regression-intercept.zh-CN.md) | 滚动线性回归拟合的截距。 |
| [`LinearRegressionSlope`](documentation/algorithms/linear-regression-slope.zh-CN.md) | 滚动线性回归拟合的斜率。 |
| [`Low`](documentation/algorithms/low.zh-CN.md) | 滚动最低值。 |
| [`LowIndex`](documentation/algorithms/low-index.zh-CN.md) | 滚动最低值的偏移量/索引。 |
| [`MACDFix`](documentation/algorithms/macd-fix.zh-CN.md) | 移动平均周期固定为 12/26 的 MACD。 |
| [`MassIndex`](documentation/algorithms/mass-index.zh-CN.md) | 根据价格区间扩张识别反转的指标。 |
| [`MarketOpenCloseTransitionDetector`](documentation/algorithms/market-open-close-transition-detector.zh-CN.md) | 根据交易时段进度检测开盘和收盘区间转换。 |
| [`MatchedFlowConformalSignal`](documentation/algorithms/matched-flow-conformal-signal.zh-CN.md) | 日内 OHLCV 匹配流信号，带类共形滚动误差区间和目标仓位诊断。 |
| [`MedianPrice`](documentation/algorithms/median-price.zh-CN.md) | 最高价与最低价的平均值。 |
| [`MesaAdaptiveMovingAverage`](documentation/algorithms/mesa-adaptive-moving-average.zh-CN.md) | 由主导周期相位驱动的 Ehlers MAMA/FAMA 自适应移动平均。 |
| [`MicrostructureNoiseRegimeDetector`](documentation/algorithms/microstructure-noise-regime-detector.zh-CN.md) | 按报价价差归一化的 EWMA 成交价相对中点噪声检测器。 |
| [`MidPoint`](documentation/algorithms/mid-point.zh-CN.md) | 单序列滚动最高值与最低值的中点。 |
| [`MidPrice`](documentation/algorithms/mid-price.zh-CN.md) | 滚动最高价序列与最低价序列的中点。 |
| [`MinusDirectionalIndicator`](documentation/algorithms/minus-directional-indicator.zh-CN.md) | 负方向指标。 |
| [`MinusDirectionalMovement`](documentation/algorithms/minus-directional-movement.zh-CN.md) | 负方向运动。 |
| [`Momentum`](documentation/algorithms/momentum.zh-CN.md) | 当前值与先前值之差。 |
| [`MoneyFlowIndex`](documentation/algorithms/money-flow-index.zh-CN.md) | 类似 RSI、按成交量加权的资金流振荡器。 |
| [`NadarayaWatsonEnvelope`](documentation/algorithms/nadaraya-watson-envelope.zh-CN.md) | 使用高斯核的 Nadaraya-Watson 平滑器及加权残差区间。 |
| [`NegativeVolumeIndex`](documentation/algorithms/negative-volume-index.zh-CN.md) | 只在成交量下降期发生变化的累计指标。 |
| [`NormalizedATR`](documentation/algorithms/normalized-atr.zh-CN.md) | 按收盘价归一化的 ATR。 |
| [`OnBalanceVolume`](documentation/algorithms/on-balance-volume.zh-CN.md) | 根据收盘方向加减成交量得到的累计指标。 |
| [`OnlineGaussianMixtureRegimeFilter`](documentation/algorithms/online-gaussian-mixture-regime-filter.zh-CN.md) | 分量数量有界的在线高斯混合状态滤波器。 |
| [`OnlineHMMRegimeFilter`](documentation/algorithms/online-hmm-regime-filter.zh-CN.md) | 转移持续性固定的在线高斯隐马尔可夫状态滤波器。 |
| [`OnlineMarkovSwitchingVolatilityFilter`](documentation/algorithms/online-markov-switching-volatility-filter.zh-CN.md) | 根据相邻收盘价变动运行的在线双状态马尔可夫切换波动率滤波器。 |
| [`OrderFlowImbalance`](documentation/algorithms/order-flow-imbalance.zh-CN.md) | 滚动报价更新窗口内，最优买卖价和挂单量变化形成的压力。 |
| [`OrderFlowImbalanceRegimeDetector`](documentation/algorithms/order-flow-imbalance-regime-detector.zh-CN.md) | 带买卖压力回滞的 EWMA 订单流失衡状态检测器。 |
| [`PageHinkley`](documentation/algorithms/page-hinkley.zh-CN.md) | 带上涨/下跌方向输出的因果 Page-Hinkley 均值位移事件检测器。 |
| [`PairsSpreadRegimeDetector`](documentation/algorithms/pairs-spread-regime-detector.zh-CN.md) | 使用流式 EWMA 对冲比率残差 z 分数检测配对价差状态。 |
| [`ParticleFilterTrend`](documentation/algorithms/particle-filter-trend.zh-CN.md) | 使用确定性随机种子与拉普拉斯测量似然，并输出有效样本量的粒子趋势滤波器。 |
| [`ParabolicSAR`](documentation/algorithms/parabolic-sar.zh-CN.md) | 抛物线止损反转跟踪趋势指标。 |
| [`PercentagePrice`](documentation/algorithms/percentage-price.zh-CN.md) | 百分比价格振荡器。 |
| [`PercentageVolume`](documentation/algorithms/percentage-volume.zh-CN.md) | 百分比成交量振荡器。 |
| [`PlusDirectionalIndicator`](documentation/algorithms/plus-directional-indicator.zh-CN.md) | 正方向指标。 |
| [`PlusDirectionalMovement`](documentation/algorithms/plus-directional-movement.zh-CN.md) | 正方向运动。 |
| [`PredictionErrorDriftDetector`](documentation/algorithms/prediction-error-drift-detector.zh-CN.md) | 基于 EWMA 的预测绝对误差漂移检测器。 |
| [`QuoteMessageRateRegimeDetector`](documentation/algorithms/quote-message-rate-regime-detector.zh-CN.md) | 检测相对 EWMA 报价消息速率状态。 |
| [`QuoteStuffingDetector`](documentation/algorithms/quote-stuffing-detector.zh-CN.md) | 通过 EWMA 报价/成交消息比率检测报价填塞时段。 |
| [`RateOfChangePercentage`](documentation/algorithms/rate-of-change-percentage.zh-CN.md) | 以小数表示的相邻周期变动率。 |
| [`RateOfChangeRatio`](documentation/algorithms/rate-of-change-ratio.zh-CN.md) | 当前值相对先前值的变动率比率。 |
| [`RateOfChangeRatio100`](documentation/algorithms/rate-of-change-ratio-100.zh-CN.md) | 乘以 100 的变动率比率。 |
| [`RenkoBrickGenerator`](documentation/algorithms/renko-brick-generator.zh-CN.md) | 事件驱动的砖形图价格变换，根据收盘价更新输出带符号砖块数及当前砖块状态。 |
| [`ResidualDriftDetector`](documentation/algorithms/residual-drift-detector.zh-CN.md) | 带符号回滞输出的 EWMA 残差 z 分数漂移检测器。 |
| [`RelativeVigorIndex`](documentation/algorithms/relative-vigor-index.zh-CN.md) | 平滑后的收盘减开盘动量相对高低区间的指标，并带信号线。 |
| [`RealizedVarianceRegimeDetector`](documentation/algorithms/realized-variance-regime-detector.zh-CN.md) | 根据相邻收盘价变化平方计算的滚动已实现方差状态检测器。 |
| [`RollingBetaShiftDetector`](documentation/algorithms/rolling-beta-shift-detector.zh-CN.md) | 因果相邻窗口 beta 位移检测器。 |
| [`RollingCorrelationShiftDetector`](documentation/algorithms/rolling-correlation-shift-detector.zh-CN.md) | 因果相邻窗口相关性位移检测器。 |
| [`RollingMeanShiftDetector`](documentation/algorithms/rolling-mean-shift-detector.zh-CN.md) | 使用双样本 z 分数的因果相邻窗口均值位移检测器。 |
| [`RollingMeanVarianceShiftDetector`](documentation/algorithms/rolling-mean-variance-shift-detector.zh-CN.md) | 因果相邻窗口均值与方差联合位移检测器。 |
| [`RollingSpreadLiquidityShiftDetector`](documentation/algorithms/rolling-spread-liquidity-shift-detector.zh-CN.md) | 因果相邻窗口报价价差/深度流动性压力位移检测器。 |
| [`RollingVarianceShiftDetector`](documentation/algorithms/rolling-variance-shift-detector.zh-CN.md) | 使用方差比对数的因果相邻窗口方差位移检测器。 |
| [`SavitzkyGolayFilter`](documentation/algorithms/savitzky-golay-filter.zh-CN.md) | 滚动多项式最小二乘平滑器，并输出一阶和二阶导数。 |
| [`SchaffTrendCycle`](documentation/algorithms/schaff-trend-cycle.zh-CN.md) | 结合 MACD 与随机指标的周期振荡器。 |
| [`SpreadFeatures`](documentation/algorithms/spread-features.zh-CN.md) | 根据成交及同期报价估计报价价差、有效价差和已实现价差。 |
| [`SpreadExplosionDetector`](documentation/algorithms/spread-explosion-detector.zh-CN.md) | 基于 EWMA 的相对报价价差爆发检测器。 |
| [`SpreadRegimeDetector`](documentation/algorithms/spread-regime-detector.zh-CN.md) | 使用相对买卖价差的有状态报价价差状态检测器。 |
| [`StdDev`](documentation/algorithms/std-dev.zh-CN.md) | 滚动标准差。 |
| [`StickyHMMRegimeFilter`](documentation/algorithms/sticky-hmm-regime-filter.zh-CN.md) | 自转移持续性很高的在线高斯 HMM 状态滤波器。 |
| [`StochRSI`](documentation/algorithms/stoch-rsi.zh-CN.md) | 对 RSI 数值应用随机振荡器。 |
| [`Stochastic`](documentation/algorithms/stochastic.zh-CN.md) | 慢速随机振荡器。 |
| [`SuperTrend`](documentation/algorithms/super-trend.zh-CN.md) | 基于 ATR 区间的趋势跟踪指标。 |
| [`Summation`](documentation/algorithms/summation.zh-CN.md) | 滚动求和。 |
| [`T3MovingAverage`](documentation/algorithms/t-3-moving-average.zh-CN.md) | Tillson T3 多重 EMA 移动平均。 |
| [`ThresholdRegimeDetector`](documentation/algorithms/threshold-regime-detector.zh-CN.md) | 带上下回滞区间的有状态阈值状态检测器。 |
| [`TimeSeriesForecast`](documentation/algorithms/time-series-forecast.zh-CN.md) | 通过滚动线性回归计算的时间序列预测。 |
| [`TradeIntensityRegimeDetector`](documentation/algorithms/trade-intensity-regime-detector.zh-CN.md) | 检测相对 EWMA 成交笔数强度状态。 |
| [`TrendChopRegimeDetector`](documentation/algorithms/trend-chop-regime-detector.zh-CN.md) | 使用真实波幅和效率比率区分趋势与震荡状态。 |
| [`TwoFactorKalmanTrendFilter`](documentation/algorithms/two-factor-kalman-trend-filter.zh-CN.md) | 双状态短期/长期卡尔曼趋势贡献模型。 |
| [`TrueRange`](documentation/algorithms/true-range.zh-CN.md) | 最高价减最低价及相对前收盘价跳空中的最大值。 |
| [`TriangularMovingAverage`](documentation/algorithms/triangular-moving-average.zh-CN.md) | 双重平滑的三角移动平均。 |
| [`TripleEMA`](documentation/algorithms/triple-ema.zh-CN.md) | 减少滞后的 TEMA 移动平均。 |
| [`Trix`](documentation/algorithms/trix.zh-CN.md) | 三重平滑的变动率振荡器。 |
| [`TypicalPrice`](documentation/algorithms/typical-price.zh-CN.md) | 最高价、最低价和收盘价的平均值。 |
| [`UltimateOscillator`](documentation/algorithms/ultimate-oscillator.zh-CN.md) | 多窗口加权买方压力振荡器。 |
| [`UlcerIndex`](documentation/algorithms/ulcer-index.zh-CN.md) | 基于回撤衡量下行风险的指标。 |
| [`VPIN`](documentation/algorithms/vpin.zh-CN.md) | 使用批量成交量分类和滚动成交量桶失衡计算的成交量同步知情交易概率。 |
| [`Variance`](documentation/algorithms/variance.zh-CN.md) | 滚动方差。 |
| [`VariableIndexDynamicAverage`](documentation/algorithms/variable-index-dynamic-average.zh-CN.md) | 使用 CMO 绝对值作为平滑系数的 VIDYA 自适应 EMA。 |
| [`VolatilityBreakoutDetector`](documentation/algorithms/volatility-breakout-detector.zh-CN.md) | 通过 EWMA z 分数检测异常大的相邻收盘价波动率突破。 |
| [`VolatilityCompressionExpansionDetector`](documentation/algorithms/volatility-compression-expansion-detector.zh-CN.md) | 根据短期与长期 EWMA 波动率之比检测压缩和扩张状态。 |
| [`VolatilityRegimeDetector`](documentation/algorithms/volatility-regime-detector.zh-CN.md) | 带高低回滞区间的 EWMA 收盘价变化波动率状态检测器。 |
| [`VolumeProfile`](documentation/algorithms/volume-profile.zh-CN.md) | 滚动价位成交量直方图，输出控制点及价值区域高低边界。 |
| [`VolumePriceTrend`](documentation/algorithms/volume-price-trend.zh-CN.md) | 按价格百分比变化调整的累计成交量。 |
| [`VolumeRegimeDetector`](documentation/algorithms/volume-regime-detector.zh-CN.md) | 带高低回滞区间的 EWMA 相对成交量状态检测器。 |
| [`VolumeWeightedAveragePrice`](documentation/algorithms/volume-weighted-average-price.zh-CN.md) | 按成交量加权的 VWAP 价格。 |
| [`VolumeWeightedMovingAverage`](documentation/algorithms/volume-weighted-moving-average.zh-CN.md) | 以成交量为权重的滚动收盘价 VWMA。 |
| [`Vortex`](documentation/algorithms/vortex.zh-CN.md) | 正向/负向 Vortex 趋势运动指标。 |
| [`WeightedClosePrice`](documentation/algorithms/weighted-close-price.zh-CN.md) | 使用最高价、最低价和收盘价计算的加权收盘价变换。 |
| [`WeightedMovingAverage`](documentation/algorithms/weighted-moving-average.zh-CN.md) | 对近期样本赋予更大权重的加权移动平均。 |
| [`WilliamsR`](documentation/algorithms/williams-r.zh-CN.md) | Williams %R 超买/超卖振荡器。 |
| [`ZigZagSwingDetector`](documentation/algorithms/zig-zag-swing-detector.zh-CN.md) | 基于收盘价的波段检测器，滤除低于百分比阈值的价格变动并输出已确认枢轴点。 |
