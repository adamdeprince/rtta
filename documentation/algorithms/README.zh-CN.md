# 算法文档

本目录收录 RTTA 公共技术分析算法的详细 Markdown 源页面。每个页面都会说明公开的 `update(...)` 调用形式、实际实现的工作原理，以及从 C++ 实现提炼出来、每次用一个样本更新状态的递推公式。

| 算法 | 摘要 |
|---|---|
| [`ATR`](atr.zh-CN.md) | 在滚动窗口内计算平均真实波幅，用于衡量波动率。 |
| [`ATRP`](atrp.zh-CN.md) | 以价格百分比表示的平均真实波幅。 |
| [`ATRRegimeDetector`](atr-regime-detector.zh-CN.md) | 带高低回滞区间的有状态 ATR 状态检测器。 |
| [`ADWIN`](adwin.zh-CN.md) | 历史长度有界、可输出位移方向的自适应窗口均值漂移检测器。 |
| [`EMA`](ema.zh-CN.md) | 对近期样本赋予更大权重的指数移动平均。 |
| [`EWMA`](ewma.zh-CN.md) | 通过 alpha、span 或 com 参数化的指数加权移动平均。 |
| [`EWMAZScoreShiftDetector`](ewmaz-score-shift-detector.zh-CN.md) | 用于检测超过阈值位移的因果 EWMA 均值/方差 z 分数事件检测器。 |
| [`MACD`](macd.zh-CN.md) | 多输出移动平均收敛/发散振荡器，包含 MACD、信号线和柱状图。 |
| [`ROC`](roc.zh-CN.md) | 以回看期内百分比变化表示的变动率动量。 |
| [`RSI`](rsi.zh-CN.md) | 相对强弱指数动量振荡器。 |
| [`SMA`](sma.zh-CN.md) | 滚动窗口内的简单移动平均。 |
| [`TSI`](tsi.zh-CN.md) | 经过双重平滑的真实强弱指数动量振荡器。 |
| [`AbsolutePriceOscillator`](absolute-price-oscillator.zh-CN.md) | 以价格单位表示的快慢移动平均之差。 |
| [`AccumulationDistribution`](accumulation-distribution.zh-CN.md) | 结合成交量与价格的累积/派发线。 |
| [`AlphaBetaGammaTrackingFilter`](alpha-beta-gamma-tracking-filter.zh-CN.md) | 类似稳态卡尔曼滤波的价格、速度和加速度跟踪器。 |
| [`AmihudIlliquidity`](amihud-illiquidity.zh-CN.md) | 单位美元成交量绝对收益的滚动平均。 |
| [`AnchoredVWAP`](anchored-vwap.zh-CN.md) | 从任意锚点或重置事件开始累计的 VWAP，而不是固定时段或滚动窗口 VWAP。 |
| [`Aroon`](aroon.zh-CN.md) | 根据近期最高价与最低价距今时间计算的 Aroon Up/Down 趋势指标。 |
| [`AroonOscillator`](aroon-oscillator.zh-CN.md) | Aroon Up 与 Aroon Down 之差。 |
| [`AverageDirectionalMovementIndex`](average-directional-movement-index.zh-CN.md) | 衡量趋势强度的 ADX 指标。 |
| [`AverageDirectionalMovementIndexRating`](average-directional-movement-index-rating.zh-CN.md) | 对 ADX 进一步平滑得到的 ADXR 趋势强度评级。 |
| [`AveragePrice`](average-price.zh-CN.md) | 开盘价、最高价、最低价和收盘价的平均值。 |
| [`AuctionContinuousMarketTransitionDetector`](auction-continuous-market-transition-detector.zh-CN.md) | 用于区分集合竞价与连续交易阶段信号的回滞检测器。 |
| [`AwesomeOscillator`](awesome-oscillator.zh-CN.md) | 短期与长期中间价移动平均之差。 |
| [`BalanceOfPower`](balance-of-power.zh-CN.md) | 根据开、高、低、收衡量买方与卖方压力。 |
| [`Beta`](beta.zh-CN.md) | 一个序列相对另一序列的滚动 beta。 |
| [`BetaRegimeDetector`](beta-regime-detector.zh-CN.md) | 带上下回滞区间的有状态滚动 beta 状态检测器。 |
| [`BidAskBounceRegimeDetector`](bid-ask-bounce-regime-detector.zh-CN.md) | 通过 EWMA 衡量买卖方向交替，用于检测报价反弹状态。 |
| [`BollingerBands`](bollinger-bands.zh-CN.md) | 根据标准差构造的移动平均包络。 |
| [`BoundedBOCPD`](bounded-bocpd.zh-CN.md) | 使用恒定危险率、内存有界的贝叶斯在线变点检测器。 |
| [`CalibrationDriftDetector`](calibration-drift-detector.zh-CN.md) | 基于 EWMA 的概率校准误差漂移检测器。 |
| [`ChaikinMoneyFlow`](chaikin-money-flow.zh-CN.md) | 窗口内按成交量加权的资金流。 |
| [`ChaikinOscillator`](chaikin-oscillator.zh-CN.md) | 对累积/派发线计算的 MACD 式振荡器。 |
| [`ChandeMomentumOscillator`](chande-momentum-oscillator.zh-CN.md) | 根据近期上涨与下跌幅度之和计算的动量振荡器。 |
| [`ChoppinessIndex`](choppiness-index.zh-CN.md) | 根据真实波幅相对高低价区间衡量震荡与趋势的 CHOP 指标。 |
| [`ClosePressureReversalSignal`](close-pressure-reversal-signal.zh-CN.md) | 日终横截面反转信号，综合当日剩余时段收益、成交量/笔数压力、VWAP 位置和类共形滚动误差区间。 |
| [`CointegrationBreakdownMonitor`](cointegration-breakdown-monitor.zh-CN.md) | 使用 EWMA 对冲比率估计，以流式残差 z 分数监控配对关系失效。 |
| [`ConnorsRSI`](connors-rsi.zh-CN.md) | 对价格 RSI、连续涨跌期 RSI 和单期价格变化百分位排名取平均的复合振荡器。 |
| [`CommodityChannelIndex`](commodity-channel-index.zh-CN.md) | 衡量典型价格偏离其移动平均程度的 CCI。 |
| [`CoppockCurve`](coppock-curve.zh-CN.md) | 长短变动率加权移动平均。 |
| [`Correlation`](correlation.zh-CN.md) | 两个序列之间的滚动 Pearson 相关系数。 |
| [`CorrelationRegimeDetector`](correlation-regime-detector.zh-CN.md) | 带上下回滞区间的有状态滚动相关性状态检测器。 |
| [`CrossAssetCorrelationBreakDetector`](cross-asset-correlation-break-detector.zh-CN.md) | 根据两种资产短期与长期滚动相关性之差检测相关性失效。 |
| [`CumulativeReturn`](cumulative-return.zh-CN.md) | 从第一个收盘价开始计算的累计收益。 |
| [`CUSUM`](cusum.zh-CN.md) | 用于检测超过阈值方向性变动的因果累积和事件滤波器。 |
| [`DDM`](ddm.zh-CN.md) | 面向伯努利预测误差流的漂移检测方法。 |
| [`DailyLogReturn`](daily-log-return.zh-CN.md) | 相邻收盘价之间的对数收益。 |
| [`DailyReturn`](daily-return.zh-CN.md) | 相邻收盘价之间的百分比收益。 |
| [`Delay`](delay.zh-CN.md) | 固定样本数之前的滞后值。 |
| [`DetrendedPriceOscillator`](detrended-price-oscillator.zh-CN.md) | 将价格与错位移动平均比较的 DPO 周期指标。 |
| [`DirectionalMovementIndex`](directional-movement-index.zh-CN.md) | 衡量方向运动趋势强度的 DX 指标。 |
| [`DoubleEMA`](double-ema.zh-CN.md) | 减少滞后的 DEMA 移动平均。 |
| [`DonchianChannel`](donchian-channel.zh-CN.md) | 由滚动最高价与最低价构成的通道。 |
| [`EDDM`](eddm.zh-CN.md) | 根据预测误差之间的距离进行早期漂移检测。 |
| [`EhlersOptimalTrackingFilter`](ehlers-optimal-tracking-filter.zh-CN.md) | 使用 Ehlers 价格不确定性跟踪指数的自适应跟踪滤波器。 |
| [`ElderRayIndex`](elder-ray-index.zh-CN.md) | 以最高价/最低价距收盘价 EMA 的距离衡量多头与空头力量。 |
| [`EaseOfMovement`](ease-of-movement.zh-CN.md) | 衡量价格移动难易程度的成交量/区间指标。 |
| [`ExecutionCostSlippageRegimeDetector`](execution-cost-slippage-regime-detector.zh-CN.md) | 根据成交价相对报价中点的偏离检测相对执行成本/滑点状态。 |
| [`FastStochastic`](fast-stochastic.zh-CN.md) | 快速随机指标 %K/%D 振荡器。 |
| [`FeatureDistributionDriftDetector`](feature-distribution-drift-detector.zh-CN.md) | 面向单个流式特征分布、采用有界 ADWIN 风格的漂移检测器。 |
| [`FibonacciRetracementLevels`](fibonacci-retracement-levels.zh-CN.md) | 近期高低点之间的滚动斐波那契回撤位。 |
| [`FisherTransform`](fisher-transform.zh-CN.md) | 将近期高低区间内的归一化位置变换为转折点振荡器的 Ehlers Fisher 变换。 |
| [`ForceIndex`](force-index.zh-CN.md) | 价格变化乘以成交量的振荡器。 |
| [`FractalAdaptiveMovingAverage`](fractal-adaptive-moving-average.zh-CN.md) | 使用分形维数自适应调整 EMA 平滑的 Ehlers FRAMA。 |
| [`GaussianProcessRegressionBands`](gaussian-process-regression-bands.zh-CN.md) | 使用 RBF 核的滚动高斯过程后验均值及不确定性区间。 |
| [`High`](high.zh-CN.md) | 滚动最高值。 |
| [`HighIndex`](high-index.zh-CN.md) | 滚动最高值的偏移量/索引。 |
| [`HighLow`](high-low.zh-CN.md) | 同时输出滚动最小值与最大值。 |
| [`HighLowIndex`](high-low-index.zh-CN.md) | 同时输出滚动最小值与最大值的偏移量/索引。 |
| [`HDDM`](hddm.zh-CN.md) | 面向伯努利预测误差流的 Hoeffding 界漂移检测器。 |
| [`HeikinAshiTransform`](heikin-ashi-transform.zh-CN.md) | 用于平滑 K 线的增量 Heikin-Ashi OHLC 变换。 |
| [`HiddenSemiMarkovRegimeFilter`](hidden-semi-markov-regime-filter.zh-CN.md) | 带有界持续期偏置的在线高斯隐半马尔可夫类状态滤波器。 |
| [`HitRateDriftDetector`](hit-rate-drift-detector.zh-CN.md) | 使用失误率回滞检测命中率下降的 EWMA 检测器。 |
| [`HullMovingAverage`](hull-moving-average.zh-CN.md) | 减少滞后的 HMA 加权移动平均。 |
| [`Ichimoku`](ichimoku.zh-CN.md) | 一目均衡表的转换线、基准线、原始及位移先行带和迟行带。 |
| [`IntradayClockEchoSignal`](intraday-clock-echo-signal.zh-CN.md) | 从过去多个聚合 K 线交易日学习的日内相同时刻收益周期性信号。 |
| [`InteractingMultipleModelFilter`](interacting-multiple-model-filter.zh-CN.md) | 以在线概率混合低波动、高波动、趋势和震荡模型的四状态 IMM 卡尔曼跟踪器。 |
| [`KSTOscillator`](kst-oscillator.zh-CN.md) | Pring Know Sure Thing 多变动率平滑振荡器。 |
| [`KalmanExtremumTrend`](kalman-extremum-trend.zh-CN.md) | 将卡尔曼趋势与近期极值区间内随机指标式位置结合。 |
| [`KalmanHedgeRatio`](kalman-hedge-ratio.zh-CN.md) | 在线卡尔曼回归对冲比率及配对价差。 |
| [`KalmanInnovationZScore`](kalman-innovation-z-score.zh-CN.md) | 用预测新息标准差归一化的带符号测量新息。 |
| [`KalmanLocalLinearTrend`](kalman-local-linear-trend.zh-CN.md) | 局部水平/趋势状态空间的卡尔曼估计器。 |
| [`KalmanMovingAverage`](kalman-moving-average.zh-CN.md) | 使用局部线性价格/速度模型的卡尔曼价格滤波器。 |
| [`KalmanPredictionBands`](kalman-prediction-bands.zh-CN.md) | 根据预测测量不确定性给出上下区间的一步卡尔曼预测。 |
| [`KalmanRegressionChannel`](kalman-regression-channel.zh-CN.md) | 带预测通道与价差的在线卡尔曼回归。 |
| [`KalmanTrendSignal`](kalman-trend-signal.zh-CN.md) | 卡尔曼滤波趋势线，以及根据价格相对趋势线位置产生的买卖信号。 |
| [`KalmanVelocityOscillator`](kalman-velocity-oscillator.zh-CN.md) | 恒定速度卡尔曼价格模型中以零为中心的速度状态。 |
| [`Kama`](kama.zh-CN.md) | Kaufman 自适应移动平均。 |
| [`KeltnerChannel`](keltner-channel.zh-CN.md) | 基于 EMA 与 ATR 的波动率通道。 |
| [`KeltnerChannelOriginal`](keltner-channel-original.zh-CN.md) | 原始的 SMA/价格区间 Keltner 通道变体。 |
| [`KlingerVolumeOscillator`](klinger-volume-oscillator.zh-CN.md) | 使用快慢 EMA 和信号线的成交量力振荡器。 |
| [`KSWIN`](kswin.zh-CN.md) | Kolmogorov-Smirnov 滑动窗口漂移检测器。 |
| [`KyleLambda`](kyle-lambda.zh-CN.md) | 收益率相对带符号美元成交量平方根的滚动价格冲击斜率。 |
| [`LeadLagRegimeDetector`](lead-lag-regime-detector.zh-CN.md) | 判断两个序列中哪一个领先的 EWMA 交叉滞后检测器。 |
| [`LiquidityDroughtDetector`](liquidity-drought-detector.zh-CN.md) | 使用下阈值回滞检测相对成交量/深度枯竭。 |
| [`LiquidityRegimeDetector`](liquidity-regime-detector.zh-CN.md) | 以单位美元成交量绝对收益衡量流动性状态的 EWMA 检测器。 |
| [`LinearRegression`](linear-regression.zh-CN.md) | 滚动最小二乘拟合值。 |
| [`LinearRegressionAngle`](linear-regression-angle.zh-CN.md) | 滚动线性回归斜率对应的角度。 |
| [`LinearRegressionIntercept`](linear-regression-intercept.zh-CN.md) | 滚动线性回归拟合的截距。 |
| [`LinearRegressionSlope`](linear-regression-slope.zh-CN.md) | 滚动线性回归拟合的斜率。 |
| [`Low`](low.zh-CN.md) | 滚动最低值。 |
| [`LowIndex`](low-index.zh-CN.md) | 滚动最低值的偏移量/索引。 |
| [`MACDFix`](macd-fix.zh-CN.md) | 快慢 EMA 周期固定为 12/26、信号周期可配置的 MACD。 |
| [`MassIndex`](mass-index.zh-CN.md) | 根据价格区间扩张识别反转的指标。 |
| [`MarketOpenCloseTransitionDetector`](market-open-close-transition-detector.zh-CN.md) | 根据交易时段进度检测开盘和收盘区间转换。 |
| [`MatchedFlowConformalSignal`](matched-flow-conformal-signal.zh-CN.md) | 日内 OHLCV 匹配流信号，带类共形滚动误差区间和目标仓位诊断。 |
| [`MedianPrice`](median-price.zh-CN.md) | 最高价与最低价的平均值。 |
| [`MesaAdaptiveMovingAverage`](mesa-adaptive-moving-average.zh-CN.md) | 由主导周期相位驱动的 Ehlers MAMA/FAMA 自适应移动平均。 |
| [`MicrostructureNoiseRegimeDetector`](microstructure-noise-regime-detector.zh-CN.md) | 按报价价差归一化的 EWMA 成交价相对中点噪声检测器。 |
| [`MidPoint`](mid-point.zh-CN.md) | 单序列滚动最高值与最低值的中点。 |
| [`MidPrice`](mid-price.zh-CN.md) | 滚动最高价序列与最低价序列的中点。 |
| [`MinusDirectionalIndicator`](minus-directional-indicator.zh-CN.md) | 负方向指标。 |
| [`MinusDirectionalMovement`](minus-directional-movement.zh-CN.md) | 负方向运动。 |
| [`Momentum`](momentum.zh-CN.md) | 当前值与先前值之差。 |
| [`MoneyFlowIndex`](money-flow-index.zh-CN.md) | 类似 RSI、按成交量加权的资金流振荡器。 |
| [`NadarayaWatsonEnvelope`](nadaraya-watson-envelope.zh-CN.md) | 使用高斯核的 Nadaraya-Watson 平滑器及加权残差区间。 |
| [`NegativeVolumeIndex`](negative-volume-index.zh-CN.md) | 只在成交量下降期发生变化的累计指标。 |
| [`NormalizedATR`](normalized-atr.zh-CN.md) | 按收盘价归一化的 ATR。 |
| [`OnBalanceVolume`](on-balance-volume.zh-CN.md) | 根据收盘方向加减成交量得到的累计指标。 |
| [`OnlineGaussianMixtureRegimeFilter`](online-gaussian-mixture-regime-filter.zh-CN.md) | 分量数量有界的在线高斯混合状态滤波器。 |
| [`OnlineHMMRegimeFilter`](online-hmm-regime-filter.zh-CN.md) | 转移持续性固定的在线高斯隐马尔可夫状态滤波器。 |
| [`OnlineMarkovSwitchingVolatilityFilter`](online-markov-switching-volatility-filter.zh-CN.md) | 根据相邻收盘价变动运行的在线双状态马尔可夫切换波动率滤波器。 |
| [`OrderFlowImbalance`](order-flow-imbalance.zh-CN.md) | 滚动报价更新窗口内，最优买卖价和挂单量变化形成的压力。 |
| [`OrderFlowImbalanceRegimeDetector`](order-flow-imbalance-regime-detector.zh-CN.md) | 带买卖压力回滞的 EWMA 订单流失衡状态检测器。 |
| [`PageHinkley`](page-hinkley.zh-CN.md) | 带上涨/下跌方向输出的因果 Page-Hinkley 均值位移事件检测器。 |
| [`PairsSpreadRegimeDetector`](pairs-spread-regime-detector.zh-CN.md) | 使用流式 EWMA 对冲比率残差 z 分数检测配对价差状态。 |
| [`ParticleFilterTrend`](particle-filter-trend.zh-CN.md) | 使用确定性随机种子与拉普拉斯测量似然，并输出有效样本量的粒子趋势滤波器。 |
| [`ParabolicSAR`](parabolic-sar.zh-CN.md) | 抛物线止损反转跟踪趋势指标。 |
| [`PercentagePrice`](percentage-price.zh-CN.md) | 百分比价格振荡器。 |
| [`PercentageVolume`](percentage-volume.zh-CN.md) | 百分比成交量振荡器。 |
| [`PlusDirectionalIndicator`](plus-directional-indicator.zh-CN.md) | 正方向指标。 |
| [`PlusDirectionalMovement`](plus-directional-movement.zh-CN.md) | 正方向运动。 |
| [`PredictionErrorDriftDetector`](prediction-error-drift-detector.zh-CN.md) | 基于 EWMA 的预测绝对误差漂移检测器。 |
| [`QuoteMessageRateRegimeDetector`](quote-message-rate-regime-detector.zh-CN.md) | 检测相对 EWMA 报价消息速率状态。 |
| [`QuoteStuffingDetector`](quote-stuffing-detector.zh-CN.md) | 通过 EWMA 报价/成交消息比率检测报价填塞时段。 |
| [`RateOfChangePercentage`](rate-of-change-percentage.zh-CN.md) | 以小数表示的相邻周期变动率。 |
| [`RateOfChangeRatio`](rate-of-change-ratio.zh-CN.md) | 当前值相对先前值的变动率比率。 |
| [`RateOfChangeRatio100`](rate-of-change-ratio-100.zh-CN.md) | 乘以 100 的变动率比率。 |
| [`RenkoBrickGenerator`](renko-brick-generator.zh-CN.md) | 事件驱动的砖形图价格变换，根据收盘价更新输出带符号砖块数及当前砖块状态。 |
| [`ResidualDriftDetector`](residual-drift-detector.zh-CN.md) | 带符号回滞输出的 EWMA 残差 z 分数漂移检测器。 |
| [`RelativeVigorIndex`](relative-vigor-index.zh-CN.md) | 平滑后的收盘减开盘动量相对高低区间的指标，并带信号线。 |
| [`RealizedVarianceRegimeDetector`](realized-variance-regime-detector.zh-CN.md) | 根据相邻收盘价变化平方计算的滚动已实现方差状态检测器。 |
| [`RollingBetaShiftDetector`](rolling-beta-shift-detector.zh-CN.md) | 因果相邻窗口 beta 位移检测器。 |
| [`RollingCorrelationShiftDetector`](rolling-correlation-shift-detector.zh-CN.md) | 因果相邻窗口相关性位移检测器。 |
| [`RollingMeanShiftDetector`](rolling-mean-shift-detector.zh-CN.md) | 使用双样本 z 分数的因果相邻窗口均值位移检测器。 |
| [`RollingMeanVarianceShiftDetector`](rolling-mean-variance-shift-detector.zh-CN.md) | 因果相邻窗口均值与方差联合位移检测器。 |
| [`RollingSpreadLiquidityShiftDetector`](rolling-spread-liquidity-shift-detector.zh-CN.md) | 因果相邻窗口报价价差/深度流动性压力位移检测器。 |
| [`RollingVarianceShiftDetector`](rolling-variance-shift-detector.zh-CN.md) | 使用方差比对数的因果相邻窗口方差位移检测器。 |
| [`SavitzkyGolayFilter`](savitzky-golay-filter.zh-CN.md) | 滚动多项式最小二乘平滑器，并输出一阶和二阶导数。 |
| [`SchaffTrendCycle`](schaff-trend-cycle.zh-CN.md) | 结合 MACD 与随机指标的周期振荡器。 |
| [`SpreadFeatures`](spread-features.zh-CN.md) | 根据成交及同期报价估计报价价差、有效价差和已实现价差。 |
| [`SpreadExplosionDetector`](spread-explosion-detector.zh-CN.md) | 基于 EWMA 的相对报价价差爆发检测器。 |
| [`SpreadRegimeDetector`](spread-regime-detector.zh-CN.md) | 使用相对买卖价差的有状态报价价差状态检测器。 |
| [`StdDev`](std-dev.zh-CN.md) | 滚动标准差。 |
| [`StickyHMMRegimeFilter`](sticky-hmm-regime-filter.zh-CN.md) | 自转移持续性很高的在线高斯 HMM 状态滤波器。 |
| [`StochRSI`](stoch-rsi.zh-CN.md) | 对 RSI 数值应用随机振荡器。 |
| [`Stochastic`](stochastic.zh-CN.md) | 慢速随机振荡器。 |
| [`SuperTrend`](super-trend.zh-CN.md) | 基于 ATR 区间的趋势跟踪指标。 |
| [`Summation`](summation.zh-CN.md) | 滚动求和。 |
| [`T3MovingAverage`](t-3-moving-average.zh-CN.md) | Tillson T3 多重 EMA 移动平均。 |
| [`ThresholdRegimeDetector`](threshold-regime-detector.zh-CN.md) | 带上下回滞区间的有状态阈值状态检测器。 |
| [`TimeSeriesForecast`](time-series-forecast.zh-CN.md) | 通过滚动线性回归计算的时间序列预测。 |
| [`TradeIntensityRegimeDetector`](trade-intensity-regime-detector.zh-CN.md) | 检测相对 EWMA 成交笔数强度状态。 |
| [`TrendChopRegimeDetector`](trend-chop-regime-detector.zh-CN.md) | 使用真实波幅和效率比率区分趋势与震荡状态。 |
| [`TwoFactorKalmanTrendFilter`](two-factor-kalman-trend-filter.zh-CN.md) | 双状态短期/长期卡尔曼趋势贡献模型。 |
| [`TrueRange`](true-range.zh-CN.md) | 最高价减最低价及相对前收盘价跳空中的最大值。 |
| [`TriangularMovingAverage`](triangular-moving-average.zh-CN.md) | 双重平滑的三角移动平均。 |
| [`TripleEMA`](triple-ema.zh-CN.md) | 减少滞后的 TEMA 移动平均。 |
| [`Trix`](trix.zh-CN.md) | 三重平滑的变动率振荡器。 |
| [`TypicalPrice`](typical-price.zh-CN.md) | 最高价、最低价和收盘价的平均值。 |
| [`UltimateOscillator`](ultimate-oscillator.zh-CN.md) | 多窗口加权买方压力振荡器。 |
| [`UlcerIndex`](ulcer-index.zh-CN.md) | 基于回撤衡量下行风险的指标。 |
| [`VPIN`](vpin.zh-CN.md) | 使用批量成交量分类和滚动成交量桶失衡计算的成交量同步知情交易概率。 |
| [`Variance`](variance.zh-CN.md) | 滚动方差。 |
| [`VariableIndexDynamicAverage`](variable-index-dynamic-average.zh-CN.md) | 使用 CMO 绝对值作为平滑系数的 VIDYA 自适应 EMA。 |
| [`VolatilityBreakoutDetector`](volatility-breakout-detector.zh-CN.md) | 通过 EWMA z 分数检测异常大的相邻收盘价波动率突破。 |
| [`VolatilityCompressionExpansionDetector`](volatility-compression-expansion-detector.zh-CN.md) | 根据短期与长期 EWMA 波动率之比检测压缩和扩张状态。 |
| [`VolatilityRegimeDetector`](volatility-regime-detector.zh-CN.md) | 带高低回滞区间的 EWMA 收盘价变化波动率状态检测器。 |
| [`VolumeProfile`](volume-profile.zh-CN.md) | 滚动价位成交量直方图，输出控制点及价值区域高低边界。 |
| [`VolumePriceTrend`](volume-price-trend.zh-CN.md) | 按价格百分比变化调整的累计成交量。 |
| [`VolumeRegimeDetector`](volume-regime-detector.zh-CN.md) | 带高低回滞区间的 EWMA 相对成交量状态检测器。 |
| [`VolumeWeightedAveragePrice`](volume-weighted-average-price.zh-CN.md) | 按成交量加权的 VWAP 价格。 |
| [`VolumeWeightedMovingAverage`](volume-weighted-moving-average.zh-CN.md) | 以成交量为权重的滚动收盘价 VWMA。 |
| [`Vortex`](vortex.zh-CN.md) | 正向/负向 Vortex 趋势运动指标。 |
| [`WeightedClosePrice`](weighted-close-price.zh-CN.md) | 使用最高价、最低价和收盘价计算的加权收盘价变换。 |
| [`WeightedMovingAverage`](weighted-moving-average.zh-CN.md) | 对近期样本赋予更大权重的加权移动平均。 |
| [`WilliamsR`](williams-r.zh-CN.md) | Williams %R 超买/超卖振荡器。 |
| [`ZigZagSwingDetector`](zig-zag-swing-detector.zh-CN.md) | 基于收盘价的波段检测器，滤除低于百分比阈值的价格变动并输出已确认枢轴点。 |
| [`AccelerationBands`](acceleration-bands.zh-CN.md) | TA-Lib 风格加速带：对缩放后的最高价和最低价极值计算 SMA，并以 SMA 为中轨。 |
| [`AcceleratorOscillator`](accelerator-oscillator.zh-CN.md) | Bill Williams 加速振荡器：Awesome Oscillator 减去其 SMA。 |
| [`AccumulativeSwingIndex`](accumulative-swing-index.zh-CN.md) | Wilder 摆动指数的累积和。 |
| [`Alligator`](alligator.zh-CN.md) | Bill Williams 鳄鱼线：中间价的位移 SMMA 颚线、齿线和唇线。 |
| [`AndrewsPitchfork`](andrews-pitchfork.zh-CN.md) | 根据百分比波段枢轴点流式构建 Andrews 叉形线中线和平行通道。 |
| [`ArnaudLegouxMovingAverage`](arnaud-legoux-moving-average.zh-CN.md) | 由 offset 和 sigma 控制高斯权重的 Arnaud Legoux 移动平均。 |
| [`BearsPower`](bears-power.zh-CN.md) | Elder 空头力量：最低价减收盘价 EMA。 |
| [`Bias`](bias.zh-CN.md) | 价格相对其简单移动平均的百分比偏差。 |
| [`BollingerBandwidth`](bollinger-bandwidth.zh-CN.md) | 布林带宽度：滚动均值与标准差包络的（上轨−下轨）/中轨。 |
| [`BollingerPercentB`](bollinger-percent-b.zh-CN.md) | 布林 %B：价格在滚动均值与标准差包络中的位置。 |
| [`BullsPower`](bulls-power.zh-CN.md) | Elder 多头力量：最高价减收盘价 EMA。 |
| [`CamarillaPivotPoints`](camarilla-pivot-points.zh-CN.md) | 根据前一根 K 线 HLC 计算 Camarilla 支撑位和阻力位。 |
| [`CDL3BlackCrows`](cdl-3-black-crows.zh-CN.md) | 三只乌鸦：连续三根逐步下行的阴线实体。 |
| [`CDL3Inside`](cdl-3-inside.zh-CN.md) | 三内部上涨/下跌：孕线加确认 K 线。 |
| [`CDL3Outside`](cdl-3-outside.zh-CN.md) | 三外部上涨/下跌：吞没形态加确认 K 线。 |
| [`CDL3WhiteSoldiers`](cdl-3-white-soldiers.zh-CN.md) | 三白兵（红三兵）：连续三根逐步上行的阳线实体。 |
| [`CDLBeltHold`](cdl-belt-hold.zh-CN.md) | 捉腰带线：在极值处开盘并形成长实体。 |
| [`CDLClosingMarubozu`](cdl-closing-marubozu.zh-CN.md) | 收盘光头光脚线：实体收于或接近 K 线极值。 |
| [`CDLCounterAttack`](cdl-counter-attack.zh-CN.md) | 反击线：方向相反的长实体收于前一收盘价附近。 |
| [`CDLDarkCloudCover`](cdl-dark-cloud-cover.zh-CN.md) | 乌云盖顶：阴线收盘穿过前一根阳线实体中点。 |
| [`CDLDojiStar`](cdl-doji-star.zh-CN.md) | 十字星：长实体之后出现十字线，提示反转风险。 |
| [`CDLDoji`](cdl-doji.zh-CN.md) | 十字线：真实实体相对全幅极小，表示犹豫。 |
| [`CDLDragonflyDoji`](cdl-dragonfly-doji.zh-CN.md) | 蜻蜓十字：带长下影线的十字线，表示看涨拒绝。 |
| [`CDLEngulfing`](cdl-engulfing.zh-CN.md) | 吞没形态：当前实体完全吞没前一实体，提示反转。 |
| [`CDLEveningDojiStar`](cdl-evening-doji-star.zh-CN.md) | 黄昏十字星：中间 K 线为十字线的黄昏星。 |
| [`CDLEveningStar`](cdl-evening-star.zh-CN.md) | 黄昏星：三根 K 线组成的看跌反转形态。 |
| [`CDLGravestoneDoji`](cdl-gravestone-doji.zh-CN.md) | 墓碑十字：带长上影线的十字线，表示看跌拒绝。 |
| [`CDLHammer`](cdl-hammer.zh-CN.md) | 锤头线：下跌趋势中带长下影线的看涨形态。 |
| [`CDLHangingMan`](cdl-hanging-man.zh-CN.md) | 上吊线：上涨趋势中出现锤头外形的看跌形态。 |
| [`CDLHaramiCross`](cdl-harami-cross.zh-CN.md) | 十字孕线：前一实体内部出现十字线。 |
| [`CDLHarami`](cdl-harami.zh-CN.md) | 孕线：小实体位于前一实体内部，表示反转或停顿。 |
| [`CDLHighWave`](cdl-high-wave.zh-CN.md) | 高浪线：极小实体配合很长的影线。 |
| [`CDLInvertedHammer`](cdl-inverted-hammer.zh-CN.md) | 倒锤头线：下跌趋势中带长上影线的看涨形态。 |
| [`CDLLongLeggedDoji`](cdl-long-legged-doji.zh-CN.md) | 长脚十字：上下影线都很长的十字线。 |
| [`CDLLongLine`](cdl-long-line.zh-CN.md) | 长实体线：真实实体相对近期平均实体较大。 |
| [`CDLMarubozu`](cdl-marubozu.zh-CN.md) | 光头光脚线：全幅主要由真实实体构成，方向性很强。 |
| [`CDLMatchingLow`](cdl-matching-low.zh-CN.md) | 相同低价：两根阴线收盘价相近，提示支撑。 |
| [`CDLMorningDojiStar`](cdl-morning-doji-star.zh-CN.md) | 晨星十字：中间 K 线为十字线的晨星。 |
| [`CDLMorningStar`](cdl-morning-star.zh-CN.md) | 晨星：三根 K 线组成的看涨反转形态。 |
| [`CDLPatternPack`](cdl-pattern-pack.zh-CN.md) | 一次 OHLC 更新即可计算多种常见 CDL 形态的多输出组合。 |
| [`CDLPiercing`](cdl-piercing.zh-CN.md) | 刺透形态：阳线收盘穿过前一根阴线实体中点。 |
| [`CDLShootingStar`](cdl-shooting-star.zh-CN.md) | 流星线：上涨趋势中带长上影线的看跌形态。 |
| [`CDLShortLine`](cdl-short-line.zh-CN.md) | 短实体线：真实实体相对近期平均实体较小。 |
| [`CDLSpinningTop`](cdl-spinning-top.zh-CN.md) | 纺锤线：小实体配合上下影线，表示犹豫。 |
| [`CDLTriStar`](cdl-tri-star.zh-CN.md) | 三星十字：连续三根十字线构成的反转形态。 |
| [`ChaikinVolatility`](chaikin-volatility.zh-CN.md) | 最高价—最低价区间 EMA 的百分比变化率。 |
| [`ChandeForecastOscillator`](chande-forecast-oscillator.zh-CN.md) | Chande 预测振荡器：收盘价距时间序列预测值的百分比。 |
| [`ChandelierExit`](chandelier-exit.zh-CN.md) | 根据滚动高低极值构建的 ATR 跟踪型多头/空头吊灯止损位。 |
| [`ComparativeRelativeStrength`](comparative-relative-strength.zh-CN.md) | 两个价格序列的比率，即 A/B 比较相对强弱。 |
| [`ConformalBands`](conformal-bands.zh-CN.md) | 以 SMA 为中心、滚动绝对残差分位数为半径的流式拆分保形式预测带。 |
| [`CrossAssetOrderFlowImbalance`](cross-asset-order-flow-imbalance.zh-CN.md) | 本资产收益率对同类资产 OFI 的滚动 beta，以及隐含冲击和残差。 |
| [`DeMarker`](de-marker.zh-CN.md) | 根据最高价向上延伸和最低价向下延伸压力构造的 DeMarker 振荡器。 |
| [`DecomposedOrderFlowImbalance`](decomposed-order-flow-imbalance.zh-CN.md) | 将 Cont 风格报价压力分解为新增、撤单和成交通道。 |
| [`DirectionalChangeDetector`](directional-change-detector.zh-CN.md) | 带超调跟踪的方向变化内在时间事件检测器。 |
| [`DollarBarGenerator`](dollar-bar-generator.zh-CN.md) | 价格×成交量累计到阈值时结束的信息驱动成交额 K 线。 |
| [`DollarRunBarGenerator`](dollar-run-bar-generator.zh-CN.md) | 同号的价格绝对值×成交量累计到阈值时结束的成交额游程 K 线。 |
| [`EfficiencyRatio`](efficiency-ratio.zh-CN.md) | 滚动窗口内净方向移动与路径长度之比，即 Kaufman 效率比率。 |
| [`EhlersCenterOfGravity`](ehlers-center-of-gravity.zh-CN.md) | 滚动价格窗口的 Ehlers 重心振荡器，带滞后触发线。 |
| [`EhlersCyberCycle`](ehlers-cyber-cycle.zh-CN.md) | 带触发线的 Ehlers Cyber Cycle 带通式周期振荡器。 |
| [`EhlersDecycler`](ehlers-decycler.zh-CN.md) | Ehlers 去周期趋势估计及残差振荡器。 |
| [`EhlersInstantaneousTrendline`](ehlers-instantaneous-trendline.zh-CN.md) | 带两根 K 线外推触发线的 Ehlers 瞬时趋势线。 |
| [`EhlersRoofingFilter`](ehlers-roofing-filter.zh-CN.md) | Ehlers Roofing 滤波器：高通级加 Super Smoother 低通级。 |
| [`EhlersSuperSmoother`](ehlers-super-smoother.zh-CN.md) | Ehlers 二极点 Super Smoother 低通滤波器。 |
| [`ElderThermometer`](elder-thermometer.zh-CN.md) | Elder K 线区间温度计：当前区间相对前一区间的比率与升温标志。 |
| [`FibonacciPivotPoints`](fibonacci-pivot-points.zh-CN.md) | 根据前一根 K 线 HLC 计算斐波那契支撑位和阻力位。 |
| [`FlowPressureCapacitySignal`](flow-pressure-capacity-signal.zh-CN.md) | 按事件时间比较主动订单流与对手方 L1 容量，并修正队列补单、撤单脆弱性和瞬时失衡。 |
| [`FOCuS`](focus.zh-CN.md) | 带候选项修剪的函数式在线 CUSUM 均值变点检测器。 |
| [`FourierResidueIdentity`](fourier-residue-identity.zh-CN.md) | 傅里叶—剩余类恒等式：把收益率自相关拆成可检验的方向（符号，k=2）与幅度（k=4）通道，并逐通道计算 Fejér 方差比。 |
| [`GatorOscillator`](gator-oscillator.zh-CN.md) | 根据鳄鱼线颚线—齿线与齿线—唇线距离计算 Bill Williams 鳄鱼振荡器。 |
| [`GeometricMovingAverage`](geometric-moving-average.zh-CN.md) | 对数价格 SMA 取指数所得几何移动平均。 |
| [`GuppyMMARibbon`](guppy-mma-ribbon.zh-CN.md) | 完整顾比 MMA 带：六条短期 EMA、六条长期 EMA 及两组平均值。 |
| [`GuppyMultipleMovingAverage`](guppy-multiple-moving-average.zh-CN.md) | 顾比 MMA 的短期/长期 EMA 组平均值及其价差。 |
| [`HawkesIntensity`](hawkes-intensity.zh-CN.md) | 用于事件时间的指数型 Hawkes 自激强度过程。 |
| [`HilbertDominantCyclePeriod`](hilbert-dominant-cycle-period.zh-CN.md) | 与 TA-Lib HT_DCPERIOD 兼容的主导周期长度。 |
| [`HilbertDominantCyclePhase`](hilbert-dominant-cycle-phase.zh-CN.md) | 与 TA-Lib HT_DCPHASE 兼容、以度表示的主导周期相位。 |
| [`HilbertPhasor`](hilbert-phasor.zh-CN.md) | 与 TA-Lib HT_PHASOR 兼容的同相分量和正交分量。 |
| [`HilbertSineWave`](hilbert-sine-wave.zh-CN.md) | 与 TA-Lib HT_SINE 兼容的正弦波和超前正弦波。 |
| [`HilbertTrendMode`](hilbert-trend-mode.zh-CN.md) | 与 TA-Lib HT_TRENDMODE 兼容的趋势/周期模式标志（1=趋势，0=周期）。 |
| [`HilbertTrendline`](hilbert-trendline.zh-CN.md) | 与 TA-Lib HT_TRENDLINE 兼容的瞬时趋势线。 |
| [`HistoricalVolatility`](historical-volatility.zh-CN.md) | 对数收益率滚动标准差的年化值。 |
| [`ImbalanceBarGenerator`](imbalance-bar-generator.zh-CN.md) | 有符号成交量绝对值达到阈值时结束的成交量失衡 K 线。 |
| [`Inertia`](inertia.zh-CN.md) | Dorsey 惯性指标：相对波动率指数的线性回归。 |
| [`IntegratedOrderFlowImbalance`](integrated-order-flow-imbalance.zh-CN.md) | 投影到在线第一主成分上的多档 Cont OFI。 |
| [`IntradayIntensity`](intraday-intensity.zh-CN.md) | 滚动成交量加权日内强度（2C−H−L）/（H−L）。 |
| [`IntradayMomentumIndex`](intraday-momentum-index.zh-CN.md) | 按每根 K 线开盘到收盘涨跌构造的 RSI 风格振荡器。 |
| [`InverseFisherRSI`](inverse-fisher-rsi.zh-CN.md) | 对 RSI 应用反 Fisher 变换，以产生更清晰的转折点。 |
| [`KagiChart`](kagi-chart.zh-CN.md) | 流式卡吉线、方向与反转事件。 |
| [`KalmanInnovationResidualBOCPD`](kalman-innovation-residual-bocpd.zh-CN.md) | 把 Kalman 新息 z 分数残差送入 ResidualBOCPD 变点检测。 |
| [`KalmanInnovationResidualFOCuS`](kalman-innovation-residual-focus.zh-CN.md) | 把 Kalman 新息 z 分数残差送入 FOCuS 变点检测。 |
| [`MACDExt`](macd-ext.zh-CN.md) | 快线、慢线和信号线均可选择 SMA/EMA 类型的 MACD。 |
| [`MarketFacilitationIndex`](market-facilitation-index.zh-CN.md) | K 线区间除以成交量，即 Bill Williams 市场促进指数。 |
| [`McGinleyDynamic`](mc-ginley-dynamic.zh-CN.md) | 趋势中自动加快、震荡中减慢的 McGinley Dynamic 自适应移动平均。 |
| [`MessageEventOrderFlowImbalance`](message-event-order-flow-imbalance.zh-CN.md) | 根据离散限价订单簿/成交消息事件（新增、撤单、成交）滚动累积 OFI。 |
| [`MovingAverageEnvelope`](moving-average-envelope.zh-CN.md) | 简单移动平均线上下的百分比包络带。 |
| [`MovingAverageVariablePeriod`](moving-average-variable-period.zh-CN.md) | 每根 K 线周期可变的 SMA（TA-Lib MAVP 风格）。 |
| [`MultiLevelOrderFlowImbalance`](multi-level-order-flow-imbalance.zh-CN.md) | 各订单簿档位的 Cont 风格订单流失衡，并提供总和/均值。 |
| [`MultiPeerOrderFlowImbalance`](multi-peer-order-flow-imbalance.zh-CN.md) | 等权同类资产篮子 OFI，以及其对本资产收益率的滚动 beta 冲击。 |
| [`ParabolicSARExtended`](parabolic-sar-extended.zh-CN.md) | 多头/空头使用独立 AF 链的扩展抛物线 SAR（SAREXT 风格）。 |
| [`PivotPoints`](pivot-points.zh-CN.md) | 根据前一根 K 线计算的经典场内枢轴点（PP/R1–R3/S1–S3）。 |
| [`PointAndFigure`](point-and-figure.zh-CN.md) | 流式点数图格值价格、方向和反转。 |
| [`PositiveVolumeIndex`](positive-volume-index.zh-CN.md) | 仅在成交量增加期间发生变化的累积指标。 |
| [`PrettyGoodOscillator`](pretty-good-oscillator.zh-CN.md) | 收盘价减 SMA，再按 ATR 标准化的 PGO。 |
| [`ProjectionOscillator`](projection-oscillator.zh-CN.md) | 收盘价在最高价和最低价线性回归投影带内的随机指标式振荡器。 |
| [`PsychologicalLine`](psychological-line.zh-CN.md) | 滚动窗口内上涨收盘 K 线的百分比。 |
| [`QStick`](q-stick.zh-CN.md) | 收盘价减开盘价的简单移动平均。 |
| [`RainbowMovingAverage`](rainbow-moving-average.zh-CN.md) | Mel Widner 彩虹线：递归 SMA 层及其外层、最高、最低、中点和宽度。 |
| [`RainbowOscillator`](rainbow-oscillator.zh-CN.md) | 彩虹振荡器：递归 SMA 各层的百分比宽度和价格位置。 |
| [`RandomWalkIndex`](random-walk-index.zh-CN.md) | 相对于 ATR 缩放区间的随机游走指数高侧/低侧值。 |
| [`RangeActionVerificationIndex`](range-action-verification-index.zh-CN.md) | RAVI：短期与长期 SMA 绝对距离占长期 SMA 的百分比。 |
| [`RelativeVolatilityIndex`](relative-volatility-index.zh-CN.md) | 对收盘价滚动标准差应用 RSI 风格计算的相对波动率指数。 |
| [`ResidualBOCPD`](residual-bocpd.zh-CN.md) | 应用于残差/新息序列的有界 BOCPD 变点检测器。 |
| [`ResidualFOCuS`](residual-focus.zh-CN.md) | 应用于残差/新息序列、面向模型变点检测的 FOCuS。 |
| [`RollingMedian`](rolling-median.zh-CN.md) | 价格窗口的滚动中位数。 |
| [`RunBarGenerator`](run-bar-generator.zh-CN.md) | 连续同号逐笔达到阈值后结束的逐笔游程 K 线。 |
| [`SmoothedMovingAverage`](smoothed-moving-average.zh-CN.md) | 以初始 SMA 窗口为种子的 Wilder/SMMA/RMA 平滑移动平均。 |
| [`SqueezeMomentum`](squeeze-momentum.zh-CN.md) | TTM 风格布林带位于肯特纳通道内的挤压标志及线性回归动量。 |
| [`StochasticMomentumIndex`](stochastic-momentum-index.zh-CN.md) | 带信号线的双重平滑随机动量指数。 |
| [`SwingIndex`](swing-index.zh-CN.md) | Wilder 衡量相邻 K 线价格行为的摆动指数。 |
| [`TrendIntensityIndex`](trend-intensity-index.zh-CN.md) | SMA 正偏差占绝对偏差的百分比。 |
| [`TwiggsMoneyFlow`](twiggs-money-flow.zh-CN.md) | 使用真实最高价/最低价和 EMA 成交量归一化的 Twiggs 资金流量。 |
| [`VerticalHorizontalFilter`](vertical-horizontal-filter.zh-CN.md) | 净移动相对路径长度的趋势强度指标（VHF）。 |
| [`VolumeBarGenerator`](volume-bar-generator.zh-CN.md) | 成交量达到阈值时结束的信息驱动成交量 K 线。 |
| [`VolumeOscillator`](volume-oscillator.zh-CN.md) | 成交量短期与长期简单移动平均之间的百分比差。 |
| [`VolumeRunBarGenerator`](volume-run-bar-generator.zh-CN.md) | 同号累计成交量达到阈值时结束的成交量游程 K 线。 |
| [`WaveTrend`](wave-trend.zh-CN.md) | 基于 HLC3 的 LazyBear WaveTrend 振荡器（wt1/wt2）。 |
| [`WeightedMultiPeerOrderFlowImbalance`](weighted-multi-peer-order-flow-imbalance.zh-CN.md) | 显式加权同类资产篮子 OFI，以及其对本资产收益率的滚动 beta 冲击。 |
| [`WilliamsAD`](williams-ad.zh-CN.md) | Williams 累积/派发累计线。 |
| [`WilliamsFractals`](williams-fractals.zh-CN.md) | 确认滞后两根 K 线的五根 K 线上下分形枢轴。 |
| [`WoodiePivotPoints`](woodie-pivot-points.zh-CN.md) | 根据前一根 K 线 H + L + 2C 计算的 Woodie 场内枢轴点。 |
| [`ZeroLagEMA`](zero-lag-ema.zh-CN.md) | 将去滞后价格送入 EMA 的零滞后指数移动平均。 |
