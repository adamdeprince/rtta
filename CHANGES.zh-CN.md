# 变更记录

## 0.2.3

- 竞赛研究信号：`FlowPressureCapacitySignal`——按事件时间计算的 L1 主动订单流除以对手方显示容量，并加入因果队列补单/撤单推断、持续性失衡滤波、有界公允价值和离散迟滞。既提供精简的有符号订单流 API，也提供详细的买入/卖出订单流 API；另附 Massive 成交/报价合并、按延迟报价支付价差的模拟账簿，以及未来中间价 alpha 诊断。

- 竞赛研究信号：`FourierResidueIdentity`——傅里叶—剩余类恒等式的流式实现（Portnaya，arXiv:2606.29591，2026 年 6 月）。它把收益率自相关拆成可单独检验的方向（符号，k=2）通道和幅度（k=4）通道，并提供逐通道 Fejér 方差比、Lo–MacKinlay 异方差稳健 z* 统计量和半样本持久性诊断。只有符号通道本身达到显著性时，`signal` 才会触发，因此纯粹由幅度造成的反转不会伪装成方向 alpha；当方向没有依据时，`magnitude_forecast` 保留仍有统计依据的波动率头寸规模信息。（这是通道分离工具，不是买卖价反跳过滤器——详见算法页面。）另新增论文之外的 `elliptical_ratio` 输出，把符号通道与 Grothendieck 基准 `(2/π)·arcsin(ρ)` 比较；这正是论文诊断的不受尺度影响形式。

- 竞赛研究信号：`SqrtImpactFlowSignal`——平方根市场冲击残差订单流（未使用冲击的延续 + 超调的回归），可选由逐笔确定符号的成交额和 K 线 VWAP（适用于 Polygon/Massive 聚合数据或逐笔数据）。

- K 线 / CDL 形态包（TA-Lib 风格的 `+100` / `0` / `-100` 输出）：32 个独立 `CDL*` 检测器，以及多输出 `CDLPatternPack`。

- 零散零售指标：`RainbowMovingAverage`、`RainbowOscillator`、`ChandeForecastOscillator`、`RangeActionVerificationIndex`（RAVI）、`BullsPower`、`BearsPower`、`ProjectionOscillator` 和 Dorsey `Inertia`。
- 研究深度扩展：`MessageEventOrderFlowImbalance`（消息流 OFI）、`HawkesIntensity`、`WeightedMultiPeerOrderFlowImbalance`，以及流式 `ConformalBands`。

- 经典指标补遗：`FibonacciPivotPoints`、`GuppyMMARibbon`（完整 12 条 EMA 带）、`AndrewsPitchfork` 和 `ElderThermometer`。

- 残差变点便利组件：`KalmanInnovationResidualFOCuS` 和 `KalmanInnovationResidualBOCPD`（一次更新即可完成新息 z 分数 → FOCuS/BOCPD）。
- 新增面向同类资产篮子 OFI 压力的 `MultiPeerOrderFlowImbalance`。
- 新增同号游程 K 线 `VolumeRunBarGenerator` 与 `DollarRunBarGenerator`。

- 后续研究组件：`CrossAssetOrderFlowImbalance`、`ResidualBOCPD`、`RunBarGenerator`、`WoodiePivotPoints`、`CamarillaPivotPoints` 和 `MovingAverageVariablePeriod`（MAVP）。
- `Ichimoku` 现在还会输出位移云带（`span_a_displaced`、`span_b_displaced`）。

- 多档 OFI 完善：`MultiLevelOrderFlowImbalance` / `IntegratedOrderFlowImbalance` 支持 float32 更新、形状为 `(n_samples, levels)` 的批处理/replay，以及注册表深度订单簿基准挂钩（`bid_prices`/`bid_sizes`/`ask_prices`/`ask_sizes`）。

- 新增第 4 波研究/微观结构组件：`MultiLevelOrderFlowImbalance`、`IntegratedOrderFlowImbalance`、`DecomposedOrderFlowImbalance`、`VolumeBarGenerator`、`DollarBarGenerator`、`ImbalanceBarGenerator`、`FOCuS`、`ResidualFOCuS` 和 `DirectionalChangeDetector`。

- 修正 `VolumeOscillator` 注册表中的 `batch_inputs`（此前错误地设为 `"input"`，导致数组批处理与 pandas 表批处理不一致）。
- 放宽 Schaff Trend Cycle 与 `ta` 的比较容差，以容纳累积浮点漂移（约 1e-10）。
- 在 Apple M4 Max、Intel Xeon 6975P-C 和 Loongson-3A6000 上重新测量 247 种算法的全注册表逐笔延迟（RTTA `0.2.3`）；注册表 `advance(...)` 中位数分别约为 29.9 / 39.2 / 104 ns/update（见 `BENCHMARK.md`）。

- 新增经典 Tier C/D：`EhlersSuperSmoother`、`EhlersRoofingFilter`、`EhlersCyberCycle`、`EhlersCenterOfGravity`、`EhlersInstantaneousTrendline`、`EhlersDecycler`、`ParabolicSARExtended`、`KagiChart`、`PointAndFigure`、`GuppyMultipleMovingAverage`、`RollingMedian` 和 `GeometricMovingAverage`。
- `Ichimoku` 现在接收 `close` 并返回 `lagging_span`（输入/输出扩展，属于破坏性变更）。

- 完成经典 Tier A/B：多输出 `MACD`/`MACDFix`、`MACDExt`、`RelativeVolatilityIndex`、`PivotPoints`、`QStick`、`PsychologicalLine`、`Bias`、`WilliamsFractals`、`MarketFacilitationIndex`、`SwingIndex`、`AccumulativeSwingIndex`、`VerticalHorizontalFilter`、`RandomWalkIndex`、`PrettyGoodOscillator`、`TrendIntensityIndex`、`WilliamsAD`、`IntradayIntensity`、`TwiggsMoneyFlow`、`ComparativeRelativeStrength` 和 `InverseFisherRSI`。
- `MACD`/`MACDFix` 现在返回 `macd`、`signal` 和 `histogram`（相对于此前只返回标量信号线的接口，属于破坏性变更）。

- 新增第 2 波经典指标：`StochasticMomentumIndex`、`DeMarker`、`IntradayMomentumIndex`、`AccelerationBands`、`ChandelierExit`、`Alligator`、`GatorOscillator`、`AcceleratorOscillator`、`SqueezeMomentum`、`WaveTrend`，以及 Hilbert 套件（`HilbertDominantCyclePeriod`、`HilbertDominantCyclePhase`、`HilbertPhasor`、`HilbertSineWave`、`HilbertTrendMode`、`HilbertTrendline`）；每种指标都附带英文算法文档、注册表条目和测试。
- 用与 TA-Lib 兼容的流式实现替换 Hilbert 套件引擎，覆盖 `HT_DCPERIOD`、`HT_DCPHASE`、`HT_PHASOR`、`HT_SINE`、`HT_TRENDMODE` 和 `HT_TRENDLINE`（奇偶去趋势器、WMA 平滑、周期截断、相位 DFT、趋势线和趋势模式规则）。

- 新增经典完整性指标：`SmoothedMovingAverage`（SMMA/RMA/Wilder）、`ZeroLagEMA`、`ArnaudLegouxMovingAverage`、`McGinleyDynamic`、`BollingerPercentB`、`BollingerBandwidth`、`MovingAverageEnvelope`、`PositiveVolumeIndex`、`VolumeOscillator`、`EfficiencyRatio`、`HistoricalVolatility` 和 `ChaikinVolatility`；每种指标都附带英文算法文档、注册表条目和正确性测试。

## 0.2.2

- 对 `indicator.cpp` 中的 C++ 热路径进行微优化：仅求和滚动窗口、分支/2 次幂环形索引、ConnorsRSI 排名扫描、布林带单次均值/方差、价格区间稳定时的 VolumeProfile 增量直方图、ADWIN/KSWIN 环形缓冲与预分配暂存、粒子滤波与高斯过程暂存复用及平稳 GP Cholesky 缓存，以及原始指针批处理循环。
- 启用面向 Release 的构建选项（`-O3`、`-DNDEBUG`、`-ffp-contract=off`，在支持时启用 LTO）。
- 在 Apple M4 Max、Intel Xeon 6975P-C 与 Loongson-3A6000 上重新测量全注册表逐笔延迟；中位 `advance(...)` 分别约为 28.5 / 35.9 / 101 ns/update（见 `BENCHMARK.md`）。
- 新增 `tools/run_latency_benchmarks.py`，用于发现新指标、运行多主机延迟基准并重新生成基准文档。
- 补全基准 harness 所需的 `IntradayClockEchoSignal` batch/replay 绑定。
- 软件包版本升级至 `0.2.2`。

## 0.2.1

- 新增生成文档站点所需的源文件，包括各算法的 Markdown 页面、按 CPU 类型拆分的基准测试页面，以及 HTML 构建工具。
- 新增 RTTA 静态站点的 favicon 和彩蛋图片资源，并通过 Git LFS 管理。
- 软件包版本升级至 `0.2.1`。

## 0.2.0

- Python 发行包更名为 `pyrtta`，导入包名仍保留为 `rtta`。
- 为卡尔曼滤波指标新增固定版本的 Python 与构建依赖 `fast-kalman==0.2.2`。
- 移除随仓库提供的 `third_party/fast-kalman` 子模块路径；CMake 现在直接使用固定版本 `fast-kalman` 构建依赖中的 C++ 头文件，因此源码发行包无需针对子模块进行额外配置即可正常构建。
- 新增 `KalmanMovingAverage`：采用恒定速度模型的卡尔曼价格滤波器，支持 update、advance、replay、batch、pandas 表批处理、记录列表批处理，以及该指标专用的不可变调优结果。
- 新增 `KalmanLocalLinearTrend`：采用局部水平/趋势状态空间模型的卡尔曼指标，其调优参数既可配置，也可训练。
- 新增 `KalmanVelocityOscillator`：公开同一恒定速度卡尔曼价格模型中的速度状态，并提供可配置、可训练的调优参数。
- 新增 `KalmanInnovationZScore`：输出带符号的卡尔曼测量新息，并以其预测标准差进行标准化。
- 新增 `KalmanPredictionBands`：根据卡尔曼滤波预测的测量不确定性，输出一步价格预测区间。
- 新增 `KalmanTrendSignal`：输出经卡尔曼滤波的趋势线，以及根据价格穿越该趋势线产生的买卖信号。
- 新增 `ConnorsRSI`、`RelativeVigorIndex`、`KlingerVolumeOscillator`、`ElderRayIndex`、`CoppockCurve`、`FisherTransform`、`FractalAdaptiveMovingAverage`、`MesaAdaptiveMovingAverage` 和 `EhlersOptimalTrackingFilter`。
- 新增 `OrderFlowImbalance`：在报价层面衡量最优买卖价及其挂单量所形成的压力，支持增量 update/advance、replay、数组批处理、记录列表批处理、pandas 表批处理，并纳入基准测试与正确性测试。
- 新增 `CUSUM`：因果累积和事件滤波器，支持 update/advance、replay、数组批处理、记录列表批处理、pandas 表批处理，并纳入基准测试与正确性测试。
- 新增 `PageHinkley`：具有方向输出的因果 Page-Hinkley 均值位移事件检测器，支持 update/advance、replay、数组批处理、记录列表批处理、pandas 表批处理，并纳入基准测试与正确性测试。
- 新增 `EWMAZScoreShiftDetector`：因果 EWMA 均值/方差 z 分数位移事件检测器，支持 update/advance、replay、数组批处理、记录列表批处理、pandas 表批处理，并纳入基准测试与正确性测试。
- 新增一组相邻滚动窗口位移检测器，覆盖均值、方差、均值与方差联合变化、相关性、beta，以及报价价差/深度所反映的流动性压力。每种检测器都支持 update/advance、replay、数组批处理、记录列表批处理、pandas 表批处理，并纳入基准测试与正确性测试。
- 新增 `ThresholdRegimeDetector`：带上下回滞阈值的有状态状态检测器，并提供标准的 update/advance/replay/batch API。
- 新增流式状态检测器，覆盖波动率、ATR、已实现方差、趋势/震荡、流动性、价差、成交量、成交强度、订单流失衡、相关性、beta、配对价差残差 z 分数、协整失效和执行成本/滑点状态。这些检测器都采用因果 update/advance/replay/batch API，支持数组、记录列表和 pandas 表批处理，并具备基准测试与正确性测试。
- 新增漂移与模型健康组件（`ADWIN`、`DDM`、`EDDM`、`HDDM`、`KSWIN`，以及残差、预测误差、命中率、校准和特征分布漂移检测器）；新增概率型在线状态组件（在线 HMM、粘性 HMM 类模型、马尔可夫切换波动率、有界 BOCPD、在线高斯混合和隐半马尔可夫类滤波器）；另新增针对金融实时数据的组件，用于检测波动率突破、压缩与扩张、微观结构噪声、买卖价反弹、报价消息速率、报价填塞、领先/滞后、流动性枯竭、价差爆发、开盘/收盘转换、集合竞价/连续交易转换和跨资产相关性失效。所有组件均提供因果 update/advance/replay/batch API，支持数组、记录列表和 pandas 表批处理，并包含基准测试与针对性一致性测试。
- 新增 `AlphaBetaGammaTrackingFilter`：类似稳态卡尔曼滤波的价格/速度/加速度跟踪器，提供不可变多输出结果、标量字段访问器、replay 输出批处理，以及数组、表格和记录批处理。
- 新增 `InteractingMultipleModelFilter`：四状态 IMM 卡尔曼跟踪器，以在线模型概率混合低波动、高波动、趋势和震荡模型。
- 新增 `ParticleFilterTrend`：使用确定性随机种子的非高斯粒子趋势滤波器，采用拉普拉斯测量似然和系统重采样，并输出信号及有效样本量诊断。
- 新增 `SavitzkyGolayFilter`：因果滚动多项式平滑器，预先计算端点卷积系数，可输出平滑价格、一阶导数和二阶导数。
- 新增 `NadarayaWatsonEnvelope`：采用高斯核的非参数平滑器，并根据加权残差生成上下边界。
- 新增 `GaussianProcessRegressionBands`：采用 RBF 核的滚动高斯过程平滑器，输出后验均值与不确定性区间。
- 新增 `ZigZagSwingDetector`：基于收盘价百分比变动的波段检测器，用于滤除噪声并标记已确认的枢轴点。
- 新增 `RenkoBrickGenerator`：事件驱动的收盘价变换，输出带符号的砖块数量、当前砖块状态、方向与反转。
- 新增 `HeikinAshiTransform`：增量式 OHLC K 线变换，输出 Heikin-Ashi 的开、高、低、收数值。
- 新增 `AnchoredVWAP`：一种可由任意锚定事件重置累计价量状态的 VWAP 变体。
- 新增 `VolumeProfile`：滚动价位成交量直方图，输出控制点以及价值区域的上沿和下沿。
- 新增 `VPIN`：基于成交量时钟的订单流毒性指标，采用批量成交量分类和滚动成交量桶失衡。
- 新增 `KyleLambda`：根据收益率与带符号的美元成交量平方根滚动估计市场冲击。
- 新增 `AmihudIlliquidity`：以单位美元成交量对应的绝对收益率滚动估计流动性。
- 新增 `SpreadFeatures`：衡量报价与成交执行质量的指标，输出报价价差、有效价差和延迟已实现价差。
- 新增 `MatchedFlowConformalSignal`：日内 OHLCV 匹配流信号，带有类共形滚动预测误差区间、标量访问器，以及 batch/replay 路径。
- 新增 `ClosePressureReversalSignal`：日终反转信号，提供压力诊断、标量访问器和 batch/replay 路径；同时加入横截面大幅加速示例，交易排名最高的 10%。
- 新增 `KalmanRegressionChannel`、`KalmanHedgeRatio`、`TwoFactorKalmanTrendFilter` 和 `KalmanExtremumTrend`，用于在线卡尔曼配对回归和混合趋势滤波。
- 新增 `ALGOS.md`，以简短说明和文档链接列出公共指标；若 ChartSchool 存在直接对应页面，则优先链接至该页面。
- 新增 `CITATION.bib`，并在 README 中加入引用章节，为引用本软件包的用户提供 BibTeX 条目。
- 新增 `VariableIndexDynamicAverage`，以 CMO 调节 EMA 平滑系数实现 VIDYA，支持 update、advance、replay、数组批处理、记录列表批处理、pandas 表批处理、基准测试和正确性测试。
- 新增独立基准测试工具 `benchmarks/benchmark_indicators.py`。它报告每样本纳秒数，并将批处理之间的比较与 RTTA `update()` 延迟分开呈现。
- 新增 `benchmarks/benchmark_update_latency.py`，用于优先跟踪 RTTA 增量延迟。它分别报告 Python 循环开销、`update()` 总延迟、扣除循环后的 update 延迟，以及未来的 `advance()` 延迟列。
- 将基准测试报告拆分为 RTTA ndarray 批处理、RTTA 记录列表批处理、RTTA pandas 表批处理、第三方批处理和 RTTA `update()` 计时列。
- 为 EMA、SMA、Summation、ROC、Kama 和 PercentagePrice 新增记录列表批处理重载。
- 为过去只公开增量 `update()` 调用的指标新增 C++ 批处理循环，使 NumPy/表格批处理在 C++ 一侧迭代，而不再为每个样本跨越一次 Python 边界。
- 新增零拷贝 pandas 表批处理。表格批处理会提取预期的列名，并且只接受内存连续的 float32/float64 列。
- 在现有 float64 重载之外，新增显式的 float32 NumPy 批处理重载，并将 ndarray 参数设为禁止转换，避免 nanobind 在幕后静默转换或复制输入而进入更慢的路径。
- pandas 表的 dtype 分派从 dtype 名称字符串比较改为整数形式的 NumPy `dtype.num` 接口和 `switch` 分派。
- 针对加权移动平均、beta/相关性、方差、线性回归输出及若干派生自极值的指标，专门优化热点滚动批处理核心，使批处理免于不必要的窗口反复扫描以及多余的最小值、最大值和求和维护。
- 为 UltimateOscillator、TrueRange、Momentum、ChandeMomentumOscillator、MoneyFlowIndex、HighLow 和 HighLowIndex 新增针对公式设计的批处理循环，使其 NumPy/表格批处理不再经过通用增量 update 包装器。
- 在 `indicator.cpp` 中新增明确的 `batch_kernels` 区域，并将与 TA-Lib 差距最大的剩余指标迁移到原始循环批处理内核，同时保留原有面向对象的 `update()` 和 `batch()` API。
- 将以 deque 实现的滚动最小/最大值队列替换为固定容量的 vector 环形缓冲区，并显式内联热点滚动辅助访问器。
- 为 `MidPrice` 新增专门的小窗口原始扫描批处理内核，并提供状态重建逻辑，使 `batch()` 之后仍可继续调用增量 `update()`。
- 为小窗口、由极值派生的数值指标（`High`、`Low`、`HighLow`、`MidPoint`、`WilliamsR` 和随机指标 fast-k）新增针对全新批次的快速路径；索引/Aroon 输出继续采用更快的单调队列路径。同时还扁平化了 `T3MovingAverage` 的 EMA 链，加入直接动量循环、前缀式统计内核，以及用于 ATR 和方向运动指标的扁平化 Wilder 平滑循环。
- 专门优化 EMA、SMA、Summation、ROC 和 Kama 的批处理循环，避免批处理中逐样本执行 nanobind 索引，以及不必要的 `update()` 分派。
- 新增 `PercentagePrice.batch_ppo()`，使仅 PPO 的基准测试与 TA-Lib 的 PPO 批处理函数比较相同形状的输出。
- 比较库仍不写入 `pyproject.toml`；基准测试文档改为说明可选的 `pip install ta==0.11.0 TA-Lib` 安装方式。
- 更新 README 的性能章节，使其指向基准测试工具及其 Markdown/CSV 输出模式。
- 修改 `make_array`，将已移动的 `std::vector<double>` 所有权直接转交 NumPy，而不再分配第二个缓冲区并复制。
- 重写 `RollingWindow`，使其维护增量和以及单调最小/最大值队列，避免在滚动 `sum`、`min`、`max` 和最小/最大值偏移查询时重新扫描。
- 将 Python `dict` 结果对象替换为类型明确的 C++ 结果结构体，并通过 nanobind 的只读字段公开。
- 为多输出指标新增标量 `update_<field>()` 和 `last_<field>()` 访问器，使只需要一个输出的调用者无需分配不可变结果对象。
- 为多输出指标新增 `replay_update_outputs()`：在 C++ 中迭代增量 update 内核，同时返回与 `batch()` 相同形状的批处理结果。
- 在 `README.md` 中说明指标 API 约定，包括 `update()`、`advance()`、不可变多输出结果结构体、标量字段访问器、校验和 replay 方法及 replay 输出批处理。
- 新增 `SuperTrend`，支持增量 `update()`/`advance()`、标量字段访问器、C++ replay 输出批处理，以及数组、表格和记录批处理。
- 新增 `ChoppinessIndex`，根据滚动真实波幅之和及滚动最高/最低价区间计算 CHOP，支持增量 `update()`/`advance()` 以及数组、表格和记录批处理。
- 新增 `HullMovingAverage`，按 `WMA(2*WMA(n/2) - WMA(n), sqrt(n))` 实现 HMA，支持增量 `update()`/`advance()` 以及数组、表格和记录批处理。
- 新增 `VolumeWeightedMovingAverage`，按滚动 `sum(close * volume) / sum(volume)` 实现 VWMA，支持增量 `update()`/`advance()` 以及数组、表格和记录批处理。
- 新增 `FibonacciRetracementLevels`，针对上涨或下跌锚点输出滚动 0%、23.6%、38.2%、50%、61.8% 和 100% 回撤位，并提供不可变多输出结果、标量字段访问器、replay 输出批处理，以及数组、表格和记录批处理。
- 修正 ATR、ATRP 和 NormalizedATR，使其采用增量 Wilder 平滑，并在预热结束后与 TA-Lib 一致。
- 更新受类型化结果 API 影响的测试。
- 为每个参与基准测试的指标新增 512 样本真实序列测试，并针对有对应实现的技术分析指标，与 TA-Lib 和 `ta` 进行第三方正确性比较。
- 软件包版本升级至 `0.2.0`。
