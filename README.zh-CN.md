# RTTA

`pyrtta` 是一个基于 C++23 和 nanobind 的低延迟库，用于逐笔技术分析、在线变化检测、市场状态监控和研究信号。Python 导入包名为 `rtta`。

RTTA 的设计目标，是尽量避免在无意中使用未来数据。每种算法都是有状态的因果对象：调用者每次通过 `update(...)` 或 `advance(...)` 输入一个 tick，对象只能根据已经见过的数据作出反应。这套接口适用于实时系统、交互式研究，以及必须在下一条观测到达前作出下单决策的市场模拟器。

## 范围

当前的基准测试注册表包含 188 种算法：

- 经典技术指标：移动平均、振荡器、趋势、波动率、价格变换、带状指标、通道、成交量指标和收益率指标。
- 状态空间模型和自适应滤波器：多种卡尔曼滤波器、粒子滤波器、交互式多模型、基于高斯过程与核方法的包络，以及跟踪滤波器。
- 市场微观结构与流动性组件：订单流失衡、买卖价反弹、价差特征、报价/成交强度、VPIN、Amihud 非流动性、Kyle lambda、流动性枯竭、价差爆发，以及执行成本/滑点状态。
- 在线变化与漂移检测：CUSUM、Page-Hinkley、ADWIN、DDM、EDDM、HDDM、KSWIN、EWMA z 分数位移，残差/误差/命中率/校准/特征漂移，以及面向均值、方差、相关性、beta 和价差/流动性的滚动双窗口位移检测器。
- 在线状态滤波器：阈值与回滞状态，波动率、ATR、已实现方差、趋势/震荡、流动性、价差、成交量、订单流、相关性、beta 和配对价差状态，以及有界 BOCPD、在线 HMM、粘性 HMM 类模型、马尔可夫切换波动率模型、高斯混合模型和半马尔可夫类滤波器。
- 金融市场专用实时组件：波动率突破、压缩/扩张、微观结构噪声、报价填塞、领先/滞后、开盘/收盘与集合竞价/连续交易转换、跨资产相关性失效，以及基于流式残差的协整失效监控。

## 安装

```bash
pip install pyrtta
```

## 用法

```python
import rtta

rsi = rtta.RSI()

for close in close_stream:
    value = rsi.update(close)
    if value > 70.0:
        reduce_position()
```

如果只需要推进状态，而不需要为当前 tick 创建 Python 结果对象，请使用 `advance(...)`：

```python
ema = rtta.EMA(window=30.0)

for close in warmup_ticks:
    ema.advance(close)

current = ema.update(next_close)
```

大多数指标提供以下方法：

- `update(...)`：接收一个样本，返回当前值或结果。
- `advance(...)`：接收一个样本，返回 `None`。
- `last()` 或 `last_<field>()`：读取最近一次状态，但不推进状态。
- `batch(...)`：为重启或研究工作流提供因果批量追赶。输入按时间顺序处理；调用完成后，对象可继续接收下一条实时 tick。
- `replay_update(...)`、`replay_advance(...)` 和 `replay_update_outputs(...)`：用于因果追赶和延迟基准测试的 C++ 回放路径。

多输出指标返回不可变的 C++ 结果结构体，其字段只读。若指标包含命名字段，还会提供 `update_upper(...)`、`last_upper()` 等标量便捷方法。

## 基准测试

延迟结果见独立的[基准测试页面](BENCHMARK.zh-CN.md)。该页面记录了当前 Intel、Apple Silicon 和龙芯测试所使用的 CPU 与运行时环境。

## 开发

本项目使用 `scikit-build-core` 和 CMake 构建为基于 C++23 的 nanobind 扩展。

```bash
poetry install --with build,dev,docs --no-root
poetry run python -m pip install --no-build-isolation -e .
poetry run pytest
```

构建静态 HTML 文档：

```bash
poetry run python tools/build_html_docs.py
```

构建 wheel：

```bash
poetry run python -m build --wheel
```

## 引用

如果在研究、基准测试或公开发表的工作中使用 RTTA，请按以下格式引用：

```bibtex
@misc{deprince2026pyrtta,
  author       = {DePrince, Adam},
  title        = {{pyrtta}: Low Latency Incremental Technical Analysis},
  year         = {2026},
  version      = {0.2.2},
  howpublished = {\url{https://github.com/adamdeprince/rtta}},
  note         = {Python package name: pyrtta; import package name: rtta}
}
```

同一条目也收录在 `CITATION.bib` 中。
