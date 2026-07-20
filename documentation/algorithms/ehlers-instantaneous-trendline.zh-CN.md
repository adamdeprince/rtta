# Ehlers 瞬时趋势线（EhlersInstantaneousTrendline）

## 摘要

`EhlersInstantaneousTrendline` 是 RTTA 对 Ehlers 瞬时趋势线的流式实现：一种二极点递归平滑器，采用四样本输入平均，并提供向前外推两根 K 线的**触发线**用于择时。

## 更新 API

```python
result = rtta.EhlersInstantaneousTrendline(period=20, fillna=True).update(price)
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `period`  | `20`    | 临界周期 \(P\)（最小为 2） |
| `fillna`  | `True`  | 若为 `False`，取得 `period` 个样本前返回 NaN |

`update(...)` 返回：

- `trendline`——瞬时趋势线
- `trigger`——\(2\cdot\text{trendline}_t-\text{trendline}_{t-2}\)

`advance(price)` 更新状态；`last()` 返回缓存的结果。

## 工作原理

Ehlers 瞬时趋势线是一种低滞后价格平滑线，旨在跟踪周期成分的局部均值，而无需等待较长的移动平均窗口。它的系数与 Super Smoother 的极点配置（\(\sqrt2\,\pi/P\)）相同，但前馈增益以 \(c_1=(1-c_2-c_3)/4\) 的形式分配到三项价格平均 \((x_t+2x_{t-1}+x_{t-2})\) 上；这是 Ehlers 介绍 iTrend 时所采用的离散形式。

**触发线**是趋势线向前两根 K 线的线性外推（\(2y_t-y_{t-2}\)）。趋势线与触发线的交叉可标记平滑线的短期转折，其作用类似信号线，但无需再增加一层 EMA。

本指标不同于 TA-Lib / RTTA 的 [`HilbertTrendline`](hilbert-trendline.zh-CN.md)：后者依据 Hilbert 主导周期估计值自适应调整平均长度，而不是使用固定周期。

## 递推公式

令 \(P=\max(\texttt{period},2)\)。预先计算：

\[
\theta = \frac{\sqrt{2}\,\pi}{P},\qquad
a_1 = e^{-\theta},\qquad
b_1 = 2 a_1 \cos(\theta)
\]

\[
c_2 = b_1,\qquad
c_3 = -a_1^{2},\qquad
c_1 = \frac{1 - c_2 - c_3}{4}
\]

对于前两个样本：

\[
I_t = x_t
\]

此后：

\[
I_t = c_1\,(x_t + 2 x_{t-1} + x_{t-2}) + c_2\, I_{t-1} + c_3\, I_{t-2}
\]

触发线（使用状态移位**之前**的趋势状态；更新后，\(t2\) 是此前再前一根的趋势值）：

\[
Trig_t = 2 I_t - I_{t-2}
\]

代码为 `trigger = 2.0 * trend - t2_`；计算时 `t2_` 仍是旧的两根 K 线前趋势值，随后状态按 `t2_ ← t1_ ← trend` 移位。

结果：`trendline` \(=I_t\)，`trigger` \(=Trig_t\)。

当 `fillna=False` 且已处理的样本少于 \(P\) 个时，两个字段均为 NaN。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class EhlersInstantaneousTrendline` 中实现。
- 结果类型：`EhlersInstantaneousTrendlineResult`（`trendline`、`trigger`）。
- 价格滞后状态为 `p1_`、`p2_`；趋势滞后状态为 `t1_`、`t2_`。
- 批量辅助函数：`batch_ehlers_itrend`。

## 参考资料

- [MESA Software——Ehlers 论文](https://www.mesasoftware.com/)
- [瞬时趋势线 / Super Smoother 资料](https://www.mesasoftware.com/papers/ZeroLag.pdf)
