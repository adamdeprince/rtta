# Ehlers Cyber Cycle（EhlersCyberCycle）

## 摘要

`EhlersCyberCycle` 是 RTTA 对 John Ehlers Cyber Cycle 振荡器的流式实现。它先对价格作轻度平滑，再用二极点高通滤波器提取主导周期分量。周期值滞后一根 K 线的结果作为触发线返回，可用于交叉择时。

## 更新 API

```python
result = rtta.EhlersCyberCycle(period=20, fillna=True).update(price)
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `period`  | `20`    | 高通滤波的临界周期（最小为 2） |
| `fillna`  | `True`  | 若为 `False`，取得 `period` 个样本前返回 NaN |

`update(...)` 返回：

- `cycle`——当前 Cyber Cycle 值
- `trigger`——前一根 K 线的周期值（滞后一根 K 线，用作触发线）

`advance(price)` 更新状态；`last()` 返回缓存的结果。

## 工作原理

Cyber Cycle 旨在消除趋势并分离周期成分。RTTA 采用 Ehlers 的实用流式形式：

1. 当历史数据足够时，对价格作 **4 根 K 线加权平滑**：\((x_t+2x_{t-1}+2x_{t-2}+x_{t-3})/6\)。
2. 使用二极点高通滤波器处理平滑结果；系数 \(\alpha\) 由临界周期确定（角度构造与 Roofing 高通相同，为 \(0.707\cdot2\pi/P\)）。
3. 在较短的启动阶段（前 7 个样本），使用简单的二阶差分代理量，使完整递归状态建立之前滤波器也有定义。
4. **触发线**是前一个周期值，因此周期线与触发线的交叉可标记短期周期转折。

周期值为正表示周期分量位于其局部零轴上方；为负则表示位于下方。

## 递推公式

### Alpha

令 \(P=\max(\texttt{period},2)\)：

\[
\theta = \frac{0.707 \cdot 2\pi}{P},\qquad
\alpha = \frac{\cos\theta + \sin\theta - 1}{\cos\theta}
\]

### 价格平滑

当样本索引 \(t\ge3\)（从零开始计数且不小于 3）时：

\[
s_t = \frac{x_t + 2 x_{t-1} + 2 x_{t-2} + x_{t-3}}{6}
\]

否则 \(s_t=x_t\)。

### 周期启动阶段（\(t<7\)）

\[
C_t = \frac{x_t - 2 x_{t-1} + x_{t-2}}{4}
\]

### 周期递推（\(t\ge7\)）

令 \(a=1-\alpha/2\)：

\[
\begin{aligned}
C_t &= a^{2}\,(s_t - 2 s_{t-1} + s_{t-2}) \\
&\quad + 2(1-\alpha)\, C_{t-1} \\
&\quad - (1-\alpha)^{2}\, C_{t-2}
\end{aligned}
\]

### 触发线

\[
T_t = C_{t-1}
\]

（在代码中，`trigger = c1_` 使用的是把新周期值写入状态**之前**的周期值。）

结果：`cycle` \(=C_t\)，`trigger` \(=T_t\)。

当 `fillna=False` 且已处理的样本少于 \(P\) 个时，两个字段均为 NaN。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class EhlersCyberCycle` 中实现。
- 结果类型：`EhlersCyberCycleResult`（`cycle`、`trigger`）。
- 价格滞后状态为 `p1_`、`p2_`、`p3_`；平滑值滞后状态为 `s1_`、`s2_`；周期滞后状态为 `c1_`、`c2_`。
- 批量辅助函数：`batch_ehlers_cyber_cycle`。

## 参考资料

- [MESA Software——Ehlers 论文](https://www.mesasoftware.com/)
- [Cyber Cycle 背景资料](https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf)
