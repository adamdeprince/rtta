# 方向变化检测器（DirectionalChangeDetector）

## 摘要

`DirectionalChangeDetector` 通过方向变化（DC）事件按**内在时间**采样：价格相对最近极值移动 \(\theta\) 即定义一次 DC；此后沿该方向继续延伸的路径报告为超调。输出包括事件标志、超调、当前极值和趋势模式。

## 更新 API

```python
import rtta

ind = rtta.DirectionalChangeDetector(threshold=0.01)  # 1% 相对阈值
result = ind.update(price)
# result.event ∈ {-1, 0, +1},
# result.overshoot, result.extremum, result.direction
```

`threshold` 是相对比例（\(0.01=1\%\)）。在上升趋势（等待向下反转）中，`direction`/模式为 \(+1\)；在下降趋势（等待向上反转）中为 \(-1\)；首次 DC 之前为 \(0\)。

## 工作原理

方向变化方法（Glattfelder、Dupuis、Olsen 等人的内在时间研究）以事件采样取代日历时间采样：价格相对局部极值移动固定比例时，时间才向前推进。发生向上 DC 后，算法将新高作为极值持续跟踪，直到价格下跌 \(\theta\)；发生向下 DC 后，则持续跟踪低点，直到价格上涨 \(\theta\)。超调衡量当前模式下，价格相对最近一次 DC 价格继续前进了多远。

## 递推公式

令 \(\theta=\)`threshold` \(>0\)。对于首个价格 \(p_0\)：极值 \(E=p_0\)，最近 DC 价格 \(p^{\mathrm{dc}}=p_0\)，模式 \(m=0\)，事件为 \(0\)。

**启动阶段**（\(m=0\)）：

\[
\begin{aligned}
p_t \ge E(1+\theta) &\Rightarrow m\leftarrow +1,\ \mathrm{event}\leftarrow +1,\
p^{\mathrm{dc}},E\leftarrow p_t,\\
p_t \le E(1-\theta) &\Rightarrow m\leftarrow -1,\ \mathrm{event}\leftarrow -1,\
p^{\mathrm{dc}},E\leftarrow p_t,\\
\text{否则} &\Rightarrow E\leftarrow \mathrm{clip\ extend}(E,p_t).
\end{aligned}
\]

**上升趋势**（\(m=+1\)）：\(E\leftarrow\max(E,p_t)\)。若 \(p_t\le E(1-\theta)\)，则令 \(m\leftarrow-1\)、\(\mathrm{event}\leftarrow-1\)、\(p^{\mathrm{dc}},E\leftarrow p_t\)。

**下降趋势**（\(m=-1\)）：\(E\leftarrow\min(E,p_t)\)。若 \(p_t\ge E(1+\theta)\)，则令 \(m\leftarrow+1\)、\(\mathrm{event}\leftarrow+1\)、\(p^{\mathrm{dc}},E\leftarrow p_t\)。

超调（当 \(p^{\mathrm{dc}}>0\) 时）：

\[
\mathrm{overshoot}_t =
\begin{cases}
(p_t - p^{\mathrm{dc}})/p^{\mathrm{dc}}, & m=+1,\\
(p^{\mathrm{dc}} - p_t)/p^{\mathrm{dc}}, & m=-1,\\
0, & m=0.
\end{cases}
\]

输出：\(\mathrm{extremum}_t=E\)，\(\mathrm{direction}_t=m\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class DirectionalChangeDetector` 中实现。结果类型为 `DirectionalChangeResult`。

## 参考资料

- [Glattfelder、Dupuis 与 Olsen，《Patterns in high-frequency FX data: discovery of 12 empirical scaling laws》（arXiv:0809.1040）](https://arxiv.org/abs/0809.1040)
