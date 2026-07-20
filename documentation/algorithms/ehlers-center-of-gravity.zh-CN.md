# Ehlers 重心振荡器（EhlersCenterOfGravity）

## 摘要

`EhlersCenterOfGravity` 是 RTTA 对 John Ehlers 重心（CG）振荡器的流式实现。它在滚动价格窗口内测量价格路径的“平衡点”，再重新居中，使振荡器围绕零轴波动。前一个 CG 以 `lag` 返回，可用于触发式交叉信号。

## 更新 API

```python
result = rtta.EhlersCenterOfGravity(window=10, fillna=True).update(price)
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `window`  | `10`    | 滚动回看期（最小为 2） |
| `fillna`  | `True`  | 若为 `False`，窗口填满前返回 NaN |

`update(...)` 返回：

- `cg`——当前重心振荡器值
- `lag`——前一根 K 线的 `cg`（用于交叉信号）

`advance(price)` 更新状态；`last()` 返回缓存的结果。

## 工作原理

把最近 \(n\) 个价格视作沿 K 线时间轴排列、质量相等的质点。Ehlers 为**最新** K 线赋予权重 \(1\)，为**最旧** K 线赋予权重 \(n\)。因此，当市场下跌进入窗口时，一阶矩偏向较早价格；上涨进入窗口时，则偏向较新价格。用价格之和（而不是权重之和）作除数可得到原始 CG 坐标；再减去 \((n+1)/2\)，便将零位居中到窗口的几何中点，使振荡器近似具有零均值。

由于 CG 在转折点处领先于普通平滑振荡器，Ehlers 常将 CG 与其滞后一根 K 线的值绘制在一起；RTTA 通过 `lag` 字段提供这个滞后值。

## 递推公式

令 \(n\) 为缓冲区中当前价格数量（\(n\le W\)，其中 \(W\) 为构造函数窗口）。以 \(i=0\) 表示**最新**样本，以 \(i=n-1\) 表示最旧样本。权重 \(w_i=i+1\)（最新样本权重为 1，最旧样本权重为 \(n\)）：

\[
N_t = \sum_{i=0}^{n-1} w_i\, x_{t-i},\qquad
D_t = \sum_{i=0}^{n-1} x_{t-i}
\]

\[
CG_t =
\begin{cases}
0 & \text{若 } D_t = 0 \\
-\dfrac{N_t}{D_t} + \dfrac{n+1}{2} & \text{其他情况}
\end{cases}
\]

\[
lag_t = CG_{t-1}
\]

（首次更新之前，第一根 K 线的 `lag` 等于构造函数的初始值 `0.0`。）

当 `fillna=False` 且滚动缓冲区尚未填满时，两个字段均为 NaN。当 `fillna=True` 时，部分窗口按相同的权重约定使用现有的 \(n\) 个样本。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class EhlersCenterOfGravity` 中实现。
- 结果类型：`EhlersCenterOfGravityResult`（`cg`、`lag`）。
- 使用 `RollingBuffer prices_`；最新值为 `prices_.at(n-1)`，与按权重索引 \(i\) 访问的循环 `prices_.at(n - 1 - i)` 一致。
- 批量辅助函数：`batch_ehlers_cg`。

## 参考资料

- [MESA Software——Ehlers 论文](https://www.mesasoftware.com/)
- [Ehlers CG / 随机 CG 讨论](https://www.mesasoftware.com/papers/TheCGOscillator.pdf)
