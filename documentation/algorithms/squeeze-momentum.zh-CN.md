# 挤压动量（SqueezeMomentum）

## 摘要

`SqueezeMomentum` 是 RTTA 对 TTM 风格挤压指标的流式实现：当布林带位于肯特纳通道之内（波动率压缩）时给出二元标志，并计算价格相对于混合中线的线性回归动量。

## 更新 API

```python
result = rtta.SqueezeMomentum(window=20, bb_mult=2.0, kc_mult=1.5, fillna=True).update(close, high, low)
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `window`  | `20`    | BB、ATR/KC、Donchian 和线性回归的回看期 |
| `bb_mult` | `2.0`   | 布林带标准差乘数 |
| `kc_mult` | `1.5`   | 肯特纳通道 ATR 乘数 |
| `fillna`  | `True`  | 若为 `False`，取得 `window` 个样本前返回 NaN |

`update(...)` 返回：

- `on`——处于挤压状态（BB 位于 KC 内）时为 `1.0`，否则为 `0.0`
- `momentum`——去均值序列的线性回归值

`advance(...)` 更新状态；`last()` 返回缓存的结果。

## 工作原理

**挤压开启**是指布林带（均值 ± \(k\cdot\sigma\)）完全位于肯特纳通道（均值 ± \(m\cdot ATR\)）之内。这是经典 TTM Squeeze 的压缩状态：已实现波动率（BB）相对于平均真实波幅宽度（KC）较低。当挤压“释放”（`on` 从 1 变为 0）时，积蓄的能量常以方向性价格移动释放。

**动量**衡量收盘价相对于一个基准的位置：该基准取 Donchian 中点与收盘价 SMA 的平均值，再进行滚动线性回归（与 LazyBear / TTM 风格柱状图相同）。正动量偏向向上释放，负动量偏向向下释放。

## 递推公式

令 \(n\) 为窗口长度，\(k=\texttt{bb\_mult}\)，\(m=\texttt{kc\_mult}\)。

收盘价的滚动均值与总体标准差：

\[
\mu_t = \operatorname{mean}_n(c),\qquad
\sigma_t = \operatorname{stddev}_n(c)
\]

\[
BB^{hi}_t = \mu_t + k\sigma_t,\qquad
BB^{lo}_t = \mu_t - k\sigma_t
\]

使用长度为 \(n\) 的 ATR：

\[
KC^{hi}_t = \mu_t + m\cdot ATR_t,\qquad
KC^{lo}_t = \mu_t - m\cdot ATR_t
\]

挤压标志：

\[
on_t =
\begin{cases}
1 & BB^{lo}_t > KC^{lo}_t \;\land\; BB^{hi}_t < KC^{hi}_t \\
0 & \text{其他情况}
\end{cases}
\]

Donchian 中点与混合基准：

\[
HH_t = \max_{0\le i<n} h_{t-i},\qquad
LL_t = \min_{0\le i<n} l_{t-i}
\]

\[
basis_t = \frac{1}{2}\left(\frac{HH_t+LL_t}{2} + \mu_t\right)
\]

\[
\delta_t = c_t - basis_t
\]

\[
momentum_t = \operatorname{LinReg}_n(\delta)_t
\]

（通过 `LinearRegressionCore` 取得当前 K 线处的回归拟合值。）

当 `fillna=False` 且取得的样本少于 \(n\) 个时，两个字段均为 NaN。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class SqueezeMomentum` 中实现。
- 使用滚动 sum/sum2 计算均值/方差，使用 `RollingExtreme` 计算 HH/LL，并使用 `ATR` 和 `LinearRegressionCore linreg_`。
- 结果类型：`SqueezeMomentumResult`（`on`、`momentum`）。
- 批量辅助函数：`batch_squeeze_momentum`。
- 注意：肯特纳通道中线使用收盘价 SMA（`mean`），而不是典型价格的另一条 EMA——这与本 C++ 实现一致。

## 参考资料

- [StockCharts——TTM Squeeze](https://school.stockcharts.com/doku.php?id=technical_indicators:ttm_squeeze)
- [LazyBear Squeeze Momentum（TradingView 社区）](https://www.tradingview.com/script/nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/)
