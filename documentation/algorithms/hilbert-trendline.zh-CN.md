# Hilbert 趋势线（HilbertTrendline）

## 摘要

`HilbertTrendline` 是 RTTA 对周期自适应瞬时趋势线的流式实现（TA-Lib `HT_TRENDLINE`）。它先按当前主导周期长度对原始价格取平均，再对该平均值应用 4 根 K 线加权移动平均。

## 更新 API

```python
result = rtta.HilbertTrendline(fillna=True).update(value)
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `fillna`  | `True`  | 若为 `False`，超过 63 个样本的回看期之前返回 NaN |

`update(value)` 以标量形式返回当前自适应趋势线。

## 工作原理

固定周期 SMA 的滞后量始终相同，而 Hilbert 趋势线会在主导周期较短时缩短平均窗口，在主导周期较长时延长窗口。TA-Lib 对最近 \(N=\operatorname{clip}(\lfloor\overline P+0.5\rfloor,1,50)\) 根 K 线的**原始**价格取平均（不是 WMA 平滑值），再以处理输入价格时相同的四抽头权重平滑该平均值：

\[
(4 A_t + 3 A_{t-1} + 2 A_{t-2} + A_{t-3})/10.
\]

它是 [`EhlersInstantaneousTrendline`](ehlers-instantaneous-trendline.zh-CN.md) 的自适应对应形式；后者不采用 Hilbert 周期，而是使用固定临界周期和二极点递推。

## 递推公式

令 \(\overline P_t\) 为平滑后的主导周期，并令：

\[
N = \operatorname{clip}\!\big(\lfloor \overline{P}_t + 0.5 \rfloor,\; 1,\; 50\big).
\]

对最近 \(N\) 个原始价格求平均（最多保留 64 个样本的价格历史）：

\[
A_t = \frac{1}{N}\sum_{j=0}^{N-1} x_{t-j}
\]

（早期 K 线中，若现有价格少于 \(N\) 个，缺失历史按零贡献处理。）

对平均值作四根 K 线 WMA：

\[
TL_t = \frac{4 A_t + 3 A_{t-1} + 2 A_{t-2} + A_{t-3}}{10}
\]

返回值：\(TL_t\)。

当 `fillna=False` 时，在完成超过 63 次更新之前，输出为 NaN。

## 实现说明

- 对 `HilbertCycleEngine::trendline()` 的轻量封装（`class HilbertTrendline`）。
- 引擎保留 `prices_`（最多 64 个值），并以 `i_trend1_`、`i_trend2_`、`i_trend3_` 保存 \(A_t\) 的滞后值。
- 回看期：`lookback_phase_ = 63`。

## 参考资料

- [TA-Lib HT_TRENDLINE](https://ta-lib.org/functions/ht_trendline)
- [MESA Software——Ehlers 论文](https://www.mesasoftware.com/)
