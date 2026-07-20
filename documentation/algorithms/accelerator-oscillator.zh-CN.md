# AcceleratorOscillator

## 摘要

`AcceleratorOscillator` 是 RTTA 对 Bill Williams 加速振荡器（AC）的流式
实现：用 Awesome Oscillator 减去其自身的短周期 SMA。它衡量的不是市场驱动力
本身，而是驱动力的*变化*（加速度）。

## 更新 API

```python
result = rtta.AcceleratorOscillator(ao_slow=34, ao_fast=5, smooth=5, fillna=True).update(high, low)
```

| 参数 | 默认值 | 含义 |
|------|-------:|------|
| `ao_slow` | `34` | Awesome Oscillator 内部慢速 SMA 的周期 |
| `ao_fast` | `5` | Awesome Oscillator 内部快速 SMA 的周期 |
| `smooth` | `5` | 对 AO 再次平滑的 SMA 周期 |
| `fillna` | `True` | 若为 `False`，在预热期结束前返回 NaN |

`update(high, low)` 返回一个标量 AC 值。

## 工作原理

Bill Williams 的 Awesome Oscillator（AO）是中间价 \((h+l)/2\) 的快速 SMA
与慢速 SMA 之差。加速振荡器则用 AO 减去 AO 的另一条 SMA：

\[
AC_t = AO_t - \operatorname{SMA}(AO)_t.
\]

AC 位于零线上方且继续上升时，上行动力正在加速；AC 位于零线下方且继续下降时，
下行动力正在加速。Williams 体系中的零线与颜色变化规则，会将 AC 用作 AO 和
Alligator 交易形态的确认过滤器。

RTTA 将现有的 [`AwesomeOscillator`](awesome-oscillator.zh-CN.md) 与周期为
`smooth` 的 SMA 组合起来实现该指标。

## 递推公式

中间价与 AO 的计算如下（与 `AwesomeOscillator` 一致；最高价和最低价会先排序，
使 \(h\ge l\)）：

\[
m_t = \frac{h_t + l_t}{2}
\]

\[
AO_t = \operatorname{SMA}_{n_f}(m_t) - \operatorname{SMA}_{n_s}(m_t)
\]

默认 \(n_f=5\)、\(n_s=34\)。

令 \(k=\texttt{smooth}\)（默认值为 5），再将 AO 输入一条 SMA。若 AO 为
NaN，则以 `0.0` 更新这条 SMA，以保证其内部状态仍会推进：

\[
\overline{AO}_t = \operatorname{SMA}_k(AO_t)
\]

\[
AC_t = AO_t - \overline{AO}_t
\]

当 `fillna=False` 时，必须取得 \(\max(n_s,n_f)+k\) 个样本后才返回非 NaN
结果。若 AO 本身仍为 NaN，则 `fillna=True` 时返回 `0.0`，否则返回 NaN。

## 实现说明

- 实现在 `src/rtta/indicator.cpp` 的 `class AcceleratorOscillator` 中。
- 内部由始终使用 `fillna=True` 的 `AwesomeOscillator ao_`，以及
  `SMA ao_sma_` 组成。
- 构造函数参数顺序为 `(ao_slow, ao_fast, smooth)`，与 AO 的
  `(window_1=slow, window_2=fast)` 顺序一致。

## 参考资料

- [Investopedia：Bill Williams Alligator 与 AC 的应用背景](https://www.investopedia.com/articles/trading/06/alligator.asp)
- [ChartSchool：Awesome Oscillator 背景](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/awesome-oscillator-ao)
