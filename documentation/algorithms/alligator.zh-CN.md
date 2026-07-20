# Alligator

## 摘要

`Alligator` 是 RTTA 对 Bill Williams 鳄鱼线的流式实现：它由中间价的三条
平移平滑移动平均线——**颚线**、**齿线**和**唇线**——组成。随着趋势与盘整
交替出现，三条线会像鳄鱼的嘴一样张开和闭合。

## 更新 API

```python
result = rtta.Alligator(
    jaw_window=13, teeth_window=8, lips_window=5,
    jaw_shift=8, teeth_shift=5, lips_shift=3,
    fillna=True,
).update(high, low)
```

| 参数 | 默认值 | 含义 |
|------|-------:|------|
| `jaw_window` | `13` | 颚线（蓝色）的 SMMA 周期 |
| `teeth_window` | `8` | 齿线（红色）的 SMMA 周期 |
| `lips_window` | `5` | 唇线（绿色）的 SMMA 周期 |
| `jaw_shift` | `8` | 颚线的前移量（以延迟的 K 线数实现） |
| `teeth_shift` | `5` | 齿线的延迟量 |
| `lips_shift` | `3` | 唇线的延迟量 |
| `fillna` | `True` | 若为 `False`，只要任一线仍处在平移预热期并为 NaN，便返回 NaN |

`update(high, low)` 返回 `jaw`、`teeth` 和 `lips`。
`advance(...)` 更新状态；`last()` 返回缓存的结果。

## 工作原理

中间价 \(m_t=(h_t+l_t)/2\) 分别经过三条独立的 Wilder SMMA（RMA）平滑，
各自使用不同的周期。在传统图表中，这些线会被*绘制到未来*（向前平移）。在因果的
流式 API 中，相同的前移效果由等长的**延迟缓冲区**实现：今天报告的是
`shift` 根 K 线以前的 SMMA 值。这与完整图表把该 SMMA 前移后对齐到当前 K 线
的显示方式一致。

- **唇线**速度最快，在趋势开始时最先反应。
- **齿线**是居中的平衡线。
- **颚线**最慢，代表休眠状态或深层趋势。

当三条线相互缠绕（嘴巴闭合）时，“鳄鱼正在睡觉”，市场处于区间或震荡状态。
当三条线按顺序展开时，嘴巴张开，趋势处于活跃状态。
[`GatorOscillator`](gator-oscillator.zh-CN.md) 会量化这些线之间的距离。

## 递推公式

\[
m_t = \frac{h_t + l_t}{2}
\]

令 \(\operatorname{SMMA}_n\) 表示
[`SmoothedMovingAverage`](smoothed-moving-average.zh-CN.md)（先用 SMA 初始化，
再以 \(\alpha=1/n\) 进行 Wilder 递推）。令 \(\Delta_k\) 表示 \(k\) 根 K 线的
因果延迟（`ShiftBuffer` 在收到 \(k\) 个样本前返回 NaN，之后返回 \(k\) 根
K 线以前的值）。

\[
jaw_t = \Delta_{\texttt{jaw\_shift}}\big(\operatorname{SMMA}_{\texttt{jaw\_window}}(m)\big)_t
\]

\[
teeth_t = \Delta_{\texttt{teeth\_shift}}\big(\operatorname{SMMA}_{\texttt{teeth\_window}}(m)\big)_t
\]

\[
lips_t = \Delta_{\texttt{lips\_shift}}\big(\operatorname{SMMA}_{\texttt{lips\_window}}(m)\big)_t
\]

默认配置为：颚线 \(13/8\)、齿线 \(8/5\)、唇线 \(5/3\)。

当 `fillna=False` 且三条延迟线中的任一条仍为 NaN 时，返回结构体的所有字段
均为 NaN。

## 实现说明

- 实现在 `src/rtta/indicator.cpp` 的 `class Alligator` 中。
- SMMA 实例始终使用 `fillna=True`，以便形成不完整周期的 SMA；平移缓冲区在
  填满之前产生 NaN。
- 结果类型为 `AlligatorResult`（`jaw`、`teeth`、`lips`）。
- 批量辅助函数为 `batch_alligator`。

## 参考资料

- [Investopedia：Alligator Indicator](https://www.investopedia.com/articles/trading/06/alligator.asp)
- [ChartSchool：Bill Williams Alligator 概述](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/gator-oscillator)
