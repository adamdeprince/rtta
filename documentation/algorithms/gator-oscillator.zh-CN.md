# 鳄鱼振荡器（GatorOscillator）

## 摘要

`GatorOscillator` 是 RTTA 对 Bill Williams 鳄鱼振荡器的流式实现。它绘制鳄鱼线颚线—齿线之间的绝对距离（上方柱状图），以及齿线—唇线之间带负号的绝对距离（下方柱状图），直观显示鳄鱼的嘴正在张开还是闭合。

## 更新 API

```python
result = rtta.GatorOscillator(
    jaw_window=13, teeth_window=8, lips_window=5,
    jaw_shift=8, teeth_shift=5, lips_shift=3,
    fillna=True,
).update(high, low)
```

参数与 [`Alligator`](alligator.zh-CN.md) 一致（窗口和位移的默认值也相同）。

`update(high, low)` 返回：

- `upper`——\(|jaw-teeth|\)
- `lower`——\(-|teeth-lips|\)

`advance(...)` 更新状态；`last()` 返回缓存的结果。

## 工作原理

鳄鱼振荡器并不引入新的平滑器；它只是对鳄鱼线进行变换。绝对距离较大表示嘴部张开（趋势行情），距离较小表示嘴部闭合（休眠 / 盘整）。按照惯例，下方序列绘制在零轴之下，因此同一个柱状图窗格可以同时显示两段距离。

Williams 的颜色规则（每段距离相对前一根 K 线扩张还是收缩）可由调用方根据连续的 `upper` / `lower` 值应用；RTTA 只返回原始的有符号幅度。

## 递推公式

令 \((J_t,T_t,L_t)\) 为时刻 \(t\) 的鳄鱼线颚线、齿线和唇线：

\[
upper_t = |J_t - T_t|
\]

\[
lower_t = -|T_t - L_t|
\]

若任一鳄鱼线分量为 NaN，两个字段均为 NaN。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class GatorOscillator` 中实现。
- 内部持有一个采用相同构造参数的 `Alligator`。
- 结果类型：`GatorOscillatorResult`（`upper`、`lower`）。
- 批量辅助函数：`batch_gator`。

## 参考资料

- [StockCharts——Gator Oscillator](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/gator-oscillator)
- [Investopedia——Alligator / Gator 背景资料](https://www.investopedia.com/articles/trading/06/alligator.asp)
