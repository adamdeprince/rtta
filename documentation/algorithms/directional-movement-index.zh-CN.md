# DirectionalMovementIndex

## 摘要

`DirectionalMovementIndex` 计算 DX 方向运动趋势强度指标。

## 更新 API

```python
result = rtta.DirectionalMovementIndex().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

该指标属于 Wilder 方向运动体系：比较当前与上一根 K 线的高低点延伸，平滑正负方向运动和真实波幅，再计算归一化方向差异。

## 递推公式

\[
DI^+_t=100\frac{\operatorname{WilderEMA}_n(DM^+_t)}{\operatorname{ATR}_n(TR_t)},\qquad DI^-_t=100\frac{\operatorname{WilderEMA}_n(DM^-_t)}{\operatorname{ATR}_n(TR_t)}
\]

\[
DX_t=100\frac{|DI^+_t-DI^-_t|}{DI^+_t+DI^-_t}
\]

\[
ADX_t=\operatorname{WilderEMA}_n(DX_t),\qquad ADXR_t=\frac{ADX_t+ADX_{t-n}}2
\]

各方向运动类依次返回 \(DI^+_t\)、\(DI^-_t\)、\(DX_t\)、\(ADX_t\) 和 \(ADXR_t\)；本类返回 \(DX_t\)。

## 组合基础组件

[`ATR`](atr.zh-CN.md)

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class DirectionalMovementIndex` 中实现。

## 参考资料

- [ChartSchool：平均趋向指数 ADX](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx)
