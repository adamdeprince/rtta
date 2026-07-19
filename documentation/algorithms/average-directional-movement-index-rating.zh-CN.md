# AverageDirectionalMovementIndexRating

## 摘要

`AverageDirectionalMovementIndexRating` 计算 ADXR，即经过进一步平滑的 ADX 趋势强度评级。

## 更新 API

```python
result = rtta.AverageDirectionalMovementIndexRating().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

该指标属于 Wilder 方向运动体系。更新时比较当前 K 线相对上一根 K 线的高低点延伸，平滑方向运动和真实波幅，并进一步平滑趋势强度。

## 递推公式

令 \(z_t = (close_t, high_t, low_t)\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
DI^+_t=100\frac{\operatorname{WilderEMA}_n(DM^+_t)}{\operatorname{ATR}_n(TR_t)}, \qquad
DI^-_t=100\frac{\operatorname{WilderEMA}_n(DM^-_t)}{\operatorname{ATR}_n(TR_t)}
\]

\[
DX_t=100\frac{|DI^+_t-DI^-_t|}{DI^+_t+DI^-_t}
\]

\[
ADX_t=\operatorname{WilderEMA}_n(DX_t), \qquad
ADXR_t=\frac{ADX_t+ADX_{t-n}}{2}
\]

`PlusDirectionalIndicator` 返回 \(DI^+_t\)，`MinusDirectionalIndicator` 返回 \(DI^-_t\)，`DirectionalMovementIndex` 返回 \(DX_t\)，`AverageDirectionalMovementIndex` 返回 \(ADX_t\)，`AverageDirectionalMovementIndexRating` 返回 \(ADXR_t\)。

## 组合基础组件

[`AverageDirectionalMovementIndex`](average-directional-movement-index.zh-CN.md)、[`Delay`](delay.zh-CN.md)

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class AverageDirectionalMovementIndexRating` 中实现。

## 参考资料

- [ChartSchool：平均趋向指数 ADX](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx)
