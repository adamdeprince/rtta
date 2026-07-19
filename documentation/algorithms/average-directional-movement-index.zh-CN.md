# AverageDirectionalMovementIndex

## 摘要

`AverageDirectionalMovementIndex` 计算 ADX 趋势强度指标。

## 更新 API

```python
result = rtta.AverageDirectionalMovementIndex().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

ADX 属于 Wilder 方向运动体系。更新时比较当前 K 线相对上一根 K 线的高低点延伸，平滑正负方向运动和真实波幅，再计算归一化方向差异及其趋势强度。

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

各方向运动类分别返回 \(DI^+_t\)、\(DI^-_t\)、\(DX_t\)、\(ADX_t\) 和 \(ADXR_t\)；本类返回 \(ADX_t\)。

## 组合基础组件

[`ATR`](atr.zh-CN.md)

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class AverageDirectionalMovementIndex` 中实现。

## 参考资料

- [ChartSchool：平均趋向指数 ADX](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx)
