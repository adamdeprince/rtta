# PlusDirectionalIndicator

## 摘要

`PlusDirectionalIndicator` 计算正方向指标 \(DI^+\)。

## 更新 API

```python
result = rtta.PlusDirectionalIndicator().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

该指标属于 Wilder 方向运动体系，比较当前 K 线相对上一根的高低点延伸，并平滑正方向运动和真实波幅。

## 递推公式

\[
DI^+_t=100\frac{\operatorname{WilderEMA}_n(DM^+_t)}{\operatorname{ATR}_n(TR_t)},\qquad DI^-_t=100\frac{\operatorname{WilderEMA}_n(DM^-_t)}{\operatorname{ATR}_n(TR_t)}
\]

\[
DX_t=100\frac{|DI^+_t-DI^-_t|}{DI^+_t+DI^-_t},\qquad ADX_t=\operatorname{WilderEMA}_n(DX_t),\qquad ADXR_t=\frac{ADX_t+ADX_{t-n}}2
\]

各方向运动类依次返回 \(DI^+_t\)、\(DI^-_t\)、\(DX_t\)、\(ADX_t\) 和 \(ADXR_t\)；本类返回 \(DI^+_t\)。

## 组合基础组件

[`ATR`](atr.zh-CN.md)

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class PlusDirectionalIndicator` 中实现。

## 参考资料

- [ChartSchool：平均趋向指数 ADX](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx)
