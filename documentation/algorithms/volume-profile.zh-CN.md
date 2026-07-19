# VolumeProfile

## 摘要

`VolumeProfile` 维护滚动的按价格分组成交量直方图，并输出成交量最大价位和价值区域上下界。

## 更新 API

```python
result = rtta.VolumeProfile().update(close, volume)
```

`update(...)` 每次接收 `close` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现把每条观测的成交量归入相应价格桶，并在滚动窗口内维护分布，以识别成交量最大价位和主要价值区域。

## 递推公式

\[
H_t=\max_{i\in W_t}high_i,\qquad L_t=\min_{i\in W_t}low_i,\qquad y_t=G(H_t,L_t,close_t)
\]

`update(...)` 返回含 `point_of_control`、`value_area_high` 和 `value_area_low` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class VolumeProfile` 中实现。

## 参考资料

- [Schwab：Volume Profile](https://www.schwab.com/learn/story/using-volume-profile-indicator)
