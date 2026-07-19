# DonchianChannel

## 摘要

`DonchianChannel` 以滚动最高价和最低价构造通道。

## 更新 API

```python
result = rtta.DonchianChannel().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护滚动高低极值和区间统计量，每个样本只更新一次状态。

## 递推公式

\[
H_t=\max_{i\in W_t}high_i,\qquad L_t=\min_{i\in W_t}low_i
\]

\[
y_t=G(H_t,L_t,close_t)
\]

`update(...)` 返回含 `upper`、`lower`、`middle`、`width` 和 `percent` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class DonchianChannel` 中实现。

## 参考资料

- [ChartSchool：Donchian 通道](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/donchian-channels)
