# KSTOscillator

## 摘要

`KSTOscillator` 是 Pring 的 Know Sure Thing 指标，对多个周期的变动率进行平滑和加权。

## 更新 API

```python
result = rtta.KSTOscillator().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标以严格因果方式维护多个 ROC 及其平滑状态，再组合为 KST、信号线和差值。

## 递推公式

\[
U_t,D_t=\operatorname{directional\_components}(z_t,z_{t-1})
\]

\[
y_t=100\frac{\operatorname{smooth}(U_t)}{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

`update(...)` 返回含 `kst`、`signal` 和 `difference` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class KSTOscillator` 中实现。

## 参考资料

- [ChartSchool：Pring's KST](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/pring-s-know-sure-thing-kst)
