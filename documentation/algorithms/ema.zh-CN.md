# EMA

## 摘要

`EMA` 计算单一标量数据流的指数移动平均。对象只保存最新的 EMA 值、样本计数器，以及由 `window` 推导出的固定平滑系数。

## 更新 API

```python
value = rtta.EMA(window=30.0, fillna=False).update(value)
```

同一个标量数据流可以通过 `update(...)`、`advance(...)` 或 `batch(...)` 输入。

## 工作原理

指数移动平均对最新样本赋予固定权重，并以其互补权重保留此前的平均值。RTTA 用第一个输入样本初始化状态，因此内部的第一个 EMA 值就等于第一个输入值。

## 递推公式

令 \(x_t\) 为新样本，\(n\) 为 `window`。RTTA 会把 `window` 的下限限制为 1，并采用：

\[
\alpha = \frac{2}{1+n}
\]

初始化：

\[
E_0 = x_0
\]

后续每次更新：

\[
E_t = \alpha x_t + (1-\alpha)E_{t-1}
\]

当 `fillna=False` 时，内部状态从第一个样本起就会更新，但在内部计数器达到 `window` 之前，返回值均为 `NaN`。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class EMA` 中实现。

## 参考资料

- [ChartSchool：简单移动平均与指数移动平均](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential)
