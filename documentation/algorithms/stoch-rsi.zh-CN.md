# StochRSI

## 摘要

`StochRSI` 对 RSI 数值应用随机振荡器。

## 更新 API

```python
result = rtta.StochRSI().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标根据 RSI 在近期高低区间中的位置进行归一化，全部滚动和平滑状态均严格因果。

## 递推公式

\[
U_t,D_t=\operatorname{directional\_components}(z_t,z_{t-1}),\qquad y_t=100\frac{\operatorname{smooth}(U_t)}{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class StochRSI` 中实现。

## 参考资料

- [ChartSchool：StochRSI](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochrsi)
