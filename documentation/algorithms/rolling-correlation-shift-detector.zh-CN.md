# RollingCorrelationShiftDetector

## 摘要

`RollingCorrelationShiftDetector` 以因果方式比较相邻窗口的相关性变化。

## 更新 API

```python
result = rtta.RollingCorrelationShiftDetector(window=20, threshold=0.25).update(real0, real1)
```

`update(...)` 每次接收 `real0` 和 `real1`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器维护相邻的参考窗口和近期窗口；两窗口相关系数之差超过阈值时输出其符号。

## 递推公式

\[
\rho^R_t=\operatorname{corr}(R^x_t,R^y_t),\qquad \rho^B_t=\operatorname{corr}(B^x_t,B^y_t),\qquad q_t=\rho^R_t-\rho^B_t
\]

\[
r_t=\begin{cases}1,&q_t>h\\-1,&q_t<-h\\0,&\text{否则}\end{cases}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class RollingCorrelationShiftDetector` 中实现。

## 参考资料

- [ChartSchool：相关系数](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/correlation-coefficient)
