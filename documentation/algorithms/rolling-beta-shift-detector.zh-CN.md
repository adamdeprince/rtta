# RollingBetaShiftDetector

## 摘要

`RollingBetaShiftDetector` 以因果方式比较相邻窗口的 beta 变化。

## 更新 API

```python
result = rtta.RollingBetaShiftDetector(window=20, threshold=0.25).update(real0, real1)
```

`update(...)` 每次接收 `real0` 和 `real1`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器比较参考窗口与近期窗口；过期的近期样本移入参考窗口。两窗口 beta 之差超过阈值时输出其符号。

## 递推公式

\[
\beta^R_t=\frac{\operatorname{cov}(R^x_t,R^y_t)}{\operatorname{var}(R^y_t)},\qquad \beta^B_t=\frac{\operatorname{cov}(B^x_t,B^y_t)}{\operatorname{var}(B^y_t)},\qquad q_t=\beta^R_t-\beta^B_t
\]

\[
r_t=\begin{cases}1,&q_t>h\\-1,&q_t<-h\\0,&\text{否则}\end{cases}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class RollingBetaShiftDetector` 中实现。

## 参考资料

- [背景资料：Beta](https://www.investopedia.com/terms/b/beta.asp)
