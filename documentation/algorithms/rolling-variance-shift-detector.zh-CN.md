# RollingVarianceShiftDetector

## 摘要

`RollingVarianceShiftDetector` 以对数方差比检测相邻窗口之间的因果方差位移。

## 更新 API

```python
result = rtta.RollingVarianceShiftDetector(window=20, threshold=1.0).update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器维护参考窗口与近期窗口，比较两者方差的对数比；绝对值超过阈值时输出方向。

## 递推公式

\[
\sigma^{2,R}_t=\operatorname{var}(R_t),\qquad \sigma^{2,B}_t=\operatorname{var}(B_t)
\]

\[
q_t=\log\left(\frac{\sigma^{2,R}_t+\epsilon}{\sigma^{2,B}_t+\epsilon}\right),\qquad r_t=\begin{cases}1,&q_t>h\\-1,&q_t<-h\\0,&\text{否则}\end{cases}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class RollingVarianceShiftDetector` 中实现。

## 参考资料

- [背景资料：F 检验](https://en.wikipedia.org/wiki/F-test)
