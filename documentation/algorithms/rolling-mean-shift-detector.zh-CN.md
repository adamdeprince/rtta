# RollingMeanShiftDetector

## 摘要

`RollingMeanShiftDetector` 用双样本 z 分数检测相邻窗口之间的因果均值位移。

## 更新 API

```python
result = rtta.RollingMeanShiftDetector(window=20, threshold=3.0).update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器维护参考窗口和近期窗口，并比较两者均值；标准化差异超过阈值时输出方向。

## 递推公式

\[
\bar x^R_t,\sigma^{2,R}_t=\operatorname{stats}(R_t),\qquad \bar x^B_t,\sigma^{2,B}_t=\operatorname{stats}(B_t)
\]

\[
q_t=\frac{\bar x^R_t-\bar x^B_t}{\sqrt{\sigma^{2,R}_t/n+\sigma^{2,B}_t/n+\epsilon}},\qquad r_t=\begin{cases}1,&q_t>h\\-1,&q_t<-h\\0,&\text{否则}\end{cases}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class RollingMeanShiftDetector` 中实现。

## 参考资料

- [背景资料：双样本检验](https://en.wikipedia.org/wiki/Student%27s_t-test)
