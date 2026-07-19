# RollingMeanVarianceShiftDetector

## 摘要

`RollingMeanVarianceShiftDetector` 联合检测相邻窗口之间的因果均值和方差位移。

## 更新 API

```python
result = rtta.RollingMeanVarianceShiftDetector(window=20, threshold=3.0).update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器分别计算均值差的 z 分数和方差比的对数，再组合其幅度；信号方向由贡献较大的分量确定。

## 递推公式

\[
z^\mu_t=\frac{\bar x^R_t-\bar x^B_t}{\sqrt{\sigma^{2,R}_t/n+\sigma^{2,B}_t/n+\epsilon}},\qquad z^\sigma_t=\log\left(\frac{\sigma^{2,R}_t+\epsilon}{\sigma^{2,B}_t+\epsilon}\right)
\]

\[
q_t=\sqrt{(z^\mu_t)^2+w(z^\sigma_t)^2},\qquad d_t=\begin{cases}z^\mu_t,&|z^\mu_t|\ge|\sqrt w z^\sigma_t|\\\sqrt w z^\sigma_t,&\text{否则}\end{cases}
\]

\[
y_t=\begin{cases}\operatorname{sgn}(d_t),&q_t>h\\0,&\text{否则}\end{cases}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class RollingMeanVarianceShiftDetector` 中实现。

## 参考资料

- [背景资料：变化检测](https://en.wikipedia.org/wiki/Change_detection)
