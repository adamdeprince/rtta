# EWMAZScoreShiftDetector

## 摘要

`EWMAZScoreShiftDetector` 用因果 EWMA 均值和方差构造 z 分数，检测超过阈值的位移事件。

## 更新 API

```python
result = rtta.EWMAZScoreShiftDetector(alpha=0.05, threshold=3.0).update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

当前变动先用此前样本估计的 EWMA 均值和方差标准化。检测器再以所得 z 分数发出方向信号；单个异常事件触发后会重置基线，避免噪声演变为持续状态。

## 递推公式

\[
z_t=\frac{close_t-\mu_{t-1}}{\sqrt{\max(\sigma^2_{t-1},\epsilon)}}
\]

\[
y_t=\begin{cases}1,&z_t>h\\-1,&z_t<-h\\0,&\text{否则}\end{cases}
\]

\[
\mu_t=\mu_{t-1}+\alpha(close_t-\mu_{t-1}),\qquad \sigma^2_t=(1-\alpha)(\sigma^2_{t-1}+\alpha(close_t-\mu_{t-1})^2)
\]

当 (y_t\ne0) 时，C++ 实现把 (mu_t) 重置为当前收盘价，并清空方差估计。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class EWMAZScoreShiftDetector` 中实现。

## 参考资料

- [pandas：指数加权窗口](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
