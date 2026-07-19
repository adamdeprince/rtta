# PredictionErrorDriftDetector

## 摘要

`PredictionErrorDriftDetector` 用 EWMA 检测绝对预测误差的漂移。

## 更新 API

```python
result = rtta.PredictionErrorDriftDetector().update(prediction, actual)
```

`update(...)` 每次接收 `prediction` 和 `actual`；只推进状态时可调用 `advance(...)`。

## 工作原理

绝对预测误差先相对此前样本估计的 EWMA 均值和方差标准化，再通过上侧回滞生成持续的漂移状态。

## 递推公式

\[
e_t=|actual_t-prediction_t|,\qquad q_t=\frac{e_t-\mu_{t-1}}{\sqrt{\max(\sigma^2_{t-1},\epsilon)}}
\]

\[
\mu_t=\mu_{t-1}+\alpha(e_t-\mu_{t-1}),\qquad \sigma^2_t=(1-\alpha)(\sigma^2_{t-1}+\alpha(e_t-\mu_{t-1})^2)
\]

\[
r_t=\begin{cases}1,&r_{t-1}=0\text{ 且 }q_t\ge e\\0,&r_{t-1}=1\text{ 且 }q_t\le x\\r_{t-1},&\text{否则}\end{cases},\qquad x<e
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class PredictionErrorDriftDetector` 中实现。

## 参考资料

- [背景资料：概念漂移](https://en.wikipedia.org/wiki/Concept_drift)
