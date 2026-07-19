# CalibrationDriftDetector

## 摘要

`CalibrationDriftDetector` 用 EWMA 跟踪概率校准误差的漂移。

## 更新 API

```python
result = rtta.CalibrationDriftDetector().update(probability, outcome)
```

`update(...)` 每次接收 `probability` 和 `outcome`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器计算预测概率与实际二元结果之间的绝对校准误差，对误差做 EWMA 平滑，再用进入/退出回滞生成稳定状态。

## 递推公式

令 \(z_t = (probability_t, outcome_t)\) 为一次更新接收的观测。

\[
e_t=|\mathbf{1}[outcome_t>0]-\operatorname{clip}(probability_t,0,1)|
\]

\[
q_t=\alpha e_t+(1-\alpha)q_{t-1}
\]

\[
r_t =
\begin{cases}
1, & r_{t-1} = 0 \text{ 且 } q_t \ge e \\
0, & r_{t-1} = 1 \text{ 且 } q_t \le x \\
r_{t-1}, & \text{否则}
\end{cases}, \qquad x < e
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class CalibrationDriftDetector` 中实现。

## 参考资料

- [背景资料：统计校准](https://en.wikipedia.org/wiki/Calibration_(statistics))
