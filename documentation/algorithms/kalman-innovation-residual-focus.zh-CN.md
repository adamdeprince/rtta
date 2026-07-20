# Kalman 新息残差 FOCuS（KalmanInnovationResidualFOCuS）

## 摘要

`KalmanInnovationResidualFOCuS` 把 **Kalman 新息 z 分数**残差送入 **FOCuS** 变点检测。每次价格更新都会产生一个标准化新息；FOCuS 监测该残差的均值漂移，并返回信号、FOCuS 统计量（`score`）和残差本身。

## 更新 API

```python
import rtta

ind = rtta.KalmanInnovationResidualFOCuS(
    focus_threshold=10.0,
    focus_sigma=1.0,
    initial_price=float("nan"),
    dt=1.0,
    measurement_variance=0.25,
    fillna=True,
)
result = ind.update(close)
# result.signal, result.score, result.residual
```

若新息不是有限值，FOCuS 不会推进；`signal` / `score` 保持为 0，非有限残差则原样输出。

## 工作原理

价格水平是非平稳的；在模型成立时，恒速度 Kalman 滤波器的新息近似为白噪声残差。用预测新息标准差进行缩放，可得到适合零均值 FOCuS（\(\mu_0=0\)）的 z 分数残差。此后，变点可标记结构性断点：导致新息尺度失配的波动率跳跃、速度状态变化，或被 FOCuS 视作残差均值漂移的测量异常。

内部 Kalman 默认值与便利构造函数使用的 `KalmanInnovationZScore` 过程 / 测量设置一致（位置/速度过程方差分别为 \(10^{-4}\) / \(10^{-3}\)，初始方差为 1，等等）。

## 递推公式

**1. 新息 z 分数**（恒速度 Kalman，观测 \(z_t=c_t\)）：

\[
\hat{x}_{t|t-1} = F\hat{x}_{t-1|t-1},\quad
P_{t|t-1} = F P_{t-1|t-1} F^\top + Q,
\]

\[
\nu_t = z_t - H\hat{x}_{t|t-1},\quad
S_t = H P_{t|t-1} H^\top + R,\quad
\rho_t = \frac{\nu_t}{\sqrt{S_t}}.
\]

**2. 对 \(\rho_t\) 应用 FOCuS**，阈值 \(h=\)`focus_threshold`，\(\mu_0=0\)，\(\sigma=\)`focus_sigma`（参见 [FOCuS](focus.zh-CN.md)）：

\[
y_t = \rho_t,\quad
\Lambda_t = \max_{(S,n)} \frac{S^2}{2\sigma^2 n},\quad
\mathrm{signal}_t =
\begin{cases}
\pm 1, & \Lambda_t \ge h,\\
0, & \text{其他情况}.
\end{cases}
\]

输出：\(\mathrm{score}_t=\Lambda_t\)，\(\mathrm{residual}_t=\rho_t\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class KalmanInnovationResidualFOCuS` 中实现（`KalmanInnovationZScore innov_` + `FOCuS focus_`）。结果类型为 `InnovationChangepointResult`。

## 参考资料

- [Romano 等，FOCuS（arXiv:2110.08205）](https://arxiv.org/abs/2110.08205)
- [Welch 与 Bishop，《An Introduction to the Kalman Filter》](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)
