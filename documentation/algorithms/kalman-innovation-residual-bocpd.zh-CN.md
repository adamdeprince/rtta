# Kalman 新息残差 BOCPD（KalmanInnovationResidualBOCPD）

## 摘要

`KalmanInnovationResidualBOCPD` 把 **Kalman 新息 z 分数**残差送入 **ResidualBOCPD**（有界 BOCPD）。它检测滤波器新息过程中的贝叶斯在线变点，并返回信号、变化概率（`score`）和残差。

## 更新 API

```python
import rtta

ind = rtta.KalmanInnovationResidualBOCPD(
    max_run_length=128,
    hazard=0.01,
    threshold=0.5,
    min_variance=1e-6,
    initial_price=float("nan"),
    dt=1.0,
    measurement_variance=0.25,
    fillna=True,
)
result = ind.update(close)
# result.signal, result.score（即概率）, result.residual
```

非有限新息不会送入 BOCPD；此时信号与分数返回零，残差则原样输出。

## 工作原理

残差构造与 `KalmanInnovationResidualFOCuS` 相同：恒速度 Kalman 滤波器产生 \(\rho_t=\nu_t/\sqrt{S_t}\)。这里不使用 FOCuS GLR，而由 **BOCPD** 建立残差游程长度的后验分布，并在 \(P(r_t=0)\ge\tau\) 时标记变点。它与 FOCuS 互为补充：BOCPD 为每个假设建模危险率以及在线残差均值/方差，而不是使用固定高斯 CUSUM 阈值。

## 递推公式

**1. 新息 z 分数** \(\rho_t\)，与 `KalmanInnovationZScore` / 残差 FOCuS 文档相同：

\[
\nu_t = c_t - H\hat{x}_{t|t-1},\qquad
\rho_t = \nu_t / \sqrt{H P_{t|t-1} H^\top + R}.
\]

**2. 对 \(\rho_t\) 应用 ResidualBOCPD**，参数为 `(max_run_length, hazard, threshold, min_variance)`——完整递推过程见 [残差 BOCPD](residual-bocpd.zh-CN.md) / [有界 BOCPD](bounded-bocpd.zh-CN.md)：

\[
\mathrm{score}_t = \pi_t(0),\qquad
\mathrm{signal}_t = \mathbf{1}\{\pi_t(0) \ge \tau\},\qquad
\mathrm{residual}_t = \rho_t.
\]

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class KalmanInnovationResidualBOCPD` 中实现（`KalmanInnovationZScore innov_` + `ResidualBOCPD bocpd_`）。结果类型为 `InnovationChangepointResult`（`score` 保存 BOCPD 概率）。

## 参考资料

- [Adams 与 MacKay，《Bayesian Online Changepoint Detection》（arXiv:0710.3742）](https://arxiv.org/abs/0710.3742)
- [Welch 与 Bishop，《An Introduction to the Kalman Filter》](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)
