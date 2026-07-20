# Hawkes 强度（HawkesIntensity）

## 摘要

`HawkesIntensity` 是 RTTA 对事件时间指数型 Hawkes 过程强度的流式实现：常数基准强度加上一个自激残差；该残差在事件之间指数衰减，并在新事件到达时跳升。

## 更新 API

```python
result = rtta.HawkesIntensity(
    mu=1.0, alpha=0.5, beta=1.0, fillna=True
).update(time, jump=1.0)
# result.intensity, result.excitation, result.baseline
```

`time` 是实数值事件时钟（秒、交易所时间戳，或任何与 `beta` 单位一致的单调尺度）。`jump` 是事件的标记大小（默认为 `1.0`）。`batch(time, jump)` 或使用单位跳跃的 `batch(time)` 返回多输出数组。

## 工作原理

一元指数 Hawkes 过程的条件强度为：

\[
\lambda(t) = \mu + \sum_{t_i < t} \alpha\, e^{-\beta(t-t_i)}
\]

其中 \(\mu\) 是基准强度，\(\alpha\) 是激发幅度，\(\beta\) 是衰减率。RTTA 以递归形式维护残余激发量 \(A_t\)，使每次更新均为 \(O(1)\)：先将此前激发量乘以 \(e^{-\beta\Delta t}\) 衰减，再加上 \(\alpha\cdot\operatorname{jump}\)。事件聚集会把强度推升到基准值以上；平静期则使其回落到 \(\mu\)。

## 递推公式

状态为残余激发量 \(A\) 和最近事件时间 \(t_{\mathrm{last}}\)。在时刻 \(t\) 到达的首个事件，其跳跃大小为 \(j\)：

\[
A \leftarrow \alpha j, \qquad
\lambda = \mu + A
\]

后续事件在时刻 \(t\) 到达，跳跃大小为 \(j\)，且 \(\Delta t=t-t_{\mathrm{last}}\)：

\[
A \leftarrow A\, e^{-\beta\,\lvert\Delta t\rvert} + \alpha j
\]

\[
\lambda_t = \mu + A, \qquad
\operatorname{excitation}_t = A, \qquad
\operatorname{baseline}_t = \mu
\]

（若时间倒退，RTTA 仍使用 \(\lvert\Delta t\rvert\) 作衰减并应用跳跃，使数据流始终有定义。）

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class HawkesIntensity` 中实现。参数设有下限，使 \(\mu,\alpha\ge0\) 且 \(\beta>0\)。

## 参考资料

- [Hawkes，《Spectra of some self-exciting and mutually exciting point processes》，*Biometrika*，1971](https://doi.org/10.1093/biomet/58.1.83)
- [Bacry、Mastromatteo、Muzy，《Hawkes processes in finance》，arXiv:1502.04592](https://arxiv.org/abs/1502.04592)
