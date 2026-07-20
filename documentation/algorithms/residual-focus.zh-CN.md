# 残差 FOCuS（ResidualFOCuS）

## 摘要

`ResidualFOCuS` 把 FOCuS 均值变点检测器应用于**残差或新息**序列（模型误差、对冲残差、滤波器新息等）。其引擎与 `FOCuS` 完全相同；类名用于明确预期的输入语义。

## 更新 API

```python
import rtta

ind = rtta.ResidualFOCuS(threshold=10.0, mu0=0.0, sigma=1.0, max_candidates=200)
result = ind.update(residual)
# result.signal ∈ {-1, 0, +1}, result.statistic
```

`advance(...)` 更新状态但不返回结果。构造函数参数与 `FOCuS` 一致。

## 工作原理

基于模型的监控通常可以归结为“残差是否仍是零均值噪声？”把预白化残差或模型残差送入 FOCuS，可以检测均值漂移；若直接对价格水平使用 CUSUM，这种漂移可能与趋势混淆。典型管线为：

1. 在线拟合预测模型（回归、Kalman、配对残差）。
2. 把 \(r_t=y_t-\hat y_t\)（或经过 z 分数标准化的新息）流式送入 `ResidualFOCuS`。
3. 当残差均值的变化超过 GLR 阈值时触发。

在模型设定正确时，应取 \(\mu_0=0\)，并使 \(\sigma\) 与残差尺度匹配。

## 递推公式

与 `FOCuS` 完全相同，只是把观测 \(x_t\) 替换为残差 \(r_t\)：

\[
y_t = r_t - \mu_0,
\]

候选项 \((S,n)\) 更新为 \((S+y_t,n+1)\)，并新增 \((y_t,1)\)；随后按均值支配关系修剪，并计算：

\[
\Lambda_t = \max \frac{S^2}{2\sigma^2 n},\qquad
\mathrm{signal}_t = \operatorname{sign}(S^\star)\ \text{若}\ \Lambda_t \ge h,\ \text{否则为}\ 0.
\]

完整 FOCuS 递推和修剪细节见 [FOCuS](focus.zh-CN.md)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class ResidualFOCuS` 中实现，成员为 `FOCuS focus_`；`update` / `advance` / `batch_array` / `last` 都转发给该引擎。

## 参考资料

- [Romano 等，FOCuS（arXiv:2110.08205）](https://arxiv.org/abs/2110.08205)
