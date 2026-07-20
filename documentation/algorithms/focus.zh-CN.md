# FOCuS

## 摘要

`FOCuS` 是一种双侧**函数式在线 CUSUM（Functional Online CUSUM）**均值变点检测器，按 Romano、Eckley、Fearnhead 与 Rigaill 的方法修剪候选项。每次更新都会返回 \(\{-1,0,+1\}\) 中的信号，以及所有活跃候选项中的最大似然比统计量。

## 更新 API

```python
import rtta

ind = rtta.FOCuS(threshold=10.0, mu0=0.0, sigma=1.0, max_candidates=200)
result = ind.update(value)
# result.signal ∈ {-1, 0, +1}, result.statistic ≥ 0
```

`advance(...)` 更新状态但不返回结果。触发后（`signal ≠ 0`），候选集会被清空，检测从头开始。

## 工作原理

FOCuS 维护一个经过修剪的候选变点位置集合。对于已知变化前均值 \(\mu_0\) 和方差 \(\sigma^2\) 的高斯观测，每个候选项保存从假定变点以来的中心化累积和与长度。累积和为 \(S\)、长度为 \(n\) 的候选项，其高斯均值广义似然比（GLR）统计量为：

\[
\Lambda = \frac{S^2}{2\sigma^2 n}.
\]

函数式修剪依据均值顺序与较短长度移除被支配的候选项，使计算成本与少量候选项线性相关（每侧上限为 `max_candidates`）。当 \(\max\Lambda\ge h\) 时，检测器输出获胜累积和的符号并重置。

## 递推公式

将观测中心化：\(y_t=x_t-\mu_0\)。令 \(t-1\) 时的候选集由二元组 \((S^{(j)},n^{(j)})\) 构成。形成更新后的多重集：

\[
\mathcal{C}'_t = \bigl\{(y_t, 1)\bigr\}
\cup
\bigl\{(S^{(j)} + y_t,\ n^{(j)}+1)\bigr\}_j.
\]

对 \(\mathcal C'_t\) 中累积和为正与为负的候选项分别修剪：按均值 \(S/n\) 排序（正值升序、负值降序），只保留长度严格递减的候选项（支配修剪），然后将每侧限制为 `max_candidates` 个。将修剪后的集合记为 \(\mathcal C_t\)。

统计量与信号为：

\[
\Lambda_t = \max_{(S,n)\in\mathcal{C}_t,\,n>0}
\frac{S^2}{2\sigma^2 n}
\quad
\bigl(\text{实现为 } S^2 \cdot (1/(2\sigma^2)) / n\bigr),
\]

\[
\mathrm{signal}_t =
\begin{cases}
\operatorname{sign}(S^\star), & \Lambda_t \ge h,\\
0, & \text{其他情况},
\end{cases}
\]

其中 \(S^\star\) 是使统计量最大的候选项累积和（\(S^\star\ge0\Rightarrow+1\)）。触发时，\(\mathcal C_t\leftarrow\emptyset\)。方差设有下限：\(\sigma^2\leftarrow\max(\sigma^2,10^{-18})\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class FOCuS`（`prune_candidates`）中实现。`ResidualFOCuS` 是一个轻量封装，把残差送入同一引擎。规范文档路径为 `focus.md`（不是 `fo-cu-s.md`）。

## 参考资料

- [Romano、Eckley、Fearnhead 与 Rigaill，《Fast Online Changepoint Detection via Functional Pruning CUSUM Statistics》（JMLR / arXiv）](https://arxiv.org/abs/2110.08205)
