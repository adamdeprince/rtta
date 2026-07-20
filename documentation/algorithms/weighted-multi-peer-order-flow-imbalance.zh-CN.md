# 加权多同类资产订单流失衡（WeightedMultiPeerOrderFlowImbalance）

## 摘要

`WeightedMultiPeerOrderFlowImbalance` 是 RTTA 对显式加权的同类资产篮子 OFI 的流式实现：每个时点先计算同类资产 OFI 的加权平均，再以该同类资产均值解释本资产收益率，得到滚动 beta、冲击和残差。

## 更新 API

```python
# peer_ofis 和 weights 的形状均为 (n_peers,)
result = rtta.WeightedMultiPeerOrderFlowImbalance(window=50).update(
    own_return, peer_ofis, weights
)
# result.beta, result.impact, result.residual, result.peer_mean

# 批量：own_return (T,)，peer_ofis (T, P)，weights (T, P)
batch = ind.batch(own_return, peer_ofis, weights)
```

权重相等时，可还原 `MultiPeerOrderFlowImbalance` 的同类资产均值。非正或非有限权重会被跳过；若所有权重都无效，实现会退回等权计算。

## 工作原理

等权篮子 OFI 对所有同类资产一视同仁。加权多同类资产 OFI 允许调用方提供流动性、ADV、beta 或行业权重，使同类资产压力指数成为一个投资组合，而不是简单平均。用该加权同类资产均值解释本资产收益率作滚动 OLS，得到 Cont 风格跨资产冲击及残余的特质收益率。

## 递推公式

令 \(r_t\) 为本资产收益率，\(f_{t,i}\) 为同类资产 OFI，\(w_{t,i}\) 为同类资产权重，其中 \(i=1,\ldots,P\)。令 \(W_t=\sum_iw_{t,i}\)，求和只包括为正的有限权重。

\[
\bar f_t = \frac{\sum_i w_{t,i} f_{t,i}}{W_t}
\]

对最近 \(n\) 个样本（`window`）的 \((r_t,\bar f_t)\) 维护滚动配对统计量，并计算 OLS beta \(\hat\beta_t\)。随后：

\[
\begin{aligned}
\operatorname{impact}_t &= \hat\beta_t\,\bar f_t \\
\operatorname{residual}_t &= r_t - \operatorname{impact}_t \\
\operatorname{peer\_mean}_t &= \bar f_t
\end{aligned}
\]

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class WeightedMultiPeerOrderFlowImbalance` 中实现，使用 `RollingPairStats`。与等权 `MultiPeerOrderFlowImbalance` 不同，每次更新都必须提供权重向量（批量接口则为权重矩阵）。

## 参考资料

- [Cont、Kukanov 与 Stoikov，《The Price Impact of Order Book Events》，arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- [市场微观结构综述中的跨资产 / 多标的订单流压力背景](https://arxiv.org/abs/1011.6402)
