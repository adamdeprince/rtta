# 多同类资产订单流失衡（MultiPeerOrderFlowImbalance）

## 摘要

`MultiPeerOrderFlowImbalance` 把跨资产 OFI 冲击扩展到一个**同类资产篮子**。每个时点先对同类资产 OFI 等权平均，得到 `peer_mean`；再以该均值解释本资产收益率作滚动 OLS，得到 `beta`、`impact` 和 `residual`。

## 更新 API

```python
import numpy as np
import rtta

ind = rtta.MultiPeerOrderFlowImbalance(window=50, fillna=True)
result = ind.update(own_return, peer_ofis)  # peer_ofis 形状为 (n_peers,)
# result.beta, result.impact, result.residual, result.peer_mean

# 批量：own_return (T,)，peer_ofis (T, n_peers)
batch = ind.batch(own_returns, peer_ofi_matrix)
```

`advance(...)` 更新状态但不返回结果。同类资产向量为空时返回 NaN。

## 工作原理

多标的订单流通常以同类资产 OFI 向量的形式到达。等权缩减：

\[
\bar{f}_t = \frac{1}{P}\sum_{p=1}^{P} f_{t,p}
\]

可作为单一跨资产冲击解释变量。用 \(\bar f\) 解释本资产收益率所得的滚动 \(\beta\)，等价于以 \(\bar f_t\) 作为同类资产特征的 `CrossAssetOrderFlowImbalance`。相较带权多同类资产变体，这是一个轻量级篮子方案。

## 递推公式

对于同类资产向量 \(f_t\in\mathbb R^P\)（\(P\ge1\)）：

\[
\bar{f}_t = \frac{1}{P}\sum_{p=1}^{P} f_{t,p}.
\]

把 \((x_t,y_t)=(r_t,\bar f_t)\) 推入长度为 \(W\) 的 `RollingPairStats`。设滚动和为 \(S_x,S_y,S_{xy},S_{y^2}\)，且 \(n=|W_t|\)：

\[
\beta_t = \frac{n\,S_{xy} - S_x S_y}{n\,S_{y^2} - S_y^2},
\]

\[
\mathrm{impact}_t = \beta_t\,\bar{f}_t,\qquad
\mathrm{residual}_t = r_t - \mathrm{impact}_t,\qquad
\mathrm{peer\_mean}_t = \bar{f}_t.
\]

若 `fillna=False` 且窗口尚未填满，\(\beta\)、冲击和残差均为 NaN；`peer_mean` 仍为 \(\bar f_t\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class MultiPeerOrderFlowImbalance` 中实现。批量输入至少需要一个同类资产列。支持 float32 和 float64 同类资产向量。

## 参考资料

- [Cont、Kukanov 与 Stoikov，《The Price Impact of Order Book Events》（arXiv:1011.6402）](https://arxiv.org/abs/1011.6402)
