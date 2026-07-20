# 跨资产订单流失衡（CrossAssetOrderFlowImbalance）

## 摘要

`CrossAssetOrderFlowImbalance` 估计**同类资产**订单流失衡序列对**本资产**收益率的滚动线性冲击：\(\beta\)、拟合冲击 \(\beta\cdot\mathrm{peer\_ofi}\) 以及残差收益率。它是一种 Cont 风格的多标的跨资产冲击特征。

## 更新 API

```python
import rtta

ind = rtta.CrossAssetOrderFlowImbalance(window=50, fillna=True)
result = ind.update(own_return, peer_ofi)
# result.beta, result.impact, result.residual, result.peer_ofi
```

`advance(...)` 更新状态但不返回结果。批量接口接受已对齐的 `own_return` 与 `peer_ofi` 数组。

## 工作原理

给定配对观测 \((r_t, f_t)\)，其中 \(r_t\) 是本资产收益率，\(f_t\) 是同类资产 OFI（或任意同类资产订单流特征），该类会在长度为 \(W\) 的窗口内维护 \(r\)、\(f\)、\(r^2\)、\(f^2\) 与 \(rf\) 的滚动和。Beta 是用 \(f\) 回归 \(r\) 的 OLS 斜率（`RollingPairStats::beta` 所用的矩形式不含截距项）：

\[
\beta = \frac{\mathrm{Cov}(r,f)}{\mathrm{Var}(f)}.
\]

随后由线性模型 \(r \approx \beta f\) 得出跨资产冲击和残差。

## 递推公式

每次将 \((x_t, y_t) = (r_t, f_t)\) 推入容量为 \(W\) 的窗口后，设运行总和为 \(S_x, S_y, S_{x^2}, S_{y^2}, S_{xy}\)，且 \(n = |W_t|\)：

\[
\beta_t = \frac{n\,S_{xy} - S_x S_y}{n\,S_{y^2} - S_y^2}
\quad\text{（安全除法；方差为零时取 \(0\)）}.
\]

\[
\mathrm{impact}_t = \beta_t\, f_t,\qquad
\mathrm{residual}_t = r_t - \mathrm{impact}_t,\qquad
\mathrm{peer\_ofi}_t = f_t.
\]

若 `fillna=False` 且窗口尚未填满，\(\beta\)、冲击与残差均为 `NaN`，但 `peer_ofi` 仍原样返回 \(f_t\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class CrossAssetOrderFlowImbalance` 中实现，使用 `RollingPairStats`（`x = own_return`、`y = peer_ofi`）。窗口长度至少为 2。

## 参考资料

- [Cont、Kukanov 与 Stoikov，《The Price Impact of Order Book Events》（arXiv:1011.6402）](https://arxiv.org/abs/1011.6402)
