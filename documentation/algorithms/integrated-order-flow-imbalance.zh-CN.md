# 综合订单流失衡（IntegratedOrderFlowImbalance）

## 摘要

`IntegratedOrderFlowImbalance` 将多档 Cont 风格 OFI 投影到各档事件协方差的在线第一主成分上。每个深度档位都贡献一个滚动 Cont 事件；EMA 协方差矩阵与幂迭代权重共同形成单一综合标量 `ofi`，并输出第一档权重 `weight_l1`。

## 更新 API

```python
import rtta

ind = rtta.IntegratedOrderFlowImbalance(
    levels=5, window=1, ema_alpha=0.05, fillna=True
)
result = ind.update(bid_price, bid_size, ask_price, ask_size)  # 长度为 `levels` 的向量
# result.ofi, result.weight_l1

batch = ind.batch(bid_prices, bid_sizes, ask_prices, ask_sizes)  # (n, levels)
```

`advance(...)` 更新状态但不返回结果。支持 float32/float64 深度向量和批量矩阵。

## 工作原理

原始多档 OFI 是向量 \(v_t\in\mathbb R^L\)。实际数据中的各分量通常高度共线；Cont 风格的“综合”OFI 取事件协方差矩阵的首要特征向量及其对应线性组合，把盘口深度压缩成一个因子。RTTA 维护：

1. 每档 Cont 事件及滚动和 \(v^{(\ell)}_t\)（映射与多档 OFI 相同）。
2. 向量 \(v_t\) 的指数移动协方差 \(\Sigma_t\)。
3. 从前一个权重向量开始，对 \(\Sigma_t\) 执行四步幂迭代，并重新定向，使盘口第一档权重非负。
4. 标量投影 \(\mathrm{ofi}_t=w_t^\top v_t\)。

初始权重为 \(w=(1,0,\ldots,0)\)。当 `fillna=False` 时，在 `count >= window` 之前输出为 NaN。

## 递推公式

Cont 事件 \(e^{(\ell)}_t\) 与滚动和 \(v^{(\ell)}_t=S^{(\ell)}_t\) 和 `MultiLevelOrderFlowImbalance` 完全相同。构造 \(v\) 时，未填满的窗口在 `fillna=True` 逻辑下可使用现有总和；否则强制为 \(0\)。

协方差 EMA 使用 \(\alpha=\mathrm{ema\_alpha}\)（截断到 \([10^{-6},1]\)）：

\[
\Sigma_t = (1-\alpha)\,\Sigma_{t-1} + \alpha\, v_t v_t^\top.
\]

从此前权重 \(w_{t-1}\) 开始作四步幂迭代：

\[
\tilde{w}^{(0)} = w_{t-1},\qquad
\tilde{w}^{(k+1)} = \frac{\Sigma_t \tilde{w}^{(k)}}{\|\Sigma_t \tilde{w}^{(k)}\|_2},
\quad k=0,1,2,3.
\]

翻转符号，使 \(\tilde w^{(4)}_0\ge0\)：

\[
w_t =
\begin{cases}
-\tilde{w}^{(4)}, & \tilde{w}^{(4)}_0 < 0,\\
\tilde{w}^{(4)}, & \text{其他情况}.
\end{cases}
\]

输出：

\[
\mathrm{ofi}_t = w_t^\top v_t,\qquad
\mathrm{weight\_l1}_t = w_{t,0}.
\]

若 `fillna=False` 且 \(t<W\)，两个字段均为 NaN。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class IntegratedOrderFlowImbalance` 中实现。协方差以行优先顺序存储在长度为 \(L^2\) 的向量中。批量输入形状必须为 `(n_samples, levels)`。

## 参考资料

- [Cont、Kukanov 与 Stoikov，《The Price Impact of Order Book Events》（arXiv:1011.6402）](https://arxiv.org/abs/1011.6402)
