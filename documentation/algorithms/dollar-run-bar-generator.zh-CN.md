# 成交额游程 K 线生成器（DollarRunBarGenerator）

## 摘要

`DollarRunBarGenerator` 生成**按成交额加权的游程 K 线**：只要逐笔规则的符号保持不变，就持续累积名义成交额 \(|\mathrm{close}|\cdot\mathrm{volume}\)；当同向游程成交额达到阈值时结束该 K 线。

## 更新 API

```python
import rtta

ind = rtta.DollarRunBarGenerator(threshold=1.0e6)
result = ind.update(close, volume)
# result.bar_open, bar_close, bar_high, bar_low, bar_volume,
# result.direction, result.complete, result.bars
```

`bar_volume` 是当前游程的成交额累计值。价格不变的逐笔不会改变游程；符号翻转时，则以当前笔的成交额重新开始游程。

## 工作原理

成交额游程 K 线把订单流方向的持续性与名义金额采样结合起来。当单边连续成交达到固定金额时，它就结束当前 K 线，融合了信息驱动 K 线文献中成交额 K 线与逐笔游程 K 线的思想。

## 递推公式

令 \(D^\star>0\) 为 `threshold`，\(d_t=|c_t|\,\max(v_t,0)\)。逐笔符号 \(s_t\) 的定义与成交量游程 K 线相同。

若 \(s_t\ne0\)：

\[
\begin{aligned}
s_t = \sigma &\Rightarrow D_{\mathrm{run}} \leftarrow D_{\mathrm{run}} + d_t,\\
s_t \neq \sigma &\Rightarrow \sigma \leftarrow s_t,\ D_{\mathrm{run}} \leftarrow d_t,\ O,H,L \leftarrow c_t.
\end{aligned}
\]

当 \(D_{\mathrm{run}}\ge D^\star\) 时：

\[
\begin{aligned}
\mathrm{complete}&=1,\ \mathrm{bars}=1,\ \mathrm{direction}=\sigma,\\
\mathrm{bar\_volume}&=D_{\mathrm{run}},\\
D_{\mathrm{run}}&\leftarrow 0,\ \sigma\leftarrow 0,\ O,H,L\leftarrow c_t.
\end{aligned}
\]

未完成 K 线的 \(\mathrm{bar\_volume}=D_{\mathrm{run}}\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class DollarRunBarGenerator` 中实现。默认阈值为 \(10^6\)。

## 参考资料

- [López de Prado，《Advances in Financial Machine Learning》（成交额 / 游程 K 线）](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
