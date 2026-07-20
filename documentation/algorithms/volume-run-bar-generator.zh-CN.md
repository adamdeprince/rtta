# 成交量游程 K 线生成器（VolumeRunBarGenerator）

## 摘要

`VolumeRunBarGenerator` 生成**按成交量加权的游程 K 线**：只要逐笔规则符号保持相同，就持续累积成交量；当游程成交量达到阈值时结束 K 线（而不是按逐笔计数）。

## 更新 API

```python
import rtta

ind = rtta.VolumeRunBarGenerator(threshold=10000.0)
result = ind.update(close, volume)
# result.bar_open, bar_close, bar_high, bar_low, bar_volume,
# result.direction, result.complete, result.bars
```

平盘逐笔不会改变游程。符号相反时，以当前笔的成交量开始新游程，并把 OHLC 重置为该收盘价。

## 工作原理

标准游程 K 线按逐笔计数；成交量游程 K 线则按每笔同号成交的数量加权，因此少数大额成交可以和许多小额成交一样快速完成一根 K 线。它是在符号持续状态下，成交量 K 线所对应的游程形式。

## 递推公式

令 \(V^\star>0\) 为 `threshold`，并令：

\[
s_t =
\begin{cases}
+1, & c_t > c_{t-1},\\
-1, & c_t < c_{t-1},\\
0, & \text{平盘}.
\end{cases}
\]

若 \(s_t\ne0\)：

\[
\begin{aligned}
s_t = \sigma &\Rightarrow V_{\mathrm{run}} \leftarrow V_{\mathrm{run}} + v_t^+,\\
s_t \neq \sigma &\Rightarrow \sigma \leftarrow s_t,\ V_{\mathrm{run}} \leftarrow v_t^+,\ O,H,L \leftarrow c_t,
\end{aligned}
\]

其中 \(v_t^+=\max(v_t,0)\)。游程活跃期间，最高价/最低价跟踪 \(c_t\)。

当 \(V_{\mathrm{run}}\ge V^\star\) 时：

\[
\begin{aligned}
\mathrm{complete}&=1,\ \mathrm{bars}=1,\ \mathrm{direction}=\sigma,\\
\mathrm{bar\_volume}&=V_{\mathrm{run}},\\
V_{\mathrm{run}}&\leftarrow 0,\ \sigma\leftarrow 0,\ O,H,L\leftarrow c_t.
\end{aligned}
\]

否则 \(\mathrm{bar\_volume}=V_{\mathrm{run}}\)，`complete` / `bars` / `direction` 均为零（未完成游程只在完成时才报告方向）。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class VolumeRunBarGenerator` 中实现。结果类型为 `InformationBarResult`。

## 参考资料

- [López de Prado，《Advances in Financial Machine Learning》（游程 K 线 / 成交量采样）](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
