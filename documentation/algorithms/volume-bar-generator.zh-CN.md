# 成交量 K 线生成器（VolumeBarGenerator）

## 摘要

`VolumeBarGenerator` 是 López de Prado 风格的**信息 K 线**：累计成交量达到固定阈值时结束一根 K 线。每次 `update(close, volume)` 都推进尚未完成的 OHLC；若成交量超出阈值，还可能一次完成一根或多根 K 线。

## 更新 API

```python
import rtta

ind = rtta.VolumeBarGenerator(threshold=10000.0)
result = ind.update(close, volume)
# result.bar_open, bar_close, bar_high, bar_low, bar_volume,
# result.direction, result.complete, result.bars
```

本次逐笔至少结束一根 K 线时，`complete` 为 \(1\)；`bars` 是本次逐笔结束的 K 线数量（如果单笔成交量跨过多个阈值，可以大于 1）。若已完成 K 线的收盘价 \(\ge\) 开盘价，`direction` 为 \(+1\)，否则为 \(-1\)（没有完成时为 0）。未完成的逐笔报告运行中的 `bar_volume = accum`。

## 工作原理

时钟时间并不均匀；成交量时间按市场活跃度归一化，使每根 K 线大致包含相同成交数量。RTTA 累积 \(\max(\mathrm{volume},0)\)。只要累计值不低于阈值，就输出一根成交量等于 `threshold` 的完整 K 线，减去该数量，再以当前收盘价重新开始 OHLC（因此大额成交可以在一次调用中生成多根 K 线）。

## 递推公式

令 \(V^\star>0\) 为 `threshold`。首个时点令 \(O=H=L=c_t\)、\(A_t=v_t^+\)；若 \(A_t\ge V^\star\)，则完成一根 K 线并重置。

此后：

\[
H \leftarrow \max(H, c_t),\quad L \leftarrow \min(L, c_t),\quad
A_t \leftarrow A_{t-1} + v_t^+,\quad v_t^+ = \max(v_t, 0).
\]

当 \(A_t\ge V^\star\) 时重复：

\[
\begin{aligned}
\mathrm{complete} &\leftarrow 1,\\
\mathrm{bars} &\leftarrow \mathrm{bars}+1,\\
\mathrm{direction} &\leftarrow \mathbf{1}\{c_t \ge O\} - \mathbf{1}\{c_t < O\},\\
\text{输出 OHLC} &\leftarrow (O, c_t, H, L),\quad
\mathrm{bar\_volume} \leftarrow V^\star,\\
A_t &\leftarrow A_t - V^\star,\quad
O,H,L \leftarrow c_t.
\end{aligned}
\]

若没有完成任何 K 线，报告的成交量为运行累计值 \(A_t\)；否则，本次逐笔返回最后一根完整 K 线的字段（成交量为 \(V^\star\)）。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class VolumeBarGenerator` 中实现。结果类型为 `InformationBarResult`。

## 参考资料

- [López de Prado，《Advances in Financial Machine Learning》（信息驱动 K 线）](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
