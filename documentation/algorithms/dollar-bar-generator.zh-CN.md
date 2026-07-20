# 成交额 K 线生成器（DollarBarGenerator）

## 摘要

`DollarBarGenerator` 在累计**成交额** \(|\mathrm{close}|\cdot\mathrm{volume}\) 达到阈值时结束一根信息 K 线。在 López de Prado 的信息驱动 K 线体系中，它是成交量 K 线按名义金额计量的对应形式。

## 更新 API

```python
import rtta

ind = rtta.DollarBarGenerator(threshold=1.0e6)
result = ind.update(close, volume)
# result.bar_open, bar_close, bar_high, bar_low, bar_volume,
# result.direction, result.complete, result.bars
```

`bar_volume` 是成交额累计值（与阈值单位相同），不是股数。`complete`、`bars` 和 `direction` 遵循与 `VolumeBarGenerator` 相同的约定。

## 工作原理

成交额（名义金额）K 线在每有固定金额完成交易时对市场采样。这样可以部分消除价格水平的影响：无论证券价格是 \$10 还是 \$1000，\$1M 的阈值都大致代表相同规模的经济活动。RTTA 累积 \(|c_t|\cdot v_t^+\)，并可在一次更新内将超出阈值的部分拆分到多根 K 线。

## 递推公式

令 \(D^\star>0\) 为 `threshold`，\(d_t=|c_t|\,\max(v_t,0)\)。

在更新 \(c_t\) 的运行最高价和最低价后，累积 \(A_t\leftarrow A_{t-1}+d_t\)。当 \(A_t\ge D^\star\) 时重复：

\[
\begin{aligned}
\mathrm{complete} &\leftarrow 1,\quad
\mathrm{bars} \mathrel{+}= 1,\\
\mathrm{direction} &\leftarrow
\begin{cases}+1,& c_t \ge O\\ -1,& c_t < O\end{cases},\\
\text{输出} &\ (O,c_t,H,L),\ \mathrm{bar\_volume}=D^\star,\\
A_t &\leftarrow A_t - D^\star,\quad O,H,L \leftarrow c_t.
\end{aligned}
\]

未完成 K 线的 \(\mathrm{bar\_volume}=A_t\)。首个时点以 \(c_t\) 初始化 OHLC；若 \(d_t\ge D^\star\)，也可以立即完成一根 K 线。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class DollarBarGenerator` 中实现。默认阈值为 \(10^6\)。

## 参考资料

- [López de Prado，《Advances in Financial Machine Learning》（成交额 K 线）](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
