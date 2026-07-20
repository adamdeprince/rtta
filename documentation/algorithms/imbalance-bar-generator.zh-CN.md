# 失衡 K 线生成器（ImbalanceBarGenerator）

## 摘要

`ImbalanceBarGenerator` 构建 López de Prado 风格的**成交量失衡 K 线**：有符号成交量（按收盘价逐笔规则确定符号）持续累积，直到其绝对值达到阈值；随后结束当前 K 线并重置失衡量。

## 更新 API

```python
import rtta

ind = rtta.ImbalanceBarGenerator(threshold=10000.0)
result = ind.update(close, volume)
# result.bar_open, bar_close, bar_high, bar_low, bar_volume,
# result.direction, result.complete, result.bars
```

`bar_volume` 为 \(|I_t|\)（有符号成交量累计值的绝对值）。完成 K 线时，`direction` 为失衡量的符号。平盘逐笔（收盘价不变）的贡献符号为 \(0\)。

## 工作原理

当买入量与卖出量变得足够不对称时，失衡 K 线结束。因此，单边市场中的 K 线频率会上升，而订单流均衡时则下降。符号采用逐笔规则：向上跳动 \(\Rightarrow+1\)，向下跳动 \(\Rightarrow-1\)，平盘 \(\Rightarrow0\)。与成交量 K 线 / 成交额 K 线不同，单次逐笔中超出阈值的部分不会拆分成多根 K 线；完成一根后，失衡量直接重置为零。

## 递推公式

令 \(V^\star>0\) 为 `threshold`。首笔用于初始化 OHLC 和 \(I=0\)，此后：

\[
s_t =
\begin{cases}
+1, & c_t > c_{t-1},\\
-1, & c_t < c_{t-1},\\
0, & c_t = c_{t-1},
\end{cases}
\qquad
I_t = I_{t-1} + s_t\,\max(v_t,0).
\]

用 \(c_t\) 更新运行最高价 \(H\) 和最低价 \(L\)。若 \(|I_t|\ge V^\star\)：

\[
\begin{aligned}
\mathrm{complete}&=1,\ \mathrm{bars}=1,\\
\mathrm{direction}&=\operatorname{sign}(I_t)\quad(I_t\ge0\text{ 时为 }+1),\\
\mathrm{bar\_volume}&=|I_t|,\\
I &\leftarrow 0,\quad O,H,L \leftarrow c_t.
\end{aligned}
\]

否则，\(\mathrm{complete}=0\)、\(\mathrm{bar\_volume}=|I_t|\)、方向为 \(0\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class ImbalanceBarGenerator` 中实现。结果类型为 `InformationBarResult`。

## 参考资料

- [López de Prado，《Advances in Financial Machine Learning》（失衡 K 线）](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
