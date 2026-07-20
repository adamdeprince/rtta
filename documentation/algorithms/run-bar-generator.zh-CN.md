# 游程 K 线生成器（RunBarGenerator）

## 摘要

`RunBarGenerator` 实现 López de Prado 的**逐笔游程 K 线**：当连续同号逐笔的数量（按收盘价逐笔规则确定符号）达到整数阈值时，结束当前 K 线。可选的 `update(close, volume)` 仍以逐笔数决定完成时机，但会把累计游程成交量作为 `bar_volume` 报告。

## 更新 API

```python
import rtta

ind = rtta.RunBarGenerator(threshold=10)
result = ind.update(close)                 # 按逐笔数计算游程
result = ind.update(close, volume)         # 规则相同；bar_volume = 游程成交量
# result.bar_open, bar_close, bar_high, bar_low, bar_volume,
# result.direction, result.complete, result.bars
```

平盘逐笔（\(c_t=c_{t-1}\)）不会中断游程，也不会增加计数。符号相反时，以计数 1 开始新游程，并将 OHLC 重置为当前收盘价。

## 工作原理

游程 K 线不是按日历时间采样，而是在一串买方或卖方发起的成交达到指定长度时采样。它强调订单流的持续性：同号序列越长，结束 K 线的频率越高。RTTA 的主要定义使用**逐笔计数**；双参数重载会累积成交量用于报告，但仍以 `run_count` 与阈值比较。

## 递推公式

令 \(N^\star=\max(\mathrm{threshold},1)\)。逐笔符号为：

\[
s_t =
\begin{cases}
+1, & c_t > c_{t-1},\\
-1, & c_t < c_{t-1},\\
0, & \text{平盘（游程逻辑忽略）}.
\end{cases}
\]

若 \(s_t\ne0\)：

\[
\begin{aligned}
s_t = \sigma \text{（当前游程符号）} &\Rightarrow n \leftarrow n+1,\\
s_t \neq \sigma &\Rightarrow \sigma \leftarrow s_t,\ n \leftarrow 1,\ O,H,L \leftarrow c_t.
\end{aligned}
\]

当 \(n\ge N^\star\) 时：

\[
\mathrm{complete}=1,\ \mathrm{bars}=1,\ \mathrm{direction}=\sigma,\
\mathrm{bar\_volume}=n\ \text{（使用成交量重载时为游程成交量）},\
n\leftarrow0,\ \sigma\leftarrow0,\ O,H,L\leftarrow c_t.
\]

成交量重载：符号规则相同；游程继续时 \(V_{\mathrm{run}}\mathrel{+}=v_t^+\)；仍要求 \(n\ge N^\star\) 才完成，随后 \(V_{\mathrm{run}}\leftarrow0\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class RunBarGenerator` 中实现。默认阈值为 10 笔。

## 参考资料

- [López de Prado，《Advances in Financial Machine Learning》（游程 K 线）](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
