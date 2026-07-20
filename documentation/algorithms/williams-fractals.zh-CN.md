# Williams 分形（WilliamsFractals）

## 摘要

`WilliamsFractals` 是 RTTA 对 Bill Williams 五根 K 线分形检测器的流式实现。向上分形在五个最高价形态的中间 K 线上确认；向下分形则在五个最低价形态的中间 K 线上确认。确认始终滞后两根 K 线。

## 更新 API

```python
result = rtta.WilliamsFractals(fillna=True).update(high, low)
# result.up, result.down
```

取得五根 K 线之前，`fillna=True` 返回 `up=0`、`down=0`；`fillna=False` 则使两个字段都返回 `NaN`。

## 工作原理

Williams 分形标记局部转折点：高点分形所在 K 线的最高价严格高于左右各两根 K 线的最高价；低点分形所在 K 线的最低价严格低于左右各两根 K 线的最低价。由于形态需要后续两根 K 线才能确认，只有在中间 K 线形成两根 K 线之后，才知道分形是否成立。RTTA 在确认 K 线（即最新 K 线完成五根 K 线窗口时）输出分形价格；没有确认分形时输出 `0.0`。

## 递推公式

维护最近五个最高价和最低价的移位寄存器，索引 \(0\) 为最旧、\(4\) 为最新。每次更新且至少已有五根 K 线时：

\[
\begin{aligned}
\operatorname{up}_t &=
\begin{cases}
H^{(2)}, &
H^{(2)} > H^{(0)},\;
H^{(2)} > H^{(1)},\;
H^{(2)} > H^{(3)},\;
H^{(2)} > H^{(4)} \\
0, & \text{其他情况}
\end{cases}
\\[6pt]
\operatorname{down}_t &=
\begin{cases}
L^{(2)}, &
L^{(2)} < L^{(0)},\;
L^{(2)} < L^{(1)},\;
L^{(2)} < L^{(3)},\;
L^{(2)} < L^{(4)} \\
0, & \text{其他情况}
\end{cases}
\end{aligned}
\]

其中 \(H^{(i)},L^{(i)}\) 为寄存器条目。只有两个形态同时成立时，两个输出才能在同一根 K 线上都非零（并不常见）。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class WilliamsFractals` 中实现。结果字段为 `up` 和 `down`。

## 参考资料

- [Investopedia：Fractal Indicator](https://www.investopedia.com/terms/f/fractal.asp)
- [TradingPedia：Williams Fractals](https://www.tradingpedia.com/forex-trading-indicators/williams-fractals/)
