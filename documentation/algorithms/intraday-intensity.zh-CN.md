# 日内强度（IntradayIntensity）

## 摘要

`IntradayIntensity` 是 RTTA 对成交量加权日内强度的流式窗口平均。每根 K 线贡献强度流 \(((2C-H-L)/(H-L))\cdot V\)；指标返回这些流的滚动和与成交量滚动和之比。

## 更新 API

```python
value = rtta.IntradayIntensity(window=21, fillna=True).update(high, low, close, volume)
```

当 `fillna=False` 时，在窗口缓冲区填满之前输出为 `NaN`。

## 工作原理

David Bostian 的日内强度（也与累积/派发及资金流系列相关）确定收盘价在 K 线最高价—最低价区间中的位置，再按成交量为该位置加权。收盘接近最高价会产生正强度，接近最低价则产生负强度。在滚动窗口内用成交量对订单流取平均，得到大致位于 \([-1,1]\) 的标准化参与程度。

## 递推公式

令 \(H_t,L_t,C_t,V_t\) 为最高价、最低价、收盘价和成交量，\(n\) 为 `window`（默认 \(21\)）。

\[
II^{\text{raw}}_t =
\begin{cases}
0, & H_t = L_t \\
\dfrac{2C_t - H_t - L_t}{H_t - L_t}\, V_t, & \text{其他情况}
\end{cases}
\]

在最近 \(\min(t,n)\) 个样本上维护滚动和（固定容量缓冲区，填满后按 FIFO 淘汰）：

\[
F_t = \sum_{i \in W_t} II^{\text{raw}}_i, \qquad
U_t = \sum_{i \in W_t} V_i
\]

\[
\operatorname{IntradayIntensity}_t = \frac{F_t}{U_t}
\quad\text{（安全除法）}
\]

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class IntradayIntensity` 中实现，使用两个滚动求和缓冲区（`flow_`、`vol_`）。

## 参考资料

- [Investopedia：Intraday Intensity Index](https://www.investopedia.com/terms/i/intradayintensityindex.asp)
- [ChartSchool：Accumulation Distribution Line](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/accumulation-distribution-line)
