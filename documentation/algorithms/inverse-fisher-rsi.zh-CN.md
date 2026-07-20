# RSI 反 Fisher 变换（InverseFisherRSI）

## 摘要

`InverseFisherRSI` 是 RTTA 对 RSI 应用 Ehlers 反 Fisher 变换的流式实现。RSI 先映射到近似对称的数域，再经 WMA 平滑，最后通过反 Fisher 映射，得到通常位于 \((-1,1)\) 的敏锐振荡器。

## 更新 API

```python
value = rtta.InverseFisherRSI(rsi_window=5, wma_window=9, fillna=True).update(close)
```

当 `fillna=False` 时，在处理 `rsi_window + wma_window` 个样本之前，输出为 `NaN`。

## 工作原理

John Ehlers 的反 Fisher 变换会放大接近零的数值并压缩极值，使平滑振荡器呈现更清晰的转折点。用于 RSI 的常见步骤为：

1. 计算 RSI。
2. 缩放 RSI，使中点（50）映射到接近 0，极值映射到接近 \(\pm5\)。
3. 使用短周期加权移动平均进行平滑。
4. 对平滑值应用等价于 \(\tanh\) 的反 Fisher 变换。

RTTA 在第 1–3 步使用自己的流式 `RSI` 和 `WMA` 基础组件。

## 递推公式

令 \(c_t\) 为收盘价，\(n_r\) 为 `rsi_window`（默认 \(5\)），\(n_w\) 为 `wma_window`（默认 \(9\)）。

\[
RSI_t = \operatorname{RSI}_{n_r}(c_t)
\]

\[
x_t = 0.1\,(RSI_t - 50)
\]

\[
y_t = \operatorname{WMA}_{n_w}(x_t)
\]

\[
e_t = \exp(2 y_t), \qquad
IFR_t = \frac{e_t - 1}{e_t + 1}
\quad\text{（安全除法）}
\]

注意 \(\frac{e^{2y}-1}{e^{2y}+1}=\tanh(y)\)。内部 RSI/WMA 以 `fillna=True` 构造；外层预热计数为 \(n_r+n_w\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class InverseFisherRSI` 中实现。另请参阅 [`RSI`](rsi.zh-CN.md)。

## 参考资料

- [John F. Ehlers，《The Inverse Fisher Transform》，*Stocks & Commodities*](https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf)
- [ChartSchool：Relative Strength Index](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi)
