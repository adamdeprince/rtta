# 几何移动平均（GeometricMovingAverage）

## 摘要

`GeometricMovingAverage` 是 RTTA 对近期价格几何平均数的流式实现：先对价格取对数并计算 SMA，再取指数。价格必须为严格正的有限值；无效样本返回 `NaN`，且不会推进可用的对数均值状态。

## 更新 API

```python
value = rtta.GeometricMovingAverage(window=14, fillna=True).update(price)
```

内部对数 SMA 会继承 `fillna`。`price` 为非正值或非有限值时，本次调用返回 `NaN`。

## 工作原理

正价格 \(x_1,\ldots,x_n\) 的几何平均数为：

\[
\Bigl(\prod_{i=1}^{n} x_i\Bigr)^{1/n} = \exp\!\Bigl(\tfrac1n\sum_{i=1}^{n}\log x_i\Bigr).
\]

它是乘法型（类似收益率）过程的自然平滑器：幅度相等的百分比涨跌对几何平均数产生对称影响，而算术 SMA 对绝对点数赋予相同权重。RTTA 实现恒等式 \(\operatorname{GMA}=\exp(\operatorname{SMA}(\log price))\)。

## 递推公式

令 \(x_t\) 为输入价格，\(n\) 为 `window`（默认 \(14\)）。

若 \(x_t\le0\) 或 \(x_t\) 不是有限值，则返回 `NaN`。否则：

\[
\ell_t = \log x_t, \qquad
L_t = \operatorname{SMA}_n(\ell_t)
\]

\[
GMA_t = \exp(L_t)
\]

若内部 SMA 返回 `NaN`（`fillna=False` 时的预热阶段），输出也是 `NaN`。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class GeometricMovingAverage` 中实现。成员 `logs_` 是一个以 `std::log(price)` 为输入的 `SMA`。

## 参考资料

- [Investopedia：Geometric Mean](https://www.investopedia.com/terms/g/geometricmean.asp)
- [Wikipedia：Geometric mean](https://en.wikipedia.org/wiki/Geometric_mean)
