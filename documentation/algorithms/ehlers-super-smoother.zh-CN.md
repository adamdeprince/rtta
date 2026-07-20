# Ehlers Super Smoother（EhlersSuperSmoother）

## 摘要

`EhlersSuperSmoother` 是 RTTA 对 John F. Ehlers 二极点 Super Smoother 低通滤波器的流式实现。与具有相近效果的指数移动平均相比，它能以显著更小的滞后衰减高频市场噪声，同时保持因果性，适合逐根 K 线流式使用。

## 更新 API

```python
result = rtta.EhlersSuperSmoother(period=10, fillna=True).update(price)
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `period`  | `10`    | 低通滤波器的临界周期 \(P\)（最小为 2） |
| `fillna`  | `True`  | 若为 `False`，取得 `period` 个样本前返回 NaN |

每次调用 `update(...)` 处理一个标量价格观测值，并返回当前滤波结果。该类不公开 `advance(...)`；批量序列请使用 `update` 或 `batch` / `batch_array`。

## 工作原理

Ehlers 将 Super Smoother 设计为二极点递归低通滤波器，其极点位于由临界周期 \(P\) 确定的圆周上。连续时间原型以 \(\sqrt2\,\pi/P\) 作为一对共轭复极点的角度；离散滤波器系数 \(c_1,c_2,c_3\) 由该角度的指数函数和余弦函数导出，并选择 \(c_1\) 使直流增益为 1。

进入递归部分之前，输入先取两样本平均 \((x_t+x_{t-1})/2\)，以减弱离散价格每根 K 线之间的阶梯跳变。前两个样本用原始价格初始化滤波器状态，从而无需较长的预热缓冲区也能为递推提供有限历史。

与具有相近噪声抑制能力的 EMA 相比，Super Smoother 在越过临界周期后滚降更快，能为后接的周期指标提供更干净的平滑结果（例如 Roofing 滤波器的低通级）。

## 递推公式

令 \(x_t\) 为输入价格，\(P=\max(\texttt{period},2)\) 为临界周期。在构造函数中预先计算：

\[
\theta = \frac{\sqrt{2}\,\pi}{P},\qquad
a_1 = e^{-\theta},\qquad
b_1 = 2 a_1 \cos(\theta)
\]

\[
c_2 = b_1,\qquad
c_3 = -a_1^{2},\qquad
c_1 = 1 - c_2 - c_3
\]

每根 K 线处理后的状态包括前一个价格 \(x_{t-1}\)，以及前两个滤波输出 \(y_{t-1}\)、\(y_{t-2}\)。对于前两个样本（\(t<2\)）：

\[
y_t = x_t
\]

此后：

\[
y_t = c_1 \cdot \frac{x_t + x_{t-1}}{2} + c_2\, y_{t-1} + c_3\, y_{t-2}
\]

当 `fillna=False` 且已处理的样本少于 \(P\) 个时，返回值为 NaN；否则返回 \(y_t\)。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class EhlersSuperSmoother` 中实现。
- 系数使用常数 \(\sqrt2\approx1.4142135623730951\) 和与 C++ 源码一致的 \(\pi\)。
- 状态变量：`price1_`、`filt1_`、`filt2_`、样本 `count_`。
- 输出为标量 `double`，不是结果结构体。

## 参考资料

- [Ehlers——Super Smoother（MESA Software 论文）](https://www.mesasoftware.com/papers/UsingTheFisherTransform.pdf)
- [Ehlers 滤波器概述（Rocket Science for Traders / 周期工具）](https://www.mesasoftware.com/)
