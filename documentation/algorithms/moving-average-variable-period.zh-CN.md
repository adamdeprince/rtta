# 可变周期移动平均（MovingAverageVariablePeriod）

## 摘要

`MovingAverageVariablePeriod` 是 RTTA 对 TA-Lib 风格 MAVP 的流式实现：`value` 的简单移动平均线，其回看长度由每根 K 线单独提供，并截断到 `[min_period, max_period]`。

## 更新 API

```python
value = rtta.MovingAverageVariablePeriod(
    max_period=30, min_period=2, fillna=True
).update(value, period)
```

`period` 通过 `llround` 舍入到最近整数，再截断。当 `fillna=False` 时，若现有样本少于截断后的周期，输出为 `NaN`。

## 工作原理

可变周期平均线允许另一个序列——效率比率、波动率、周期长度等——在每根 K 线上控制平滑器的记忆长度。RTTA 保留长度为 `max_period` 的滚动缓冲区，每次更新只对最近 \(p\) 个样本求平均，其中 \(p\) 是截断后的请求周期。这符合 TA-Lib MAVP 的常见解读：长度动态变化的普通 SMA。

## 递推公式

令 \(x_t\) 为 `value`，\(p^{\text{raw}}_t\) 为 `period`，\(p_{\min}\) 为 `min_period`，\(p_{\max}\) 为 `max_period`。

\[
p_t = \operatorname{clamp}\!\bigl(\operatorname{round}(p^{\text{raw}}_t),\, p_{\min},\, p_{\max}\bigr)
\]

维护一个最多保存最近 \(p_{\max}\) 个值的 FIFO 缓冲区。令 \(n_t\) 为当前缓冲区大小，\(u_t=\min(n_t,p_t)\)。对最近 \(u_t\) 个样本求和：

\[
MAVP_t = \frac{1}{u_t}\sum_{i=0}^{u_t-1} x_{t-i}
\]

若 `fillna=False` 且 \(n_t<p_t\)，返回 `NaN`。默认值：\(p_{\max}=30\)、\(p_{\min}=2\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class MovingAverageVariablePeriod` 中实现。缓冲区容量始终为 `max_period`。

## 参考资料

- [TA-Lib：MAVP——Moving average with variable period](https://ta-lib.org/functions/MAVP/)
- [Investopedia：Moving Average](https://www.investopedia.com/terms/m/movingaverage.asp)
