# ADWIN

## 摘要

`ADWIN` 是一种自适应窗口均值漂移检测器；它限制历史长度，并输出漂移方向。

## 更新 API

```python
result = rtta.ADWIN().update(value)
```

`update(...)` 每次接收一个 `value`。如果只需推进状态，可用相同输入调用 `advance(...)`。

## 工作原理

`ADWIN` 维护一个自适应的近期窗口，并检查每个允许的切分点，判断新旧子窗口的均值差异是否具有统计意义。信号方向取最显著且通过检验的均值变化方向；接受切分后会丢弃较早的窗口前缀。

## 递推公式

令 \(z_t = value_t\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
W_t=\operatorname{tail}_{max\_window}(W_{t-1}\cup\{x_t\})
\]

\[
\epsilon(c)=R_t
\sqrt{\frac{1}{2}\log\left(\frac{4}{\delta}\right)
\left(\frac{1}{c}+\frac{1}{|W_t|-c}\right)}
\]

\[
c^\*=\arg\max_c |\bar{x}_{c:|W_t|}-\bar{x}_{1:c}|
\quad \text{s.t.}\quad
|\bar{x}_{c:|W_t|}-\bar{x}_{1:c}|>\epsilon(c)
\]

\[
y_t =
\begin{cases}
\operatorname{sgn}(\bar{x}_{c^\*:|W_t|}-\bar{x}_{1:c^\*}), & c^\* \text{ 存在}\\
0, & \text{否则}
\end{cases}
\]

接受切分后，较早的前缀会被丢弃，保留的后缀成为下一个自适应窗口。返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ADWIN` 中实现。

## 参考资料

- [背景资料：概念漂移](https://en.wikipedia.org/wiki/Concept_drift)
