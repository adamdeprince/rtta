# KSWIN

## 摘要

`KSWIN` 使用 Kolmogorov-Smirnov 检验检测滑动窗口内的分布漂移。

## 更新 API

```python
result = rtta.KSWIN().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器以 KS 上确界距离比较近期子窗口与较早的参考部分。统计量超过临界值时，以两个子窗口均值之差的符号作为漂移方向。

## 递推公式

\[
A_t=W_t[1:|W_t|-m],\qquad B_t=W_t[|W_t|-m+1:|W_t|]
\]

\[
D_t=\sup_x|\widehat F_{A_t}(x)-\widehat F_{B_t}(x)|
\]

\[
c_\alpha=\sqrt{-\frac12\log(\alpha/2)\left(\frac1{|A_t|}+\frac1{|B_t|}\right)}
\]

\[
y_t=\begin{cases}\operatorname{sgn}(\bar B_t-\bar A_t),&D_t>c_\alpha\\0,&\text{否则}\end{cases}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class KSWIN` 中实现。

## 参考资料

- [背景资料：Kolmogorov-Smirnov 检验](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
