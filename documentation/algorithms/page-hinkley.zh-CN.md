# PageHinkley

## 摘要

`PageHinkley` 是因果均值位移事件检测器，输出向上或向下的变化方向。

## 更新 API

```python
result = rtta.PageHinkley(threshold=1.0, delta=0.0).update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器围绕在线均值累计正负偏离，并扣除很小的漂移容忍量。当任一累计路径相对自身历史最小值上升超过阈值时发出信号，随后以当前收盘价重置。

## 递推公式

\[
\mu_t=\mu_{t-1}+\frac{close_t-\mu_{t-1}}t
\]

\[
P_t=P_{t-1}+close_t-\mu_t-\delta,\qquad N_t=N_{t-1}+\mu_t-close_t-\delta
\]

\[
S^+_t=P_t-\min_{i\le t}P_i,\qquad S^-_t=N_t-\min_{i\le t}N_i
\]

\[
y_t=\begin{cases}1,&S^+_t>h\text{ 且 }S^+_t\ge S^-_t\\-1,&S^-_t>h\\0,&\text{否则}\end{cases}
\]

发出非零信号后，C++ 实现把运行均值和累计值重置到当前收盘价。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class PageHinkley` 中实现。

## 参考资料

- [Page-Hinkley 背景资料](https://menelaus.readthedocs.io/en/dev/menelaus.change_detection.html)
