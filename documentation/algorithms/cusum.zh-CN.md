# CUSUM

## 摘要

`CUSUM` 是因果累积和事件滤波器，用于检测超过阈值的方向性变动。

## 更新 API

```python
result = rtta.CUSUM(threshold=1.0, drift=0.0).update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

每次更新只使用新收盘价和先前状态，分别累计向上与向下偏移；任一累计值越过阈值时发出方向信号。

## 递推公式

\[
\Delta_t=close_t-close_{t-1}
\]

\[
S^+_t=\max(0,S^+_{t-1}+\Delta_t-\kappa), \qquad S^-_t=\min(0,S^-_{t-1}+\Delta_t+\kappa)
\]

\[
y_t=\begin{cases}1,&S^+_t>h\\-1,&S^-_t<-h\\0,&\text{否则}\end{cases}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class CUSUM` 中实现。

## 参考资料

- [背景资料：CUSUM](https://en.wikipedia.org/wiki/CUSUM)
