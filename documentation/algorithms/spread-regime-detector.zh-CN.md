# SpreadRegimeDetector

## 摘要

`SpreadRegimeDetector` 根据相对买卖价差检测有状态的报价价差区间。

## 更新 API

```python
result = rtta.SpreadRegimeDetector().update(bid_price, ask_price)
```

`update(...)` 每次接收买价和卖价；只推进状态时可调用 `advance(...)`。

## 工作原理

非负报价价差先除以报价中点的绝对值，再通过双向进入/退出回滞生成稳定状态。

## 递推公式

\[
mid_t=\frac{bid_t+ask_t}{2},\qquad q_t=\frac{\max(ask_t-bid_t,0)}{\max(|mid_t|,\epsilon)}
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class SpreadRegimeDetector` 中实现。

## 参考资料

- [背景资料](https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf)
