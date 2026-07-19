# TradeIntensityRegimeDetector

## 摘要

`TradeIntensityRegimeDetector` 以相对 EWMA 基线检测成交笔数强度状态。

## 更新 API

```python
result = rtta.TradeIntensityRegimeDetector().update(transactions)
```

`update(...)` 每次接收 `transactions`；只推进状态时可调用 `advance(...)`。

## 工作原理

当前非负成交笔数先除以上一期 EWMA 基线，再通过双向回滞生成状态；计算比率后才更新基线。

## 递推公式

\[
b_t=\alpha\max(x_t,0)+(1-\alpha)b_{t-1},\qquad q_t=\frac{\max(x_t,0)}{\max(b_{t-1},\epsilon)}
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class TradeIntensityRegimeDetector` 中实现。

## 参考资料

- [背景资料：市场微观结构](https://en.wikipedia.org/wiki/Market_microstructure)
