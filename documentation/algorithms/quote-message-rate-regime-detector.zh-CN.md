# QuoteMessageRateRegimeDetector

## 摘要

`QuoteMessageRateRegimeDetector` 以相对 EWMA 基线检测报价消息速率状态。

## 更新 API

```python
result = rtta.QuoteMessageRateRegimeDetector().update(quote_messages)
```

`update(...)` 每次接收 `quote_messages`；只推进状态时可调用 `advance(...)`。

## 工作原理

当前报价消息数先除以上一期 EWMA 基线，再用双向回滞生成状态；评估比率后才把当前观测纳入基线，避免前视偏差。

## 递推公式

\[
b_t=\alpha\max(x_t,0)+(1-\alpha)b_{t-1},\qquad q_t=\frac{\max(x_t,0)}{\max(b_{t-1},\epsilon)}
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class QuoteMessageRateRegimeDetector` 中实现。

## 参考资料

- [背景资料：报价填塞](https://en.wikipedia.org/wiki/Quote_stuffing)
