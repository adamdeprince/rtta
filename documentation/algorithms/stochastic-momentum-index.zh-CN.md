# 随机动量指数（StochasticMomentumIndex）

## 摘要

`StochasticMomentumIndex` 是 RTTA 对带信号线的双重平滑随机动量指数的流式实现。

## 更新 API

```python
result = rtta.StochasticMomentumIndex().update(close, high, low)
```

每次调用 `update(...)` 都使用 `close`、`high` 和 `low` 处理一个观测值。如果调用方只想更新状态而不生成 Python 返回值，可以用相同输入调用 `advance(...)`。

## 工作原理

`StochasticMomentumIndex` 衡量收盘价相对于近期最高价—最低价区间中点的位置，随后对距离和区间分别进行双重平滑，再作标准化。指标还会生成一条信号 EMA。

## 递推公式

令 \(c_t,h_t,l_t\) 为收盘价、最高价和最低价，\(n\) 为区间窗口，\(s_1,s_2,s\) 为平滑周期。

\[
HH_t=\max_{0\le i<n}h_{t-i},\quad
LL_t=\min_{0\le i<n}l_{t-i},\quad
M_t=\frac{HH_t+LL_t}{2}
\]

\[
D_t=c_t-M_t,\quad
R_t=HH_t-LL_t
\]

\[
SM_t=\operatorname{EMA}_{s_2}(\operatorname{EMA}_{s_1}(D_t)),\quad
SR_t=\operatorname{EMA}_{s_2}(\operatorname{EMA}_{s_1}(R_t))
\]

\[
SMI_t=100\frac{SM_t}{0.5\,SR_t},\quad
signal_t=\operatorname{EMA}_s(SMI_t)
\]

`update(...)` 返回一个结果结构体，字段为 `smi` 和 `signal`。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class StochasticMomentumIndex` 中实现。

## 参考资料

- [背景资料](https://www.investopedia.com/terms/s/stochmomentum.asp)
