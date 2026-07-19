# MarketOpenCloseTransitionDetector

## 摘要

`MarketOpenCloseTransitionDetector` 根据交易时段进度检测开盘区间和收盘区间之间的转换。

## 更新 API

```python
result = rtta.MarketOpenCloseTransitionDetector().update(session_progress)
```

`update(...)` 每次接收一个 `session_progress`；只推进状态时可调用 `advance(...)`。

## 工作原理

输入先被限制在 0 到 1，表示交易时段的完成比例。开盘和收盘各有独立的进入/退出阈值，回滞避免状态在边界附近反复切换。

## 递推公式

\[
p_t=\operatorname{clip}(session\_progress_t,0,1)
\]

\[
r_t=\begin{cases}1,&r_{t-1}=0\text{ 且 }p_t\le open_e\\0,&r_{t-1}=1\text{ 且 }p_t\ge open_x\\-1,&r_{t-1}=0\text{ 且 }p_t\ge close_e\\0,&r_{t-1}=-1\text{ 且 }p_t\le close_x\\r_{t-1},&\text{否则}\end{cases}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class MarketOpenCloseTransitionDetector` 中实现。

## 参考资料

- [背景资料：交易日](https://en.wikipedia.org/wiki/Trading_day)
