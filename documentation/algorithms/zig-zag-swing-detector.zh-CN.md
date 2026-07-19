# ZigZagSwingDetector

## 摘要

`ZigZagSwingDetector` 根据收盘价识别摆动，过滤小于百分比阈值的变动，并输出已确认拐点。

## 更新 API

```python
result = rtta.ZigZagSwingDetector().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器维护当前摆动方向、活动极值和最后确认的拐点。只有价格从活动极值反向移动达到设定百分比后，才确认新拐点，从而过滤较小波动。

## 递推公式

\[
\tau=\frac{percent\_change}{100}
\]

\[
direction_t=\begin{cases}1,&direction_{t-1}=0\text{ 且 }close_t\ge start(1+\tau)\\-1,&direction_{t-1}=0\text{ 且 }close_t\le start(1-\tau)\\-1,&direction_{t-1}=1\text{ 且 }close_t\le extreme_{t-1}(1-\tau)\\1,&direction_{t-1}=-1\text{ 且 }close_t\ge extreme_{t-1}(1+\tau)\\direction_{t-1},&\text{否则}\end{cases}
\]

\[
extreme_t=\begin{cases}\max(extreme_{t-1},close_t),&direction_t=1\\\min(extreme_{t-1},close_t),&direction_t=-1\\close_t\text{（若其离 }start\text{ 更远）},&direction_t=0\end{cases}
\]

方向翻转时，此前极值成为已确认拐点，当前收盘价则作为新一轮极值搜索的起点。

`update(...)` 返回含 `value`、`direction`、`pivot` 和 `pivot_index` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ZigZagSwingDetector` 中实现。

## 参考资料

- [ChartSchool：ZigZag](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/zigzag)
