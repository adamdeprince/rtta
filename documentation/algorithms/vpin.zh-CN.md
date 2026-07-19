# VPIN

## 摘要

`VPIN` 采用批量成交量分类和滚动成交量桶失衡，估计成交量同步的知情交易概率。

## 更新 API

```python
result = rtta.VPIN().update(close, volume)
```

`update(...)` 每次接收 `close` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标根据价格变化把成交量划分到买卖两侧，按固定成交量桶累计失衡，再在滚动桶窗口内归一化。

## 递推公式

\[
PV_t=PV_{t-1}+price_t\,volume_t,\qquad V_t=V_{t-1}+volume_t,\qquad y_t=G(PV_t,V_t,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class VPIN` 中实现。

## 参考资料

- [VPIN 论文](https://www.quantresearch.org/VPIN.pdf)
