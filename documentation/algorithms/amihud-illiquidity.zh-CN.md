# AmihudIlliquidity

## 摘要

`AmihudIlliquidity` 计算单位成交金额所对应绝对收益率的滚动平均值。

## 更新 API

```python
result = rtta.AmihudIlliquidity().update(close, volume)
```

`update(...)` 每次接收 `close` 和 `volume`。如果只需推进状态，可用相同输入调用 `advance(...)`。

## 工作原理

该指标以因果方式更新紧凑的滚动状态，并返回当前的非流动性估计。

## 递推公式

令 \(z_t = (close_t, volume_t)\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
PV_t = PV_{t-1}+price_t\,volume_t
\]

\[
V_t = V_{t-1}+volume_t, \qquad y_t = G(PV_t,V_t,z_t)
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class AmihudIlliquidity` 中实现。

## 参考资料

- [Amihud 非流动性估计量讲义](https://ba-odegaard.no/teach/notes/liquidity_estimators/amihud_estimator/amihud_lectures.pdf)
