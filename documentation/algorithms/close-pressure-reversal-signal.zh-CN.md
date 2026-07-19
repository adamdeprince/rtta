# ClosePressureReversalSignal

## 摘要

`ClosePressureReversalSignal` 是临近收盘的横截面反转信号，综合当日剩余收益、成交量与成交笔数压力、VWAP 位置，以及滚动保形误差带。

## 更新 API

```python
result = rtta.ClosePressureReversalSignal().update(open, high, low, close, volume)
```

`update(...)` 每次接收一根 K 线的 `open`、`high`、`low`、`close` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

该类把尾盘反转思路转换为因果 K 线流：在设定截止时点冻结当日收益，按已实现日内波动率标准化输家/赢家压力，再结合成交量、成交笔数和 VWAP 位置进行调整，并且只在尾盘窗口内发出入场信号。下方链接的研究说明详述了经验依据和参数含义。

## 递推公式

令 \(z_t = (open_t, high_t, low_t, close_t, volume_t)\) 为一次更新接收的观测。

\[
ROD_t=\log(close_t)-\log(anchor), \qquad F_t=ROD_{t_c}
\]

\[
DV_t=close_t\max(volume_t,0), \qquad vwap\_gap_t=\frac{close_t}{VWAP_t}-1
\]

\[
\sigma_{intra,t}=\sqrt{N_{t_c}\operatorname{Var}(r^{(1)}_1,\ldots,r^{(1)}_{t_c})},
\qquad L_t=\frac{\max(0,-F_t)}{\sigma_{intra,t}}, \quad W_t=\frac{\max(0,F_t)}{\sigma_{intra,t}}
\]

\[
M^V_t=1+0.20\,\operatorname{clip}(\log(DV_t/NDV_t),-2,4), \qquad
M^X_t=1+0.10\,\operatorname{clip}(\log(X_t/NX_t),-2,4)
\]

\[
P^{long}_t=L_tM^V_tM^X_t\left(1+0.50\,\operatorname{clip}\left(\frac{-vwap\_gap_t}{\sigma_{intra,t}},0,3\right)\right)
\]

\[
\widehat{r}_t=slope\cdot \max(0,-F_t)\,\operatorname{clip}(P^{long}_t/2,0,2)
\]

\[
\mathcal{E}_t=\{|r^{entry\to exit}_i-\widehat{r}_i|:\ i \text{ 已到期}\}, \qquad radius_t=\max(Q_{\tau}(\mathcal{E}_t), cost)
\]

\[
score_t=\frac{\widehat{r}_t}{radius_t+cost}
\]

`update(...)` 返回包含 `bar_number`、`rod_return`、`frozen_rod_return`、`loser_z`、`winner_z`、`range_z`、`volume_shock`、`transaction_shock`、`vwap_gap`、`pressure_score`、`prediction`、`radius`、`score`、`signal`、`target_fraction`、`max_trade_dollars`、`realized_error`、`entry_window`、`exit_window`、`frozen` 和 `news_guard` 的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ClosePressureReversalSignal` 中实现。

## 参考资料

- [详细研究说明](../close_pressure_reversal_signal.zh-CN.md)
