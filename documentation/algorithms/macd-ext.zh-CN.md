# 扩展 MACD（MACDExt）

## 摘要

`MACDExt` 是 RTTA 的扩展 MACD：结构与经典 MACD 相同，仍由 macd / signal / histogram 组成，但快线、慢线和信号线各级都可以独立选择 SMA 或 EMA 平滑器（TA-Lib `MACDEXT` 风格）。

## 更新 API

```python
result = rtta.MACDExt(
    fast=12, slow=26, signal=9,
    fast_ma_type=1, slow_ma_type=1, signal_ma_type=1,
    fillna=True,
).update(value)
```

| 参数 | 默认值 | 含义 |
|-------------------|---------|---------|
| `fast`            | `12`    | 快速 MA 长度 |
| `slow`            | `26`    | 慢速 MA 长度 |
| `signal`          | `9`     | 信号 MA 长度 |
| `fast_ma_type`    | `1`     | `0` = SMA，`1` = EMA |
| `slow_ma_type`    | `1`     | `0` = SMA，`1` = EMA |
| `signal_ma_type`  | `1`     | `0` = SMA，`1` = EMA |
| `fillna`          | `True`  | 若为 `False`，预热结束前返回 NaN |

`update(value)` 返回 `macd`、`signal`、`histogram`（字段与 [`MACD`](macd.zh-CN.md) 相同）。`advance(value)` 更新状态；`last()` 返回缓存值。

## 工作原理

经典 MACD 全程使用 EMA。有些平台和 TA-Lib 允许混合使用 SMA 与 EMA，例如在研究中使用 SMA 信号线或全 SMA 的 MACD。`MACDExt` 通过小型 `SelectableMA` 辅助类实现这种灵活性（`0→SMA`，其他值→EMA）。

解读方式不变：MACD 是快线减慢线的距离，信号线是该距离的移动平均，柱状图是 MACD 减信号线。零轴交叉与信号线交叉仍具有通常的动量含义。

## 递推公式

令 \(x_t\) 为输入。若 `type=0`，则 \(\operatorname{MA}^{(type)}_n\) 为 SMA；若 `type=1`，则为 EMA。

\[
F_t = \operatorname{MA}^{(f)}_{n_f}(x_t),\qquad
S_t = \operatorname{MA}^{(s)}_{n_s}(x_t)
\]

\[
MACD_t = F_t - S_t
\]

\[
signal_t = \operatorname{MA}^{(g)}_{n_g}(MACD_t)
\]

\[
hist_t = MACD_t - signal_t
\]

内部 MA 始终以 `fillna=True` 运行，因此中间值始终存在。当外层 `fillna=False` 时，在处理 \(\max(n_f,n_s)+n_g\) 个样本之前，三个输出均为 NaN。

所有 `ma_type=1` 且使用默认长度时，除预热 / `fillna` 约定外，结果与基于 EMA 的 MACD 一致。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class MACDExt` 和 `class SelectableMA` 中实现。
- 结果类型：共用的 `MACDResult`（`macd`、`signal`、`histogram`）。
- 批量辅助函数：`batch_macd_ext`。
- 此版本只能选择 SMA 与 EMA（不包括完整 TA-Lib MA 类型枚举中的 DEMA/TEMA 等）。

## 参考资料

- [TA-Lib MACDEXT](https://ta-lib.org/functions/macdext)
- [StockCharts——MACD](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator)
