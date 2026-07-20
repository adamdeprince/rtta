# MACDFix

## 摘要

`MACDFix` 是 RTTA 的固定周期 MACD 封装：快速/慢速 EMA 周期锁定为经典的 12/26，信号周期仍可配置。它是 [`MACD`](macd.zh-CN.md) 上的一层轻量便利接口。

## 更新 API

```python
result = rtta.MACDFix(signal=9, fillna=False).update(close)
```

| 参数 | 默认值 | 含义 |
|-----------|----------|---------|
| `signal`  | `9`      | 信号 EMA 长度 (c) |
| `fillna`  | `False`  | 若为 `False`，预热结束前返回 NaN |

`update(close)` 通过内部 `MACD` 结果结构体返回 `macd`、`signal` 和 `histogram`。`advance(close)` 与 `last()` 均委托给内部 MACD。

## 工作原理

许多图表 API 提供“MACD Fix”，即快速/慢速均线固定为 12/26、只有信号周期可调的 MACD。RTTA 通过构造以下对象实现：

```text
MACD(a=12, b=26, c=signal, fillna=fillna)
```

并把每次更新转发给它。在数值与语义上，本指标等同于直接调用 `MACD(12, 26, signal, fillna)`。

## 递推公式

固定 \(a=12\)、\(b=26\)，并令可配置的信号长度为 (c)：

\[
F_t = \operatorname{EMA}_{12}(x_t),\qquad
S_t = \operatorname{EMA}_{26}(x_t)
\]

\[
MACD_t = F_t - S_t
\]

\[
signal_t = \operatorname{EMA}_{c}(MACD_t)
\]

\[
hist_t = MACD_t - signal_t
\]

当 `fillna=False` 时，预热方式与 `MACD` 相同：取得 \(\max(12,26)+c=26+c\) 个样本之前返回 NaN。内部 EMA 使用 `fillna=True`，因此状态始终有定义。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class MACDFix` 中实现，成员为 `MACD macd_(12, 26, signal, fillna)`。
- 批量辅助函数：`batch_macd_fix`。
- 另请参阅 [`MACD`](macd.zh-CN.md) 与 [`MACDExt`](macd-ext.zh-CN.md)。

## 参考资料

- [StockCharts——MACD](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator)
- [TA-Lib MACDFIX](https://ta-lib.org/functions/macdfix)
