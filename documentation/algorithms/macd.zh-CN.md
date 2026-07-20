# MACD

## 摘要

`MACD` 是 RTTA 对移动平均收敛/发散振荡器的多输出流式实现：快速 EMA 减慢速 EMA、该差值的信号 EMA，以及 MACD 减信号线所得柱状图。

## 更新 API

```python
result = rtta.MACD(a=12, b=26, c=9, fillna=False).update(value)
```

| 参数 | 默认值 | 含义 |
|-----------|----------|---------|
| `a`       | `12`     | 快速 EMA 周期 |
| `b`       | `26`     | 慢速 EMA 周期 |
| `c`       | `9`      | 信号 EMA 周期 |
| `fillna`  | `False`  | 若为 `False`，预热结束前返回 NaN |

`update(value)` 返回的结果包含：

- `macd`——快速 EMA − 慢速 EMA
- `signal`——`macd` 的 EMA
- `histogram`——`macd` − `signal`

`advance(value)` 更新状态；`last()` 返回缓存的结果。

相关 API：[`MACDFix`](macd-fix.zh-CN.md)（固定 12/26 周期）、[`MACDExt`](macd-ext.zh-CN.md)（可选择 SMA/EMA 类型）。

## 工作原理

MACD 衡量价格短期指数平均与长期指数平均之间的距离。快速 EMA 高于慢速 EMA 时，中期动量为正；反之则为负。信号线是进一步平滑 MACD 的 EMA，因此：

- **MACD / 信号线交叉**常用作进场触发条件。
- **柱状图**直观显示二者距离及其扩张/收缩。
- MACD 的**零轴交叉**标记快速/慢速 EMA 上下顺序的变化。

三个内部 EMA 均以 `fillna=True` 运行，因此系数和种子会在每根 K 线上推进；外层 `fillna` 标志只控制预热不完整的样本是否以 NaN 返回。

## 递推公式

令 \(x_t\) 为输入序列，\(a,b,c\) 为三个周期。

\[
F_t = \operatorname{EMA}_a(x_t),\qquad
S_t = \operatorname{EMA}_b(x_t)
\]

\[
MACD_t = F_t - S_t
\]

\[
signal_t = \operatorname{EMA}_c(MACD_t)
\]

\[
hist_t = MACD_t - signal_t
\]

当 `fillna=False` 时，只要样本计数仍小于 \(\max(a,b)+c\)，三个字段就全部为 NaN；否则返回当前 MACD 三元结果。

EMA 的初始化与 \(\alpha=2/(n+1)\) 遵循 RTTA 的 [`EMA`](ema.zh-CN.md)。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class MACD` 中实现。
- 成员：`EMA a_`、`EMA b_`、`EMA c_`；结果类型为 `MACDResult`。
- `fillna` 默认值为 `False`（RTTA 的许多其他指标默认为 True）。
- MACD 系列提供批量序列处理辅助函数。

## 参考资料

- [StockCharts——MACD](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator)
- [Investopedia——MACD](https://www.investopedia.com/terms/m/macd.asp)
