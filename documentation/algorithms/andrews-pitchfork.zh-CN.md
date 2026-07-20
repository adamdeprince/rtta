# AndrewsPitchfork

## 摘要

`AndrewsPitchfork` 是 RTTA 基于百分比 ZigZag 式枢轴点构建的流式安德鲁斯
叉线。确认三个交替枢轴点后，每根 K 线都会输出在当前索引处求值的中线及其平行
上叉线、下叉线，并同时输出枢轴与方向标志。

## 更新 API

```python
result = rtta.AndrewsPitchfork(percent_change=0.05, fillna=True).update(high, low, close)
# result.median, result.upper, result.lower, result.pivot, result.direction
```

`percent_change` 是小数形式的阈值（例如 `0.05` 表示 5%）。大于 `1.0` 的值
会被视为百分数并除以 100。第一根 K 线用于初始化状态；当 `fillna=True` 时，
早期输出用收盘价暂代三条线的值。

## 工作原理

Alan Andrews 的叉线从原点枢轴 \(P_0\) 出发，穿过后续两个枢轴
\(P_1,P_2\) 的中点绘制中线；再按照 \(P_1\) 与该中线之间的距离设置两条
平行叉线。价格经常会在这些叉线附近产生反应。

RTTA 不要求手工输入枢轴坐标，而是通过基于最高价与最低价极值的百分比变化
ZigZag 在线发现枢轴：价格从摆动高点向下反转达到阈值时确认高点枢轴；价格从
摆动低点向上反转达到阈值时确认低点枢轴。最近三个已确认枢轴定义叉线，此后的
K 线根据中线斜率外推各条线。

## 递推公式

令 \(H_t,L_t,C_t\) 分别为最高价、最低价和收盘价，\(\tau\) 为百分比阈值。
维护摆动方向 \(d_t\in\{-1,0,+1\}\)、极值价格及其索引，以及一个最多容纳八个
枢轴 \((p_i,j_i)\)（价格与 K 线索引）的环形缓冲区。

**枢轴确认（ZigZag）。** 从第一根收盘价开始，当
\(H_t\ge C_0(1+\tau)\) 时将方向设为 \(+1\)（上升摆动）；当
\(L_t\le C_0(1-\tau)\) 时设为 \(-1\)。上升摆动期间持续跟踪最高价；当
\(L_t\le E(1-\tau)\) 时确认高点枢轴。下降摆动期间持续跟踪最低价；当
\(H_t\ge E(1+\tau)\) 时确认低点枢轴。每次确认都会压入
\((E,j_E,\pm1)\)，并翻转方向。结果中的 `pivot` 在确认枢轴的 K 线上为
\(1\)，其他时候为 \(0\)。

**叉线几何。** 至少已有三个枢轴时，取最近三个枢轴作为
\(P_0,P_1,P_2\)（按这三个点内部从旧到新的顺序）：

\[
M^{\text{price}} = \tfrac12(p_1 + p_2), \qquad
M^{\text{idx}} = \tfrac12(j_1 + j_2)
\]

\[
s = \frac{M^{\text{price}} - p_0}{M^{\text{idx}} - j_0}
\quad\text{（斜率；分母为零时取 \(0\)）}
\]

在当前 K 线索引 \(t\) 处：

\[
\begin{aligned}
\operatorname{median}_t &= p_0 + s\,(t - j_0) \\
\delta &= p_1 - \bigl(p_0 + s\,(j_1 - j_0)\bigr) \\
\operatorname{upper}_t &= \operatorname{median}_t + \delta \\
\operatorname{lower}_t &= \operatorname{median}_t - \delta
\end{aligned}
\]

`direction` 是当前摆动方向 \(d_t\)。在取得三个枢轴之前，`fillna=True`
时三条线均返回收盘价，`fillna=False` 时则返回 `NaN`。

## 实现说明

递推过程实现在 `src/rtta/indicator.cpp` 的 `class AndrewsPitchfork` 中。
结果字段为 `median`、`upper`、`lower`、`pivot` 和 `direction`。枢轴缓冲区
最多保存八个点，超出后删除最旧的点。

## 参考资料

- [ChartSchool：Andrews' Pitchfork](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns/andrews-pitchfork)
- [Investopedia：Andrews' Pitchfork](https://www.investopedia.com/terms/a/andrewspitchfork.asp)
