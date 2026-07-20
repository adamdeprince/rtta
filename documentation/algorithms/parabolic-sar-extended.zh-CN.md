# 扩展抛物线止损反转（ParabolicSARExtended）

## 摘要

`ParabolicSARExtended` 是 TA-Lib **SAREXT** 风格的抛物线止损反转指标，具有相互独立的多头/空头加速因子（AF）链、可选的固定起始 SAR，以及趋势反转时应用的偏移量。每次更新返回当前 SAR 价位。

## 更新 API

```python
import rtta

ind = rtta.ParabolicSARExtended(
    start=0.0,
    offset_on_reverse=0.0,
    af_init_long=0.02,
    af_long=0.02,
    af_max_long=0.2,
    af_init_short=0.02,
    af_short=0.02,
    af_max_short=0.2,
)
sar = ind.update(high, low)
```

若 `start != 0`，该值用于初始化 SAR；否则，前几根 K 线使用最高价/最低价极值。多头与空头各自具有独立的 AF 初始值、步长和最大值参数。

## 工作原理

Wilder 的抛物线 SAR 以一个止损位跟踪价格；随着趋势持续，该止损位会加速趋近极值价格。SAREXT 对经典的单一 AF 调度作了扩展，使多头和空头趋势可以按不同速度加速，并允许在切换方向时加入偏移量。RTTA 采用通常的 SAR 更新：

\[
\mathrm{SAR}_{t} = \mathrm{SAR}_{t-1} + \mathrm{AF}\,(\mathrm{EP} - \mathrm{SAR}_{t-1}),
\]

随后进行截断，使止损位不会穿入此前两根 K 线的区间；最后检查当前最低价（多头）或最高价（空头）是否穿过止损位，从而决定是否反转。

## 递推公式

**第 0 根 K 线：**保存 \(H_0,L_0\)；若 `start` 非零，则 \(\mathrm{SAR}=\mathrm{start}\)，否则 \(\mathrm{SAR}=L_0\)；\(\mathrm{EP}=H_0\)；\(\mathrm{AF}=\mathrm{af\_init\_long}\)；方向向上。

**第 1 根 K 线：**若 \(H_1\ge H_0\)，则方向向上；SAR 取 `start`，或取 \(\min(L_0,L_1)\) / \(\max(H_0,H_1)\)；EP 取相应最高价/最低价；AF 取对应方向的初始值。

**第 \(t\ge2\) 根 K 线：**

\[
\mathrm{SAR}' = \mathrm{SAR} + \mathrm{AF}\,(\mathrm{EP} - \mathrm{SAR}).
\]

若方向向上：

\[
\mathrm{SAR}' \leftarrow \min(\mathrm{SAR}',\, \min(L_{t-1}, L_t)).
\]

若 \(L_t<\mathrm{SAR}'\)（反转为空头）：

\[
\mathrm{SAR}' \leftarrow \mathrm{EP} + \mathrm{offset\_on\_reverse},\quad
\mathrm{EP}\leftarrow L_t,\quad
\mathrm{AF}\leftarrow \mathrm{af\_init\_short}.
\]

否则，若 \(H_t>\mathrm{EP}\)：\(\mathrm{EP}\leftarrow H_t\)，\(\mathrm{AF}\leftarrow\min(\mathrm{AF}+\mathrm{af\_long},\mathrm{af\_max\_long})\)。

若方向为空头：

\[
\mathrm{SAR}' \leftarrow \max(\mathrm{SAR}',\, \max(H_{t-1}, H_t)).
\]

若 \(H_t>\mathrm{SAR}'\)（反转为多头）：

\[
\mathrm{SAR}' \leftarrow \mathrm{EP} - \mathrm{offset\_on\_reverse},\quad
\mathrm{EP}\leftarrow H_t,\quad
\mathrm{AF}\leftarrow \mathrm{af\_init\_long}.
\]

否则，若 \(L_t<\mathrm{EP}\)：\(\mathrm{EP}\leftarrow L_t\)，\(\mathrm{AF}\leftarrow\min(\mathrm{AF}+\mathrm{af\_short},\mathrm{af\_max\_short})\)。

最后提交 \(\mathrm{SAR}\leftarrow\mathrm{SAR}'\)，并把当前最高价/最低价保存为此前状态。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class ParabolicSARExtended` 中实现。每根 K 线返回一个标量 `double` SAR。

## 参考资料

- [TA-Lib SAREXT 文档](https://ta-lib.org/functions/SAREXT/)
- [Wilder，《New Concepts in Technical Trading Systems》（Parabolic SAR）](https://www.amazon.com/New-Concepts-Technical-Trading-Systems/dp/0894590278)
