# CDLMorningStar

## 摘要

`CDLMorningStar` 是 RTTA 对**启明星（Morning Star）** K 线形态的流式
检测器。这是一种三 K 线看涨反转：先是长阴线，再是小实体，最后是一根收盘高于
第一根实体中点的长阳线。

输出遵循 TA-Lib 约定：看涨匹配为 **`+100`**，看跌匹配为 **`-100`**，
形态未触发时为 **`0`**。

## 更新 API

```python
value = rtta.CDLMorningStar(fillna=True).update(open, high, low, close)
# 看涨匹配为 +100；有方向的看跌匹配为 -100；未匹配为 0
batch = rtta.CDLMorningStar(fillna=True).batch(open, high, low, close)
```

`update(...)` 每次接收一根 OHLC K 线。`advance(...)` 使用相同输入但不返回
Python 值。在新实例上调用标量 `batch(open, high, low, close)`，结果与依次
调用 `update` 一致。当 `fillna=False` 时，在至少取得 3 根 K 线前返回 NaN；
当 `fillna=True`（默认）时，未匹配的 K 线返回 `0`。

## 工作原理

K 线形态是根据柱体几何结构产生的短周期因果标签。它们本身并不预测收益，而是
标记犹豫、拒绝、吞没压力、多 K 线反转等结构，可与趋势、波动率或成交量背景
结合使用。

引擎在每根 K 线上计算：

\[
\begin{aligned}
\mathrm{body}_t &= |C_t - O_t| \\
\mathrm{range}_t &= H_t - L_t \\
\mathrm{upper}_t &= H_t - \max(O_t, C_t) \\
\mathrm{lower}_t &= \min(O_t, C_t) - L_t \\
\mathrm{top}_t &= \max(O_t, C_t),\quad
\mathrm{bot}_t = \min(O_t, C_t) \\
\mathrm{mid}_t &= \tfrac12(O_t + C_t)
\end{aligned}
\]

其中 `body` 为实体长度，`range` 为整根 K 线的全幅，`upper` 和 `lower`
分别为上影线和下影线。当 \(C_t\ge O_t\) 时为**阳线**，当 \(C_t<O_t\) 时为
**阴线**。K 线保存在一个短环形缓冲区中；年龄 \(0\) 表示最新 K 线，年龄
\(1\) 表示前一根，以此类推。

下文使用以下共享谓词：

\[
\begin{aligned}
\mathrm{doji}(b) &\iff \mathrm{body} \le 0.1\cdot \mathrm{range}
  \quad(\text{或 range}=0) \\
\mathrm{longBody}(b,\overline B,f) &\iff \mathrm{body} \ge f\cdot \overline B \\
\mathrm{shortBody}(b,\overline B,f) &\iff \mathrm{body} \le f\cdot \overline B \\
\mathrm{longLower}(b,m) &\iff \mathrm{lower} \ge m\cdot \mathrm{body} \\
\mathrm{longUpper}(b,m) &\iff \mathrm{upper} \ge m\cdot \mathrm{body} \\
\mathrm{smallUpper}(b,f) &\iff \mathrm{upper} \le f\cdot \mathrm{range} \\
\mathrm{smallLower}(b,f) &\iff \mathrm{lower} \le f\cdot \mathrm{range} \\
\mathrm{engulf}(c,p) &\iff
  \mathrm{top}_c \ge \mathrm{top}_p \land
  \mathrm{bot}_c \le \mathrm{bot}_p \land
  \mathrm{body}_c > \mathrm{body}_p \\
\mathrm{inside}(c,p) &\iff
  \mathrm{top}_c \le \mathrm{top}_p \land
  \mathrm{bot}_c \ge \mathrm{bot}_p
\end{aligned}
\]

\(\overline B\) 是环形缓冲区内最近最多五根 K 线的平均实体长度；
\(\overline R\) 是同一窗口的平均全幅。

对于锤子线家族的形态，轻量级**先行趋势**判断会比较前收盘价与此前两个收盘价
的 SMA：

\[
\tau =
\begin{cases}
+1 & C_{-1} > 1.0001\cdot \mathrm{SMA}(C_{-1},C_{-2}) \\
-1 & C_{-1} < 0.9999\cdot \mathrm{SMA}(C_{-1},C_{-2}) \\
0 & \text{其他情况}
\end{cases}
\]

## 递推公式

将 \((O_t,H_t,L_t,C_t)\) 压入环形缓冲区。只有其中至少已有 **3** 根 K 线
时才进行检测。

令 \(a=\mathrm{at}(2)\)、\(b=\mathrm{at}(1)\)、
\(c=\mathrm{at}(0)\)，\(\overline B\) 为平均实体长度。

满足下式时输出 \(+100\)：

\[
a\ \text{为阴线}\land \mathrm{longBody}(a,\overline B,0.8)
\land \mathrm{shortBody}(b,\overline B,0.6)
\land c\ \text{为阳线}\land \mathrm{longBody}(c,\overline B,0.7)
\land C_c > \mathrm{mid}_a.
\]

谓词不成立时输出 \(0\)。C++ 路径每次更新的复杂度为 \(O(1)\)：使用固定大小
环形缓冲区，除少数几根 K 线外不重新扫描完整窗口。

## 实现说明

递推过程实现在 `src/rtta/indicator.cpp` 的 `class CDLMorningStar` 中，并建立在
共享的 `CdlHistory` 和 `cdl::` 几何辅助函数之上。阈值采用适合流式计算的
几何规则，**并非**对 TA-Lib 每一张平均实体回看表逐行移植。

## 参考资料

- [Investopedia：Morning Star](https://www.investopedia.com/terms/m/morningstar.asp)
- [ChartSchool：K 线形态词典](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/candlestick-pattern-dictionary)
- [ChartSchool：K 线入门](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/introduction-to-candlesticks)
