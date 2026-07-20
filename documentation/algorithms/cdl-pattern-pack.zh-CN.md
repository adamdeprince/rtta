# CDLPatternPack

## 摘要

`CDLPatternPack` 是 RTTA 的流式 **CDL 形态组合包**检测器。它通过一次 OHLC
更新运行一组固定的常用 CDL 检测器，并把所有代码一起返回。

输出遵循 TA-Lib 约定：看涨匹配为 **`+100`**，看跌匹配为 **`-100`**，
形态未触发时为 **`0`**。

## 更新 API

```python
result = rtta.CDLPatternPack(fillna=True).update(open, high, low, close)
# result.doji, result.hammer, result.engulfing, result.morning_star, ...
batch = rtta.CDLPatternPack(fillna=True).batch(open, high, low, close)
```

`update(...)` 每次接收一根 OHLC K 线，并返回多字段结果；每个已打包形态都使用
TA-Lib 风格代码（`+100` / `0` / `-100`）。`advance(...)` 更新状态但不
返回 Python 对象。数组 `batch(...)` 的结果与在新实例上依次调用 `update`
一致。

## 工作原理

K 线形态是根据柱体几何结构产生的短周期因果标签。它们本身并不预测收益，而是
标记犹豫、拒绝、吞没压力、多 K 线反转等结构，可与趋势、波动率或成交量背景
结合使用。

当需要许多标签、但不希望实例化 32 个独立对象时，该组合包很有用。在相同数据流
上，每个字段都与其对应的独立 `CDL*` 类完全相同。

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

每根 K 线都会分别更新并输出下列已打包检测器：

| 字段 | 检测器 |
|------|--------|
| `doji` | `CDLDoji` |
| `hammer` | `CDLHammer` |
| `hanging_man` | `CDLHangingMan` |
| `inverted_hammer` | `CDLInvertedHammer` |
| `shooting_star` | `CDLShootingStar` |
| `engulfing` | `CDLEngulfing` |
| `harami` | `CDLHarami` |
| `piercing` | `CDLPiercing` |
| `dark_cloud_cover` | `CDLDarkCloudCover` |
| `morning_star` | `CDLMorningStar` |
| `evening_star` | `CDLEveningStar` |
| `three_white_soldiers` | `CDL3WhiteSoldiers` |
| `three_black_crows` | `CDL3BlackCrows` |
| `marubozu` | `CDLMarubozu` |
| `spinning_top` | `CDLSpinningTop` |

每个分量都维护自己的环形状态；组合包不会在检测器之间共享历史，其行为与并排运行
这些检测器相同。

谓词不成立时输出 \(0\)。C++ 路径每次更新的复杂度为 \(O(1)\)：使用固定大小
环形缓冲区，除少数几根 K 线外不重新扫描完整窗口。

## 实现说明

递推过程实现在 `src/rtta/indicator.cpp` 的 `class CDLPatternPack` 中，并建立在
共享的 `CdlHistory` 和 `cdl::` 几何辅助函数之上。阈值采用适合流式计算的
几何规则，**并非**对 TA-Lib 每一张平均实体回看表逐行移植。

## 参考资料

- [ChartSchool：K 线入门](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/introduction-to-candlesticks)
- [ChartSchool：K 线形态词典](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/candlestick-pattern-dictionary)
- [ChartSchool：K 线入门](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/introduction-to-candlesticks)
