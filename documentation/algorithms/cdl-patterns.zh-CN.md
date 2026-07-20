# K 线（CDL）形态

RTTA 的 CDL 家族是一组**流式** K 线形态检测器。每次调用
`update(open, high, low, close)` 都会接收一根 K 线，并返回 TA-Lib 风格的
代码：

| 代码 | 含义 |
|------|------|
| `+100` | 匹配看涨形态 |
| `-100` | 匹配看跌形态 |
| `0` | 未匹配（或 `fillna=True` 时处于预热期） |
| `NaN` | `fillna=False` 时处于预热期 |

## 共享几何结构

所有检测器都使用相同的 K 线特征和短期因果环形缓冲区（即
`src/rtta/indicator.cpp` 中的 `CdlHistory`）。实体、全幅、影线的定义以及
共享谓词（`doji`、`engulf`、`inside`、先行趋势 \(\tau\) 等）可参阅任一
形态页面的**工作原理**一节。

## 目录

### 单根 K 线
- [`CDLDoji`](cdl-doji.zh-CN.md)、[`CDLDragonflyDoji`](cdl-dragonfly-doji.zh-CN.md)、[`CDLGravestoneDoji`](cdl-gravestone-doji.zh-CN.md)、[`CDLLongLeggedDoji`](cdl-long-legged-doji.zh-CN.md)
- [`CDLHammer`](cdl-hammer.zh-CN.md)、[`CDLHangingMan`](cdl-hanging-man.zh-CN.md)、[`CDLInvertedHammer`](cdl-inverted-hammer.zh-CN.md)、[`CDLShootingStar`](cdl-shooting-star.zh-CN.md)
- [`CDLMarubozu`](cdl-marubozu.zh-CN.md)、[`CDLClosingMarubozu`](cdl-closing-marubozu.zh-CN.md)
- [`CDLSpinningTop`](cdl-spinning-top.zh-CN.md)、[`CDLHighWave`](cdl-high-wave.zh-CN.md)
- [`CDLLongLine`](cdl-long-line.zh-CN.md)、[`CDLShortLine`](cdl-short-line.zh-CN.md)、[`CDLBeltHold`](cdl-belt-hold.zh-CN.md)

### 两根 K 线
- [`CDLEngulfing`](cdl-engulfing.zh-CN.md)、[`CDLHarami`](cdl-harami.zh-CN.md)、[`CDLHaramiCross`](cdl-harami-cross.zh-CN.md)
- [`CDLPiercing`](cdl-piercing.zh-CN.md)、[`CDLDarkCloudCover`](cdl-dark-cloud-cover.zh-CN.md)
- [`CDLDojiStar`](cdl-doji-star.zh-CN.md)、[`CDLMatchingLow`](cdl-matching-low.zh-CN.md)、[`CDLCounterAttack`](cdl-counter-attack.zh-CN.md)

### 三根 K 线
- [`CDLMorningStar`](cdl-morning-star.zh-CN.md)、[`CDLEveningStar`](cdl-evening-star.zh-CN.md)
- [`CDLMorningDojiStar`](cdl-morning-doji-star.zh-CN.md)、[`CDLEveningDojiStar`](cdl-evening-doji-star.zh-CN.md)
- [`CDL3WhiteSoldiers`](cdl-3-white-soldiers.zh-CN.md)、[`CDL3BlackCrows`](cdl-3-black-crows.zh-CN.md)
- [`CDL3Inside`](cdl-3-inside.zh-CN.md)、[`CDL3Outside`](cdl-3-outside.zh-CN.md)、[`CDLTriStar`](cdl-tri-star.zh-CN.md)

### 组合包
- [`CDLPatternPack`](cdl-pattern-pack.zh-CN.md)——一次多输出更新即可计算常用形态

## 用法示例

```python
import rtta

eng = rtta.CDLEngulfing()
code = eng.update(o, h, l, c)   # 100、-100 或 0

pack = rtta.CDLPatternPack()
r = pack.update(o, h, l, c)
# r.doji、r.engulfing、r.morning_star 等
```

## 注意事项

- 这些定义是**适合流式计算的几何规则**，不保证与 TA-Lib 完整的平均实体阈值表
  逐位相同。
- 锤子线家族的形态使用轻量级先行趋势过滤器；其他大多数形态只使用几何结构。
- 与其单独依据 CDL 标签交易，更适合把它们与趋势或波动率特征组合使用。

## 参考资料

- [ChartSchool：K 线形态词典](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/candlestick-pattern-dictionary)
- [ChartSchool：K 线入门](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/introduction-to-candlesticks)
