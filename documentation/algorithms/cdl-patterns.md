# Candlestick (CDL) patterns

RTTA’s CDL family is a set of **streaming** candlestick pattern detectors. Each
`update(open, high, low, close)` call consumes one bar and returns a TA-Lib-style
code:

| Code | Meaning |
|------|---------|
| `+100` | Bullish pattern match |
| `-100` | Bearish pattern match |
| `0` | No match (or warmup with `fillna=True`) |
| `NaN` | Warmup when `fillna=False` |

## Shared geometry

All detectors use the same bar features and short causal ring (`CdlHistory` in
`src/rtta/indicator.cpp`). See any individual page’s **Theory Of Operation** for
the body/range/shadow definitions and shared predicates
(`doji`, `engulf`, `inside`, prior-trend \(\tau\), …).

## Catalog

### Single-bar
- [`CDLDoji`](cdl-doji.md), [`CDLDragonflyDoji`](cdl-dragonfly-doji.md), [`CDLGravestoneDoji`](cdl-gravestone-doji.md), [`CDLLongLeggedDoji`](cdl-long-legged-doji.md)
- [`CDLHammer`](cdl-hammer.md), [`CDLHangingMan`](cdl-hanging-man.md), [`CDLInvertedHammer`](cdl-inverted-hammer.md), [`CDLShootingStar`](cdl-shooting-star.md)
- [`CDLMarubozu`](cdl-marubozu.md), [`CDLClosingMarubozu`](cdl-closing-marubozu.md)
- [`CDLSpinningTop`](cdl-spinning-top.md), [`CDLHighWave`](cdl-high-wave.md)
- [`CDLLongLine`](cdl-long-line.md), [`CDLShortLine`](cdl-short-line.md), [`CDLBeltHold`](cdl-belt-hold.md)

### Two-bar
- [`CDLEngulfing`](cdl-engulfing.md), [`CDLHarami`](cdl-harami.md), [`CDLHaramiCross`](cdl-harami-cross.md)
- [`CDLPiercing`](cdl-piercing.md), [`CDLDarkCloudCover`](cdl-dark-cloud-cover.md)
- [`CDLDojiStar`](cdl-doji-star.md), [`CDLMatchingLow`](cdl-matching-low.md), [`CDLCounterAttack`](cdl-counter-attack.md)

### Three-bar
- [`CDLMorningStar`](cdl-morning-star.md), [`CDLEveningStar`](cdl-evening-star.md)
- [`CDLMorningDojiStar`](cdl-morning-doji-star.md), [`CDLEveningDojiStar`](cdl-evening-doji-star.md)
- [`CDL3WhiteSoldiers`](cdl-3-white-soldiers.md), [`CDL3BlackCrows`](cdl-3-black-crows.md)
- [`CDL3Inside`](cdl-3-inside.md), [`CDL3Outside`](cdl-3-outside.md), [`CDLTriStar`](cdl-tri-star.md)

### Pack
- [`CDLPatternPack`](cdl-pattern-pack.md) — common patterns in one multi-output update

## Usage sketch

```python
import rtta

eng = rtta.CDLEngulfing()
code = eng.update(o, h, l, c)   # 100, -100, or 0

pack = rtta.CDLPatternPack()
r = pack.update(o, h, l, c)
# r.doji, r.engulfing, r.morning_star, ...
```

## Notes

- Definitions are **geometric streaming rules**, not a guaranteed bit-identical
  port of TA-Lib’s full average-body threshold tables.
- Hammer-family patterns use a light prior-trend filter; most others are pure
  geometry.
- Prefer composing CDL labels with trend/volatility features rather than trading
  them in isolation.

## Reference

- [ChartSchool: Candlestick pattern dictionary](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/candlestick-pattern-dictionary)
- [ChartSchool: Introduction to candlesticks](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/introduction-to-candlesticks)
