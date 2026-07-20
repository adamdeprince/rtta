# Hilbert 正弦波（HilbertSineWave）

## 摘要

`HilbertSineWave` 是 RTTA 对由主导周期相位产生的 Hilbert 正弦波与超前正弦波的流式实现（TA-Lib `HT_SINE`）。两条波形组成周期振荡器，其交叉可用于识别周期转折。

## 更新 API

```python
result = rtta.HilbertSineWave(fillna=True).update(value)
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `fillna`  | `True`  | 若为 `False`，超过 63 个样本的回看期之前返回 NaN |

`update(...)` 返回：

- `sine`——\(\sin(\phi_t)\)
- `lead_sine`——\(\sin(\phi_t+45^\circ)\)

其中 \(\phi_t\) 是以度为单位的主导周期相位。`advance(value)` 更新状态；`last()` 返回缓存的结果。

## 工作原理

[`HilbertDominantCyclePhase`](hilbert-dominant-cycle-phase.zh-CN.md) 产生相位 \(\phi_t\) 后，通过正弦函数映射可得到 \([-1,1]\) 范围内的标准化周期波形。超前正弦波向前移动 \(45^\circ\)，使正弦波与超前正弦波的交叉早于纯正弦波的波峰和波谷出现——TA-Lib / Ehlers 把这些交叉用作周期模式的择时事件（[`HilbertTrendMode`](hilbert-trend-mode.zh-CN.md) 也会使用它们）。

由于相位来自自适应周期，当估计周期长度变化时，正弦波也会相应拉伸或压缩。

## 递推公式

令 \(\phi_t\) 为共用引擎给出的主导周期相位（角度制）。使用 `deg2rad = 1/rad2deg`、`rad2deg = 45/\operatorname{atan}(1)` 转换：

\[
sine_t = \sin(\phi_t \cdot \texttt{deg2rad})
\]

\[
lead\_sine_t = \sin\big((\phi_t + 45)\cdot \texttt{deg2rad}\big)
\]

当 `fillna=False` 时，在处理超过 63 个样本之前，两个字段均为 NaN。

## 实现说明

- 以 `class HilbertSineWave` 实现，封装 `HilbertCycleEngine::sine()` 与 `::lead_sine()`。
- 结果类型：`HilbertSineWaveResult`（`sine`、`lead_sine`）。
- 回看期：`lookback_phase_ = 63`（与相位 / 趋势线 / 趋势模式相同）。
- 批量辅助函数：`batch_hilbert_sine_wave`。

## 参考资料

- [TA-Lib HT_SINE](https://ta-lib.org/functions/ht_sine)
- [MESA Software——Ehlers 论文](https://www.mesasoftware.com/)
