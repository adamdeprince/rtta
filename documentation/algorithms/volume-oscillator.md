# VolumeOscillator

## Summary

`VolumeOscillator` is RTTA's streaming implementation of: Percent difference between short and long simple moving averages of volume.

## Update API

```python
result = rtta.VolumeOscillator().update(volume)
```

The `update(...)` call consumes one observation using `volume`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`VolumeOscillator` measures whether short-term volume is running above or below
a longer volume baseline. It is the SMA-based volume analog of a price percent
oscillator (distinct from [`PercentageVolume`](percentage-volume.md), which uses
EMAs and a signal line).

## Recurrence

Let \(v_t = volume_t\), \(n_s\) the short window, and \(n_l\) the long window.

\[
S_t = \operatorname{SMA}_{n_s}(v_t), \qquad
L_t = \operatorname{SMA}_{n_l}(v_t)
\]

\[
VO_t = 100 \cdot \frac{S_t - L_t}{L_t}
\]

The return value is the current scalar indicator value.

## Composed Primitives

[`SMA`](sma.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class VolumeOscillator`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/percentage-volume-oscillator-pvo)
