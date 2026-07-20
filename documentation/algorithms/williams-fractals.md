# WilliamsFractals

## Summary

`WilliamsFractals` is RTTA's streaming 5-bar Bill Williams fractal detector.
An up fractal is confirmed at the middle bar of a five-high pattern; a down
fractal at the middle of a five-low pattern. Confirmation always lags by two
bars.

## Update API

```python
result = rtta.WilliamsFractals(fillna=True).update(high, low)
# result.up, result.down
```

Until five bars are available, `fillna=True` returns `up=0`, `down=0`;
`fillna=False` returns `NaN` for both fields.

## Theory Of Operation

Williams fractals mark local turning points: a high fractal is a bar whose high
is strictly greater than the highs of the two bars on either side; a low fractal
is strictly less than the lows of the two bars on either side. Because the
pattern needs two subsequent bars to confirm, the fractal is only known two bars
after the middle bar forms. RTTA emits the fractal price level on the
confirmation bar (when the newest bar completes the five-bar window), or `0.0`
when no fractal is confirmed.

## Recurrence

Maintain a shift register of the last five highs and lows, indices \(0\) oldest
through \(4\) newest. After each update with at least five bars:

\[
\begin{aligned}
\operatorname{up}_t &=
\begin{cases}
H^{(2)}, &
H^{(2)} > H^{(0)},\;
H^{(2)} > H^{(1)},\;
H^{(2)} > H^{(3)},\;
H^{(2)} > H^{(4)} \\
0, & \text{otherwise}
\end{cases}
\\[6pt]
\operatorname{down}_t &=
\begin{cases}
L^{(2)}, &
L^{(2)} < L^{(0)},\;
L^{(2)} < L^{(1)},\;
L^{(2)} < L^{(3)},\;
L^{(2)} < L^{(4)} \\
0, & \text{otherwise}
\end{cases}
\end{aligned}
\]

where \(H^{(i)}, L^{(i)}\) are the register entries. Both can be non-zero on the
same bar only if both patterns hold (uncommon).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class WilliamsFractals`. Result fields are `up` and `down`.

## Reference

- [Investopedia: Fractal Indicator](https://www.investopedia.com/terms/f/fractal.asp)
- [TradingPedia: Williams Fractals](https://www.tradingpedia.com/forex-trading-indicators/williams-fractals/)
