# ParabolicSARExtended

## Summary

`ParabolicSARExtended` is a TA-Lib **SAREXT**-style Parabolic Stop and Reverse with
independent long/short acceleration-factor (AF) chains, optional fixed start SAR, and an
offset applied when the trend reverses. Each update returns the current SAR level.

## Update API

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

If `start != 0`, that value seeds the SAR; otherwise the first bars use extremes of
high/low. Long and short sides have separate AF init, step, and max parameters.

## Theory Of Operation

Wilder's Parabolic SAR trails price with a stop that accelerates toward the extreme
price as the trend persists. SAREXT generalizes the classic single AF schedule so long
and short trends can accelerate at different rates, and allows an additive offset when
flipping sides. RTTA follows the usual SAR update

\[
\mathrm{SAR}_{t} = \mathrm{SAR}_{t-1} + \mathrm{AF}\,(\mathrm{EP} - \mathrm{SAR}_{t-1}),
\]

clamped so the stop does not penetrate the prior two bars' range, then checks for
penetration of the current low (long) or high (short) to reverse.

## Recurrence

**Bar 0:** store \(H_0,L_0\); \(\mathrm{SAR} = \mathrm{start}\) if nonzero else \(L_0\);
\(\mathrm{EP}=H_0\); \(\mathrm{AF}=\mathrm{af\_init\_long}\); rising.

**Bar 1:** set rising if \(H_1 \ge H_0\); SAR from `start` or \(\min(L_0,L_1)\) /
\(\max(H_0,H_1)\); EP from highs/lows; AF from the corresponding side's init.

**Bars \(t \ge 2\):**

\[
\mathrm{SAR}' = \mathrm{SAR} + \mathrm{AF}\,(\mathrm{EP} - \mathrm{SAR}).
\]

If rising:

\[
\mathrm{SAR}' \leftarrow \min(\mathrm{SAR}',\, \min(L_{t-1}, L_t)).
\]

If \(L_t < \mathrm{SAR}'\) (reverse to short):

\[
\mathrm{SAR}' \leftarrow \mathrm{EP} + \mathrm{offset\_on\_reverse},\quad
\mathrm{EP}\leftarrow L_t,\quad
\mathrm{AF}\leftarrow \mathrm{af\_init\_short}.
\]

Else if \(H_t > \mathrm{EP}\): \(\mathrm{EP}\leftarrow H_t\),
\(\mathrm{AF}\leftarrow \min(\mathrm{AF}+\mathrm{af\_long},\, \mathrm{af\_max\_long})\).

If short:

\[
\mathrm{SAR}' \leftarrow \max(\mathrm{SAR}',\, \max(H_{t-1}, H_t)).
\]

If \(H_t > \mathrm{SAR}'\) (reverse to long):

\[
\mathrm{SAR}' \leftarrow \mathrm{EP} - \mathrm{offset\_on\_reverse},\quad
\mathrm{EP}\leftarrow H_t,\quad
\mathrm{AF}\leftarrow \mathrm{af\_init\_long}.
\]

Else if \(L_t < \mathrm{EP}\): \(\mathrm{EP}\leftarrow L_t\),
\(\mathrm{AF}\leftarrow \min(\mathrm{AF}+\mathrm{af\_short},\, \mathrm{af\_max\_short})\).

Commit \(\mathrm{SAR}\leftarrow\mathrm{SAR}'\) and store current high/low as previous.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class ParabolicSARExtended`. Returns a scalar `double` SAR each bar.

## Reference

- [TA-Lib SAREXT documentation](https://ta-lib.org/functions/SAREXT/)
- [Wilder, *New Concepts in Technical Trading Systems* (Parabolic SAR)](https://www.amazon.com/New-Concepts-Technical-Trading-Systems/dp/0894590278)
