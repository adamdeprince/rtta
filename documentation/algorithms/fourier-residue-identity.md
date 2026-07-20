# FourierResidueIdentity

## Summary

`FourierResidueIdentity` is a streaming implementation of the **Fourier-Residue
Identity (FRI)**, which splits return autocorrelation into a **direction (sign)**
channel and a **magnitude** channel that are individually testable and neither
redundant.

It answers a question no scalar autocorrelation can: when a series mean-reverts,
*is the direction predictable, or only the size?* The two cases call for opposite
trades. A directional reversal says "go long after a down day". A magnitude-only
reversal says "expect a smaller move tomorrow, of unknown sign" — a volatility
signal, not a directional one, and a contrarian bet on it has no statistical
warrant at all.

The motivating fact from the source paper: SPY's lag-1 autocorrelation is
\(\hat\rho(1) = -0.081\), which is \(7.4\) standard errors below zero — one of the
most significant regularities in empirical equity finance. Yet the FRI sign test
on the same data returns \(z_{\mathrm{sign}} = -1.59\) (\(p = 0.11\)). Knowing SPY
fell yesterday tells you essentially nothing about whether it rises or falls
today. **The bounce has no direction.**

## Reference

V. Portnaya, *"The Bounce Has No Direction: Sign, Magnitude, and the
Microstructure of Equity Return Predictability — Fourier-Residue Identities,
Fejér Sums, and Evidence from US Equity and Cross-Asset Markets, 1993–2026"*,
[arXiv:2606.29591](https://arxiv.org/abs/2606.29591) (June 2026).

## Update API

```python
out = rtta.FourierResidueIdentity().update(close)

# OHLC-compatible overload (open/high/low ignored)
out = rtta.FourierResidueIdentity().update(open, high, low, close)

indicator = rtta.FourierResidueIdentity()
indicator.advance(close)          # no return value
out = indicator.last()
indicator.reset()

batch = rtta.FourierResidueIdentity().batch(close_array)
```

Constructor knobs:

| Argument | Default | Role |
|----------|---------|------|
| `max_lag` | `8` | lags \(M\) tracked; raised to cover `horizon - 1` and `test_lag` |
| `horizon` | `2` | variance-ratio horizon \(q\) |
| `test_lag` | `1` | lag \(m\) reported by the scalar outputs |
| `span` | `512.0` | EWMA memory in observations |
| `median_window` | `256` | rolling window for the median \(\lvert r\rvert\) that defines the \(k=4\) buckets |
| `entry_z` / `exit_z` | `2.0` / `1.0` | hysteresis on sign-channel evidence for `signal` |
| `fillna` | `True` | `0` vs `NaN` during warmup |

## Outputs

| Field | Meaning |
|-------|---------|
| `rho` | scalar autocorrelation \(\hat\rho(m)\) |
| `rho_sign` | sign channel \(\gamma_{1,2}(m) = 2p_{m,0} - 1\) |
| `rho_magnitude` | magnitude channel \(\operatorname{Re}\gamma_{1,4}(m)\) |
| `z_rho` | Bartlett \(z\) for `rho` |
| `z_sign` | binomial \(z\) for `rho_sign` |
| `directional_share` | \(\lvert z_{\text{sign}}\rvert / (\lvert z_{\text{sign}}\rvert + \lvert z_\rho\rvert)\) |
| `elliptical_ratio` | `rho_sign` divided by its Gaussian benchmark (see below) |
| `variance_ratio` | \(\mathrm{VR}(q)\) |
| `variance_ratio_sign` | \(\mathrm{VR}_2(q)\), direction channel |
| `variance_ratio_magnitude` | \(\mathrm{VR}_4(q)\), magnitude channel |
| `z_variance_ratio` | Lo–MacKinlay heteroskedasticity-robust \(z^*\) |
| `persistence` | half-period ratio \(R_N\) |
| `signal` | `-1` / `0` / `+1`, gated on sign-channel significance |
| `score` | continuous directional score in \([-1, 1]\) |
| `magnitude_forecast` | conditional \(\mathbb{E}\lvert r_{t+1}\rvert\) |

## Theory Of Operation

### Fejér / variance-ratio identity

The Lo–MacKinlay variance ratio admits an exact autocorrelation representation
(Proposition 2.2):

\[
\mathrm{VR}(q) \;=\; 1 + 2\sum_{m=1}^{q-1}\Bigl(1 - \tfrac{m}{q}\Bigr)\hat\rho(m)
\;=\; 1 + 2\,\mathcal{C}_q
\]

The Fejér weights \(w_m = 1 - m/q\) taper linearly to zero at lag \(q\), giving
short lags — where microstructure lives — the most weight.

### The FRI decomposition

Encode each return as a \(k\)-ary symbol \(s_t \in \{0,\dots,k-1\}\) and evaluate
the characters of the cyclic group \(\mathbb{Z}/k\mathbb{Z}\) (Definition 2.4):

\[
\gamma_{A,k}(m) \;=\; \frac{1}{N-m}\sum_t \omega^{A(s_t - s_{t+m})},
\qquad \omega = e^{2\pi i/k}
\]

**Sign channel (\(k=2\)).** With \(s_t = \mathbb{1}[r_t > 0]\) and \(\omega = -1\),
the character is \(+1\) when successive signs agree and \(-1\) when they disagree,
collapsing to a closed form (Proposition 2.5):

\[
\gamma_{1,2}(m) \;=\; 2p_{m,0} - 1 \;=:\; \hat\rho_{\mathrm{sign}}(m)
\]

where \(p_{m,0}\) is the probability of closing on the same side of zero \(m\)
periods apart. Under the random-walk null \(p_{m,0} = \tfrac12\). This is a
**magnitude-free** test of directional dependence: positive means momentum,
negative means genuine contrarian reversal.

**Magnitude channel (\(k=4\)).** Returns are bucketed at the median
\(\lvert r\rvert\) into a signed size ladder
\(\{\text{large-down},\text{small-down},\text{small-up},\text{large-up}\} = \{0,1,2,3\}\)
with \(A = 1\), \(\omega = i\). This measures whether the *size* bucket persists,
independently of whether direction agrees.

Applying the Fejér identity per channel gives \(\mathrm{VR}_2(q)\) and
\(\mathrm{VR}_4(q)\) (Equation 5). The channels are nonnested: a series with sign
momentum but no magnitude clustering has \(\mathrm{VR}_2 > 1\) and
\(\mathrm{VR}_4 \approx 1\), and vice versa.

### Which mechanism is which

| Mechanism | Sign \(\mathrm{VR}_2\) | Magn. \(\mathrm{VR}_4\) | Lag range |
|---|---|---|---|
| Bid-ask bounce | \(\approx 1\) | \(< 1\) | lag 1 only |
| Non-synchronous trading | \(\approx 1\) | \(< 1\) | lags 1–3 |
| Dealer inventory | \(\approx 1\) | \(< 1\) | lags 1–2 |
| Adverse selection | \(\neq 1\) | \(\neq 1\) | lags 2–5 |
| Partial price adjustment | \(\neq 1\) | \(\neq 1\) | lags 2–7 |
| Volatility clustering | \(\approx 1\) | \(> 1\) | all lags |

Only the mechanisms with \(\mathrm{VR}_2 \neq 1\) are directionally tradeable.

### Subsample persistence

The half-period ratio (Definition 2.6) answers whether a detected deviation will
survive out of sample:

\[
G_N = \max_{1\le m\le M}\lvert\hat\rho_N(m)\rvert,
\qquad R_N = G_{N/2}\,/\,G_N
\]

Under IID noise \(R_N \to \sqrt2 \approx 1.41\); under genuine serial dependence
\(R_N \to 1\) (Proposition 2.7). Halving the sample inflates a *noise* maximum by
\(\sqrt2\) but barely moves a *structural* one.

### Streaming form

The paper estimates full-sample; this implementation is bounded-memory and
online. Sample means become debiased EWMAs of span `span` (so early updates
behave like an expanding sample rather than a biased ramp), and \(n\) is replaced
by the effective sample size \(n_{\text{eff}} = \min(\text{count},\,(2-\alpha)/\alpha)\).

`persistence` is computed by running a second parallel estimator at half the
span, which is the streaming analogue of the \(G_{N/2}/G_N\) construction. It is
only meaningful for a **finite** `span`; with an effectively infinite span both
estimators coincide and the ratio degenerates to 1.

`z_variance_ratio` uses the Lo–MacKinlay M2 statistic with
\(\hat\delta(j)\) in its standard \(O(1/n)\) normalisation, so \(\phi_2(q)\)
collapses to \(\phi_1(q)\) under an IID null. Daily equity returns have strong
GARCH effects, and the homoskedastic \(z\) over-rejects at 10–12% where the
robust \(z^*\) holds its 5% size.

### The elliptical benchmark (extension beyond the paper)

The sign channel is not *free* of \(\rho\) — it has a predictable null. For a
bivariate normal pair, Grothendieck's identity gives

\[
\mathbb{E}[\operatorname{sgn}X \operatorname{sgn}Y] = \tfrac{2}{\pi}\arcsin\rho
\]

so any elliptical process with autocorrelation \(\rho\) *must* show a sign channel
of about \(0.64\rho\). `elliptical_ratio` divides the observed `rho_sign` by that
benchmark, giving a scale-free diagnostic with a null of 1:

- \(\approx 1\) — the autocorrelation is exactly as directional as a Gaussian
  process with the same \(\rho\).
- \(\approx 0\) — the predictability is carried by magnitude alone.

This matters because it sharpens the paper's own conclusion. A simulated pure
Roll bounce scores **0.95** here, not 0 — a bounce *does* leak into the sign
channel, because sign correlation is pinned to \(\rho\) for near-Gaussian data.
What makes SPY genuinely unusual is that its pair \((\rho, \rho_{\mathrm{sign}}) =
(-0.081, -0.017)\) scores **0.34**: far *less* directional than any elliptical
process with that \(\rho\). The reversal is concentrated in large moves — which
dominate the covariance — while a typical day's direction stays a coin flip.

Only interpret `elliptical_ratio` when the scalar ACF is itself detectable
(\(\lvert z_\rho\rvert\) large); it is returned as `NaN` when \(\rho\) is too close
to zero for the ratio to be stable.

## Recurrence

State: previous close; a ring buffer of the last \(M\) returns with their signs
and \(k=4\) codes; a rolling median of \(\lvert r\rvert\); debiased EWMA pairs
\((v,w)\) for \(\mu\), \(\mu_2\), \(\lvert r\rvert\), \(\lvert r\rvert^2\); and per
lag \(m \le M\) the EWMAs \(c_m\) (cross-product), \(g_m\) (sign agreement),
\(h_m\) (\(\operatorname{Re}\) magnitude character), \(a_m\)
(\(\lvert r\rvert\) cross-product) and \(d_m\) (quartic, for \(\hat\delta\)).
A parallel \(\mu, \mu_2, c_m\) set runs at half the span for \(R_N\).

Each debiased EWMA accumulates \(v \leftarrow (1-\alpha)v + \alpha x\) and
\(w \leftarrow (1-\alpha)w + \alpha\), reporting \(v/w\).

1. \(r_t = \log(C_t/C_{t-1})\); \(\sigma_t = \operatorname{sign}(r_t)\) as \(\pm1\);
   \(\ell_t = \mathbb{1}[\lvert r_t\rvert > \operatorname{med}_t]\);
   code \(s_t \in \{0,1,2,3\}\) from \((\sigma_t, \ell_t)\).
2. For \(m = 1\ldots\min(\text{count}, M)\), against \(r_{t-m}\) held at ring slot
   \(m-1\): push \(r_t r_{t-m}\) into \(c_m\); \(\sigma_t\sigma_{t-m}\) into \(g_m\);
   \(\cos\!\bigl(\tfrac{\pi}{2}(s_t - s_{t-m})\bigr)\) into \(h_m\) via a
   4-entry table on \((s_t-s_{t-m}) \bmod 4\); \(\lvert r_t r_{t-m}\rvert\) into
   \(a_m\); and \((r_t-\mu)^2(r_{t-m}-\mu)^2\) into \(d_m\).
3. Push \(r_t\) into the global moment EWMAs, then into the ring buffer.
4. \(\operatorname{Var} = \mu_2 - \mu^2\);
   \(\hat\rho(m) = (c_m - \mu^2)/\operatorname{Var}\);
   \(\hat\rho_{\mathrm{sign}}(m) = g_m\);
   \(\operatorname{Re}\gamma_{1,4}(m) = h_m\).
5. \(n_{\text{eff}} = \min(\text{count}, (2-\alpha)/\alpha)\);
   \(z_\rho = \hat\rho / \sqrt{(1 + 2\sum_{k<m}\hat\rho(k)^2)/n_{\text{eff}}}\);
   \(z_{\mathrm{sign}} = \hat\rho_{\mathrm{sign}}\sqrt{n_{\text{eff}} - m}\).
6. Fejér-weight lags \(1\ldots q-1\) into \(\mathrm{VR}\), \(\mathrm{VR}_2\),
   \(\mathrm{VR}_4\); accumulate \(\phi_2\) from
   \(\hat\delta(j) = d_j/(n_{\text{eff}}\operatorname{Var}^2)\).
7. \(G\) and \(G_{1/2}\) are the running maxima of \(\lvert\hat\rho(m)\rvert\) over
   the full- and half-span sets; \(R_N = G_{1/2}/G\).
8. Score \(= \hat\rho_{\mathrm{sign}}\sigma_{t+1-m}\); arm/disarm on
   \(\lvert z_{\mathrm{sign}}\rvert\) against `entry_z`/`exit_z`; emit the signed
   score when armed.

Each update is \(O(M)\) with \(M =\) `max_lag` (default 8) and causal. The rolling
median dominates at \(O(\text{median\_window})\) via `nth_element`; its scratch
buffer reaches full size during warmup and is not reallocated afterwards, so the
steady-state hot path is allocation-free. Lower `median_window` if the \(k=4\)
bucket boundary does not need that much history.

## Trading Interpretation

`signal` is non-zero only while the **sign** channel itself clears `entry_z`,
with hysteresis at `exit_z`:

- `rho_sign < 0` and significant → stance opposes the sign of \(r_{t+1-m}\)
  (contrarian).
- `rho_sign > 0` and significant → stance follows it (momentum).
- otherwise → `0`, regardless of how significant `rho` itself is.

`magnitude_forecast` carries the content that remains statistically warranted
even when direction does not: a conditional forecast of the next absolute
return, for volatility sizing, straddle/strangle timing, or scaling a
delta-hedged book.

A practical reading of the two together:

| `z_rho` | `z_sign` | Reading |
|---|---|---|
| large | large, same sign | genuine directional dependence — trade the direction |
| large | small | magnitude-only — size positions, do not bet on direction |
| small | large | direction pattern hidden from the scalar ACF by offsetting magnitudes |
| small | small | no exploitable structure |

### What this does *not* do

Be precise about the limits of the sign gate. Simulating a pure Roll bounce
(martingale efficient price, IID trade direction, half-spread \(0.45\sigma\))
produces \(\rho = -0.139\) **and** \(\rho_{\mathrm{sign}} = -0.101\) at
\(z_{\mathrm{sign}} = -6.5\) — a thoroughly significant sign channel. So
`signal` will fire on a simulated bid-ask bounce; it is *not* a bounce filter.
Nothing computed from close prices alone can be, because the observed series
genuinely does reverse — what makes the bounce untradeable is the spread you
cross, which is not in the data.

What the sign channel does deliver is the separation itself: when
\(\lvert z_\rho\rvert\) is large and \(\lvert z_{\mathrm{sign}}\rvert\) is not,
you know the direction of a typical bar is a coin flip and only sizing is
warranted. Real SPY is that case; a simulated Roll bounce is not.

The actionable consequence of a low `elliptical_ratio` is *where* the edge sits.
On a simulated magnitude-carried reversal (`elliptical_ratio` = 0.37), taking the
contrarian stance only after an above-median move retains 97% of the gross P&L
while trading half as many bars, nearly doubling per-bar edge. On a uniform
directional reversal (`elliptical_ratio` = 1.00) the same restriction retains
only 78%. A low ratio tells you to concentrate risk on large moves rather than
to trade every bar.

## Notes

- `max_lag` is raised automatically to cover both `horizon - 1` and `test_lag`,
  so an understated `max_lag` cannot read outside the ring buffer.
- The lag-3 sign channel is worth watching even when \(\hat\rho(3)\) is
  negligible: the paper finds \(z_{\mathrm{sign}}(3) = -2.32\) (\(p = 0.02\)) for
  SPY where the scalar ACF gives \(p = 0.50\) — a partial-price-adjustment
  channel invisible to the standard test. Set `test_lag=3` to read it.
- Non-finite or non-positive prices are rejected without damaging estimator
  state; the next valid observation resumes normally.
