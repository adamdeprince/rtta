# ClosePressureReversalSignal

`ClosePressureReversalSignal` is a late-session cross-sectional reversal signal.
It is designed around a specific empirical idea: strong intraday losers can
experience price pressure near the close, and that pressure may partially reverse
over the final bars.

The primary paper reference is:

- Baltussen, Da, and Soebhag, "End-of-Day Reversal," SSRN, 2024/2025. The paper
  documents cross-sectional stock return reversals in the final 30 minutes of
  the trading day, especially from positive price pressure on intraday losers.
  SSRN page: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5039009

The RTTA implementation is not a verbatim replication. It converts the paper's
idea into an incremental signal that can run on aggregate bars and optionally be
used cross-sectionally through `ClosePressureReversalUniverse`.

## API

```python
rtta.ClosePressureReversalSignal(
    cutoff_after_bars=66,
    entry_start_after_bars=72,
    entry_end_after_bars=75,
    exit_after_bars=77,
    calibration_window=120,
    calibration_quantile=0.80,
    reversal_slope=0.025,
    entry_z=1.0,
    cost_buffer=0.0005,
    max_abs_target_fraction=0.04,
    participation_cap=0.02,
    allow_short_winners=False,
    fillna=False,
    max_loser_z=6.0,
    max_range_z=5.0,
    max_volume_shock=6.0,
)
```

`update(open, high, low, close, volume, vwap=nan, transactions=nan,
previous_session_close=nan, normal_dollar_volume=nan, normal_transactions=nan,
reset_session=False)` consumes one intraday aggregate bar.

The default timing assumes 5-minute bars in a 390-minute US regular session:

- bar 66 is the cutoff, roughly 3:00 p.m. Eastern,
- bars 72 to 75 are the entry window, roughly 3:30 to 3:50 p.m.,
- bar 77 is the exit window, roughly the close.

If the bar interval changes, the timing parameters should change too.

## Session State

At session start the indicator resets:

- bar count,
- previous log close,
- anchor log close,
- cumulative session VWAP state,
- return sums up to cutoff,
- the frozen rest-of-day return,
- any pending entry-window prediction.

Longer-lived calibration state remains unless `reset()` is called:

- rolling realized-error quantile,
- normal dollar-volume EWMA,
- normal transaction-count EWMA,
- range EWMA.

## Core Idea

The paper studies end-of-day reversal in the cross-section. RTTA turns that into
a bar-by-bar score:

1. Measure how far the stock is down or up on the day by the cutoff.
2. Normalize that move by realized intraday volatility up to the cutoff.
3. Increase the score when volume and transaction count are unusually high.
4. Increase the long score when the stock is below session VWAP.
5. Only emit entries in the configured late-session entry window.
6. Force exits in the configured exit window.

The default implementation is long-biased because the paper emphasizes reversal
from intraday losers. `allow_short_winners=True` enables a weaker short side for
intraday winners.

## Step-by-Step Algorithm

For each valid bar:

1. Increment `bar_number`.
2. Establish `anchor_log_close` from `previous_session_close` if supplied, or
   from the first bar close.
3. Compute close-to-close log return `ret1`.
4. Before and including the cutoff bar, accumulate return mean and variance
   terms.
5. Compute rest-of-day return:

   ```text
   rod_return = log(close) - anchor_log_close
   ```

6. At `cutoff_after_bars`, freeze that value:

   ```text
   frozen_rod_return = rod_return at cutoff
   ```

7. Estimate intraday volatility from returns up to cutoff:

   ```text
   intraday_vol = sqrt(var(ret1_to_cutoff) * count)
   ```

8. Convert the frozen return into loser and winner z-scores:

   ```text
   loser_z = max(0, -frozen_rod_return) / intraday_vol
   winner_z = max(0,  frozen_rod_return) / intraday_vol
   ```

9. Compute activity shocks:

   ```text
   volume_shock = log(dollar_volume / normal_dollar_volume)
   transaction_shock = log(transactions / normal_transactions)
   ```

10. Compute session VWAP gap:

    ```text
    vwap_gap = close / session_vwap - 1
    ```

11. Compute range z-score from the current high-low range relative to the range
    EWMA.

12. Build pressure multipliers:

    ```text
    volume_mult = 1 + 0.20 * clamp(volume_shock, -2, 4)
    tx_mult     = 1 + 0.10 * clamp(transaction_shock, -2, 4)
    ```

13. For long reversal pressure, reward losers below VWAP:

    ```text
    long_vwap_mult = 1 + 0.50 * clamp((-vwap_gap) / intraday_vol, 0, 3)
    long_pressure_score = loser_z * volume_mult * tx_mult * long_vwap_mult
    ```

14. If shorting winners is allowed, compute a weaker short pressure score for
    winners above VWAP:

    ```text
    short_vwap_mult = 1 + 0.50 * clamp(vwap_gap / intraday_vol, 0, 3)
    short_pressure_score = winner_z * volume_mult * tx_mult * short_vwap_mult
    ```

15. Convert pressure into an expected reversal:

    ```text
    long_prediction =
        reversal_slope * loser_return * clamp(long_pressure_score / 2, 0, 2)

    short_prediction =
       -0.50 * reversal_slope * winner_return * clamp(short_pressure_score / 2, 0, 2)
    ```

16. Block entries when the move looks too extreme:

    ```text
    news_guard =
        loser_z > max_loser_z
        or winner_z > max_loser_z
        or range_z > max_range_z
        or volume_shock > max_volume_shock
    ```

## Error Band

When an entry-window prediction is made, it is stored. At or after
`exit_after_bars`, the realized log return from the prediction bar to the current
bar is compared to the prediction:

```text
realized_error = abs(realized_return - prediction)
```

That error is pushed into a rolling quantile. Once there are enough calibration
samples, `radius` is:

```text
radius = max(rolling_error_quantile, cost_buffer)
```

Before calibration is ready, fallback radius is:

```text
radius = max(2 * cost_buffer, max(0.0005, 1.25 * intraday_vol))
```

As with the other RTTA moonshot signals, this is conformal-inspired empirical
calibration, not a formal paper replication with guaranteed coverage.

## Trading Outputs

The score is:

```text
score = prediction / (radius + cost_buffer)
```

Entries are allowed only when:

- the current bar is inside `entry_window`,
- `news_guard` is false,
- the score clears `entry_z`,
- the relevant side is enabled.

The signal is forced flat in the exit window:

```text
signal = 0 if exit_window or news_guard or not entry_window
signal = +1 if prediction > entry_z * (radius + cost_buffer)
signal = -1 if allow_short_winners and prediction < -entry_z * (radius + cost_buffer)
```

`target_fraction` is score-scaled and capped by `max_abs_target_fraction`.
`max_trade_dollars = participation_cap * normal_dollar_volume`.

## ClosePressureReversalUniverse

`ClosePressureReversalUniverse` holds one `ClosePressureReversalSignal` per
symbol and provides a cross-sectional selector:

```python
selected, exits = universe.update(
    indices, open, high, low, close, volume, vwap, transactions, top_fraction
)
```

For each bar group:

- it updates all supplied symbols,
- records any symbol in its exit window,
- ranks positive-score entry-window candidates,
- selects the top `ceil(candidate_count * top_fraction)`.

This is the preferred API for the Massive/Polygon example because the paper's
effect is cross-sectional.

## Outputs

- `bar_number`: 1-based bar count in the current session.
- `rod_return`: current return from previous close or first bar anchor.
- `frozen_rod_return`: return frozen at cutoff.
- `loser_z` / `winner_z`: normalized frozen down/up move.
- `range_z`: current bar range relative to range EWMA.
- `volume_shock`: log dollar-volume shock.
- `transaction_shock`: log transaction-count shock.
- `vwap_gap`: close relative to session VWAP.
- `pressure_score`: active side's pressure score.
- `prediction`: expected reversal log return.
- `radius`: empirical uncertainty band.
- `score`: prediction divided by radius plus cost buffer.
- `signal`: `-1`, `0`, or `+1`.
- `target_fraction`: capped suggested exposure.
- `max_trade_dollars`: liquidity cap.
- `realized_error`: latest matured entry-to-exit prediction error.
- `entry_window`, `exit_window`, `frozen`, `news_guard`: boolean state flags.

## Intended Use

Use this on regular intraday bars with correct session resets. For US equities,
5-minute bars are the intended default. The example script ranks symbols
cross-sectionally and uses quotes for delayed fills. The indicator itself does
not place orders and does not know about borrow, halt, limit-up/limit-down, or
portfolio constraints.
