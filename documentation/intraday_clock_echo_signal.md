# IntradayClockEchoSignal

`IntradayClockEchoSignal` is a same-clock intraday periodicity signal. It learns
which times of day have historically shown positive or negative residual returns
and uses that clock pattern to forecast the next `horizon_bars` of returns.

The main paper references are:

- Heston, Korajczyk, and Sadka, "Intraday Patterns in the Cross-Section of Stock
  Returns," Journal of Finance, 2010. The paper documents return continuation at
  intraday intervals that are exact multiples of a trading day, with effects
  lasting many trading days. SSRN page: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1107590
- Haendler, Heston, Korajczyk, and Sadka, "The Intra-Day Stock Return
  Periodicity Puzzle," SSRN, 2025. The paper revisits the effect out of sample
  and studies explanations including VWAP-like trading and market-on-close
  trading. SSRN page: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5749704

The RTTA implementation is not a full cross-sectional replication. It is an
online, per-symbol indicator that learns slot-level residual return behavior
from prior bars or from explicit training days.

## API

```python
rtta.IntradayClockEchoSignal(
    slots_per_session=195,
    horizon_bars=15,
    lookback_days=40,
    min_slot_samples=10,
    calibration_window=500,
    calibration_quantile=0.80,
    entry_z=1.0,
    cost_buffer=0.0005,
    max_abs_target_fraction=0.03,
    participation_cap=0.02,
    allow_short=True,
    fillna=False,
)
```

For 2-minute bars in a 390-minute US session, `slots_per_session=195`. For
5-minute bars, use 78. The example script computes this from the interval unless
it is supplied.

`update(open, high, low, close, volume, vwap=nan, transactions=nan,
market_return=0.0, normal_dollar_volume=nan, slot=0, reset_session=False)`
consumes one bar.

The `train(days)` method accepts a sequence of day records. Each day is an
iterable of dict-like or tuple-like records containing at least open, high, low,
close, volume, and optionally vwap, transactions, market_return,
normal_dollar_volume, and slot.

## What "Clock Echo" Means

The paper result is about time-of-day repetition. If a stock tends to rise at a
particular half-hour slot, related behavior may recur at the same slot on later
days. RTTA stores this as `slot_echo_[slot]`, an exponentially weighted average
of residual returns for each time-of-day slot.

The indicator does not assume raw return is the full signal. It subtracts the
optional `market_return` first:

```text
bar_return = log(close_t / close_{t-1})
residual_return = bar_return - market_return
```

This makes the learned pattern closer to idiosyncratic same-clock behavior when
a market or ETF return is supplied.

## Training and Online State

For each slot, the indicator stores:

- `slot_echo`: EWMA of residual returns at that slot,
- `slot_abs_err`: EWMA of absolute residual returns at that slot,
- `slot_volume`: EWMA of dollar volume at that slot,
- `slot_count`: number of observations for that slot.

The EWMA learning rate is:

```text
alpha = 2 / (lookback_days + 1)
```

`train(days)` simply replays prior bars through `update(...)`, resetting
intraday state between days. It updates the slot-level long-lived state and the
rolling prediction-error calibration state.

`reset_session=True` clears only intraday state:

- previous log close,
- pending predictions,
- last result.

It does not clear learned slot echoes, slot volume, or residual calibration.

## Prediction

On each update, the indicator forecasts the next `horizon_bars` by looking at
future slots:

```text
future_slot_j = (current_slot + j) % slots_per_session
```

Each future slot receives:

```text
reliability = min(1, slot_count[future_slot] / min_slot_samples)
weight = exp(-0.10 * (j - 1)) * reliability
```

The clock echo is the weighted average of future slot echoes:

```text
clock_echo = sum(weight_j * slot_echo[future_slot_j]) / sum(weight_j)
prediction = clock_echo * horizon_bars
```

The multiplication by `horizon_bars` turns an average per-slot residual return
into a horizon-level expected log return.

## Flow and Volume Adjustments

After the clock prediction is made, current bar flow can amplify or soften it:

```text
dollar_volume = close * max(volume, 0)
normal_dv = supplied normal_dollar_volume, or slot_volume[slot], or dollar_volume
signed_flow = sign(bar_return) * dollar_volume / normal_dv
flow_confirm = sign(prediction) * signed_flow
prediction *= 1 + 0.20 * tanh(flow_confirm)
```

If current dollar volume is radically different from the slot's normal dollar
volume, the prediction is dampened:

```text
volume_sync = log(dollar_volume / slot_volume[slot])
if abs(volume_sync) > 4:
    prediction *= 0.5
```

This prevents the same-clock pattern from being trusted too much during highly
abnormal activity.

## Error Band

When a prediction is ready, it is stored with:

- remaining bars until maturity,
- entry log close,
- prediction.

Each subsequent update decrements the pending horizon. When a prediction matures:

```text
realized = current_log_close - entry_log_close
realized_error = abs(realized - prediction)
```

The error is pushed into a rolling quantile. Once enough calibration samples
exist, radius is:

```text
radius = max(rolling_error_quantile, cost_buffer)
```

Before calibration is ready, fallback radius is based on the slot's absolute
residual return:

```text
radius = max(cost_buffer, slot_abs_err[slot] * horizon_bars)
```

## Readiness

`ready` is true only when:

- the relevant future slots have enough history to produce a weighted
  prediction,
- the rolling residual calibration has enough matured prediction errors.

If `fillna=False`, not-ready outputs are intentionally blanked:

- `prediction = NaN`
- `radius = NaN`
- `score = NaN`
- `signal = 0`
- `target_fraction = 0`

With `fillna=True`, fallback values are emitted earlier. That is useful for
experimentation, but live trading systems should usually care about `ready`.

## Trading Outputs

The score is:

```text
score = prediction / (radius + cost_buffer)
```

The signal is:

```text
signal = +1 if score > entry_z
signal = -1 if allow_short and score < -entry_z
signal =  0 otherwise
```

Target fraction is capped:

```text
target_fraction = max_abs_target_fraction * clamp(score / 3, side bounds)
```

`max_trade_dollars = participation_cap * normal_dollar_volume` is an execution
liquidity hint.

## Outputs

- `slot`: current time-of-day slot.
- `samples_for_slot`: number of historical observations for that slot.
- `bar_return`: current close-to-close log return.
- `residual_return`: bar return minus optional market return.
- `clock_echo`: weighted same-clock residual return forecast.
- `flow_confirm`: signed current flow in the direction of the prediction.
- `volume_sync`: current dollar volume relative to normal slot volume.
- `prediction`: horizon-level expected log return.
- `radius`: empirical uncertainty band.
- `score`: prediction divided by radius plus cost buffer.
- `signal`: `-1`, `0`, or `+1`.
- `target_fraction`: capped suggested exposure.
- `max_trade_dollars`: liquidity cap.
- `realized_error`: latest matured prediction error.
- `ready`: whether history and calibration are sufficient.

## Intended Use

Use this when bars have stable session slots. It is sensitive to missing bars,
half-days, and incorrect session resets because the time-of-day slot is the
feature. In the Massive/Polygon example, each symbol trains from prior day
aggregate bars, then the live day is scored by aligned window start.

The indicator should generally be paired with a market-return input when
available. Without market adjustment, broad intraday market moves can be learned
as if they were symbol-specific clock behavior.
