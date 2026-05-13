# MatchedFlowConformalSignal

`MatchedFlowConformalSignal` is an intraday OHLCV signal that tries to trade
only when directional price movement, signed volume flow, VWAP location, and
relative activity point in the same direction and the expected move is larger
than a recent empirical error band.

It is a composite research prototype rather than a direct implementation of one
paper. The main references are:

- Chordia, Roll, and Subrahmanyam, "Order imbalance, liquidity, and market
  returns," Journal of Financial Economics, 2002. The paper studies order
  imbalance as a trading-activity measure and links imbalance to liquidity and
  market returns. Public metadata: https://authors.library.caltech.edu/95132/
- Chordia and Subrahmanyam, "Order imbalance and individual stock returns:
  Theory and evidence," Journal of Financial Economics, 2004. The paper studies
  how autocorrelated imbalances can create price pressure in individual stocks.
  SSRN page: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=354122
- Xu and Xie, "Sequential Predictive Conformal Inference for Time Series,"
  arXiv:2212.03463, 2022. The paper motivates sequential calibration of
  nonconformity scores for non-exchangeable time series. arXiv/HF page:
  https://huggingface.co/papers/2212.03463

The RTTA implementation is deliberately simpler than the papers. It does not
estimate true order imbalance from signed trades or quotes. Instead, it uses the
bar close-to-close return sign as a cheap proxy for signed flow and scales that
by dollar volume.

## API

```python
rtta.MatchedFlowConformalSignal(
    horizon_bars=12,
    calibration_window=250,
    calibration_quantile=0.80,
    entry_z=1.0,
    cost_buffer=0.0005,
    max_abs_target_fraction=0.05,
    participation_cap=0.02,
    fillna=False,
)
```

`update(open, high, low, close, volume, normal_dollar_volume=nan,
market_cap=nan, reset_session=False)` consumes one aggregate bar. The example
script uses Massive/Polygon stock trade aggregates, but any regular OHLCV bar
stream can be used.

## State

The indicator maintains:

- close lags for 3, 6, and 12 bar momentum,
- rolling return volatility over 6 and 12 bars,
- rolling sums of `alpha_flow` over 3, 6, and 12 bars,
- rolling sums of participation flow over 3 and 6 bars,
- an intraday session VWAP built from bar VWAP or close times volume,
- an EWMA fallback for normal dollar volume,
- a fixed-size queue of pending predictions,
- a rolling quantile of realized prediction errors.

`reset_session=True` clears intraday state such as close lags, flow windows,
pending predictions, and session VWAP. It does not clear the longer-lived
normal-dollar-volume EWMA or the calibration residual quantile.

## Feature Construction

For each valid bar:

1. Compute `dollar_volume = close * max(volume, 0)`.
2. Choose `normal_dollar_volume` from the supplied value, the internal EWMA, or
   current dollar volume.
3. Compute close-to-close log return `ret1`.
4. Use `sign(ret1)` as the signed-flow direction.
5. Update session VWAP from cumulative price-volume and volume.
6. Compute:
   - `vwap_gap = close / session_vwap - 1`
   - `rel_dollar_volume = dollar_volume / normal_dollar_volume`
   - `alpha_flow = sign(ret1) * dollar_volume / market_cap`, when market cap is
     supplied, otherwise scaled by normal dollar volume
   - `participation = sign(ret1) * dollar_volume / normal_dollar_volume`
7. Clamp `alpha_flow` and `participation` to `[-10, 10]`.
8. Push these values into the rolling flow windows.

Momentum is a weighted combination of 3, 6, and 12 bar log returns:

```text
momentum = 0.20 * ret_3 + 0.35 * ret_6 + 0.45 * ret_12
```

`flow_score` compresses accumulated signed flow through `tanh`, with a different
normalization depending on whether `market_cap` is supplied:

```text
flow_score = tanh(alpha_flow_12_sum / alpha_norm
                 + 0.50 * participation_6_sum / 6)
```

Relative activity is also squashed:

```text
activity_score = tanh((rel_dollar_volume - 1) / 2)
```

The high-low range dampens predictions in wide-spread/noisy bars:

```text
spread_proxy = max(0, high - low) / close
spread_dampen = 1 / (1 + 25 * spread_proxy)
```

## Prediction

The raw next-horizon log-return forecast is:

```text
raw_prediction =
    0.35   * momentum
  + 0.0010 * flow_score
  + 0.05   * vwap_gap
  + 0.0005 * activity_score

prediction = spread_dampen * raw_prediction
```

This is intentionally heuristic. The goal is not to fit a model in the
indicator; the goal is to expose a stable, incremental signal that reacts only
when several intraday conditions line up.

## Error Band

Every prediction is stored with its entry close. After `horizon_bars` updates,
the indicator compares the realized log return to the stored prediction:

```text
realized = log(current_close / old_close)
error = abs(realized - old_prediction)
```

Errors are pushed into a rolling quantile. Once enough errors are available,
`radius` becomes:

```text
radius = max(rolling_error_quantile, cost_buffer)
```

Before calibration is ready, it uses a fallback:

```text
radius = max(2 * cost_buffer,
             max(0.0007, 1.25 * volatility_12 * sqrt(horizon_bars)))
```

This is conformal-inspired but not a formal finite-sample conformal guarantee.
It is an online empirical error band.

## Trading Outputs

The score is:

```text
score = prediction / (radius + cost_buffer)
```

The signal is:

```text
signal = +1 if prediction > entry_z * (radius + cost_buffer)
signal = -1 if prediction < -entry_z * (radius + cost_buffer)
signal =  0 otherwise
```

The target fraction is capped:

```text
target_fraction = max_abs_target_fraction * clamp(score / 3, -1, 1)
```

`max_trade_dollars = participation_cap * normal_dollar_volume` is a liquidity
cap for an external execution layer.

## Outputs

- `prediction`: expected next-horizon log return.
- `radius`: empirical uncertainty/error band.
- `score`: prediction divided by radius plus cost buffer.
- `signal`: `-1`, `0`, or `+1`.
- `target_fraction`: capped suggested exposure.
- `alpha_flow`: signed dollar volume scaled by market cap or normal dollar
  volume.
- `participation`: signed dollar volume scaled by normal dollar volume.
- `flow_score`: compressed recent signed-flow measure.
- `momentum`: weighted multi-bar momentum.
- `volatility`: rolling 12-bar return volatility.
- `vwap_gap`: close relative to session VWAP.
- `rel_dollar_volume`: current dollar volume relative to normal dollar volume.
- `max_trade_dollars`: liquidity cap.
- `realized_error`: newest matured prediction error, when available.

## Intended Use

Use this on liquid intraday bars, typically 1 to 5 minutes. Reset it at session
boundaries. It works best as a ranking or gating feature inside a broader system
that also handles spreads, fees, inventory, exposure, and execution.
