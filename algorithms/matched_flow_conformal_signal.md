# MatchedFlowConformalSignal

**Status:** proposed RTTA indicator / research prototype  
**Target module:** `rtta.indicator`  
**Suggested C++ class:** `MatchedFlowConformalSignal`  
**Suggested result struct:** `MatchedFlowConformalSignalResult`

`MatchedFlowConformalSignal` is a medium-speed technical trading signal for 5-minute OHLCV bars. It turns recent price movement, VWAP location, signed volume flow, relative volume, and an online error band into a simple trading output:

```text
prediction        estimated next-horizon log return
radius            adaptive recent error/noise band
score             prediction / uncertainty
signal            -1, 0, or +1
target_fraction   suggested portfolio fraction
max_trade_dollars liquidity cap for execution
```

The design is meant for programmers who already have market data and order APIs wired up, for example Polygon/Massive for bars and Alpaca for execution. The RTTA object should **not** place orders. It should only emit stateful technical-analysis outputs that an execution layer can consume.

This is **not** financial advice and not a guaranteed-profitable strategy. It is a clean, testable technical translation of several current research ideas into an incremental C++ indicator.

---

## One-sentence idea

Trade only when recent directional flow, momentum, and VWAP position suggest next-hour continuation **and** the predicted move is larger than the indicator's own recent prediction error.

In plain programmer terms:

```text
if prediction > recent_error_band + estimated_cost:
    emit LONG
elif prediction < -(recent_error_band + estimated_cost):
    emit SHORT
else:
    emit FLAT
```

The useful part is the `recent_error_band`. It prevents the indicator from trading every tiny noisy wiggle.

---

## Why this belongs in RTTA

RTTA is designed as a very low-latency incremental technical-analysis toolkit. Its README describes the package as avoiding batch-only pandas-style computation by updating indicators one sample at a time. New multi-output indicators are expected to expose `update(...)`, `advance(...)`, immutable result structs, scalar `update_<field>(...)` methods, and `last_<field>()` accessors.

`MatchedFlowConformalSignal` fits that pattern because it is stateful and incremental:

1. Each call consumes exactly one OHLCV bar.
2. It updates rolling state in O(1) or near-O(1) time.
3. It emits a result struct with multiple read-only fields.
4. It keeps old predictions internally and later compares them to realized returns.
5. It does not require pandas, Python callbacks, or a full ML model at runtime.

---

## Inputs

Recommended `update(...)` signature:

```cpp
MatchedFlowConformalSignalResult update(
    double open,
    double high,
    double low,
    double close,
    double volume,
    double normal_dollar_volume = NaN,
    double market_cap = NaN,
    bool reset_session = false
);
```

### Required inputs

| Input | Meaning |
|---|---|
| `open` | Bar open price. Currently included for API symmetry; not heavily used. |
| `high` | Bar high price. Used for a spread/noise proxy. |
| `low` | Bar low price. Used for a spread/noise proxy. |
| `close` | Bar close price. Main price input. |
| `volume` | Bar share volume. Used to compute dollar volume and flow. |

### Optional inputs

| Input | Meaning |
|---|---|
| `normal_dollar_volume` | Same-clock historical median/average dollar volume. Example: median dollar volume for this symbol at 10:35 over the last 20 sessions. Used for relative volume and execution caps. |
| `market_cap` | Current market capitalization. If present, signed flow can be normalized by market cap. If missing, the indicator falls back to normal-dollar-volume normalization. |
| `reset_session` | Set `true` on the first regular-session bar so VWAP, intraday lags, and pending intraday predictions reset cleanly. |

---

## Outputs

Suggested result struct:

```cpp
struct MatchedFlowConformalSignalResult {
    double prediction;
    double radius;
    double score;
    double signal;
    double target_fraction;

    double alpha_flow;
    double participation;
    double flow_score;
    double momentum;
    double volatility;
    double vwap_gap;
    double rel_dollar_volume;

    double max_trade_dollars;
    double realized_error;
};
```

### Main fields

| Field | Meaning |
|---|---|
| `prediction` | Estimated log return over the configured horizon, usually 12 bars = 60 minutes on 5-minute data. |
| `radius` | Recent prediction-error quantile. This is the noise/uncertainty band. |
| `score` | `prediction / (radius + cost_buffer)`. Larger absolute values mean stronger conviction. |
| `signal` | `+1` long, `0` flat, `-1` short. |
| `target_fraction` | Suggested portfolio fraction. Execution code can multiply this by account equity. |
| `max_trade_dollars` | Per-bar notional cap based on normal dollar volume and `participation_cap`. |

### Diagnostic fields

| Field | Meaning |
|---|---|
| `alpha_flow` | Signed dollar flow normalized for alpha extraction. Uses market cap when available. |
| `participation` | Signed dollar flow normalized by normal dollar volume. Useful for execution/liquidity context. |
| `flow_score` | Squashed rolling flow score. |
| `momentum` | Recent log-return momentum blend. |
| `volatility` | Recent return volatility. |
| `vwap_gap` | `(close / session_vwap) - 1`. Positive means price is above session VWAP. |
| `rel_dollar_volume` | Current dollar volume divided by normal same-clock dollar volume. |
| `realized_error` | Error from a matured old prediction, when available. |

---

## Algorithm walkthrough

Assume 5-minute bars and `horizon_bars = 12`, so the signal is trying to estimate the next 60 minutes.

### 1. Reset intraday state at the start of each session

On the first regular-session bar, call:

```cpp
sig.update(open, high, low, close, volume, normal_dollar_volume, market_cap, true);
```

This resets session VWAP, close lags, short rolling stats, and pending predictions. It should **not** wipe the longer calibration/error window unless the user calls `reset()`.

Why: the indicator is meant for intraday behavior. Letting yesterday's close or VWAP leak into today's first bars would make the signal harder to interpret.

---

### 2. Compute basic bar values

```text
dollar_volume = close * volume
ret_1         = log(close / previous_close)
signed_bar    = sign(ret_1)
```

`signed_bar` is a simple trade-direction proxy. If the bar closes up versus the previous bar, dollar volume is treated as positive flow. If it closes down, dollar volume is treated as negative flow.

This is less precise than quote-level order-flow imbalance, but it works with ordinary OHLCV bars.

---

### 3. Maintain session VWAP

```text
session_vwap = cumulative(close * volume) / cumulative(volume)
vwap_gap     = close / session_vwap - 1
```

Interpretation:

```text
vwap_gap > 0  -> price is above today's average traded price
vwap_gap < 0  -> price is below today's average traded price
```

This gives the signal a simple location feature. A strong upside flow while price is above VWAP is treated differently from a tiny bounce while price is still below VWAP.

---

### 4. Compute two signed-flow measures

The indicator keeps two related, but intentionally different, flow values.

#### Alpha flow

```text
alpha_flow = sign(ret_1) * dollar_volume / market_cap
```

If `market_cap` is missing:

```text
alpha_flow = sign(ret_1) * dollar_volume / normal_dollar_volume
```

Use this for directional signal extraction.

The market-cap version comes from the matched-filter order-flow idea: informed flow may scale with firm value, while raw daily trading volume can inject turnover noise. This is not proven for every intraday U.S. equity use case, so this feature should be tested with ablations.

#### Participation flow

```text
participation = sign(ret_1) * dollar_volume / normal_dollar_volume
```

Use this for execution/liquidity context.

The participation-style version says, "How large was this signed bar compared with normal activity for this time of day?"

---

### 5. Compute short momentum and volatility

Suggested rolling windows on 5-minute bars:

```text
mom_3  = log(close / close_3_bars_ago)    # 15 minutes
mom_6  = log(close / close_6_bars_ago)    # 30 minutes
mom_12 = log(close / close_12_bars_ago)   # 60 minutes

vol_12 = stddev(last 12 one-bar log returns)
```

Then blend momentum:

```text
momentum = 0.20 * mom_3 + 0.35 * mom_6 + 0.45 * mom_12
```

The exact weights are intentionally simple. They should be easy to tune and easy to explain.

---

### 6. Convert rolling flow into a bounded score

Example:

```text
flow_score = tanh(alpha_flow_12_sum / alpha_norm
                  + 0.50 * participation_6_sum / part_norm)
```

Why use `tanh`? It prevents one insane volume bar from dominating the indicator forever.

---

### 7. Estimate next-horizon return

A deliberately simple heuristic prediction:

```text
activity_score = tanh((rel_dollar_volume - 1.0) / 2.0)
spread_proxy   = max(0, high - low) / close
spread_dampen  = 1 / (1 + 25 * spread_proxy)

raw_prediction = 0.35   * momentum
               + 0.0010 * flow_score
               + 0.05   * vwap_gap
               + 0.0005 * activity_score

prediction = spread_dampen * raw_prediction
```

This is a technical-analysis approximation of a trained model. It is intentionally deterministic and fast.

The prediction is in log-return units. For example:

```text
prediction = 0.0020
```

means roughly:

```text
estimated next-hour return is +0.20%
```

---

### 8. Store the prediction and later measure its error

Every time the indicator emits a prediction, store:

```text
pending_prediction = { close_now, prediction_now }
```

After `horizon_bars` updates, compare it to the realized return:

```text
realized_return = log(close_now / old_prediction.close)
error           = abs(realized_return - old_prediction.prediction)
```

Push `error` into a rolling calibration window.

This is the key trick: the indicator continuously asks, "How wrong have I been recently?"

---

### 9. Build the conformal-style error band

Keep the most recent `calibration_window` absolute prediction errors. Then compute a high quantile, such as 80%:

```text
radius = quantile(abs_errors, 0.80)
```

This `radius` becomes the uncertainty band.

Interpretation:

```text
prediction = +0.0030
radius     =  0.0018
cost       =  0.0005
```

The predicted move is larger than recent model error plus estimated costs, so a long signal is allowed.

---

### 10. Generate signal

```text
denom = radius + cost_buffer
score = prediction / denom

if prediction > entry_z * denom:
    signal = +1
elif prediction < -entry_z * denom:
    signal = -1
else:
    signal = 0
```

Default values:

```text
entry_z = 1.0
cost_buffer = 0.0005
```

`cost_buffer = 0.0005` is approximately 5 basis points in log-return units. It should be replaced with your own estimate of spread, fees, slippage, and crossing costs.

---

### 11. Suggest a target size

```text
if signal != 0:
    target_fraction = max_abs_target_fraction * clamp(score / 3, -1, +1)
else:
    target_fraction = 0
```

Default:

```text
max_abs_target_fraction = 0.05
```

So the indicator will never suggest more than 5% account allocation per symbol by default.

---

### 12. Cap per-bar execution size

```text
max_trade_dollars = participation_cap * normal_dollar_volume
```

Default:

```text
participation_cap = 0.02
```

So if normal dollar volume for the current 5-minute slot is `$2,000,000`, the execution layer should not trade more than:

```text
0.02 * 2,000,000 = $40,000
```

in that bar.

The indicator does **not** know the account's current position. The execution layer should compute the delta:

```text
target_notional = account_equity * result.target_fraction
current_notional = current_shares * last_price
wanted_delta = target_notional - current_notional

order_notional = clamp(
    wanted_delta,
    -result.max_trade_dollars,
    +result.max_trade_dollars
)
```

---

## Default parameters

| Parameter | Suggested default | Meaning |
|---|---:|---|
| `horizon_bars` | `12` | Prediction horizon. On 5-minute bars, this is 60 minutes. |
| `calibration_window` | `250` | Number of matured errors used for the rolling error band. |
| `calibration_quantile` | `0.80` | Error quantile used as `radius`. Higher means fewer trades. |
| `entry_z` | `1.0` | Required prediction strength relative to `radius + cost`. |
| `cost_buffer` | `0.0005` | Estimated round-trip friction in return units. |
| `max_abs_target_fraction` | `0.05` | Max suggested allocation per symbol. |
| `participation_cap` | `0.02` | Max per-bar trading notional as fraction of normal dollar volume. |
| `fillna` | `false` | Follow RTTA convention: return NaNs until populated unless `fillna=true`. |

---

## Expected RTTA API surface

### C++ class

```cpp
class MatchedFlowConformalSignal {
public:
    MatchedFlowConformalSignal(
        std::size_t horizon_bars = 12,
        std::size_t calibration_window = 250,
        double calibration_quantile = 0.80,
        double entry_z = 1.0,
        double cost_buffer = 0.0005,
        double max_abs_target_fraction = 0.05,
        double participation_cap = 0.02,
        bool fillna = false
    );

    MatchedFlowConformalSignalResult update(
        double open,
        double high,
        double low,
        double close,
        double volume,
        double normal_dollar_volume = NaN,
        double market_cap = NaN,
        bool reset_session = false
    );

    void advance(...same args...);

    double update_prediction(...same args...);
    double update_radius(...same args...);
    double update_score(...same args...);
    double update_signal(...same args...);
    double update_target_fraction(...same args...);

    MatchedFlowConformalSignalResult last() const;
    double last_prediction() const;
    double last_radius() const;
    double last_score() const;
    double last_signal() const;
    double last_target_fraction() const;
    double last_max_trade_dollars() const;
    double last_realized_error() const;

    void reset();
    void reset_intraday();
};
```

### Python usage

```python
from rtta.indicator import MatchedFlowConformalSignal

sig = MatchedFlowConformalSignal(
    horizon_bars=12,
    calibration_window=250,
    calibration_quantile=0.80,
    entry_z=1.0,
    cost_buffer=0.0005,
    max_abs_target_fraction=0.05,
    participation_cap=0.02,
    fillna=True,
)

for bar in bars:
    result = sig.update(
        bar.open,
        bar.high,
        bar.low,
        bar.close,
        bar.volume,
        normal_dollar_volume=bar.normal_dollar_volume,
        market_cap=bar.market_cap,
        reset_session=bar.is_first_regular_session_bar,
    )

    if result.signal > 0:
        print("LONG", result.score, result.target_fraction, result.max_trade_dollars)
    elif result.signal < 0:
        print("SHORT", result.score, result.target_fraction, result.max_trade_dollars)
    else:
        print("FLAT", result.score)
```

---

## Execution-layer sketch

The indicator should not import Alpaca, Polygon/Massive, pandas, or any broker client. Keep it pure.

A broker layer can consume the output like this:

```python
target_notional = account_equity * result.target_fraction
current_notional = current_qty * latest_price

delta = target_notional - current_notional

delta = max(
    -result.max_trade_dollars,
    min(result.max_trade_dollars, delta),
)

shares = int(delta / latest_price)

if shares > 0:
    submit_buy(symbol, shares)
elif shares < 0:
    submit_sell(symbol, abs(shares))
```

Practical execution rules that should live outside the indicator:

1. Use paper trading first.
2. Block trading in the first few bars after the open until state is warmed up.
3. Flatten or reduce positions before market close if the strategy is intended to be intraday-only.
4. Enforce account-level exposure limits.
5. Enforce symbol-level borrow/shorting rules.
6. Handle broker errors, partial fills, rejected orders, and market halts.
7. Log every signal and order decision.

---

## What to test

### Unit tests

1. **NaN behavior**  
   With `fillna=false`, early outputs should be NaN for `prediction`, `radius`, and `score` until the object is populated.

2. **Session reset**  
   Calling `reset_session=true` should reset VWAP, close lags, pending predictions, and short rolling windows.

3. **Calibration maturation**  
   After `horizon_bars` updates, old predictions should mature and push absolute errors into the residual window.

4. **Signal threshold**  
   If `prediction <= radius + cost_buffer`, signal should be flat. If it exceeds the threshold, signal should be directional.

5. **Participation cap**  
   `max_trade_dollars` should equal `participation_cap * normal_dollar_volume` when normal dollar volume is supplied.

6. **Missing market cap fallback**  
   If `market_cap` is NaN, `alpha_flow` should fall back to normal-dollar-volume normalization.

### Backtest diagnostics

At minimum, log:

```text
symbol
timestamp
close
prediction
radius
score
signal
target_fraction
max_trade_dollars
realized_error
future_return
```

Then report:

```text
trade count
average holding period
gross PnL
net PnL after estimated costs
hit rate
average win
average loss
max drawdown
PnL by symbol
PnL by time of day
signal count by time of day
calibration coverage: percent(|future_return - prediction| <= radius)
```

The calibration coverage check is especially important. If `calibration_quantile=0.80`, then roughly 80% of matured absolute errors should fall inside the radius over a stable evaluation period. It will not be exact, especially under drift, but large deviations are a warning sign.

---

## Known limitations

1. **The C++ object is not the same as the Python ML model.**  
   The original idea used scikit-style training. This RTTA version is a deterministic streaming indicator inspired by that pipeline.

2. **Signed OHLCV flow is a proxy.**  
   `sign(ret_1) * dollar_volume` is not true order-flow imbalance. True OFI needs bid/ask quote updates and sizes.

3. **Market-cap normalization is experimental intraday.**  
   The matched-filter paper tests Korean stock-day observations. Applying the idea to 5-minute U.S. bars is a research hypothesis, not a proven fact.

4. **The error band is conformal-style, not a formal guarantee.**  
   Rolling residual quantiles borrow the conformal calibration idea, but the simplified indicator does not implement the full theory from the cited papers.

5. **Costs can dominate.**  
   If spread/slippage are larger than `cost_buffer`, the signal can look good in backtests and fail live.

6. **Regime shifts still hurt.**  
   The rolling error band adapts, but abrupt news, halts, earnings, Fed events, or market-wide shocks can still break it.

---

## Suggested ablations

Ablations are the fastest way to find out whether the indicator is doing anything real.

1. **No conformal filter**  
   Trade all nonzero predictions. Compare against the filtered version.

2. **No market-cap normalization**  
   Replace `alpha_flow` with participation-style flow only.

3. **Momentum only**  
   Remove flow, VWAP gap, and activity score.

4. **Flow only**  
   Remove momentum and VWAP gap.

5. **Different horizons**  
   Test `horizon_bars = 6`, `9`, `12`, and `18`.

6. **Different quantiles**  
   Test `calibration_quantile = 0.70`, `0.80`, `0.90`.

7. **Time-of-day filters**  
   Compare all-day trading against skipping the open and closing hour.

---

## Documents and research references

### RTTA implementation references

| Reference | Why it matters |
|---|---|
| [RTTA branch `0.2.0`](https://github.com/adamdeprince/rtta/tree/0.2.0) | Target repository and branch. |
| [RTTA README](https://raw.githubusercontent.com/adamdeprince/rtta/0.2.0/README.md) | Describes RTTA's incremental update goal, C++23/nanobind build, `fillna` convention, and indicator API conventions. |
| [RTTA `ALGOS.md`](https://raw.githubusercontent.com/adamdeprince/rtta/0.2.0/ALGOS.md) | Public indicator list where this algorithm should be added. |
| [RTTA `indicator.cpp`](https://raw.githubusercontent.com/adamdeprince/rtta/0.2.0/src/rtta/indicator.cpp) | Existing C++/nanobind indicator implementation file. |
| [RTTA `__init__.py`](https://raw.githubusercontent.com/adamdeprince/rtta/0.2.0/src/rtta/__init__.py) | Python export file where new bindings may need to be exposed. |
| [nanobind class docs](https://nanobind.readthedocs.io/en/latest/classes.html) | Binding style for exposing C++ classes to Python. |

### Market data and execution references

| Reference | Why it matters |
|---|---|
| [Massive/Polygon Python client](https://github.com/massive-com/client-python) | Official Python client for Polygon.io/Massive data; useful for OHLCV bar retrieval in external data pipelines. |
| [Massive/Polygon aggregate bars docs](https://github.com/polygon-io/client-python/blob/master/docs/source/Aggs.rst) | Documents `RESTClient.list_aggs`, useful for generating 5-minute bars. |
| [Alpaca-py order API](https://alpaca.markets/sdks/python/api_reference/trading/orders.html) | Documents `TradingClient.submit_order(...)` for a separate execution layer. |
| [Alpaca working with orders](https://docs.alpaca.markets/docs/working-with-orders) | Practical order-management documentation. |

### Original and supporting research

| Reference | Design idea used here |
|---|---|
| [The Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting](https://arxiv.org/abs/2602.03395) | Motivates testing intermediate target horizons instead of assuming the final trading horizon is the best training label. This matters more for the Python ML version, but it informs why the C++ default horizon should be tunable. |
| [Optimal Signal Extraction from Order Flow: A Matched Filter Perspective on Normalization and Market Microstructure](https://arxiv.org/abs/2512.18648) | Motivates separating market-cap-normalized flow for signal extraction from volume-normalized participation for execution/liquidity context. |
| [Temporal Conformal Prediction: A Distribution-Free Statistical and Machine Learning Framework for Adaptive Risk Forecasting](https://arxiv.org/abs/2507.05470) | Motivates rolling/adaptive calibration of prediction uncertainty in nonstationary financial time series. |
| [Taming Tail Risk in Financial Markets: Conformal Risk Control for Nonstationary Portfolio VaR](https://arxiv.org/abs/2602.03903) | Motivates recency-weighted and regime-aware conformal calibration. The current indicator uses a simpler rolling quantile version. |
| [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170) | General foundation for online conformal-style adaptation under changing distributions. |
| [Online Conformal Model Selection for Nonstationary Time Series](https://arxiv.org/abs/2506.05544) | Supports the broader idea that static model choice can be brittle under nonstationarity. This is more relevant to a future multi-expert version. |
| [Forecasting Intraday Volume in Equity Markets with Machine Learning](https://arxiv.org/abs/2505.08180) | Motivates using expected intraday volume for execution caps and later replacing `normal_dollar_volume` with a learned volume forecast. |
| [The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402) | Classic order-flow imbalance reference; useful if this OHLCV indicator is later upgraded to quote-level OFI. |
| [Order-Flow Filtration and Directional Association with Short-Horizon Returns](https://arxiv.org/abs/2507.22712) | Motivates filtering noisy high-frequency order-flow signals before using them directionally. Useful for a quote-level extension. |

---

## Future extensions

### 1. Quote-level OFI version

Replace OHLCV signed flow:

```text
sign(ret_1) * dollar_volume
```

with true order-flow imbalance computed from best bid/ask price and size updates.

This would require a different update signature, for example:

```cpp
update_quote(
    double bid_price,
    double bid_size,
    double ask_price,
    double ask_size,
    double trade_price,
    double trade_size,
    ...
)
```

### 2. Weighted conformal radius

Replace the simple rolling quantile with exponentially weighted residuals:

```text
weight_i = exp(-age_i / tau)
radius   = weighted_quantile(errors, weights, calibration_quantile)
```

This would adapt faster after regime shifts.

### 3. Regime-aware radius

Add a simple regime feature:

```text
regime = recent_volatility / long_volatility
```

Then weight old residuals by both age and regime similarity:

```text
weight_i = exp(-age_i / tau) * exp(-gamma * abs(regime_i - regime_now))
```

This is closer to the regime-weighted conformal risk-control research.

### 4. Learned offline coefficients

The current coefficients are fixed:

```text
0.35, 0.0010, 0.05, 0.0005
```

A future version could expose them as constructor parameters or fit them offline in Python, then pass them into the C++ indicator.

### 5. Volume forecast input

Instead of passing same-clock median dollar volume, pass a forecast from a separate model:

```text
normal_dollar_volume = predicted_next_bar_or_next_hour_dollar_volume
```

This makes `max_trade_dollars` more responsive to unusual trading days.

---

## Suggested `ALGOS.md` row

```markdown
| `MatchedFlowConformalSignal` | Medium-speed OHLCV signal combining intraday momentum, VWAP gap, signed flow, relative volume, and an online residual/error band to emit prediction, score, direction, target fraction, and execution cap. | https://arxiv.org/abs/2507.05470 |
```

---

## Suggested Codex task

```text
Add a new RTTA indicator named MatchedFlowConformalSignal.

Use the design in docs/MatchedFlowConformalSignal.md.

Requirements:
1. Add MatchedFlowConformalSignalResult and MatchedFlowConformalSignal to src/rtta/indicator.cpp.
2. Follow existing RTTA multi-output indicator conventions.
3. Expose update(...), advance(...), update_prediction(...), update_radius(...), update_score(...), update_signal(...), update_target_fraction(...).
4. Expose last(), last_prediction(), last_radius(), last_score(), last_signal(), last_target_fraction(), last_max_trade_dollars(), last_realized_error().
5. Add Python bindings with nanobind.
6. Export the names from src/rtta/__init__.py if required by the current package layout.
7. Add an ALGOS.md row.
8. Add unit tests for warmup/fillna behavior, session reset, residual maturation, signal thresholding, and participation caps.
9. Do not add broker/execution code inside the indicator.
10. Do not add pandas or Python callbacks to the C++ hot path.
```
