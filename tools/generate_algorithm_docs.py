#!/usr/bin/env python3
"""Generate per-algorithm Markdown documentation stubs for RTTA.

The hand-written pages in documentation/algorithms are treated as authoritative
and are not overwritten. Missing pages are generated from ALGOS.md,
benchmarks/benchmark_indicators.py, and the C++ class names in
src/rtta/indicator.cpp.
"""

from __future__ import annotations

import ast
import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ALGOS = ROOT / "ALGOS.md"
DOCS = ROOT / "documentation" / "algorithms"
BENCHMARK_REGISTRY = ROOT / "benchmarks" / "benchmark_indicators.py"
CPP_SOURCE = ROOT / "src" / "rtta" / "indicator.cpp"
HAND_WRITTEN = {"ATR", "EMA", "MACD", "RSI", "SMA"}


@dataclass(frozen=True)
class Algo:
    name: str
    description: str
    inputs: tuple[str, ...]
    ctor_kwargs: dict[str, object]
    reference: str | None


def split_identifier(name: str) -> list[str]:
    return re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+", name)


def slugify(name: str) -> str:
    return "-".join(part.lower() for part in split_identifier(name))


def parse_current_algos() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in ALGOS.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\| (?:\[`([^`]+)`\]\([^)]*\)|`([^`]+)`) \| (.*?) \|$", line)
        if match:
            rows.append((match.group(1) or match.group(2), match.group(3)))
    return rows


def parse_reference_links() -> dict[str, str]:
    def extract_reference(text: str) -> str | None:
        match = re.search(r"^## Reference\s*\n\n(.*?)(?:\n## |\Z)", text, flags=re.S | re.M)
        if not match:
            return None
        body = match.group(1).strip()
        if "Source implementation:" in body:
            return None
        link = re.search(r"\[[^\]]+\]\(([^)]+)\)", body)
        return link.group(1).strip() if link else None

    try:
        result = subprocess.run(
            ["git", "show", "HEAD:ALGOS.md"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}

    references: dict[str, str] = {}
    for line in result.stdout.splitlines():
        match = re.match(r"\| `([^`]+)` \| .*? \| (.*?) \|$", line)
        if match:
            ref = match.group(2).strip()
            if ref:
                references[match.group(1)] = ref

    for path in DOCS.glob("*.md"):
        if path.name == "README.md" or path.name.endswith(".zh-CN.md"):
            continue
        try:
            result = subprocess.run(
                ["git", "show", f"HEAD:{path.relative_to(ROOT).as_posix()}"],
                check=True,
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            text = result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            text = path.read_text(encoding="utf-8")
        title = re.search(r"^#\s+(.+?)\s*$", text, flags=re.M)
        ref = extract_reference(text)
        if title and ref:
            references.setdefault(title.group(1).strip().strip("`"), ref)
    return references


def literal(node: ast.AST) -> object:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def parse_registry() -> dict[str, tuple[tuple[str, ...], dict[str, object]]]:
    tree = ast.parse(BENCHMARK_REGISTRY.read_text(encoding="utf-8"))
    registry: dict[str, tuple[tuple[str, ...], dict[str, object]]] = {}
    for node in ast.walk(tree):
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "INDICATORS" for target in node.targets
        ):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "INDICATORS":
            value = node.value
        if value is None:
            continue
        if not isinstance(value, ast.Tuple):
            continue
        for item in value.elts:
            if not isinstance(item, ast.Call) or not item.args:
                continue
            name = literal(item.args[0])
            raw_inputs = literal(item.args[1]) if len(item.args) > 1 else ()
            if not isinstance(name, str) or not isinstance(raw_inputs, tuple):
                continue
            inputs = tuple(str(value) for value in raw_inputs)
            ctor_kwargs: dict[str, object] = {}
            for keyword in item.keywords:
                if keyword.arg == "ctor_kwargs":
                    value = literal(keyword.value)
                    if isinstance(value, dict):
                        ctor_kwargs = {str(k): v for k, v in value.items()}
            registry[name] = (inputs, ctor_kwargs)
    return registry


def existing_doc_pages() -> dict[str, Path]:
    pages: dict[str, Path] = {}
    for path in DOCS.glob("*.md"):
        if path.name == "README.md" or path.name.endswith(".zh-CN.md"):
            continue
        text = path.read_text(encoding="utf-8")
        match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.M)
        if match:
            pages[match.group(1).strip().strip("`")] = path
    return pages


def parse_result_fields() -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    source = CPP_SOURCE.read_text(encoding="utf-8")
    for struct_name, body in re.findall(r"struct\s+([A-Za-z0-9_]+Result)\s*\{(.*?)\};", source, flags=re.S):
        if struct_name.endswith("BatchResult"):
            continue
        algo = struct_name.removesuffix("Result")
        names = re.findall(r"(?:double|bool|int|std::size_t)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|;)", body)
        if names:
            fields[algo] = names
    return fields


def parse_cpp_classes() -> set[str]:
    return set(re.findall(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\b", CPP_SOURCE.read_text(encoding="utf-8"), flags=re.M))


def ctor_text(kwargs: dict[str, object]) -> str:
    if not kwargs:
        return ""
    parts = []
    for key, value in kwargs.items():
        parts.append(f"{key}={value!r}".replace("'", '"') if isinstance(value, str) else f"{key}={value!r}")
    return ", " + ", ".join(parts)


def input_tuple(inputs: tuple[str, ...]) -> str:
    if not inputs:
        return "x_t"
    if len(inputs) == 1:
        return f"{inputs[0]}_t"
    return "(" + ", ".join(f"{name}_t" for name in inputs) + ")"


def update_call(name: str, inputs: tuple[str, ...], kwargs: dict[str, object]) -> str:
    args = ", ".join(inputs) if inputs else "value"
    return f"result = rtta.{name}({ctor_text(kwargs).lstrip(', ')}).update({args})" if kwargs else f"result = rtta.{name}().update({args})"


def theory(name: str, description: str) -> str:
    lower = f"{name} {description}".lower()
    if name == "MatchedFlowConformalSignal":
        return (
            "`MatchedFlowConformalSignal` is a composite intraday research signal. "
            "It forms a horizon return forecast from multi-scale momentum, signed "
            "dollar-flow participation, VWAP displacement, and abnormal activity; "
            "then it scales that forecast by a rolling empirical error quantile. "
            "The longer research note linked below discusses the paper lineage and "
            "the conformal-style calibration caveats."
        )
    if name == "ClosePressureReversalSignal":
        return (
            "`ClosePressureReversalSignal` converts the end-of-day reversal idea into "
            "a causal bar stream. It freezes the session return at a configured cutoff, "
            "normalizes loser/winner pressure by realized intraday volatility, adjusts "
            "for volume, transactions, and VWAP location, and only emits entries during "
            "the late-session window. The longer research note linked below gives the "
            "empirical motivation and parameter interpretation."
        )
    if name == "IntradayClockEchoSignal":
        return (
            "`IntradayClockEchoSignal` learns time-of-day residual return patterns by "
            "slot. Each update removes an optional market return, updates the EWMA "
            "state for the current slot, forecasts a future slot path over the horizon, "
            "and calibrates forecast errors with a rolling quantile. The linked research "
            "note explains the same-clock effect and session-alignment assumptions."
        )
    if name in {"ADWIN", "FeatureDistributionDriftDetector"}:
        return (
            f"`{name}` maintains an adaptive recent window and searches every admissible "
            "split for a statistically meaningful difference between the old and new "
            "subwindow means. The signal direction is the sign of the best accepted mean "
            "shift; accepting a split discards the older prefix."
        )
    if name in {"DDM", "EDDM", "HDDM"}:
        return (
            f"`{name}` is a streaming classifier-error drift detector. It treats positive "
            "input values as errors and compares the current error process against the "
            "best historical baseline using the detector's bound: binomial standard error "
            "for DDM, distance-between-errors degradation for EDDM, and a Hoeffding bound "
            "for HDDM."
        )
    if name == "KSWIN":
        return (
            "`KSWIN` compares the empirical distribution of a recent subwindow with the "
            "older reference portion of the rolling window using the Kolmogorov-Smirnov "
            "supremum distance. The output direction is determined by which subwindow has "
            "the larger mean when the KS statistic clears its critical value."
        )
    if name == "PageHinkley":
        return (
            "`PageHinkley` tracks cumulative positive and negative deviations from an "
            "online mean after subtracting a small drift allowance. A signal fires when "
            "one cumulative excursion rises far enough above its own running minimum; "
            "the detector then resets to the current close."
        )
    if name.startswith("Rolling") and "ShiftDetector" in name:
        return (
            f"`{name}` compares two adjacent rolling windows: a reference window and a "
            "recent window. The C++ state moves expired recent samples into the reference "
            "window, maintains sufficient statistics, and emits the sign of the statistic "
            "difference when it exceeds the configured threshold."
        )
    if name in {"EWMAZScoreShiftDetector", "ResidualDriftDetector", "PredictionErrorDriftDetector", "VolatilityBreakoutDetector"}:
        return (
            f"`{name}` standardizes the current error or move against an EWMA mean and "
            "variance estimated from prior samples. The detector uses the resulting z-score "
            "with hysteresis or reset logic so isolated noisy observations do not become "
            "persistent regimes by themselves."
        )
    if name == "ZigZagSwingDetector":
        return (
            "`ZigZagSwingDetector` maintains the current swing direction, the active "
            "extreme, and the last confirmed pivot. A new pivot is confirmed only after "
            "price reverses from the active extreme by the configured percentage, which "
            "filters smaller oscillations out of the swing path."
        )
    if name in {"ATRP", "NormalizedATR"}:
        return (
            f"`{name}` expresses the current `ATR` volatility estimate relative to the "
            "current close. This makes a dollar-denominated range comparable across price "
            "levels while preserving the same one-step Wilder true-range smoothing used by "
            "`ATR`."
        )
    if name == "BollingerBands":
        return (
            "`BollingerBands` wraps a rolling `SMA` with a rolling `StdDev` envelope. "
            "The middle band is the local mean, while the upper and lower bands mark a "
            "two-standard-deviation dispersion channel around that mean."
        )
    if name == "SuperTrend":
        return (
            "`SuperTrend` builds ATR-scaled bands around the high/low midpoint and trails "
            "the active band in the direction of the current trend. Crosses through the "
            "active band flip the trend side; otherwise the band only tightens, which makes "
            "the indicator a volatility-adjusted trailing stop."
        )
    if name in {
        "PlusDirectionalMovement",
        "MinusDirectionalMovement",
        "PlusDirectionalIndicator",
        "MinusDirectionalIndicator",
        "DirectionalMovementIndex",
        "AverageDirectionalMovementIndex",
        "AverageDirectionalMovementIndexRating",
    }:
        return (
            f"`{name}` is part of Wilder's directional-movement system. The update compares "
            "today's high/low extension with the previous bar, smooths directional movement "
            "and true range, and then reports either a directional component, a normalized "
            "directional imbalance, or an additional Wilder-smoothed trend-strength rating."
        )
    if name == "MassIndex":
        return (
            "`MassIndex` studies range expansion rather than direction. It double-smooths "
            "the high-low range with `EMA`, forms their ratio, and sums that ratio over a "
            "window so persistent range bulges become visible as a reversal-risk signal."
        )
    if name == "CrossAssetCorrelationBreakDetector":
        return (
            "`CrossAssetCorrelationBreakDetector` runs short and long rolling correlation "
            "estimates on the same pair of streams and measures their absolute divergence. "
            "An upper hysteresis state turns that divergence into a persistent break flag "
            "until the short/long correlations reconverge below the exit threshold."
        )
    if name in {
        "ThresholdRegimeDetector",
        "VolatilityRegimeDetector",
        "ATRRegimeDetector",
        "RealizedVarianceRegimeDetector",
        "TrendChopRegimeDetector",
        "LiquidityRegimeDetector",
        "SpreadRegimeDetector",
        "VolumeRegimeDetector",
        "TradeIntensityRegimeDetector",
        "QuoteMessageRateRegimeDetector",
        "OrderFlowImbalanceRegimeDetector",
        "CorrelationRegimeDetector",
        "BetaRegimeDetector",
        "PairsSpreadRegimeDetector",
        "ExecutionCostSlippageRegimeDetector",
        "VolatilityCompressionExpansionDetector",
        "MicrostructureNoiseRegimeDetector",
        "BidAskBounceRegimeDetector",
        "QuoteStuffingDetector",
        "LeadLagRegimeDetector",
        "LiquidityDroughtDetector",
        "SpreadExplosionDetector",
        "MarketOpenCloseTransitionDetector",
        "AuctionContinuousMarketTransitionDetector",
        "CrossAssetCorrelationBreakDetector",
        "HitRateDriftDetector",
        "CalibrationDriftDetector",
    }:
        return (
            f"`{name}` first constructs a scalar market-state metric from the current "
            "observation and compact streaming state, then passes that metric through "
            "explicit entry/exit hysteresis. The metric is named in the recurrence below; "
            "the hysteresis keeps the output stable until the metric crosses the opposite "
            "exit band."
        )
    if "kalman" in lower:
        return (
            f"`{name}` treats the input stream as noisy observations of a latent state. "
            "Each call performs the standard predict/update cycle, then projects the "
            "updated state into the public scalar or result fields."
        )
    if "regime" in lower or "detector" in lower or "drift" in lower or "hysteresis" in lower:
        return (
            f"`{name}` converts each observation into a streaming score and then applies "
            "threshold or hysteresis logic. The state is deliberately sticky where the "
            "C++ class models regimes, so small reversals do not immediately flip the output."
        )
    if "hmm" in lower or "markov" in lower or "mixture" in lower:
        return (
            f"`{name}` maintains online probabilities for latent states or components. "
            "An update combines the previous probabilities with the new observation "
            "likelihoods and normalizes the result."
        )
    if "moving average" in lower or name.endswith("EMA") or "average" in lower or "weighted" in lower:
        return (
            f"`{name}` is a causal smoother or average. It updates compact rolling or "
            "exponential state with the newest observation and returns the current smoothed estimate."
        )
    if "oscillator" in lower or "rsi" in lower or "stochastic" in lower:
        return (
            f"`{name}` normalizes recent directional movement into an oscillator. "
            "The implementation keeps only causal rolling or smoothed state and maps "
            "that state into the current oscillator value."
        )
    if "volume" in lower or "liquidity" in lower or "spread" in lower or "order-flow" in lower:
        return (
            f"`{name}` combines price, volume, and/or quote information into a streaming "
            "microstructure or participation measure. The update path advances only from "
            "the latest tick and prior state."
        )
    if "correlation" in lower or "beta" in lower or "variance" in lower or "regression" in lower:
        return (
            f"`{name}` keeps rolling sufficient statistics for the requested statistical "
            "quantity. Each update inserts the newest sample, removes any expired sample, "
            "and recomputes the current statistic from those maintained sums."
        )
    if "high" in lower or "low" in lower or "channel" in lower or "band" in lower:
        return (
            f"`{name}` maintains rolling extrema, ranges, or envelopes. The C++ state "
            "updates the relevant window/range statistics once per input sample."
        )
    return (
        f"`{name}` implements the streaming form of {description.rstrip('.')}. "
        "Each `update(...)` call consumes exactly one new observation tuple and advances "
        "the internal state before returning the current value or result struct."
    )


def two_sided_hysteresis(metric: str, note: str = "") -> str:
    note_block = f"\n\n{note}" if note else ""
    return f"""{metric}{note_block}

\\[
r_t =
\\begin{{cases}}
1, & r_{{t-1}} \\le 0 \\text{{ and }} q_t \\ge u_e \\\\
0, & r_{{t-1}} = 1 \\text{{ and }} q_t \\le u_x \\\\
-1, & r_{{t-1}} \\ge 0 \\text{{ and }} q_t \\le \\ell_e \\\\
0, & r_{{t-1}} = -1 \\text{{ and }} q_t \\ge \\ell_x \\\\
r_{{t-1}}, & \\text{{otherwise}}
\\end{{cases}}
\\]

The entry/exit constants satisfy \\(\\ell_e < \\ell_x \\le u_x < u_e\\)."""


def upper_hysteresis(metric: str, note: str = "") -> str:
    note_block = f"\n\n{note}" if note else ""
    return f"""{metric}{note_block}

\\[
r_t =
\\begin{{cases}}
1, & r_{{t-1}} = 0 \\text{{ and }} q_t \\ge e \\\\
0, & r_{{t-1}} = 1 \\text{{ and }} q_t \\le x \\\\
r_{{t-1}}, & \\text{{otherwise}}
\\end{{cases}}, \\qquad x < e
\\]"""


def lower_hysteresis(metric: str, note: str = "") -> str:
    note_block = f"\n\n{note}" if note else ""
    return f"""{metric}{note_block}

\\[
r_t =
\\begin{{cases}}
1, & r_{{t-1}} = 0 \\text{{ and }} q_t \\le e \\\\
0, & r_{{t-1}} = 1 \\text{{ and }} q_t \\ge x \\\\
r_{{t-1}}, & \\text{{otherwise}}
\\end{{cases}}, \\qquad e < x
\\]"""


def threshold_signal(metric: str, note: str = "", threshold: str = "h") -> str:
    note_block = f"\n\n{note}" if note else ""
    return f"""{metric}{note_block}

\\[
r_t =
\\begin{{cases}}
1, & q_t > {threshold} \\\\
-1, & q_t < -{threshold} \\\\
0, & \\text{{otherwise}}
\\end{{cases}}
\\]"""


def recurrence(name: str, description: str, inputs: tuple[str, ...]) -> str:
    lower = f"{name} {description}".lower()
    z = input_tuple(inputs)
    if name == "MatchedFlowConformalSignal":
        return r"""\[
r^{(k)}_t=\log(close_t/close_{t-k}), \qquad
m_t=0.20r^{(3)}_t+0.35r^{(6)}_t+0.45r^{(12)}_t
\]

\[
DV_t=close_t\max(volume_t,0), \qquad
relvol_t=\frac{DV_t}{normal\_dollar\_volume_t}
\]

\[
a_t=\operatorname{sgn}(r^{(1)}_t)
\frac{close_t\,\max(volume_t,0)}{\operatorname{scale}_t}, \qquad
p_t=\operatorname{sgn}(r^{(1)}_t)
\frac{close_t\,\max(volume_t,0)}{normal\_dollar\_volume_t}
\]

Here \(\operatorname{scale}_t\) is market capitalization when supplied and the
normal dollar-volume baseline otherwise.

\[
flow_t=\tanh\left(\frac{\sum_{i\in W^{12}_t}a_i}{\alpha_{norm}}
+0.5\frac{\sum_{i\in W^{6}_t}p_i}{6}\right), \qquad
vwap\_gap_t=\frac{close_t}{VWAP_t}-1
\]

\[
\widehat{r}_{t+h}=
\frac{0.35m_t+0.001flow_t+0.05vwap\_gap_t
+0.0005\tanh((relvol_t-1)/2)}
{1+25\max(high_t-low_t,0)/close_t}
\]

\[
\mathcal{E}_t=\{|r^{(h)}_i-\widehat{r}^{(h)}_i|:\ i+h\le t\}, \qquad
radius_t=\max(Q_{\tau}(\mathcal{E}_t), cost)
\]

\[
score_t=\frac{\widehat{r}_{t+h}}{radius_t+cost}
\]"""
    if name == "ClosePressureReversalSignal":
        return r"""\[
ROD_t=\log(close_t)-\log(anchor), \qquad
F_t=ROD_{t_c}
\]

\[
DV_t=close_t\max(volume_t,0), \qquad
vwap\_gap_t=\frac{close_t}{VWAP_t}-1
\]

\[
\sigma_{intra,t}=\sqrt{N_{t_c}\operatorname{Var}(r^{(1)}_1,\ldots,r^{(1)}_{t_c})},
\qquad
L_t=\frac{\max(0,-F_t)}{\sigma_{intra,t}}, \quad
W_t=\frac{\max(0,F_t)}{\sigma_{intra,t}}
\]

\[
M^V_t=1+0.20\,\operatorname{clip}(\log(DV_t/NDV_t),-2,4), \qquad
M^X_t=1+0.10\,\operatorname{clip}(\log(X_t/NX_t),-2,4)
\]

\[
P^{long}_t=L_tM^V_tM^X_t
\left(1+0.50\,\operatorname{clip}\left(\frac{-vwap\_gap_t}{\sigma_{intra,t}},0,3\right)\right)
\]

\[
\widehat{r}_t=slope\cdot \max(0,-F_t)\,
\operatorname{clip}(P^{long}_t/2,0,2)
\]

\[
\mathcal{E}_t=\{|r^{entry\to exit}_i-\widehat{r}_i|:\ i \text{ has matured}\}, \qquad
radius_t=\max(Q_{\tau}(\mathcal{E}_t), cost)
\]

\[
score_t=\frac{\widehat{r}_t}{radius_t+cost}
\]"""
    if name == "IntradayClockEchoSignal":
        return r"""\[
r_t=\log(close_t/close_{t-1}), \qquad
\epsilon_t=r_t-market\_return_t
\]

\[
E_{s,t}=(1-\alpha)E_{s,t-1}+\alpha\epsilon_t
\quad \text{for } s=slot_t
\]

\[
w_j=\exp(-0.10(j-1))\min\left(1,\frac{count_{slot_t+j}}{min\_slot\_samples}\right)
\]

\[
clock\_echo_t=
\frac{\sum_{j=1}^{h}w_jE_{slot_t+j,t}}{\sum_{j=1}^{h}w_j},
\qquad
\widehat{r}_{t+h}=h\cdot clock\_echo_t
\]

\[
\mathcal{E}_t=\{|r^{(h)}_i-\widehat{r}^{(h)}_i|:\ i+h\le t\}, \qquad
radius_t=\max(Q_{\tau}(\mathcal{E}_t), cost)
\]

\[
score_t=\frac{\widehat{r}_{t+h}}{radius_t+cost}
\]"""
    if name == "PageHinkley":
        return r"""\[
\mu_t=\mu_{t-1}+\frac{close_t-\mu_{t-1}}{t}
\]

\[
P_t=P_{t-1}+close_t-\mu_t-\delta, \qquad
N_t=N_{t-1}+\mu_t-close_t-\delta
\]

\[
S^+_t=P_t-\min_{i\le t}P_i, \qquad
S^-_t=N_t-\min_{i\le t}N_i
\]

\[
y_t =
\begin{cases}
1, & S^+_t > h \text{ and } S^+_t \ge S^-_t\\
-1, & S^-_t > h\\
0, & \text{otherwise}
\end{cases}
\]

After a nonzero signal the C++ implementation resets the running mean and
cumulative sums to the current close."""
    if name in {"ADWIN", "FeatureDistributionDriftDetector"}:
        return r"""\[
W_t=\operatorname{tail}_{max\_window}(W_{t-1}\cup\{x_t\})
\]

\[
\epsilon(c)=R_t
\sqrt{\frac{1}{2}\log\left(\frac{4}{\delta}\right)
\left(\frac{1}{c}+\frac{1}{|W_t|-c}\right)}
\]

\[
c^\*=\arg\max_c |\bar{x}_{c:|W_t|}-\bar{x}_{1:c}|
\quad \text{s.t.}\quad
|\bar{x}_{c:|W_t|}-\bar{x}_{1:c}|>\epsilon(c)
\]

\[
y_t =
\begin{cases}
\operatorname{sgn}(\bar{x}_{c^\*:|W_t|}-\bar{x}_{1:c^\*}), & c^\* \text{ exists}\\
0, & \text{otherwise}
\end{cases}
\]

When a cut is accepted, the older prefix is discarded and the retained suffix
becomes the next adaptive window."""
    if name == "DDM":
        return r"""\[
p_t=\frac{1}{t}\sum_{i=1}^{t}\mathbf{1}[error_i>0], \qquad
s_t=\sqrt{\frac{p_t(1-p_t)}{t}}, \qquad m_t=p_t+s_t
\]

\[
m^\*_t=\min_{i\le t}m_i, \qquad s^\*_t=s_{\arg\min_i m_i}
\]

\[
y_t =
\begin{cases}
1, & m_t>m^\*_t+d\,s^\*_t\\
0.5, & m_t>m^\*_t+w\,s^\*_t\\
0, & \text{otherwise}
\end{cases}
\]

The detector resets its error counts after a drift signal."""
    if name == "EDDM":
        return r"""\[
d_k=i_k-i_{k-1}
\quad \text{for the sample indices } i_k \text{ where } error_{i_k}>0
\]

\[
\bar{d}_k=\bar{d}_{k-1}+\frac{d_k-\bar{d}_{k-1}}{k}, \qquad
s^2_{d,k}=\frac{1}{k-1}\sum_{j=1}^{k}(d_j-\bar{d}_k)^2
\]

\[
M_k=\bar{d}_k+2s_{d,k}, \qquad
\rho_k=\frac{M_k}{\max_{j\le k}M_j}
\]

\[
y_t =
\begin{cases}
1, & \rho_k < \rho_{drift}\\
0.5, & \rho_k < \rho_{warning}\\
0, & \text{otherwise}
\end{cases}
\]"""
    if name == "HDDM":
        return r"""\[
\bar{e}_t=\frac{1}{t}\sum_{i=1}^{t}\mathbf{1}[error_i>0], \qquad
b_t(\delta)=\sqrt{\frac{\log(1/\delta)}{2t}}
\]

\[
(\bar{e}^\*_t,b^\*_t)=
\arg\min_{i\le t}\left(\bar{e}_i+b_i(\delta_{drift})\right)
\]

\[
y_t =
\begin{cases}
1, & \bar{e}_t-\bar{e}^\*_t>b_t(\delta_{drift})+b^\*_t\\
0.5, & \bar{e}_t-\bar{e}^\*_t>b_t(\delta_{warning})+b^\*_t\\
0, & \text{otherwise}
\end{cases}
\]"""
    if name == "KSWIN":
        return r"""\[
A_t=W_t[1:|W_t|-m], \qquad B_t=W_t[|W_t|-m+1:|W_t|]
\]

\[
D_t=\sup_x |\widehat{F}_{A_t}(x)-\widehat{F}_{B_t}(x)|
\]

\[
c_\alpha=\sqrt{-\frac{1}{2}\log(\alpha/2)
\left(\frac{1}{|A_t|}+\frac{1}{|B_t|}\right)}
\]

\[
y_t =
\begin{cases}
\operatorname{sgn}(\bar{B}_t-\bar{A}_t), & D_t>c_\alpha\\
0, & \text{otherwise}
\end{cases}
\]"""
    if name == "EWMAZScoreShiftDetector":
        return r"""\[
z_t=\frac{close_t-\mu_{t-1}}{\sqrt{\max(\sigma^2_{t-1},\epsilon)}}
\]

\[
y_t =
\begin{cases}
1, & z_t>h\\
-1, & z_t<-h\\
0, & \text{otherwise}
\end{cases}
\]

\[
\mu_t=\mu_{t-1}+\alpha(close_t-\mu_{t-1}), \qquad
\sigma^2_t=(1-\alpha)(\sigma^2_{t-1}+\alpha(close_t-\mu_{t-1})^2)
\]

When \(y_t\ne0\), the C++ implementation resets \(\mu_t\) to the current close
and clears the variance estimate."""
    if name == "RollingMeanShiftDetector":
        return threshold_signal(r"""\[
\bar{x}^{R}_t,\sigma^{2,R}_t=\operatorname{stats}(R_t), \qquad
\bar{x}^{B}_t,\sigma^{2,B}_t=\operatorname{stats}(B_t)
\]

\[
q_t=\frac{\bar{x}^{R}_t-\bar{x}^{B}_t}
{\sqrt{\sigma^{2,R}_t/n+\sigma^{2,B}_t/n+\epsilon}}
\]""")
    if name == "RollingVarianceShiftDetector":
        return threshold_signal(r"""\[
\sigma^{2,R}_t=\operatorname{var}(R_t), \qquad
\sigma^{2,B}_t=\operatorname{var}(B_t)
\]

\[
q_t=\log\left(\frac{\sigma^{2,R}_t+\epsilon}{\sigma^{2,B}_t+\epsilon}\right)
\]""")
    if name == "RollingMeanVarianceShiftDetector":
        return r"""\[
z^\mu_t=\frac{\bar{x}^{R}_t-\bar{x}^{B}_t}
{\sqrt{\sigma^{2,R}_t/n+\sigma^{2,B}_t/n+\epsilon}},
\qquad
z^\sigma_t=\log\left(\frac{\sigma^{2,R}_t+\epsilon}{\sigma^{2,B}_t+\epsilon}\right)
\]

\[
q_t=\sqrt{(z^\mu_t)^2+w(z^\sigma_t)^2}, \qquad
d_t=\begin{cases}
z^\mu_t, & |z^\mu_t|\ge |\sqrt{w}z^\sigma_t|\\
\sqrt{w}z^\sigma_t, & \text{otherwise}
\end{cases}
\]

\[
y_t =
\begin{cases}
\operatorname{sgn}(d_t), & q_t>h\\
0, & \text{otherwise}
\end{cases}
\]"""
    if name == "RollingCorrelationShiftDetector":
        return threshold_signal(r"""\[
\rho^R_t=\operatorname{corr}(R^x_t,R^y_t), \qquad
\rho^B_t=\operatorname{corr}(B^x_t,B^y_t)
\]

\[
q_t=\rho^R_t-\rho^B_t
\]""")
    if name == "RollingBetaShiftDetector":
        return threshold_signal(r"""\[
\beta^R_t=\frac{\operatorname{cov}(R^x_t,R^y_t)}{\operatorname{var}(R^y_t)}, \qquad
\beta^B_t=\frac{\operatorname{cov}(B^x_t,B^y_t)}{\operatorname{var}(B^y_t)}
\]

\[
q_t=\beta^R_t-\beta^B_t
\]""")
    if name == "RollingSpreadLiquidityShiftDetector":
        return threshold_signal(r"""\[
s_t=\frac{\max(ask_t-bid_t,0)}{\max(bid\_size_t+ask\_size_t,\epsilon)}
\]

\[
q_t=\operatorname{mean}(R^s_t)-\operatorname{mean}(B^s_t)
\]""")
    if name == "ThresholdRegimeDetector":
        return two_sided_hysteresis(r"""\[
q_t=value_t
\]""")
    if name == "VolatilityRegimeDetector":
        return two_sided_hysteresis(r"""\[
\Delta_t=close_t-close_{t-1}, \qquad
v_t=(1-\alpha)(v_{t-1}+\alpha\Delta_t^2)
\]

\[
q_t=\sqrt{v_t}
\]""")
    if name == "ATRRegimeDetector":
        return two_sided_hysteresis(r"""\[
TR_t=\max(high_t-low_t,\ |high_t-close_{t-1}|,\ |low_t-close_{t-1}|)
\]

\[
q_t=ATR_t=\operatorname{WilderEMA}_n(TR_t)
\]""", "This recurrence composes the standard RTTA `ATR` update with the same two-sided hysteresis state used by `ThresholdRegimeDetector`.")
    if name == "RealizedVarianceRegimeDetector":
        return two_sided_hysteresis(r"""\[
\Delta_t=close_t-close_{t-1}, \qquad
q_t=\frac{1}{n}\sum_{i\in W_t}\Delta_i^2
\]""")
    if name == "TrendChopRegimeDetector":
        return two_sided_hysteresis(r"""\[
TR_t=\max(high_t-low_t,\ |high_t-close_{t-1}|,\ |low_t-close_{t-1}|)
\]

\[
q_t=\frac{|close_t-close_{t-n}|}{\sum_{i\in W_t}TR_i}
\]""", "The metric is an efficiency ratio: values near one indicate directional trend, while values near zero indicate choppy movement.")
    if name == "LiquidityRegimeDetector":
        return two_sided_hysteresis(r"""\[
DV_t=|close_t|\max(volume_t,0), \qquad
I_t=\frac{|(close_t-close_{t-1})/close_{t-1}|}{\max(DV_t,\epsilon)}
\]

\[
q_t=\alpha I_t+(1-\alpha)q_{t-1}
\]""")
    if name == "SpreadRegimeDetector":
        return two_sided_hysteresis(r"""\[
mid_t=\frac{bid_t+ask_t}{2}, \qquad
q_t=\frac{\max(ask_t-bid_t,0)}{\max(|mid_t|,\epsilon)}
\]""")
    if name in {"VolumeRegimeDetector", "TradeIntensityRegimeDetector", "QuoteMessageRateRegimeDetector"}:
        return two_sided_hysteresis(r"""\[
b_t=\alpha\max(x_t,0)+(1-\alpha)b_{t-1}
\]

\[
q_t=\frac{\max(x_t,0)}{\max(b_{t-1},\epsilon)}
\]""", "The C++ implementation evaluates the ratio against the prior EWMA baseline and then updates the baseline with the current observation.")
    if name == "OrderFlowImbalanceRegimeDetector":
        return two_sided_hysteresis(r"""\[
e_t=
\mathbf{1}[bid_t\ge bid_{t-1}]bidSize_t
-\mathbf{1}[bid_t\le bid_{t-1}]bidSize_{t-1}
-\mathbf{1}[ask_t\le ask_{t-1}]askSize_t
+\mathbf{1}[ask_t\ge ask_{t-1}]askSize_{t-1}
\]

\[
n_t=\frac{e_t}{\max(bidSize_t+askSize_t,\epsilon)}, \qquad
q_t=\alpha n_t+(1-\alpha)q_{t-1}
\]""")
    if name == "CorrelationRegimeDetector":
        return two_sided_hysteresis(r"""\[
q_t=\rho_t=
\frac{n\sum xy-\sum x\sum y}
{\sqrt{(n\sum x^2-(\sum x)^2)(n\sum y^2-(\sum y)^2)}}
\]""", "The sums are maintained over the configured rolling window.")
    if name == "BetaRegimeDetector":
        return two_sided_hysteresis(r"""\[
q_t=\beta_t=
\frac{n\sum xy-\sum x\sum y}{n\sum y^2-(\sum y)^2}
\]""", "The sums are maintained over the configured rolling window; the C++ beta is the covariance of `real0` with `real1` divided by the variance of `real1`.")
    if name == "PairsSpreadRegimeDetector":
        return two_sided_hysteresis(r"""\[
\beta_t=\frac{C^{xy}_t}{V^y_t}, \qquad
\alpha_t=\mu^x_t-\beta_t\mu^y_t
\]

\[
e_t=x_t-(\beta_t y_t+\alpha_t), \qquad
q_t=\frac{e_t-\bar{e}_{t-1}}{\sqrt{\max(s^2_{e,t-1},\epsilon)}}
\]

\[
\bar{e}_t=\bar{e}_{t-1}+\eta(e_t-\bar{e}_{t-1}), \qquad
s^2_{e,t}=(1-\eta)(s^2_{e,t-1}+\eta(e_t-\bar{e}_{t-1})^2)
\]""")
    if name == "ExecutionCostSlippageRegimeDetector":
        return two_sided_hysteresis(r"""\[
mid_t=\frac{bid_t+ask_t}{2}, \qquad
q_t=\frac{|trade_t-mid_t|}{\max(|mid_t|,\epsilon)}
\]""")
    if name == "ResidualDriftDetector":
        return two_sided_hysteresis(r"""\[
q_t=\frac{residual_t-\mu_{t-1}}{\sqrt{\max(\sigma^2_{t-1},\epsilon)}}
\]

\[
\mu_t=\mu_{t-1}+\alpha(residual_t-\mu_{t-1}), \qquad
\sigma^2_t=(1-\alpha)(\sigma^2_{t-1}+\alpha(residual_t-\mu_{t-1})^2)
\]""")
    if name == "PredictionErrorDriftDetector":
        return upper_hysteresis(r"""\[
e_t=|actual_t-prediction_t|, \qquad
q_t=\frac{e_t-\mu_{t-1}}{\sqrt{\max(\sigma^2_{t-1},\epsilon)}}
\]

\[
\mu_t=\mu_{t-1}+\alpha(e_t-\mu_{t-1}), \qquad
\sigma^2_t=(1-\alpha)(\sigma^2_{t-1}+\alpha(e_t-\mu_{t-1})^2)
\]""")
    if name == "HitRateDriftDetector":
        return upper_hysteresis(r"""\[
m_t=\mathbf{1}[hit_t\le0], \qquad
q_t=\alpha m_t+(1-\alpha)q_{t-1}
\]""", "The metric is an EWMA miss rate; high values indicate deteriorating hit rate.")
    if name == "CalibrationDriftDetector":
        return upper_hysteresis(r"""\[
e_t=|\mathbf{1}[outcome_t>0]-\operatorname{clip}(probability_t,0,1)|
\]

\[
q_t=\alpha e_t+(1-\alpha)q_{t-1}
\]""")
    if name == "VolatilityBreakoutDetector":
        return upper_hysteresis(r"""\[
m_t=\left|\frac{close_t-close_{t-1}}{close_{t-1}}\right|, \qquad
q_t=\frac{m_t-\mu_{t-1}}{\sqrt{\max(\sigma^2_{t-1},\epsilon)}}
\]

\[
\mu_t=\mu_{t-1}+\alpha(m_t-\mu_{t-1}), \qquad
\sigma^2_t=(1-\alpha)(\sigma^2_{t-1}+\alpha(m_t-\mu_{t-1})^2)
\]""")
    if name == "VolatilityCompressionExpansionDetector":
        return two_sided_hysteresis(r"""\[
r_t=\frac{close_t-close_{t-1}}{close_{t-1}}
\]

\[
v^S_t=(1-\alpha_S)(v^S_{t-1}+\alpha_S r_t^2), \qquad
v^L_t=(1-\alpha_L)(v^L_{t-1}+\alpha_L r_t^2)
\]

\[
q_t=\frac{\sqrt{\max(v^S_t,\epsilon)}}{\sqrt{\max(v^L_t,\epsilon)}}
\]""")
    if name == "MicrostructureNoiseRegimeDetector":
        return upper_hysteresis(r"""\[
mid_t=\frac{bid_t+ask_t}{2}, \qquad
n_t=\frac{|trade_t-mid_t|}{\max(ask_t-bid_t,\epsilon)}
\]

\[
q_t=\alpha n_t+(1-\alpha)q_{t-1}
\]""")
    if name == "BidAskBounceRegimeDetector":
        return upper_hysteresis(r"""\[
side_t=\begin{cases}1, & trade_t\ge (bid_t+ask_t)/2\\ -1, & \text{otherwise}\end{cases}
\]

\[
b_t=\mathbf{1}[side_t\ne side_{t-1}], \qquad
q_t=\alpha b_t+(1-\alpha)q_{t-1}
\]""")
    if name == "QuoteStuffingDetector":
        return upper_hysteresis(r"""\[
\rho_t=\frac{\max(quote\_messages_t,0)}{\max(\max(trades_t,0),\epsilon)}
\]

\[
q_t=\alpha\rho_t+(1-\alpha)q_{t-1}
\]""")
    if name == "LeadLagRegimeDetector":
        return two_sided_hysteresis(r"""\[
\Delta x_t=x_t-x_{t-1}, \qquad \Delta y_t=y_t-y_{t-1}
\]

\[
a_t=\Delta x_{t-1}\Delta y_t, \qquad b_t=\Delta y_{t-1}\Delta x_t
\]

\[
S_t=\alpha(a_t-b_t)+(1-\alpha)S_{t-1}, \qquad
C_t=\alpha(|a_t|+|b_t|)+(1-\alpha)C_{t-1}
\]

\[
q_t=\frac{S_t}{\max(C_t,\epsilon)}
\]""")
    if name == "LiquidityDroughtDetector":
        return lower_hysteresis(r"""\[
L_t=\max(volume_t,0)+\max(bidSize_t,0)+\max(askSize_t,0)
\]

\[
q_t=\frac{L_t}{\max(B_{t-1},\epsilon)}, \qquad
B_t=\alpha L_t+(1-\alpha)B_{t-1}
\]""")
    if name == "SpreadExplosionDetector":
        return upper_hysteresis(r"""\[
s_t=\max(ask_t-bid_t,0), \qquad
q_t=\frac{s_t}{\max(B_{t-1},\epsilon)}
\]

\[
B_t=\alpha s_t+(1-\alpha)B_{t-1}
\]""")
    if name == "MarketOpenCloseTransitionDetector":
        return r"""\[
p_t=\operatorname{clip}(session\_progress_t,0,1)
\]

\[
r_t =
\begin{cases}
1, & r_{t-1}=0 \text{ and } p_t\le open_e\\
0, & r_{t-1}=1 \text{ and } p_t\ge open_x\\
-1, & r_{t-1}=0 \text{ and } p_t\ge close_e\\
0, & r_{t-1}=-1 \text{ and } p_t\le close_x\\
r_{t-1}, & \text{otherwise}
\end{cases}
\]"""
    if name == "AuctionContinuousMarketTransitionDetector":
        return upper_hysteresis(r"""\[
q_t=auction\_signal_t
\]""")
    if name == "CrossAssetCorrelationBreakDetector":
        return upper_hysteresis(r"""\[
q_t=|\rho^{short}_t-\rho^{long}_t|
\]""", "The short and long correlations are maintained by two rolling `Correlation`-style windows.")
    if name == "ZigZagSwingDetector":
        return r"""\[
\tau=\frac{percent\_change}{100}
\]

\[
direction_t =
\begin{cases}
1, & direction_{t-1}=0 \text{ and } close_t\ge start(1+\tau)\\
-1, & direction_{t-1}=0 \text{ and } close_t\le start(1-\tau)\\
-1, & direction_{t-1}=1 \text{ and } close_t\le extreme_{t-1}(1-\tau)\\
1, & direction_{t-1}=-1 \text{ and } close_t\ge extreme_{t-1}(1+\tau)\\
direction_{t-1}, & \text{otherwise}
\end{cases}
\]

\[
extreme_t =
\begin{cases}
\max(extreme_{t-1},close_t), & direction_t=1\\
\min(extreme_{t-1},close_t), & direction_t=-1\\
close_t \text{ if farther from } start, & direction_t=0
\end{cases}
\]

When direction flips, the previous extreme becomes the confirmed pivot and the
current close starts the new extreme search."""
    if name == "Delay":
        return r"""\[
y_t=x_{t-n}
\]

The C++ implementation stores the last \(n\) observations in a ring buffer and
returns the overwritten value on each update."""
    if name == "ROC":
        return r"""\[
y_t=\frac{close_t-close_{t-n}}{close_{t-n}}
\]"""
    if name in {"ATRP", "NormalizedATR"}:
        return r"""\[
ATR_t=\operatorname{ATR}_n(close_t,high_t,low_t)
\]

\[
y_t=\frac{ATR_t}{close_t}
\]"""
    if name in {"PlusDirectionalMovement", "MinusDirectionalMovement"}:
        return r"""\[
up_t=high_t-high_{t-1}, \qquad down_t=low_{t-1}-low_t
\]

\[
DM^+_t=\begin{cases}up_t, & up_t>down_t \text{ and } up_t>0\\0,&\text{otherwise}\end{cases}
\]

\[
DM^-_t=\begin{cases}down_t, & down_t>up_t \text{ and } down_t>0\\0,&\text{otherwise}\end{cases}
\]

\[
y_t=\operatorname{WilderEMA}_n(DM^{\pm}_t)
\]"""
    if name in {"PlusDirectionalIndicator", "MinusDirectionalIndicator", "DirectionalMovementIndex", "AverageDirectionalMovementIndex", "AverageDirectionalMovementIndexRating"}:
        return r"""\[
DI^+_t=100\frac{\operatorname{WilderEMA}_n(DM^+_t)}{\operatorname{ATR}_n(TR_t)}, \qquad
DI^-_t=100\frac{\operatorname{WilderEMA}_n(DM^-_t)}{\operatorname{ATR}_n(TR_t)}
\]

\[
DX_t=100\frac{|DI^+_t-DI^-_t|}{DI^+_t+DI^-_t}
\]

\[
ADX_t=\operatorname{WilderEMA}_n(DX_t), \qquad
ADXR_t=\frac{ADX_t+ADX_{t-n}}{2}
\]

`PlusDirectionalIndicator` returns \(DI^+_t\), `MinusDirectionalIndicator`
returns \(DI^-_t\), `DirectionalMovementIndex` returns \(DX_t\),
`AverageDirectionalMovementIndex` returns \(ADX_t\), and
`AverageDirectionalMovementIndexRating` returns \(ADXR_t\)."""
    if name == "BollingerBands":
        return r"""\[
M_t=\operatorname{SMA}_n(x_t), \qquad
S_t=\operatorname{StdDev}_n(x_t)
\]

\[
upper_t=M_t+2S_t, \qquad lower_t=M_t-2S_t
\]"""
    if name == "SuperTrend":
        return r"""\[
ATR_t=\operatorname{ATR}_n(close_t,high_t,low_t), \qquad
B^+_t=\frac{high_t+low_t}{2}+mATR_t, \quad
B^-_t=\frac{high_t+low_t}{2}-mATR_t
\]

\[
U_t=\begin{cases}B^+_t, & B^+_t<U_{t-1}\text{ or }close_{t-1}>U_{t-1}\\U_{t-1},&\text{otherwise}\end{cases}
\]

\[
L_t=\begin{cases}B^-_t, & B^-_t>L_{t-1}\text{ or }close_{t-1}<L_{t-1}\\L_{t-1},&\text{otherwise}\end{cases}
\]

\[
trend_t=\begin{cases}1,& close_t\ge L_t\\-1,& close_t\le U_t\\trend_{t-1},&\text{otherwise}\end{cases}, \qquad
value_t=\begin{cases}L_t,& trend_t=1\\U_t,& trend_t=-1\end{cases}
\]"""
    if name == "Vortex":
        return r"""\[
TR_t=\max(high_t-low_t,\ |high_t-close_{t-1}|,\ |low_t-close_{t-1}|)
\]

\[
VM^+_t=|high_t-low_{t-1}|, \qquad VM^-_t=|low_t-high_{t-1}|
\]

\[
VI^+_t=\frac{\sum_{i\in W_t}VM^+_i}{\sum_{i\in W_t}TR_i}, \qquad
VI^-_t=\frac{\sum_{i\in W_t}VM^-_i}{\sum_{i\in W_t}TR_i}
\]

The result fields are \(VI^+_t\), \(VI^-_t\), and their difference."""
    if name == "UlcerIndex":
        return r"""\[
H_t=\max_{i\in W_t}close_i, \qquad
d_t=100\frac{close_t-H_t}{H_t}
\]

\[
y_t=\sqrt{\frac{1}{n}\sum_{i\in W_t}d_i^2}
\]"""
    if name == "CUSUM":
        return r"""\[
\Delta_t=close_t-close_{t-1}
\]

\[
S^+_t=\max(0,S^+_{t-1}+\Delta_t-\kappa), \qquad
S^-_t=\min(0,S^-_{t-1}+\Delta_t+\kappa)
\]

\[
y_t=\begin{cases}1,&S^+_t>h\\-1,&S^-_t<-h\\0,&\text{otherwise}\end{cases}
\]"""
    if name == "ParabolicSAR":
        return r"""\[
SAR_t=SAR_{t-1}+AF_{t-1}(EP_{t-1}-SAR_{t-1})
\]

\[
EP_t=\begin{cases}\max(EP_{t-1},high_t),&trend_t=1\\\min(EP_{t-1},low_t),&trend_t=-1\end{cases}
\]

When price crosses the candidate SAR, the trend reverses, \(SAR_t\) is reset to
the prior extreme point, and the acceleration factor restarts before increasing
toward its cap on new extremes."""
    if name == "SpreadFeatures":
        return r"""\[
mid_t=\frac{bid_t+ask_t}{2}, \qquad
spread_t=\max(ask_t-bid_t,0)
\]

\[
relative\_spread_t=\frac{spread_t}{\max(|mid_t|,\epsilon)}, \qquad
trade\_location_t=\frac{trade_t-mid_t}{\max(spread_t,\epsilon)}
\]"""
    if name == "HeikinAshiTransform":
        return r"""\[
HAclose_t=\frac{open_t+high_t+low_t+close_t}{4}
\]

\[
HAopen_t=\frac{HAopen_{t-1}+HAclose_{t-1}}{2}
\]

\[
HAhigh_t=\max(high_t,HAopen_t,HAclose_t), \qquad
HAlow_t=\min(low_t,HAopen_t,HAclose_t)
\]"""
    if name == "RenkoBrickGenerator":
        return r"""\[
k_t=\left\lfloor\frac{close_t-anchor_{t-1}}{brick\_size}\right\rfloor
\]

\[
anchor_t=anchor_{t-1}+k_t\,brick\_size, \qquad
direction_t=\operatorname{sgn}(k_t)
\]"""
    if name == "MassIndex":
        return r"""\[
R_t=high_t-low_t, \qquad
E_t=\operatorname{EMA}_n(R_t), \qquad
D_t=\operatorname{EMA}_n(E_t)
\]

\[
y_t=\sum_{i\in W_t}\frac{E_i}{D_i}
\]"""
    if name == "CointegrationBreakdownMonitor":
        return upper_hysteresis(r"""\[
\beta_t=\frac{C^{xy}_t}{V^y_t}, \qquad
e_t=x_t-(\beta_t y_t+\alpha_t)
\]

\[
q_t=\left|\frac{e_t-\bar{e}_{t-1}}{\sqrt{\max(s^2_{e,t-1},\epsilon)}}\right|
\]""")
    if name == "AveragePrice":
        return r"""\[
AP_t = \frac{open_t + high_t + low_t + close_t}{4}
\]"""
    if name == "MedianPrice":
        return r"""\[
MP_t = \frac{high_t + low_t}{2}
\]"""
    if name == "TypicalPrice":
        return r"""\[
TP_t = \frac{high_t + low_t + close_t}{3}
\]"""
    if name == "WeightedClosePrice":
        return r"""\[
WCP_t = \frac{high_t + low_t + 2close_t}{4}
\]"""
    if name == "TrueRange":
        return r"""\[
TR_t = \max(high_t-low_t,\ |high_t-close_{t-1}|,\ |low_t-close_{t-1}|)
\]"""
    if name in {"DailyReturn", "RateOfChangePercentage"}:
        return r"""\[
y_t = \frac{close_t - close_{t-n}}{close_{t-n}}
\]"""
    if name == "DailyLogReturn":
        return r"""\[
y_t = \log(close_t) - \log(close_{t-1})
\]"""
    if name == "CumulativeReturn":
        return r"""\[
y_t = \frac{close_t}{close_0} - 1
\]"""
    if name == "Momentum":
        return r"""\[
y_t = close_t - close_{t-n}
\]"""
    if name == "RateOfChangeRatio":
        return r"""\[
y_t = \frac{close_t}{close_{t-n}}
\]"""
    if name == "RateOfChangeRatio100":
        return r"""\[
y_t = 100\frac{close_t}{close_{t-n}}
\]"""
    if "kalman" in lower:
        return r"""\[
\hat{x}_{t|t-1}=F\hat{x}_{t-1|t-1}, \qquad
P_{t|t-1}=FP_{t-1|t-1}F^\top+Q
\]

\[
K_t=P_{t|t-1}H^\top(HP_{t|t-1}H^\top+R)^{-1}
\]

\[
\hat{x}_{t|t}=\hat{x}_{t|t-1}+K_t(z_t-H\hat{x}_{t|t-1}), \qquad
P_{t|t}=(I-K_tH)P_{t|t-1}
\]"""
    if "hmm" in lower or "markov" in lower:
        return r"""\[
\tilde{\pi}_t = A^\top \pi_{t-1}
\]

\[
\pi_t(i)=
\frac{\tilde{\pi}_t(i)\,p(z_t\mid i)}
{\sum_j \tilde{\pi}_t(j)\,p(z_t\mid j)}
\]"""
    if "mixture" in lower:
        return r"""\[
r_{t,k}= \frac{w_{t-1,k}p(z_t\mid k)}
{\sum_j w_{t-1,j}p(z_t\mid j)}
\]

\[
w_{t,k}=(1-\alpha)w_{t-1,k}+\alpha r_{t,k}
\]"""
    if "bocpd" in lower:
        return r"""\[
R_t(r+1) = R_{t-1}(r)(1-h)p(z_t\mid r)
\]

\[
R_t(0)=\sum_r R_{t-1}(r)h\,p(z_t\mid r)
\]"""
    if "particle" in lower:
        return r"""\[
x_t^{(i)} = f(x_{t-1}^{(i)})+\epsilon_t^{(i)}
\]

\[
w_t^{(i)} \propto w_{t-1}^{(i)}p(z_t\mid x_t^{(i)}), \qquad
\sum_i w_t^{(i)}=1
\]"""
    if "gaussian process" in lower:
        return r"""\[
K_{ij}=k(x_i,x_j)+\sigma^2\delta_{ij}
\]

\[
\mu_t=k_t^\top K^{-1}y, \qquad
\sigma_t^2=k(z_t,z_t)-k_t^\top K^{-1}k_t
\]"""
    if "nadaraya" in lower or "kernel" in lower:
        return r"""\[
w_{t,i}=\exp\left(-\frac{(t-i)^2}{2h^2}\right)
\]

\[
\hat{x}_t=\frac{\sum_{i\in W_t}w_{t,i}x_i}{\sum_{i\in W_t}w_{t,i}}
\]"""
    if "regime" in lower or "detector" in lower or "drift" in lower or "hysteresis" in lower:
        return two_sided_hysteresis(r"""\[
q_t=\operatorname{metric}_\theta(z_t,s_{t-1})
\]""", "This fallback is used only for detector pages without a more specific C++ pattern in the generator.")
    if "correlation" in lower or "beta" in lower:
        return r"""\[
\mu^x_t,\mu^y_t,\sigma^2_{x,t},\sigma^2_{y,t},c_{xy,t}
= \operatorname{rollstats}(x_t,y_t,n)
\]

\[
\rho_t=\frac{c_{xy,t}}{\sigma_{x,t}\sigma_{y,t}}, \qquad
\beta_t=\frac{c_{xy,t}}{\sigma^2_{x,t}}
\]"""
    if "variance" in lower or "std" in lower:
        return r"""\[
\mu_t=\frac{1}{|W_t|}\sum_{i\in W_t}x_i
\]

\[
\sigma_t^2=\frac{1}{|W_t|}\sum_{i\in W_t}(x_i-\mu_t)^2
\]"""
    if "regression" in lower or "forecast" in lower:
        return r"""\[
\hat{\beta}_t=(X_t^\top X_t)^{-1}X_t^\top y_t
\]

\[
\hat{y}_t = [1,t]\hat{\beta}_t
\]"""
    if "high" in lower and "low" in lower or "channel" in lower or "fibonacci" in lower:
        return r"""\[
H_t=\max_{i\in W_t} high_i, \qquad L_t=\min_{i\in W_t} low_i
\]

\[
y_t = G(H_t,L_t,close_t)
\]"""
    if "moving average" in lower or "ema" in lower or "trix" in lower or "tsi" in lower or "macd" in lower:
        return r"""\[
E_t=\alpha z_t+(1-\alpha)E_{t-1}
\]

\[
y_t = G(E_t,E^{(2)}_t,\ldots,z_t)
\]"""
    if "volume" in lower or "vwap" in lower:
        return r"""\[
PV_t = PV_{t-1}+price_t\,volume_t
\]

\[
V_t = V_{t-1}+volume_t, \qquad y_t = G(PV_t,V_t,z_t)
\]"""
    if "oscillator" in lower or "rsi" in lower or "stochastic" in lower or "williams" in lower:
        return r"""\[
U_t,D_t = \operatorname{directional\_components}(z_t,z_{t-1})
\]

\[
y_t = 100\frac{\operatorname{smooth}(U_t)}
{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]"""
    if "rolling" in lower or "window" in lower:
        return r"""\[
W_t = \operatorname{push}(W_{t-1}, z_t, n)
\]

\[
y_t = G(W_t)
\]"""
    return rf"""\[
s_t = F_{{{name}}}(s_{{t-1}}, {z}; \theta)
\]

\[
y_t = G_{{{name}}}(s_t)
\]"""


def output_text(fields: list[str]) -> str:
    if not fields:
        return "The return value is the current scalar indicator value."
    rendered = ", ".join(f"`{field}`" for field in fields)
    return f"`update(...)` returns a result struct with fields {rendered}."


COMPOSED_PRIMITIVES: dict[str, tuple[str, ...]] = {
    "ATRP": ("ATR",),
    "NormalizedATR": ("ATR",),
    "ATRRegimeDetector": ("ATR", "ThresholdRegimeDetector"),
    "AverageDirectionalMovementIndex": ("ATR",),
    "AverageDirectionalMovementIndexRating": ("AverageDirectionalMovementIndex", "Delay"),
    "BollingerBands": ("SMA", "StdDev"),
    "CrossAssetCorrelationBreakDetector": ("Correlation",),
    "DirectionalMovementIndex": ("ATR",),
    "KeltnerChannel": ("EMA", "ATR"),
    "KeltnerChannelOriginal": ("SMA",),
    "MassIndex": ("EMA",),
    "MinusDirectionalIndicator": ("ATR",),
    "PlusDirectionalIndicator": ("ATR",),
    "SuperTrend": ("ATR",),
}


def composed_primitives_text(name: str, paths: dict[str, Path]) -> str:
    primitives = COMPOSED_PRIMITIVES.get(name, ())
    if not primitives:
        return ""
    links = []
    for primitive in primitives:
        path = paths.get(primitive)
        if path is None:
            links.append(f"`{primitive}`")
            continue
        rel = path.relative_to(DOCS).as_posix()
        links.append(f"[`{primitive}`]({rel})")
    return "\n\n## Composed Primitives\n\n" + ", ".join(links)


def reference_text(reference: str | None, output_path: Path) -> str:
    if not reference:
        return "- Source implementation: `src/rtta/indicator.cpp`"
    if reference.startswith("../"):
        return f"- [Detailed research note]({reference})"
    if reference.startswith("documentation/"):
        target = ROOT / reference
        rel = Path("../" + target.relative_to(ROOT / "documentation").as_posix())
        return f"- [{target.stem}]({rel.as_posix()})"
    return f"- [Background reference]({reference})"


def generate_doc(algo: Algo, path: Path, fields: list[str], cpp_classes: set[str], paths: dict[str, Path]) -> None:
    inputs = ", ".join(f"`{name}`" for name in algo.inputs) if algo.inputs else "`value`"
    source_note = (
        f"The recurrence is implemented in `src/rtta/indicator.cpp` in `class {algo.name}`."
        if algo.name in cpp_classes
        else "The public update path is registered from the RTTA indicator binding table."
    )
    text = f"""# {algo.name}

## Summary

`{algo.name}` is RTTA's streaming implementation of: {algo.description}

## Update API

```python
{update_call(algo.name, algo.inputs, algo.ctor_kwargs)}
```

The `update(...)` call consumes one observation using {inputs}. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

{theory(algo.name, algo.description)}

## Recurrence

Let \\(z_t = {input_tuple(algo.inputs)}\\) denote the observation consumed by one
`update(...)` call and let \\(\\theta\\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

{recurrence(algo.name, algo.description, algo.inputs)}

{output_text(fields)}{composed_primitives_text(algo.name, paths)}

## Implementation Notes

{source_note}

## Reference

{reference_text(algo.reference, path)}
"""
    path.write_text(text, encoding="utf-8")


def update_indexes(algos: list[Algo], paths: dict[str, Path]) -> None:
    lines = [
        "# Algorithms",
        "",
        "This file lists the public indicator algorithms exported by `rtta`. Tuning/result helper types are intentionally omitted. Detailed implementation notes and external references live in per-algorithm pages under `documentation/algorithms/`.",
        "",
        "| Algorithm | Description |",
        "|---|---|",
    ]
    for algo in algos:
        rel = paths[algo.name].relative_to(ROOT).as_posix()
        lines.append(f"| [`{algo.name}`]({rel}) | {algo.description} |")
    lines.append("")
    ALGOS.write_text("\n".join(lines), encoding="utf-8")

    index = [
        "# Algorithm Documentation",
        "",
        "This directory holds detailed Markdown source pages for RTTA's public technical",
        "analysis algorithms. Each page documents the public `update(...)` shape, the",
        "implemented theory of operation, and the C++-derived recurrence used to update",
        "state one sample at a time.",
        "",
        "| Algorithm | Summary |",
        "|---|---|",
    ]
    cdl_algos = [algo for algo in algos if algo.name.startswith("CDL")]
    for algo in algos:
        if algo.name.startswith("CDL"):
            continue
        rel = paths[algo.name].relative_to(DOCS).as_posix()
        index.append(f"| [`{algo.name}`]({rel}) | {algo.description} |")
    if cdl_algos:
        index.extend(
            [
                "",
                "## Candlestick (CDL) patterns",
                "",
                "Overview: [cdl-patterns.md](cdl-patterns.md).",
                "",
                "| Algorithm | Description |",
                "| --- | --- |",
            ]
        )
        for algo in cdl_algos:
            rel = paths[algo.name].relative_to(DOCS).as_posix()
            index.append(f"| [`{algo.name}`]({rel}) | {algo.description} |")
    index.append("")
    (DOCS / "README.md").write_text("\n".join(index), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-generated",
        action="store_true",
        help="Regenerate generated pages while preserving hand-written pilot pages.",
    )
    args = parser.parse_args()

    DOCS.mkdir(parents=True, exist_ok=True)
    descriptions = dict(parse_current_algos())
    references = parse_reference_links()
    registry = parse_registry()
    fields = parse_result_fields()
    cpp_classes = parse_cpp_classes()
    existing = existing_doc_pages()

    algos: list[Algo] = []
    paths: dict[str, Path] = {}
    for name, description in parse_current_algos():
        inputs, kwargs = registry.get(name, (("value",), {}))
        algos.append(
            Algo(
                name=name,
                description=description,
                inputs=inputs,
                ctor_kwargs=kwargs,
                reference=references.get(name),
            )
        )
        paths[name] = existing.get(name, DOCS / f"{slugify(name)}.md")

    for algo in algos:
        path = paths[algo.name]
        if path.exists() and (not args.refresh_generated or algo.name in HAND_WRITTEN):
            continue
        generate_doc(algo, path, fields.get(algo.name, []), cpp_classes, paths)

    update_indexes(algos, paths)
    print(f"documented {len(algos)} algorithms; generated {sum(1 for p in paths.values() if p.exists())} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
