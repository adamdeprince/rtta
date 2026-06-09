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
        if path.name == "README.md":
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


def recurrence(name: str, description: str, inputs: tuple[str, ...]) -> str:
    lower = f"{name} {description}".lower()
    z = input_tuple(inputs)
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
        return r"""\[
s_t = F(s_{t-1}, z_t)
\]

\[
r_t =
\begin{cases}
1, & score(s_t) \ge u \\
-1, & score(s_t) \le l \\
r_{t-1}, & \text{otherwise}
\end{cases}
\]"""
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


def reference_text(reference: str | None, output_path: Path) -> str:
    if not reference:
        return "- Source implementation: `src/rtta/indicator.cpp`"
    if reference.startswith("documentation/"):
        target = ROOT / reference
        rel = Path("../" + target.relative_to(ROOT / "documentation").as_posix())
        return f"- [{target.stem}]({rel.as_posix()})"
    return f"- [Background reference]({reference})"


def generate_doc(algo: Algo, path: Path, fields: list[str], cpp_classes: set[str]) -> None:
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

{output_text(fields)}

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
    for algo in algos:
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
        generate_doc(algo, path, fields.get(algo.name, []), cpp_classes)

    update_indexes(algos, paths)
    print(f"documented {len(algos)} algorithms; generated {sum(1 for p in paths.values() if p.exists())} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
