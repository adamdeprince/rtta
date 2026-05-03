#!/usr/bin/env python3
"""Benchmark RTTA incremental update latency.

This benchmark intentionally focuses on stateful per-sample calls. It does not
compare against TA-Lib or other batch libraries. The `advance()` columns are
reported alongside C++ replay columns that loop over the same incremental
methods in C++ to avoid one Python call per sample.

Run after installing RTTA into the active environment, for example:

    python -m pip install --no-build-isolation -e .
    python benchmarks/benchmark_update_latency.py --samples 200000
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_indicators import INDICATORS, IndicatorSpec, MarketData, generate_market_data  # noqa: E402


BLACKHOLE: Any = None


@dataclass
class LatencyResult:
    indicator: str
    arity: int
    loop_ns: float
    update_ns: float
    update_net_ns: float
    advance_ns: float | None
    advance_net_ns: float | None
    advance_speedup: float | None
    replay_update_ns: float | None
    replay_advance_ns: float | None
    replay_advance_speedup: float | None


def benchmark_runner(runner: Callable[[], Any], samples: int, repeat: int, warmup: int) -> float:
    global BLACKHOLE

    for _ in range(warmup):
        BLACKHOLE = runner()

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        best = math.inf
        for _ in range(repeat):
            start = time.perf_counter_ns()
            BLACKHOLE = runner()
            elapsed = time.perf_counter_ns() - start
            best = min(best, elapsed)
    finally:
        if was_enabled:
            gc.enable()

    return best / samples


def indicator_ctor(rtta: Any, spec: IndicatorSpec) -> Callable[[], Any]:
    indicator_cls = getattr(rtta, spec.name)

    def make_indicator() -> Any:
        return indicator_cls(*spec.ctor_args, **spec.ctor_kwargs)

    return make_indicator


def input_lists(spec: IndicatorSpec, data: MarketData) -> list[list[float]]:
    return [data.lists[name] for name in spec.update_inputs]


def input_arrays(spec: IndicatorSpec, data: MarketData) -> list[Any]:
    return [data.arrays[name] for name in spec.update_inputs]


def make_loop_runner(inputs: list[list[float]]) -> Callable[[], None]:
    if len(inputs) == 1:
        a0 = inputs[0]

        def run() -> None:
            for _x0 in a0:
                pass

        return run

    if len(inputs) == 2:
        a0, a1 = inputs

        def run() -> None:
            for _x0, _x1 in zip(a0, a1):
                pass

        return run

    if len(inputs) == 3:
        a0, a1, a2 = inputs

        def run() -> None:
            for _x0, _x1, _x2 in zip(a0, a1, a2):
                pass

        return run

    if len(inputs) == 4:
        a0, a1, a2, a3 = inputs

        def run() -> None:
            for _x0, _x1, _x2, _x3 in zip(a0, a1, a2, a3):
                pass

        return run

    if len(inputs) == 5:
        a0, a1, a2, a3, a4 = inputs

        def run() -> None:
            for _x0, _x1, _x2, _x3, _x4 in zip(a0, a1, a2, a3, a4):
                pass

        return run

    raise ValueError(f"unsupported update arity {len(inputs)}")


def make_method_runner(make_indicator: Callable[[], Any], method_name: str, inputs: list[list[float]]) -> Callable[[], Any]:
    if len(inputs) == 1:
        a0 = inputs[0]

        def run() -> Any:
            indicator = make_indicator()
            method = getattr(indicator, method_name)
            for x0 in a0:
                method(x0)
            return indicator

        return run

    if len(inputs) == 2:
        a0, a1 = inputs

        def run() -> Any:
            indicator = make_indicator()
            method = getattr(indicator, method_name)
            for x0, x1 in zip(a0, a1):
                method(x0, x1)
            return indicator

        return run

    if len(inputs) == 3:
        a0, a1, a2 = inputs

        def run() -> Any:
            indicator = make_indicator()
            method = getattr(indicator, method_name)
            for x0, x1, x2 in zip(a0, a1, a2):
                method(x0, x1, x2)
            return indicator

        return run

    if len(inputs) == 4:
        a0, a1, a2, a3 = inputs

        def run() -> Any:
            indicator = make_indicator()
            method = getattr(indicator, method_name)
            for x0, x1, x2, x3 in zip(a0, a1, a2, a3):
                method(x0, x1, x2, x3)
            return indicator

        return run

    if len(inputs) == 5:
        a0, a1, a2, a3, a4 = inputs

        def run() -> Any:
            indicator = make_indicator()
            method = getattr(indicator, method_name)
            for x0, x1, x2, x3, x4 in zip(a0, a1, a2, a3, a4):
                method(x0, x1, x2, x3, x4)
            return indicator

        return run

    raise ValueError(f"unsupported update arity {len(inputs)}")


def make_replay_runner(make_indicator: Callable[[], Any], method_name: str, arrays: list[Any]) -> Callable[[], Any]:
    def run() -> Any:
        indicator = make_indicator()
        method = getattr(indicator, method_name)
        method(*arrays)
        return indicator

    return run


def selected_indicators(names: list[str] | None) -> tuple[IndicatorSpec, ...]:
    if not names:
        return INDICATORS

    requested = {name for group in names for name in group.split(",") if name}
    indicators = tuple(spec for spec in INDICATORS if spec.name in requested)
    found = {spec.name for spec in indicators}
    missing = sorted(requested - found)
    if missing:
        raise SystemExit(f"unknown indicator(s): {', '.join(missing)}")
    return indicators


def run_benchmarks(args: argparse.Namespace) -> tuple[list[LatencyResult], dict[str, str]]:
    try:
        rtta = importlib.import_module("rtta")
    except ImportError as exc:
        raise SystemExit(
            "Could not import rtta. Install the local package first, for example:\n"
            "  python -m pip install --no-build-isolation -e ."
        ) from exc

    data = generate_market_data(args.samples, args.seed)
    loop_ns_by_arity: dict[int, float] = {}
    rows: list[LatencyResult] = []

    for spec in selected_indicators(args.indicator):
        inputs = input_lists(spec, data)
        arrays = input_arrays(spec, data)
        arity = len(inputs)
        if arity not in loop_ns_by_arity:
            loop_ns_by_arity[arity] = benchmark_runner(make_loop_runner(inputs), args.samples, args.repeat, args.warmup)
        loop_ns = loop_ns_by_arity[arity]

        make_indicator = indicator_ctor(rtta, spec)
        update_ns = benchmark_runner(
            make_method_runner(make_indicator, "update", inputs),
            args.samples,
            args.repeat,
            args.warmup,
        )

        probe = make_indicator()
        advance_ns: float | None = None
        advance_net_ns: float | None = None
        speedup: float | None = None
        if hasattr(probe, "advance"):
            advance_ns = benchmark_runner(
                make_method_runner(make_indicator, "advance", inputs),
                args.samples,
                args.repeat,
                args.warmup,
            )
            advance_net_ns = advance_ns - loop_ns
            speedup = update_ns / advance_ns if advance_ns > 0 else None

        replay_update_ns: float | None = None
        replay_advance_ns: float | None = None
        replay_advance_speedup: float | None = None
        if hasattr(probe, "replay_update"):
            replay_update_ns = benchmark_runner(
                make_replay_runner(make_indicator, "replay_update", arrays),
                args.samples,
                args.repeat,
                args.warmup,
            )
        if hasattr(probe, "replay_advance"):
            replay_advance_ns = benchmark_runner(
                make_replay_runner(make_indicator, "replay_advance", arrays),
                args.samples,
                args.repeat,
                args.warmup,
            )
        if replay_update_ns is not None and replay_advance_ns is not None and replay_advance_ns > 0:
            replay_advance_speedup = replay_update_ns / replay_advance_ns

        rows.append(
            LatencyResult(
                indicator=spec.name,
                arity=arity,
                loop_ns=loop_ns,
                update_ns=update_ns,
                update_net_ns=update_ns - loop_ns,
                advance_ns=advance_ns,
                advance_net_ns=advance_net_ns,
                advance_speedup=speedup,
                replay_update_ns=replay_update_ns,
                replay_advance_ns=replay_advance_ns,
                replay_advance_speedup=replay_advance_speedup,
            )
        )

    sort_key = {
        "name": lambda row: row.indicator,
        "update": lambda row: -row.update_ns,
        "update-net": lambda row: -row.update_net_ns,
        "advance": lambda row: math.inf if row.advance_ns is None else -row.advance_ns,
        "advance-net": lambda row: math.inf if row.advance_net_ns is None else -row.advance_net_ns,
        "replay-update": lambda row: math.inf if row.replay_update_ns is None else -row.replay_update_ns,
        "replay-advance": lambda row: math.inf if row.replay_advance_ns is None else -row.replay_advance_ns,
    }[args.sort]
    rows.sort(key=sort_key)
    if args.top is not None:
        rows = rows[: args.top]

    versions = {
        "python": platform.python_version(),
        "numpy": importlib.import_module("numpy").__version__,
        "rtta": getattr(rtta, "__version__", "installed"),
        "platform": platform.platform(),
    }
    return rows, versions


def format_ns(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 10:
        return f"{value:.2f}"
    if value < 100:
        return f"{value:.1f}"
    return f"{value:.0f}"


def format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def write_markdown(
    rows: Iterable[LatencyResult],
    versions: dict[str, str],
    args: argparse.Namespace,
    output: Path | None,
) -> None:
    lines = [
        f"<!-- Generated by benchmarks/benchmark_update_latency.py with samples={args.samples}, repeat={args.repeat}, warmup={args.warmup}. -->",
        f"<!-- Python {versions['python']}; NumPy {versions['numpy']}; RTTA {versions['rtta']}; {versions['platform']} -->",
        "",
        "| Indicator | Arity | Python loop ns/sample | update ns/sample | update net ns/sample | advance ns/sample | advance net ns/sample | update/advance | C++ replay update ns/sample | C++ replay advance ns/sample | replay update/advance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.indicator,
                    str(row.arity),
                    format_ns(row.loop_ns),
                    format_ns(row.update_ns),
                    format_ns(row.update_net_ns),
                    format_ns(row.advance_ns),
                    format_ns(row.advance_net_ns),
                    format_ratio(row.advance_speedup),
                    format_ns(row.replay_update_ns),
                    format_ns(row.replay_advance_ns),
                    format_ratio(row.replay_advance_speedup),
                ]
            )
            + " |"
        )

    text = "\n".join(lines) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.write_text(text)


def write_csv(rows: Iterable[LatencyResult], output: Path | None) -> None:
    fieldnames = [
        "indicator",
        "arity",
        "python_loop_ns_per_sample",
        "update_ns_per_sample",
        "update_net_ns_per_sample",
        "advance_ns_per_sample",
        "advance_net_ns_per_sample",
        "update_over_advance",
        "replay_update_ns_per_sample",
        "replay_advance_ns_per_sample",
        "replay_update_over_replay_advance",
    ]
    stream = sys.stdout if output is None else output.open("w", newline="")
    try:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "indicator": row.indicator,
                    "arity": row.arity,
                    "python_loop_ns_per_sample": row.loop_ns,
                    "update_ns_per_sample": row.update_ns,
                    "update_net_ns_per_sample": row.update_net_ns,
                    "advance_ns_per_sample": row.advance_ns,
                    "advance_net_ns_per_sample": row.advance_net_ns,
                    "update_over_advance": row.advance_speedup,
                    "replay_update_ns_per_sample": row.replay_update_ns,
                    "replay_advance_ns_per_sample": row.replay_advance_ns,
                    "replay_update_over_replay_advance": row.replay_advance_speedup,
                }
            )
    finally:
        if output is not None:
            stream.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RTTA incremental update and advance latency.")
    parser.add_argument("--samples", type=int, default=200_000, help="Number of generated OHLCV samples.")
    parser.add_argument("--repeat", type=int, default=5, help="Benchmark repeats; the best run is reported.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for deterministic generated market data.")
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown", help="Output format.")
    parser.add_argument("--output", type=Path, help="Output file. Defaults to stdout.")
    parser.add_argument("--indicator", action="append", help="Indicator name or comma-separated names to benchmark.")
    parser.add_argument("--top", type=int, help="Limit output to the first N rows after sorting.")
    parser.add_argument(
        "--sort",
        choices=("update", "update-net", "advance", "advance-net", "replay-update", "replay-advance", "name"),
        default="update",
        help="Sort output rows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows, versions = run_benchmarks(args)
    if args.format == "csv":
        write_csv(rows, args.output)
    else:
        write_markdown(rows, versions, args, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
