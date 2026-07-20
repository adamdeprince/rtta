#!/usr/bin/env python3
"""Orchestrate multi-host RTTA latency benchmarks and regenerate docs.

Runs ``benchmarks/benchmark_readme.py`` on:

* this machine (local)
* SSH host ``avx10`` (Intel Xeon 6975P-C)
* SSH host ``loongson`` (Loongson-3A6000)

Before running, discovers public RTTA indicator classes that are not yet in
``benchmarks/benchmark_indicators.py`` INDICATORS and appends them (so new
algorithms appear in the tables without a manual registry edit).

Typical usage from the repo root:

    python tools/run_latency_benchmarks.py
    python tools/run_latency_benchmarks.py --samples 50000 --repeat 5 --warmup 1
    python tools/run_latency_benchmarks.py --hosts local,avx10 --skip-sync

Outputs:

* raw markdown under ``.benchmark-runs/<timestamp>/``
* composed CPU pages under ``documentation/benchmarks/``
* overview ``BENCHMARK.md`` (and a matching ``BENCHMARK.zh-CN.md``)
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import datetime as dt
import importlib
import os
import re
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "benchmarks" / "benchmark_indicators.py"
COMPOSE_SCRIPT = ROOT / "tools" / "compose_benchmark_docs.py"
BENCHMARK_SCRIPT = ROOT / "benchmarks" / "benchmark_readme.py"

# Market-data column names available to benchmark runners.
MARKET_FIELDS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "transactions",
        "error",
        "residual",
        "prediction",
        "actual",
        "hit",
        "probability",
        "outcome",
        "feature",
        "quote_messages",
        "trades",
        "session_progress",
        "auction_signal",
        "anchor",
        "value",
        "input",
        "x",
        "real0",
        "real1",
        "signed_dollar_volume",
        "trade_price",
        "bid_price",
        "bid_size",
        "ask_price",
        "ask_size",
        "period",
    }
)

# Classes that are not per-tick indicators for this latency harness.
SKIP_NAME_SUFFIXES = (
    "Result",
    "BatchResult",
    "Tuning",
    "Universe",
)

DEFAULT_CTOR_HEURISTICS: dict[str, Any] = {
    "window": 30,
    "span": 30.0,
    "size": 8,
    "particles": 32,
}


@dataclass(frozen=True)
class HostConfig:
    """How to reach a benchmark machine and how to label its CPU page."""

    name: str
    kind: str  # "local" | "ssh"
    ssh_host: str | None
    remote_dir: str
    python: str
    cpu_label: str
    output_slug: str
    extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return self.kind == "local"


DEFAULT_HOSTS: dict[str, HostConfig] = {
    "local": HostConfig(
        name="local",
        kind="local",
        ssh_host=None,
        remote_dir=str(ROOT),
        python=sys.executable,
        cpu_label="Apple M4 Max",
        output_slug="apple-m4-max",
    ),
    "avx10": HostConfig(
        name="avx10",
        kind="ssh",
        ssh_host="avx10",
        remote_dir="~/rtta",
        python="/usr/bin/python3.14",
        cpu_label="Intel Xeon 6975P-C",
        output_slug="intel-xeon-6975p-c",
    ),
    "loongson": HostConfig(
        name="loongson",
        kind="ssh",
        ssh_host="loongson",
        remote_dir="~/rtta",
        # Prefer pyenv 3.14.x when present; fall back to system python3.
        python=(
            'export PATH="$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH"; '
            'eval "$(pyenv init - 2>/dev/null)" || true; '
            'if [ -x "$HOME/.pyenv/versions/3.14.4/bin/python" ]; then '
            'printf %s "$HOME/.pyenv/versions/3.14.4/bin/python"; '
            "elif command -v python3.14 >/dev/null 2>&1; then command -v python3.14; "
            "else command -v python3; fi"
        ),
        cpu_label="Loongson-3A6000",
        output_slug="loongson-3a6000",
        # System g++ is 8.3 / system cmake is 3.16; NumPy 2.x and scikit-build need newer tools.
        # Runtime also needs the matching libstdc++ from the modern GCC install.
        extra_env={
            "CC": "/opt/loongson-gcc-15.2.0/bin/gcc",
            "CXX": "/opt/loongson-gcc-15.2.0/bin/g++",
            "CMAKE": "/opt/loongson-cmake-4.3.2/bin/cmake",
            "CMAKE_EXECUTABLE": "/opt/loongson-cmake-4.3.2/bin/cmake",
            "LD_LIBRARY_PATH": "/opt/loongson-gcc-15.2.0/lib",
        },
    ),
}


@dataclass
class DiscoveredSpec:
    name: str
    update_inputs: tuple[str, ...]
    ctor_args: tuple[Any, ...] = ()
    ctor_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_indicator_spec_source(self) -> str:
        parts = [f'"{self.name}"', _tuple_literal(self.update_inputs)]
        if self.ctor_args:
            parts.append(f"ctor_args={_tuple_literal(self.ctor_args)}")
        if self.ctor_kwargs:
            parts.append(f"ctor_kwargs={_dict_literal(self.ctor_kwargs)}")
        # Prefer explicit batch_inputs matching update_inputs for array batch APIs.
        parts.append(f"batch_inputs={_tuple_literal(self.update_inputs)}")
        return f"    IndicatorSpec({', '.join(parts)}),"


def _repr_value(value: Any) -> str:
    if isinstance(value, float) and value != value:  # NaN
        return "float('nan')"
    return repr(value)


def _tuple_literal(values: Sequence[Any]) -> str:
    if len(values) == 1:
        return f"({_repr_value(values[0])},)"
    return "(" + ", ".join(_repr_value(v) for v in values) + ")"


def _dict_literal(values: dict[str, Any]) -> str:
    items = ", ".join(f"{_repr_value(k)}: {_repr_value(v)}" for k, v in values.items())
    return "{" + items + "}"


def log(message: str) -> None:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def run(
    command: Sequence[str] | str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    if isinstance(command, str):
        display = command
        shell = True
    else:
        display = " ".join(shlex.quote(part) for part in command)
        shell = False
    log(f"$ {display}" + (f"  (cwd={cwd})" if cwd else ""))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=merged,
        shell=shell,
        text=True,
        capture_output=capture,
        check=False,
    )
    if check and result.returncode != 0:
        if capture:
            sys.stderr.write(result.stdout or "")
            sys.stderr.write(result.stderr or "")
        # RuntimeError (not SystemExit) so thread-pool workers can report failures cleanly.
        raise RuntimeError(f"command failed ({result.returncode}): {display}")
    return result


def parse_nb_signature_params(method: Any) -> list[tuple[str, bool]]:
    """Return [(param_name, has_default), ...] excluding self from a nanobind method."""
    nb_sig = getattr(method, "__nb_signature__", None)
    if not nb_sig:
        return []
    # Overload tuple: (('def name(self, a: float, b: float = \\0) -> float', None, defaults), ...)
    first = nb_sig[0]
    text = first[0]
    defaults_blob = first[2] if len(first) > 2 else None
    match = re.search(r"\((.*)\)\s*(?:->|$)", text, flags=re.S)
    if not match:
        return []
    interior = match.group(1).strip()
    if not interior:
        return []

    # Split top-level parameters.
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in interior:
        if ch == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)

    params: list[tuple[str, bool]] = []
    for part in parts:
        if part == "/" or part == "*":
            continue
        name = part.split(":", 1)[0].strip()
        if name == "self":
            continue
        # Default markers look like `= \\0` or `= False` or `= 30`.
        has_default = "=" in part
        params.append((name, has_default))
    # If defaults blob is present but some params lack '=', keep has_default as parsed.
    _ = defaults_blob
    return params


def try_construct(cls: Any, ctor_kwargs: dict[str, Any]) -> Any | None:
    try:
        return cls(**ctor_kwargs)
    except TypeError:
        pass
    try:
        return cls()
    except TypeError:
        return None


def infer_ctor_kwargs(cls: Any) -> dict[str, Any] | None:
    """Pick kwargs that let us construct the class for a smoke-test tick."""
    params = parse_nb_signature_params(cls.__init__)
    required = [name for name, has_default in params if not has_default]
    kwargs: dict[str, Any] = {}
    for name in required:
        if name in DEFAULT_CTOR_HEURISTICS:
            kwargs[name] = DEFAULT_CTOR_HEURISTICS[name]
        elif name == "window":
            kwargs[name] = 30
        else:
            # Unknown required constructor argument — skip auto-add.
            return None

    # Prefer fillna=True for latency tables when supported and free.
    optional_names = {name for name, has_default in params if has_default}
    if "fillna" in optional_names:
        kwargs.setdefault("fillna", True)

    probe = try_construct(cls, kwargs)
    if probe is None:
        return None
    return kwargs


def infer_update_inputs(cls: Any, ctor_kwargs: dict[str, Any]) -> tuple[str, ...] | None:
    params = parse_nb_signature_params(cls.update)
    if not params:
        return None
    inputs: list[str] = []
    for name, has_default in params:
        if name in MARKET_FIELDS:
            inputs.append(name)
            continue
        # Stop at first non-market parameter (e.g. slot / reset_session).
        if has_default:
            break
        return None
    if not inputs:
        return None

    # Smoke-test one update with zeros-ish market values.
    probe = try_construct(cls, ctor_kwargs)
    if probe is None or not hasattr(probe, "advance"):
        return None
    sample = {
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000.0,
        "value": 100.5,
        "input": 100.5,
        "x": 100.5,
        "real0": 100.5,
        "real1": 80.0,
        "error": 0.0,
        "residual": 0.0,
        "prediction": 100.0,
        "actual": 100.5,
        "hit": 1.0,
        "probability": 0.5,
        "outcome": 1.0,
        "feature": 100.0,
        "quote_messages": 100.0,
        "trades": 10.0,
        "session_progress": 0.1,
        "auction_signal": 0.0,
        "anchor": 0.0,
        "signed_dollar_volume": 1000.0,
        "trade_price": 100.5,
        "bid_price": 100.4,
        "bid_size": 10.0,
        "ask_price": 100.6,
        "ask_size": 10.0,
        "transactions": 5.0,
    }
    args = [sample[name] for name in inputs]
    try:
        probe.update(*args)
        advance = getattr(probe, "advance", None)
        if advance is not None:
            advance(*args)
    except Exception:
        return None
    return tuple(inputs)


def discover_new_indicators() -> list[DiscoveredSpec]:
    """Find public RTTA classes missing from the benchmark registry."""
    sys.path.insert(0, str(ROOT))
    try:
        rtta = importlib.import_module("rtta")
    except ImportError as exc:
        raise SystemExit(
            "Could not import local rtta. Install first, e.g.\n"
            "  python -m pip install --no-build-isolation -e ."
        ) from exc

    from benchmarks.benchmark_indicators import INDICATORS

    known = {spec.name for spec in INDICATORS}
    discovered: list[DiscoveredSpec] = []

    for name in sorted(dir(rtta)):
        if not name[:1].isupper():
            continue
        if any(name.endswith(suffix) for suffix in SKIP_NAME_SUFFIXES):
            continue
        if name in known:
            continue
        cls = getattr(rtta, name, None)
        if cls is None or not hasattr(cls, "update"):
            continue
        if not hasattr(cls, "advance"):
            log(f"skip {name}: no advance()")
            continue

        ctor_kwargs = infer_ctor_kwargs(cls)
        if ctor_kwargs is None:
            log(f"skip {name}: could not construct with defaults")
            continue
        update_inputs = infer_update_inputs(cls, ctor_kwargs)
        if update_inputs is None:
            log(f"skip {name}: could not infer market update inputs")
            continue

        # Drop fillna from stored kwargs when True is already a class default —
        # keep only kwargs that differ from empty construction needs.
        stored_kwargs = dict(ctor_kwargs)
        discovered.append(
            DiscoveredSpec(
                name=name,
                update_inputs=update_inputs,
                ctor_kwargs=stored_kwargs,
            )
        )
        log(
            f"discovered {name}: update_inputs={update_inputs}"
            + (f" ctor_kwargs={stored_kwargs}" if stored_kwargs else "")
        )
    return discovered


def append_specs_to_registry(specs: Sequence[DiscoveredSpec]) -> int:
    """Insert new IndicatorSpec entries into the INDICATORS tuple. Returns count added."""
    if not specs:
        return 0

    text = REGISTRY_PATH.read_text(encoding="utf-8")
    # Existing names by simple scan.
    existing = set(re.findall(r'IndicatorSpec\(\s*"([^"]+)"', text))
    to_add = [spec for spec in specs if spec.name not in existing]
    if not to_add:
        return 0

    # Insert alphabetically among IndicatorSpec lines for readability.
    lines = text.splitlines(keepends=True)
    spec_line_indexes = [
        index
        for index, line in enumerate(lines)
        if re.match(r'\s+IndicatorSpec\("', line)
    ]
    if not spec_line_indexes:
        raise SystemExit(f"could not find IndicatorSpec entries in {REGISTRY_PATH}")

    # Build map name -> line index for insertion point.
    def spec_name_at(index: int) -> str | None:
        match = re.search(r'IndicatorSpec\(\s*"([^"]+)"', lines[index])
        return match.group(1) if match else None

    names_in_file = [(spec_name_at(i), i) for i in spec_line_indexes]
    names_in_file = [(name, i) for name, i in names_in_file if name]

    insertions: list[tuple[int, str]] = []
    for spec in sorted(to_add, key=lambda item: item.name):
        insert_at = names_in_file[-1][1] + 1  # default: after last
        for name, line_index in names_in_file:
            if name > spec.name:
                insert_at = line_index
                break
        else:
            # After last IndicatorSpec line.
            insert_at = names_in_file[-1][1] + 1
        insertions.append((insert_at, spec.to_indicator_spec_source() + "\n"))

    # Apply from bottom to top so indexes remain valid.
    for insert_at, source in sorted(insertions, key=lambda item: item[0], reverse=True):
        lines.insert(insert_at, source)

    REGISTRY_PATH.write_text("".join(lines), encoding="utf-8")
    # Validate syntax.
    ast.parse(REGISTRY_PATH.read_text(encoding="utf-8"))
    log(f"appended {len(to_add)} indicator(s) to {REGISTRY_PATH.relative_to(ROOT)}")
    return len(to_add)


def resolve_remote_python(host: HostConfig) -> str:
    if host.is_local:
        return host.python
    # Fixed absolute path (e.g. /usr/bin/python3.14).
    if host.python.startswith("/") and "\n" not in host.python and " " not in host.python:
        return host.python
    # Otherwise treat host.python as a remote shell snippet that prints the interpreter path.
    result = ssh_run(host, host.python, capture=True, check=True)
    path = (result.stdout or "").strip().splitlines()[-1].strip() if (result.stdout or "").strip() else ""
    if not path:
        raise SystemExit(f"could not resolve python on {host.name}")
    return path


def remote_shell_path(path: str) -> str:
    """Return a remote path that expands under bash (prefer $HOME over ~)."""
    if path.startswith("~/"):
        return f"$HOME/{path[2:]}"
    if path == "~":
        return "$HOME"
    return path


def ssh_run(host: HostConfig, remote_command: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Run one remote shell command. Always pass a single command string to ssh."""
    assert host.ssh_host is not None
    # Force bash so $HOME / pipelines behave consistently on all hosts.
    wrapped = f"bash -lc {shlex.quote(remote_command)}"
    return run(["ssh", host.ssh_host, wrapped], check=check, capture=capture)


def rsync_to_host(host: HostConfig) -> None:
    if host.is_local:
        return
    assert host.ssh_host is not None
    excludes = [
        "--exclude",
        ".git",
        "--exclude",
        "build",
        "--exclude",
        "dist",
        "--exclude",
        ".pytest_cache",
        "--exclude",
        "**/__pycache__",
        "--exclude",
        "html",
        "--exclude",
        ".benchmark-runs",
        "--exclude",
        "*.so",
        "--exclude",
        ".venv*",
    ]
    # rsync understands ~/ on the remote side when left unquoted in the destination.
    remote = f"{host.ssh_host}:{host.remote_dir.rstrip('/')}/"
    run(
        [
            "rsync",
            "-az",
            "--delete",
            *excludes,
            f"{ROOT}/",
            remote,
        ],
        check=True,
    )


def install_and_benchmark_remote(
    host: HostConfig,
    *,
    python: str,
    samples: int,
    repeat: int,
    warmup: int,
    seed: int,
    remote_output: str,
) -> None:
    assert host.ssh_host is not None
    remote_dir = remote_shell_path(host.remote_dir)
    # Optional host-specific env (e.g. modern GCC on Loongson).
    env_exports = ""
    for key, value in host.extra_env.items():
        # LD_LIBRARY_PATH is handled below so we can prepend rather than replace.
        if key == "LD_LIBRARY_PATH":
            continue
        env_exports += f"export {key}={shlex.quote(value)}\n"
    # Prefer host-provided tool directories (modern GCC / CMake) early on PATH.
    path_dirs: list[str] = []
    for key in ("CXX", "CMAKE", "CMAKE_EXECUTABLE"):
        if key in host.extra_env:
            path_dirs.append(str(Path(host.extra_env[key]).parent))
    if path_dirs:
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        ordered: list[str] = []
        for directory in path_dirs:
            if directory not in seen:
                seen.add(directory)
                ordered.append(directory)
        joined = ":".join(ordered)
        env_exports += f'export PATH={shlex.quote(joined)}:"$PATH"\n'
    if "LD_LIBRARY_PATH" in host.extra_env:
        # Prepend, don't replace, so other libs remain available.
        env_exports += (
            f'export LD_LIBRARY_PATH={shlex.quote(host.extra_env["LD_LIBRARY_PATH"])}'
            '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n'
        )

    script = textwrap.dedent(
        f"""\
        set -euo pipefail
        cd {remote_dir}
        BASE_PY={shlex.quote(python)}
        {env_exports}
        echo "host=$(hostname) base_python=$($BASE_PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"
        echo "CC=${{CC:-}}; CXX=${{CXX:-}}; cmake=$(command -v cmake || true); g++=$(command -v g++ || true)"
        command -v g++ >/dev/null && g++ --version | head -1 || true
        command -v cmake >/dev/null && cmake --version | head -1 || true

        # Always use a project venv to avoid PEP 668 system Python locks and keep deps isolated.
        VENV="$PWD/.venv"
        if [ ! -x "$VENV/bin/python" ]; then
          $BASE_PY -m venv "$VENV"
        fi
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
        PY="$VENV/bin/python"
        $PY -m pip install -q --upgrade pip setuptools wheel
        # Prefer a host CMake on PATH when available; only fall back to pip cmake wheels
        # on arches that publish them (not loongarch64).
        if ! command -v cmake >/dev/null 2>&1 || ! cmake --version | head -1 | grep -Eq 'version (3\\.(2[1-9]|[3-9][0-9])|[4-9])'; then
          $PY -m pip install -q 'cmake>=3.21' || true
        fi
        $PY -m pip install -q 'ninja' || true
        $PY -m pip install -q 'nanobind>=2.2' 'scikit-build-core>=0.12' 'fast-kalman==0.2.3' 'numpy>=2.0.2'
        $PY -m pip install -q --no-build-isolation -e .
        $PY -c "import rtta, numpy, importlib.metadata as m; print('rtta', getattr(rtta,'__version__', m.version('pyrtta')), 'numpy', numpy.__version__)"
        mkdir -p .benchmark-runs
        $PY benchmarks/benchmark_readme.py \\
            --samples {int(samples)} \\
            --repeat {int(repeat)} \\
            --warmup {int(warmup)} \\
            --seed {int(seed)} \\
            --output {shlex.quote(remote_output)}
        """
    )
    log(f"{host.name}: remote install + benchmark")
    proc = subprocess.run(
        ["ssh", host.ssh_host, "bash", "-s"],
        input=script,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"remote benchmark failed on {host.name} (exit {proc.returncode})")



def run_local_benchmark(
    host: HostConfig,
    *,
    samples: int,
    repeat: int,
    warmup: int,
    seed: int,
    output: Path,
) -> None:
    env = os.environ.copy()
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 4))
    # Ensure package is importable / up to date.
    run(
        [
            host.python,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-build-isolation",
            "-e",
            ".",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    run(
        [
            host.python,
            str(BENCHMARK_SCRIPT),
            "--samples",
            str(samples),
            "--repeat",
            str(repeat),
            "--warmup",
            str(warmup),
            "--seed",
            str(seed),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )


def fetch_remote_file(host: HostConfig, remote_path: str, local_path: Path) -> None:
    """Copy remote_path (may use ~) to local_path via scp."""
    assert host.ssh_host is not None
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # Expand ~ on the remote, print an absolute path, then scp that path.
    expanded = ssh_run(
        host,
        f"python3 -c \"import os; print(os.path.expanduser({remote_path!r}))\"",
        capture=True,
        check=True,
    ).stdout.strip()
    if not expanded:
        raise SystemExit(f"could not resolve remote path {remote_path!r} on {host.name}")
    run(["scp", f"{host.ssh_host}:{expanded}", str(local_path)], check=True)


def run_host(
    host: HostConfig,
    *,
    work_dir: Path,
    samples: int,
    repeat: int,
    warmup: int,
    seed: int,
    skip_sync: bool,
    skip_install: bool,
) -> Path:
    raw_path = work_dir / f"{host.output_slug}.raw.md"
    log(f"=== {host.name}: start ({host.cpu_label}) ===")
    if host.is_local:
        if skip_install:
            run(
                [
                    host.python,
                    str(BENCHMARK_SCRIPT),
                    "--samples",
                    str(samples),
                    "--repeat",
                    str(repeat),
                    "--warmup",
                    str(warmup),
                    "--seed",
                    str(seed),
                    "--output",
                    str(raw_path),
                ],
                cwd=ROOT,
                check=True,
            )
        else:
            run_local_benchmark(
                host,
                samples=samples,
                repeat=repeat,
                warmup=warmup,
                seed=seed,
                output=raw_path,
            )
    else:
        if not skip_sync:
            rsync_to_host(host)
        python = resolve_remote_python(host)
        log(f"{host.name}: using python {python}")
        # Paths relative to the remote checkout so ~ expansion is not required.
        remote_output_rel = f".benchmark-runs/{raw_path.name}"
        remote_dir = remote_shell_path(host.remote_dir)
        assert host.ssh_host is not None
        ssh_run(host, f"mkdir -p {remote_dir}/.benchmark-runs", check=True)
        if skip_install:
            skip_env = ""
            for key, value in host.extra_env.items():
                if key == "LD_LIBRARY_PATH":
                    skip_env += (
                        f'export LD_LIBRARY_PATH={shlex.quote(value)}'
                        '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n'
                    )
                elif key in {"CC", "CXX", "CMAKE", "CMAKE_EXECUTABLE"}:
                    continue
                else:
                    skip_env += f"export {key}={shlex.quote(value)}\n"
            script = textwrap.dedent(
                f"""\
                set -euo pipefail
                cd {remote_dir}
                {skip_env}
                if [ -x .venv/bin/python ]; then PY="$PWD/.venv/bin/python"; else PY={shlex.quote(python)}; fi
                mkdir -p .benchmark-runs
                $PY benchmarks/benchmark_readme.py \\
                    --samples {int(samples)} --repeat {int(repeat)} \\
                    --warmup {int(warmup)} --seed {int(seed)} \\
                    --output {shlex.quote(remote_output_rel)}
                """
            )
            proc = subprocess.run(
                ["ssh", host.ssh_host, "bash", "-s"],
                input=script,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"remote benchmark failed on {host.name}")
        else:
            install_and_benchmark_remote(
                host,
                python=python,
                samples=samples,
                repeat=repeat,
                warmup=warmup,
                seed=seed,
                remote_output=remote_output_rel,
            )
        fetch_remote_file(host, f"{host.remote_dir}/{remote_output_rel}", raw_path)
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        raise SystemExit(f"missing benchmark output for {host.name}: {raw_path}")
    log(f"=== {host.name}: done -> {raw_path} ===")
    return raw_path


def compose_docs(
    host_results: Sequence[tuple[HostConfig, Path]],
    *,
    run_date: str,
) -> None:
    pages: list[str] = []
    for host, raw_path in host_results:
        out = f"documentation/benchmarks/{host.output_slug}.md"
        pages.append(f"{host.cpu_label}|{out}|{raw_path}")

    cmd = [
        sys.executable,
        str(COMPOSE_SCRIPT),
        "--date",
        run_date,
    ]
    for page in pages:
        cmd.extend(["--page", page])
    run(cmd, cwd=ROOT, check=True)

    # Lightweight Chinese overview mirror (CPU pages keep bilingual pair when present).
    write_chinese_overview(run_date, host_results)
    log("wrote BENCHMARK.md")


def write_chinese_overview(
    run_date: str,
    host_results: Sequence[tuple[HostConfig, Path]],
) -> None:
    """Regenerate BENCHMARK.zh-CN.md from the English overview medians."""
    english = (ROOT / "BENCHMARK.md").read_text(encoding="utf-8")
    # Pull result bullets and translate the shell.
    bullets = re.findall(r"^- \[.*?$", english, flags=re.M)
    lines = [
        "# 基准测试",
        "",
        "基准测试页面报告面向 Python 的实时逐 tick 路径：",
        "",
        "- `advance(...)`：仅更新状态。",
        "- `update(...)`：更新状态并返回 Python 值/结果。",
        "",
        f"这些结果采集于 {run_date}（含 SSH 远端）。公开文档有意省略主机名；"
        "每个完整结果页以 CPU 类型标识。",
        "",
        "每个 CPU 页面包含各自的系统元数据以及完整算法延迟表。",
        "",
        "运行命令：",
        "",
        "```bash",
        "python benchmarks/benchmark_readme.py --samples 50000 --repeat 5 --warmup 1 --output <benchmark-output.md>",
        "```",
        "",
        "## 按 CPU 的结果",
        "",
    ]
    # Rewrite links to zh-CN siblings when they exist.
    for bullet in bullets:
        match = re.match(r"^- \[([^\]]+)\]\(([^)]+)\)(:.*)$", bullet)
        if not match:
            lines.append(bullet)
            continue
        label, path, rest = match.group(1), match.group(2), match.group(3)
        zh_path = path.replace(".md", ".zh-CN.md")
        if (ROOT / zh_path).exists() or path.endswith(".md"):
            # Prefer zh-CN page path even if not yet regenerated; compose only wrote EN.
            target = path
            zh_candidate = Path(path).with_suffix("").as_posix()
            # documentation/benchmarks/foo.md -> foo.zh-CN.md
            if path.startswith("documentation/benchmarks/"):
                slug = Path(path).name.replace(".md", "")
                zh_rel = f"documentation/benchmarks/{slug}.zh-CN.md"
                # Create/refresh zh CPU page as a thin translated wrapper if missing raw compose.
                en_page = ROOT / path
                zh_page = ROOT / zh_rel
                if en_page.exists():
                    _write_chinese_cpu_page(en_page, zh_page, label, run_date)
                target = zh_rel
            lines.append(f"- [{label}]({target}){rest}")
        else:
            lines.append(bullet)
    lines.append("")
    (ROOT / "BENCHMARK.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    log("wrote BENCHMARK.zh-CN.md")


def _write_chinese_cpu_page(en_page: Path, zh_page: Path, cpu_label: str, run_date: str) -> None:
    """Translate the static header of a CPU benchmark page; keep the data table."""
    text = en_page.read_text(encoding="utf-8")
    # Split at latency snapshot / table.
    body_match = re.search(r"(## Latency Snapshot[\s\S]*)", text)
    body = body_match.group(1) if body_match else text
    body = body.replace("## Latency Snapshot", "## 延迟快照", 1)
    body = body.replace("Benchmarked algorithms in registry", "注册表中的基准算法数")
    body = body.replace("Algorithms shown", "展示的算法数")
    body = body.replace(
        "Median `advance(...)` latency, update state only",
        "中位 `advance(...)` 延迟（仅更新状态）",
    )
    body = body.replace(
        "Median `update(...)` latency, update state and return a value/result",
        "中位 `update(...)` 延迟（更新状态并返回值/结果）",
    )
    body = body.replace(
        "| Algorithm | Inputs | update only: `advance(...)` ns/update | update + return: `update(...)` ns/update |",
        "| 算法 | 输入数 | 仅更新：`advance(...)` ns/update | 更新并返回：`update(...)` ns/update |",
    )

    # System metadata block from English.
    system_fields = re.findall(r"^- (.*?)$", text.split("## System", 1)[-1].split("## ", 1)[0], flags=re.M)
    field_map = {
        "CPU:": "CPU：",
        "Runtime processor string:": "运行时处理器字符串：",
        "Architecture:": "架构：",
        "Platform:": "平台：",
        "Python:": "Python：",
        "NumPy:": "NumPy：",
        "RTTA:": "RTTA：",
        "Samples:": "样本数：",
        "Repeats:": "重复次数：",
        "Warmup repeats:": "预热重复次数：",
        "Seed:": "随机种子：",
    }
    zh_system: list[str] = []
    for field in system_fields:
        line = field
        for eng, zh in field_map.items():
            if line.startswith(eng):
                line = zh + line[len(eng) :]
                break
        zh_system.append(f"- {line}")

    header = (
        f"# {cpu_label} 基准测试\n\n"
        f"以下结果采集于 {run_date}。公开文档以 CPU 类型而不是主机名标识基准测试系统。\n\n"
        "## 系统\n\n"
        + "\n".join(zh_system)
        + "\n\n运行命令：\n\n"
        "```bash\n"
        "python benchmarks/benchmark_readme.py --samples 50000 --repeat 5 --warmup 1 "
        "--output <benchmark-output.md>\n"
        "```\n\n"
    )
    zh_page.write_text(header + body.lstrip(), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RTTA latency benchmarks on local/avx10/loongson and regenerate BENCHMARK.md"
    )
    parser.add_argument(
        "--hosts",
        default="local,avx10,loongson",
        help="Comma-separated host keys (default: local,avx10,loongson).",
    )
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="Date stamp written into documentation (default: today).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for raw outputs (default: .benchmark-runs/<timestamp>).",
    )
    parser.add_argument(
        "--skip-discover",
        action="store_true",
        help="Do not scan rtta for indicators missing from the registry.",
    )
    parser.add_argument(
        "--skip-registry-update",
        action="store_true",
        help="Discover new indicators but do not rewrite benchmark_indicators.py.",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Do not rsync the tree to SSH hosts.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Assume RTTA is already installed on each host; only run the benchmark.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run hosts one after another instead of in parallel.",
    )
    parser.add_argument(
        "--compose-only",
        action="store_true",
        help="Only compose docs from existing raw files in --work-dir.",
    )
    return parser.parse_args(argv)


def select_hosts(names: str) -> list[HostConfig]:
    selected: list[HostConfig] = []
    for raw in names.split(","):
        key = raw.strip()
        if not key:
            continue
        if key not in DEFAULT_HOSTS:
            known = ", ".join(sorted(DEFAULT_HOSTS))
            raise SystemExit(f"unknown host {key!r}; expected one of: {known}")
        selected.append(DEFAULT_HOSTS[key])
    if not selected:
        raise SystemExit("no hosts selected")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(ROOT)
    hosts = select_hosts(args.hosts)

    work_dir = args.work_dir
    if work_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        work_dir = ROOT / ".benchmark-runs" / stamp
    work_dir = work_dir if work_dir.is_absolute() else ROOT / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    log(f"work dir: {work_dir}")

    if not args.skip_discover:
        discovered = discover_new_indicators()
        if discovered and not args.skip_registry_update:
            append_specs_to_registry(discovered)
            # Reload notice for subsequent local import.
            importlib.invalidate_caches()
        elif discovered:
            log(
                f"discovered {len(discovered)} new indicator(s) but registry update skipped: "
                + ", ".join(spec.name for spec in discovered)
            )
        else:
            log("no new indicators discovered")
    else:
        log("indicator discovery skipped")

    if args.compose_only:
        host_results: list[tuple[HostConfig, Path]] = []
        for host in hosts:
            raw = work_dir / f"{host.output_slug}.raw.md"
            if not raw.exists():
                raise SystemExit(f"--compose-only missing {raw}")
            host_results.append((host, raw))
        compose_docs(host_results, run_date=args.date)
        return 0

    # After registry updates, re-sync is required so remotes see new specs.
    results: dict[str, Path] = {}
    errors: dict[str, str] = {}

    def _worker(host: HostConfig) -> tuple[str, Path]:
        path = run_host(
            host,
            work_dir=work_dir,
            samples=args.samples,
            repeat=args.repeat,
            warmup=args.warmup,
            seed=args.seed,
            skip_sync=args.skip_sync,
            skip_install=args.skip_install,
        )
        return host.name, path

    if args.sequential or len(hosts) == 1:
        for host in hosts:
            try:
                name, path = _worker(host)
                results[name] = path
            except Exception as exc:  # noqa: BLE001 - collect per-host failures
                errors[host.name] = str(exc)
                log(f"ERROR {host.name}: {exc}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
            futures = {pool.submit(_worker, host): host for host in hosts}
            for future in concurrent.futures.as_completed(futures):
                host = futures[future]
                try:
                    name, path = future.result()
                    results[name] = path
                except Exception as exc:  # noqa: BLE001
                    errors[host.name] = str(exc)
                    log(f"ERROR {host.name}: {exc}")

    if errors and not results:
        raise SystemExit("all hosts failed:\n" + "\n".join(f"  {k}: {v}" for k, v in errors.items()))
    if errors:
        log("partial success; failed hosts: " + ", ".join(sorted(errors)))

    host_results = [(host, results[host.name]) for host in hosts if host.name in results]
    compose_docs(host_results, run_date=args.date)

    log("summary:")
    overview = (ROOT / "BENCHMARK.md").read_text(encoding="utf-8")
    for line in overview.splitlines():
        if line.startswith("- ["):
            log("  " + line)
    if errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
