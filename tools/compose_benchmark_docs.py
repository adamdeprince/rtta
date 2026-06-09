#!/usr/bin/env python3
"""Compose benchmark documentation pages from raw benchmark_readme output."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ALGORITHM_DOCS = ROOT / "documentation" / "algorithms"


@dataclass(frozen=True)
class BenchmarkPage:
    cpu_label: str
    output_path: Path
    source_path: Path


@dataclass(frozen=True)
class BenchmarkResult:
    page: BenchmarkPage
    generated: dict[str, str]
    versions: dict[str, str]
    advance_median: str
    update_median: str
    body: str


def parse_key_values(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in re.findall(r"(\w+)=([^,.)]+)", text):
        result[key] = value.strip()
    return result


def parse_page(value: str) -> BenchmarkPage:
    parts = value.split("|")
    if len(parts) != 3:
        raise SystemExit("--page must be formatted as 'CPU label|output.md|input.md'")
    label, output, source = (part.strip() for part in parts)
    if not label or not output or not source:
        raise SystemExit("--page entries may not contain empty fields")
    return BenchmarkPage(label, Path(output), Path(source))


def parse_versions(comment: str) -> dict[str, str]:
    parts = [part.strip() for part in comment.split(";")]
    if len(parts) < 6:
        raise SystemExit(f"could not parse benchmark version comment: {comment!r}")

    versions = {
        "python": parts[0].removeprefix("Python ").strip(),
        "numpy": parts[1].removeprefix("NumPy ").strip(),
        "rtta": parts[2].removeprefix("RTTA ").strip(),
        "platform": parts[3],
    }
    for part in parts[4:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        versions[key.strip()] = value.strip()
    return versions


def algorithm_detail_pages() -> dict[str, Path]:
    links: dict[str, Path] = {}
    for path in sorted(ALGORITHM_DOCS.glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.M)
        if not title_match:
            continue
        name = title_match.group(1).strip().strip("`")
        links[name] = path.relative_to(ROOT)
    return links


def relative_markdown_link(from_path: Path, to_path: Path) -> str:
    source_dir = (ROOT / from_path).parent
    target = ROOT / to_path
    return Path(os.path.relpath(target, source_dir)).as_posix()


def link_algorithm_table_rows(body: str, output_path: Path, detail_pages: dict[str, Path]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        target = detail_pages.get(name)
        if target is None:
            return match.group(0)
        link = relative_markdown_link(output_path, target)
        return f"| [{name}]({link}) {match.group(2)}"

    return re.sub(r"(?m)^\| ([A-Za-z][A-Za-z0-9]*) (\| .*)$", replace, body)


def parse_benchmark(page: BenchmarkPage) -> BenchmarkResult:
    text = page.source_path.read_text(encoding="utf-8")
    comments = re.findall(r"^<!--\s*(.*?)\s*-->$", text, flags=re.M)
    if len(comments) < 2:
        raise SystemExit(f"{page.source_path} does not look like benchmark_readme output")

    generated = parse_key_values(comments[0])
    versions = parse_versions(comments[1])

    advance_match = re.search(
        r"Median `advance\(\.\.\.\)` latency.*?\*\*([^*]+?) ns/update\*\*",
        text,
    )
    update_match = re.search(
        r"Median `update\(\.\.\.\)` latency.*?\*\*([^*]+?) ns/update\*\*",
        text,
    )
    if not advance_match or not update_match:
        raise SystemExit(f"{page.source_path} does not contain median latency bullets")

    body = re.sub(r"^<!--.*?-->\n", "", text, count=2, flags=re.M).lstrip()
    body = body.replace("### README Latency Snapshot", "## Latency Snapshot", 1)
    body = link_algorithm_table_rows(body, page.output_path, algorithm_detail_pages())
    return BenchmarkResult(
        page=page,
        generated=generated,
        versions=versions,
        advance_median=advance_match.group(1),
        update_median=update_match.group(1),
        body=body,
    )


def repo_relative(path: Path) -> Path:
    path = path if path.is_absolute() else ROOT / path
    return path.relative_to(ROOT)


def write_cpu_page(result: BenchmarkResult, run_date: str) -> None:
    output_path = ROOT / result.page.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    versions = result.versions
    generated = result.generated
    processor = versions.get("processor", "unknown")
    extra_processor = ""
    if processor and processor != result.page.cpu_label:
        extra_processor = f"- Runtime processor string: `{processor}`\n"

    text = (
        f"# {result.page.cpu_label} Benchmarks\n\n"
        f"These results were collected on {run_date}. Public documentation identifies "
        "benchmark systems by CPU type rather than hostname.\n\n"
        "## System\n\n"
        f"- CPU: **{result.page.cpu_label}**\n"
        f"{extra_processor}"
        f"- Architecture: `{versions.get('machine', 'unknown')}`\n"
        f"- Platform: `{versions.get('platform', 'unknown')}`\n"
        f"- Python: `{versions.get('python', 'unknown')}`\n"
        f"- NumPy: `{versions.get('numpy', 'unknown')}`\n"
        f"- RTTA: `{versions.get('rtta', 'unknown')}`\n"
        f"- Samples: `{generated.get('samples', 'unknown')}`\n"
        f"- Repeats: `{generated.get('repeat', 'unknown')}`\n"
        f"- Warmup repeats: `{generated.get('warmup', 'unknown')}`\n"
        f"- Seed: `{generated.get('seed', 'unknown')}`\n\n"
        "Run command:\n\n"
        "```bash\n"
        "python benchmarks/benchmark_readme.py "
        f"--samples {generated.get('samples', '50000')} "
        f"--repeat {generated.get('repeat', '5')} "
        f"--warmup {generated.get('warmup', '1')} "
        "--output <benchmark-output.md>\n"
        "```\n\n"
        f"{result.body}"
    )
    output_path.write_text(text, encoding="utf-8")


def write_overview(results: list[BenchmarkResult], run_date: str) -> None:
    lines = [
        "# Benchmarks",
        "",
        "The benchmark pages report the live per-tick Python-facing path:",
        "",
        "- `advance(...)`: update state only.",
        "- `update(...)`: update state and return a Python value/result.",
        "",
        f"These runs were collected on {run_date} over SSH. Hostnames are intentionally "
        "omitted from the documentation; each full result page is identified by CPU type.",
        "",
        "Each CPU page carries its own system metadata and full 188-algorithm latency table.",
        "",
        "Run command:",
        "",
        "```bash",
        "python benchmarks/benchmark_readme.py --samples 50000 --repeat 5 --warmup 1 --output <benchmark-output.md>",
        "```",
        "",
        "## Results By CPU",
        "",
    ]
    for result in results:
        page = repo_relative(result.page.output_path).as_posix()
        versions = result.versions
        platform = versions.get("platform", "unknown")
        machine = versions.get("machine", "unknown")
        lines.append(
            f"- [{result.page.cpu_label}]({page}): `{platform}`, `{machine}`, "
            f"Python `{versions.get('python', 'unknown')}`, NumPy `{versions.get('numpy', 'unknown')}`, "
            f"RTTA `{versions.get('rtta', 'unknown')}`; median `advance(...)` "
            f"**{result.advance_median} ns/update**, median `update(...)` "
            f"**{result.update_median} ns/update**."
        )
    lines.append("")
    (ROOT / "BENCHMARK.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Benchmark collection date")
    parser.add_argument(
        "--page",
        action="append",
        required=True,
        help="Benchmark page as 'CPU label|output.md|input.md'",
    )
    args = parser.parse_args()

    pages = [parse_page(value) for value in args.page]
    results = [parse_benchmark(page) for page in pages]
    for result in results:
        write_cpu_page(result, args.date)
    write_overview(results, args.date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
