#!/usr/bin/env python3
"""Build the RTTA static HTML documentation from Markdown sources.

Generated HTML is written to html/ and is intentionally ignored by git. The
source of truth remains README.md, ALGOS.md, and documentation/**/*.md.
"""

from __future__ import annotations

import html as html_escape
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "html"
DOCS = ROOT / "documentation"
STYLE_SOURCE = DOCS / "assets" / "style.css"
STRIDE_HTML = Path.home() / "dev" / "stride-align" / "html"

FAVICON_FILES = (
    "favicon.ico",
    "favicon-16.png",
    "favicon-32.png",
    "favicon-48.png",
    "favicon-64.png",
    "favicon-128.png",
    "favicon-180.png",
    "favicon-192.png",
    "favicon-256.png",
    "favicon-512.png",
    "apple-touch-icon.png",
    "icon-192.png",
    "icon-512.png",
)


def markdown_renderer():
    try:
        import markdown
    except ImportError as exc:
        raise SystemExit(
            "Python-Markdown is required to build docs. Install it with "
            "`python3 -m pip install Markdown` or use an environment that "
            "already provides the `markdown` package."
        ) from exc
    return markdown.Markdown(
        extensions=["fenced_code", "tables", "sane_lists", "toc", "attr_list"],
        output_format="html5",
    )


def extract_math(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    replacements: list[tuple[str, str, str]] = []

    def replace_display(match: re.Match[str]) -> str:
        key = f"RTTA_MATH_DISPLAY_{len(replacements)}"
        replacements.append((key, "display", match.group(1)))
        return key

    def replace_inline(match: re.Match[str]) -> str:
        key = f"RTTA_MATH_INLINE_{len(replacements)}"
        replacements.append((key, "inline", match.group(1)))
        return key

    text = re.sub(r"\\\[(.*?)\\\]", replace_display, text, flags=re.S)
    text = re.sub(r"\\\((.*?)\\\)", replace_inline, text, flags=re.S)
    return text, replacements


def restore_math(body: str, replacements: list[tuple[str, str, str]]) -> str:
    for key, kind, content in replacements:
        escaped = html_escape.escape(content.strip())
        if kind == "display":
            replacement = f'<div class="math">\\[{escaped}\\]</div>'
            body = body.replace(f"<p>{key}</p>", replacement)
        else:
            replacement = f'\\({escaped}\\)'
        body = body.replace(key, replacement)
    return body


def render_markdown(text: str) -> str:
    text, math_replacements = extract_math(text)
    md = markdown_renderer()
    return restore_math(md.convert(text), math_replacements)


def strip_first_h1(body: str) -> tuple[str, str | None]:
    match = re.search(r"<h1[^>]*>(.*?)</h1>", body, flags=re.S)
    if not match:
        return body, None
    title = re.sub(r"<.*?>", "", match.group(1)).strip()
    body = body[: match.start()] + body[match.end() :]
    return body.lstrip(), title


def rewrite_markdown_links(body: str) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix, target = match.group(1), match.group(2)
        anchor = ""
        if "#" in target:
            target, anchor = target.split("#", 1)
            anchor = "#" + anchor
        if target == "README.md":
            target = "index.html"
        elif target.endswith("/README.md"):
            target = target[:-9] + "index.html"
        elif target.endswith(".md"):
            target = target[:-3] + ".html"
        return f'{prefix}{target}{anchor}"'

    return re.sub(r'(href=")([^"]+\.md(?:#[^"]*)?)"', replace, body)


def wrap_tables(body: str) -> str:
    return re.sub(r"(<table>.*?</table>)", r'<div class="table-scroll">\1</div>', body, flags=re.S)


def asset_prefix(output_rel: Path) -> str:
    depth = len(output_rel.parent.parts)
    return "../" * depth


def template(*, title: str, masthead_title: str, tagline: str, body: str, output_rel: Path) -> str:
    prefix = asset_prefix(output_rel)
    escaped_title = html_escape.escape(title)
    escaped_masthead = html_escape.escape(masthead_title)
    masthead_content = escaped_masthead
    if masthead_title == "RTTA":
        masthead_content = f'<a href="{prefix}rtta.png">{escaped_masthead}</a>'
    return f"""<!doctype html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<meta name="description" content="RTTA technical analysis documentation">
<link rel="icon" href="{prefix}favicon.ico" sizes="any">
<link rel="icon" type="image/png" href="{prefix}favicon-32.png" sizes="32x32">
<link rel="icon" type="image/png" href="{prefix}favicon-16.png" sizes="16x16">
<link rel="apple-touch-icon" href="{prefix}apple-touch-icon.png" sizes="180x180">
<link rel="manifest" href="{prefix}site.webmanifest">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{prefix}style.css">
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']]
  }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
<div class="page">
<header class="masthead">
<h1>{masthead_content}</h1>
<p class="tagline">{tagline}</p>
</header>
<nav class="nav">
<a class="home" href="{prefix}index.html">README</a><a class="github" href="{prefix}ALGOS.html">Algorithms</a><a class="github" href="{prefix}BENCHMARK.html">Benchmarks</a><a class="github" href="https://github.com/adamdeprince/rtta" rel="noopener">Source Code</a>
</nav>
<main class="content">
{body}
</main>
<footer class="footer">
RTTA · low-latency incremental technical analysis
</footer>
</div>
</body>
</html>
"""


def output_for_markdown(path: Path) -> Path:
    rel = path.relative_to(ROOT)
    if rel.name == "README.md":
        if rel.parent == Path("."):
            return Path("index.html")
        return rel.parent / "index.html"
    return rel.with_suffix(".html")


def build_page(path: Path) -> None:
    output_rel = output_for_markdown(path)
    body = render_markdown(path.read_text(encoding="utf-8"))
    body = rewrite_markdown_links(body)
    body = wrap_tables(body)
    body, page_h1 = strip_first_h1(body)
    title_base = page_h1 or path.stem
    out_html = template(
        title=f"{title_base} - RTTA",
        masthead_title=title_base,
        tagline="Incremental, causal technical analysis documentation",
        body=body.rstrip(),
        output_rel=output_rel,
    )
    out_path = OUT / output_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_html, encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)}")


def copy_assets() -> None:
    OUT.mkdir(exist_ok=True)
    shutil.copy2(STYLE_SOURCE, OUT / "style.css")

    for name in FAVICON_FILES:
        target = OUT / name
        if target.exists():
            continue
        source = STRIDE_HTML / name
        if source.exists():
            shutil.copy2(source, target)

    (OUT / "site.webmanifest").write_text(
        '{\n'
        '  "icons": [\n'
        '    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },\n'
        '    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png" }\n'
        '  ]\n'
        '}\n',
        encoding="utf-8",
    )


def clean_generated_html() -> None:
    if not OUT.exists():
        return
    for path in OUT.rglob("*.html"):
        path.unlink()


def markdown_sources() -> list[Path]:
    sources = [ROOT / "README.md", ROOT / "ALGOS.md", ROOT / "BENCHMARK.md"]
    sources.extend(sorted(DOCS.rglob("*.md")))
    return sources


def main() -> int:
    clean_generated_html()
    copy_assets()
    for path in markdown_sources():
        build_page(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
