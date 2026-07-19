#!/usr/bin/env python3
"""Build the RTTA static HTML documentation from Markdown sources.

Generated documentation is written to html/ and is intentionally ignored by
git. The hand-authored English and Simplified Chinese landing pages and their
goblin.png mascot are preserved; README files are rendered beside them.
"""

from __future__ import annotations

import html as html_escape
import posixpath
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "html"
DOCS = ROOT / "documentation"
STYLE_SOURCE = DOCS / "assets" / "style.css"
ZH_SUFFIX = ".zh-CN"
LANDING_PAGES = {OUT / "index.html", OUT / "index.zh-CN.html"}
SKIP_MD_PARTS = {".git", ".pytest_cache", "build", "dist", "html", "TEST_TMP"}


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
        if target == "README.zh-CN.md":
            target = "README.zh-CN.html"
        elif target.endswith("/README.zh-CN.md"):
            target = target[:-15] + "index.zh-CN.html"
        elif target == "README.md":
            target = "README.html"
        elif target.endswith("/README.md"):
            target = target[:-9] + "index.html"
        elif target.endswith(".md"):
            target = target[:-3] + ".html"
        return f'{prefix}{target}{anchor}"'

    return re.sub(r'(href=")([^"]+\.md(?:#[^"]*)?)"', replace, body)


def wrap_tables(body: str) -> str:
    return re.sub(r"(<table>.*?</table>)", r'<div class="table-scroll">\1</div>', body, flags=re.S)


def align_chinese_heading_ids(body: str, english_text: str, path: Path) -> str:
    """Use the English page's stable heading IDs on its Chinese counterpart."""
    heading_re = re.compile(r'<h([1-6]) id="([^"]+)">')
    english_headings = heading_re.findall(render_markdown(english_text))
    chinese_headings = heading_re.findall(body)
    if [level for level, _ in english_headings] != [
        level for level, _ in chinese_headings
    ]:
        raise ValueError(
            f"heading structure changed in {path.relative_to(ROOT)}: "
            f"{len(english_headings)} English headings vs "
            f"{len(chinese_headings)} Chinese headings"
        )
    english_ids = iter(identifier for _, identifier in english_headings)
    return heading_re.sub(
        lambda match: f'<h{match.group(1)} id="{next(english_ids)}">',
        body,
    )


def asset_prefix(output_rel: Path) -> str:
    depth = len(output_rel.parent.parts)
    return "../" * depth


def template(
    *,
    title: str,
    masthead_title: str,
    body: str,
    output_rel: Path,
    language: str,
    counterpart_href: str,
) -> str:
    prefix = asset_prefix(output_rel)
    is_zh = language == "zh-CN"
    escaped_title = html_escape.escape(title)
    escaped_masthead = html_escape.escape(masthead_title)
    masthead_content = escaped_masthead
    if masthead_title == "RTTA":
        landing = "index.zh-CN.html" if is_zh else "index.html"
        masthead_content = f'<a href="{prefix}{landing}">{escaped_masthead}</a>'
    landing = "index.zh-CN.html" if is_zh else "index.html"
    readme = "README.zh-CN.html" if is_zh else "README.html"
    algorithms = "ALGOS.zh-CN.html" if is_zh else "ALGOS.html"
    benchmarks = "BENCHMARK.zh-CN.html" if is_zh else "BENCHMARK.html"
    labels = {
        "skip": "跳到正文" if is_zh else "Skip to content",
        "documentation": "文档" if is_zh else "Documentation",
        "credit": "由 Goblin Reactor 打造" if is_zh else "Built by Goblin Reactor",
        "tagline": "增量式、因果技术分析文档" if is_zh else "Incremental, causal technical analysis documentation",
        "nav": "文档导航" if is_zh else "Documentation navigation",
        "home": "首页" if is_zh else "Home",
        "readme": "使用说明" if is_zh else "README",
        "algorithms": "算法" if is_zh else "Algorithms",
        "benchmarks": "基准测试" if is_zh else "Benchmarks",
        "source": "源代码" if is_zh else "Source Code",
        "language": "English" if is_zh else "简体中文",
        "hreflang": "en" if is_zh else "zh-CN",
        "footer": "低延迟增量式技术分析" if is_zh else "low-latency incremental technical analysis",
    }
    return f"""<!doctype html>
<html lang="{language}" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<meta name="description" content="{'RTTA 技术分析文档' if is_zh else 'RTTA technical analysis documentation'}">
<link rel="icon" type="image/png" href="{prefix}goblin.png">
<link rel="apple-touch-icon" href="{prefix}goblin.png">
<link rel="manifest" href="{prefix}site.webmanifest">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo+Black&amp;family=IBM+Plex+Mono:wght@400;500;600&amp;family=Inter:wght@400;500;600;700&amp;family=Noto+Sans+SC:wght@400;500;600;700&amp;display=swap" rel="stylesheet">
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
<a class="skip-link" href="#content">{labels['skip']}</a>
<div class="page">
<header class="masthead">
<div class="docs-topline">
<a class="docs-brand" href="{prefix}{landing}"><span class="docs-mascot" aria-hidden="true"><img src="{prefix}goblin.png" alt=""></span><strong>RTTA</strong><small>{labels['documentation']}</small></a>
<a class="docs-credit" href="https://goblinreactor.com" rel="noopener">{labels['credit']} <b>↗</b></a>
</div>
<div class="masthead-copy">
<h1>{masthead_content}</h1>
<p class="tagline">{labels['tagline']}</p>
</div>
</header>
<nav class="nav" aria-label="{labels['nav']}">
<a class="home" href="{prefix}{landing}">{labels['home']}</a><a class="github" href="{prefix}{readme}">{labels['readme']}</a><a class="github" href="{prefix}{algorithms}">{labels['algorithms']}</a><a class="github" href="{prefix}{benchmarks}">{labels['benchmarks']}</a><a class="github" href="https://github.com/adamdeprince/rtta" rel="noopener">{labels['source']}</a><a class="language-link" href="{counterpart_href}" hreflang="{labels['hreflang']}" lang="{labels['hreflang']}">{labels['language']}</a>
</nav>
<main class="content" id="content">
{body}
</main>
<footer class="footer">
<span>RTTA · {labels['footer']}</span><a href="https://goblinreactor.com" rel="noopener">Goblin Reactor ↗</a>
</footer>
</div>
</body>
</html>
"""


def is_chinese_source(path: Path) -> bool:
    return path.name.endswith(f"{ZH_SUFFIX}.md")


def counterpart_source(path: Path) -> Path:
    if is_chinese_source(path):
        return path.with_name(path.name.removesuffix(f"{ZH_SUFFIX}.md") + ".md")
    return path.with_name(path.stem + f"{ZH_SUFFIX}.md")


def output_for_markdown(path: Path) -> Path:
    rel = path.relative_to(ROOT)
    is_zh = is_chinese_source(path)
    source_name = rel.name.removesuffix(f"{ZH_SUFFIX}.md") + ".md" if is_zh else rel.name
    html_suffix = f"{ZH_SUFFIX}.html" if is_zh else ".html"
    if rel.parent == Path(".") and source_name == "README.md":
        return Path(f"README{html_suffix}") if is_zh else Path("README.html")
    if source_name == "README.md":
        return rel.parent / f"index{ZH_SUFFIX if is_zh else ''}.html"
    return rel.with_suffix(".html")


def build_page(path: Path) -> None:
    output_rel = output_for_markdown(path)
    language = "zh-CN" if is_chinese_source(path) else "en"
    counterpart_rel = output_for_markdown(counterpart_source(path))
    counterpart_href = posixpath.relpath(
        counterpart_rel.as_posix(),
        start=output_rel.parent.as_posix() if output_rel.parent != Path(".") else ".",
    )
    body = render_markdown(path.read_text(encoding="utf-8"))
    if language == "zh-CN":
        english_path = counterpart_source(path)
        body = align_chinese_heading_ids(
            body,
            english_path.read_text(encoding="utf-8"),
            path,
        )
    body = rewrite_markdown_links(body)
    body = wrap_tables(body)
    body, page_h1 = strip_first_h1(body)
    title_base = page_h1 or path.stem
    out_html = template(
        title=f"{title_base} - RTTA",
        masthead_title=title_base,
        body=body.rstrip(),
        output_rel=output_rel,
        language=language,
        counterpart_href=counterpart_href,
    )
    out_path = OUT / output_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_html, encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)}")


def copy_assets() -> None:
    OUT.mkdir(exist_ok=True)
    shutil.copy2(STYLE_SOURCE, OUT / "style.css")

    (OUT / "site.webmanifest").write_text(
        '{\n'
        '  "icons": [\n'
        '    { "src": "/goblin.png", "sizes": "1254x1254", "type": "image/png" }\n'
        '  ]\n'
        '}\n',
        encoding="utf-8",
    )


def clean_generated_html() -> None:
    if not OUT.exists():
        return
    for path in OUT.rglob("*.html"):
        if path in LANDING_PAGES:
            continue
        path.unlink()


def markdown_sources() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not any(part in SKIP_MD_PARTS for part in path.relative_to(ROOT).parts)
    )


def main() -> int:
    clean_generated_html()
    copy_assets()
    for path in markdown_sources():
        build_page(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
