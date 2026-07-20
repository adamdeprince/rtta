import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_html_docs import (
    counterpart_source,
    is_chinese_source,
    markdown_sources,
)
from tools.generate_algorithm_docs import existing_doc_pages


HEADING = re.compile(r"^(#{1,6})\s+")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def _heading_levels(path: Path) -> list[int]:
    levels: list[int] = []
    fence: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        marker = FENCE.match(line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token[0]
            elif token[0] == fence:
                fence = None
            continue
        if fence is None and (heading := HEADING.match(line)):
            levels.append(len(heading.group(1)))
    return levels


def test_chinese_pages_preserve_english_heading_structure():
    for chinese in markdown_sources():
        if not is_chinese_source(chinese):
            continue
        english = counterpart_source(chinese)
        if english.exists():
            assert _heading_levels(chinese) == _heading_levels(english), chinese


def test_english_pages_do_not_link_to_chinese_markdown():
    offenders = [
        path
        for path in markdown_sources()
        if not is_chinese_source(path)
        and ".zh-CN.md" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_algorithm_index_generator_uses_english_source_pages():
    assert all(
        not path.name.endswith(".zh-CN.md")
        for path in existing_doc_pages().values()
    )
