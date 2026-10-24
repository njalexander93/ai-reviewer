"""Locate a conservative Markdown heading subset in captured source, without rendering.

ATX headings and single-line setext headings have inclusive LF section ranges.
Fenced code is excluded. This is advisory navigation, not a CommonMark parser:
containers, HTML and multiline setext headings are intentionally unsupported.
"""

import re
from typing import Any

from ai_reviewer.patch import lines


def section_index(filename: str, text: str) -> dict[str, Any]:
    """Return source-ordered heading candidates and nesting-aware section ranges.

    Only .md/.markdown paths (case-insensitive) are supported. Names retain inline
    markup. A section ends before the next equal/shallower indexed heading or EOF.
    Up to three leading spaces are allowed; indented code is excluded. Backtick
    and tilde fences must close with the same character and at least opening length.
    HTML/container blocks are conservatively skipped until a blank line; comments
    are skipped through their terminator. Missing headings never prove absence.
    Bare CR coordinates are unavailable. Input is never executed or changed.
    """
    if not filename.lower().endswith((".md", ".markdown")):
        return {"status": "unsupported_language", "entries": []}
    if "\r" in text.replace("\r\n", "\n"):
        return {"status": "unsupported_line_endings", "entries": []}
    records = lines(text)
    entries: list[dict[str, Any]] = []
    parents: list[dict[str, Any]] = []
    fence = ""
    skip_block = False
    comment = False
    paragraph: list[tuple[int, str]] = []
    # Recognize only supported block forms, never headings inside fenced examples.
    for number, record in enumerate(records, 1):
        line = record.rstrip("\r\n")
        if fence:
            if re.fullmatch(
                r" {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*", line
            ):
                fence = ""
            continue
        if comment:
            comment = "-->" not in line
            continue
        opening = re.match(r" {0,3}(`{3,}|~{3,})(.*)$", line)
        if opening and (opening[1][0] == "~" or "`" not in opening[2]):
            skip_block = False
            fence = opening[1]
            paragraph = []
            continue
        if skip_block:
            skip_block = bool(line.strip())
            continue
        if line.lstrip().startswith("<!--"):
            comment = "-->" not in line
            paragraph = []
            continue
        if re.match(r" {0,3}(?:<|>|[-+*][ \t]|[0-9]+[.)][ \t])", line) and not re.fullmatch(
            r" {0,3}-+[ \t]*", line
        ):
            skip_block = True
            paragraph = []
            continue
        heading = re.fullmatch(r" {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)", line)
        underline = re.fullmatch(r" {0,3}(=+|-+)[ \t]*", line)
        if heading:
            level = len(heading[1])
            name = re.sub(r"(?:^|[ \t]+)#+[ \t]*$", "", heading[2] or "").strip()
            start = number
        elif underline and len(paragraph) == 1:
            level = 1 if underline[1][0] == "=" else 2
            start, name = paragraph[0]
        else:
            if not line.strip() or line.startswith(("    ", "\t")) or underline:
                paragraph = []
            else:
                paragraph.append((number, line.strip()))
            continue
        paragraph = []
        # Close siblings/ancestors while preserving nested sections in their parent.
        while parents and parents[-1]["level"] >= level:
            parents.pop()["end"] = start - 1
        entry = {"name": name, "level": level, "start": start, "end": len(records)}
        entries.append(entry)
        parents.append(entry)
    return {"status": "partial", "entries": entries}
