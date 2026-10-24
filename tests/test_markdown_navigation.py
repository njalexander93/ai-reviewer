"""Verify advisory heading coordinates against explicit Markdown source examples."""

import pytest

from ai_reviewer.markdown_navigation import section_index


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_sections_preserve_hierarchy_and_physical_lines(newline: str) -> None:
    """Nested/duplicate headings and setext titles span source, not rendered lines."""
    source = newline.join(
        [
            "# Root #",
            "intro\u2028still one Git line",
            "## Child",
            "body",
            "### Nested",
            "text",
            "## Child",
            "other",
            "",
            "Next root",
            "===",
            "last",
        ]
    )
    assert section_index("GUIDE.MD", source) == {
        "status": "partial",
        "entries": [
            {"name": "Root", "level": 1, "start": 1, "end": 9},
            {"name": "Child", "level": 2, "start": 3, "end": 6},
            {"name": "Nested", "level": 3, "start": 5, "end": 6},
            {"name": "Child", "level": 2, "start": 7, "end": 9},
            {"name": "Next root", "level": 1, "start": 10, "end": 12},
        ],
    }


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_fences_exclude_heading_examples_and_require_matching_close(fence: str) -> None:
    """Short/wrong/annotated closing fences do not expose example headings."""
    other = "~~~" if fence[0] == "`" else "```"
    source = "\n".join(
        [
            "# Visible",
            "   " + fence + "python",
            "## hidden",
            other,
            "## still hidden",
            fence[:-1],
            "Title",
            "---",
            fence + " not a close",
            "## also hidden",
            fence + fence[0],
            "## Visible child",
            "text",
            fence,
            "# Unclosed example",
        ]
    )
    assert section_index("guide.markdown", source)["entries"] == [
        {"name": "Visible", "level": 1, "start": 1, "end": 15},
        {"name": "Visible child", "level": 2, "start": 12, "end": 15},
    ]


def test_subset_omits_unsupported_blocks_and_invalid_heading_forms() -> None:
    """Unsupported constructs stay advisory; comments and indented code are excluded."""
    source = "\n".join(
        [
            "#no-space",
            "####### too many",
            "    # code",
            "",
            "<!--",
            "",
            "# comment",
            "-->",
            "",
            "> # quote",
            "",
            "- item",
            "  ## contained",
            "",
            "<div>",
            "# html",
            "</div>",
            "",
            "multiline",
            "title",
            "===",
            "",
            "Title",
            "---",
            "   #### **Raw** ###",
            "#",
            "text",
        ]
    )
    assert section_index("guide.md", source)["entries"] == [
        {"name": "Title", "level": 2, "start": 23, "end": 25},
        {"name": "**Raw**", "level": 4, "start": 25, "end": 25},
        {"name": "", "level": 1, "start": 26, "end": 27},
    ]


@pytest.mark.parametrize(
    "path,source,status",
    [
        ("guide.txt", "# Heading", "unsupported_language"),
        ("guide.md", "# Heading\r## Child", "unsupported_line_endings"),
        ("guide.md", "", "partial"),
    ],
)
def test_unavailable_or_partial_navigation_is_explicit(path: str, source: str, status: str) -> None:
    """No entries never implies complete Markdown parsing or proof of absence."""
    assert section_index(path, source) == {"status": status, "entries": []}


@pytest.mark.parametrize("underline", ["- ", "---", "=\t"])
def test_short_setext_and_fence_interrupting_list(underline: str) -> None:
    """A top-level fence after a list still suppresses headings across blank lines."""
    source = "Title\n" + underline + "\n- item\n```\n\n# example\n```\n# Real\n"
    assert section_index("guide.md", source)["entries"] == [
        {"name": "Title", "level": 1 if underline.startswith("=") else 2, "start": 1, "end": 7},
        {"name": "Real", "level": 1, "start": 8, "end": 8},
    ]
