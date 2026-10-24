"""Check packet ownership and approval gates against explicit synthetic changes."""

import pytest

from ai_reviewer.chunking import file_declarations, plan
from ai_reviewer.contracts import ChangedFile
from tests.test_github import Server, bundle


def test_small_packet_preserves_whole_context() -> None:
    """Packets below the existing ceiling keep the exact original reviewer payload."""
    context, _, _ = bundle(Server())
    assert plan(context).whole == context.model_dump_json()


def test_large_file_is_split_without_losing_changes() -> None:
    """A 900 KB added text file owns every original patch record exactly once."""
    context, _, _ = bundle(Server())
    text = "value = '" + "x" * 90 + "'\n"
    content = text * 9000
    context.files = [
        ChangedFile(
            filename="big.py",
            previous_filename=None,
            status="added",
            patch="@@ -0,0 +1,9000 @@\n" + "".join("+" + text for _ in range(9000)),
            before="",
            after=content,
            before_mode=None,
            after_mode="100644",
        )
    ]
    # Plan the oversized file, then require disjoint record coverage and bounded packets.
    result = plan(context)
    assert result.whole is None
    assert len(result.chunks) > 1
    units = [unit for chunk in result.chunks for unit in chunk.units]
    assert "".join(unit.patch for unit in units) == context.files[0].patch
    assert all(len(chunk.payload(result.metadata).encode()) <= 600_000 for chunk in result.chunks)


def test_metadata_overflow_is_not_silently_trimmed() -> None:
    """Required PR prose alone beyond 600,000 bytes makes review explicitly incomplete."""
    import pytest

    context, _, _ = bundle(Server())
    context.description = "🚀" * 150_001
    # A file is needed to carry metadata through the partitioned route.
    context.files = [
        ChangedFile(
            filename="a",
            previous_filename=None,
            status="added",
            patch="@@ -0,0 +1 @@\n+x\n",
            before="",
            after="x\n",
            before_mode=None,
            after_mode="100644",
        )
    ]
    with pytest.raises(ValueError, match="metadata"):
        plan(context)


def test_fragments_keep_unicode_newlines_modes_and_absolute_coordinates() -> None:
    """LF, rather than Unicode separators or CR, determines Git line coordinates."""
    from ai_reviewer.chunking import units
    from ai_reviewer.github import rebuild_patch

    before = ("old\u2028---still-one-line\r\n" * 10_000) + "last"
    after = ("new\u2028+++still-one-line\r\n" * 10_000) + "last\n"
    file = ChangedFile(
        filename="renamed.txt",
        previous_filename="old.txt",
        status="renamed",
        patch=rebuild_patch(before, after),
        before=before,
        after=after,
        before_mode="100644",
        after_mode="100755",
    )
    split = units(file, file_declarations(file))
    assert len(split) > 1
    assert "".join(u.patch for u in split) == file.patch
    for unit in split:
        assert unit.previous_filename == "old.txt"
        assert unit.after_mode == "100755"
        original = before.split("\n")
        count = unit.before.count("\n")
        expected = "\n".join(original[unit.before_start - 1 : unit.before_start - 1 + count])
        if unit.before.endswith("\n"):
            assert unit.before == expected + "\n"
    # An interior removal fragment must start at an absolute old-side coordinate.
    interior = next(u for u in split if u.old_cursor > 1 and u.old_cursor < 9000)
    assert interior.patch.startswith("-")
    assert interior.old_cursor > interior.before_start
    assert any("\\ No newline at end of file" in u.patch for u in split)


def test_packing_is_stable_under_file_inventory_order() -> None:
    """API ordering changes cannot alter partition ownership or chunk ordering."""
    context, _, _ = bundle(Server())
    context.files = [
        ChangedFile(
            filename=f"pkg/{name}.txt",
            previous_filename=None,
            status="added",
            patch="@@ -0,0 +1,2000 @@\n" + ("+" + "a" * 100 + "\n") * 2000,
            before="",
            after=("a" * 100 + "\n") * 2000,
            before_mode=None,
            after_mode="100644",
        )
        for name in ("b", "a", "test_a")
    ]
    # Only inventory order changes; both plans must retain identical ownership.
    first = plan(context)
    context.files.reverse()
    second = plan(context)
    assert first.chunks == second.chunks


def test_large_mode_change_preserves_complete_source() -> None:
    """A patchless executable-bit change still reviews the entire large text blob."""
    from ai_reviewer.chunking import units

    content = "line\n" * 80_000
    file = ChangedFile(
        filename="script",
        previous_filename="old-script",
        status="renamed",
        patch="",
        before=content,
        after=content,
        before_mode="100644",
        after_mode="100755",
    )
    # Split patchless evidence without losing either version or reusing unit IDs.
    split = units(file, file_declarations(file))
    assert len(split) > 1
    assert len({u.id for u in split}) == len(split)
    assert "".join(u.before for u in split) == content
    assert "".join(u.after for u in split) == content


def test_single_oversized_physical_line_fails_explicitly() -> None:
    """An unsplittable line cannot disappear from the planned evidence."""
    import pytest

    from ai_reviewer.chunking import units

    file = ChangedFile(
        filename="huge.txt",
        previous_filename=None,
        status="added",
        patch="@@ -0,0 +1 @@\n+" + "x" * 700_000,
        before="",
        after="x" * 700_000,
        before_mode=None,
        after_mode="100644",
    )
    # Require explicit rejection rather than slicing away an indivisible line.
    with pytest.raises(ValueError, match="Indivisible"):
        units(file, file_declarations(file))


@pytest.mark.parametrize(
    "before,after",
    [
        ("old\n", "new\n"),
        ("", "added"),
        ("removed", ""),
        ("x", "x\n"),
        ("x\ny\nz\n", "x\nz\n"),
        ("x\nz\n", "x\ny\nz\n"),
        ("a\u2028b\r\n", "a\u2028c\r\n"),
    ],
)
def test_patch_validation_accepts_exact_git_changes(before: str, after: str) -> None:
    """Actual Git patches preserve additions, deletions, Unicode and newline changes."""
    from ai_reviewer.chunking import validate_patch
    from ai_reviewer.github import rebuild_patch

    file = ChangedFile(
        filename="a",
        previous_filename=None,
        status="modified",
        patch=rebuild_patch(before, after),
        before=before,
        after=after,
        before_mode="100644",
        after_mode="100644",
    )
    # Validate a real Git-produced patch without changing either source fixture.
    validate_patch(file)
    assert file.before == before and file.after == after


@pytest.mark.parametrize(
    "patch",
    [
        "",
        "@@ -1,2 +1,2 @@\n-old\n+new\n",
        "@@ -1 +1 @@\n-wrong\n+new\n",
        "@@ -1 +1 @@\n-old\n+wrong\n",
        "@@ -1 +1 @@\n-old\n+new\n",
    ],
)
def test_partial_patches_cannot_hide_a_second_change(patch: str) -> None:
    """Every malformed or truncated patch omits the independently known second edit."""
    from ai_reviewer.chunking import validate_patch

    file = ChangedFile(
        filename="a",
        previous_filename=None,
        status="modified",
        patch=patch,
        before="old\nalso old\n",
        after="new\nalso new\n",
        before_mode="100644",
        after_mode="100644",
    )
    with pytest.raises(ValueError, match="patch"):
        validate_patch(file)


def test_direct_reference_outside_changed_hunk_remains_in_integration() -> None:
    """An import at line 1 survives planning when only a distant caller line changes."""
    from ai_reviewer.chunking import direct_relations, relation_key, units
    from ai_reviewer.contracts import Review
    from ai_reviewer.github import rebuild_patch
    from ai_reviewer.work import integration_assignments

    context, _, _ = bundle(Server())
    prefix = "from helpers.target import call\n"
    prefix += ("x = 1 # " + "x" * 100 + "\n") * 3000
    before, after = prefix + "call(1)\n", prefix + "call(2)\n"
    patch = rebuild_patch(before, after)
    # GitHub's normal file patch contains hunks, not Git's file headers.
    patch = patch[patch.index("@@") :]
    main = ChangedFile(
        filename="src/main.py",
        previous_filename=None,
        status="modified",
        patch=patch,
        before=before,
        after=after,
        before_mode="100644",
        after_mode="100644",
    )
    helper = ChangedFile(
        filename="helpers/target.py",
        previous_filename=None,
        status="added",
        patch="@@ -0,0 +1 @@\n+def call(x): return x\n",
        before="",
        after="def call(x): return x\n",
        before_mode=None,
        after_mode="100644",
    )
    context.files = [main, helper]
    split = units(main, file_declarations(main))
    assert all("helpers.target" not in u.after for u in split)
    links = direct_relations(context)
    assert links[relation_key(main.filename, helper.filename)][0].start == 1
    # Force the two known scopes apart to inspect their integration evidence.
    from ai_reviewer.chunking import Chunk

    planned = plan(context).model_copy(
        update={
            "chunks": [
                Chunk(id="chunk-0000", units=split),
                Chunk(id="chunk-0001", units=units(helper, file_declarations(helper))),
            ]
        }
    )
    reports = {
        (c.id, role): Review(complete=True, summary="Local done", findings=[])
        for c in planned.chunks
        for role in ("claude", "openai")
    }
    edges = [
        a
        for a in integration_assignments(context, planned, reports, ("claude", "openai"))
        if a.id.startswith("edge-")
    ]
    assert edges
    import json

    evidence = [
        section["text"] for a in edges for section in json.loads(a.payload)["reference_source"]
    ]
    assert any("from helpers.target import call\n" in text for text in evidence)
    # If packing co-locates the tail and helper, integration must still see the import.
    together = plan(context)
    assert len(together.chunks) == 1
    same_chunk = integration_assignments(context, together, reports, ("claude", "openai"))
    actual = [
        section["text"] for a in same_chunk for section in json.loads(a.payload)["reference_source"]
    ]
    assert any("from helpers.target import call\n" in text for text in actual)


def test_reference_index_matches_complete_tokens_renames_and_first_line_evidence() -> None:
    """Complete path/module matches retain the first reference without suffix collisions."""
    from ai_reviewer.chunking import direct_relations

    context, _, _ = bundle(Server())
    template = ChangedFile(
        filename="unused",
        previous_filename=None,
        status="modified",
        patch="",
        before="",
        after="",
        before_mode="100644",
        after_mode="100644",
    )
    context.files = [
        template.model_copy(
            update={
                "filename": "src/caller.py",
                "before": "prefix\nlib.target\n" + "padding\n" * 40 + "lib.target\n",
                "after": "old/name.py\n" + "padding\n" * 40 + "lib.target\n",
            }
        ),
        template.model_copy(
            update={"filename": "lib/target.py", "previous_filename": "old/name.py"}
        ),
        template.model_copy(update={"filename": "target.py"}),
        template.model_copy(update={"filename": "tests/test_target.py"}),
        template.model_copy(update={"filename": "unrelated.py"}),
    ]
    # Check exact known relationships, then require evidence near the first references.
    links = direct_relations(context)
    assert set(links) == {
        ("lib/target.py", "src/caller.py"),
        ("lib/target.py", "tests/test_target.py"),
        ("target.py", "tests/test_target.py"),
    }
    evidence = links[("lib/target.py", "src/caller.py")]
    assert [(e.path, e.side) for e in evidence] == [
        ("src/caller.py", "LEFT"),
        ("src/caller.py", "RIGHT"),
    ]
    assert evidence[0].start <= 2 <= evidence[0].end < 43
    assert evidence[1].start <= 1 <= evidence[1].end < 42
    context.files.reverse()
    assert direct_relations(context) == links


def test_common_filenames_and_short_substrings_do_not_create_relations() -> None:
    """201 identical stems must not exhaust the 20,000 relationship allowance."""
    from ai_reviewer.chunking import direct_relations

    context, _, _ = bundle(Server())
    template = ChangedFile(
        filename="unused",
        previous_filename=None,
        status="modified",
        patch="",
        before="",
        after="",
        before_mode="100644",
        after_mode="100644",
    )
    context.files = [
        template.model_copy(update={"filename": f"pkg{i}/__init__.py", "before": "", "after": ""})
        for i in range(201)
    ] + [
        template.model_copy(update={"filename": name, "before": "", "after": text})
        for name, text in (("go.mod", ""), ("a.py", ""), ("caller.py", "cargo data setup"))
    ]
    # Reject substring/name-only collisions before adding one complete module token.
    assert direct_relations(context) == {}
    context.files[-1].after = "import a\n"
    assert set(direct_relations(context)) == {("a.py", "caller.py")}


def test_incremental_packet_sizes_match_serialized_unicode_and_shared_comments() -> None:
    """Actual UTF-8 serialization is the oracle for fit checks, including deduplication."""
    from ai_reviewer.chunking import Chunk, PacketSizer, packet_metadata, units
    from ai_reviewer.contracts import PriorComment

    context, _, _ = bundle(Server())
    context.files = [
        ChangedFile(
            filename=name,
            previous_filename=None,
            status="added",
            patch='@@ -0,0 +1 @@\n+🚀"\t\n',
            before="",
            after='🚀"\t\n',
            before_mode=None,
            after_mode="100644",
        )
        for name in ("a.py", "b.py")
    ]
    context.previous_comments = [
        PriorComment.model_validate(
            {"author": "reviewer", "body": 'é"\n', "path": path, "line": None, "side": None}
        )
        for path in ("unchanged.py", "a.py", "b.py")
    ]
    # Compare independent full serialization with incremental admission and duplicate admission.
    metadata = packet_metadata(context)
    evidence = [u for f in context.files for u in units(f, file_declarations(f))]
    sizer = PacketSizer(metadata, evidence)
    budget = sizer.budget("chunk-0000")
    chunk = Chunk(id="chunk-0000", units=[])
    assert budget.size == len(chunk.payload(metadata).encode())
    for unit in evidence:
        budget.include(sizer.additions([unit]))
        chunk.units.append(unit)
        assert budget.size == len(chunk.payload(metadata).encode())
        budget.include(sizer.additions([unit]))
        assert budget.size == len(chunk.payload(metadata).encode())


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_declaration_navigation_keeps_lexical_names_and_git_ranges(newline: str) -> None:
    """Decorators, nested async methods and duplicate names retain distinct LF ranges."""
    from ai_reviewer.chunking import declaration_index

    source = newline.join(
        [
            "raise RuntimeError('must never execute')",
            "@decorate",
            "class Adapter:",
            "    @decorate",
            "    async def handle(self):",
            "        def nested():",
            "            return 1",
            "        return nested()",
            "def handle():",
            "    return 2",
        ]
    )
    assert declaration_index("adapter.py", source) == {
        "status": "indexed",
        "entries": [
            {"name": "Adapter", "start": 2, "end": 8},
            {"name": "Adapter.handle", "start": 4, "end": 8},
            {"name": "Adapter.handle.nested", "start": 6, "end": 7},
            {"name": "handle", "start": 9, "end": 10},
        ],
    }


@pytest.mark.parametrize(
    "filename,source,status",
    [
        ("app.js", "function handle() {}", "unsupported_language"),
        ("app.py", "def handle(:", "parse_error"),
        ("app.py", "\x00", "parse_error"),
        ("app.py", "pass\rdef handle(): pass", "unsupported_line_endings"),
        ("app.py", "", "indexed"),
    ],
)
def test_unavailable_navigation_is_explicit(filename: str, source: str, status: str) -> None:
    """An empty valid file differs from unsupported or malformed captured source."""
    from ai_reviewer.chunking import declaration_index

    assert declaration_index(filename, source) == {"status": status, "entries": []}


def test_parenthesized_decorator_includes_opening_at_line() -> None:
    """Requesting a declaration must include the complete multiline decorator."""
    from ai_reviewer.chunking import declaration_index

    source = "@(\n    decorate\n)\ndef handle():\n    pass\n"
    assert declaration_index("app.py", source) == {
        "status": "indexed",
        "entries": [{"name": "handle", "start": 1, "end": 5}],
    }


@pytest.mark.parametrize("description_size", [0, 48_000])
def test_large_navigation_is_bounded_without_losing_patch_coverage(description_size: int) -> None:
    """Thousands of declarations cannot crowd required source out of the smaller plan."""
    from ai_reviewer.limits import CAPACITY_PACKET_BYTES
    from ai_reviewer.serialization import encode

    context, _, _ = bundle(Server())
    context.description = "x" * description_size
    source = "".join(f"def handler_{n}(): return {n}\n" for n in range(3000))
    context.files = [
        ChangedFile(
            filename="many.py",
            previous_filename=None,
            status="added",
            before="",
            after=source,
            patch="@@ -0,0 +1,3000 @@\n"
            + "".join("+" + line for line in source.splitlines(keepends=True)),
            before_mode=None,
            after_mode="100644",
        )
    ]
    planned = plan(context, packet_bytes=CAPACITY_PACKET_BYTES)
    inventory = planned.metadata.overview["inventory"]
    assert planned.metadata.overview["declaration_navigation"] == "truncated"
    assert len(encode(inventory[0]["declarations"]).encode()) < 16_000
    assert "".join(u.patch for c in planned.chunks for u in c.units) == context.files[0].patch
    assert all(
        len(c.payload(planned.metadata).encode()) <= CAPACITY_PACKET_BYTES for c in planned.chunks
    )


def test_plan_reuses_discovery_for_splitting_and_packet_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan-time parsing happens once per version; packing reuses its complete ranges."""
    import ast

    from ai_reviewer.limits import CAPACITY_PACKET_BYTES

    context, _, _ = bundle(Server())
    source = "@decorate\ndef handle():\n    return 1\n"
    context.files = [
        ChangedFile(
            filename="app.py",
            previous_filename=None,
            status="added",
            before="",
            after=source,
            patch="@@ -0,0 +1,3 @@\n+@decorate\n+def handle():\n+    return 1\n",
            before_mode=None,
            after_mode="100644",
        )
    ]
    parse = ast.parse
    parsed = []

    def record_parse(text: str) -> ast.Module:
        """Record parse inputs while retaining real syntax and coordinate behavior."""
        parsed.append(text)
        return parse(text)

    monkeypatch.setattr(ast, "parse", record_parse)
    planned = plan(context, packet_bytes=CAPACITY_PACKET_BYTES)
    payloads = [c.payload(planned.metadata) for c in planned.chunks]
    assert parsed == ["", source]
    assert '"name":"handle","start":1,"end":3' in payloads[0]


def test_navigation_projection_does_not_mutate_captured_metadata() -> None:
    """Small and full projections remain independent of call order and shared sizers."""
    from copy import deepcopy

    from ai_reviewer.chunking import bound_navigation

    captured = {
        "inventory": [
            {
                "filename": "app.py",
                "declarations": {
                    "RIGHT": {
                        "status": "indexed",
                        "entries": [{"name": "handle", "start": 1, "end": 3}],
                    }
                },
            }
        ]
    }
    original = deepcopy(captured)
    omitted = bound_navigation(captured, 0)
    full = bound_navigation(captured, 16000)
    assert captured == original
    assert omitted == {"inventory": [{"filename": "app.py"}], "declaration_navigation": "truncated"}
    assert full == {**original, "declaration_navigation": "complete"}


def test_markdown_navigation_reaches_contract_beyond_previous_excerpt() -> None:
    """The reported 500/549 gap is discoverable on each captured side after a rename."""
    from ai_reviewer.chunking import overview

    context, _, _ = bundle(Server())
    source = "\n" * 535 + "### Task-free parent operations\n" + "\n" * 12
    source += "External file edits contract\n\n### Next section\n"
    context.files = [
        ChangedFile(
            filename="docs/host-hooks.md",
            previous_filename="docs/old.markdown",
            status="renamed",
            patch="",
            before=source,
            after="\n" + source,
            before_mode="100644",
            after_mode="100644",
        )
    ]
    item = overview(context)["inventory"][0]
    assert item["sections"]["LEFT"]["entries"][0] == {
        "name": "Task-free parent operations",
        "level": 3,
        "start": 536,
        "end": 550,
    }
    assert item["sections"]["RIGHT"]["entries"][0] == {
        "name": "Task-free parent operations",
        "level": 3,
        "start": 537,
        "end": 551,
    }
    assert source.splitlines()[548] == "External file edits contract"


def test_markdown_navigation_shares_byte_budget_without_mutating_source() -> None:
    """Section omissions are explicit and indexed ranges survive metadata projection."""
    from copy import deepcopy

    from ai_reviewer.chunking import bound_navigation, omit_navigation
    from ai_reviewer.markdown_navigation import section_index
    from ai_reviewer.serialization import encode

    index = section_index("guide.md", "# Root\n" + "## Child\n" * 100)
    data = {"inventory": [{"filename": "guide.md", "sections": {"RIGHT": index}}]}
    original = deepcopy(data)
    bounded = bound_navigation(data, 400)
    assert bounded["declaration_navigation"] == "truncated"
    entries = bounded["inventory"][0]["sections"]["RIGHT"]["entries"]
    assert 0 < len(entries) < 101
    assert entries[0] == {"name": "Root", "level": 1, "start": 1, "end": 101}
    assert len(encode(bounded).encode()) - len(encode(omit_navigation(data)).encode()) <= 400
    assert bound_navigation(data, 0) == {
        "inventory": [{"filename": "guide.md"}],
        "declaration_navigation": "truncated",
    }
    assert data == original
