"""Check packet ownership and approval gates against explicit synthetic changes."""

import pytest

from ai_reviewer.chunking import plan
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
    split = units(file)
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
    split = units(file)
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
    with pytest.raises(ValueError, match="Indivisible"):
        units(file)


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
    split = units(main)
    assert all("helpers.target" not in u.after for u in split)
    links = direct_relations(context)
    assert links[relation_key(main.filename, helper.filename)][0].start == 1
    # Force the two known scopes apart to inspect their integration evidence.
    from ai_reviewer.chunking import Chunk

    planned = plan(context).model_copy(
        update={
            "chunks": [
                Chunk(id="chunk-0000", units=split),
                Chunk(id="chunk-0001", units=units(helper)),
            ]
        }
    )
    reports = {
        (c.id, role): Review(complete=True, summary="Local done", findings=[])
        for c in planned.chunks
        for role in ("claude", "openai")
    }
    edges = [
        a for a in integration_assignments(context, planned, reports) if a.id.startswith("edge-")
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
    same_chunk = integration_assignments(context, together, reports)
    actual = [
        section["text"] for a in same_chunk for section in json.loads(a.payload)["reference_source"]
    ]
    assert any("from helpers.target import call\n" in text for text in actual)


def test_reference_index_retains_overlaps_renames_and_first_line_evidence() -> None:
    """Literal path/module matches retain the first reference, even with shared suffixes."""
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
    links = direct_relations(context)
    assert set(links) == {
        ("lib/target.py", "src/caller.py"),
        ("src/caller.py", "target.py"),
        ("lib/target.py", "target.py"),
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
