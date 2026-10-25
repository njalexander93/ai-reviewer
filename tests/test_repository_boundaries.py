"""Protect immutable revision identity and host isolation at the source-tool boundary."""

import json
from pathlib import Path

import pytest

from ai_reviewer.repository import RepositorySnapshotError, load_repository
from tests.test_repository import make_repository


@pytest.mark.parametrize("revision", ["head", "base", "diff_base"])
def test_tools_reject_changed_revision_identity(tmp_path: Path, revision: str) -> None:
    """Changing any captured revision cannot reuse a previously saved source index."""
    context, _ = make_repository(tmp_path)
    setattr(context, revision, "f" * 40)
    with pytest.raises(RepositorySnapshotError, match="revisions"):
        load_repository(context, tmp_path / "repository")


def test_reads_ignore_working_files_and_reject_blob_symlinks(tmp_path: Path) -> None:
    """Source comes from pinned blobs; even a matching external symlink is rejected."""
    _, repository = make_repository(tmp_path)
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/roles.py").write_text("DEFAULT_ROLE = 'modified'\n")
    arguments = {"revision": "head", "path": "lib/roles.py", "start": 1, "end": 1}
    assert json.loads(repository.invoke("read_repository_lines", arguments))["text"] == (
        "DEFAULT_ROLE = 'reviewer'\n"
    )
    entry = repository.entries["head"]["lib/roles.py"]
    blob = repository.blob_dir / entry.blob
    outside = tmp_path / "outside.txt"
    outside.write_bytes(blob.read_bytes())
    blob.unlink()
    blob.symlink_to(outside)
    with pytest.raises(RepositorySnapshotError, match="unsafe"):
        repository.invoke("read_repository_lines", arguments)


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside.txt", "lib/../outside.txt"])
def test_model_paths_cannot_address_host_files(tmp_path: Path, path: str) -> None:
    """Absolute and traversal paths fail before a source file can be opened."""
    _, repository = make_repository(tmp_path)
    result = json.loads(
        repository.invoke(
            "read_repository_lines", {"revision": "head", "path": path, "start": 1, "end": 1}
        )
    )
    assert "error" in result
    assert "text" not in result
