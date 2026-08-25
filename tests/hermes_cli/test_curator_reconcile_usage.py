"""RED tests for ``hermes curator reconcile-usage``.

The command is intentionally absent from upstream/main.  The assertions below
pin the on-disk contract before any production implementation is added.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import stat
from pathlib import Path

import pytest


@pytest.fixture
def reconcile_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    import tools.skill_usage as skill_usage

    importlib.reload(hermes_constants)
    importlib.reload(skill_usage)
    monkeypatch.setattr(skill_usage, "_prune_builtins_enabled", lambda: False)
    return home


def _write_skill(
    home: Path,
    *,
    category: str,
    directory: str,
    name: str,
) -> Path:
    skill_dir = home / "skills" / category / directory
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: reconcile fixture\n"
        "---\n\n"
        "# fixture\n",
        encoding="utf-8",
    )
    return skill_dir


def _record(**overrides):
    record = {
        "created_by": None,
        "use_count": 0,
        "view_count": 0,
        "patch_count": 0,
        "last_used_at": None,
        "last_viewed_at": None,
        "last_patched_at": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "patch_generation": 0,
        "last_reused_patch_generation": 0,
        "state": "active",
        "pinned": False,
        "archived_at": None,
    }
    record.update(overrides)
    return record


def _usage_path(home: Path) -> Path:
    return home / "skills" / ".usage.json"


def _write_usage(home: Path, records: dict) -> Path:
    path = _usage_path(home)
    path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _read_usage(home: Path) -> dict:
    return json.loads(_usage_path(home).read_text(encoding="utf-8"))


def _run_reconcile(*, apply: bool) -> int:
    """Invoke the CLI boundary, turning absent parser wiring into a RED assert."""
    from hermes_cli import curator

    argv = ["reconcile-usage"] + (["--apply"] if apply else [])
    try:
        return curator.cli_main(argv)
    except SystemExit as exc:  # argparse's current upstream behavior
        pytest.fail(
            "reconcile-usage is not implemented/registered; expected a "
            f"controlled command result (argparse exit {exc.code})"
        )


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    snapshot = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            snapshot[str(path.relative_to(root))] = (
                path.read_bytes(),
                stat.S_IMODE(path.stat().st_mode),
            )
    return snapshot


def _matching_backup_files(home: Path, original: bytes) -> list[Path]:
    return [
        path
        for path in home.rglob("*")
        if path.is_file()
        and path != _usage_path(home)
        and path.read_bytes() == original
        and path.parent != home / "skills"
    ]


def test_default_reconcile_is_dry_run_and_hash_invariant(reconcile_home, capsys):
    skill_dir = _write_skill(
        reconcile_home,
        category="research",
        directory="directory-alias",
        name="canonical-skill",
    )
    usage = _write_usage(
        reconcile_home,
        {
            "research/directory-alias": _record(
                use_count=2,
                created_by="agent",
                token="DO-NOT-PRINT",
            ),
            str(skill_dir): _record(view_count=3, state="stale"),
        },
    )
    before_bytes = usage.read_bytes()
    before_hash = hashlib.sha256(before_bytes).hexdigest()
    before_tree = _tree_snapshot(reconcile_home)

    assert _run_reconcile(apply=False) == 0

    assert usage.read_bytes() == before_bytes
    assert hashlib.sha256(usage.read_bytes()).hexdigest() == before_hash
    assert _tree_snapshot(reconcile_home) == before_tree
    captured = capsys.readouterr()
    report = captured.out + captured.err
    assert "DO-NOT-PRINT" not in report


def test_apply_merges_aliases_and_removes_keys_only_after_private_backup(
    reconcile_home,
    capsys,
):
    skill_dir = _write_skill(
        reconcile_home,
        category="research",
        directory="directory-alias",
        name="canonical-skill",
    )
    original = {
        "research/directory-alias": _record(
            use_count=2,
            view_count=3,
            patch_count=4,
            created_at="2026-01-03T00:00:00+00:00",
            last_used_at="2026-02-01T00:00:00+00:00",
            patch_generation=2,
            last_reused_patch_generation=2,
            created_by="agent",
            pinned=False,
            state="archived",
            archived_at="2026-02-02T00:00:00+00:00",
            token="DO-NOT-PRINT",
        ),
        "research:canonical-skill": _record(
            use_count=5,
            view_count=7,
            patch_count=11,
            created_at="2026-01-01T00:00:00+00:00",
            last_viewed_at="2026-03-01T00:00:00+00:00",
            last_patched_at="2026-04-01T00:00:00+00:00",
            patch_generation=5,
            last_reused_patch_generation=99,
            created_by="installed",
            pinned=True,
            state="stale",
            archived_at=None,
            token="DO-NOT-PRINT",
        ),
        str(skill_dir / "SKILL.md"): _record(
            use_count=1,
            created_at="2026-01-02T00:00:00+00:00",
            last_used_at="2026-02-15T00:00:00+00:00",
            state="active",
        ),
    }
    usage = _write_usage(reconcile_home, original)
    before_bytes = usage.read_bytes()

    assert _run_reconcile(apply=True) == 0

    merged = _read_usage(reconcile_home)
    assert set(merged) == {"canonical-skill"}
    record = merged["canonical-skill"]
    assert record["use_count"] == 8
    assert record["view_count"] == 10
    assert record["patch_count"] == 15
    assert record["created_at"] == "2026-01-01T00:00:00+00:00"
    assert record["last_used_at"] == "2026-02-15T00:00:00+00:00"
    assert record["last_viewed_at"] == "2026-03-01T00:00:00+00:00"
    assert record["last_patched_at"] == "2026-04-01T00:00:00+00:00"
    assert record["patch_generation"] == 5
    assert record["last_reused_patch_generation"] == 5
    # Provenance follows the actual local skill. A stale alias marked
    # ``installed`` must not evict an existing ``agent`` ownership marker and
    # silently remove the skill from Curator management.
    assert record["created_by"] == "agent"
    assert record["pinned"] is True
    # active outranks stale and archived; archived_at is only legal for an
    # archived result.
    assert record["state"] == "active"
    assert record["archived_at"] is None

    backups = _matching_backup_files(reconcile_home, before_bytes)
    assert backups, "apply must create a byte-identical private backup first"
    backup = backups[0]
    assert stat.S_IMODE(backup.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    report = capsys.readouterr().out
    assert "DO-NOT-PRINT" not in report


def test_archived_group_retains_archived_at_only_when_result_is_archived(reconcile_home):
    _write_skill(
        reconcile_home,
        category="research",
        directory="directory-alias",
        name="archived-skill",
    )
    _write_usage(
        reconcile_home,
        {
            "research/directory-alias": _record(
                state="archived",
                archived_at="2026-02-01T00:00:00+00:00",
            ),
            "research:archived-skill": _record(
                state="archived",
                archived_at="2026-03-01T00:00:00+00:00",
            ),
        },
    )

    assert _run_reconcile(apply=True) == 0

    merged = _read_usage(reconcile_home)["archived-skill"]
    assert merged["state"] == "archived"
    assert merged["archived_at"] is not None


def test_apply_is_idempotent_without_a_second_backup(reconcile_home):
    skill_dir = _write_skill(
        reconcile_home,
        category="research",
        directory="directory-alias",
        name="canonical-skill",
    )
    usage = _write_usage(
        reconcile_home,
        {
            "research/directory-alias": _record(use_count=3),
            str(skill_dir): _record(view_count=4),
        },
    )
    original_bytes = usage.read_bytes()

    assert _run_reconcile(apply=True) == 0
    after_first = usage.read_bytes()
    backups_after_first = _matching_backup_files(reconcile_home, original_bytes)
    assert backups_after_first

    assert _run_reconcile(apply=True) == 0
    assert usage.read_bytes() == after_first
    assert _matching_backup_files(reconcile_home, original_bytes) == backups_after_first


def test_conflicting_unknown_fields_block_apply_without_mutation(reconcile_home):
    _write_skill(
        reconcile_home,
        category="research",
        directory="directory-alias",
        name="canonical-skill",
    )
    usage = _write_usage(
        reconcile_home,
        {
            "research/directory-alias": _record(extension_payload="one"),
            "research:canonical-skill": _record(extension_payload="two"),
        },
    )
    before = usage.read_bytes()

    assert _run_reconcile(apply=True) != 0
    assert usage.read_bytes() == before
    assert set(_read_usage(reconcile_home)) == {
        "research/directory-alias",
        "research:canonical-skill",
    }


def test_ambiguous_unknown_and_plugin_groups_are_preserved_while_safe_groups_merge(
    reconcile_home,
):
    _write_skill(
        reconcile_home,
        category="one",
        directory="same-dir",
        name="ambiguous-skill",
    )
    _write_skill(
        reconcile_home,
        category="two",
        directory="same-dir",
        name="ambiguous-skill",
    )
    _write_skill(
        reconcile_home,
        category="local",
        directory="skill",
        name="skill",
    )
    _write_skill(
        reconcile_home,
        category="safe",
        directory="directory-alias",
        name="safe-skill",
    )
    _write_usage(
        reconcile_home,
        {
            "ambiguous-skill": _record(use_count=1),
            "unknown-alias": _record(view_count=1),
            "plugin:skill": _record(patch_count=1),
            "skill": _record(use_count=2),
            "safe/directory-alias": _record(use_count=3),
            "safe:safe-skill": _record(view_count=4),
        },
    )

    assert _run_reconcile(apply=True) == 0

    after = _read_usage(reconcile_home)
    assert after["ambiguous-skill"]["use_count"] == 1
    assert after["unknown-alias"]["view_count"] == 1
    assert after["plugin:skill"]["patch_count"] == 1
    assert after["skill"]["use_count"] == 2
    assert "plugin:skill" in after
    assert "skill" in after
    assert "safe/directory-alias" not in after
    assert "safe:safe-skill" not in after
    assert after["safe-skill"]["use_count"] == 3
    assert after["safe-skill"]["view_count"] == 4


@pytest.mark.parametrize("payload", [b"{not-json", b"[]", b'{"alias": []}'])
def test_corrupted_or_malformed_sidecar_fails_controlled_and_preserves_bytes(
    reconcile_home,
    payload,
    capsys,
):
    usage = _usage_path(reconcile_home)
    usage.write_bytes(payload)

    assert _run_reconcile(apply=True) != 0
    assert usage.read_bytes() == payload
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "Traceback" not in output
    assert "not-json" not in output


def test_apply_save_usage_false_without_write_is_nonzero_and_preserves_source(
    reconcile_home,
    monkeypatch,
):
    skill_dir = _write_skill(
        reconcile_home,
        category="research",
        directory="directory-alias",
        name="canonical-skill",
    )
    usage = _write_usage(
        reconcile_home,
        {
            "research/directory-alias": _record(use_count=2),
            str(skill_dir): _record(view_count=3),
        },
    )
    before_bytes = usage.read_bytes()

    from tools import skill_usage

    monkeypatch.setattr(skill_usage, "save_usage", lambda data: False)

    apply_rc = _run_reconcile(apply=True)
    assert apply_rc != 0
    assert usage.read_bytes() == before_bytes
    assert set(_read_usage(reconcile_home)) == {
        "research/directory-alias",
        str(skill_dir),
    }


def test_apply_accepts_save_usage_false_after_original_commit(
    reconcile_home,
    monkeypatch,
):
    skill_dir = _write_skill(
        reconcile_home,
        category="research",
        directory="directory-alias",
        name="canonical-skill",
    )
    usage = _write_usage(
        reconcile_home,
        {
            "research/directory-alias": _record(use_count=2),
            str(skill_dir): _record(view_count=3),
        },
    )

    from tools import skill_usage

    original_save_usage = skill_usage.save_usage

    def save_then_report_false(data):
        assert original_save_usage(data) is True
        return False

    monkeypatch.setattr(skill_usage, "save_usage", save_then_report_false)

    assert _run_reconcile(apply=True) == 0
    assert set(_read_usage(reconcile_home)) == {"canonical-skill"}
    assert stat.S_IMODE(usage.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "\\x01canonical-skill",
        "\\x1b[31mcanonical-skill\\x1b[0m",
        "canonical\\tname",
        "canonical\\x7fname",
        "canonical\\x80name",
    ],
)
def test_cli_reconcile_never_emits_control_chars_in_canonical_name(
    reconcile_home,
    monkeypatch,
    capsys,
    unsafe_name,
):
    unsafe_name = bytes(unsafe_name, "ascii").decode("unicode_escape")
    from hermes_cli import curator
    from tools import skill_usage

    monkeypatch.setattr(
        skill_usage,
        "reconcile_usage_report",
        lambda: {
            "status": "ok",
            "counts": {
                "records": 1,
                "groups": 1,
                "aliases": 1,
                "possible_conflicts": 0,
                "skipped": 0,
            },
            "skipped": {"unknown": 0, "ambiguous": 0, "plugin": 0},
            "groups": [
                {
                    "canonical": unsafe_name,
                    "aliases": ["research/directory-alias"],
                    "record_count": 1,
                    "possible_conflicts": 0,
                }
            ],
        },
    )

    control = next(
        char
        for char in unsafe_name
        if ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F
    )
    safe_name = curator._safe_reconcile_display_name(unsafe_name)
    assert safe_name is None or control not in safe_name

    assert _run_reconcile(apply=False) == 0
    output = capsys.readouterr().out
    assert control not in output


def test_unknown_extension_numeric_types_conflict_in_apply_and_dry_run(
    reconcile_home,
    capsys,
):
    _write_skill(
        reconcile_home,
        category="research",
        directory="directory-alias",
        name="canonical-skill",
    )
    usage = _write_usage(
        reconcile_home,
        {
            "research/directory-alias": _record(extension_value=1),
            "research:canonical-skill": _record(extension_value=1.0),
        },
    )
    before_bytes = usage.read_bytes()

    assert _run_reconcile(apply=False) == 0
    report = capsys.readouterr().out
    assert "possible conflicts: 1" in report

    apply_rc = _run_reconcile(apply=True)
    after_bytes = usage.read_bytes()
    assert apply_rc != 0 and after_bytes == before_bytes
