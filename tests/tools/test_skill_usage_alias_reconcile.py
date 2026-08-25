"""RED contract tests for canonical skill-usage aliases.

These tests intentionally target behavior that is not present on upstream/main.
They exercise the real telemetry sidecar and the future ``reconcile-usage`` CLI
without adding a production implementation in this change.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def usage_home(tmp_path, monkeypatch):
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
    skills_dir: Path,
    *,
    category: str,
    directory: str,
    frontmatter_name: str,
) -> Path:
    skill_dir = skills_dir / category / directory
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {frontmatter_name}\n"
        "description: alias reconciliation fixture\n"
        "---\n\n"
        "# fixture\n",
        encoding="utf-8",
    )
    return skill_dir


def _write_usage(home: Path, records: dict) -> Path:
    path = home / "skills" / ".usage.json"
    path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    return path


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


def test_all_telemetry_writes_resolve_aliases_to_frontmatter_name(usage_home):
    """Bare, category, path, directory, and frontmatter aliases share one key."""
    from tools import skill_usage

    skill_dir = _write_skill(
        usage_home / "skills",
        category="research",
        directory="directory-alias",
        frontmatter_name="canonical-skill",
    )
    skill_md = skill_dir / "SKILL.md"

    # Each write path is intentionally different so every telemetry writer is
    # covered, including the two trusted absolute-path forms.
    aliases_and_writers = [
        ("canonical-skill", skill_usage.bump_view),
        ("research/canonical-skill", skill_usage.bump_use),
        ("research:canonical-skill", skill_usage.bump_patch),
        (str(skill_dir), skill_usage.bump_view),
        (str(skill_md), skill_usage.bump_use),
        ("directory-alias", skill_usage.bump_patch),
    ]
    for alias, writer in aliases_and_writers:
        writer(alias)

    data = skill_usage.load_usage()
    assert set(data) == {"canonical-skill"}
    assert data["canonical-skill"]["view_count"] == 2
    assert data["canonical-skill"]["use_count"] == 2
    assert data["canonical-skill"]["patch_count"] == 2


def test_ambiguous_and_unknown_aliases_keep_events_without_collapsing(usage_home):
    """Fail-safe lookup must preserve an event when no unique local target exists."""
    from tools import skill_usage

    skills = usage_home / "skills"
    _write_skill(
        skills,
        category="one",
        directory="same-dir",
        frontmatter_name="ambiguous-skill",
    )
    _write_skill(
        skills,
        category="two",
        directory="same-dir",
        frontmatter_name="ambiguous-skill",
    )

    # Neither alias can be safely assigned to one local skill. The original
    # event key is the lossless fallback; it must not disappear or be guessed.
    skill_usage.bump_view("ambiguous-skill")
    skill_usage.bump_use("unknown-skill-alias")

    data = skill_usage.load_usage()
    assert data["ambiguous-skill"]["view_count"] == 1
    assert data["unknown-skill-alias"]["use_count"] == 1


def test_plugin_namespace_is_never_collapsed_into_local_bare_name(usage_home):
    from tools import skill_usage

    _write_skill(
        usage_home / "skills",
        category="local",
        directory="skill",
        frontmatter_name="skill",
    )
    skill_usage.bump_view("plugin:skill")
    skill_usage.bump_use("plugin:skill")
    skill_usage.bump_patch("plugin:skill")

    data = skill_usage.load_usage()
    assert "plugin:skill" in data
    assert data["plugin:skill"]["view_count"] == 1
    assert data["plugin:skill"]["use_count"] == 1
    assert data["plugin:skill"]["patch_count"] == 1
    assert "skill" not in data


def test_identity_cache_discovers_new_category_path_alias_without_root_touch(
    usage_home,
):
    from tools import skill_usage

    skills = usage_home / "skills"
    _write_skill(
        skills,
        category="cat",
        directory="one",
        frontmatter_name="one",
    )

    # Index the category while it contains only the first skill. A later skill
    # is created below the existing category directory, so the skills-root
    # signature itself does not change.
    assert skill_usage._canonicalize_skill_name("cat/one") == "one"
    root_signature = skill_usage._skills_root_signature(skills)
    _write_skill(
        skills,
        category="cat",
        directory="two",
        frontmatter_name="two",
    )
    assert skill_usage._skills_root_signature(skills) == root_signature

    assert skill_usage._canonicalize_skill_name("cat/two") == "two"
    skill_usage.bump_use("cat/two")
    data = skill_usage.load_usage()
    assert data["two"]["use_count"] == 1
    assert "cat/two" not in data


def test_identity_cache_watches_empty_category_after_last_skill_deleted(
    usage_home, monkeypatch
):
    from agent import skill_utils
    from tools import skill_usage

    skills = usage_home / "skills"
    first = _write_skill(
        skills,
        category="cat",
        directory="one",
        frontmatter_name="one",
    )
    assert skill_usage._canonicalize_skill_name("cat/one") == "one"

    (first / "SKILL.md").unlink()
    first.rmdir()
    assert skill_usage._canonicalize_skill_name("cat/one") == "cat/one"

    root_signature = skill_usage._skills_root_signature(skills)
    _write_skill(
        skills,
        category="cat",
        directory="two",
        frontmatter_name="two",
    )
    assert skill_usage._skills_root_signature(skills) == root_signature
    assert skill_usage._canonicalize_skill_name("cat/two") == "two"

    real_iter = skill_utils.iter_skill_index_files
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_iter(*args, **kwargs)

    monkeypatch.setattr(skill_utils, "iter_skill_index_files", counted)
    assert skill_usage._canonicalize_skill_name("cat/two") == "two"
    assert calls == 0


def test_reconcile_cli_is_registered_and_defaults_to_report_only(usage_home):
    """The new command must exist before its filesystem behavior is exercised."""
    import argparse

    from hermes_cli import curator

    parser = argparse.ArgumentParser(prog="hermes curator")
    curator.register_cli(parser)
    try:
        args = parser.parse_args(["reconcile-usage"])
    except SystemExit as exc:  # current upstream: command is absent
        pytest.fail(
            "reconcile-usage is not registered; expected a report-only default "
            f"(argparse exit {exc.code})"
        )
    assert args.curator_command == "reconcile-usage"
    assert args.apply is False


def test_reconcile_cli_has_explicit_apply_switch(usage_home):
    import argparse

    from hermes_cli import curator

    parser = argparse.ArgumentParser(prog="hermes curator")
    curator.register_cli(parser)
    try:
        args = parser.parse_args(["reconcile-usage", "--apply"])
    except SystemExit as exc:
        pytest.fail(
            "reconcile-usage --apply is not registered; explicit apply is "
            f"required (argparse exit {exc.code})"
        )
    assert args.apply is True
