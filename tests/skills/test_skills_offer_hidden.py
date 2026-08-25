"""RED regressions for the ``skills.offer_hidden`` offer-time policy.

The production feature is intentionally not implemented in this branch.  These
regressions define the contract for a canonical-name list under
``skills.offer_hidden``:

* hidden skills disappear from offer surfaces (prompt index, ``skills_list``,
  slash/autocomplete);
* explicit ``skill_view`` and ``--skills`` preload remain usable;
* ``skills.disabled`` remains a hard block;
* caches notice config changes;
* malformed ``offer_hidden`` values fail safe; and
* the administrative installed-skill listing keeps hidden skills visible while
  identifying them as on-demand.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

import agent.prompt_builder as prompt_builder
import agent.skill_commands as skill_commands
import agent.skill_utils as skill_utils
import tools.skills_tool as skills_tool


_MISSING = object()


def _write_skill(
    skills_dir: Path,
    directory_name: str,
    canonical_name: str,
    *,
    description: str | None = None,
) -> Path:
    skill_dir = skills_dir / directory_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    description = description or f"Description for {canonical_name}."
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {canonical_name}\n"
        f"description: {description}\n"
        "---\n\n"
        f"# {canonical_name}\n\n"
        f"Instructions for {canonical_name}.\n",
        encoding="utf-8",
    )
    return skill_dir


def _write_config(
    home: Path,
    *,
    offer_hidden=_MISSING,
    disabled=_MISSING,
) -> None:
    lines = ["skills:"]
    if offer_hidden is not _MISSING:
        if isinstance(offer_hidden, list):
            lines.append("  offer_hidden:")
            lines.extend(f"    - {name}" for name in offer_hidden)
        else:
            lines.append(f"  offer_hidden: {offer_hidden}")
    if disabled is not _MISSING:
        if isinstance(disabled, list):
            lines.append("  disabled:")
            lines.extend(f"    - {name}" for name in disabled)
        else:
            lines.append(f"  disabled: {disabled}")
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Keep this helper from making the raw-config cache, rather than the
    # offer-hidden cache boundary, determine the result of a test mutation.
    skill_utils._raw_config_cache_clear()


@pytest.fixture()
def isolated_skills(monkeypatch, tmp_path):
    home = tmp_path / "home"
    skills_dir = home / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)

    # Keep every surface on the same small, profile-local tree.  In particular,
    # do not let the checkout's project skills or the user's real profile enter
    # these contract tests.
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skills_tool, "_skills_dir", lambda: skills_dir)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [])
    monkeypatch.setattr(skill_utils, "get_project_skills_dirs", lambda: [])
    monkeypatch.setattr(prompt_builder, "get_all_skills_dirs", lambda: [skills_dir])

    skills_tool._SKILLS_CACHE.clear()
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
    skill_utils._raw_config_cache_clear()
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)
    monkeypatch.setattr(skill_commands, "_skill_commands_home", None)

    yield home, skills_dir

    skills_tool._SKILLS_CACHE.clear()
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
    skill_utils._raw_config_cache_clear()


def _names_from_skills_list(raw: str) -> set[str]:
    payload = json.loads(raw)
    assert payload["success"] is True
    return {entry["name"] for entry in payload["skills"]}


def test_offer_hidden_excludes_canonical_name_from_system_prompt_index(
    isolated_skills,
):
    home, skills_dir = isolated_skills
    _write_skill(skills_dir, "alias-hidden", "hidden-canonical")
    _write_skill(skills_dir, "visible-dir", "visible-skill")
    _write_config(home, offer_hidden=["hidden-canonical"])

    prompt = prompt_builder.build_skills_system_prompt(
        skills_dir_override=skills_dir,
    )

    assert "visible-skill" in prompt
    assert "hidden-canonical" not in prompt


def test_offer_hidden_excludes_canonical_name_from_skills_list_tool(
    isolated_skills,
):
    home, skills_dir = isolated_skills
    _write_skill(skills_dir, "alias-hidden", "hidden-canonical")
    _write_skill(skills_dir, "visible-dir", "visible-skill")
    _write_config(home, offer_hidden=["hidden-canonical"])

    offered_names = _names_from_skills_list(skills_tool.skills_list())

    assert "visible-skill" in offered_names
    assert "hidden-canonical" not in offered_names


def test_offer_hidden_excludes_canonical_name_from_slash_and_autocomplete(
    isolated_skills,
):
    home, skills_dir = isolated_skills
    _write_skill(skills_dir, "alias-hidden", "hidden-canonical")
    _write_skill(skills_dir, "visible-dir", "visible-skill")
    _write_config(home, offer_hidden=["hidden-canonical"])

    commands = skill_commands.scan_skill_commands()
    autocomplete_commands = skill_commands.get_skill_commands()

    assert "/visible-skill" in commands
    assert "/hidden-canonical" not in commands
    assert "/visible-skill" in autocomplete_commands
    assert "/hidden-canonical" not in autocomplete_commands


def test_offer_hidden_only_changes_offers_explicit_view_and_preload_still_work(
    isolated_skills,
):
    home, skills_dir = isolated_skills
    _write_skill(skills_dir, "alias-hidden", "hidden-canonical")
    _write_config(home, offer_hidden=["hidden-canonical"])

    commands = skill_commands.scan_skill_commands()
    viewed = json.loads(skills_tool.skill_view("hidden-canonical"))
    preloaded_prompt, loaded_names, missing = skill_commands.build_preloaded_skills_prompt(
        ["hidden-canonical"]
    )

    assert "/hidden-canonical" not in commands
    assert viewed["success"] is True
    assert "Instructions for hidden-canonical." in viewed["content"]
    assert loaded_names == ["hidden-canonical"]
    assert missing == []
    assert "hidden-canonical" in preloaded_prompt


def test_skills_disabled_remains_hard_block_even_when_offer_hidden_is_set(
    isolated_skills,
):
    home, skills_dir = isolated_skills
    _write_skill(skills_dir, "alias-hidden", "hidden-canonical")
    _write_config(
        home,
        offer_hidden=["hidden-canonical"],
        disabled=["hidden-canonical"],
    )

    prompt = prompt_builder.build_skills_system_prompt(
        skills_dir_override=skills_dir,
    )
    offered_names = _names_from_skills_list(skills_tool.skills_list())
    commands = skill_commands.scan_skill_commands()
    viewed = json.loads(skills_tool.skill_view("hidden-canonical"))
    _preloaded_prompt, loaded_names, missing = skill_commands.build_preloaded_skills_prompt(
        ["hidden-canonical"]
    )

    assert "hidden-canonical" not in prompt
    assert "hidden-canonical" not in offered_names
    assert "/hidden-canonical" not in commands
    assert viewed["success"] is False
    assert "disabled" in viewed["error"].lower()
    assert loaded_names == []
    assert missing == ["hidden-canonical"]


def test_offer_hidden_config_change_invalidates_system_prompt_cache(
    isolated_skills,
):
    home, skills_dir = isolated_skills
    _write_skill(skills_dir, "alias-hidden", "hidden-canonical")
    _write_config(home, offer_hidden=[])

    before = prompt_builder.build_skills_system_prompt(
        skills_dir_override=skills_dir,
    )
    assert "hidden-canonical" in before

    _write_config(home, offer_hidden=["hidden-canonical"])
    after = prompt_builder.build_skills_system_prompt(
        skills_dir_override=skills_dir,
    )

    assert "hidden-canonical" not in after


def test_offer_hidden_config_change_invalidates_slash_cache(isolated_skills):
    home, skills_dir = isolated_skills
    _write_skill(skills_dir, "alias-hidden", "hidden-canonical")
    _write_config(home, offer_hidden=[])

    before = skill_commands.get_skill_commands()
    assert "/hidden-canonical" in before

    _write_config(home, offer_hidden=["hidden-canonical"])
    after = skill_commands.get_skill_commands()

    assert "/hidden-canonical" not in after


def test_offer_hidden_config_change_invalidates_skills_list_cache(isolated_skills):
    home, skills_dir = isolated_skills
    _write_skill(skills_dir, "alias-hidden", "hidden-canonical")
    _write_config(home, offer_hidden=[])

    before = _names_from_skills_list(skills_tool.skills_list())
    assert "hidden-canonical" in before

    _write_config(home, offer_hidden=["hidden-canonical"])
    after = _names_from_skills_list(skills_tool.skills_list())

    assert "hidden-canonical" not in after


@pytest.mark.parametrize(
    "malformed_value",
    [
        "hidden-canonical",
        "{unexpected: value}",
        "42",
    ],
)
def test_malformed_offer_hidden_fails_safe_without_hiding_skills(
    isolated_skills,
    malformed_value,
):
    home, skills_dir = isolated_skills
    _write_skill(skills_dir, "alias-hidden", "hidden-canonical")
    _write_config(home, offer_hidden=["hidden-canonical"])

    hidden_prompt = prompt_builder.build_skills_system_prompt(
        skills_dir_override=skills_dir,
    )
    assert "hidden-canonical" not in hidden_prompt

    _write_config(home, offer_hidden=malformed_value)
    restored_prompt = prompt_builder.build_skills_system_prompt(
        skills_dir_override=skills_dir,
    )
    restored_names = _names_from_skills_list(skills_tool.skills_list())
    restored_commands = skill_commands.get_skill_commands()

    assert "hidden-canonical" in restored_prompt
    assert "hidden-canonical" in restored_names
    assert "/hidden-canonical" in restored_commands


def test_admin_list_keeps_installed_offer_hidden_skill_and_marks_on_demand(
    isolated_skills,
    monkeypatch,
):
    home, _skills_dir = isolated_skills
    _write_config(home, offer_hidden=["hidden-canonical"])

    import hermes_cli.skills_hub as cli_hub
    import tools.skills_hub as tools_hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool_module

    installed = [
        {
            "name": "hidden-canonical",
            "category": "optional",
            "description": "Only offered on demand.",
        },
        {
            "name": "visible-skill",
            "category": "core",
            "description": "Shown by default.",
        },
    ]
    monkeypatch.setattr(tools_hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(
        tools_hub,
        "HubLockFile",
        lambda: type("Lock", (), {"list_installed": lambda self: []})(),
    )
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {})
    monkeypatch.setattr(
        skills_tool_module,
        "_find_all_skills",
        lambda **_kwargs: list(installed),
    )

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None, width=120)
    cli_hub.do_list(console=console)
    output = sink.getvalue().lower()

    assert "hidden-canonical" in output
    assert "visible-skill" in output
    assert "on-demand" in output
