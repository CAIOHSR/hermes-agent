"""Skill usage telemetry + provenance for the Curator: a sidecar ``~/.hermes/skills/.usage.json`` keyed by
skill name (never frontmatter — keeps telemetry out of user-authored SKILL.md and off bundled/hub skills).
Counter bumps are best-effort (DEBUG-logged failures never break the tool call); writes are atomic under a
cross-process lock. Curator management is an explicit ``created_by: agent`` marker written by skill_manage —
never inferred from location. Lifecycle: active -> stale -> archived (moved to .archive/); ``pinned`` opts
out of auto transitions, orthogonal to state."""

from __future__ import annotations

import json
import logging
import os
import secrets
import stat
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home
from agent.skill_utils import is_excluded_skill_path, is_external_skill_path
from utils import atomic_write_text

logger = logging.getLogger(__name__)

# fcntl is Unix-only; on Windows use msvcrt for file locking.
msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-specific fallback
    fcntl = None
    with suppress(ImportError):
        import msvcrt


STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED = "active", "stale", "archived"
_VALID_STATES = {STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED}

# Load-bearing built-ins (by frontmatter ``name``) the curator must NEVER archive/consolidate regardless of
# ``curator.prune_builtins``, pins or LLM judgment — archiving one breaks its slash command. Keep tiny.
PROTECTED_BUILTIN_SKILLS: Set[str] = set()


def is_protected_builtin(skill_name: str) -> bool:
    return skill_name in PROTECTED_BUILTIN_SKILLS


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def _usage_file() -> Path:
    return _skills_dir() / ".usage.json"


def _archive_dir() -> Path:
    return _skills_dir() / ".archive"


def _flock(fd, lock: bool) -> None:
    if fcntl:
        return fcntl.flock(fd, fcntl.LOCK_EX if lock else fcntl.LOCK_UN)
    fd.seek(0)
    msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK if lock else msvcrt.LK_UNLCK, 1)


@contextmanager
def _usage_file_lock():
    """Serialize .usage.json read-modify-write cycles across processes."""
    lock_path = _usage_file().with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None and msvcrt is None:
        yield
        return
    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        lock_path.write_text(" ", encoding="utf-8")  # msvcrt needs a non-empty byte range to lock
    with open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8") as fd:
        _flock(fd, True)
        try:
            yield
        finally:
            with suppress(OSError, IOError):
                _flock(fd, False)


def _read_lines(path: Path, fail_log: str) -> List[str]:
    """Stripped, non-empty lines of a small metadata file ([] if missing/unreadable)."""
    if not path.exists():
        return []
    try:
        return [s for s in (line.strip() for line in path.read_text(encoding="utf-8").splitlines()) if s]
    except OSError as e:
        logger.debug(fail_log, e)
        return []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value)) if value else None
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed and parsed.tzinfo is None else parsed


def latest_activity_at(record: Dict[str, Any]) -> Optional[str]:
    """Newest use/view/patch timestamp; ``created_at`` is excluded so never-active skills stay distinguishable."""
    stamps = [(dt, str(raw)) for raw in (record.get(k) for k in ("last_used_at", "last_viewed_at", "last_patched_at"))
              if (dt := _parse_iso_timestamp(raw)) is not None]
    return max(stamps, key=lambda t: t[0])[1] if stamps else None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _non_negative_int(value: Any) -> int:
    return 0 if isinstance(value, bool) else max(0, _int_or_zero(value))


def activity_count(record: Dict[str, Any]) -> int:
    """Total observed use+view+patch events."""
    return sum(_int_or_zero(record.get(key)) for key in ("use_count", "view_count", "patch_count"))


# --- Provenance — which skills are agent-created (and thus eligible for curation) ---
def _read_bundled_manifest_names() -> Set[str]:
    """Names from ``.bundled_manifest`` ("name:hash" per line); empty if missing/unreadable."""
    lines = _read_lines(_skills_dir() / ".bundled_manifest", "Failed to read bundled manifest: %s")
    return {n for n in (line.split(":", 1)[0].strip() for line in lines) if n}


def _read_hub_installed_names() -> Set[str]:
    """Hub-installed names (``.hub/lock.json``) plus the frontmatter name of each in-tree ``install_path``."""
    skills_dir = _skills_dir()
    lock_path = skills_dir / ".hub" / "lock.json"
    if not lock_path.exists():
        return set()
    # The whole walk sits under one handler (BASE semantics): an OSError anywhere — including the
    # per-skill SKILL.md read — logs and yields an empty set rather than a partial one.
    try:
        # errors="replace": hub descriptions can carry Windows-1252 high bytes; a strict read raises
        # UnicodeDecodeError (a ValueError, not caught below) and would 500 the whole /api/skills endpoint.
        data = json.loads(lock_path.read_text(encoding="utf-8", errors="replace"))
        installed = (data.get("installed") or {}) if isinstance(data, dict) else None
        if not isinstance(installed, dict):
            return set()
        names = {str(k) for k in installed}
        paths = (e.get("install_path") for e in installed.values() if isinstance(e, dict))
        for install_path in (p for p in paths if isinstance(p, str) and p.strip()):
            try:  # ValueError: install_path escapes the skills dir
                resolved = (skills_dir / install_path).resolve()
                resolved.relative_to(skills_dir.resolve())
            except (OSError, ValueError):
                continue
            if (resolved / "SKILL.md").exists():
                names.add(_read_skill_name(resolved / "SKILL.md", fallback=resolved.name))
        return names
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read hub lock file: %s", e)
    return set()


def _prune_builtins_enabled() -> bool:
    """``curator.prune_builtins`` (default True); lazy config import keeps this module importable during update/sync."""
    try:
        from hermes_cli.config import load_config
        cur = load_config().get("curator")
        return bool(cur.get("prune_builtins", True)) if isinstance(cur, dict) else True
    except Exception as e:  # pragma: no cover — best-effort config read
        logger.debug("Failed to read curator.prune_builtins: %s", e)
        return True


def read_suppressed_names() -> Set[str]:
    """Built-ins the curator pruned (``.curator_suppressed``); the update-time re-seeder must leave these archived."""
    lines = _read_lines(_skills_dir() / ".curator_suppressed", "Failed to read curator suppression list: %s")
    return {line for line in lines if not line.startswith("#")}


def _toggle_suppressed_name(skill_name: str, *, add: bool) -> None:
    """Add (built-in pruned) or drop (restored) *skill_name* in the suppression list; no-op when unchanged."""
    if not skill_name or (skill_name in (names := read_suppressed_names())) == add:
        return
    (names.add if add else names.discard)(skill_name)
    try:
        atomic_write_text(_skills_dir() / ".curator_suppressed", "\n".join(sorted(names)) + ("\n" if names else ""),
                          tmp_prefix=".curator_suppressed_")
    except Exception as e:
        logger.debug("Failed to write curator suppression list: %s", e, exc_info=True)


def _iter_skill_mds(base: Path, *, local_only: bool) -> Iterator[Tuple[str, Path]]:
    """``(frontmatter name, SKILL.md)`` under *base* minus metadata/VCS/venv/cache dirs; *local_only* also skips
    external skill dirs mounted below the tree (curation must not touch them)."""
    for skill_md in base.rglob("SKILL.md"):
        if not (is_excluded_skill_path(skill_md) or (local_only and is_external_skill_path(skill_md))):
            yield _read_skill_name(skill_md, fallback=skill_md.parent.name), skill_md


def _scan_local_skills(keep: Callable[[str, Path, Set[str], Dict[str, Any]], bool]) -> List[str]:
    """Sorted local skill names passing *keep(name, skill_md, bundled, usage)*; hub/protected names never reach it."""
    if not (base := _skills_dir()).exists():
        return []
    hub, bundled, usage = _read_hub_installed_names(), _read_bundled_manifest_names(), load_usage()
    return sorted({name for name, skill_md in _iter_skill_mds(base, local_only=True)
                   if name not in hub and not is_protected_builtin(name) and keep(name, skill_md, bundled, usage)})


def list_agent_created_skill_names() -> List[str]:
    """Curator-manageable skills: ``created_by: agent`` records plus, with ``curator.prune_builtins``, bundled
    built-ins (which never carry a managed record, so the record gate applies only to local skills). Never hub."""
    prune_builtins = _prune_builtins_enabled()  # read once, before the walk
    return _scan_local_skills(
        lambda name, _md, bundled, usage: prune_builtins if name in bundled else _is_curator_managed_record(usage.get(name)))


def list_archived_skill_names() -> List[str]:
    """Skills in ``.archive/`` — flat layout (``archive_skill`` flattens), so dir name == skill name."""
    root = _archive_dir()
    return sorted({p.name for p in root.iterdir() if p.is_dir()}) if root.exists() else []


def _read_skill_name(skill_md: Path, fallback: str) -> str:
    """The frontmatter ``name:`` field of a SKILL.md (first 4000 chars), else *fallback*."""
    try:
        lines = [line.strip() for line in skill_md.read_text(encoding="utf-8", errors="replace")[:4000].split("\n")]
    except OSError:
        return fallback
    if "---" not in lines:
        return fallback
    block = lines[lines.index("---") + 1:]  # frontmatter runs to the closing --- or (truncated) end of text
    block = block[:block.index("---")] if "---" in block else block
    values = (line.split(":", 1)[1].strip().strip("\"'") for line in block if line.startswith("name:"))
    return next((v for v in values if v), fallback)


def is_agent_created(skill_name: str) -> bool:
    """Neither bundled nor hub-installed (and not only present in an external dir)."""
    skill_name = _canonicalize_skill_name(skill_name)
    return not (is_bundled(skill_name) or is_hub_installed(skill_name)) and (
        _find_skill_dir(skill_name) is not None or _find_external_skill_dir(skill_name) is None)


def is_hub_installed(skill_name: str) -> bool:
    skill_name = _canonicalize_skill_name(skill_name)
    return skill_name in _read_hub_installed_names()


def is_bundled(skill_name: str) -> bool:
    skill_name = _canonicalize_skill_name(skill_name)
    return skill_name in _read_bundled_manifest_names()


def _external_read_only_message(skill_name: str) -> str:
    return f"skill '{skill_name}' lives in skills.external_dirs; external skills are read-only to the curator"


def is_curation_eligible(skill_name: str, skill_path: Optional[Path] = None) -> bool:
    """Agent-created: yes. Bundled: only with ``curator.prune_builtins``. Hub / external-dir / protected built-ins:
    never (external owner). Org-shared skills are eligible here but protected from ARCHIVE/DELETE elsewhere."""
    skill_name = _canonicalize_skill_name(skill_name)
    if ((skill_path is not None and is_external_skill_path(skill_path)) or is_protected_builtin(skill_name)
            or is_hub_installed(skill_name)):
        return False
    if is_bundled(skill_name):
        return _prune_builtins_enabled()
    local_dir = _find_skill_dir(skill_name)
    return not is_external_skill_path(local_dir) if local_dir else _find_external_skill_dir(skill_name) is None


def _is_curator_managed_record(record: Any) -> bool:
    """``created_by`` is a curator-management OPT-IN flag, not proof of authorship (``curator adopt`` flips it);
    the key name is kept because it lives in every user's ``.usage.json``.

    NAMING (issue #67140): the on-disk field is ``created_by``, which reads like provenance but is consumed
    as a **curator-management opt-in policy flag**. The two are not the same question:
    """
    return isinstance(record, dict) and (record.get("created_by") == "agent" or record.get("agent_created") is True)


def is_curator_managed(skill_name: str) -> bool:
    return _is_curator_managed_record(load_usage().get(skill_name))


def list_unmanaged_skill_names() -> List[str]:
    """Curation-ELIGIBLE skills without a provenance marker (pre-``created_by`` records, or foreground creates that
    belong to the user). Invisible to ``curated_report()`` and auto transitions; only ``curator adopt`` hands
    them over — provenance is declared, never inferred from activity."""
    return _scan_local_skills(
        lambda name, md, bundled, usage: name not in bundled and not _is_curator_managed_record(usage.get(name))
        and is_curation_eligible(name, md))


def unmanaged_report() -> List[Dict[str, Any]]:
    """Rows for :func:`list_unmanaged_skill_names`; ``has_provenance_key`` (False = pre-dates ``created_by``) explains
    WHY, it is not a signal to adopt on."""
    usage = load_usage()
    return [_report_row(n, usage.get(n), has_provenance_key="created_by" in usage.get(n, {}), has_record=n in usage)
            for n in list_unmanaged_skill_names()]


def adopt_skill(skill_name: str) -> Tuple[bool, str]:
    """User-declared handover: writes the ``created_by: agent`` marker (inactivity clock NOT reset). Refuses hub,
    external, bundled and protected skills. Returns (ok, message)."""
    skill_name = _canonicalize_skill_name(skill_name)
    if not skill_name:
        return False, "no skill name given"
    if is_protected_builtin(skill_name):
        return False, f"'{skill_name}' is a protected built-in; the curator never manages it"
    if is_hub_installed(skill_name):
        return False, f"'{skill_name}' is hub-installed; its upstream owns it"
    if is_bundled(skill_name):  # governed by prune_builtins; stamping created_by=agent would change nothing
        return False, f"'{skill_name}' is a bundled built-in — it is governed by curator.prune_builtins, not by adoption"
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None:
        if _find_external_skill_dir(skill_name) is not None:
            return False, f"'{skill_name}' lives in skills.external_dirs and is read-only to the curator"
        return False, f"skill '{skill_name}' not found"
    if is_external_skill_path(skill_dir):
        return False, _external_read_only_message(skill_name)
    if is_curator_managed(skill_name):
        return True, f"'{skill_name}' is already curator-managed"
    mark_agent_created(skill_name)
    if is_curator_managed(skill_name):
        return True, f"adopted '{skill_name}' into curator management"
    return False, f"could not mark '{skill_name}' as curator-managed"


# --- Sidecar I/O ---
def _empty_record() -> Dict[str, Any]:
    return {"created_by": None, "use_count": 0, "view_count": 0, "last_used_at": None, "last_viewed_at": None,
            "patch_count": 0, "patch_generation": 0, "last_reused_patch_generation": 0, "last_patched_at": None,
            "created_at": _now_iso(), "state": STATE_ACTIVE, "pinned": False, "archived_at": None}


def _backfilled(rec: Any) -> Dict[str, Any]:
    """*rec* with every missing default key appended (a fresh record when not a dict)."""
    if not isinstance(rec, dict):
        return _empty_record()
    return {**rec, **{k: v for k, v in _empty_record().items() if k not in rec}}


def _report_row(name: str, raw: Any, **extra: Any) -> Dict[str, Any]:
    row = {"name": name, **_backfilled(raw), **extra}
    row.update(last_activity_at=latest_activity_at(row), activity_count=activity_count(row))
    return row


def load_usage() -> Dict[str, Dict[str, Any]]:
    """The whole .usage.json map (non-dict values dropped); {} on missing/corrupt."""
    path = _usage_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read %s: %s", path, e)
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def save_usage(data: Dict[str, Dict[str, Any]]) -> bool:
    """Write the usage map atomically; True when it committed."""
    path = _usage_file()
    try:
        atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), tmp_prefix=".usage_", create_mode=0o600)
        return True
    except Exception as e:
        logger.debug("Failed to write %s: %s", path, e, exc_info=True)
        return False


def get_record(skill_name: str) -> Dict[str, Any]:
    """The (backfilled) record for *skill_name*; fresh defaults if missing."""
    skill_name = _canonicalize_skill_name(skill_name)
    return _backfilled(load_usage().get(skill_name))


def _locked_update(skill_name: str, op: Callable[[Dict[str, Dict[str, Any]]], Tuple[Any, bool]], fail_log: str,
                   guard: Optional[Callable[[], bool]] = None) -> Any:
    """*op(data) -> (result, dirty)* under the file lock, saving only when dirty; *guard* runs before locking.
    None when the guard failed, the save did not land, or anything raised (DEBUG-logged via *fail_log*)."""
    try:
        if guard is not None and not guard():
            return None
        with _usage_file_lock():
            usage_file_existed = _usage_file().exists()
            data = load_usage()
            result, dirty = op(data)
            if dirty:
                if not save_usage(data):
                    return None
                if not usage_file_existed:
                    _refresh_identity_cache_signature()
            return result
    except Exception as e:
        logger.debug(fail_log, skill_name, e, exc_info=True)
        return None


def seed_record_if_missing(skill_name: str) -> None:
    """Baseline record for a curation-eligible skill so its inactivity clock starts at first sight, not epoch."""
    skill_name = _canonicalize_skill_name(skill_name)
    if skill_name and is_curation_eligible(skill_name):
        # load_usage() already dropped non-dict values, so "missing" == key absent; dirty only when inserted.
        def _seed(data):
            return None, skill_name not in data and data.setdefault(skill_name, _empty_record()) is not None
        _locked_update(skill_name, _seed, "skill_usage.seed_record_if_missing(%s) failed: %s")


def _mutate(skill_name: str, mutator, *, require_curation_eligible: bool = False) -> Any:
    """Load, apply *mutator(record)* in place, save; the mutator result (None if nothing landed). Telemetry is
    recorded for ANY skill; lifecycle mutators pass ``require_curation_eligible=True`` (never write onto unmanaged)."""
    skill_name = _canonicalize_skill_name(skill_name)
    if not skill_name:
        return None
    return _locked_update(skill_name, lambda data: (mutator(data.setdefault(skill_name, _empty_record())), True),
                          "skill_usage._mutate(%s) failed: %s",
                          (lambda: is_curation_eligible(skill_name)) if require_curation_eligible else None)


def _set_field(skill_name: str, key: str, value: Any) -> bool:
    """Curation-gated single-field write; True only when the write landed."""
    return bool(_mutate(skill_name, lambda rec: rec.update({key: value}) or True, require_curation_eligible=True))


def _bump(rec: Dict[str, Any], count_key: str, ts_key: str) -> None:
    rec[count_key] = _non_negative_int(rec.get(count_key)) + 1
    rec[ts_key] = _now_iso()


def telemetry_provenance(skill_name: str, record: Optional[Dict[str, Any]] = None) -> str:
    """Bounded provenance label for shared skill metrics."""
    skill_name = _canonicalize_skill_name(skill_name)
    if is_hub_installed(skill_name) or is_bundled(skill_name):
        return "installed"
    if ":" in skill_name:
        with suppress(Exception):
            from hermes_cli.plugins import get_plugin_manager
            if get_plugin_manager().find_plugin_skill(skill_name) is not None:
                return "installed"
    if label := {"installed": "installed", "agent": "agent_created"}.get(
            record.get("created_by") if isinstance(record, dict) else None):
        return label
    if _find_external_skill_dir(skill_name) is not None:
        return "external"
    return "local" if _find_skill_dir(skill_name) is not None or isinstance(record, dict) else "unknown"


def _emit_skill_lifecycle(skill_name: str, action: str, *, record: Optional[Dict[str, Any]] = None,
                          task_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
    """Best-effort lifecycle hook after an authoritative state change; facts absent from *record* go as None."""
    facts = record or {}
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook
        if has_hook("on_skill_lifecycle"):
            invoke_hook("on_skill_lifecycle", action=action, skill_name=skill_name,
                        provenance=telemetry_provenance(skill_name, record), task_id=task_id or "",
                        session_id=session_id or "", use_count=facts.get("use_count"), reused=facts.get("reused"),
                        reuse_after_patch=facts.get("reuse_after_patch"))
    except Exception:
        logger.debug("skill_usage lifecycle hook failed for %s/%s", skill_name, action, exc_info=True)


def _mutate_and_emit(skill_name: str, action: str, mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
                     **hook_kwargs: Any) -> None:
    """``_mutate`` then emit *action* with the mutator's facts as the record — only if the write landed."""
    if isinstance(facts := _mutate(skill_name, mutator), dict):
        _emit_skill_lifecycle(skill_name, action, record=facts, **hook_kwargs)


# --- Counter bumps — telemetry for ALL skills regardless of provenance (observability only) ---
def bump_view(skill_name: str) -> None:
    skill_name = _canonicalize_skill_name(skill_name)
    _mutate(skill_name, lambda rec: _bump(rec, "view_count", "last_viewed_at"))


def bump_use(skill_name: str, *, task_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
    """Skill actively used (loaded into the prompt path / referenced from an assistant turn)."""
    skill_name = _canonicalize_skill_name(skill_name)
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        uses = _non_negative_int(rec.get("use_count"))
        gen = _non_negative_int(rec.get("patch_generation"))
        last_reused = min(_non_negative_int(rec.get("last_reused_patch_generation")), gen)
        reuse_after_patch = uses > 0 and gen > last_reused
        rec.update(use_count=uses + 1, last_used_at=_now_iso(), patch_generation=gen,
                   last_reused_patch_generation=gen if reuse_after_patch else last_reused)
        return {"created_by": rec.get("created_by"), "use_count": uses + 1, "reused": uses > 0,
                "reuse_after_patch": reuse_after_patch}
    _mutate_and_emit(skill_name, "loaded", _apply, task_id=task_id, session_id=session_id)


def bump_patch(skill_name: str, *, action: str = "patch", task_id: Optional[str] = None,
               session_id: Optional[str] = None) -> None:
    """Called from skill_manage (patch/edit)."""
    skill_name = _canonicalize_skill_name(skill_name)
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        _bump(rec, "patch_count", "last_patched_at")
        rec["patch_generation"] = _non_negative_int(rec.get("patch_generation")) + 1
        return {"created_by": rec.get("created_by")}
    _mutate_and_emit(skill_name, "patched" if action == "patch" else "edited", _apply, task_id=task_id,
                     session_id=session_id)


def record_created(skill_name: str, *, agent_created: bool, task_id: Optional[str] = None,
                   session_id: Optional[str] = None) -> None:
    """Persist creation provenance and emit a create fact; the record is reset (a create is a new logical skill)."""
    skill_name = _canonicalize_skill_name(skill_name)
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        rec.clear()
        rec.update(_empty_record(), created_by="agent" if agent_created else None)
        return {"created_by": rec["created_by"]}
    _mutate_and_emit(skill_name, "created", _apply, task_id=task_id, session_id=session_id)


def record_installed(skill_name: str) -> None:
    """Record a successful Skills Hub install without exporting its name."""
    skill_name = _canonicalize_skill_name(skill_name)
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        rec.update(created_by="installed", state=STATE_ACTIVE, archived_at=None)
        return {"created_by": "installed"}
    _mutate_and_emit(skill_name, "installed", _apply)


def mark_agent_created(skill_name: str) -> None:
    """Opt a skill into curator management — the only thing that makes it eligible for automatic curation."""
    skill_name = _canonicalize_skill_name(skill_name)
    _set_field(skill_name, "created_by", "agent")


def set_state(skill_name: str, state: str) -> None:
    """Set lifecycle state (no-op if invalid / unmanageable). Emits archived/stale/restored; active<-stale is silent."""
    skill_name = _canonicalize_skill_name(skill_name)
    if state not in _VALID_STATES:
        logger.debug("set_state: invalid state %r for %s", state, skill_name)
        return

    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        previous = rec.get("state")
        if previous != state:
            rec["state"] = state
            if state != STATE_STALE:
                rec["archived_at"] = _now_iso() if state == STATE_ARCHIVED else None
        return {"changed": previous != state, "created_by": rec.get("created_by"), "previous_state": previous}
    facts = _mutate(skill_name, _apply, require_curation_eligible=True)
    if isinstance(facts, dict) and facts["changed"]:
        restored = state == STATE_ACTIVE and facts["previous_state"] == STATE_ARCHIVED
        action = "restored" if restored else {STATE_ARCHIVED: "archived", STATE_STALE: "stale"}.get(state)
        if action is not None:
            _emit_skill_lifecycle(skill_name, action, record=facts)


def set_pinned(skill_name: str, pinned: bool) -> bool:
    """False when the write did not land (not curation-eligible).

    (skill not curation-eligible), True on success — so callers can report failure instead of a false
    success (issue #92993).
    """
    skill_name = _canonicalize_skill_name(skill_name)
    return _set_field(skill_name, "pinned", bool(pinned))


def set_sync(skill_name: str, sync: bool) -> None:
    """Opt-in ``sync`` flag (read by ``skills_sync_client``); curation-gated so bundled/hub/external can't be marked."""
    skill_name = _canonicalize_skill_name(skill_name)
    _set_field(skill_name, "sync", bool(sync))


def is_sync_enabled(skill_name: str) -> bool:
    return get_record(skill_name).get("sync") is True


def forget(skill_name: str) -> None:
    skill_name = _canonicalize_skill_name(skill_name)
    if skill_name:
        _locked_update(skill_name, lambda d: (None, d.pop(skill_name, None) is not None), "skill_usage.forget(%s) failed: %s")


# --- Archive / restore ---
def _relocate(src: Path, dest: Path, skill_name: str, action: str, **capture_kwargs: Any) -> Tuple[bool, str]:
    """Move *src* to *dest* for *action* ("archive" | "restore") inside a best-effort audit-ledger entry, then apply
    suppression + state side effects; rename falls back to shutil.move across devices."""
    try:
        from tools import skill_ledger as _ledger
        _ledger_before = _ledger.capture_before(src, **capture_kwargs)
    except Exception:
        _ledger = _ledger_before = None  # type: ignore[assignment]
    try:
        src.rename(dest)
    except OSError:
        import shutil
        try:
            shutil.move(str(src), str(dest))
        except Exception as e:
            return False, f"failed to {action}: {e}"
    archiving = action == "archive"
    if not archiving or is_bundled(skill_name):  # pruning a built-in only sticks if the re-seeder skips it
        _toggle_suppressed_name(skill_name, add=archiving)
    set_state(skill_name, STATE_ARCHIVED if archiving else STATE_ACTIVE)
    with suppress(Exception):
        if _ledger is not None:
            _ledger.record_mutation(action, skill_name, before=_ledger_before or [], after_root=dest)
    return True, f"{action}d to {dest}"


def archive_skill(skill_name: str) -> Tuple[bool, str]:
    """Move a curator-eligible skill dir to ``.archive/`` (flattened; timestamp suffix on collision). Never hub;
    bundled built-ins only with ``curator.prune_builtins`` (and then suppressed from re-seeding)."""
    skill_name = _canonicalize_skill_name(skill_name)
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None and _find_external_skill_dir(skill_name) is not None:
        return False, _external_read_only_message(skill_name)
    if not is_curation_eligible(skill_name, skill_dir):
        if is_protected_builtin(skill_name):
            return False, f"skill '{skill_name}' is a protected built-in; it backs load-bearing UX and is never archived or consolidated"
        if is_hub_installed(skill_name):
            return False, f"skill '{skill_name}' is hub-installed; never archive"
        return False, f"skill '{skill_name}' is a bundled built-in; enable curator.prune_builtins to allow pruning it"
    if skill_dir is None:
        return False, f"skill '{skill_name}' not found"
    if is_external_skill_path(skill_dir):
        return False, _external_read_only_message(skill_name)
    dest = _archive_dir() / skill_dir.name
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"failed to create archive dir: {e}"
    if dest.exists():
        dest = dest.with_name(f"{skill_dir.name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    # complete_package: consolidation may have re-homed support files first, so a disk-only capture can come
    # back hollow; the fill from the newest curator backup keeps rollback restorable.
    return _relocate(skill_dir, dest, skill_name, "archive", complete_package=True, skill=skill_name)


def restore_skill(skill_name: str) -> Tuple[bool, str]:
    """Move an archived skill back to the flat layout (nesting NOT reconstructed). Refuses a name now colliding with
    a hub skill, or a bundled built-in unless ``curator.prune_builtins`` is on (restoring lifts a prune)."""
    skill_name = _canonicalize_skill_name(skill_name)
    if is_hub_installed(skill_name):
        return False, f"skill '{skill_name}' is now hub-installed; restore would shadow the upstream version"
    if is_bundled(skill_name) and not _prune_builtins_enabled():
        return False, f"skill '{skill_name}' is now bundled; restore would shadow the upstream version"
    archive_root = _archive_dir()
    if not archive_root.exists():
        return False, "no archive directory"
    # Exact name first (recursive: older archives left nested layouts), then the timestamped duplicate. Only
    # "<skill>-YYYYMMDDHHMMSS" counts — a bare startswith("<skill>-") would let restoring "git" steal "git-helpers".
    dirs = [p for p in archive_root.rglob("*") if p.is_dir()]
    prefix = f"{skill_name}-"
    candidates = [p for p in dirs if p.name == skill_name] or sorted(
        (p for p in dirs if p.name.startswith(prefix) and len(p.name) - len(prefix) == 14
         and p.name[len(prefix):].isdigit()), reverse=True)
    if not candidates:
        return False, f"skill '{skill_name}' not found in archive"
    if (dest := _skills_dir() / skill_name).exists():
        return False, f"destination already exists: {dest}"
    return _relocate(candidates[0], dest, skill_name, "restore")


def _match_skill_dir(skill_mds: Iterable[Path], skill_name: str) -> Optional[Path]:
    return next((p.parent for p in skill_mds if _read_skill_name(p, fallback=p.parent.name) == skill_name), None)


def _find_skill_dir(skill_name: str) -> Optional[Path]:
    """Skill dir by frontmatter ``name`` (flat or nested); the gated index iterator sees only the active org mirror."""
    from agent.skill_utils import iter_skill_index_files
    base = _skills_dir()
    return _match_skill_dir((p for p in iter_skill_index_files(base, "SKILL.md") if not is_external_skill_path(p)),
                            skill_name) if base.exists() else None


def _find_external_skill_dir(skill_name: str) -> Optional[Path]:
    """Skill dir under configured external dirs by frontmatter name."""
    from agent.skill_utils import get_all_skills_dirs
    return next((found for base in get_all_skills_dirs()[1:] if base.exists()
                 if (found := _match_skill_dir((p for p in base.rglob("SKILL.md") if not is_excluded_skill_path(p)),
                                               skill_name)) is not None), None)


# --- Reporting — for the curator CLI / slash command ---
def curated_report() -> List[Dict[str, Any]]:
    """One backfilled row per curator-managed skill with ``provenance`` and ``_persisted`` (real record exists; fresh
    backfills get their inactivity clock seeded instead of counting as ancient)."""
    data = load_usage()
    names = set(list_agent_created_skill_names())
    # Pinned-but-unmanaged skills stay visible or their pin silently vanishes from `curator status`; the local-dir
    # guard keeps stale records for deleted dirs from rendering as ghost rows.
    names.update(name for name, rec in data.items()
                 if rec.get("pinned") and is_curation_eligible(name) and _find_skill_dir(name) is not None)
    return [_report_row(n, data.get(n), _persisted=n in data, provenance=provenance(n)) for n in sorted(names)]


def provenance(skill_name: str) -> str:
    """'hub' | 'bundled' | 'agent' (the latter also covers local manually-authored skills)."""
    skill_name = _canonicalize_skill_name(skill_name)
    return "hub" if is_hub_installed(skill_name) else "bundled" if is_bundled(skill_name) else "agent"


def usage_report() -> List[Dict[str, Any]]:
    """Usage rows for EVERY skill on disk (built-ins and hub included); ``curated_report()`` is the managed subset."""
    if not (base := _skills_dir()).exists():
        return []
    data = load_usage()
    return [_report_row(n, data.get(n), provenance=provenance(n), _persisted=n in data)
            for n in sorted({name for name, _md in _iter_skill_mds(base, local_only=False)})]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import tempfile  # noqa: F401,E402
import os  # noqa: F401,E402
import os  # noqa: F401,E402
import tempfile  # noqa: F401,E402

def _suppressed_file() -> Path:
    return _skills_dir() / ".curator_suppressed"

def _write_suppressed_names(names: Set[str]) -> None:
    path = _suppressed_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = "\n".join(sorted(names)) + ("\n" if names else "")
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".curator_suppressed_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("Failed to write curator suppression list: %s", e, exc_info=True)

def add_suppressed_name(skill_name: str) -> None:
    """Record that a built-in skill was pruned, so sync won't restore it."""
    if not skill_name:
        return
    names = read_suppressed_names()
    if skill_name not in names:
        names.add(skill_name)
        _write_suppressed_names(names)

def agent_created_report() -> List[Dict[str, Any]]:
    """DEPRECATED — use :func:`curated_report` instead.

    Used to return everything :func:`curated_report` returns (including bundled
    skills when ``curator.prune_builtins`` is enabled), which made the
    "agent-created" name misleading. Kept as a compatibility alias for
    external callers; new code should call ``curated_report()``.
    """
    return curated_report()

def remove_suppressed_name(skill_name: str) -> None:
    """Clear a built-in's suppression entry (e.g. on restore)."""
    if not skill_name:
        return
    names = read_suppressed_names()
    if skill_name in names:
        names.discard(skill_name)
        _write_suppressed_names(names)
# ---- END PLUGIN-COMPAT ----


_SKILL_IDENTITY_CACHE: Dict[
    Path,
    Tuple[
        Tuple[Tuple[str, int, int, int, int, int], ...],
        Dict[str, Set[Path]],
        Dict[Path, str],
        Tuple[Path, ...],
    ],
] = {}


def _identity_watch_signature(
    paths: Iterable[Path],
) -> Tuple[Tuple[str, int, int, int, int, int], ...]:
    """Fingerprint known skill files and the dirs that can gain children."""
    signature = []
    for path in sorted(set(paths), key=lambda item: str(item)):
        try:
            path_stat = os.stat(path, follow_symlinks=False)
            row = (
                str(path),
                path_stat.st_dev,
                path_stat.st_ino,
                path_stat.st_mtime_ns,
                path_stat.st_ctime_ns,
                path_stat.st_size,
            )
        except OSError:
            row = (str(path), -1, -1, -1, -1, -1)
        signature.append(row)
    return tuple(signature)


def _skills_root_signature(root: Path) -> Optional[Tuple[int, int, int, int]]:
    """Return the root-only signature kept for diagnostics/backward compatibility."""
    try:
        root_stat = root.stat()
    except OSError:
        return None
    return (
        root_stat.st_dev,
        root_stat.st_ino,
        root_stat.st_mtime_ns,
        root_stat.st_ctime_ns,
    )


def _add_identity_alias(
    aliases: Dict[str, Set[Path]], alias: str, skill_dir: Path
) -> None:
    if alias:
        aliases.setdefault(alias, set()).add(skill_dir)


def _build_skill_identity_index(
    root: Path,
) -> Tuple[Dict[str, Set[Path]], Dict[Path, str], Tuple[Path, ...]]:
    """Build aliases for local skills, preserving ambiguity as a set of paths."""
    aliases: Dict[str, Set[Path]] = {}
    canonical_by_dir: Dict[Path, str] = {}
    watched_paths: Set[Path] = set()
    try:
        from agent.skill_utils import iter_skill_index_files

        resolved_root = root.resolve()
        watched_paths.add(resolved_root)
        # Preserve empty category directories in the watch set. This walk runs
        # only while rebuilding an already-invalidated index; stable cache hits
        # perform stat calls over watched paths and never recurse the tree.
        for directory, child_dirs, _ in os.walk(resolved_root, followlinks=False):
            child_dirs[:] = [name for name in child_dirs if not name.startswith(".")]
            watched_paths.add(Path(directory))
        for skill_md in iter_skill_index_files(root, "SKILL.md"):
            try:
                resolved_md = skill_md.resolve()
                resolved_md.relative_to(resolved_root)
                skill_dir = resolved_md.parent
            except (OSError, RuntimeError, ValueError):
                # Absolute paths are trusted only when they resolve inside the
                # active profile's skills root. This also rejects symlink escapes.
                continue
            if is_excluded_skill_path(skill_md) or is_external_skill_path(skill_md):
                continue

            watched_paths.add(resolved_md)
            parent = skill_dir
            while True:
                watched_paths.add(parent)
                if parent == resolved_root or resolved_root not in parent.parents:
                    break
                parent = parent.parent

            canonical = _read_skill_name(skill_md, fallback=skill_dir.name)
            if not canonical:
                continue
            canonical_by_dir[skill_dir] = canonical

            # The frontmatter name and on-disk directory basename are both
            # accepted bare aliases. The relative directory path is also a
            # trusted alias (e.g. ``research/directory-alias``).
            _add_identity_alias(aliases, canonical, skill_dir)
            _add_identity_alias(aliases, skill_dir.name, skill_dir)
            try:
                relative_dir = skill_dir.relative_to(resolved_root)
            except ValueError:
                continue
            relative_alias = relative_dir.as_posix()
            _add_identity_alias(aliases, relative_alias, skill_dir)

            # Categorized callers use either category/name or category:name.
            # Accept the canonical frontmatter name as well as the directory
            # alias in both forms; only a unique path may canonicalize.
            if len(relative_dir.parts) >= 2:
                category = "/".join(relative_dir.parts[:-1])
                for name in (canonical, skill_dir.name):
                    _add_identity_alias(aliases, f"{category}/{name}", skill_dir)
                    _add_identity_alias(aliases, f"{category}:{name}", skill_dir)
    except Exception as e:
        # Telemetry must never make a skill operation fail because discovery is
        # unavailable or a malformed skill tree was encountered.
        logger.debug("Failed to build skill identity index: %s", e, exc_info=True)
    return aliases, canonical_by_dir, tuple(watched_paths)


def _skill_identity_index() -> Tuple[Dict[str, Set[Path]], Dict[Path, str]]:
    """Return the cached local identity index for the active Hermes profile."""
    root = _skills_dir()
    try:
        cache_root = root.expanduser().resolve()
    except (OSError, RuntimeError):
        cache_root = root.expanduser()
    cached = _SKILL_IDENTITY_CACHE.get(cache_root)
    if cached is not None and cached[0] == _identity_watch_signature(cached[3]):
        return cached[1], cached[2]

    aliases, canonical_by_dir, watched_paths = _build_skill_identity_index(root)
    _SKILL_IDENTITY_CACHE[cache_root] = (
        _identity_watch_signature(watched_paths),
        aliases,
        canonical_by_dir,
        watched_paths,
    )
    return aliases, canonical_by_dir


def _refresh_identity_cache_signature() -> None:
    """Account for the first sidecar creation without rescanning skills."""
    root = _skills_dir()
    try:
        cache_root = root.expanduser().resolve()
    except (OSError, RuntimeError):
        cache_root = root.expanduser()
    cached = _SKILL_IDENTITY_CACHE.get(cache_root)
    if cached is not None:
        _SKILL_IDENTITY_CACHE[cache_root] = (
            _identity_watch_signature(cached[3]),
            cached[1],
            cached[2],
            cached[3],
        )


def _registered_plugin_skill(skill_name: str) -> bool:
    """Whether a qualified key belongs to a registered plugin skill."""
    if ":" not in skill_name:
        return False
    try:
        from hermes_cli.plugins import get_plugin_manager

        return get_plugin_manager().find_plugin_skill(skill_name) is not None
    except Exception:
        # Plugin discovery is optional from the telemetry module's perspective.
        return False


def _canonicalize_skill_name(skill_name: str) -> str:
    """Resolve a local skill alias to its unique frontmatter ``name``.

    This is intentionally fail-safe: plugin-qualified names, unknown aliases,
    ambiguous aliases, and paths outside the active local skills root are
    returned byte-for-byte unchanged so an event is never lost or attributed
    to the wrong skill.
    """
    if not skill_name:
        return skill_name
    raw = os.fspath(skill_name) if isinstance(skill_name, os.PathLike) else str(skill_name)
    if not raw:
        return raw

    # A registered plugin skill always owns its qualified namespace. Do this
    # before local category matching so a local ``plugin/skill`` directory can
    # never collapse ``plugin:skill`` into a local record.
    if _registered_plugin_skill(raw):
        return raw

    aliases, canonical_by_dir = _skill_identity_index()
    target_dirs: Optional[Set[Path]] = None

    # Trusted absolute skill directory and SKILL.md forms are resolved through
    # the path index, not by accepting arbitrary filesystem paths.
    try:
        candidate_path = Path(raw).expanduser()
        if candidate_path.is_absolute():
            resolved = candidate_path.resolve(strict=False)
            if resolved.name == "SKILL.md":
                resolved = resolved.parent
            if resolved in canonical_by_dir:
                target_dirs = {resolved}
    except (OSError, RuntimeError, TypeError, ValueError):
        target_dirs = None

    if target_dirs is None:
        target_dirs = aliases.get(raw)

    # A path can have several matching skill directories (same directory alias
    # or duplicate frontmatter names). Refuse to guess and retain the event key.
    if target_dirs is None or len(target_dirs) != 1:
        return raw
    return canonical_by_dir.get(next(iter(target_dirs)), raw)


class UsageReconcileError(ValueError):
    """A controlled, user-safe error while inspecting the usage sidecar."""

    def __init__(self, code: str) -> None:
        self.code = code
        # Do not retain the underlying exception or path: callers may safely
        # render this exception at a CLI boundary without leaking sidecar data.
        super().__init__(code)


_RECONCILE_RECORD_FIELDS = {
    "created_by",
    "use_count",
    "view_count",
    "last_used_at",
    "last_viewed_at",
    "patch_count",
    "patch_generation",
    "last_reused_patch_generation",
    "last_patched_at",
    "created_at",
    "state",
    "pinned",
    "archived_at",
    "sync",
}


_RECONCILE_SKIP_REASONS = ("unknown", "ambiguous", "plugin")


def _strict_json_object_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Reject duplicate JSON keys instead of silently taking the last value."""
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject NaN/Infinity, which are not valid JSON sidecar values."""
    raise ValueError(value)


def _strict_usage_bytes() -> Tuple[bytes, bool]:
    """Read the usage sidecar as bytes without following a symlink."""
    path = _usage_file()
    try:
        path_stat = os.lstat(path)
    except FileNotFoundError:
        return b"", False
    except OSError:
        raise UsageReconcileError("usage_sidecar_unreadable")
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise UsageReconcileError("usage_sidecar_type")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except (OSError, ValueError):
        raise UsageReconcileError("usage_sidecar_unreadable")
    try:
        opened_stat = os.fstat(fd)
        if stat.S_ISLNK(opened_stat.st_mode) or not stat.S_ISREG(opened_stat.st_mode):
            raise UsageReconcileError("usage_sidecar_type")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            return handle.read(), True
    except UsageReconcileError:
        raise
    except (OSError, ValueError):
        raise UsageReconcileError("usage_sidecar_unreadable")
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def _strict_parse_usage(raw: bytes) -> Dict[str, Dict[str, Any]]:
    """Parse a complete usage sidecar without lossy coercions."""
    try:
        data = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError, TypeError):
        raise UsageReconcileError("invalid_usage_sidecar")
    if not isinstance(data, dict):
        raise UsageReconcileError("usage_sidecar_not_dict")
    for record in data.values():
        if not isinstance(record, dict):
            raise UsageReconcileError("usage_record_not_dict")
    return data


def _read_usage_strict_for_reconcile() -> Tuple[Dict[str, Dict[str, Any]], bool]:
    """Read the sidecar without the forgiving behavior of :func:`load_usage`."""
    raw, present = _strict_usage_bytes()
    if not present:
        return {}, False
    return _strict_parse_usage(raw), True


def _reconcile_alias_target(raw_key: str) -> Tuple[str, str]:
    """Return ``(canonical_name, disposition)`` for one raw sidecar key."""
    try:
        canonical = _canonicalize_skill_name(raw_key)
    except Exception:
        # Preserve an event that cannot be classified rather than aborting the
        # whole report because one skill tree entry is malformed.
        return raw_key, "unknown"

    try:
        if _registered_plugin_skill(raw_key):
            return canonical, "plugin"

        aliases, canonical_by_dir = _skill_identity_index()
        target_dirs: Optional[Set[Path]] = None
        candidate_path = Path(raw_key).expanduser()
        if candidate_path.is_absolute():
            resolved = candidate_path.resolve(strict=False)
            if resolved.name == "SKILL.md":
                resolved = resolved.parent
            if resolved in canonical_by_dir:
                target_dirs = {resolved}
        if target_dirs is None:
            target_dirs = aliases.get(raw_key)
    except (OSError, RuntimeError, TypeError, ValueError):
        return raw_key, "unknown"

    if target_dirs is None:
        return raw_key, "unknown"
    if len(target_dirs) != 1:
        return raw_key, "ambiguous"
    return canonical, "local"


def _safe_reconcile_alias(raw_key: str) -> Optional[str]:
    """Return a non-sensitive alias suitable for report metadata."""
    try:
        if Path(raw_key).expanduser().is_absolute():
            return None
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if not raw_key or "\n" in raw_key or "\r" in raw_key:
        return None
    return raw_key[:256]


def _reconcile_group_conflict_count(records: List[Dict[str, Any]]) -> int:
    """Count groups needing review before a future mutating merge."""
    if len(records) < 2:
        return 0
    unknown_fields = set().union(
        *(set(record) - _RECONCILE_RECORD_FIELDS for record in records)
    )
    for field in unknown_fields:
        values = [record[field] for record in records if field in record]
        if any(not _reconcile_values_equal(values[0], value) for value in values[1:]):
            return 1
    return 0


_RECONCILE_COUNTER_FIELDS = ("use_count", "view_count", "patch_count")


_RECONCILE_ACTIVITY_FIELDS = (
    "last_used_at",
    "last_viewed_at",
    "last_patched_at",
)


_RECONCILE_BOOL_FIELDS = ("pinned", "sync")


_RECONCILE_TIMESTAMP_FIELDS = (
    "created_at",
    *_RECONCILE_ACTIVITY_FIELDS,
    "archived_at",
)


_RECONCILE_STATE_PRIORITY = {
    STATE_ARCHIVED: 0,
    STATE_STALE: 1,
    STATE_ACTIVE: 2,
}


def _reconcile_groups_for_data(
    data: Dict[str, Dict[str, Any]],
) -> Tuple[List[Tuple[str, List[Tuple[str, Dict[str, Any]]]]], Dict[str, int]]:
    """Classify one already-read sidecar into safe groups and skipped rows."""
    groups_by_name: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    skipped = {reason: 0 for reason in _RECONCILE_SKIP_REASONS}
    for raw_key, record in data.items():
        canonical, disposition = _reconcile_alias_target(raw_key)
        if disposition != "local":
            skipped[disposition] += 1
            continue
        groups_by_name.setdefault(canonical, []).append((raw_key, record))

    groups = []
    for canonical in sorted(groups_by_name):
        entries = groups_by_name[canonical]
        if any(raw_key != canonical for raw_key, _ in entries):
            groups.append((canonical, entries))
    return groups, skipped


def _reconcile_validate_record(record: Dict[str, Any]) -> None:
    """Validate known fields before any backup or write is attempted."""
    for field in _RECONCILE_COUNTER_FIELDS + (
        "patch_generation",
        "last_reused_patch_generation",
    ):
        if field not in record:
            continue
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise UsageReconcileError("invalid_known_field")

    for field in _RECONCILE_BOOL_FIELDS:
        if field in record and not isinstance(record[field], bool):
            raise UsageReconcileError("invalid_known_field")

    if "state" in record and record["state"] not in _VALID_STATES:
        raise UsageReconcileError("invalid_known_field")

    if "created_by" in record:
        value = record["created_by"]
        if value is not None and not isinstance(value, str):
            raise UsageReconcileError("invalid_known_field")

    for field in _RECONCILE_TIMESTAMP_FIELDS:
        if field not in record:
            continue
        value = record[field]
        if value is None and field in _RECONCILE_ACTIVITY_FIELDS + ("archived_at",):
            continue
        if not isinstance(value, str) or _parse_iso_timestamp(value) is None:
            raise UsageReconcileError("invalid_known_timestamp")


def _reconcile_values_equal(left: Any, right: Any) -> bool:
    """Compare JSON values strictly, preserving numeric JSON representation."""
    try:
        return json.dumps(
            left, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ) == json.dumps(
            right, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return False


def _usage_commit_matches(expected: Dict[str, Dict[str, Any]]) -> bool:
    """Verify whether a reportedly failed save already committed *expected*."""
    try:
        path_stat = os.lstat(_usage_file())
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            return False
        if os.name != "nt" and stat.S_IMODE(path_stat.st_mode) != 0o600:
            return False
        raw, present = _strict_usage_bytes()
        return present and _reconcile_values_equal(_strict_parse_usage(raw), expected)
    except (OSError, UsageReconcileError):
        return False


def _reconcile_timestamp_result(
    records: List[Dict[str, Any]],
    field: str,
    *,
    minimum: bool = False,
) -> Optional[str]:
    values = [record[field] for record in records if field in record]
    valid = [(dt, str(raw)) for raw in values if (dt := _parse_iso_timestamp(raw)) is not None]
    if not valid:
        return None
    return (min if minimum else max)(valid, key=lambda item: item[0])[1]


def _reconcile_merge_record(
    canonical: str,
    records: List[Dict[str, Any]],
    *,
    bundled_names: Set[str],
    hub_names: Set[str],
) -> Dict[str, Any]:
    """Merge one safe local identity group according to field-specific rules."""
    for record in records:
        _reconcile_validate_record(record)

    merged: Dict[str, Any] = {}
    unknown_fields = sorted(
        set().union(*(set(record) - _RECONCILE_RECORD_FIELDS for record in records))
    )
    for field in unknown_fields:
        values = [record[field] for record in records if field in record]
        if any(not _reconcile_values_equal(values[0], value) for value in values[1:]):
            raise UsageReconcileError("conflicting_unknown_fields")
        # JSON values are immutable for this operation; save_usage serializes a
        # fresh representation, so retaining this value cannot mutate the read.
        merged[field] = values[0]

    for field in _RECONCILE_COUNTER_FIELDS:
        if any(field in record for record in records):
            merged[field] = sum(record.get(field, 0) for record in records)

    if any("created_at" in record for record in records):
        merged["created_at"] = _reconcile_timestamp_result(
            records, "created_at", minimum=True
        )

    for field in _RECONCILE_ACTIVITY_FIELDS:
        if any(field in record for record in records):
            merged[field] = _reconcile_timestamp_result(records, field)

    if any("patch_generation" in record for record in records):
        merged["patch_generation"] = max(
            record.get("patch_generation", 0) for record in records
        )
    if any("last_reused_patch_generation" in record for record in records):
        merged["last_reused_patch_generation"] = min(
            max(record.get("last_reused_patch_generation", 0) for record in records),
            merged.get("patch_generation", 0),
        )

    created_by_values = [
        record["created_by"] for record in records if "created_by" in record
    ]
    if canonical in bundled_names or canonical in hub_names:
        merged["created_by"] = "installed"
    elif any(value == "agent" for value in created_by_values):
        merged["created_by"] = "agent"
    elif any(value == "installed" for value in created_by_values):
        merged["created_by"] = "installed"
    elif created_by_values:
        first = created_by_values[0]
        if any(not _reconcile_values_equal(first, value) for value in created_by_values[1:]):
            raise UsageReconcileError("conflicting_created_by")
        merged["created_by"] = first

    if any("pinned" in record for record in records):
        merged["pinned"] = any(record.get("pinned", False) for record in records)

    if any("state" in record for record in records):
        merged_state = max(
            (record.get("state", STATE_ACTIVE) for record in records),
            key=lambda state: _RECONCILE_STATE_PRIORITY[state],
        )
        merged["state"] = merged_state
    else:
        merged_state = STATE_ACTIVE

    if any("archived_at" in record for record in records) or merged_state == STATE_ARCHIVED:
        merged["archived_at"] = (
            _reconcile_timestamp_result(records, "archived_at")
            if merged_state == STATE_ARCHIVED
            else None
        )

    if any("sync" in record for record in records):
        merged["sync"] = any(record.get("sync", False) for record in records)

    return merged


def _reconcile_plan_counts(
    data: Dict[str, Dict[str, Any]],
    groups: List[Tuple[str, List[Tuple[str, Dict[str, Any]]]]],
    skipped: Dict[str, int],
) -> Dict[str, int]:
    aliases = sum(
        sum(1 for raw_key, _ in entries if raw_key != canonical)
        for canonical, entries in groups
    )
    return {
        "records": len(data),
        "groups": len(groups),
        "aliases": aliases,
        "possible_conflicts": sum(
            _reconcile_group_conflict_count([record for _, record in entries])
            for _, entries in groups
        ),
        "skipped": sum(skipped.values()),
    }


def _open_private_backup_dir() -> Tuple[Path, int]:
    """Open ``.usage_backups`` after rejecting symlinks and wrong types."""
    skills_dir = _skills_dir()
    try:
        skills_stat = os.lstat(skills_dir)
    except OSError:
        raise UsageReconcileError("backup_directory_unavailable")
    if stat.S_ISLNK(skills_stat.st_mode) or not stat.S_ISDIR(skills_stat.st_mode):
        raise UsageReconcileError("backup_directory_type")

    backup_dir = skills_dir / ".usage_backups"
    try:
        os.mkdir(backup_dir, 0o700)
    except FileExistsError:
        pass
    except OSError:
        raise UsageReconcileError("backup_directory_unavailable")

    try:
        backup_stat = os.lstat(backup_dir)
    except OSError:
        raise UsageReconcileError("backup_directory_unavailable")
    if stat.S_ISLNK(backup_stat.st_mode) or not stat.S_ISDIR(backup_stat.st_mode):
        raise UsageReconcileError("backup_directory_type")
    try:
        os.chmod(backup_dir, 0o700, follow_symlinks=False)
    except (NotImplementedError, TypeError):  # pragma: no cover - Windows
        os.chmod(backup_dir, 0o700)

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(backup_dir, flags)
        opened_stat = os.fstat(fd)
    except (OSError, ValueError):
        raise UsageReconcileError("backup_directory_unavailable")
    if stat.S_ISLNK(opened_stat.st_mode) or not stat.S_ISDIR(opened_stat.st_mode):
        os.close(fd)
        raise UsageReconcileError("backup_directory_type")
    return backup_dir, fd


def _create_usage_backup(raw: bytes) -> None:
    """Create a collision-safe, byte-identical private sidecar backup."""
    backup_dir, directory_fd = _open_private_backup_dir()
    try:
        for index in range(128):
            suffix = "" if index == 0 else f"-{index}"
            name = (
                "usage-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
                f"{secrets.token_hex(8)}{suffix}.json"
            )
            candidate = backup_dir / name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            try:
                fd = os.open(candidate, flags, 0o600)
            except FileExistsError:
                continue
            try:
                if hasattr(os, "fchmod"):
                    os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    fd = -1
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.chmod(candidate, 0o600, follow_symlinks=False)
                except (NotImplementedError, TypeError):  # pragma: no cover - Windows
                    os.chmod(candidate, 0o600)
                os.fsync(directory_fd)
                return
            except BaseException:
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                try:
                    candidate.unlink()
                except OSError:
                    pass
                raise UsageReconcileError("backup_failed")
        raise UsageReconcileError("backup_collision")
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            pass


def reconcile_usage_report() -> Dict[str, Any]:
    """Build a sanitized, read-only alias reconciliation report.

    Only keys that resolve to one local skill directory enter ``groups``.
    Unknown, ambiguous, and registered-plugin keys remain untouched and are
    represented only by disposition counts. The report contains no record
    values, extension field names, or absolute paths.
    """
    data, present = _read_usage_strict_for_reconcile()
    groups_by_name: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    skipped = {reason: 0 for reason in _RECONCILE_SKIP_REASONS}

    for raw_key, record in data.items():
        canonical, disposition = _reconcile_alias_target(raw_key)
        if disposition != "local":
            skipped[disposition] += 1
            continue
        groups_by_name.setdefault(canonical, []).append((raw_key, record))

    groups: List[Dict[str, Any]] = []
    possible_conflicts = 0
    alias_count = 0
    for canonical in sorted(groups_by_name):
        entries = groups_by_name[canonical]
        noncanonical_entries = [
            (raw_alias, record)
            for raw_alias, record in entries
            if raw_alias != canonical
        ]
        if not noncanonical_entries:
            continue
        safe_aliases: List[str] = []
        for raw_alias, _ in sorted(noncanonical_entries, key=lambda item: item[0]):
            safe_alias = _safe_reconcile_alias(raw_alias)
            if safe_alias is not None:
                safe_aliases.append(safe_alias)
        conflict = _reconcile_group_conflict_count([record for _, record in entries])
        possible_conflicts += conflict
        alias_count += len(noncanonical_entries)
        groups.append(
            {
                "canonical": canonical,
                "aliases": safe_aliases,
                "record_count": len(entries),
                "possible_conflicts": conflict,
            }
        )

    skipped_total = sum(skipped.values())
    counts = {
        "records": len(data),
        "groups": len(groups),
        "aliases": alias_count,
        "possible_conflicts": possible_conflicts,
        "skipped": skipped_total,
    }
    return {
        "status": "clean" if not data else "ok",
        "dry_run": True,
        "source_present": present,
        "counts": counts,
        "groups": groups,
        "skipped": skipped,
        "possible_conflicts": possible_conflicts,
    }


def reconcile_usage_apply() -> Dict[str, Any]:
    """Apply a freshly planned, fail-closed alias reconciliation.

    The lock deliberately surrounds the strict byte read, planning, backup, and
    save.  A report made before this function is only advisory; the apply path
    always re-reads and re-plans the current sidecar before creating a backup.
    """
    try:
        with _usage_file_lock():
            raw, present = _strict_usage_bytes()
            if not present:
                return {
                    "status": "clean",
                    "dry_run": False,
                    "source_present": False,
                    "counts": {
                        "records": 0,
                        "groups": 0,
                        "aliases": 0,
                        "possible_conflicts": 0,
                        "skipped": 0,
                    },
                    "skipped": {reason: 0 for reason in _RECONCILE_SKIP_REASONS},
                }

            data = _strict_parse_usage(raw)
            # Validate skipped records too.  An apply must never silently carry
            # a malformed known field forward while mutating another group.
            for record in data.values():
                _reconcile_validate_record(record)

            groups, skipped = _reconcile_groups_for_data(data)
            counts = _reconcile_plan_counts(data, groups, skipped)
            if not groups:
                # Unknown/ambiguous/plugin records are intentionally preserved;
                # with no safe alias there is no mutation and no backup.
                return {
                    "status": "clean",
                    "dry_run": False,
                    "source_present": True,
                    "counts": counts,
                    "skipped": skipped,
                }

            bundled_names = _read_bundled_manifest_names()
            hub_names = _read_hub_installed_names()
            merged_by_canonical: Dict[str, Dict[str, Any]] = {}
            for canonical, entries in groups:
                merged_by_canonical[canonical] = _reconcile_merge_record(
                    canonical,
                    [record for _, record in entries],
                    bundled_names=bundled_names,
                    hub_names=hub_names,
                )

            replanned = dict(data)
            for canonical, entries in groups:
                for raw_key, _ in entries:
                    if raw_key != canonical:
                        replanned.pop(raw_key, None)
                replanned[canonical] = merged_by_canonical[canonical]

            # The backup is the exact byte sequence that was parsed and is made
            # only after every group has passed validation/conflict checks.
            _create_usage_backup(raw)
            if not save_usage(replanned) and not _usage_commit_matches(replanned):
                raise UsageReconcileError("save_failed")
            return {
                "status": "applied",
                "dry_run": False,
                "source_present": True,
                "counts": counts,
                "skipped": skipped,
            }
    except UsageReconcileError:
        raise
    except Exception:
        # Keep the CLI boundary free of filesystem paths, values, and tracebacks.
        raise UsageReconcileError("apply_failed")
