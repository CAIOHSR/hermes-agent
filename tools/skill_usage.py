"""Skill usage telemetry + provenance tracking for the Curator feature.

Tracks per-skill usage metadata in a sidecar JSON file (~/.hermes/skills/.usage.json)
keyed by skill name. Counters are bumped by the existing skill tools (skill_view,
skill_manage); the curator orchestrator reads the derived activity timestamp to
decide lifecycle transitions.

Design notes:
  - Sidecar, not frontmatter. Keeps operational telemetry out of user-authored
    SKILL.md content and avoids conflict pressure for bundled/hub skills.
  - Atomic writes via tempfile + os.replace (same pattern as .bundled_manifest).
  - All counter bumps are best-effort: failures log at DEBUG and return silently.
    A broken sidecar never breaks the underlying tool call.
  - Provenance filter: curator-managed skills are explicitly marked when
    created through skill_manage. Bundled / hub-installed skills stay
    off-limits, and manually authored skills are not inferred from location.

Lifecycle states:
    active    -> default
    stale     -> unused > stale_after_days (config)
    archived  -> unused > archive_after_days (config); moved to .archive/
    pinned    -> opt-out from auto transitions (boolean flag, orthogonal to state)
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home
from agent.skill_utils import is_excluded_skill_path, is_external_skill_path

logger = logging.getLogger(__name__)

# fcntl is Unix-only; on Windows use msvcrt for file locking.
msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-specific fallback
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass


STATE_ACTIVE = "active"
STATE_STALE = "stale"
STATE_ARCHIVED = "archived"
_VALID_STATES = {STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED}

# Load-bearing bundled built-ins the curator must NEVER archive or consolidate,
# regardless of ``curator.prune_builtins``, pin state, or LLM judgment. These
# back advertised UX paths; silently archiving one turns its slash command
# into "Unknown command" with no signal to the user.
# Protection is by skill ``name`` (frontmatter ``name:``), matching the keys used
# throughout this module. Keep this list tiny and intentional — it is not a
# substitute for ``curator.prune_builtins: false``, which exempts ALL built-ins.
# (``plan`` used to live here; it is now a first-class built-in command with
# no skill on disk, so the set is currently empty.)
PROTECTED_BUILTIN_SKILLS: Set[str] = set()


def is_protected_builtin(skill_name: str) -> bool:
    """Whether *skill_name* is a load-bearing built-in the curator never touches.

    Protected built-ins are exempt from archival and consolidation on every
    path: the automatic state-transition walk, the LLM consolidation pass (they
    are dropped from the candidate list), and direct ``archive_skill`` calls.
    """
    return skill_name in PROTECTED_BUILTIN_SKILLS


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def _usage_file() -> Path:
    return _skills_dir() / ".usage.json"


@contextmanager
def _usage_file_lock():
    """Serialize .usage.json read-modify-write cycles across processes."""
    lock_path = _usage_file().with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if fcntl is None and msvcrt is None:
        yield
        return

    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        lock_path.write_text(" ", encoding="utf-8")

    fd = open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8")
    try:
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            fd.seek(0)
            msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        if fcntl:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except (OSError, IOError):
                pass
        elif msvcrt:
            try:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
        fd.close()


def _archive_dir() -> Path:
    return _skills_dir() / ".archive"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO timestamp defensively for activity comparisons."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def latest_activity_at(record: Dict[str, Any]) -> Optional[str]:
    """Return the newest actual activity timestamp for a usage record.

    "Activity" means a skill was used, viewed, or patched. Creation time is
    intentionally excluded so callers can still distinguish never-active skills;
    lifecycle code can fall back to ``created_at`` as its own anchor.
    """
    latest_dt: Optional[datetime] = None
    latest_raw: Optional[str] = None
    for key in ("last_used_at", "last_viewed_at", "last_patched_at"):
        raw = record.get(key)
        dt = _parse_iso_timestamp(raw)
        if dt is None:
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt = dt
            latest_raw = str(raw)
    return latest_raw


def activity_count(record: Dict[str, Any]) -> int:
    """Return the total observed activity count across use/view/patch events."""
    total = 0
    for key in ("use_count", "view_count", "patch_count"):
        try:
            total += int(record.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return total


# ---------------------------------------------------------------------------
# Provenance — which skills are agent-created (and thus eligible for curation)
# ---------------------------------------------------------------------------

def _read_bundled_manifest_names() -> Set[str]:
    """Return the set of skill names that were seeded from the bundled repo.

    Reads ~/.hermes/skills/.bundled_manifest (format: "name:hash" per line).
    Returns empty set if the file is missing or unreadable.
    """
    manifest = _skills_dir() / ".bundled_manifest"
    if not manifest.exists():
        return set()
    names: Set[str] = set()
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            name = line.split(":", 1)[0].strip()
            if name:
                names.add(name)
    except OSError as e:
        logger.debug("Failed to read bundled manifest: %s", e)
    return names


def _read_hub_installed_names() -> Set[str]:
    """Return the set of skill names installed via the Skills Hub.

    Reads ~/.hermes/skills/.hub/lock.json (see tools/skills_hub.py :: HubLockFile).
    """
    lock_path = _skills_dir() / ".hub" / "lock.json"
    if not lock_path.exists():
        return set()
    try:
        # Tolerate non-UTF-8 bytes in the lock file. Hub descriptions can carry
        # Windows-1252 typographic chars (em-dash 0x97, smart quotes, bullets)
        # written as single high bytes; a strict utf-8 read raises
        # UnicodeDecodeError, which is a ValueError sibling (not OSError/
        # JSONDecodeError) so it escapes the handler below and 500s the whole
        # /api/skills endpoint. errors="replace" degrades the offending byte to
        # U+FFFD, keeping the (structurally valid) JSON — and every other
        # skill — readable. See #68053.
        data = json.loads(lock_path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, dict):
            installed = data.get("installed") or {}
            if isinstance(installed, dict):
                names = {str(k) for k in installed.keys()}
                skills_dir = _skills_dir()
                for entry in installed.values():
                    if not isinstance(entry, dict):
                        continue
                    install_path = entry.get("install_path")
                    if not isinstance(install_path, str) or not install_path.strip():
                        continue
                    skill_dir = Path(install_path)
                    if not skill_dir.is_absolute():
                        skill_dir = skills_dir / skill_dir
                    try:
                        resolved = skill_dir.resolve()
                        resolved.relative_to(skills_dir.resolve())
                    except (OSError, ValueError):
                        continue
                    skill_md = resolved / "SKILL.md"
                    if skill_md.exists():
                        names.add(_read_skill_name(skill_md, fallback=resolved.name))
                return names
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read hub lock file: %s", e)
    return set()


def _prune_builtins_enabled() -> bool:
    """Whether bundled built-in skills are eligible for curator pruning.

    Reads ``curator.prune_builtins`` from config (default True). Lazy import
    keeps this module importable without the CLI config layer (e.g. in the
    update/sync context); on any failure we fall back to the default. The real
    safety against a mass-prune is the curator's seed-on-first-sight, not this
    flag — built-ins only archive after a fresh inactivity window.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        cur = cfg.get("curator") if isinstance(cfg, dict) else None
        if isinstance(cur, dict):
            return bool(cur.get("prune_builtins", True))
    except Exception as e:  # pragma: no cover — best-effort config read
        logger.debug("Failed to read curator.prune_builtins: %s", e)
    return True


def _suppressed_file() -> Path:
    return _skills_dir() / ".curator_suppressed"


def read_suppressed_names() -> Set[str]:
    """Built-in skills the curator pruned — the re-seeder must leave archived.

    One skill name per line in ``~/.hermes/skills/.curator_suppressed``. This is
    what makes pruning a built-in durable: without it, ``hermes update`` would
    re-copy the bundled skill on the next sync.
    """
    path = _suppressed_file()
    if not path.exists():
        return set()
    names: Set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.add(line)
    except OSError as e:
        logger.debug("Failed to read curator suppression list: %s", e)
    return names


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


def remove_suppressed_name(skill_name: str) -> None:
    """Clear a built-in's suppression entry (e.g. on restore)."""
    if not skill_name:
        return
    names = read_suppressed_names()
    if skill_name in names:
        names.discard(skill_name)
        _write_suppressed_names(names)


def list_agent_created_skill_names() -> List[str]:
    """Enumerate skills the curator may manage.

    Always includes agent-authored skills (those marked in ``.usage.json`` via
    ``skill_manage(action="create")``). When ``curator.prune_builtins`` is
    enabled, bundled built-in skills are ALSO included even though they have no
    agent-created usage record — their inactivity clock is anchored on first
    sight (see ``apply_automatic_transitions``). Hub-installed skills are never
    included; manually authored skills are not inferred from filesystem
    location.
    """
    base = _skills_dir()
    if not base.exists():
        return []
    hub = _read_hub_installed_names()
    bundled = _read_bundled_manifest_names()
    prune_builtins = _prune_builtins_enabled()
    usage = load_usage()

    names: List[str] = []
    # Top-level SKILL.md files (flat layout) AND nested category/skill/SKILL.md
    for skill_md in base.rglob("SKILL.md"):
        # Skip Hermes metadata, VCS, virtualenv/dependency, and cache dirs
        if is_excluded_skill_path(skill_md):
            continue
        # External skill dirs can be mounted below the local skills tree.
        # Discovery may see them, but autonomous lifecycle curation must not.
        if is_external_skill_path(skill_md):
            continue
        try:
            skill_md.relative_to(base)
        except ValueError:
            continue
        name = _read_skill_name(skill_md, fallback=skill_md.parent.name)
        # Hub-installed skills are always off-limits.
        if name in hub:
            continue
        # Protected built-ins are never curation candidates — exempt from the
        # automatic transition walk AND the LLM consolidation pass.
        if is_protected_builtin(name):
            continue
        if name in bundled:
            # Built-ins are only candidates when pruning is enabled. They never
            # carry a curator-managed record, so the record gate is skipped.
            if not prune_builtins:
                continue
            names.append(name)
            continue
        # Agent-authored (or local-manual) skills must opt in via their record.
        if not _is_curator_managed_record(usage.get(name)):
            continue
        names.append(name)
    return sorted(set(names))


def list_archived_skill_names() -> List[str]:
    """Enumerate skills in ``~/.hermes/skills/.archive/``.

    Archive layout is flat (``.archive/<skill>/``) as set by ``archive_skill``,
    so the directory name is the skill name. Used by ``hermes curator
    list-archived`` to help users pass a name to ``hermes curator restore``.
    """
    archive_root = _archive_dir()
    if not archive_root.exists():
        return []
    return sorted({p.name for p in archive_root.iterdir() if p.is_dir()})


def _read_skill_name(skill_md: Path, fallback: str) -> str:
    """Parse the `name:` field from a SKILL.md YAML frontmatter."""
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return fallback
    in_frontmatter = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "---":
            if in_frontmatter:
                break
            in_frontmatter = True
            continue
        if in_frontmatter and stripped.startswith("name:"):
            value = stripped.split(":", 1)[1].strip().strip("\"'")
            if value:
                return value
    return fallback


# The identity index is deliberately scoped to the active profile's local
# skills root. Known SKILL.md files and their ancestor directories are watched
# with cheap stat calls; a new nested skill changes one of those parent dirs.
# The recursive walk is therefore amortized across bumps rather than paid on
# every event. The root path remains part of the key so switching HERMES_HOME
# (including profile overrides in one process) cannot leak names.
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


def is_agent_created(skill_name: str) -> bool:
    """Whether *skill_name* is neither bundled nor hub-installed."""
    skill_name = _canonicalize_skill_name(skill_name)
    off_limits = _read_bundled_manifest_names() | _read_hub_installed_names()
    if skill_name in off_limits:
        return False
    return not (
        _find_skill_dir(skill_name) is None
        and _find_external_skill_dir(skill_name) is not None
    )


def is_hub_installed(skill_name: str) -> bool:
    """Whether *skill_name* was installed via the Skills Hub."""
    skill_name = _canonicalize_skill_name(skill_name)
    return skill_name in _read_hub_installed_names()


def is_bundled(skill_name: str) -> bool:
    """Whether *skill_name* was seeded from the bundled repo skills."""
    skill_name = _canonicalize_skill_name(skill_name)
    return skill_name in _read_bundled_manifest_names()


def _external_read_only_message(skill_name: str) -> str:
    return (
        f"skill '{skill_name}' lives in skills.external_dirs; "
        "external skills are read-only to the curator"
    )


def is_curation_eligible(skill_name: str, skill_path: Optional[Path] = None) -> bool:
    """Whether the curator may track/archive *skill_name*.

    Agent-created skills are always eligible. Bundled built-ins become eligible
    only when ``curator.prune_builtins`` is enabled. Hub-installed and external
    skill-dir skills are NEVER eligible — they have an external upstream owner.
    Org-shared skills ARE eligible for improvement (the curator may patch them
    like any other skill; edits stay local until proposed) but are protected
    from ARCHIVE/DELETE elsewhere — removing a shared skill is an org-admin
    action, not a local curation decision.
    Protected built-ins (``PROTECTED_BUILTIN_SKILLS``) are NEVER eligible
    regardless of any flag — they back load-bearing UX and must never be
    archived or consolidated.
    """
    skill_name = _canonicalize_skill_name(skill_name)
    if skill_path is not None and is_external_skill_path(skill_path):
        return False
    if is_protected_builtin(skill_name):
        return False
    if is_hub_installed(skill_name):
        return False
    if is_bundled(skill_name):
        return _prune_builtins_enabled()
    local_dir = _find_skill_dir(skill_name)
    if local_dir is not None:
        return not is_external_skill_path(local_dir)
    if _find_external_skill_dir(skill_name) is not None:
        return False
    return True


def _is_curator_managed_record(record: Any) -> bool:
    """Return True when a usage record opts a skill into curator management.

    NAMING (issue #67140): the on-disk field is ``created_by``, which reads
    like provenance but is consumed as a **curator-management opt-in policy
    flag**. The two are not the same question:

    * provenance = "who authored this file" — historical fact, and for records
      written before the marker existed it is simply unrecoverable.
    * management = "may autonomous curation mutate/archive this" — a policy
      decision the user can change at any time via ``hermes curator adopt``.

    ``created_by: "agent"`` therefore means "curator-managed", NOT "proof the
    agent wrote it". The field name is retained because it is already on disk
    in every user's ``.usage.json``; renaming it would strand those records.
    Read it as policy, and prefer ``is_curator_managed()`` at call sites so the
    intent is unambiguous.
    """
    if not isinstance(record, dict):
        return False
    return record.get("created_by") == "agent" or record.get("agent_created") is True


def is_curator_managed(skill_name: str) -> bool:
    """Whether *skill_name* is opted into curator management.

    Policy-intent alias for the ``created_by``-marker check, so call sites read
    as the question they are actually asking (see ``_is_curator_managed_record``
    for why the stored field name says "created_by").
    """
    return _is_curator_managed_record(load_usage().get(skill_name))


def list_unmanaged_skill_names() -> List[str]:
    """Enumerate curation-ELIGIBLE skills that carry no provenance marker.

    These are skills the curator *could* manage (they are not hub-installed,
    not external, not protected built-ins) but never will, because nothing
    ever wrote ``created_by: agent`` onto their usage record. Two ways a skill
    lands here:

    * It predates the provenance mechanism entirely — records written before
      ``created_by`` existed carry no key at all, so their authorship is
      unknowable from the record alone.
    * It was created by a FOREGROUND ``skill_manage(action="create")`` call,
      which deliberately does not mark provenance (skills a user asks for
      belong to the user).

    Either way the skill is invisible to ``curated_report()`` and therefore to
    every automatic transition. ``hermes curator status`` surfaces this count
    so the blind spot is legible instead of silent, and ``hermes curator
    adopt`` lets the user hand specific skills over explicitly.

    Provenance is a DECLARATION, never an inference: this function only
    reports, and callers must not auto-adopt what it returns. Heavy patch or
    use counts are evidence of maintenance, not of authorship — the agent
    edits user-authored skills on the user's behalf routinely.
    """
    base = _skills_dir()
    if not base.exists():
        return []
    hub = _read_hub_installed_names()
    bundled = _read_bundled_manifest_names()
    usage = load_usage()

    names: List[str] = []
    for skill_md in base.rglob("SKILL.md"):
        if is_excluded_skill_path(skill_md) or is_external_skill_path(skill_md):
            continue
        try:
            skill_md.relative_to(base)
        except ValueError:
            continue
        name = _read_skill_name(skill_md, fallback=skill_md.parent.name)
        # Anything with an external owner or a bundled/protected identity is
        # outside the adoption question entirely.
        if name in hub or name in bundled or is_protected_builtin(name):
            continue
        if _is_curator_managed_record(usage.get(name)):
            continue
        if not is_curation_eligible(name, skill_md):
            continue
        names.append(name)
    return sorted(set(names))


def unmanaged_report() -> List[Dict[str, Any]]:
    """Rows for every skill :func:`list_unmanaged_skill_names` returns.

    Each row carries the usual activity fields plus ``has_provenance_key``:
    False when the record has no ``created_by`` key at all (pre-dates the
    mechanism), True when the key is present but unset (a foreground create
    under the current policy). The distinction matters for explaining WHY a
    skill is unmanaged; it is not a signal to adopt on.
    """
    usage = load_usage()
    rows: List[Dict[str, Any]] = []
    for name in list_unmanaged_skill_names():
        raw = usage.get(name)
        rec: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else _empty_record()
        for k, v in _empty_record().items():
            rec.setdefault(k, v)
        row = {"name": name, **rec}
        row["has_provenance_key"] = isinstance(raw, dict) and "created_by" in raw
        row["has_record"] = isinstance(raw, dict)
        row["last_activity_at"] = latest_activity_at(row)
        row["activity_count"] = activity_count(row)
        rows.append(row)
    return rows


def adopt_skill(skill_name: str) -> Tuple[bool, str]:
    """Hand *skill_name* to the curator by user declaration.

    Writes the same ``created_by: agent`` marker the background review fork
    writes, so the skill joins ``curated_report()`` and the automatic
    transition walk. The inactivity clock is NOT reset: the skill's existing
    ``last_activity_at`` still governs staleness, so adopting something idle
    for months does not buy it a fresh window (nor does it archive it on the
    spot — the state machine decides on the next pass).

    Returns (ok, message). Refuses hub-installed, external, and protected
    built-in skills, which have an owner other than the user.
    """
    skill_name = _canonicalize_skill_name(skill_name)
    if not skill_name:
        return False, "no skill name given"
    if is_protected_builtin(skill_name):
        return False, f"'{skill_name}' is a protected built-in; the curator never manages it"
    if is_hub_installed(skill_name):
        return False, f"'{skill_name}' is hub-installed; its upstream owns it"
    if is_bundled(skill_name):
        # Bundled skills already fall under the curator via
        # ``curator.prune_builtins``; stamping created_by=agent on one would
        # claim Hermes' own shipped skill was agent-authored and change nothing
        # about its eligibility.
        return False, (
            f"'{skill_name}' is a bundled built-in — it is governed by "
            "curator.prune_builtins, not by adoption"
        )
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None:
        if _find_external_skill_dir(skill_name) is not None:
            return False, f"'{skill_name}' lives in skills.external_dirs and is read-only to the curator"
        return False, f"skill '{skill_name}' not found"
    if is_external_skill_path(skill_dir):
        return False, _external_read_only_message(skill_name)
    usage = load_usage()
    if _is_curator_managed_record(usage.get(skill_name)):
        return True, f"'{skill_name}' is already curator-managed"
    mark_agent_created(skill_name)
    if not _is_curator_managed_record(load_usage().get(skill_name)):
        return False, f"could not mark '{skill_name}' as curator-managed"
    return True, f"adopted '{skill_name}' into curator management"


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------

def _empty_record() -> Dict[str, Any]:
    return {
        "created_by": None,
        "use_count": 0,
        "view_count": 0,
        "last_used_at": None,
        "last_viewed_at": None,
        "patch_count": 0,
        "patch_generation": 0,
        "last_reused_patch_generation": 0,
        "last_patched_at": None,
        "created_at": _now_iso(),
        "state": STATE_ACTIVE,
        "pinned": False,
        "archived_at": None,
    }


def load_usage() -> Dict[str, Dict[str, Any]]:
    """Read the entire .usage.json map. Returns empty dict on missing/corrupt."""
    path = _usage_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    # Defensive: coerce any non-dict values to a fresh empty record
    clean: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            clean[str(k)] = v
    return clean


def save_usage(data: Dict[str, Dict[str, Any]]) -> bool:
    """Write the usage map atomically and report whether it committed."""
    path = _usage_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=".usage_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                # The usage sidecar contains operational history and may carry
                # extension fields.  Keep the replacement private before it is
                # made visible, rather than relying on the process umask.
                if hasattr(os, "fchmod"):
                    os.fchmod(f.fileno(), 0o600)
                json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            # fchmod above is the normal path; chmod is retained for platforms
            # without fchmod and makes the postcondition explicit.
            try:
                os.chmod(path, 0o600, follow_symlinks=False)
            except (NotImplementedError, TypeError):  # pragma: no cover - Windows
                os.chmod(path, 0o600)
            return True
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("Failed to write %s: %s", path, e, exc_info=True)
        return False


def get_record(skill_name: str) -> Dict[str, Any]:
    """Return the record for *skill_name*, creating a fresh one if missing."""
    skill_name = _canonicalize_skill_name(skill_name)
    data = load_usage()
    rec = data.get(skill_name)
    if not isinstance(rec, dict):
        return _empty_record()
    # Backfill any missing keys so callers don't need to handle old files
    base = _empty_record()
    for k, v in base.items():
        rec.setdefault(k, v)
    return rec


def seed_record_if_missing(skill_name: str) -> None:
    """Persist a baseline usage record for a curation-eligible skill.

    Built-ins carry no usage record until something touches them, which leaves
    their inactivity clock with no anchor. Seeding a record here fixes
    ``created_at`` to the moment the curator first sees the skill, so the
    archive/stale clock measures non-use FROM THEN — not from epoch. No-op when
    a record already exists or the skill isn't curation-eligible.
    """
    skill_name = _canonicalize_skill_name(skill_name)
    if not skill_name or not is_curation_eligible(skill_name):
        return
    try:
        with _usage_file_lock():
            usage_file_existed = _usage_file().exists()
            data = load_usage()
            if isinstance(data.get(skill_name), dict):
                return
            data[skill_name] = _empty_record()
            if save_usage(data) and not usage_file_existed:
                _refresh_identity_cache_signature()
    except Exception as e:
        logger.debug("skill_usage.seed_record_if_missing(%s) failed: %s", skill_name, e, exc_info=True)


def _mutate(skill_name: str, mutator, *, require_curation_eligible: bool = False) -> Any:
    """Load, apply *mutator(record)* in place, save. Best-effort.

    By default this records telemetry for ANY skill — bundled, hub-installed,
    or agent-created — because usage tracking is pure observability and is
    orthogonal to whether a skill is ever curated. Lifecycle mutators
    (``set_state``, ``set_pinned``, ``mark_agent_created``) pass
    ``require_curation_eligible=True`` so they never write meaningless state
    onto a skill the curator can't manage (e.g. an ``archived`` flag on a
    hub-installed skill).
    """
    skill_name = _canonicalize_skill_name(skill_name)
    if not skill_name:
        return None
    try:
        if require_curation_eligible and not is_curation_eligible(skill_name):
            return None
        with _usage_file_lock():
            usage_file_existed = _usage_file().exists()
            data = load_usage()
            rec = data.get(skill_name)
            if not isinstance(rec, dict):
                rec = _empty_record()
            result = mutator(rec)
            data[skill_name] = rec
            if not save_usage(data):
                return None
            if not usage_file_existed:
                _refresh_identity_cache_signature()
            return result
    except Exception as e:
        logger.debug("skill_usage._mutate(%s) failed: %s", skill_name, e, exc_info=True)
        return None


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def telemetry_provenance(
    skill_name: str,
    record: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the bounded provenance used by shared skill metrics."""
    skill_name = _canonicalize_skill_name(skill_name)
    if is_hub_installed(skill_name) or is_bundled(skill_name):
        return "installed"
    if ":" in skill_name:
        try:
            from hermes_cli.plugins import get_plugin_manager

            if get_plugin_manager().find_plugin_skill(skill_name) is not None:
                return "installed"
        except Exception:
            pass
    if isinstance(record, dict):
        created_by = record.get("created_by")
        if created_by == "installed":
            return "installed"
        if created_by == "agent":
            return "agent_created"
    if _find_external_skill_dir(skill_name) is not None:
        return "external"
    if _find_skill_dir(skill_name) is not None or isinstance(record, dict):
        return "local"
    return "unknown"


def _emit_skill_lifecycle(
    skill_name: str,
    action: str,
    *,
    record: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    use_count: Optional[int] = None,
    reused: Optional[bool] = None,
    reuse_after_patch: Optional[bool] = None,
) -> None:
    """Emit one best-effort lifecycle fact after authoritative state changes."""
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook

        if not has_hook("on_skill_lifecycle"):
            return
        invoke_hook(
            "on_skill_lifecycle",
            action=action,
            skill_name=skill_name,
            provenance=telemetry_provenance(skill_name, record),
            task_id=task_id or "",
            session_id=session_id or "",
            use_count=use_count,
            reused=reused,
            reuse_after_patch=reuse_after_patch,
        )
    except Exception:
        logger.debug(
            "skill_usage lifecycle hook failed for %s/%s",
            skill_name,
            action,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Public counter-bump helpers — telemetry for ALL skills (observability only)
# ---------------------------------------------------------------------------

def bump_view(skill_name: str) -> None:
    """Bump view_count and last_viewed_at. Called from skill_view().

    Tracks every skill regardless of provenance — built-ins and hub skills
    included. Usage telemetry is observability, not a curation signal.
    """
    skill_name = _canonicalize_skill_name(skill_name)

    def _apply(rec: Dict[str, Any]) -> None:
        rec["view_count"] = _non_negative_int(rec.get("view_count")) + 1
        rec["last_viewed_at"] = _now_iso()
    _mutate(skill_name, _apply)


def bump_use(
    skill_name: str,
    *,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Bump use_count and last_used_at. Called when a skill is actively used
    (e.g. loaded into the prompt path or referenced from an assistant turn).

    Tracks every skill regardless of provenance.
    """
    skill_name = _canonicalize_skill_name(skill_name)

    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        previous_use_count = _non_negative_int(rec.get("use_count"))
        patch_generation = _non_negative_int(rec.get("patch_generation"))
        last_reused_generation = min(
            _non_negative_int(rec.get("last_reused_patch_generation")),
            patch_generation,
        )
        reused = previous_use_count > 0
        reuse_after_patch = reused and patch_generation > last_reused_generation
        rec["use_count"] = previous_use_count + 1
        rec["last_used_at"] = _now_iso()
        rec["patch_generation"] = patch_generation
        rec["last_reused_patch_generation"] = last_reused_generation
        if reuse_after_patch:
            rec["last_reused_patch_generation"] = patch_generation
        return {
            "created_by": rec.get("created_by"),
            "use_count": rec["use_count"],
            "reused": reused,
            "reuse_after_patch": reuse_after_patch,
        }

    facts = _mutate(skill_name, _apply)
    if isinstance(facts, dict):
        _emit_skill_lifecycle(
            skill_name,
            "loaded",
            record=facts,
            task_id=task_id,
            session_id=session_id,
            use_count=facts["use_count"],
            reused=facts["reused"],
            reuse_after_patch=facts["reuse_after_patch"],
        )


def bump_patch(
    skill_name: str,
    *,
    action: str = "patch",
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Bump patch_count and last_patched_at. Called from skill_manage (patch/edit).

    Tracks every skill regardless of provenance.
    """
    skill_name = _canonicalize_skill_name(skill_name)
    lifecycle_action = "patched" if action == "patch" else "edited"

    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        rec["patch_count"] = _non_negative_int(rec.get("patch_count")) + 1
        rec["patch_generation"] = _non_negative_int(rec.get("patch_generation")) + 1
        rec["last_patched_at"] = _now_iso()
        return {"created_by": rec.get("created_by")}

    facts = _mutate(skill_name, _apply)
    if isinstance(facts, dict):
        _emit_skill_lifecycle(
            skill_name,
            lifecycle_action,
            record=facts,
            task_id=task_id,
            session_id=session_id,
        )


def record_created(
    skill_name: str,
    *,
    agent_created: bool,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Persist explicit creation provenance and emit a successful create fact."""
    skill_name = _canonicalize_skill_name(skill_name)

    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        # A successful create is a new logical skill even if stale sidecar
        # state survived an earlier deletion or manual filesystem change.
        rec.clear()
        rec.update(_empty_record())
        if agent_created:
            rec["created_by"] = "agent"
        return {"created_by": rec["created_by"]}

    facts = _mutate(skill_name, _apply)
    if isinstance(facts, dict):
        _emit_skill_lifecycle(
            skill_name,
            "created",
            record=facts,
            task_id=task_id,
            session_id=session_id,
        )


def record_installed(skill_name: str) -> None:
    """Record a successful Skills Hub install without exporting its name."""
    skill_name = _canonicalize_skill_name(skill_name)

    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        rec["created_by"] = "installed"
        rec["state"] = STATE_ACTIVE
        rec["archived_at"] = None
        return {"created_by": rec["created_by"]}

    facts = _mutate(skill_name, _apply)
    if isinstance(facts, dict):
        _emit_skill_lifecycle(skill_name, "installed", record=facts)


def mark_agent_created(skill_name: str) -> None:
    """Opt a skill created by skill_manage into curator management.

    Viewing or invoking a manually authored skill may still create telemetry,
    but only this explicit marker makes it eligible for automatic curation.
    """
    skill_name = _canonicalize_skill_name(skill_name)

    def _apply(rec: Dict[str, Any]) -> None:
        rec["created_by"] = "agent"
    _mutate(skill_name, _apply, require_curation_eligible=True)


def set_state(skill_name: str, state: str) -> None:
    """Set lifecycle state. No-op if *state* is invalid or the skill isn't
    curator-manageable (hub skills, or built-ins with pruning disabled)."""
    skill_name = _canonicalize_skill_name(skill_name)
    if state not in _VALID_STATES:
        logger.debug("set_state: invalid state %r for %s", state, skill_name)
        return
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        previous_state = rec.get("state")
        if previous_state == state:
            return {"changed": False, "created_by": rec.get("created_by")}
        rec["state"] = state
        if state == STATE_ARCHIVED:
            rec["archived_at"] = _now_iso()
        elif state == STATE_ACTIVE:
            rec["archived_at"] = None
        return {
            "changed": True,
            "created_by": rec.get("created_by"),
            "previous_state": previous_state,
        }

    facts = _mutate(skill_name, _apply, require_curation_eligible=True)
    if not isinstance(facts, dict) or not facts.get("changed"):
        return
    action = {
        STATE_ARCHIVED: "archived",
        STATE_STALE: "stale",
    }.get(state)
    if state == STATE_ACTIVE and facts.get("previous_state") == STATE_ARCHIVED:
        action = "restored"
    if action is not None:
        _emit_skill_lifecycle(skill_name, action, record=facts)


def set_pinned(skill_name: str, pinned: bool) -> bool:
    """Set/clear the pin flag. Returns False when the write did not land
    (skill not curation-eligible), True on success — so callers can report
    failure instead of a false success (issue #92993)."""
    skill_name = _canonicalize_skill_name(skill_name)

    def _apply(rec: Dict[str, Any]) -> Any:
        rec["pinned"] = bool(pinned)
        return True  # non-None sentinel: _mutate propagates the mutator result
    return bool(_mutate(skill_name, _apply, require_curation_eligible=True))


def set_sync(skill_name: str, sync: bool) -> None:
    """Set the sync opt-in flag on a skill's usage record.

    Sync is OPT-IN: nothing propagates to the sync plane unless the user marks
    a skill with ``sync: true`` here. Sits alongside ``pinned``/``created_by``
    on the ``.usage.json`` sidecar and is read by
    ``tools.skills_sync_client.list_synced_skill_names``. Gated on curation
    eligibility so bundled/hub/external skills (which never sync) can't be
    marked. Provisional per the M1-D default.
    """
    skill_name = _canonicalize_skill_name(skill_name)

    def _apply(rec: Dict[str, Any]) -> None:
        rec["sync"] = bool(sync)
    _mutate(skill_name, _apply, require_curation_eligible=True)


def is_sync_enabled(skill_name: str) -> bool:
    """Whether a skill is opted into sync (``sync: true`` in its record)."""
    return get_record(skill_name).get("sync") is True


def forget(skill_name: str) -> None:
    """Drop a skill's usage entry entirely. Called when the skill is deleted."""
    skill_name = _canonicalize_skill_name(skill_name)
    if not skill_name:
        return
    try:
        with _usage_file_lock():
            data = load_usage()
            if skill_name in data:
                del data[skill_name]
                save_usage(data)
    except Exception as e:
        logger.debug("skill_usage.forget(%s) failed: %s", skill_name, e, exc_info=True)


# ---------------------------------------------------------------------------
# Archive / restore
# ---------------------------------------------------------------------------

def archive_skill(skill_name: str) -> Tuple[bool, str]:
    """Move a curator-eligible skill directory to ~/.hermes/skills/.archive/.

    Returns (ok, message). Never archives hub-installed skills. Bundled
    built-ins are only archivable when ``curator.prune_builtins`` is enabled;
    when one is archived, its name is added to the suppression list so the
    update-time re-seeder leaves it archived instead of restoring it.
    """
    skill_name = _canonicalize_skill_name(skill_name)
    local_skill_dir = _find_skill_dir(skill_name)
    if local_skill_dir is None and _find_external_skill_dir(skill_name) is not None:
        return False, _external_read_only_message(skill_name)

    if not is_curation_eligible(skill_name, local_skill_dir):
        if is_protected_builtin(skill_name):
            return False, (
                f"skill '{skill_name}' is a protected built-in; it backs "
                "load-bearing UX and is never archived or consolidated"
            )
        if is_hub_installed(skill_name):
            return False, f"skill '{skill_name}' is hub-installed; never archive"
        return False, (
            f"skill '{skill_name}' is a bundled built-in; enable "
            "curator.prune_builtins to allow pruning it"
        )

    skill_dir = local_skill_dir
    if skill_dir is None:
        return False, f"skill '{skill_name}' not found"
    if is_external_skill_path(skill_dir):
        return False, _external_read_only_message(skill_name)

    archive_root = _archive_dir()
    try:
        archive_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"failed to create archive dir: {e}"

    # Flatten any category nesting into a single ".archive/<skill>/" so restores
    # are simple. If a collision exists, append a timestamp.
    dest = archive_root / skill_dir.name
    if dest.exists():
        dest = archive_root / f"{skill_dir.name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    # Audit ledger pre-capture (best-effort; never blocks the archive).
    # complete_package: consolidation may have re-homed support files out of
    # the tree first, so a disk-only capture can come back hollow; the fill
    # from the newest curator backup keeps rollback restorable (#96962).
    _ledger_before = None
    try:
        from tools import skill_ledger as _ledger
        _ledger_before = _ledger.capture_before(
            skill_dir, complete_package=True, skill=skill_name
        )
    except Exception:
        _ledger = None  # type: ignore[assignment]

    try:
        skill_dir.rename(dest)
    except OSError:
        # Cross-device — fall back to shutil.move
        import shutil
        try:
            shutil.move(str(skill_dir), str(dest))
        except Exception as e2:
            return False, f"failed to archive: {e2}"

    # Pruning a built-in only sticks if the re-seeder is told to leave it alone.
    if is_bundled(skill_name):
        add_suppressed_name(skill_name)

    set_state(skill_name, STATE_ARCHIVED)
    try:
        if _ledger is not None:
            _ledger.record_mutation(
                "archive",
                skill_name,
                before=_ledger_before if _ledger_before is not None else [],
                after_root=dest,
            )
    except Exception:
        pass
    return True, f"archived to {dest}"


def restore_skill(skill_name: str) -> Tuple[bool, str]:
    """Move an archived skill back to ~/.hermes/skills/. Restores to the flat
    top-level layout; original category nesting is NOT reconstructed.

    Refuses to restore under a name that now collides with a hub-installed
    skill — that would shadow the upstream version. Also refuses to restore
    over a bundled built-in UNLESS ``curator.prune_builtins`` is enabled (in
    which case built-ins are curator-managed and restoring is the documented
    way to lift a prune). Restoring clears any suppression entry so future
    updates may re-seed the built-in again.
    """
    skill_name = _canonicalize_skill_name(skill_name)
    # Hub skills always have an external upstream owner — never shadow them.
    if is_hub_installed(skill_name):
        return False, (
            f"skill '{skill_name}' is now hub-installed; "
            "restore would shadow the upstream version"
        )
    # A bundled built-in is upstream-owned UNLESS prune_builtins is on. With the
    # flag off, restoring over it would shadow the bundled version.
    if is_bundled(skill_name) and not _prune_builtins_enabled():
        return False, (
            f"skill '{skill_name}' is now bundled; "
            "restore would shadow the upstream version"
        )
    archive_root = _archive_dir()
    if not archive_root.exists():
        return False, "no archive directory"

    # Try exact name match first, then the timestamped-duplicate fallback.
    # Recursive walk handles nested archive layouts (e.g. .archive/<category>/<skill>/)
    # left behind by older archive paths or external imports.
    candidates = [p for p in archive_root.rglob("*") if p.is_dir() and p.name == skill_name]
    if not candidates:
        # A name collision makes archive_skill() disambiguate by appending its
        # UTC timestamp ("<skill>-YYYYMMDDHHMMSS", a 14-digit suffix), so only
        # that exact shape is another copy of THIS skill. A bare
        # startswith(f"{skill_name}-") also swallows unrelated sibling skills —
        # restoring "git" would otherwise pull an archived "git-helpers" out of
        # the archive and rename it to "git", destroying the sibling's only
        # copy. Require the suffix to be the timestamp archive_skill writes.
        prefix = f"{skill_name}-"
        candidates = sorted(
            [
                p for p in archive_root.rglob("*")
                if p.is_dir()
                and p.name.startswith(prefix)
                and len(p.name) - len(prefix) == 14
                and p.name[len(prefix):].isdigit()
            ],
            reverse=True,
        )
    if not candidates:
        return False, f"skill '{skill_name}' not found in archive"

    src = candidates[0]
    dest = _skills_dir() / skill_name
    if dest.exists():
        return False, f"destination already exists: {dest}"

    # Audit ledger pre-capture (best-effort; never blocks the restore).
    _ledger_before = None
    try:
        from tools import skill_ledger as _ledger
        _ledger_before = _ledger.capture_before(src)
    except Exception:
        _ledger = None  # type: ignore[assignment]

    try:
        src.rename(dest)
    except OSError:
        import shutil
        try:
            shutil.move(str(src), str(dest))
        except Exception as e:
            return False, f"failed to restore: {e}"

    # Restoring a pruned built-in lifts its suppression so updates can manage it.
    remove_suppressed_name(skill_name)

    set_state(skill_name, STATE_ACTIVE)
    try:
        if _ledger is not None:
            _ledger.record_mutation(
                "restore",
                skill_name,
                before=_ledger_before if _ledger_before is not None else [],
                after_root=dest,
            )
    except Exception:
        pass
    return True, f"restored to {dest}"


def _find_skill_dir(skill_name: str) -> Optional[Path]:
    """Locate the directory for a skill by its frontmatter `name:` field.

    Handles both flat (~/.hermes/skills/<skill>/SKILL.md) and category-nested
    (~/.hermes/skills/<category>/<skill>/SKILL.md) layouts. Uses the gated
    index iterator so M2 org mirrors resolve ONLY for the active org
    (stale ``_org/<other>/`` trees never match).
    """
    base = _skills_dir()
    if not base.exists():
        return None
    from agent.skill_utils import iter_skill_index_files

    for skill_md in iter_skill_index_files(base, "SKILL.md"):
        if is_external_skill_path(skill_md):
            continue
        if _read_skill_name(skill_md, fallback=skill_md.parent.name) == skill_name:
            return skill_md.parent
    return None


def _find_external_skill_dir(skill_name: str) -> Optional[Path]:
    """Locate a skill under configured external dirs by frontmatter name."""
    from agent.skill_utils import get_all_skills_dirs

    for base in get_all_skills_dirs()[1:]:
        if not base.exists():
            continue
        for skill_md in base.rglob("SKILL.md"):
            if is_excluded_skill_path(skill_md):
                continue
            if _read_skill_name(skill_md, fallback=skill_md.parent.name) == skill_name:
                return skill_md.parent
    return None


# ---------------------------------------------------------------------------
# Reporting — for the curator CLI / slash command
# ---------------------------------------------------------------------------


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


def curated_report() -> List[Dict[str, Any]]:
    """Return a list of {name, provenance, state, pinned, last_activity_at, ...}
    records for every curator-managed skill. Missing usage records are
    backfilled with defaults so callers can always index fields.

    ``provenance`` is 'agent', 'bundled', or 'hub' (see :func:`provenance`).
    Bundled skills are only included when ``curator.prune_builtins`` is enabled.
    Hub-installed skills are never included.

    Each row carries ``_persisted``: True when a real record exists in
    ``.usage.json``, False when the row is a fresh backfill (e.g. a built-in
    seen for the first time). The curator uses this to seed the inactivity
    clock instead of treating an unrecorded skill as ancient.
    """
    data = load_usage()
    rows: List[Dict[str, Any]] = []
    names = set(list_agent_created_skill_names())
    # Issue #92993: a successfully pinned skill must be visible in the report
    # even when it lacks the created_by marker (eligible-but-unmanaged), or
    # its pin silently vanishes from `curator status`. The local-dir guard
    # keeps stale records for deleted skill dirs from rendering as ghost rows;
    # `curator unpin` is the cleanup path for those.
    for name, rec in data.items():
        if (
            isinstance(rec, dict)
            and rec.get("pinned")
            and is_curation_eligible(name)
            and _find_skill_dir(name) is not None
        ):
            names.add(name)
    for name in sorted(names):
        raw = data.get(name)
        persisted = isinstance(raw, dict)
        rec: Dict[str, Any] = raw if isinstance(raw, dict) else _empty_record()
        base = _empty_record()
        for k, v in base.items():
            rec.setdefault(k, v)
        row = {"name": name, **rec, "_persisted": persisted}
        row["last_activity_at"] = latest_activity_at(row)
        row["activity_count"] = activity_count(row)
        row["provenance"] = provenance(name)
        rows.append(row)
    return rows


def agent_created_report() -> List[Dict[str, Any]]:
    """DEPRECATED — use :func:`curated_report` instead.

    Used to return everything :func:`curated_report` returns (including bundled
    skills when ``curator.prune_builtins`` is enabled), which made the
    "agent-created" name misleading. Kept as a compatibility alias for
    external callers; new code should call ``curated_report()``.
    """
    return curated_report()


def provenance(skill_name: str) -> str:
    """Classify a skill's origin: 'hub', 'bundled', or 'agent'.

    'agent' covers both agent-authored and local manually-authored skills —
    anything not seeded from the bundled repo or installed via the hub.
    """
    skill_name = _canonicalize_skill_name(skill_name)
    if is_hub_installed(skill_name):
        return "hub"
    if is_bundled(skill_name):
        return "bundled"
    return "agent"


def usage_report() -> List[Dict[str, Any]]:
    """Return usage telemetry for EVERY skill on disk, with provenance.

    Unlike ``curated_report()`` (which is scoped to curator-managed
    candidates), this surfaces all skills — bundled built-ins and
    hub-installed included — so callers can answer "how often is this skill
    used" independent of whether it's ever curated. Rows carry a
    ``provenance`` field ('agent' | 'bundled' | 'hub') and ``_persisted``
    (whether a real ``.usage.json`` record backs the row).
    """
    base = _skills_dir()
    if not base.exists():
        return []
    data = load_usage()
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for skill_md in base.rglob("SKILL.md"):
        if is_excluded_skill_path(skill_md):
            continue
        name = _read_skill_name(skill_md, fallback=skill_md.parent.name)
        if name in seen:
            continue
        seen.add(name)
        raw = data.get(name)
        persisted = isinstance(raw, dict)
        rec: Dict[str, Any] = raw if isinstance(raw, dict) else _empty_record()
        base_rec = _empty_record()
        for k, v in base_rec.items():
            rec.setdefault(k, v)
        row = {
            "name": name,
            **rec,
            "provenance": provenance(name),
            "_persisted": persisted,
        }
        row["last_activity_at"] = latest_activity_at(row)
        row["activity_count"] = activity_count(row)
        rows.append(row)
    return sorted(rows, key=lambda r: r["name"])
