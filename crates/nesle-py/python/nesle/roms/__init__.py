"""ROM access for the Rust-native NESLE package.

Holds ROM-file resolution + the packaged ``*.nes`` ROMs (this directory). env-id
construction/resolution lives in ``nesle.registration``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from nesle import _nesle

ROM_ENV_VAR = "NESLE_ROMS_DIR"
REGISTRY_FILE = "registry.json"


def game_metadata() -> dict[str, dict[str, object]]:
    return {
        game_id: {
            "gym_id": gym_id,
            "display_name": display_name,
            "sha1": sha1,
            "players": players,
        }
        for game_id, gym_id, display_name, sha1, players in _nesle.game_metadata()
    }


def get_all_game_ids() -> list[str]:
    """All registered game-ids. NESLE keys
    games by the Rust ``GameSpec.id`` (e.g. ``super_c_2p``), not the ROM filename."""
    return sorted(game_metadata())


def start_state_metadata() -> dict[str, list[tuple[str, str]]]:
    states: dict[str, list[tuple[str, str]]] = {}
    for game_id, state_id, env_suffix in _nesle.start_state_metadata():
        states.setdefault(game_id, []).append((state_id, env_suffix))
    return states


def roms_dir() -> Path:
    return Path(os.environ.get(ROM_ENV_VAR, Path.home() / ".local" / "share" / "nesle" / "roms"))


def _packaged_roms_dir() -> Path:
    """Directory of ROMs shipped inside the wheel (this package dir)."""
    return Path(__file__).resolve().parent


def get_rom_path(game_id: str) -> Path | None:
    """Path to the packaged ROM for ``game_id``.

    Returns ``None`` when no packaged ROM matches. The direct ``{game_id}.nes``
    name is tried first; failing that (e.g. a multiplayer id that shares a ROM
    with its 1P variant) the packaged ``*.nes`` are scanned for a sha1 match
    against the GameSpec table -- the single source of truth.
    """
    base = _packaged_roms_dir()
    if not base.is_dir():
        return None
    expected = str(game_metadata().get(game_id, {}).get("sha1", ""))
    direct = base / f"{game_id}.nes"
    if direct.exists() and (not expected or sha1_file(direct) == expected):
        return direct
    if expected:
        for candidate in sorted(base.glob("*.nes")):
            if sha1_file(candidate) == expected:
                return candidate
    return None


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(directory: Path | None = None) -> dict[str, dict[str, str]]:
    base = directory or roms_dir()
    registry = base / REGISTRY_FILE
    if not registry.exists():
        return {}
    return json.loads(registry.read_text(encoding="utf-8"))


def save_registry(registry: dict[str, dict[str, str]], directory: Path | None = None) -> None:
    base = directory or roms_dir()
    base.mkdir(parents=True, exist_ok=True)
    (base / REGISTRY_FILE).write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def import_roms(
    source: str | Path, directory: str | Path | None = None
) -> dict[str, dict[str, str]]:
    src = Path(source)
    base = Path(directory) if directory is not None else roms_dir()
    base.mkdir(parents=True, exist_ok=True)
    registry = load_registry(base)

    candidates = [src] if src.is_file() else sorted(src.rglob("*.nes"))
    for path in candidates:
        game_id = identify_game(path)
        if game_id is None:
            continue
        expected_sha1 = game_metadata().get(game_id, {}).get("sha1", "")
        actual_sha1 = sha1_file(path)
        if expected_sha1 and actual_sha1 != expected_sha1:
            continue
        target = base / f"{game_id}.nes"
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        registry[game_id] = {
            "path": str(target),
            "sha1": sha1_file(target),
            "display_name": game_metadata().get(game_id, {}).get("display_name", game_id),
        }

    save_registry(registry, base)
    return registry


def resolve_rom(game_id: str, rom_path: str | Path | None = None) -> Path:
    # Resolution order: explicit path > packaged ROM > NESLE_ROMS_DIR registry.
    if rom_path is not None:
        path = Path(rom_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    packaged = get_rom_path(game_id)
    if packaged is not None:
        return packaged

    registry = load_registry()
    entry = registry.get(game_id)
    if entry is None:
        raise FileNotFoundError(
            f"ROM for {game_id!r} is not packaged or registered. "
            "Pass rom_path=... or run nesle.import_roms(...)."
        )
    path = Path(entry["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    actual_sha1 = sha1_file(path)
    expected_sha1 = entry.get("sha1")
    if expected_sha1 and actual_sha1 != expected_sha1:
        raise ValueError(f"ROM hash mismatch for {game_id!r}: {actual_sha1} != {expected_sha1}")
    return path


def identify_game(path: Path) -> str | None:
    """Identify a ROM by exact SHA1 against the registered GameSpecs.

    Works for every game with a registered ``sha1`` (no per-title name
    heuristics): the GameSpec table is the single source of truth.
    """
    file_sha1 = sha1_file(path)
    for game_id, meta in game_metadata().items():
        if meta.get("sha1") and meta["sha1"] == file_sha1:
            return game_id
    return None
