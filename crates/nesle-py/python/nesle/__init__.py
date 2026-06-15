"""NESLE Python facade for the Rust-native package line."""

from nesle.registration import ParsedEnvId, parse_env_id, register_envs
from nesle.roms import get_all_game_ids, get_rom_path, import_roms, resolve_rom, roms_dir

register_envs()

__all__ = [
    "ParsedEnvId",
    "get_all_game_ids",
    "get_rom_path",
    "import_roms",
    "parse_env_id",
    "register_envs",
    "resolve_rom",
    "roms_dir",
]
