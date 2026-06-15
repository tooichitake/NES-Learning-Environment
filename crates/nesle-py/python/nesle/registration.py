"""Gymnasium registration + env-id parsing for NESLE environments.

An env-id is layered ``NESLE/{game-id}[-{game-mode}]-{game-level}-v{env-version}``;
``parse_env_id`` splits it back into those layers. See ``docs/env_ids.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from nesle.roms import game_metadata, start_state_metadata

from . import _nesle

_EPISODE_FRAME_CAP = 108_000

_RAW_PARAMS: dict[str, Any] = {
    "obs_type": "rgb",
    "frame_skip": 1,
    "screen_size": None,
    "max_pool": False,
    "remove_sprite_limit": False,
    "noop_max": 0,
    "terminal_on_life_loss": False,
    "repeat_action_probability": 0.0,
    "scale_obs": False,
    "clip_reward": False,
}

_STANDARD_PARAMS: dict[str, Any] = {
    "preprocessed": True,
    "frame_skip": 4,
    "screen_size": 112,  # NES native 256x240 is ~1.83x Atari's pixels; 112 = 84*4/3 matches ALE detail-retention
    "max_pool": True,
    "remove_sprite_limit": False,
    "noop_max": 30,
    "terminal_on_life_loss": True,
    "repeat_action_probability": 0.0,
    "scale_obs": False,
    "clip_reward": False,
}

_NOFLICKER_PARAMS: dict[str, Any] = {
    **_STANDARD_PARAMS,
    "max_pool": False,
    "remove_sprite_limit": True,
}

# v3 = v2 + sticky actions 0.25 (Machado et al. 2018); params are ALE-correct at raw-frame granularity.
_STICKY_PARAMS: dict[str, Any] = {
    **_NOFLICKER_PARAMS,
    "repeat_action_probability": 0.25,
}

_PROFILE_PARAMS: dict[str, dict[str, Any]] = {
    "-v1": _STANDARD_PARAMS,
    "-v2": _NOFLICKER_PARAMS,
    "-v3": _STICKY_PARAMS,
    "NoFrameskip-v1": {**_STANDARD_PARAMS, "frame_skip": 1},
    "NoFrameskip-v2": {**_NOFLICKER_PARAMS, "frame_skip": 1},
    "NoFrameskip-v3": {**_STICKY_PARAMS, "frame_skip": 1},
}


def vector_profile_params(env_id: str) -> dict[str, Any]:
    """Preprocessing params for a *vectorized* env-id, keyed by its ``-vN`` suffix.

    The single source of truth (``_PROFILE_PARAMS``), shared by single-agent
    ``gym.make_vec`` (via registration) and multi-player vectorization (via
    ``vector_env.make_multiplayer_vector_env``). ``-v0`` is the RAW profile and is not
    a vectorized/preprocessed profile, so it raises here. ``screen_size`` defaults
    to 84 when the profile leaves it unset.
    """
    # Longest suffix first so "NoFrameskip-v1" wins over the "-v1" substring.
    for suffix in sorted(_PROFILE_PARAMS, key=len, reverse=True):
        if env_id.endswith(suffix):
            params = dict(_PROFILE_PARAMS[suffix])
            if params.get("screen_size") is None:
                params["screen_size"] = 84
            return params
    raise ValueError(
        f"{env_id!r} has no preprocessed (-v1/-v2/-v3) profile suffix; "
        "vectorized envs require a preprocessed profile"
    )


@dataclass(frozen=True)
class ParsedEnvId:
    """The layers of a NESLE env-id.

    env-id = ``NESLE/{game_name}[-{game_mode}]-{game_level}-v{env_version}``. The
    ``rom-id`` (on-disk ROM filename) is a sixth layer resolved by sha1 from
    ``game_id``, not encoded in the string. See ``docs/env_ids.md``.
    """

    env_id: str
    game_id: str  # Rust GameSpec.id (e.g. "super_c_2p"); resolves the ROM + spec
    game_name: str  # game-id layer: PascalCase name token (e.g. "SuperC")
    game_mode: str | None  # game-mode layer (e.g. "2P"/"1P"/"VS"/"Normal"); None if modeless
    game_level: str  # game-level layer: start-state suffix (e.g. "2", "1-1")
    env_version: int  # env-version layer (0..3)
    no_frameskip: bool  # the NoFrameskip env-version variant (frame_skip=1)
    state_id: str  # formal start-state id backing game_level


def _iter_env_ids():
    """Yield every registered ``ParsedEnvId`` -- the single construction point for
    env-id assembly (``env_ids_for_game`` / ``parse_env_id``).

    Single-agent families (a 1-player spec, plus the 1P variant of a 2P spec) carry
    v0 raw + v1/v2/v3 + NoFrameskip-v{1,2,3}; multi-player specs carry v0..v3 only
    (v0 raw PettingZoo + v1/v2/v3 self-play). game-id + game-mode come from the stem
    (game names are hyphen-free, so the mode is whatever follows the first hyphen);
    game-level is the start-state suffix, which may itself contain hyphens (SMB 1-1).
    """
    states = start_state_metadata()
    for game_id, meta in game_metadata().items():
        families: list[tuple[str, bool]] = []  # (stem, single_agent_family)
        gym_id = str(meta["gym_id"])
        if int(meta["players"]) == 1:
            families.append((gym_id, True))
        else:
            if not gym_id.endswith("-v0"):
                raise ValueError(f"multiplayer gym_id must end with -v0: {gym_id!r}")
            families.append((gym_id[:-3], False))
        for stem, single_agent in families:
            game_name, _, mode_token = stem[len("NESLE/") :].partition("-")
            game_mode = mode_token or None
            for state_id, level in states.get(game_id, []):
                base = f"{stem}-{level}"
                if single_agent:
                    variants = (
                        (f"{base}-v0", 0, False),
                        (f"{base}-v1", 1, False),
                        (f"{base}-v2", 2, False),
                        (f"{base}-v3", 3, False),
                        (f"{base}NoFrameskip-v1", 1, True),
                        (f"{base}NoFrameskip-v2", 2, True),
                        (f"{base}NoFrameskip-v3", 3, True),
                    )
                else:
                    variants = tuple((f"{base}-v{n}", n, False) for n in range(4))
                for env_id, env_version, no_frameskip in variants:
                    yield ParsedEnvId(
                        env_id=env_id,
                        game_id=game_id,
                        game_name=game_name,
                        game_mode=game_mode,
                        game_level=level,
                        env_version=env_version,
                        no_frameskip=no_frameskip,
                        state_id=state_id,
                    )


def env_ids_for_game(game_id: str) -> list[str]:
    return [p.env_id for p in _iter_env_ids() if p.game_id == game_id]


def parse_env_id(env_id: str) -> ParsedEnvId:
    """Split an env-id into its layers (``ParsedEnvId``; see ``docs/env_ids.md``).

    Raises ``ValueError`` for an unregistered id.
    """
    for parsed in _iter_env_ids():
        if parsed.env_id == env_id:
            return parsed
    raise ValueError(f"unknown NESLE env id: {env_id!r}")


def resolve_env_id(env_id: str) -> tuple[str, str]:
    parsed = parse_env_id(env_id)
    return parsed.game_id, parsed.state_id


def rom_key_for_env_id(env_id: str) -> str:
    return resolve_env_id(env_id)[0]


def _register(env_id: str, *, entry_point: str, kwargs: dict[str, Any]) -> None:
    if env_id in gym.envs.registry:
        return
    gym.register(
        id=env_id,
        entry_point=entry_point,
        vector_entry_point="nesle.vector_env:NESSinglePlayerVectorEnv",
        kwargs=kwargs,
    )
    # NESLE uses -v0/-v1/-v2 as fixed observation profiles, not a deprecation chain.
    spec = gym.envs.registry[env_id]
    spec.name = env_id.rsplit("/", 1)[-1]
    spec.version = None


def _preprocessed_kwargs(game_id: str, level_state: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "_level_state": level_state,
        "preprocessed": params["preprocessed"],
        "screen_size": params["screen_size"],
        "frame_skip": params["frame_skip"],
        "noop_max": params["noop_max"],
        "terminal_on_life_loss": params["terminal_on_life_loss"],
        "scale_obs": params["scale_obs"],
        "max_pool": params["max_pool"],
        "remove_sprite_limit": params["remove_sprite_limit"],
        "clip_reward": params["clip_reward"],
        "repeat_action_probability": params["repeat_action_probability"],
        "max_num_frames_per_episode": _EPISODE_FRAME_CAP,
    }


def _register_single_agent_family(game_id: str, base: str, states: dict) -> None:
    """Register one single-agent gym family for the ``base`` gym_id stem:
    ``{base}-{level}-v0`` (raw, NESSinglePlayerEnv) + ``-v1/-v2/-v3`` + NoFrameskip
    (preprocessed). NESSinglePlayerEnv / NESSinglePlayerVectorEnv force one controller
    port, so a multi-capable spec registered here runs as its 1P mode (port 0)."""
    raw = _RAW_PARAMS
    for level_state, env_suffix in states.get(game_id, []):
        stem = f"{base}-{env_suffix}"
        _register(
            f"{stem}-v0",
            entry_point="nesle.env:NESSinglePlayerEnv",
            kwargs={
                "game_id": game_id,
                "_level_state": level_state,
                "obs_type": raw["obs_type"],
                "frame_skip": raw["frame_skip"],
                "repeat_action_probability": raw["repeat_action_probability"],
                "max_num_frames_per_episode": _EPISODE_FRAME_CAP,
            },
        )
        for suffix, params in _PROFILE_PARAMS.items():
            _register(
                f"{stem}{suffix}",
                entry_point="nesle.preprocessing:_make_preprocessed_env",
                kwargs=_preprocessed_kwargs(game_id, level_state, dict(params)),
            )


def register_envs() -> None:
    states = start_state_metadata()
    for game_id, meta in game_metadata().items():
        if meta.get("players") == 1:
            _register_single_agent_family(game_id, meta["gym_id"], states)
