"""Gymnasium facade for the Rust-native NESLE environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle

from . import _nesle
from nesle.registration import resolve_env_id
from nesle.roms import resolve_rom

# PettingZoo is OPTIONAL (only NESMultiPlayerEnv needs it); single-agent users never pull it in.
try:
    from pettingzoo import ParallelEnv

    _HAS_PETTINGZOO = True
except ImportError:  # pragma: no cover - exercised only without pettingzoo installed
    ParallelEnv = object
    _HAS_PETTINGZOO = False


class NESSinglePlayerEnv(gym.Env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        *,
        game_id: str = "super_mario_bros",
        rom_path: str | Path | None = None,
        obs_type: str = "ram",
        render_mode: str | None = None,
        frameskip: int = 1,
        repeat_action_probability: float = 0.0,
        full_action_space: bool = False,
        remove_sprite_limit: bool = False,
        max_num_frames_per_episode: int | None = None,
        _level_state: str | None = None,
        mode: int = 0,
        difficulty: int = 0,
    ) -> None:
        EzPickle.__init__(
            self,
            game_id=game_id,
            rom_path=rom_path,
            obs_type=obs_type,
            render_mode=render_mode,
            frameskip=frameskip,
            repeat_action_probability=repeat_action_probability,
            full_action_space=full_action_space,
            remove_sprite_limit=remove_sprite_limit,
            max_num_frames_per_episode=max_num_frames_per_episode,
            _level_state=_level_state,
            mode=mode,
            difficulty=difficulty,
        )
        if obs_type not in ("ram", "rgb", "grayscale"):
            raise ValueError("obs_type must be one of: ram, rgb, grayscale")
        if render_mode not in (None, "rgb_array", "human"):
            raise ValueError(f"unsupported render_mode: {render_mode}")
        if mode != 0:
            raise ValueError(f"mode {mode} unsupported (NES has no ALE-style game modes; only 0)")
        if difficulty != 0:
            raise ValueError(f"difficulty {difficulty} unsupported (NES has no ALE-style difficulties; only 0)")
        self.game_id = game_id
        self.obs_type = obs_type
        self.frameskip = frameskip
        self.render_mode = render_mode
        self.full_action_space = full_action_space
        self._env = _nesle.NesEnv(game_id)
        self._env.set_players(1)  # single-agent facade -> one port (1P for a 2P-capable spec)
        if _level_state is not None:
            self._env.set_start_state(str(_level_state))
        self._env.load_rom_bytes(resolve_rom(game_id, rom_path).read_bytes())
        self._env.set_max_episode_frames(max_num_frames_per_episode)
        self._env.set_action_repeat(frameskip, repeat_action_probability)
        self._env.set_remove_sprite_limit(remove_sprite_limit)
        self._actions = (
            self._env.full_action_set() if full_action_space else self._env.minimal_action_set()
        )
        self.action_space = spaces.Discrete(len(self._actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self._obs_shape(), dtype=np.uint8)
        self._human = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.seed(int(seed))
        if options:
            unknown = ", ".join(sorted(options))
            raise ValueError(f"unsupported reset options: {unknown}")
        frame_number, episode_frame_number, lives = self._env.reset()
        if self.render_mode == "human":
            self._render_human()
        info = self._info(frame_number, episode_frame_number, lives)
        if seed is not None:
            info["seeds"] = (int(seed),)
        return self._obs(), info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action index: {action}")
        _, mask = self._actions[int(action)]
        if self.render_mode == "human":
            window = self._ensure_human_window()
            reward, terminated, truncated, frame_number, episode_frame_number, lives = (
                self._env.step_human(mask, window)
            )
        else:
            reward, terminated, truncated, frame_number, episode_frame_number, lives = self._env.step(mask)
        info = self._info(frame_number, episode_frame_number, lives)
        return (
            self._obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        if self.render_mode == "human":
            self._render_human()
            return None
        if self.render_mode == "rgb_array":
            return self._screen_rgb()
        return None

    def close(self) -> None:
        if self._human is not None:
            self._human.close()
            self._human = None
        return None

    def _ensure_human_window(self):
        """Create the optional SDL2 window used by render_mode='human'."""
        if self._human is None:
            window_cls = getattr(_nesle, "HumanWindow", None)
            if window_cls is None or not hasattr(self._env, "step_human"):
                raise RuntimeError(
                    "render_mode='human' needs the in-process SDL2 window, built behind the "
                    "extension's optional `viewer` feature (the ALE SDL_SUPPORT analogue). "
                    "Rebuild with `maturin develop --features viewer`, or use the standalone "
                    "native viewer: `python -m nesle.play <game>`."
                )
            self._human = window_cls(f"NESLE - {self.game_id}", 3)
        return self._human

    def _render_human(self) -> None:
        window = self._ensure_human_window()
        if window.present(self._env.screen_rgb()):  # user closed the window
            window.close()
            self._human = None

    def get_ram(self) -> np.ndarray:
        return np.frombuffer(self._env.ram(), dtype=np.uint8).copy()

    def set_ram(self, ram) -> None:
        arr = np.asarray(ram, dtype=np.uint8)
        if arr.shape != (2048,):
            raise ValueError(f"RAM must have shape (2048,), got {arr.shape}")
        self._env.set_ram(arr.tobytes())

    def clone_state(self):
        return self._env.clone_state()

    def restore_state(self, state) -> None:
        self._env.restore_state(state)

    def save_state_blob(self) -> bytes:
        return bytes(self._env.save_state_blob())

    def restore_state_blob(self, blob: bytes | bytearray | memoryview) -> None:
        self._env.restore_state_blob(bytes(blob))

    def get_action_meanings(self) -> list[str]:
        """Return names for the active action set."""
        return [name for name, _ in self._actions]

    def get_minimal_action_set(self) -> list[int]:
        """Return indices for this game's minimal action set."""
        return list(range(len(self._env.minimal_action_set())))

    def get_legal_action_set(self) -> list[int]:
        """Return indices for the unified 36-action NES set."""
        return list(range(len(self._env.full_action_set())))

    def get_keys_to_action(self) -> dict[tuple[str, ...], int]:
        """Keyboard-combo -> action-index map for ``gymnasium.utils.play.play``
        (the ALE ``get_keys_to_action`` analogue). NESSinglePlayerEnv only exposes the env +
        this map; gymnasium does the actual keyboard playing::

            from gymnasium.utils.play import play
            play(gymnasium.make("NESLE/SuperMarioBros-1-1-v0", render_mode="rgb_array"))

        Keys: W/A/S/D = D-pad, k = A, j = B, n = Select, m = Start; no keys =
        NOOP. (A standalone native keyboard player is also ``python -m nesle.play``.)
        """
        button_key = {0x10: "w", 0x20: "s", 0x40: "a", 0x80: "d",
                      0x01: "k", 0x02: "j", 0x04: "n", 0x08: "m"}
        mapping: dict[tuple[str, ...], int] = {}
        for index, (_, mask) in enumerate(self._actions):
            if mask == 0:
                continue  # NOOP is the no-keys default
            mapping[tuple(sorted(ch for bit, ch in button_key.items() if mask & bit))] = index
        return mapping

    def _obs(self) -> np.ndarray:
        if self.obs_type == "ram":
            return np.frombuffer(self._env.ram(), dtype=np.uint8).copy()
        if self.obs_type == "rgb":
            return self._screen_rgb()
        return self._screen_gray()

    def _obs_shape(self) -> tuple[int, ...]:
        if self.obs_type == "ram":
            return (2048,)
        if self.obs_type == "rgb":
            return (240, 256, 3)
        return (240, 256)

    def _screen_rgb(self) -> np.ndarray:
        return np.frombuffer(self._env.screen_rgb(), dtype=np.uint8).reshape((240, 256, 3)).copy()

    def _screen_gray(self) -> np.ndarray:
        return np.frombuffer(self._env.screen_gray(), dtype=np.uint8).reshape((240, 256)).copy()

    def get_screen_grayscale(self) -> np.ndarray:
        """Return the native (240, 256) grayscale frame as a read-only view."""
        return np.frombuffer(self._env.screen_gray(), dtype=np.uint8).reshape((240, 256))

    def set_render_enabled(self, enabled: bool) -> None:
        self._env.set_render_enabled(enabled)

    @staticmethod
    def _info(frame_number: int, episode_frame_number: int, lives: int) -> dict[str, int]:
        return {
            "frame_number": int(frame_number),
            "episode_frame_number": int(episode_frame_number),
            "lives": int(lives),
        }


# Multi-agent facade: PettingZoo ParallelEnv sibling of NESSinglePlayerEnv (separate class: gym.Env vs ParallelEnv are incompatible).

_MULTI_OBS_SHAPE = {"ram": (2048,), "rgb": (240, 256, 3), "grayscale": (240, 256)}


class NESMultiPlayerEnv(ParallelEnv):
    """PettingZoo ParallelEnv facade for shared-screen NES multiplayer games."""

    metadata = {"render_modes": ["rgb_array"], "name": "nesle_multiplayer_v0", "is_parallelizable": True}

    def __init__(
        self,
        *,
        env_id: str = "NESLE/SuperC-2P-2-v0",
        rom_path: str | Path | None = None,
        obs_type: str = "rgb",
        frameskip: int = 4,
        max_num_frames_per_episode: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if not _HAS_PETTINGZOO:
            raise ImportError(
                "NESMultiPlayerEnv requires the optional 'pettingzoo' dependency "
                "(pip install pettingzoo)."
            )
        if obs_type not in _MULTI_OBS_SHAPE:
            raise ValueError("obs_type must be one of: ram, rgb, grayscale")
        game_id, start_state = resolve_env_id(env_id)
        self.game_id = game_id
        self.env_id = env_id
        self.obs_type = obs_type
        self.render_mode = render_mode
        # One unified NesEnv drives 1..=game.players ports; fresh defaults to the spec's full count.
        self._env = _nesle.NesEnv(game_id)
        self._env.set_start_state(start_state)
        self._env.load_rom_bytes(resolve_rom(game_id, rom_path).read_bytes())
        self._env.set_action_repeat(frameskip, 0.0)
        self._env.set_max_episode_frames(max_num_frames_per_episode)
        self._n = int(self._env.num_players())
        self.possible_agents: list[str] = [f"player_{i}" for i in range(self._n)]
        self.agents: list[str] = []
        self._actions = self._env.minimal_action_set()
        self._obs_space = spaces.Box(low=0, high=255, shape=_MULTI_OBS_SHAPE[obs_type], dtype=np.uint8)
        self._act_space = spaces.Discrete(len(self._actions))
        self.observation_spaces = {a: self._obs_space for a in self.possible_agents}
        self.action_spaces = {a: self._act_space for a in self.possible_agents}

    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._act_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._env.seed(int(seed))
        del options
        self.agents = self.possible_agents[:]
        _, _, lives = self._env.reset_ports()
        obs = self._multi_obs()
        observations = {a: obs.copy() for a in self.agents}
        infos = {a: {"lives": int(lives[i])} for i, a in enumerate(self.possible_agents)}
        return observations, infos

    def step(self, actions: dict[str, int]):
        masks = [0] * self._n
        for i, agent in enumerate(self.possible_agents):
            if agent in self.agents:
                masks[i] = self._multi_mask(actions[agent])

        rewards_arr, terminated_arr, truncated, _, _, lives_arr = self._env.step_ports(masks)

        obs = self._multi_obs()
        observations = {a: obs.copy() for a in self.agents}
        rewards = {a: float(rewards_arr[self.possible_agents.index(a)]) for a in self.agents}
        terminations = {a: bool(terminated_arr[self.possible_agents.index(a)]) for a in self.agents}
        truncations = {a: bool(truncated) for a in self.agents}
        infos = {a: {"lives": int(lives_arr[self.possible_agents.index(a)])} for a in self.agents}

        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self._multi_screen_rgb()

    def close(self) -> None:
        return None

    def _multi_mask(self, action) -> int:
        index = int(action)
        if index < 0 or index >= len(self._actions):
            raise ValueError(f"invalid action index: {index}")
        return int(self._actions[index][1])

    def _multi_obs(self) -> np.ndarray:
        # The per-agent dicts copy this shared view when reset()/step() returns.
        if self.obs_type == "ram":
            return np.frombuffer(self._env.ram(), dtype=np.uint8)
        if self.obs_type == "rgb":
            return self._multi_screen_rgb()
        return np.frombuffer(self._env.screen_gray(), dtype=np.uint8).reshape((240, 256))

    def _multi_screen_rgb(self) -> np.ndarray:
        return np.frombuffer(self._env.screen_rgb(), dtype=np.uint8).reshape((240, 256, 3))


def parallel_env(
    *,
    env_id: str = "NESLE/SuperC-2P-2-v0",
    rom_path: str | Path | None = None,
    obs_type: str = "rgb",
    frameskip: int = 4,
    max_num_frames_per_episode: int | None = None,
    render_mode: str | None = None,
) -> NESMultiPlayerEnv:
    """Return a PettingZoo-style NESLE multiplayer env."""
    return NESMultiPlayerEnv(
        env_id=env_id,
        rom_path=rom_path,
        obs_type=obs_type,
        frameskip=frameskip,
        max_num_frames_per_episode=max_num_frames_per_episode,
        render_mode=render_mode,
    )
