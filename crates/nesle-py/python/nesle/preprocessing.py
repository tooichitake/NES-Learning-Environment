"""Gymnasium-style NES preprocessing backed by the shared Rust pipeline."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from nesle.env import NESSinglePlayerEnv

ScreenSize = int | tuple[int, int]


def _screen_size_shape(screen_size: ScreenSize) -> tuple[int, int]:
    if isinstance(screen_size, int) and screen_size > 0:
        return screen_size, screen_size
    if (
        isinstance(screen_size, tuple)
        and len(screen_size) == 2
        and all(isinstance(size, int) and size > 0 for size in screen_size)
    ):
        return screen_size
    raise AssertionError(f"screen_size must be a positive int or (width, height), got {screen_size!r}")


class NESPreprocessing(gym.Wrapper):
    """NES analogue of ``gymnasium.wrappers.AtariPreprocessing``."""

    def __init__(
        self,
        env: gym.Env,
        *,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: ScreenSize = 112,
        terminal_on_life_loss: bool = False,
        grayscale_newaxis: bool = False,
        scale_obs: bool = False,
        maxpool: bool = True,
        render_skip: bool = True,
    ) -> None:
        super().__init__(env)
        assert frame_skip > 0
        width, height = _screen_size_shape(screen_size)
        if frame_skip > 1 and getattr(env.unwrapped, "frameskip", None) != 1:
            raise ValueError(
                "base env must have frameskip=1; NESPreprocessing performs the frame-skip itself"
            )
        assert noop_max >= 0
        if noop_max > 0:
            assert env.unwrapped.get_action_meanings()[0] == "NOOP"
        if not hasattr(env.unwrapped, "get_screen_grayscale"):
            raise TypeError("NESPreprocessing requires a NESSinglePlayerEnv-style env with get_screen_grayscale()")

        self.noop_max = noop_max
        self.frame_skip = frame_skip
        self.screen_size = (width, height)
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_newaxis = grayscale_newaxis
        self.scale_obs = scale_obs
        self.maxpool = maxpool
        self.render_skip = render_skip

        self.lives = 0
        self.game_over = False
        env.unwrapped._env.configure_obs_shape(
            frame_skip, width, height, maxpool, render_skip, terminal_on_life_loss
        )

        _low, _high, _dtype = (0, 1, np.float32) if scale_obs else (0, 255, np.uint8)
        _shape = (height, width, 1) if grayscale_newaxis else (height, width)
        self.observation_space = Box(low=_low, high=_high, shape=_shape, dtype=_dtype)

    def step(self, action):
        _, mask = self.env.unwrapped._actions[int(action)]
        obs_bytes, reward, terminated, truncated, frame, ep_frame, lives = (
            self.env.unwrapped._env.observe_step(mask)
        )
        self.lives = int(lives)
        self.game_over = bool(terminated)
        info = {
            "frame_number": int(frame),
            "episode_frame_number": int(ep_frame),
            "lives": int(lives),
        }
        return self._format(obs_bytes), float(reward), bool(terminated), bool(truncated), info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _, reset_info = self.env.reset(seed=seed, options=options)
        obs_bytes, frame, ep_frame, lives = self.env.unwrapped._env.observe_reset(self.noop_max)
        self.lives = int(lives)
        self.game_over = False
        reset_info.update(
            frame_number=int(frame), episode_frame_number=int(ep_frame), lives=int(lives)
        )
        return self._format(obs_bytes), reset_info

    def _format(self, obs_bytes) -> np.ndarray:
        width, height = self.screen_size
        obs = np.frombuffer(obs_bytes, dtype=np.uint8).reshape((height, width))
        if self.scale_obs:
            obs = obs.astype(np.float32) / 255.0
        else:
            obs = obs.copy()  # frombuffer view is read-only; match gymnasium's writable obs
        if self.grayscale_newaxis:
            obs = np.expand_dims(obs, axis=-1)
        return obs


class ClipRewardWrapper(gym.RewardWrapper):
    """Clip reward to its sign {-1, 0, +1}."""

    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


def _make_preprocessed_env(
    *,
    game_id: str,
    preprocessed: bool = True,
    screen_size: ScreenSize = 84,
    frameskip: int = 4,
    noop_max: int = 30,
    terminal_on_life_loss: bool = False,
    grayscale_newaxis: bool = False,
    scale_obs: bool = False,
    remove_sprite_limit: bool = False,
    render_skip: bool = True,
    clip_reward: bool = False,
    maxpool: bool = True,
    repeat_action_probability: float = 0.0,
    full_action_space: bool = False,
    max_num_frames_per_episode: int | None = 108_000,
    rom_path: Any = None,
    _level_state: str | None = None,
) -> gym.Env:
    """Build the registered Gymnasium preprocessing pipeline (always grayscale,
    mirroring gymnasium.AtariPreprocessing). ``preprocessed`` is the shared env-spec
    flag the vector entry point reads; it must be True on this (preprocessed) path."""
    if not preprocessed:
        raise ValueError("_make_preprocessed_env builds the preprocessed pipeline; preprocessed must be True")
    env: gym.Env = NESSinglePlayerEnv(
        game_id=game_id,
        obs_type="ram",
        frameskip=1,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        remove_sprite_limit=remove_sprite_limit,
        max_num_frames_per_episode=max_num_frames_per_episode,
        rom_path=rom_path,
        _level_state=_level_state,
    )
    env = NESPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frameskip,
        screen_size=screen_size,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_newaxis=grayscale_newaxis,
        scale_obs=scale_obs,
        maxpool=maxpool,
        render_skip=render_skip,
    )
    if clip_reward:
        env = ClipRewardWrapper(env)
    return env
