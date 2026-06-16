"""Gymnasium VectorEnv facade for the Rust-native NESLE vector path.

One class: ONE Rust
worker-pool engine, and ``batch_size`` selects the consumption discipline.

* ``batch_size`` in ``{0, num_envs}`` -> synchronous full batch: ``reset``/``step``
  push N tasks and drain ALL N completions sorted by env_id (deterministic, fixed
  order). This is the ``make_vec`` + Stable-Baselines3 target and the default.
* ``0 < batch_size < num_envs`` -> envpool-style asynchronous: drive it with
  ``async_reset`` then a ``recv``/``send`` loop; ``recv`` returns the first
  ``batch_size`` envs to finish, tagged via ``info["env_id"]`` (out-of-order, GIL
  released). ``reset``/``step`` are unavailable in this mode.

Frame stacking (``stack_num``) is built in for both modes: preprocessed
(grayscale) observations default to ``frame_stack=4`` so ``make_vec`` hands back a
ready-to-train ``(num_envs, frame_stack, H, W)`` tensor, exactly like
``FrameStackObservation + AtariPreprocessing``. Raw obs (rgb/ram) default to no
stacking and are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from nesle.registration import resolve_env_id, vector_profile_params
from nesle.roms import resolve_rom

from . import _nesle


class NESSinglePlayerVectorEnv(VectorEnv):
    def __init__(
        self,
        *,
        num_envs: int,
        game_id: str = "super_mario_bros",
        _level_state: str | None = None,
        rom_path: str | Path | None = None,
        obs_type: str = "rgb",
        preprocessed: bool = False,
        frame_skip: int = 1,
        frame_stack: int | None = None,
        repeat_action_probability: float = 0.0,
        full_action_space: bool = False,
        max_num_frames_per_episode: int | None = None,
        max_episode_steps: int | None = None,
        autoreset_mode: AutoresetMode | str = AutoresetMode.NEXT_STEP,
        seed: int = 1,
        screen_size: int | tuple[int, int] | None = None,
        noop_max: int = 0,
        terminal_on_life_loss: bool = False,
        remove_sprite_limit: bool = False,
        max_pool: bool = False,
        clip_reward: bool = False,
        scale_obs: bool = False,
        batch_size: int = 0,
        num_threads: int = 0,
    ) -> None:
        self.num_envs = int(num_envs)
        self.game_id = game_id
        self._level_state = _level_state
        self.obs_type = obs_type
        self.full_action_space = full_action_space
        self.render_mode = None
        # Preprocessing is a distinct layer (AtariPreprocessing-style), NOT an obs_type value.
        self._preprocessed = bool(preprocessed)
        self._noop_max = int(noop_max)
        self._clip_reward = bool(clip_reward)
        autoreset = _gym_autoreset_mode(autoreset_mode)
        self._autoreset = autoreset
        self.metadata = {"render_modes": [], "autoreset_mode": autoreset}

        # stack_num: preprocessed envs default to a 4-frame stack; raw obs to none.
        if frame_stack is None:
            frame_stack = 4 if self._preprocessed else 1
        self.frame_stack = int(frame_stack)

        # batch_size selects the backend: full batch -> sync (reset/step); partial -> envpool async.
        batch_size = int(batch_size)
        self._async = 0 < batch_size < self.num_envs
        self.batch_size = batch_size if self._async else self.num_envs
        if self._async and not self._preprocessed:
            raise ValueError("async vector env (batch_size < num_envs) requires preprocessed=True")

        action_repeat = int(frame_skip)
        # Honor BOTH episode caps by taking the tighter one (both in NES-frame units).
        _caps = []
        if max_num_frames_per_episode is not None:
            _caps.append(int(max_num_frames_per_episode))
        if max_episode_steps is not None:
            _caps.append(int(max_episode_steps) * action_repeat)
        max_frames = min(_caps) if _caps else None

        proto = _nesle.NesEnv(game_id)
        self._actions = proto.full_action_set() if full_action_space else proto.minimal_action_set()
        self.single_action_space = spaces.Discrete(len(self._actions))
        self.action_space = spaces.MultiDiscrete([len(self._actions)] * self.num_envs)
        rom_bytes = resolve_rom(game_id, rom_path).read_bytes()

        if self._async:
            single_shape = self._build_async(
                rom_bytes,
                num_threads,
                action_repeat,
                screen_size,
                max_pool,
                terminal_on_life_loss,
                remove_sprite_limit,
                repeat_action_probability,
                seed,
                max_frames,
                autoreset,
            )
        else:
            single_shape = self._build_sync(
                rom_bytes,
                max_frames,
                autoreset,
                action_repeat,
                repeat_action_probability,
                seed,
                remove_sprite_limit,
                screen_size,
                max_pool,
                terminal_on_life_loss,
            )

        self._hw = single_shape
        stacked_shape = (self.frame_stack, *single_shape) if self.frame_stack > 1 else single_shape
        dtype = np.float32 if scale_obs else np.uint8
        high = 1.0 if scale_obs else 255
        self.single_observation_space = spaces.Box(
            low=0, high=high, shape=stacked_shape, dtype=dtype
        )
        self.observation_space = spaces.Box(
            low=0, high=high, shape=(self.num_envs, *stacked_shape), dtype=dtype
        )
        self._scale_obs = scale_obs
        self._recv_env_ids: np.ndarray | None = None

    # -- construction --------------------------------------------------------

    def _build_sync(
        self,
        rom_bytes,
        max_frames,
        autoreset,
        action_repeat,
        repeat_action_probability,
        seed,
        remove_sprite_limit,
        screen_size,
        max_pool,
        terminal_on_life_loss,
    ) -> tuple[int, ...]:
        obs_mode = "preprocessed" if self._preprocessed else self.obs_type
        height, width = _screen_hw(screen_size) if self._preprocessed else (84, 84)
        self._env = _nesle.NesVectorEnv(
            self.num_envs,
            game_id=self.game_id,
            rom=rom_bytes,
            obs_mode=obs_mode,
            batch_size=self.num_envs,
            frame_skip=action_repeat,
            width=width,
            height=height,
            maxpool=bool(max_pool),
            stack_num=self.frame_stack,
            terminal_on_life_loss=bool(terminal_on_life_loss),
            repeat_action_probability=float(repeat_action_probability),
            noop_max=self._noop_max,
            seed=int(seed),
            max_episode_frames=max_frames,
            autoreset_mode=autoreset.value,
            remove_sprite_limit=bool(remove_sprite_limit),
            start_state=self._level_state,
        )
        return self._single_obs_shape(screen_size)

    def _build_async(
        self,
        rom_bytes,
        num_threads,
        action_repeat,
        screen_size,
        max_pool,
        terminal_on_life_loss,
        remove_sprite_limit,
        repeat_action_probability,
        seed,
        max_frames,
        autoreset,
    ) -> tuple[int, ...]:
        height, width = _screen_hw(screen_size)
        self._env = _nesle.NesVectorEnv(
            num_envs=self.num_envs,
            batch_size=self.batch_size,
            rom=rom_bytes,
            game_id=self.game_id,
            obs_mode="preprocessed",
            num_threads=int(num_threads),
            frame_skip=action_repeat,
            width=width,
            height=height,
            maxpool=bool(max_pool),
            terminal_on_life_loss=bool(terminal_on_life_loss),
            repeat_action_probability=float(repeat_action_probability),
            noop_max=self._noop_max,
            seed=int(seed),
            max_episode_frames=max_frames,
            autoreset_mode=autoreset.value,
            remove_sprite_limit=bool(remove_sprite_limit),
            start_state=self._level_state,
        )
        return (height, width)

    # -- synchronous API (reset/step + full-batch send/recv) -----------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self._async:
            raise RuntimeError("async vector env: use async_reset() + recv()/send(), not reset()")
        super().reset(seed=seed)
        del options
        infos = self._env.reset()
        return self._maybe_scale(self._single_obs()), self._infos(infos)

    def step(self, actions):
        if self._async:
            raise RuntimeError("async vector env: use recv()/send(), not step()")
        masks = self._action_masks(actions, self.num_envs)
        steps = self._env.step(masks)
        return self._sync_step_result(steps)

    def send(self, actions) -> None:
        """Queue actions. Sync: one action per env (fixed order). Async: one action
        per env from the most recent ``recv`` batch, in completion order."""
        if self._async:
            if self._recv_env_ids is None:
                raise RuntimeError("async send() requires a preceding recv()")
            env_ids = self._recv_env_ids
            masks = self._action_masks(actions, len(env_ids))
            self._env.async_send([int(e) for e in env_ids], masks)
        else:
            self._env.send(self._action_masks(actions, self.num_envs))

    def recv(self):
        if self._async:
            return self._async_recv()
        return self._sync_step_result(self._env.recv())

    # -- asynchronous API (envpool-style) ------------------------------------

    def async_reset(self) -> None:
        """Start every env stepping (each with its noop start); the first recv is full."""
        if not self._async:
            raise RuntimeError(
                "async_reset() requires a vector env built with batch_size < num_envs"
            )
        self._recv_env_ids = None
        self._env.async_reset()

    def _async_recv(self):
        (
            obs_bytes,
            env_ids,
            rewards,
            terminated,
            truncated,
            frame_number,
            episode_frame_number,
            lives,
        ) = self._env.async_recv()
        height, width = self._hw
        if self.frame_stack > 1:
            frames = np.frombuffer(obs_bytes, np.uint8).reshape(
                len(env_ids), self.frame_stack, height, width
            )
        else:
            frames = np.frombuffer(obs_bytes, np.uint8).reshape(len(env_ids), height, width)
        env_ids = np.asarray(env_ids, dtype=np.int64)
        self._recv_env_ids = env_ids
        obs = self._maybe_scale(frames)
        infos = {
            "env_id": env_ids,
            "frame_number": np.asarray(frame_number, dtype=np.int64),
            "episode_frame_number": np.asarray(episode_frame_number, dtype=np.int64),
            "lives": np.asarray(lives, dtype=np.int64),
        }
        return (
            obs,
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            infos,
        )

    def close(self) -> None:
        super().close()

    def close_extras(self, **kwargs: Any) -> None:
        del kwargs
        if self._async:
            del self._env

    # -- frame stacking ------------------------------------------------------

    def _maybe_scale(self, obs: np.ndarray) -> np.ndarray:
        if self._scale_obs:
            return obs.astype(np.float32) / 255.0
        return obs

    # -- helpers -------------------------------------------------------------

    def _single_obs_shape(self, screen_size: int | tuple[int, int] | None) -> tuple[int, ...]:
        if not self._preprocessed:
            if self.obs_type not in ("ram", "rgb", "grayscale"):
                raise ValueError("obs_type must be one of: ram, rgb, grayscale")
            return {"ram": (2048,), "rgb": (240, 256, 3), "grayscale": (240, 256)}[self.obs_type]

        height, width = _screen_hw(screen_size)
        self.obs_type = "grayscale"
        return (height, width)

    def get_ram(self) -> np.ndarray:
        """Per-env CPU RAM ``(num_envs, 2048)`` -- live game state the preprocessed obs drops."""
        return np.frombuffer(self._env.ram_batch(), dtype=np.uint8).reshape(self.num_envs, 2048)

    def get_nametable(self) -> np.ndarray:
        """Per-env PPU nametable / CIRAM tile field ``(num_envs, vram_len)``."""
        return np.frombuffer(self._env.nametable_batch(), dtype=np.uint8).reshape(self.num_envs, -1)

    def get_screen_grayscale(self) -> np.ndarray:
        """Per-env native grayscale screen ``(num_envs, 240, 256)`` (independent of obs_type)."""
        return np.frombuffer(self._env.grayscale_batch(), dtype=np.uint8).reshape(
            self.num_envs, 240, 256
        )

    def get_screen_rgb(self) -> np.ndarray:
        """Per-env native RGB screen ``(num_envs, 240, 256, 3)`` (independent of obs_type)."""
        return np.frombuffer(self._env.rgb_batch(), dtype=np.uint8).reshape(
            self.num_envs, 240, 256, 3
        )

    def set_ram(self, ram) -> None:
        """Overwrite every env's CPU RAM; ``ram`` is ``(num_envs, 2048)`` uint8."""
        arr = np.ascontiguousarray(ram, dtype=np.uint8)
        if arr.shape != (self.num_envs, 2048):
            raise ValueError(f"RAM must have shape ({self.num_envs}, 2048), got {arr.shape}")
        self._env.set_ram_batch(arr.tobytes())

    def _single_obs(self) -> np.ndarray:
        if self._preprocessed:
            shape = (
                (self.num_envs, self.frame_stack, *self._hw)
                if self.frame_stack > 1
                else (self.num_envs, *self._hw)
            )
            return np.frombuffer(self._env.observation_batch(), dtype=np.uint8).reshape(shape)
        if self.obs_type == "ram":
            return np.frombuffer(self._env.ram_batch(), dtype=np.uint8).reshape(
                (self.num_envs, 2048)
            )
        if self.obs_type == "rgb":
            return np.frombuffer(self._env.rgb_batch(), dtype=np.uint8).reshape(
                (self.num_envs, 240, 256, 3)
            )
        return np.frombuffer(self._env.grayscale_batch(), dtype=np.uint8).reshape(
            (self.num_envs, 240, 256)
        )

    def _action_masks(self, actions, expected: int) -> list[int]:
        if len(actions) != expected:
            raise ValueError(f"action count must match: {len(actions)} != {expected}")
        masks: list[int] = []
        for action in actions:
            index = int(action)
            if index < 0 or index >= len(self._actions):
                raise ValueError(f"invalid action index: {index}")
            masks.append(int(self._actions[index][1]))
        return masks

    def _sync_step_result(self, steps):
        single = self._single_obs()
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        terminated = np.zeros((self.num_envs,), dtype=bool)
        truncated = np.zeros((self.num_envs,), dtype=bool)
        infos: dict[str, np.ndarray] = {
            "frame_number": np.zeros((self.num_envs,), dtype=np.int64),
            "episode_frame_number": np.zeros((self.num_envs,), dtype=np.int64),
            "lives": np.zeros((self.num_envs,), dtype=np.int64),
            "final_observation": np.zeros((self.num_envs,), dtype=bool),
        }
        for env_id, reward, term, trunc, frame, episode_frame, lives, final_obs in steps:
            rewards[env_id] = np.sign(reward) if self._clip_reward else reward
            terminated[env_id] = term
            truncated[env_id] = trunc
            infos["frame_number"][env_id] = frame
            infos["episode_frame_number"][env_id] = episode_frame
            infos["lives"][env_id] = lives
            infos["final_observation"][env_id] = final_obs
        obs = self._maybe_scale(single)
        return obs, rewards, terminated, truncated, infos

    def _infos(self, raw_infos):
        count = len(raw_infos)
        infos: dict[str, np.ndarray] = {
            "frame_number": np.zeros((count,), dtype=np.int64),
            "episode_frame_number": np.zeros((count,), dtype=np.int64),
            "lives": np.zeros((count,), dtype=np.int64),
        }
        for env_id, (frame, episode_frame, lives) in enumerate(raw_infos):
            infos["frame_number"][env_id] = frame
            infos["episode_frame_number"][env_id] = episode_frame
            infos["lives"][env_id] = lives
        return infos


def _screen_hw(screen_size: int | tuple[int, int] | None) -> tuple[int, int]:
    if isinstance(screen_size, int):
        return int(screen_size), int(screen_size)
    if (
        isinstance(screen_size, tuple)
        and len(screen_size) == 2
        and all(isinstance(size, int) for size in screen_size)
    ):
        width, height = screen_size
        return int(height), int(width)
    raise ValueError("preprocessed vector env requires screen_size as int or (width, height)")


def _gym_autoreset_mode(value: AutoresetMode | str) -> AutoresetMode:
    if isinstance(value, AutoresetMode):
        return value
    if value == AutoresetMode.NEXT_STEP.value:
        return AutoresetMode.NEXT_STEP
    if value == AutoresetMode.SAME_STEP.value:
        return AutoresetMode.SAME_STEP
    raise ValueError(f"unsupported autoreset_mode: {value}")


class NESMultiPlayerVectorEnv:
    """Multi-player vectorized env (the multi sibling of NESSinglePlayerVectorEnv).

    Naming follows the regular ``NES[MultiPlayer][Vector]Env`` scheme — single-agent
    is the unmarked default (``NESSinglePlayerEnv`` / ``NESSinglePlayerVectorEnv``, like gym.Env); only the multi-player variants carry ``MultiPlayer``. Unlike the other
    three it has NO standard base class: there is no community-standard multi-agent
    *vectorized* base (Gymnasium ``VectorEnv`` is single-agent, PettingZoo
    ``ParallelEnv`` is non-vectorized), so this is a deliberate custom API. It is
    typically driven as parameter-sharing self-play.

    Both this and ``NESSinglePlayerVectorEnv`` wrap the SAME Rust vector engine
    (``_nesle.NesVectorEnv``), this one in FlatSlots mode (``players=N``).
    K worker units (parallel matches) each drive N controller ports -> ``K*N``
    agent slots stepped by ONE shared policy. Each slot sees its unit's shared NES
    screen (grayscale, resized + frame-stacked) plus a fixed ``player_id`` so the
    policy can tell the co-located players apart. ``SameStep`` autoreset, so the obs
    on a done step is already the next episode's first frame.

    Slot layout is unit-major: slot ``u*N + p`` is match ``u``'s port ``p`` (the same
    flat unit-major order the action stream uses). ``recv_ports`` returns one per-unit
    tuple with the full per-port arrays, demuxed back to slots here.

    Per-port termination is verbatim. In a round-ends-together mode (Bomberman 2 VS,
    2P) a death ends the round, so ``terminated[slot]`` coincides with the unit reset
    (``final_observation``) -- clean PPO boundaries. Modes where one agent can die
    mid-match (Bomberman 2 Battle, 3P) keep stepping survivors; a faithful trainer
    must stop collecting that slot until its unit's ``final_observation``.
    """

    def __init__(
        self,
        *,
        env_id: str,
        num_envs: int,
        rom_path: str | Path | None = None,
        screen_size: int = 84,
        frame_stack: int = 4,
        frame_skip: int = 4,
        max_pool: bool = False,
        terminal_on_life_loss: bool = False,
        noop_max: int = 0,
        max_num_frames_per_episode: int | None = None,
        seed: int = 1,
        num_threads: int = 0,
        repeat_action_probability: float = 0.0,
        remove_sprite_limit: bool = True,
    ) -> None:
        game_id, start_state = resolve_env_id(env_id)
        self.game_id = game_id
        self.env_id = env_id
        self.num_envs = int(num_envs)
        self.screen_size = int(screen_size)
        self.frame_stack = int(frame_stack)

        proto = _nesle.NesEnv(game_id)
        self.num_players = int(proto.num_players())  # N ports = agents per match
        self.num_agents = self.num_envs * self.num_players
        self._actions = proto.minimal_action_set()
        self.num_actions = len(self._actions)
        self._masks = np.array([mask for _, mask in self._actions], dtype=np.uint8)
        # Fixed per-slot player index (slot u*N+p -> player p).
        self.player_ids = np.tile(np.arange(self.num_players, dtype=np.int64), self.num_envs)

        rom = resolve_rom(game_id, rom_path).read_bytes()
        # PREPROCESSED obs_mode: the Rust ObsPipeline does grayscale/resize/maxpool/stack per unit; Python fans it out K -> K*N.
        self._env = _nesle.NesVectorEnv(
            self.num_envs,
            game_id=game_id,
            rom=rom,
            obs_mode="preprocessed",
            players=self.num_players,
            frame_skip=int(frame_skip),
            width=self.screen_size,
            height=self.screen_size,
            maxpool=bool(max_pool),
            stack_num=self.frame_stack,
            terminal_on_life_loss=bool(terminal_on_life_loss),
            noop_max=int(noop_max),
            seed=int(seed),
            num_threads=int(num_threads),
            repeat_action_probability=float(repeat_action_probability),
            max_episode_frames=max_num_frames_per_episode,
            autoreset_mode="SameStep",
            remove_sprite_limit=bool(remove_sprite_limit),
            start_state=start_state,
        )
        # Per-slot "still alive this match" flag (spectator masking for mid-match deaths).
        self._active = np.ones(self.num_agents, dtype=bool)

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (self.frame_stack, self.screen_size, self.screen_size)

    def get_ram(self) -> np.ndarray:
        """Per-env CPU RAM ``(num_envs, 2048)`` -- one shared NES RAM per match.
        For reward-shaping / scripted opponents that read game state (player
        positions, alive flags, ...) which the preprocessed obs does not carry."""
        return np.frombuffer(self._env.ram_batch(), dtype=np.uint8).reshape(self.num_envs, 2048)

    def get_nametable(self) -> np.ndarray:
        """Per-env PPU nametable / CIRAM ``(num_envs, vram_len)`` -- the rendered
        field's tile ids (walls / soft bricks / floor / bombs / flames). The field
        is NOT a per-tile array in CPU RAM, so brick-clearing reward shaping and
        scripted bots read it from here."""
        flat = np.frombuffer(self._env.nametable_batch(), dtype=np.uint8)
        return flat.reshape(self.num_envs, -1)

    def get_screen_grayscale(self) -> np.ndarray:
        """Per-match native grayscale screen ``(num_envs, 240, 256)`` -- one shared screen per match."""
        return np.frombuffer(self._env.grayscale_batch(), dtype=np.uint8).reshape(
            self.num_envs, 240, 256
        )

    def get_screen_rgb(self) -> np.ndarray:
        """Per-match native RGB screen ``(num_envs, 240, 256, 3)`` -- one shared screen per match."""
        return np.frombuffer(self._env.rgb_batch(), dtype=np.uint8).reshape(
            self.num_envs, 240, 256, 3
        )

    def set_ram(self, ram) -> None:
        """Overwrite every match's CPU RAM; ``ram`` is ``(num_envs, 2048)`` uint8."""
        arr = np.ascontiguousarray(ram, dtype=np.uint8)
        if arr.shape != (self.num_envs, 2048):
            raise ValueError(f"RAM must have shape ({self.num_envs}, 2048), got {arr.shape}")
        self._env.set_ram_batch(arr.tobytes())

    def _stacked_obs(self) -> np.ndarray:
        """Per-slot frame-stacked obs. The Rust ObsPipeline already produced one
        ``(frame_stack, H, W)`` stack per unit (shared screen, resize + stack done in
        Rust); fan it out K -> K*N so every co-located player sees its unit's stack."""
        h = w = self.screen_size
        units = np.frombuffer(self._env.observation_batch(), np.uint8).reshape(
            self.num_envs, self.frame_stack, h, w
        )
        return np.repeat(units, self.num_players, axis=0)

    def reset(self):
        self._env.reset()
        self._active[:] = True
        return self._stacked_obs(), {"player_id": self.player_ids}

    def step(self, actions):
        masks = self._masks[np.asarray(actions, dtype=np.int64)]
        self._env.send(masks.astype(np.uint8).tolist())  # flat unit-major
        ports = self._env.recv_ports()

        n = self.num_players
        rewards = np.zeros(self.num_agents, np.float32)
        # `done` = per-agent episode end (spectator-masked): a dead player ends once, then spectates (reward 0) until the unit resets.
        done = np.zeros(self.num_agents, bool)
        unit_done = np.zeros(self.num_envs, bool)
        for env_id, rew4, term4, trunc, _frame, _ep_frame, _lives4, final_obs in ports:
            base = env_id * n
            unit_done[env_id] = final_obs
            for p in range(n):
                slot = base + p
                if final_obs:
                    # SameStep reset: a still-active slot signals done now; all slots reactivate.
                    if self._active[slot]:
                        rewards[slot] = rew4[p]
                        done[slot] = True
                    self._active[slot] = True
                elif self._active[slot]:
                    rewards[slot] = rew4[p]
                    if bool(term4[p]) or bool(trunc):
                        done[slot] = True
                        self._active[slot] = False  # spectate until the unit resets

        # SameStep autoreset + Rust frame-stacking: `observation_batch()` carries the rolled stack.
        obs = self._stacked_obs()
        agent_unit_done = np.repeat(unit_done, n)
        infos = {"player_id": self.player_ids, "final_observation": agent_unit_done}
        truncated = np.zeros(self.num_agents, bool)  # truncation folded into `done`
        return obs, rewards, done, truncated, infos

    def close(self) -> None:
        if hasattr(self, "_env"):
            del self._env


def make_multiplayer_vector_env(
    env_id: str,
    num_envs: int,
    *,
    frame_stack: int = 4,
    max_episode_steps: int | None = None,
    seed: int = 1,
    num_threads: int = 0,
    **overrides: Any,
) -> NESMultiPlayerVectorEnv:
    """Build a ``NESMultiPlayerVectorEnv`` with the env-id's ``-vN`` profile auto-applied
    -- the multi-player analog of ``gym.make_vec`` (the class is not gym-registered, so it
    gets an explicit factory instead of going through the gym registry).

    ``NESMultiPlayerVectorEnv`` is profile-dumb (explicit preprocessing kwargs, like
    ``NESSinglePlayerVectorEnv``); this factory is the profile-applying layer: it reads
    ``registration.vector_profile_params`` (the ``_PROFILE_PARAMS`` source of truth)
    from the ``-v1/-v2/-v3`` suffix and passes screen_size / frame_skip / max_pool /
    noop_max / sticky to the env, so the trainer just names ``...-v3`` instead of
    hand-setting the profile. ``terminal_on_life_loss`` is forced ``False``: in
    last-standing multi-player (Bomberman) death is handled by the game's
    ``per_agent_lives_termination`` + spectator masking, and ALE episodic-life would
    mis-end Battle-3P units on the first death. ``frame_stack`` and
    ``max_episode_steps`` are training knobs (not part of the obs profile); any
    keyword in ``overrides`` wins over the profile.
    """
    params = vector_profile_params(env_id)
    frame_skip = int(params["frame_skip"])
    max_frames = max_episode_steps * frame_skip if max_episode_steps else None
    kwargs: dict[str, Any] = dict(
        env_id=env_id,
        num_envs=num_envs,
        screen_size=int(params["screen_size"]),
        frame_stack=frame_stack,
        frame_skip=frame_skip,
        max_pool=bool(params["max_pool"]),
        noop_max=int(params["noop_max"]),
        repeat_action_probability=float(params["repeat_action_probability"]),
        terminal_on_life_loss=False,
        remove_sprite_limit=bool(params["remove_sprite_limit"]),
        max_num_frames_per_episode=max_frames,
        seed=seed,
        num_threads=num_threads,
    )
    kwargs.update(overrides)
    return NESMultiPlayerVectorEnv(**kwargs)
