# NESLE — NES Learning Environment

[![build-wheels](https://github.com/tooichitake/nesle/actions/workflows/build-wheels.yml/badge.svg)](https://github.com/tooichitake/nesle/actions/workflows/build-wheels.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: GPL-2.0](https://img.shields.io/badge/license-GPL--2.0-blue.svg)](LICENSE)

NESLE is an original, Rust-native reinforcement-learning environment for the
Nintendo Entertainment System (NES). The cycle-accurate NES core, the RL
environment layer, the preprocessing pipeline, and batched/vectorized stepping
are all written from scratch in Rust and exposed to Python through a single
compiled extension.

NESLE **implements the standard Gymnasium (single-agent) and PettingZoo
(multi-agent) interfaces**, so it drops straight into the existing RL ecosystem
(Stable-Baselines3, CleanRL, …) with no glue code. The interfaces are the only
thing borrowed — the emulator, environments, observation/preprocessing pipeline,
vectorization, memory access, and tooling are all NESLE's own.

```python
import gymnasium as gym
import nesle  # importing registers the NESLE/* environments

env = gym.make("NESLE/SuperMarioBros-1-1-v2", rom_path="smb.nes")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Features

- **Rust-native NES core** — deterministic and integer-only, so a run is
  bit-for-bit reproducible across platforms; verified cycle-accurate against a
  reference emulator (RAM + rendered framebuffer).
- **Gymnasium-compatible single-agent envs** and **PettingZoo-compatible
  multi-agent envs** (2–4 players, last-standing / versus / co-op).
- **Three observation types** — RGB `(240, 256, 3)`, grayscale `(240, 256)`, and
  raw RAM `(2048,)`.
- **Built-in preprocessing** — 84×84 grayscale, frame-skip, 2-frame max-pool,
  frame-stacking, sticky actions, and no-op resets, selected per env version.
- **Built-in vectorization in Rust** — a synchronous batched backend and an
  asynchronous (envpool-style) backend, both with the GIL released.
- **Direct memory access** — read the full NES RAM and the PPU tile field, and
  save / restore emulator state (see [Memory & state interface](#memory--state-interface)).
- **abi3 wheels** — one wheel per platform covers Python 3.10+.

## Installation

### From a release wheel

Pre-built, optimized abi3 wheels (Linux x86-64, Windows x86-64, macOS arm64) are
attached to each [GitHub Release](https://github.com/tooichitake/nesle/releases).

```bash
pip install nesle-<version>-cp310-abi3-<platform>.whl
```

### From source

Requires a Rust toolchain (1.85+) and [maturin](https://www.maturin.rs/).

```bash
pip install maturin
maturin develop --release          # build + install into the active environment
# optional features:
#   --features viewer        # in-process SDL window for render_mode="human"
#   --features audio-synth   # APU audio synthesis (off by default for training)
```

## ROMs

NESLE ships **no game ROMs** — you supply your own `.nes` files. A ROM is
resolved by its **SHA-1** (filename-agnostic, validated against NESLE's game
table), tried in this order:

1. an explicit `rom_path=` argument;
2. a ROM bundled inside the installed wheel (if you built a wheel with ROMs);
3. a directory you registered with `nesle.import_roms(...)`, or pointed at with
   the `NESLE_ROMS_DIR` environment variable.

```python
import gymnasium as gym
import nesle

# (1) explicit path
env = gym.make("NESLE/SuperMarioBros-1-1-v2", rom_path="/path/to/smb.nes")

# (3) register a folder once (copied + indexed by SHA-1), then make envs by id:
nesle.import_roms("/path/to/roms")
#   or per-process:  export NESLE_ROMS_DIR=/path/to/roms

nesle.get_all_game_ids()       # every supported game id
nesle.get_rom_path(game_id)    # packaged path, if present
```

## Usage

### Single-agent (Gymnasium)

```python
import gymnasium as gym
import nesle

# Raw env (v0): pick the observation type.
raw = gym.make("NESLE/SuperMarioBros-1-1-v0", obs_type="rgb")        # or "grayscale" / "ram"

# Standard preprocessed training env (v2): 84×84 grayscale, frame-skip 4.
env = gym.make("NESLE/SuperMarioBros-1-1-v2")
obs, info = env.reset(seed=0)
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

### Vectorized

```python
import gymnasium as gym

# Synchronous: preprocessed profiles build in a 4-frame stack.
vec = gym.make_vec("NESLE/SuperMarioBros-1-1-v2", num_envs=8)
batch_obs, infos = vec.reset()                       # (8, 4, 84, 84)

# Asynchronous (envpool-style): 0 < batch_size < num_envs.
avec = gym.make_vec("NESLE/SuperMarioBros-1-1-v3", num_envs=12, batch_size=4)
avec.async_reset()
obs, rewards, terms, truncs, info = avec.recv()      # info["env_id"] demuxes
avec.send(actions)                                   # one action per env in the recv batch
```

### Multi-agent (PettingZoo)

```python
from nesle.env import parallel_env

env = parallel_env(env_id="NESLE/SuperC-2P-2-v0", rom_path="superc.nes")
obs, infos = env.reset()
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, terminations, truncations, infos = env.step(actions)
```

## Environments

Env ids are `NESLE/<Game>-<level>-<version>` (multi-player stems add a player
tag, e.g. `SuperC-2P`). The version suffix selects the observation / preprocessing
profile:

| Version | Profile |
|---|---|
| `v0` | Raw observation (`obs_type` ∈ `rgb` / `grayscale` / `ram`), configurable action repeat. |
| `v1` | 84×84 grayscale, frame-skip 4, 2-frame max-pool, terminal-on-life-loss. |
| `v2` | Same as v1 with sprite-limit removal and max-pool disabled. |
| `v3` | v2 + sticky actions (`repeat_action_probability = 0.25`). |

`NoFrameskip` variants keep the observation semantics but set action repeat to 1.
Episode summaries (`info["episode"]`) are produced by Gymnasium's
`RecordEpisodeStatistics` wrapper, not by the env itself.

## Memory & state interface

NESLE exposes the live machine state, not just pixels — for reward shaping,
scripted agents/opponents, RAM-map reverse engineering, and debugging.

```python
import gymnasium as gym
env = gym.make("NESLE/SuperMarioBros-1-1-v0", obs_type="ram")
obs, info = env.reset()          # obs IS the 2048-byte NES RAM (uint8)

# Read RAM at any time, independent of obs_type:
ram = env.unwrapped.get_ram()    # np.uint8 (2048,)
value = int(ram[0x075A])         # e.g. some game-specific address

# Other live-state accessors on the single-agent env:
env.unwrapped.get_screen_grayscale()   # (240, 256) uint8
env.unwrapped.get_action_meanings()    # ["NOOP", "RIGHT", ...]

# Save / restore the full emulator state:
snap = env.unwrapped.clone_state()
env.unwrapped.restore_state(snap)      # also: restore_state_blob(bytes)
```

For batched / multi-agent runs, `NESMultiPlayerVectorEnv` exposes per-match
memory directly (the preprocessed image observation does not carry it):

```python
from nesle.vector_env import make_multiplayer_vector_env
venv = make_multiplayer_vector_env("NESLE/Bomberman2-VS-1-v3", num_envs=16)
venv.reset()
ram = venv.get_ram()          # (num_envs, 2048) CPU RAM, one per match
field = venv.get_nametable()  # (num_envs, vram) PPU tile field (walls / bricks / bombs / …)
```

## Building wheels & CI/CD

`.github/workflows/build-wheels.yml` builds optimized abi3 wheels for Linux
x86-64, Windows x86-64, and macOS arm64 on every push to `main`, and attaches
them to the GitHub Release on a `v*` tag. Each wheel is a release build
(thin-LTO, `codegen-units=1`, `-Ctarget-cpu` baseline) with an optional PGO pass
driven by a mapper-diverse workload (`scripts/pgo_workload.py`).

Because NES ROMs are copyrighted, the public repo contains none. CI injects them
at build time from a **private** source so released wheels can ship complete:

- Secret `ROMS_DEPLOY_KEY` — a read-only SSH deploy key for the private ROM repo.
- Variable `ROMS_REPO` — `owner/name` of that repo (default `<owner>/nesle-roms`).

CI checks out the private repo, stages its `*.nes` into the package
(`scripts/stage_roms.py`), then builds. With no key set (e.g. fork PRs) CI builds
a ROM-free wheel and skips PGO. To build a complete wheel locally, just keep the
ROMs in `crates/nesle-py/python/nesle/roms/` (git-ignored) and run
`maturin build --release`.

## License

[GPL-2.0-only](LICENSE). This license covers the source in this repository; it
grants no rights to NES ROMs, which belong to their respective copyright holders.
