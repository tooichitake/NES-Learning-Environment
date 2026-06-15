# NESLE — NES Learning Environment

[![build-wheels](https://github.com/tooichitake/nesle/actions/workflows/build-wheels.yml/badge.svg)](https://github.com/tooichitake/nesle/actions/workflows/build-wheels.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: GPL-2.0](https://img.shields.io/badge/license-GPL--2.0-blue.svg)](LICENSE)

A Rust-native, **ALE-py-shaped** reinforcement-learning environment for the
Nintendo Entertainment System (NES). The emulator core, environment layer, and
vectorized stepping are all implemented in Rust and exposed to Python through a
single PyO3 extension — so `nesle` plugs into the Gymnasium / PettingZoo
ecosystem the same way `ale-py` does for the Atari 2600.

```python
import gymnasium
import nesle  # importing registers the NESLE/* env ids

env = gymnasium.make("NESLE/SuperMarioBros-1-1-v2", rom_path="super-mario-bros.nes")
obs, info = env.reset()          # (84, 84) grayscale
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Highlights

- **Rust-native NES core** — cycle-accurate enough to match a Mesen2 reference on
  RAM and the indexed framebuffer; integer-only and deterministic (same result
  native and across platforms).
- **ALE-py-faithful Python API** — `gymnasium.make` / `make_vec` for single-agent
  play, raw `rgb` / `grayscale` / `ram` observations, the preprocessed
  84×84 training pipeline, sticky actions, no-op resets, and per-game minimal
  action sets.
- **Built-in vectorization** — a Rust worker-pool vector env with both a
  synchronous (`make_vec` / SB3) backend and an envpool-style asynchronous
  backend (`async_reset` / `send` / `recv`).
- **Multi-agent** — 2–4 player games are exposed as PettingZoo `ParallelEnv`s
  and pass `parallel_api_test`.
- **abi3 wheels** — one wheel per platform covers Python 3.10+.

## Architecture

The package is a small Cargo workspace; only the Python facade is user-facing.

| Crate | Responsibility |
|---|---|
| `nesle-common` | Shared primitive types. |
| `nesle-core` | The complete NES machine (CPU / PPU / APU / mappers). |
| `nesle-rl` | ALE-style environment, per-game specs, preprocessing, vectorization. |
| `nesle-py` | PyO3 extension (`_nesle`) + the `nesle` Python package (incl. the optional SDL `viewer`). |

## Installation

### From a release wheel (recommended)

Pre-built, fully-optimized abi3 wheels (Linux x86-64, Windows x86-64, macOS
arm64) are attached to each [GitHub Release](https://github.com/tooichitake/nesle/releases).
Release wheels ship the supported ROMs inside the wheel.

```bash
pip install nesle-<version>-cp310-abi3-<platform>.whl
```

### From source

Requires a Rust toolchain (1.85+) and [maturin](https://www.maturin.rs/).

```bash
pip install maturin
maturin develop --release         # build + install into the active environment
# optional features:
#   --features viewer        # in-process SDL window for render_mode="human"
#   --features audio-synth   # APU audio synthesis (off by default for training)
```

A source build produces a **ROM-free** install (see below).

## ROMs

NES ROMs are copyrighted, so — exactly like `ale-py` — **this repository does not
contain any game ROMs**, and you bring your own. There are three ways to make a
ROM available, tried in this order (`nesle.resolve_rom`):

1. **Explicit path** — pass `rom_path=` to `gymnasium.make(...)`:
   ```python
   env = gymnasium.make("NESLE/SuperMarioBros-1-1-v2", rom_path="/path/to/smb.nes")
   ```
2. **Packaged ROM** — ROMs bundled inside a release wheel resolve automatically
   by SHA-1, so `gymnasium.make("NESLE/SuperMarioBros-1-1-v2")` just works.
3. **Imported ROM directory** — register a folder once (the `ale-import-roms`
   analogue), then make envs by id with no `rom_path`:
   ```python
   import nesle
   nesle.import_roms("/path/to/my/roms")     # identifies + copies by SHA-1
   # or point at a directory per-process:
   #   export NESLE_ROMS_DIR=/path/to/my/roms   (the ALE_ROMS_DIR analogue)
   ```

`nesle.get_all_game_ids()` lists every supported game; `nesle.get_rom_path(game_id)`
returns the packaged path when present.

## Quickstart

### Single-agent (Gymnasium)

```python
import gymnasium
import nesle

# Raw env (v0): obs_type ∈ {rgb (240,256,3), grayscale (240,256), ram (2048,)}.
raw = gymnasium.make("NESLE/SuperMarioBros-1-1-v0", obs_type="rgb")

# Standard preprocessed training env (v2): 84×84 grayscale, frame-skip 4.
env = gymnasium.make("NESLE/SuperMarioBros-1-1-v2")
obs, info = env.reset(seed=0)
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

### Vectorized

```python
# Synchronous vector env: preprocessed profiles build in a 4-frame stack.
vec = gymnasium.make_vec("NESLE/SuperMarioBros-1-1-v2", num_envs=8)
batch_obs, infos = vec.reset()        # (8, 4, 84, 84)

# Asynchronous (envpool-style) backend: 0 < batch_size < num_envs.
avec = gymnasium.make_vec("NESLE/SuperMarioBros-1-1-v3", num_envs=12, batch_size=4)
avec.async_reset()
obs, rewards, terms, truncs, info = avec.recv()   # info["env_id"] demuxes
avec.send(actions)                                # one action per env in the recv batch
```

### Multi-agent (PettingZoo)

```python
from nesle.env import parallel_env

env = parallel_env(env_id="NESLE/SuperC-2P-2-v0", rom_path="superc.nes")
obs, infos = env.reset()
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, terminations, truncations, infos = env.step(actions)
```

## Environment IDs & profiles

Env ids follow `NESLE/<Game>-<level>-<version>` (multiplayer stems add a player
tag, e.g. `SuperC-2P`). The version suffix selects the observation /
preprocessing profile:

| Version | Profile |
|---|---|
| `v0` | Raw observation (`obs_type` ∈ `rgb` / `grayscale` / `ram`), action repeat configurable. |
| `v1` | Standard training: 84×84 grayscale, frame-skip 4, 2-frame max-pool, terminal-on-life-loss. |
| `v2` | No-flicker training: sprite-limit removal, max-pool disabled. |
| `v3` | Sticky-action training: v2 semantics + `repeat_action_probability=0.25`. |

`NoFrameskip` variants keep the observation semantics but set action repeat to 1.
Episode summaries (`info["episode"]`) are not emitted by the envs — wrap with
`gymnasium.wrappers.RecordEpisodeStatistics` (the ale-py / Gymnasium convention).

## Building wheels & CI/CD

`.github/workflows/build-wheels.yml` builds optimized abi3 wheels for Linux
x86-64, Windows x86-64, and macOS arm64 on every push to `main`, and attaches
them to the GitHub Release on a `v*` tag. Each wheel is a release build
(thin-LTO, `codegen-units=1`, `-Ctarget-cpu` baseline) with an optional PGO pass
driven by a mapper-diverse workload (`scripts/pgo_workload.py`).

Because the public repo is ROM-free, CI injects ROMs at build time from a
**private** source so released wheels ship complete:

- Secret `ROMS_DEPLOY_KEY` — a read-only SSH deploy key for the private ROM repo.
- Variable `ROMS_REPO` — `owner/name` of that repo (default `<owner>/nesle-roms`).

CI checks out the private repo, stages its `*.nes` into the package ROM home
(`scripts/stage_roms.py`), then builds. When `ROMS_DEPLOY_KEY` is unset (e.g. fork
PRs), CI builds a ROM-free wheel and skips PGO — it never fails on absent ROMs.
To build a complete wheel locally, just have the ROMs in
`crates/nesle-py/python/nesle/roms/` (they are git-ignored) and run
`maturin build --release`.

## License

[GPL-2.0-only](LICENSE). This license covers the source in this repository; it
does not grant any rights to NES ROMs, which are the property of their
respective copyright holders.
