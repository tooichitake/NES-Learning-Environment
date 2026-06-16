# NESLE: NES Learning Environment

[![build-wheels](https://github.com/tooichitake/nesle/actions/workflows/build-wheels.yml/badge.svg)](https://github.com/tooichitake/nesle/actions/workflows/build-wheels.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

NESLE is a Rust-native reinforcement learning environment for the Nintendo
Entertainment System (NES). It provides a cycle-accurate NES core, game-specific
RL environments, preprocessing, vectorized stepping, and Python bindings through
a single compiled extension.

NESLE implements the standard Gymnasium interface for single-agent environments
and the PettingZoo interface for multi-agent environments, so it works with
common RL libraries such as Stable-Baselines3 and CleanRL.

```python
import gymnasium as gym
import nesle

gym.register_envs(nesle)

env = gym.make("NESLE/SuperMarioBros-1-1-v3")   # release wheels bundle ROMs (no rom_path)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

![NES games rendered by NESLE](assets/games.png)

<sub>NESLE's Rust NES core rendering a sample of its supported games.</sub>

## Features

- Rust-native, deterministic NES core verified against a reference emulator.
- Gymnasium single-agent environments and PettingZoo multi-agent environments.
- RGB, grayscale, and raw RAM observations.
- ALE-style preprocessing with frame skip, max-pooling, frame stacking, no-op
  resets, and sticky actions.
- Rust vector environments with synchronous and asynchronous stepping.
- Direct RAM access, PPU tile-field access, and emulator state save/restore.
- abi3 wheels for Python 3.10+.
- Browser viewer and WebSocket server for play, debugging, and agent monitoring.

## Installation

### From a release wheel

Prebuilt abi3 wheels are attached to each
[GitHub Release](https://github.com/tooichitake/nesle/releases) for Linux x86-64,
Windows x86-64, and macOS arm64.

```bash
pip install nesle-<version>-cp310-abi3-<platform>.whl
```

### From source

Requires a Rust toolchain (1.85+) and [maturin](https://www.maturin.rs/).

```bash
pip install maturin
maturin develop --release          # build + install into the active environment
# optional features:
#   --features viewer        # optional SDL2 native window for render_mode="human"
#   --features audio-synth   # APU audio synthesis (off by default for training)
```

## ROMs

NES ROMs are copyrighted and are not stored in the public source tree. NESLE
resolves ROMs by SHA-1, independent of filename.

Release wheels bundle ROMs for the supported games, so `gym.make("NESLE/<id>")`
works without `rom_path`. Source builds are ROM-free by default. To bundle ROMs
into a local build, place `.nes` files under
`crates/nesle-py/python/nesle/roms/` before running `maturin develop` or
`maturin build`. You can also resolve ROMs at runtime.

Runtime resolution order:

1. An explicit `rom_path=` argument.
2. A ROM bundled in the installed wheel.
3. A folder registered with `nesle.import_roms(...)`, or pointed at by the
   `NESLE_ROMS_DIR` environment variable.

```python
import gymnasium as gym
import nesle

gym.register_envs(nesle)

# Release wheel: ROMs are bundled, so no rom_path is needed.
env = gym.make("NESLE/SuperMarioBros-1-1-v3")

# Custom or source-build ROMs: register a folder once.
nesle.import_roms("/path/to/roms")
# Or set NESLE_ROMS_DIR=/path/to/roms for the process.
# Or pass rom_path directly to gym.make(...).

nesle.get_all_game_ids()       # every supported game id
nesle.get_rom_path(game_id)    # bundled ROM path, if present
```

## Usage

NESLE has single-agent and multi-agent APIs, each with a vectorized variant.

| | Non-vectorized | Vectorized |
|---|---|---|
| **Single-agent** (Gymnasium) | `gym.make("NESLE/<id>")` | `gym.make_vec("NESLE/<id>", num_envs=N)` |
| **Multi-agent** (PettingZoo) | `nesle.env.parallel_env(env_id="NESLE/<id>")` | `make_multiplayer_vector_env("NESLE/<id>", num_envs=N)` |

### Single-agent Gymnasium

```python
import gymnasium as gym
import nesle

gym.register_envs(nesle)

# Raw env (v0): pick the observation type.
raw = gym.make("NESLE/SuperMarioBros-1-1-v0", obs_type="rgb")        # or "grayscale" or "ram"

# Standard training env (v3): 112x112 grayscale, frame skip 4, sticky actions.
env = gym.make("NESLE/SuperMarioBros-1-1-v3")
obs, info = env.reset(seed=0)
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

### Vectorized Gymnasium

```python
import gymnasium as gym
import nesle

gym.register_envs(nesle)

# Synchronous vector env.
vec = gym.make_vec("NESLE/SuperMarioBros-1-1-v3", num_envs=8)
batch_obs, infos = vec.reset()                       # (8, 4, 112, 112)

# Asynchronous vector env: 0 < batch_size < num_envs.
avec = gym.make_vec("NESLE/SuperMarioBros-1-1-v3", num_envs=16, batch_size=4)
avec.async_reset()
obs, rewards, terms, truncs, info = avec.recv()      # info["env_id"] demuxes
avec.send(actions)                                   # one action per env in the recv batch
```

### Multi-agent PettingZoo

```python
import nesle
from nesle.env import parallel_env

env = parallel_env(env_id="NESLE/SuperC-2P-2-v3")
obs, infos = env.reset()
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, terminations, truncations, infos = env.step(actions)
```

### Vectorized self-play

```python
import numpy as np
from nesle.vector_env import make_multiplayer_vector_env

# K parallel matches, each with `num_players` controller ports.
venv = make_multiplayer_vector_env("NESLE/Bomberman2-VS-1-v3", num_envs=16)
obs, infos = venv.reset()                            # (num_envs * num_players, 4, 112, 112)

# One action per agent slot: slot = unit * num_players + port.
actions = np.random.randint(venv.num_actions, size=venv.num_agents)
obs, rewards, dones, truncated, infos = venv.step(actions)
```

## Environments

Env ids follow `NESLE/<game>[-<mode>]-<level>-v<version>`. The optional `<mode>`
token names the cart mode or player count and is absent for single-mode games.
Bomberman 2 uses `Normal`, `VS`, and `Battle`; Super C and Ice Hockey use `1P`
and `2P`. Example ids include `SuperMarioBros-1-1-v3`, `SuperC-2P-2-v3`, and
`Bomberman2-VS-1-v3`. Call `nesle.get_all_game_ids()` for the registered ids.
The version suffix selects the observation and preprocessing profile:

| Version | Profile |
|---|---|
| `v0` | Raw observation (`obs_type` in `rgb`, `grayscale`, or `ram`), configurable action repeat. |
| `v1` | 112x112 grayscale, frame skip 4, 2-frame max-pool, terminal-on-life-loss. |
| `v2` | Same as v1 with sprite-limit removal and max-pool disabled. |
| `v3` | v2 + sticky actions (`repeat_action_probability = 0.25`). |

`v3` is the recommended training profile and matches the ALE sticky-action
setting. `NoFrameskip` variants keep the same observation semantics with action
repeat set to 1. Episode summaries (`info["episode"]`) are produced by
Gymnasium's `RecordEpisodeStatistics` wrapper.

![Observation pipeline: raw RGB to grayscale to 112x112 obs](assets/obs_pipeline.png)

<sub>The preprocessing pipeline: native RGB to grayscale to 112x112 training observation.</sub>

## Supported games

20 games ship today. Build an env id as `NESLE/<stem>-<level>-v3`; call
`nesle.get_all_game_ids()` for the authoritative list at runtime, and
`nesle.parse_env_id(env_id)` to inspect one.

**Single-agent (Gymnasium):**
SuperMarioBros, SuperMarioBros2, SuperMarioBros3, KungFu, Castlevania,
SuperC-1P, AdventureIsland, DuckTales, MegaMan2, PacMan, MarioBros,
Bomberman, Bomberman2-Normal, IceHockey-1P

**Multi-agent (PettingZoo):**
SuperC-2P, IceHockey-2P, Bomberman2-VS, Bomberman2-Battle, RCProAm2-4P,
Roundball2on2-4P

## Memory & state interface

NESLE exposes live machine state for reward shaping, scripted agents, RAM-map
reverse engineering, and debugging.

```python
import gymnasium as gym
import nesle

gym.register_envs(nesle)

env = gym.make("NESLE/SuperMarioBros-1-1-v0", obs_type="ram")
obs, info = env.reset()          # obs is the 2048-byte NES RAM (uint8)

# Read RAM at any time, independent of obs_type:
ram = env.unwrapped.get_ram()    # np.uint8, shape (2048,)
value = int(ram[0x075A])         # e.g. some game-specific address

# Other live-state accessors on the single-agent env:
env.unwrapped.get_screen_grayscale()   # (240, 256) uint8
env.unwrapped.get_action_meanings()    # ["NOOP", "RIGHT", ...]

# Save and restore the full emulator state:
snap = env.unwrapped.clone_state()
env.unwrapped.restore_state(snap)      # also: restore_state_blob(bytes)
```

For batched and multi-agent runs, `NESMultiPlayerVectorEnv` exposes per-match
memory directly:

```python
from nesle.vector_env import make_multiplayer_vector_env
venv = make_multiplayer_vector_env("NESLE/Bomberman2-VS-1-v3", num_envs=16)
venv.reset()
ram = venv.get_ram()          # (num_envs, 2048) CPU RAM
field = venv.get_nametable()  # (num_envs, vram) PPU tile field
```

## Server and viewer

`nesle-server` is a WebSocket console host with a browser viewer. It can run any
supported game, stream frames to the browser, accept controller input, and step
RL environments frame by frame. Emulation, reward logic, preprocessing, and
start-state handling stay in Rust.

![NESLE browser viewer: Play, Debug, and Agent modes](assets/server-ui.png)

<sub>The `nesle-server` browser viewer in Play, Debug, and Agent modes.</sub>

The viewer has three modes:

- Play: 60 Hz play with keyboard controllers for 1 to 4 players.
- Debug: step the preprocessed env and inspect observation, reward, lives, and
  terminal flags.
- Agent: connect RL agents as peer players and monitor per-agent observations.

It also supports replay recording and RAM dumps.

```bash
cargo run -p nesle-server                          # then open http://127.0.0.1:8090
cargo run -p nesle-server --features audio-synth   # with APU audio
```

| Env var | Default | Meaning |
|---|---|---|
| `NESLE_SERVER_ADDR` | `127.0.0.1:8090` | HTTP and WebSocket bind address |
| `NESLE_WEB_DIR` | `crates/nesle-server/web` | static UI directory |

Pick a game and level in the UI and the server loads the packaged ROM by SHA-1.
Use Upload ROM for unregistered ROMs.

### Connecting an RL agent (`nesle.agent_client`)

`Agent` mode accepts external RL agents over WebSocket. With a game loaded in the
browser, connect an agent peer that receives observations and sends actions:

```python
import asyncio
from nesle.agent_client import AgentClient

def policy(obs):                  # np.uint8, (H, W) or (H, W, channels)
    return 0                      # action index

asyncio.run(AgentClient(name="my-agent").play(
    env_id="NESLE/SuperMarioBros-1-1-v3",   # declares the agent's observation profile
    policy=policy,                           # steps defaults to None -> runs until disconnected
))
```

The CLI client sends random actions when no policy is provided:

```bash
python -m nesle.agent_client --env-id NESLE/SuperMarioBros-1-1-v3
```

The agent client requires the `agent` extra (`pip install nesle[agent]`) and a
running `nesle-server` with the matching game loaded.

## Citing

If you use NESLE in your research, please cite it:

```bibtex
@software{zhao_nesle,
  author = {Zhao, Zhiyuan},
  title  = {NESLE: The NES Learning Environment},
  url    = {https://github.com/tooichitake/nesle},
  year   = {2026},
}
```

NESLE mirrors the Arcade Learning Environment's conventions. If you use the
sticky-action profiles (`v3`, `repeat_action_probability = 0.25`), please also cite
the ALE papers:

```bibtex
@article{bellemare13arcade,
  author  = {{Bellemare}, M.~G. and {Naddaf}, Y. and {Veness}, J. and {Bowling}, M.},
  title   = {The Arcade Learning Environment: An Evaluation Platform for General Agents},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {47},
  pages   = {253--279},
  year    = {2013},
}

@article{machado18arcade,
  author  = {Marlos C. Machado and Marc G. Bellemare and Erik Talvitie and Joel Veness and Matthew J. Hausknecht and Michael Bowling},
  title   = {Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {61},
  pages   = {523--562},
  year    = {2018},
}
```

## Acknowledgments

NESLE follows conventions from related RL environment projects and is verified
against prior emulator work:

- **[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)** and
  **[PettingZoo](https://github.com/Farama-Foundation/PettingZoo)** for the
  single-agent and multi-agent interfaces.
- **Arcade Learning Environment** (Bellemare et al. 2013; Machado et al. 2018)
  for observation, preprocessing, and sticky-action conventions.
- **gym-super-mario-bros and nes-py** (Christian Kauten) for single-agent NES
  reward conventions.
- **[Mesen2](https://github.com/SourMesen/Mesen2)** as the reference emulator
  for RAM and framebuffer verification.

## License

Copyright (C) 2026 Zhiyuan Zhao.

NESLE is licensed under [GPL-3.0-only](LICENSE). This license covers the source in
this repository; it grants no rights to NES ROMs, which belong to their respective
copyright holders.
