#!/usr/bin/env python3
"""PGO profile workload for a DISTRIBUTION (general, multi-game) `nesle` wheel.

The profile must cover the game-agnostic core AND multiple mapper-dispatch arms,
so PGO does not bias to one mapper. So:
  1. Step the raw `NesInterface` core across a mapper-diverse ROM set (NROM / MMC1 /
     UxROM / MMC3 / MMC5 / AxROM / MMC2 / Sunsoft FME-7) -- exercises CPU/PPU/APU
     plus each mapper's bank-switch / IRQ path.
  2. Run the SMB1 preprocessed gray + RGB vector path (grayscale LUT + INTER_AREA
     resize SIMD + rayon vector + PyO3 boundary) -- game-agnostic, exercised once.
ROMs that fail to load (unsupported mapper on this build) are skipped, so the
workload never crashes. argv[1] (a ROM path) is accepted but ignored.
"""

import os

import gymnasium as gym
import nesle  # noqa: F401
import numpy as np

if hasattr(nesle, "register_envs"):
    nesle.register_envs()

ROMS = "crates/nesle-py/python/nesle/roms"

# One ROM per supported mapper (best-effort; any that fail to load are skipped).
MAPPER_ROMS = [
    "super-mario-bros.nes",  # NROM          (0)
    "Legend of Zelda, The (USA) (Rev 1).nes",  # MMC1          (1)
    "Mega Man (USA).nes",  # UxROM         (2)
    "Solomon's Key (USA).nes",  # CNROM         (3)
    "Super Mario Bros. 3 (USA) (Rev 1).nes",  # MMC3          (4)
    "Mega Man 3 (USA).nes",  # MMC3          (4, extra)
    "Castlevania III - Dracula's Curse (USA).nes",  # MMC5          (5)
    "Battletoads (USA).nes",  # AxROM         (7)
    "Mike Tyson's Punch-Out!! (Japan, USA) (Rev 1).nes",  # MMC2          (9)
    "Batman - Return of the Joker (USA).nes",  # Sunsoft FME-7 (69)
]
# (Mappers verified from iNES headers; covers all 9 supported: 0/1/2/3/4/5/7/9/69.)
_MASKS = np.array([0x00, 0x80, 0x82, 0x81, 0x83, 0x01, 0x40, 0x08, 0x10, 0x20], dtype=np.uint8)


def _core(rom_name, frames, seed):
    from nesle._nesle import NesInterface

    path = os.path.join(ROMS, rom_name)
    if not os.path.exists(path):
        print(f"  miss {rom_name}", flush=True)
        return
    try:
        nes = NesInterface()
        nes.load_rom_bytes(open(path, "rb").read())
    except Exception as e:  # noqa: BLE001  (unsupported mapper -> skip, don't crash)
        print(f"  skip {rom_name}: {e}", flush=True)
        return
    rng = np.random.default_rng(seed)
    nes.act_n(rng.choice(_MASKS, size=frames).tolist())  # one Rust call, `frames` frames
    print(f"  core {rom_name} x{frames}", flush=True)


def _preprocess(env_id, num_envs, steps, seed):
    env = gym.make_vec(
        env_id, num_envs=num_envs, rom_path=os.path.join(ROMS, "super-mario-bros.nes")
    )
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    n = env.single_action_space.n
    for _ in range(steps):
        env.step(rng.integers(0, n, size=num_envs))
    env.close()


def _single(env_id, steps, seed):
    env = gym.make(env_id, rom_path=os.path.join(ROMS, "super-mario-bros.nes"))
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    n = env.action_space.n
    for _ in range(steps):
        _o, _r, term, trunc, _i = env.step(int(rng.integers(0, n)))
        if term or trunc:
            env.reset()
    env.close()
    print(f"  single {env_id} x{steps}", flush=True)


def _async(rounds, seed):
    from nesle._nesle import NesEnv, NesVectorEnv

    n = len(NesEnv("super_mario_bros").minimal_action_set())
    rom = open(os.path.join(ROMS, "super-mario-bros.nes"), "rb").read()
    env = NesVectorEnv(
        12,
        game_id="super_mario_bros",
        rom=rom,
        obs_mode="preprocessed",
        batch_size=4,
        num_threads=4,
        frame_skip=4,
        width=84,
        height=84,
        maxpool=False,
        terminal_on_life_loss=True,
        repeat_action_probability=0.25,
        noop_max=30,
        seed=seed,
        max_episode_frames=None,
        autoreset_mode="SameStep",
    )
    env.async_reset()
    rng = np.random.default_rng(seed)
    done = 0
    while done < rounds:
        _obs, env_ids, _r, _t, _tr, _f, _e, _l = env.async_recv()
        env.async_send(list(env_ids), [int(rng.integers(0, n)) for _ in env_ids])
        done += len(env_ids)
    print(f"  async x{rounds}", flush=True)


def main():
    for i, rom in enumerate(MAPPER_ROMS):
        _core(rom, 600, i + 1)  # mapper-diverse core
    # Keep each pass small (instrumented builds ~10-20x slower): PGO needs each path HIT, not volume.
    _preprocess(
        "NESLE/SuperMarioBros-1-1-v2", 4, 400, 100
    )  # sync vector + gray 84x84 + resize SIMD
    _preprocess("NESLE/SuperMarioBros-1-1-v0", 2, 150, 200)  # raw RGB palette path
    # Other training paths (skipped gracefully if the API differs):
    for name, fn in (
        ("single", lambda: _single("NESLE/SuperMarioBros-1-1-v2", 400, 300)),
        ("async", lambda: _async(800, 400)),
    ):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  skip {name}: {e}", flush=True)
    print("pgo workload done", flush=True)


if __name__ == "__main__":
    main()
