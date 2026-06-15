"""WebSocket agent client for the authoritative ``nesle-server`` console."""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
from typing import Any, Callable

import nesle
import numpy as np
import websockets
from nesle.registration import rom_key_for_env_id

Policy = Callable[[np.ndarray], int]


def parse_state(buf: bytes) -> tuple[dict[str, Any], np.ndarray]:
    """Decode one binary state frame into ``(meta, obs)``."""
    (mlen,) = struct.unpack_from("<I", buf, 0)
    meta = json.loads(bytes(buf[4 : 4 + mlen]))
    off = 4 + mlen + meta["native_w"] * meta["native_h"] * 3
    channels = int(meta.get("obs_channels", 1))
    n = meta["obs_w"] * meta["obs_h"] * channels
    shape = (
        (meta["obs_h"], meta["obs_w"])
        if channels == 1
        else (meta["obs_h"], meta["obs_w"], channels)
    )
    obs = np.frombuffer(buf, dtype=np.uint8, count=n, offset=off).reshape(shape)
    return meta, obs


class AgentClient:
    """A WebSocket agent peer for the authoritative console."""

    def __init__(self, uri: str = "ws://127.0.0.1:8090/ws", name: str | None = None) -> None:
        self.uri = uri
        self.name = name  # display label; the console falls back to "Agent N" if None
        self.client_id: int | None = None
        self.players = 1
        self.actions: list[str] = []
        self.action_masks: list[int] = []
        self.ready_game: str | None = None
        self.ready_env_id: str | None = None

    async def play(
        self,
        *,
        env_id: str,
        policy: Policy | None = None,
        steps: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Run an obs-to-action loop and return the last state metadata."""
        rng = np.random.default_rng(seed)
        nesle.register_envs()
        game_id = rom_key_for_env_id(env_id)
        async with websockets.connect(self.uri, max_size=None) as ws:
            hello = {"t": "hello", "role": "agent", "name": self.name, "env_id": env_id}
            await ws.send(json.dumps(hello))
            await self._await_ready(ws)
            if self.ready_game != game_id:
                raise RuntimeError(
                    f"server loaded {self.ready_game!r}, but agent env_id belongs to {game_id!r}"
                )

            last: dict[str, Any] = {}
            seen = 0
            while steps is None or seen < steps:
                msg = await ws.recv()
                if isinstance(msg, str):
                    self._on_text(json.loads(msg))
                    continue
                last, obs = parse_state(msg)
                seen += 1
                if policy is not None:
                    idx = int(policy(obs))
                elif self.action_masks:
                    idx = int(rng.integers(0, len(self.action_masks)))
                else:
                    idx = 0
                mask = self.action_masks[idx] if self.action_masks else 0
                await ws.send(json.dumps({"t": "action", "mask": int(mask)}))
            return last

    async def _await_ready(self, ws, timeout: float | None = None) -> bool:
        # The action space arrives as a `ready` text message from the human-owned console.
        while not self.action_masks:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout) if timeout else await ws.recv()
            except asyncio.TimeoutError:
                return False
            if isinstance(msg, str):
                self._on_text(json.loads(msg))
        return True

    def _on_text(self, m: dict[str, Any]) -> None:
        t = m.get("t")
        if t == "welcome":
            self.client_id = int(m["client_id"])
        elif t == "ready":
            self.ready_env_id = str(m.get("env_id", ""))
            self.ready_game = str(m.get("game", ""))
            self.actions = list(m.get("actions", []))
            self.action_masks = list(m.get("action_masks", []))
            self.players = int(m.get("players", 1))


def _main() -> None:
    ap = argparse.ArgumentParser(description="nesle-server agent player-client")
    ap.add_argument("--uri", default="ws://127.0.0.1:8090/ws")
    ap.add_argument("--name", default=None, help="display label shown in UIs (else 'Agent N')")
    ap.add_argument("--env-id", required=True, help="Gymnasium env id declaring the requested obs")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    meta = asyncio.run(
        AgentClient(uri=args.uri, name=args.name).play(
            env_id=args.env_id,
            steps=args.steps,
            seed=args.seed,
        )
    )
    print("last state:", json.dumps(meta))


if __name__ == "__main__":
    _main()
