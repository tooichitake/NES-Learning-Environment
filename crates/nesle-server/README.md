# nesle-server

`nesle-server` hosts one authoritative NES console session and serves the browser
thin UI. It is used for human play, Debug inspection, Agent sessions,
recording, and RAM dumps. Training does not depend on the browser round trip.

## Structure

```text
src/
  main.rs       startup, env vars, router, static file service
  client.rs     WebSocket lifecycle and client message routing
  protocol.rs   client JSON message schema
  session.rs    console thread, reset/start-state, ports, recording, frames
  wire.rs       typed server JSON payloads and game/start-state metadata
web/
  index.html    DOM shell
  styles.css    macOS-like responsive UI
  app.js        app initialization, WebSocket state, controls, input
  js/
    dom.js
    downloads.js
    panels.js
    rendering.js
    select.js
```

There is no npm or frontend build chain. `index.html` loads `app.js` as an ES
module, and the server serves the directory with `tower_http::ServeDir`.

## Run

```powershell
cargo run -p nesle-server
```

With audio:

```powershell
cargo run -p nesle-server --features audio-synth
```

Environment variables:

- `NESLE_SERVER_ADDR`: bind address, default `127.0.0.1:8090`.
- `NESLE_WEB_DIR`: static UI directory, default `crates/nesle-server/web`.

## Invariants

- The browser is a thin client. It never implements emulator, reward,
  preprocessing, cheat/password, replay, or start-state restore semantics.
- Episode resets restore formal `.state` assets through the env path.
- Play mode uses the selected env id and restores its Level state on load/reset.
- Play and Agent are human-owned console modes. `Title` starts from the
  normal title/menu path; specific levels restore `.state` assets.
- Native game output stays 256 x 240 CSS pixels.
- Agent/RL obs canvases use streamed dimensions at 1:1 CSS pixels.
- Labels, owners, actions, levels, rewards, lives, and env descriptors are
  server-authoritative.

## Validate

```powershell
cargo check -p nesle-server
cargo test -p nesle-server
cargo clippy -p nesle-server --all-targets -- -D warnings
python scripts/validate_serve_ui_layout.py --url http://127.0.0.1:8090 --quick
```

When changes touch env/reset/start-state behavior:

```powershell
cargo test -p nesle-rl
```

More detail:

- `docs/server.md`: runtime and WebSocket protocol.
- `docs/viewer.md`: browser UI behavior.
- `.agents/skills/rust-nesle-server-ui`: Codex workflow rules.
- `.claude/skills/rust-nesle-server-ui`: Claude Code workflow rules.
