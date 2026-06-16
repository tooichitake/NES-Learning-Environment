//! Wire protocol, client -> server (JSON text). Server -> client is: a one-shot
//! `welcome` JSON (this connection's `client_id`) on connect, a `ready` JSON
//! (action space + native size + player count), a `ports` JSON whenever peers or
//! controller assignments change, plus a per-frame binary `state` message
//! encoded in `session.rs`.

use serde::Deserialize;

#[derive(Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum ClientMsg {
    /// Identity handshake sent once on connect by every peer.
    Hello {
        #[serde(default)]
        role: String,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        env_id: Option<String>,
    },
    /// Rename this peer. The server owns the displayed peer list.
    Rename { name: String },
    /// Load a registered game. `env_id` is the Gymnasium id and the only public
    /// selector; the console resolves it to a GameSpec. `bytes_b64` is OPTIONAL: when
    /// empty (the default), the server auto-loads the packaged ROM (resolved by the
    /// spec's sha1 from `nesle/roms/`); a non-empty value is an explicit upload
    /// override. If auto-load finds no packaged ROM, the server replies with an
    /// `error` (`code = "rom_required"`) so the UI can fall back to Upload ROM.
    LoadRom {
        env_id: String,
        #[serde(default)]
        bytes_b64: String,
    },
    /// Set the latest action mask. Browser peers include a port because one
    /// browser can drive multiple controllers; agent peers omit it and use their
    /// assigned controller.
    Action {
        #[serde(default)]
        port: Option<u8>,
        mask: u8,
    },
    /// Assign a client to a controller with a unique player name, or pass
    /// `client_id:null` to free it.
    AssignPort {
        port: u8,
        #[serde(default)]
        client_id: Option<u64>,
        #[serde(default)]
        name: Option<String>,
    },
    /// Live controls: run/pause, reset, and observation/preprocessing knobs.
    /// `rl_mode` switches the obs to true frame-skip windowing: the obs + per-step
    /// reward refresh once per `frame_skip` frames (max-pooling the window's last
    /// two when `maxpool`), while native RGB still streams every frame for a smooth
    /// human view. When `rl_mode` is false (Play mode) the obs is computed every
    /// frame as before.
    Settings {
        #[serde(default)]
        running: Option<bool>,
        #[serde(default)]
        reset: Option<bool>,
        #[serde(default)]
        obs_size: Option<usize>,
        #[serde(default)]
        rl_mode: Option<bool>,
        #[serde(default)]
        step_mode: Option<bool>,
        #[serde(default)]
        frame_skip: Option<usize>,
        #[serde(default)]
        maxpool: Option<bool>,
        // Live Debug preprocessing knobs (observation-affecting + behavior).
        #[serde(default)]
        remove_sprite_limit: Option<bool>,
        #[serde(default)]
        obs_rgb: Option<bool>,
        // Debug behavior knobs; life-loss terminal is a preprocessing policy, not a GameSpec signal.
        #[serde(default)]
        terminal_on_life_loss: Option<bool>,
        #[serde(default)]
        sticky_prob: Option<f32>,
        #[serde(default)]
        noop_max: Option<usize>,
        #[serde(default)]
        clip_pos: Option<f32>,
        #[serde(default)]
        clip_neg: Option<f32>,
    },
    /// Advance ONE agent step in Debug `step_mode` with one action `mask` PER
    /// controller port: `masks[i]` drives port `i`. A multi-agent env steps all ports
    /// together (the env requires one action per port, any order, then one step); a
    /// single-agent env sends a 1-element list. The console steps `frame_skip` frames
    /// (masks repeated), windows the obs, and broadcasts once. No auto-advance in step
    /// mode -- the action space drives time. Trailing ports default to NOOP.
    Step {
        #[serde(default)]
        masks: Vec<u8>,
    },
    /// Start (resets the episode first) or stop gameplay recording. Stopping emits
    /// a `recording` message (replay-format input trace) to download.
    Record {
        #[serde(default)]
        on: bool,
    },
    /// Request the current 2 KB work RAM (emitted as a `ram` message) -- the
    /// reverse-engineering aid for mapping a game's score/lives/state addresses.
    DumpRam,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_rom_accepts_env_id_only() {
        // No bytes_b64 -> auto-load request (default empty).
        let msg: ClientMsg =
            serde_json::from_str(r#"{"t":"load_rom","env_id":"NESLE/SuperC-1P-2-v0"}"#).unwrap();
        match msg {
            ClientMsg::LoadRom { env_id, bytes_b64 } => {
                assert_eq!(env_id, "NESLE/SuperC-1P-2-v0");
                assert!(bytes_b64.is_empty());
            }
            _ => panic!("expected load_rom"),
        }
    }

    #[test]
    fn agent_hello_declares_identity_and_env() {
        let msg: ClientMsg =
            serde_json::from_str(
                r#"{"t":"hello","role":"agent","name":"SB3 PPO","env_id":"NESLE/SuperMarioBros-1-1-v2"}"#,
            )
            .unwrap();
        match msg {
            ClientMsg::Hello { role, name, env_id } => {
                assert_eq!(role, "agent");
                assert_eq!(name.as_deref(), Some("SB3 PPO"));
                assert_eq!(env_id.as_deref(), Some("NESLE/SuperMarioBros-1-1-v2"));
            }
            _ => panic!("expected hello"),
        }
    }

    #[test]
    fn settings_accepts_reset() {
        let msg: ClientMsg = serde_json::from_str(r#"{"t":"settings","reset":true}"#).unwrap();
        match msg {
            ClientMsg::Settings { reset, .. } => {
                assert_eq!(reset, Some(true));
            }
            _ => panic!("expected settings"),
        }
    }

    #[test]
    fn assign_port_names_peer_port_and_player() {
        let msg: ClientMsg =
            serde_json::from_str(r#"{"t":"assign_port","client_id":7,"port":1,"name":"Alice"}"#)
                .unwrap();
        match msg {
            ClientMsg::AssignPort {
                client_id,
                port,
                name,
            } => {
                assert_eq!(client_id, Some(7));
                assert_eq!(port, 1);
                assert_eq!(name.as_deref(), Some("Alice"));
            }
            _ => panic!("expected assign_port"),
        }
    }

    #[test]
    fn assign_port_without_peer_releases() {
        let msg: ClientMsg =
            serde_json::from_str(r#"{"t":"assign_port","client_id":null,"port":1}"#).unwrap();
        match msg {
            ClientMsg::AssignPort {
                client_id,
                port,
                name,
            } => {
                assert_eq!(client_id, None);
                assert_eq!(port, 1);
                assert_eq!(name, None);
            }
            _ => panic!("expected assign_port"),
        }
    }
}
