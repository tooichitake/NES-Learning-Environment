//! Typed server-to-client JSON payloads.

use serde::Serialize;

#[derive(Serialize)]
pub struct LevelInfo {
    pub id: String,
    pub label: String,
}

#[derive(Serialize)]
pub struct GameInfo {
    pub id: &'static str,
    pub gym_id: &'static str,
    /// Game name only (no mode/port suffix), e.g. `"Bomberman 2"`. Drives the
    /// front-end's Game -> Mode -> Level menu grouping (specs sharing a
    /// `display_name` are the same game's modes).
    pub display_name: &'static str,
    /// In-game mode token (the title-screen selection: `"Normal"`/`"VS"`/`"Battle"`,
    /// `"1P"`/`"2P"`/`"4P"`, ...); `None` for single-mode games (SMB, Pac-Man).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<&'static str>,
    pub players: u8,
    pub sha1: &'static str,
    pub levels: Vec<LevelInfo>,
}

#[derive(Serialize)]
pub struct WelcomeMsg {
    pub t: &'static str,
    pub client_id: u64,
    pub games: Vec<GameInfo>,
}

#[derive(Serialize)]
pub struct PeerInfo {
    pub id: u64,
    pub role: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_id: Option<String>,
}

#[derive(Serialize)]
pub struct PortOwner {
    pub client_id: u64,
    pub role: String,
    pub name: String,
}

#[derive(Serialize)]
pub struct PortsMsg {
    pub t: &'static str,
    pub players: u8,
    pub ports: usize,
    pub peers: Vec<PeerInfo>,
    pub owners: Vec<Option<PortOwner>>,
}

#[derive(Serialize)]
pub struct ErrorMsg {
    pub t: &'static str,
    pub code: &'static str,
    pub message: String,
}

#[derive(Serialize)]
pub struct RecordingMsg {
    pub t: &'static str,
    pub game: &'static str,
    pub frameskip: u8,
    pub players: u8,
    pub frames: usize,
    pub actions: serde_json::Value,
}

#[derive(Serialize)]
pub struct RamMsg {
    pub t: &'static str,
    pub game: &'static str,
    pub bytes: usize,
    pub b64: String,
}

pub fn level_options(game_id: &str) -> Vec<LevelInfo> {
    let mut levels = vec![LevelInfo {
        id: "title".to_string(),
        label: "Title".to_string(),
    }];
    levels.extend(
        nesle_rl::available_start_state_ids(game_id)
            .into_iter()
            .filter_map(|state_id| {
                nesle_rl::env_suffix_for_start_state(game_id, &state_id).map(|id| LevelInfo {
                    label: id.clone(),
                    id,
                })
            }),
    );
    levels
}

pub fn game_roster() -> Vec<GameInfo> {
    // One entry per spec; player-count and scoring modes are separate specs sharing a display_name.
    let mut roster = Vec::new();
    for g in nesle_rl::games::registry::all_games() {
        roster.push(GameInfo {
            id: g.id,
            gym_id: g.gym_id,
            display_name: g.display_name,
            mode: g.mode,
            players: g.players,
            sha1: g.sha1,
            levels: level_options(g.id),
        });
    }
    roster
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welcome_payload_shape_is_stable() {
        let msg = WelcomeMsg {
            t: "welcome",
            client_id: 42,
            games: vec![GameInfo {
                id: "super_c_2p",
                gym_id: "NESLE/SuperC-2P-v0",
                display_name: "Super C",
                mode: Some("2P"),
                players: 2,
                sha1: "abc",
                levels: vec![LevelInfo {
                    id: "2".to_string(),
                    label: "2".to_string(),
                }],
            }],
        };
        let value = serde_json::to_value(msg).unwrap();
        assert_eq!(value["t"], "welcome");
        assert_eq!(value["client_id"], 42);
        assert_eq!(value["games"][0]["levels"][0]["id"], "2");
    }

    #[test]
    fn ports_payload_shape_is_stable() {
        let msg = PortsMsg {
            t: "ports",
            players: 2,
            ports: 4,
            peers: vec![PeerInfo {
                id: 7,
                role: "human".to_string(),
                name: "Alice".to_string(),
                env_id: None,
            }],
            owners: vec![
                Some(PortOwner {
                    client_id: 7,
                    role: "human".to_string(),
                    name: "Alice".to_string(),
                }),
                None,
            ],
        };
        let value = serde_json::to_value(msg).unwrap();
        assert_eq!(value["t"], "ports");
        assert_eq!(value["owners"][0]["client_id"], 7);
        assert_eq!(value["owners"][0]["name"], "Alice");
        assert!(value["owners"][1].is_null());
    }
}
