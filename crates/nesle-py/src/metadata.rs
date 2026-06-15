use nesle_rl::games::registry;
use pyo3::prelude::*;

#[pyfunction]
pub(crate) fn backend_id() -> &'static str {
    "rust-native"
}

#[pyfunction]
pub(crate) fn game_metadata() -> Vec<(String, String, String, String, u8)> {
    registry::all_games()
        .iter()
        .map(|g| {
            (
                g.id.to_string(),
                g.gym_id.to_string(),
                g.display_name.to_string(),
                g.sha1.to_string(),
                g.players,
            )
        })
        .collect()
}

#[pyfunction]
pub(crate) fn start_state_metadata() -> Vec<(String, String, String)> {
    registry::all_games()
        .iter()
        .map(|g| g.id)
        .flat_map(|game_id| {
            nesle_rl::available_start_state_ids(game_id)
                .into_iter()
                .filter_map(move |state_id| {
                    nesle_rl::env_suffix_for_start_state(game_id, &state_id)
                        .map(|suffix| (game_id.to_string(), state_id, suffix))
                })
        })
        .collect()
}
