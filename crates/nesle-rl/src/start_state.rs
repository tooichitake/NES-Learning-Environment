use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use nesle_common::{NesleError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartStateId(String);

impl StartStateId {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.is_empty() {
            return Err(NesleError::InvalidState(
                "start_state id must not be empty".to_string(),
            ));
        }
        if value == "default" || value == "random" {
            return Err(NesleError::InvalidState(format!(
                "{value:?} is a reserved start_state id"
            )));
        }
        if value.contains(['/', '\\']) || value.contains("..") {
            return Err(NesleError::InvalidState(format!(
                "start_state id must be a file stem, got {value:?}"
            )));
        }
        if let Some(rest) = value.strip_prefix("level_") {
            let has_leading_zero = rest
                .split('_')
                .any(|part| part.len() > 1 && part.starts_with('0'));
            if has_leading_zero {
                return Err(NesleError::InvalidState(format!(
                    "level start-state ids must not use leading zeroes, got {value:?}"
                )));
            }
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartState {
    FirstAvailable,
    Id(StartStateId),
    Random,
    Path(PathBuf),
}

impl StartState {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "random" => Ok(Self::Random),
            other => Ok(Self::Id(StartStateId::new(other)?)),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StartStateBlob {
    pub bytes: Vec<u8>,
}

pub(crate) fn game_start_state_dir(game_id: &str) -> PathBuf {
    let root = if let Ok(root) = std::env::var("NESLE_START_STATES_DIR") {
        PathBuf::from(root)
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("start_states")
    };
    // Multi-mode families (family != id, e.g. Bomberman 2) nest as `<family>/<mode>/`; single-spec games stay flat at `<id>/`.
    if let Some(game) = crate::games::registry::find_game(game_id) {
        if game.family != game.id {
            if let Some(mode) = game.mode {
                return root.join(game.family).join(mode.to_lowercase());
            }
        }
    }
    root.join(game_id)
}

pub(crate) fn load_start_state_blob(game_id: &str, id: &StartStateId) -> Result<StartStateBlob> {
    let path = game_start_state_dir(game_id).join(format!("{}.state", id.as_str()));
    read_start_state_path(&path).map(|bytes| StartStateBlob { bytes })
}

pub(crate) fn load_start_state_path(path: &Path) -> Result<StartStateBlob> {
    read_start_state_path(path).map(|bytes| StartStateBlob { bytes })
}

pub(crate) fn load_first_start_state_blob(game_id: &str) -> Result<StartStateBlob> {
    let mut paths = list_level_state_paths(game_id).map_err(|err| {
        NesleError::InvalidState(format!(
            "start-state reset needs .state files under {}; {err}",
            game_start_state_dir(game_id).display()
        ))
    })?;
    let Some(path) = paths.drain(..).next() else {
        return Err(NesleError::InvalidState(format!(
            "no level_*.state files found under {}",
            game_start_state_dir(game_id).display()
        )));
    };
    read_start_state_path(&path).map(|bytes| StartStateBlob { bytes })
}

pub(crate) fn load_random_start_state_blobs(game_id: &str) -> Result<Vec<StartStateBlob>> {
    let paths = list_level_state_paths(game_id).map_err(|err| {
        NesleError::InvalidState(format!(
            "start_state='random' needs .state files under {}; {err}",
            game_start_state_dir(game_id).display()
        ))
    })?;
    if paths.is_empty() {
        return Err(NesleError::InvalidState(format!(
            "no level_*.state files found under {} for start_state='random'",
            game_start_state_dir(game_id).display()
        )));
    }
    paths
        .into_iter()
        .map(|path| read_start_state_path(&path).map(|bytes| StartStateBlob { bytes }))
        .collect()
}

pub fn available_start_state_ids(game_id: &str) -> Vec<String> {
    list_level_state_paths(game_id)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|path| {
            path.file_stem()
                .and_then(OsStr::to_str)
                .and_then(|stem| StartStateId::new(stem).ok())
                .map(|id| id.as_str().to_string())
        })
        .collect()
}

pub fn env_suffix_for_start_state(game_id: &str, start_state_id: &str) -> Option<String> {
    if !available_start_state_ids(game_id)
        .iter()
        .any(|id| id == start_state_id)
    {
        return None;
    }
    level_env_suffix(start_state_id)
}

fn level_env_suffix(start_state_id: &str) -> Option<String> {
    let rest = start_state_id.strip_prefix("level_")?;
    let parts = rest
        .split('_')
        .map(|part| {
            part.parse::<u32>()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| part.to_string())
        })
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join("-"))
}

pub fn start_state_for_env_suffix(game_id: &str, env_suffix: &str) -> Option<String> {
    available_start_state_ids(game_id)
        .into_iter()
        .find(|id| env_suffix_for_start_state(game_id, id).as_deref() == Some(env_suffix))
}

fn list_level_state_paths(game_id: &str) -> std::io::Result<Vec<PathBuf>> {
    let dir = game_start_state_dir(game_id);
    let entries = std::fs::read_dir(&dir)?;
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry?.path();
        let is_state = path.extension() == Some(OsStr::new("state"));
        let is_level = path
            .file_stem()
            .and_then(OsStr::to_str)
            .is_some_and(|stem| stem.starts_with("level_") && StartStateId::new(stem).is_ok());
        if is_state && is_level {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn read_start_state_path(path: &Path) -> Result<Vec<u8>> {
    if path.extension() != Some(OsStr::new("state")) {
        return Err(NesleError::InvalidState(format!(
            "start_state_path must point to a .state file, got {}",
            path.display()
        )));
    }
    std::fs::read(path).map_err(|err| {
        NesleError::InvalidState(format!(
            "failed to read start state {}: {err}",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_suffix_maps_level_ids() {
        assert_eq!(level_env_suffix("level_1_2").as_deref(), Some("1-2"));
        assert_eq!(level_env_suffix("level_1_1").as_deref(), Some("1-1"));
        assert_eq!(level_env_suffix("boss_01"), None);
        assert!(StartStateId::new("level_01").is_err());
    }
}
