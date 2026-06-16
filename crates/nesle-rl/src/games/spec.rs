use nesle_common::ActionSet;

use crate::constants::{MAX_PLAYERS, RAM_SIZE};

pub type Ram = [u8; RAM_SIZE];
pub type MultiPlayerValues<T> = [T; MAX_PLAYERS];

/// Per-port rewards (index i = controller port i; unused trailing ports `0.0`).
/// Single-player games fill index 0 via [`solo_reward`].
pub type RewardFn = fn(previous_ram: &Ram, current_ram: &Ram) -> MultiPlayerValues<f32>;
/// Shared episode-over predicate. Receives `previous_ram` too (some games end on a
/// transition, e.g. Ice Hockey's score-reset); single-player games ignore it.
pub type TerminalFn = fn(previous_ram: &Ram, current_ram: &Ram) -> bool;
/// Per-port lives / alive-flags (index i = port i; unused trailing ports `0`).
/// Single-player games fill index 0 via [`solo_lives`].
pub type LivesFn = fn(current_ram: &Ram) -> MultiPlayerValues<u8>;
/// True while the agent is in a non-interactive level transition (cutscene / load).
/// The env fast-forwards these frames so the agent never observes the dead zone;
/// `None` = the game has no such transition.
pub type TransitionFn = fn(current_ram: &Ram) -> bool;

/// One game, or one mode of a game: a single spec drives 1..=`players`
/// controller ports -- single-player is the `players == 1` case. Modes are separate
/// specs sharing one `display_name`, told apart by `mode`: scoring modes (Bomberman II
/// Normal / VS / Battle) and player-count modes (Super C / Ice Hockey 1P vs 2P, which
/// the cart bakes into RAM at the menu, so a 2P save state can't be narrowed to 1P).
///
/// Each field defines one aspect of a game's RL contract:
/// `reward` (per-step reward), `terminal` (episode-over predicate), `lives` (per-port),
/// `minimal_actions` (the game's action set), `mode` (scoring / player-count mode;
/// NESLE models modes as separate specs, not a runtime switch), `sha1` (ROM identity),
/// `id`/`gym_id`/`display_name` (naming). Multiplayer: `players` / `four_score` /
/// `per_agent_lives_termination`. `in_transition` (cutscene fast-forward) and `family`
/// (start-state folder grouping) are extras. Reset is a cached save-state, not a
/// scripted action sequence.
#[derive(Debug, Clone, Copy)]
pub struct GameSpec {
    pub id: &'static str,
    /// Game-family folder for start-state assets. Equals `id` for single-spec games
    /// (flat `start_states/<id>/`); for games whose modes are separate specs (Bomberman
    /// 2 Normal/VS/Battle) every spec shares the family (e.g. "bomberman_2"), so assets
    /// nest as `start_states/<family>/<mode>/`.
    pub family: &'static str,
    pub gym_id: &'static str,
    /// GAME name only (no mode / port suffix), e.g. `"Bomberman II"`. Drives the
    /// server's game -> mode -> level menu grouping.
    pub display_name: &'static str,
    pub sha1: &'static str,
    /// Controller ports this spec can drive (1..=4). `set_players` may run fewer.
    pub players: u8,
    /// Four Score adapter required to reach ports 2-3.
    pub four_score: bool,
    /// In-game mode name in the cart's own wording (e.g. `Some("Battle")`);
    /// `None` for single-mode games (SMB, Pac-Man, ...).
    pub mode: Option<&'static str>,
    pub minimal_actions: ActionSet,
    pub reward: RewardFn,
    pub terminal: TerminalFn,
    pub lives: LivesFn,
    pub in_transition: Option<TransitionFn>,
    /// When true, a port terminates the frame its `lives[i]` reaches 0 (last-standing
    /// games like Bomberman); when false, only the shared `terminal` ends ports
    /// (co-op / single-player).
    pub per_agent_lives_termination: bool,
}

/// Lift a single-player scalar reward into the per-port array (port 0 only).
pub fn solo_reward(reward: f32) -> MultiPlayerValues<f32> {
    [reward, 0.0, 0.0, 0.0]
}

/// Lift a single-player lives count into the per-port array (port 0 only).
pub fn solo_lives(lives: u8) -> MultiPlayerValues<u8> {
    [lives, 0, 0, 0]
}
