use nesle_common::action::{NesAction, NesButton};

use crate::games::GameSpec;

pub static ROUNDBALL_2ON2: GameSpec = GameSpec {
    id: "roundball_2on2",
    family: "roundball_2on2",
    gym_id: "NESLE/Roundball2on2-4P-v0",
    display_name: "Roundball - 2-on-2 Challenge",
    sha1: "eb23642fd7edd4916fc61a1f9ae8c16b36f0f950",
    players: 4,
    four_score: true,
    mode: Some("4P"),
    minimal_actions: &ACTIONS,
    reward,
    terminal,
    lives,
    in_transition: None,
    per_agent_lives_termination: false,
};

pub const ACTIONS: [NesAction; 11] = [
    NesAction::new("NOOP", 0),
    NesAction::new("UP", NesButton::Up as u8),
    NesAction::new("DOWN", NesButton::Down as u8),
    NesAction::new("LEFT", NesButton::Left as u8),
    NesAction::new("RIGHT", NesButton::Right as u8),
    NesAction::new("UP_LEFT", NesButton::Up as u8 | NesButton::Left as u8),
    NesAction::new("UP_RIGHT", NesButton::Up as u8 | NesButton::Right as u8),
    NesAction::new("DOWN_LEFT", NesButton::Down as u8 | NesButton::Left as u8),
    NesAction::new("DOWN_RIGHT", NesButton::Down as u8 | NesButton::Right as u8),
    NesAction::new("A", NesButton::A as u8),
    NesAction::new("B", NesButton::B as u8),
];

const TEAM_A_SCORE: usize = 0x0430;
const TEAM_B_SCORE: usize = 0x0431;

fn reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    let a = score_delta(previous_ram, current_ram, TEAM_A_SCORE);
    let b = score_delta(previous_ram, current_ram, TEAM_B_SCORE);
    let team_a = a - b;
    [team_a, team_a, -team_a, -team_a]
}

fn terminal(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    let drop = |addr: usize| current_ram[addr] < previous_ram[addr] && previous_ram[addr] <= 99;
    drop(TEAM_A_SCORE) || drop(TEAM_B_SCORE)
}

fn lives(_current_ram: &[u8; 0x800]) -> [u8; 4] {
    [0, 0, 0, 0]
}

fn score_delta(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800], addr: usize) -> f32 {
    let delta = current_ram[addr] as i16 - previous_ram[addr] as i16;
    if (1..=3).contains(&delta) {
        delta as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_is_team_zero_sum() {
        let prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        cur[TEAM_A_SCORE] = 2;
        assert_eq!(reward(&prev, &cur), [2.0, 2.0, -2.0, -2.0]);
        cur[TEAM_A_SCORE] = 0;
        cur[TEAM_B_SCORE] = 3;
        assert_eq!(reward(&prev, &cur), [-3.0, -3.0, 3.0, 3.0]);
    }

    #[test]
    fn reward_handles_both_team_deltas_in_one_frame() {
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        prev[TEAM_A_SCORE] = 4;
        prev[TEAM_B_SCORE] = 3;
        cur[TEAM_A_SCORE] = 5;
        cur[TEAM_B_SCORE] = 5;

        assert_eq!(reward(&prev, &cur), [-1.0, -1.0, 1.0, 1.0]);
    }

    #[test]
    fn reward_ignores_implausible_score_changes() {
        let prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        cur[TEAM_A_SCORE] = 4;
        cur[TEAM_B_SCORE] = 250;

        assert_eq!(reward(&prev, &cur), [0.0, 0.0, -0.0, -0.0]);
    }

    #[test]
    fn terminal_uses_scoreboard_clear_transition() {
        let mut prev = [0u8; 0x800];
        let cur = [0u8; 0x800];
        prev[TEAM_A_SCORE] = 12;
        assert!(terminal(&prev, &cur));
        prev[TEAM_A_SCORE] = 255;
        assert!(!terminal(&prev, &cur));
        assert_eq!(lives(&cur), [0, 0, 0, 0]);
    }

    #[test]
    fn terminal_and_lives_are_shared_for_scoreboard_games() {
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        prev[TEAM_B_SCORE] = 8;
        cur[TEAM_B_SCORE] = 7;

        assert!(terminal(&prev, &cur));
        assert_eq!(lives(&cur), [0, 0, 0, 0]);
    }

    #[test]
    fn reward_terminal_and_lives_survive_boot_garbage() {
        let garbage = [0xffu8; 0x800];
        assert_eq!(reward(&garbage, &garbage), [0.0, 0.0, -0.0, -0.0]);
        assert!(!terminal(&garbage, &garbage));
        assert_eq!(lives(&garbage), [0, 0, 0, 0]);

        let clean = [0u8; 0x800];
        assert_eq!(reward(&garbage, &clean), [0.0, 0.0, -0.0, -0.0]);
        assert!(!terminal(&garbage, &clean));
        assert_eq!(lives(&clean), [0, 0, 0, 0]);
    }
}
