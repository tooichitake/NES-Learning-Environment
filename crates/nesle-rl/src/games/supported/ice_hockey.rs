use nesle_common::action::{NesAction, NesButton};

use crate::games::GameSpec;

/// Ice Hockey (Nintendo, 1988), 1-2 players on one cartridge. With `players == 1`
/// the agent drives the left team against the CPU (reward = left score delta minus
/// the CPU/right delta -- index 0 of the zero-sum reward below). With `players == 2`
/// it is head-to-head -- the competitive counterpart to the cooperative Super C:
/// each agent drives one team, zero-sum reward (a goal for you is +1, a goal against
/// is -1; PettingZoo's competitive-Atari convention, e.g. boxing). RAM verified
/// against a real 3-2 playthrough recorded in the WASM viewer (replayed
/// deterministically + a cheat-poke RAM dump, cross-checked against a 0-0 game): the
/// left team's score is $042F and the right team's is $0430 (plain goal counters, 0
/// at the face-off). A match is three periods; the menu picks 1 or 2 players, then
/// each side builds a 4-skater LINEUP and selects the central END to drop the puck.
/// 1P (vs CPU) and 2P (head-to-head) are SEPARATE specs with separate start states --
/// the cart bakes the player count into RAM at the menu, so a 2P save can't be narrowed.
pub static ICE_HOCKEY_2P: GameSpec = GameSpec {
    id: "ice_hockey_2p",
    family: "ice_hockey",
    gym_id: "NESLE/IceHockey-2P-v0",
    display_name: "Ice Hockey",
    sha1: "3f732edbabdcc06dbb15e246c4438e6e53e90ac4",
    players: 2,
    four_score: false,
    mode: Some("2P"),
    minimal_actions: &ICE_HOCKEY_ACTIONS,
    reward: ice_hockey_reward,
    terminal: ice_hockey_terminal,
    lives: ice_hockey_lives,
    in_transition: None,
    per_agent_lives_termination: false,
};

pub static ICE_HOCKEY_1P: GameSpec = GameSpec {
    id: "ice_hockey_1p",
    family: "ice_hockey",
    gym_id: "NESLE/IceHockey-1P",
    display_name: "Ice Hockey",
    sha1: "3f732edbabdcc06dbb15e246c4438e6e53e90ac4",
    players: 1,
    four_score: false,
    mode: Some("1P"),
    minimal_actions: &ICE_HOCKEY_ACTIONS,
    reward: ice_hockey_reward,
    terminal: ice_hockey_terminal,
    lives: ice_hockey_lives,
    in_transition: None,
    per_agent_lives_termination: false,
};

/// Skate (8-way) + shoot/pass (A) + check/switch (B) -- Ice Hockey's moveset.
pub const ICE_HOCKEY_ACTIONS: [NesAction; 19] = [
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
    NesAction::new("UP_A", NesButton::Up as u8 | NesButton::A as u8),
    NesAction::new("DOWN_A", NesButton::Down as u8 | NesButton::A as u8),
    NesAction::new("LEFT_A", NesButton::Left as u8 | NesButton::A as u8),
    NesAction::new("RIGHT_A", NesButton::Right as u8 | NesButton::A as u8),
    NesAction::new("UP_B", NesButton::Up as u8 | NesButton::B as u8),
    NesAction::new("DOWN_B", NesButton::Down as u8 | NesButton::B as u8),
    NesAction::new("LEFT_B", NesButton::Left as u8 | NesButton::B as u8),
    NesAction::new("RIGHT_B", NesButton::Right as u8 | NesButton::B as u8),
];

// A goal raises a team's score by 1; the large game-over board-clear drop is ignored.
fn ice_hockey_goal(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800], addr: usize) -> f32 {
    let delta = current_ram[addr] as i32 - previous_ram[addr] as i32;
    if (1..=2).contains(&delta) {
        delta as f32
    } else {
        0.0
    }
}

fn ice_hockey_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    // Zero-sum: P1 (left, $042F) is +1/own goal, -1/opponent goal; P2 (right, $0430) is the negation. players==1 reads index 0.
    let left = ice_hockey_goal(previous_ram, current_ram, 0x042f);
    let right = ice_hockey_goal(previous_ram, current_ram, 0x0430);
    let p1 = left - right;
    [p1, -p1, 0.0, 0.0]
}

fn ice_hockey_terminal(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    // Game over clears the scoreboard to 0; scores only climb during play, so a score drop uniquely marks it. The `<= 50` guard ignores the power-on 255->0 init.
    let drop = |a: usize| current_ram[a] < previous_ram[a] && previous_ram[a] <= 50;
    drop(0x042f) || drop(0x0430)
}

fn ice_hockey_lives(_current_ram: &[u8; 0x800]) -> [u8; 4] {
    // Ice Hockey has no lives -- a match is period/time-bound, not life-bound.
    [0, 0, 0, 0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::smoke_multiplayer_actions;

    #[test]
    fn real_rom_steps_training_actions() {
        smoke_multiplayer_actions(
            &ICE_HOCKEY_2P,
            "Ice Hockey (USA).nes",
            &["UP_A", "DOWN_A", "LEFT_B", "RIGHT_B"],
        );
    }

    #[test]
    fn reward_is_zero_sum_goal_delta() {
        let prev = [0u8; 0x800];
        let mut left_goal = [0u8; 0x800];
        left_goal[0x042f] = 1; // left scores
        assert_eq!(ice_hockey_reward(&prev, &left_goal), [1.0, -1.0, 0.0, 0.0]);
        let mut right_goal = [0u8; 0x800];
        right_goal[0x0430] = 1; // right scores
        assert_eq!(ice_hockey_reward(&prev, &right_goal), [-1.0, 1.0, 0.0, 0.0]);
        // game-over board clear (3 -> 0) must not leak a huge reward
        let mut hi = [0u8; 0x800];
        hi[0x042f] = 3;
        assert_eq!(ice_hockey_reward(&hi, &prev), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn terminal_only_when_score_resets_at_game_over() {
        // a goal (score climbs) is not terminal
        let mut a = [0u8; 0x800];
        a[0x042f] = 2;
        let mut b = [0u8; 0x800];
        b[0x042f] = 3;
        assert!(!ice_hockey_terminal(&a, &b));
        assert_eq!(ice_hockey_lives(&b), [0, 0, 0, 0]);
        // face-off (both 0, unchanged) is not terminal
        let zero = [0u8; 0x800];
        assert!(!ice_hockey_terminal(&zero, &zero));
        // game over clears the board: score drops to 0 -> terminal
        assert!(ice_hockey_terminal(&b, &zero));
        // power-on RAM init (255 -> 0) is NOT a game over
        let mut garbage = [0u8; 0x800];
        garbage[0x042f] = 255;
        garbage[0x0430] = 255;
        assert!(!ice_hockey_terminal(&garbage, &zero));
    }
}
