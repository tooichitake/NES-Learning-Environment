use nesle_common::action::{NesAction, NesButton};

use crate::games::GameSpec;

/// Super C (Konami, 1990) run-'n-gun, exposed as two player-count modes sharing one RAM
/// map. 2P co-op (chosen at the menu with SELECT): both players share the world and each
/// agent's reward is its own score delta -- PettingZoo's cooperative-Atari convention. 1P
/// solo is the single-agent variant (reward index 0 = P1). The cart bakes the player count
/// into RAM at the menu, so the two modes are SEPARATE specs with separate start states,
/// not one spec narrowed by `set_players`. RAM map verified against the running ROM: P1/P2
/// score $07E3/$07E6 (3-byte BCD), lives $53/$54, level $50, game-over $CA, player state
/// $A0/$A1 (2 = controllable on the ground).
pub static SUPER_C_2P: GameSpec = GameSpec {
    id: "super_c_2p",
    family: "super_c",
    gym_id: "NESLE/SuperC-2P-v0",
    display_name: "Super C",
    sha1: "032708cc9eb6e25d7ab43f50a3e572222f755239",
    players: 2,
    four_score: false,
    mode: Some("2P"),
    minimal_actions: &SUPER_C_ACTIONS,
    reward: super_c_reward,
    terminal: super_c_terminal,
    lives: super_c_lives,
    in_transition: None,
    per_agent_lives_termination: false,
};

pub static SUPER_C_1P: GameSpec = GameSpec {
    id: "super_c_1p",
    family: "super_c",
    gym_id: "NESLE/SuperC-1P",
    display_name: "Super C",
    sha1: "032708cc9eb6e25d7ab43f50a3e572222f755239",
    players: 1,
    four_score: false,
    mode: Some("1P"),
    minimal_actions: &SUPER_C_ACTIONS,
    reward: super_c_reward,
    terminal: super_c_terminal,
    lives: super_c_lives,
    in_transition: None,
    per_agent_lives_termination: false,
};

/// Move, jump, fire, movement+jump, run-fire, vertical/prone fire,
/// diagonal-up fire, and jump-fire.
pub const SUPER_C_ACTIONS: [NesAction; 18] = [
    NesAction::new("NOOP", 0),
    NesAction::new("RIGHT", NesButton::Right as u8),
    NesAction::new("LEFT", NesButton::Left as u8),
    NesAction::new("UP", NesButton::Up as u8),
    NesAction::new("DOWN", NesButton::Down as u8),
    NesAction::new("A", NesButton::A as u8),
    NesAction::new("B", NesButton::B as u8),
    NesAction::new("RIGHT_A", NesButton::Right as u8 | NesButton::A as u8),
    NesAction::new("LEFT_A", NesButton::Left as u8 | NesButton::A as u8),
    NesAction::new("RIGHT_B", NesButton::Right as u8 | NesButton::B as u8),
    NesAction::new("LEFT_B", NesButton::Left as u8 | NesButton::B as u8),
    NesAction::new("UP_B", NesButton::Up as u8 | NesButton::B as u8),
    NesAction::new("DOWN_B", NesButton::Down as u8 | NesButton::B as u8),
    NesAction::new(
        "UPRIGHT_B",
        NesButton::Up as u8 | NesButton::Right as u8 | NesButton::B as u8,
    ),
    NesAction::new(
        "UPLEFT_B",
        NesButton::Up as u8 | NesButton::Left as u8 | NesButton::B as u8,
    ),
    NesAction::new("A_B", NesButton::A as u8 | NesButton::B as u8),
    NesAction::new(
        "RIGHT_A_B",
        NesButton::Right as u8 | NesButton::A as u8 | NesButton::B as u8,
    ),
    NesAction::new(
        "LEFT_A_B",
        NesButton::Left as u8 | NesButton::A as u8 | NesButton::B as u8,
    ),
];

fn super_c_player_score(ram: &[u8; 0x800], base: usize) -> u32 {
    // 3-byte BCD, most-significant byte first (verified monotonic vs kills).
    let mut score = 0u32;
    for k in 0..3 {
        let b = ram[base + k] as u32;
        score = score * 100 + (b >> 4) * 10 + (b & 0x0f);
    }
    score
}

fn super_c_scroll(ram: &[u8; 0x800]) -> i32 {
    // 2-byte LE forward camera position at $00FD-$00FE (verified monotone under RIGHT, stable under LEFT).
    (ram[0x00fd] as i32) | ((ram[0x00fe] as i32) << 8)
}

fn super_c_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    // Per-agent SMB-style reward: shared progress (deltascroll clipped [1,5]) + stage-clear bonus, plus each port's own deltascore*0.01 and -25 death penalty.
    let scroll_prev = super_c_scroll(previous_ram);
    let scroll_cur = super_c_scroll(current_ram);
    let scroll_delta = scroll_cur - scroll_prev;
    let progress = if (1..=5).contains(&scroll_delta) {
        scroll_delta as f32
    } else {
        0.0
    };

    let p1_score = super_c_player_score(current_ram, 0x07e3)
        .saturating_sub(super_c_player_score(previous_ram, 0x07e3)) as f32
        * 0.01;
    let p2_score = super_c_player_score(current_ram, 0x07e6)
        .saturating_sub(super_c_player_score(previous_ram, 0x07e6)) as f32
        * 0.01;

    let p1_death = if current_ram[0x0053] < previous_ram[0x0053] && previous_ram[0x0053] != 0xff {
        -25.0
    } else {
        0.0
    };
    let p2_death = if current_ram[0x0054] < previous_ram[0x0054] && previous_ram[0x0054] != 0xff {
        -25.0
    } else {
        0.0
    };

    let level_prev = previous_ram[0x0050];
    let level_cur = current_ram[0x0050];
    let stage_clear = if level_cur > level_prev && level_cur - level_prev <= 2 {
        50.0
    } else {
        0.0
    };

    [
        progress + p1_score + p1_death + stage_clear,
        progress + p2_score + p2_death + stage_clear,
        0.0,
        0.0,
    ]
}

fn super_c_terminal(_previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    // Game over: the documented flag $CA, or both players out of lives.
    current_ram[0x00ca] == 1 || (current_ram[0x0053] == 0 && current_ram[0x0054] == 0)
}

fn super_c_lives(current_ram: &[u8; 0x800]) -> [u8; 4] {
    [current_ram[0x0053], current_ram[0x0054], 0, 0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::smoke_multiplayer_actions;

    #[test]
    fn real_rom_steps_training_actions() {
        smoke_multiplayer_actions(
            &SUPER_C_2P,
            "Super C (USA).nes",
            &["A_B", "RIGHT_A_B", "LEFT_A_B", "UPRIGHT_B", "UPLEFT_B"],
        );
    }

    #[test]
    fn reward_progress_shared_across_players() {
        // deltascroll is the shared world-progress component (one camera for both players).
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        prev[0x00fd] = 0x10;
        cur[0x00fd] = 0x13; // +3 scroll
        let r = super_c_reward(&prev, &cur);
        assert_eq!(r[0], 3.0);
        assert_eq!(r[1], 3.0);
        assert_eq!(r[2], 0.0);
        assert_eq!(r[3], 0.0);
    }

    #[test]
    fn reward_score_per_agent_scaled() {
        // Per-agent score delta * 0.01, so a P2-only +100 kill credits only P2.
        let prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        cur[0x07e6 + 1] = 0x01; // +100 to P2 score (mid BCD byte)
        let r = super_c_reward(&prev, &cur);
        assert!((r[0]).abs() < 1e-6); // P1 got nothing
        assert!((r[1] - 1.0).abs() < 1e-6); // P2 got +1.0 (= 100 * 0.01)
    }

    #[test]
    fn reward_death_penalty_per_agent() {
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        prev[0x0053] = 2;
        prev[0x0054] = 2;
        cur[0x0053] = 1; // P1 lost a life
        cur[0x0054] = 2; // P2 didn't
        let r = super_c_reward(&prev, &cur);
        assert_eq!(r[0], -25.0);
        assert_eq!(r[1], 0.0);
    }

    #[test]
    fn reward_stage_clear_shared() {
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        prev[0x0050] = 1;
        cur[0x0050] = 2;
        let r = super_c_reward(&prev, &cur);
        assert_eq!(r[0], 50.0);
        assert_eq!(r[1], 50.0);
    }

    #[test]
    fn action_set_includes_movement_jump_and_directional_fire() {
        let has = |name: &str, mask: u8| -> bool {
            SUPER_C_ACTIONS
                .iter()
                .any(|a| a.name == name && a.mask == mask)
        };
        assert!(has("RIGHT_A", NesButton::Right as u8 | NesButton::A as u8));
        assert!(has("LEFT_A", NesButton::Left as u8 | NesButton::A as u8));
        assert!(has("UP_B", NesButton::Up as u8 | NesButton::B as u8));
        assert!(has("DOWN_B", NesButton::Down as u8 | NesButton::B as u8));
        assert!(has("RIGHT_B", NesButton::Right as u8 | NesButton::B as u8));
        assert!(has("LEFT_B", NesButton::Left as u8 | NesButton::B as u8));
        assert!(has(
            "UPRIGHT_B",
            NesButton::Up as u8 | NesButton::Right as u8 | NesButton::B as u8
        ));
        assert!(has(
            "UPLEFT_B",
            NesButton::Up as u8 | NesButton::Left as u8 | NesButton::B as u8
        ));
    }

    #[test]
    fn reward_survives_boot_garbage() {
        // Server Play runs from power-on (before $07E3-$07E8 init); the BCD decoder's saturating_sub must yield 0, not panic.
        let garbage = [0xffu8; 0x800];
        let _ = super_c_reward(&garbage, &garbage);
        let clean = [0u8; 0x800];
        // garbage prev + clean cur is the real reset case (boot 0xff then cleared); must clamp to 0.
        assert_eq!(super_c_reward(&garbage, &clean), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn terminal_on_game_over_or_both_out_of_lives() {
        let prev = [0u8; 0x800];
        let mut ram = [0u8; 0x800];
        ram[0x0053] = 3;
        ram[0x0054] = 2;
        assert!(!super_c_terminal(&prev, &ram));
        assert_eq!(super_c_lives(&ram), [3, 2, 0, 0]);
        ram[0x0053] = 0;
        ram[0x0054] = 0;
        assert!(super_c_terminal(&prev, &ram)); // both out
        ram[0x0053] = 1;
        assert!(!super_c_terminal(&prev, &ram));
        ram[0x00ca] = 1;
        assert!(super_c_terminal(&prev, &ram)); // explicit game-over flag
    }
}
