use nesle_common::action::{NesAction, NesButton};

use crate::games::GameSpec;

/// Data Crystal's R.C. Pro-Am II RAM map marks these as one-byte-per-racer
/// arrays: $0554/$056D track position high/low, $055E laps, $0596 speed integer,
/// $074C car active, $0755 screen display type,
/// $076D finish placement, and $0799/$079D circuit point score.
pub static R_C_PRO_AM_2: GameSpec = GameSpec {
    id: "r_c_pro_am_2",
    family: "r_c_pro_am_2",
    gym_id: "NESLE/RCProAm2-4P-v0",
    display_name: "R.C. Pro-Am 2",
    sha1: "d536d1a6d87fa2c9d32f58f76c9545cc506fea8f",
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

pub const ACTIONS: [NesAction; 12] = [
    NesAction::new("NOOP", 0),
    NesAction::new("UP", NesButton::Up as u8),
    NesAction::new("DOWN", NesButton::Down as u8),
    NesAction::new("LEFT", NesButton::Left as u8),
    NesAction::new("RIGHT", NesButton::Right as u8),
    NesAction::new("A", NesButton::A as u8),
    NesAction::new("B", NesButton::B as u8),
    NesAction::new("LEFT_B", NesButton::Left as u8 | NesButton::B as u8),
    NesAction::new("RIGHT_B", NesButton::Right as u8 | NesButton::B as u8),
    NesAction::new("A_B", NesButton::A as u8 | NesButton::B as u8),
    NesAction::new(
        "LEFT_A_B",
        NesButton::Left as u8 | NesButton::A as u8 | NesButton::B as u8,
    ),
    NesAction::new(
        "RIGHT_A_B",
        NesButton::Right as u8 | NesButton::A as u8 | NesButton::B as u8,
    ),
];

const TRACK_POS_HI_ADDR: usize = 0x0554;
const LAPS_ADDR: usize = 0x055e;
const TRACK_POS_LO_ADDR: usize = 0x056d;
const SPEED_INTEGER_ADDR: usize = 0x0596;
const CAR_ACTIVE_ADDR: usize = 0x074c;
const SCREEN_DISPLAY_TYPE_ADDR: usize = 0x0755;
const FINISH_PLACEMENT_ADDR: usize = 0x076d;
const POINT_SCORE_LO_ADDR: usize = 0x0799;
const POINT_SCORE_HI_ADDR: usize = 0x079d;
const ACTIVE_RACE_SCREEN: u8 = 0x00;

fn reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    let mut rewards = [0.0; 4];
    for (player, reward) in rewards.iter_mut().enumerate() {
        *reward += progress_reward(previous_ram, current_ram, player);
        *reward += finish_reward(previous_ram, current_ram, player);
        *reward += point_score_reward(previous_ram, current_ram, player);
    }
    rewards
}

fn progress_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800], player: usize) -> f32 {
    if previous_ram[SCREEN_DISPLAY_TYPE_ADDR] != ACTIVE_RACE_SCREEN
        || current_ram[SCREEN_DISPLAY_TYPE_ADDR] != ACTIVE_RACE_SCREEN
        || current_ram[CAR_ACTIVE_ADDR + player] == 0
        || current_ram[SPEED_INTEGER_ADDR + player] == 0
    {
        return 0.0;
    }
    let previous = track_progress(previous_ram, player);
    let current = track_progress(current_ram, player);
    let delta = current - previous;
    if (1..=512).contains(&delta) {
        delta as f32 * 0.01
    } else {
        0.0
    }
}

fn finish_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800], player: usize) -> f32 {
    if previous_ram[FINISH_PLACEMENT_ADDR + player] != 0 {
        return 0.0;
    }
    match current_ram[FINISH_PLACEMENT_ADDR + player] {
        1 => 20.0,
        2 => 12.0,
        3 => 6.0,
        4 => 2.0,
        _ => 0.0,
    }
}

fn point_score_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800], player: usize) -> f32 {
    let previous = point_score(previous_ram, player);
    let current = point_score(current_ram, player);
    let delta = current.saturating_sub(previous);
    if delta <= 10_000 {
        delta as f32 * 0.1
    } else {
        0.0
    }
}

fn terminal(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    previous_ram[SCREEN_DISPLAY_TYPE_ADDR] == ACTIVE_RACE_SCREEN
        && current_ram[SCREEN_DISPLAY_TYPE_ADDR] != ACTIVE_RACE_SCREEN
}

fn lives(_current_ram: &[u8; 0x800]) -> [u8; 4] {
    [0, 0, 0, 0]
}

fn track_progress(ram: &[u8; 0x800], player: usize) -> i32 {
    ((ram[LAPS_ADDR + player] as i32) << 16)
        | ((ram[TRACK_POS_HI_ADDR + player] as i32) << 8)
        | ram[TRACK_POS_LO_ADDR + player] as i32
}

fn point_score(ram: &[u8; 0x800], player: usize) -> u32 {
    ((ram[POINT_SCORE_HI_ADDR + player] as u32) << 8) | ram[POINT_SCORE_LO_ADDR + player] as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::smoke_multiplayer_actions;

    #[test]
    fn real_rom_steps_training_actions() {
        smoke_multiplayer_actions(
            &R_C_PRO_AM_2,
            "R.C. Pro-Am II (USA).nes",
            &["LEFT_A_B", "RIGHT_A_B"],
        );
    }

    #[test]
    fn reward_shapes_per_car_progress_when_car_is_moving() {
        let prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        cur[SCREEN_DISPLAY_TYPE_ADDR] = ACTIVE_RACE_SCREEN;
        cur[CAR_ACTIVE_ADDR] = 1;
        cur[SPEED_INTEGER_ADDR] = 10;
        cur[TRACK_POS_LO_ADDR] = 5;
        let rewards = reward(&prev, &cur);
        assert!((rewards[0] - 0.05).abs() < 1e-6);
        assert_eq!(rewards[1..], [0.0, 0.0, 0.0]);

        cur[SPEED_INTEGER_ADDR] = 0;
        assert_eq!(reward(&prev, &cur), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn reward_includes_finish_and_points() {
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        cur[FINISH_PLACEMENT_ADDR] = 1;
        prev[POINT_SCORE_LO_ADDR] = 10;
        cur[POINT_SCORE_LO_ADDR] = 15;
        assert_eq!(reward(&prev, &cur)[0], 20.5);
    }

    #[test]
    fn race_ends_when_active_race_screen_exits() {
        let ram = [0u8; 0x800];
        assert_eq!(lives(&ram), [0, 0, 0, 0]);
        assert!(!terminal(&ram, &ram));
        let mut shop = ram;
        shop[SCREEN_DISPLAY_TYPE_ADDR] = 2;
        assert!(terminal(&ram, &shop));
    }

    #[test]
    fn reward_terminal_and_lives_survive_boot_garbage() {
        let garbage = [0xffu8; 0x800];
        assert_eq!(reward(&garbage, &garbage), [0.0, 0.0, 0.0, 0.0]);
        assert!(!terminal(&garbage, &garbage));
        assert_eq!(lives(&garbage), [0, 0, 0, 0]);

        let clean = [0u8; 0x800];
        assert_eq!(reward(&garbage, &clean), [0.0, 0.0, 0.0, 0.0]);
        assert!(!terminal(&garbage, &clean));
        assert_eq!(lives(&clean), [0, 0, 0, 0]);
    }
}
