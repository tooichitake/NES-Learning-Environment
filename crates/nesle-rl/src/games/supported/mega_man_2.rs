use nesle_common::action::{NesAction, NesButton};

use crate::games::{solo_lives, solo_reward, GameSpec};

pub static MEGA_MAN_2: GameSpec = GameSpec {
    id: "mega_man_2",
    family: "mega_man_2",
    gym_id: "NESLE/MegaMan2",
    display_name: "Mega Man 2",
    sha1: "54d53b46b961ba8b80ef57309bb0d78995d740a2",
    players: 1,
    four_score: false,
    mode: None,
    minimal_actions: &ACTIONS,
    reward,
    terminal,
    lives,
    in_transition: None,
    per_agent_lives_termination: false,
};

pub const ACTIONS: [NesAction; 16] = [
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
    NesAction::new("A_B", NesButton::A as u8 | NesButton::B as u8),
    NesAction::new(
        "RIGHT_A_B",
        NesButton::Right as u8 | NesButton::A as u8 | NesButton::B as u8,
    ),
    NesAction::new(
        "LEFT_A_B",
        NesButton::Left as u8 | NesButton::A as u8 | NesButton::B as u8,
    ),
    NesAction::new("UP_B", NesButton::Up as u8 | NesButton::B as u8),
    NesAction::new("DOWN_B", NesButton::Down as u8 | NesButton::B as u8),
];

const SCREEN_ADDR: usize = 0x002a;
const X_SCROLL_ADDR: usize = 0x001f;
const HEALTH_ADDR: usize = 0x06c0;
const LIVES_ADDR: usize = 0x00a8;
const MODE_ADDR: usize = 0x0027;

fn reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    let screen_bonus = if current_ram[SCREEN_ADDR] > previous_ram[SCREEN_ADDR]
        && current_ram[SCREEN_ADDR] - previous_ram[SCREEN_ADDR] <= 2
    {
        25.0
    } else {
        0.0
    };
    let scroll_delta = current_ram[X_SCROLL_ADDR] as i16 - previous_ram[X_SCROLL_ADDR] as i16;
    let progress = if (1..=6).contains(&scroll_delta) {
        scroll_delta as f32
    } else {
        0.0
    };
    solo_reward(
        progress
            + screen_bonus
            + health_delta(previous_ram, current_ram)
            + death_penalty(previous_ram, current_ram),
    )
}

fn terminal(_previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    current_ram[MODE_ADDR] != 0xff && current_ram[LIVES_ADDR] == 0xff
}

fn lives(current_ram: &[u8; 0x800]) -> [u8; 4] {
    let value = current_ram[LIVES_ADDR];
    solo_lives(if value == 0xff { 0 } else { value })
}

fn health_delta(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    let prev = previous_ram[HEALTH_ADDR];
    let cur = current_ram[HEALTH_ADDR];
    if cur < prev && prev <= 0x1c {
        -0.5 * (prev - cur) as f32
    } else {
        0.0
    }
}

fn death_penalty(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    let prev = previous_ram[LIVES_ADDR];
    let cur = current_ram[LIVES_ADDR];
    if cur < prev && prev != 0xff {
        -25.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::smoke_single_player_actions;

    #[test]
    fn real_rom_steps_training_actions() {
        smoke_single_player_actions(
            &MEGA_MAN_2,
            "Mega Man 2 (USA).nes",
            &["A_B", "RIGHT_A_B", "LEFT_A_B", "UP_B", "DOWN_B"],
            |_| {},
        );
    }

    fn ram_with_state(
        screen: u8,
        scroll: u8,
        health: u8,
        lives_value: u8,
        mode: u8,
    ) -> [u8; 0x800] {
        let mut ram = [0u8; 0x800];
        ram[SCREEN_ADDR] = screen;
        ram[X_SCROLL_ADDR] = scroll;
        ram[HEALTH_ADDR] = health;
        ram[LIVES_ADDR] = lives_value;
        ram[MODE_ADDR] = mode;
        ram
    }

    #[test]
    fn reward_clips_scroll_and_rewards_screen_advance_fixture() {
        let prev = ram_with_state(0, 10, 0x1c, 2, 1);
        let mut cur = ram_with_state(0, 13, 0x1c, 2, 1);
        assert_eq!(reward(&prev, &cur)[0], 3.0);

        cur[SCREEN_ADDR] = 1;
        assert_eq!(reward(&prev, &cur)[0], 28.0);

        cur[X_SCROLL_ADDR] = 200;
        assert_eq!(reward(&prev, &cur)[0], 25.0);
    }

    #[test]
    fn reward_penalizes_health_and_life_loss_fixture() {
        let prev = ram_with_state(0, 0, 0x1c, 2, 1);
        let mut cur = ram_with_state(0, 0, 0x18, 2, 1);
        assert_eq!(health_delta(&prev, &cur), -2.0);

        cur[LIVES_ADDR] = 1;
        assert_eq!(death_penalty(&prev, &cur), -25.0);
        assert_eq!(reward(&prev, &cur)[0], -27.0);
    }

    #[test]
    fn terminal_and_lives_use_mode_guard_fixture() {
        let prev = [0u8; 0x800];
        let mut ram = ram_with_state(0, 0, 0x1c, 2, 1);
        assert!(!terminal(&prev, &ram));
        assert_eq!(lives(&ram)[0], 2);

        ram[LIVES_ADDR] = 0;
        assert!(!terminal(&prev, &ram));
        assert_eq!(lives(&ram)[0], 0);

        ram[LIVES_ADDR] = 0xff;
        assert!(terminal(&prev, &ram));
        assert_eq!(lives(&ram)[0], 0);

        ram[MODE_ADDR] = 0xff;
        assert!(!terminal(&prev, &ram));
    }

    #[test]
    fn reward_lives_and_terminal_survive_boot_and_menu_ram() {
        let garbage = [0xffu8; 0x800];
        let menu = [0u8; 0x800];

        let _ = reward(&garbage, &garbage);
        assert!(!terminal(&garbage, &garbage));
        assert_eq!(lives(&garbage)[0], 0);

        assert_eq!(reward(&menu, &menu)[0], 0.0);
        assert!(!terminal(&menu, &menu));
        assert_eq!(lives(&menu)[0], 0);
    }
}
