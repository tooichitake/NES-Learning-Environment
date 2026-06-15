use nesle_common::action::{NesAction, NesButton};

use crate::games::{solo_lives, solo_reward, GameSpec};

pub static SUPER_MARIO_BROS_2: GameSpec = GameSpec {
    id: "super_mario_bros_2",
    family: "super_mario_bros_2",
    gym_id: "NESLE/SuperMarioBros2",
    display_name: "Super Mario Bros. 2",
    sha1: "0dde824764daba70217e6167d01f202fb1666cb0",
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
    NesAction::new("UP_B", NesButton::Up as u8 | NesButton::B as u8),
    NesAction::new("DOWN_B", NesButton::Down as u8 | NesButton::B as u8),
    NesAction::new(
        "RIGHT_A_B",
        NesButton::Right as u8 | NesButton::A as u8 | NesButton::B as u8,
    ),
    NesAction::new(
        "LEFT_A_B",
        NesButton::Left as u8 | NesButton::A as u8 | NesButton::B as u8,
    ),
    NesAction::new("DOWN_A", NesButton::Down as u8 | NesButton::A as u8),
];

const X_ADDR: usize = 0x0028;
const LIFE_METER_ADDR: usize = 0x04c2;
const LIVES_ADDR: usize = 0x04ed;

fn reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    solo_reward(x_reward(previous_ram, current_ram) + damage_penalty(previous_ram, current_ram))
}

fn x_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    let delta = current_ram[X_ADDR] as i32 - previous_ram[X_ADDR] as i32;
    if (1..=8).contains(&delta) {
        delta as f32
    } else {
        0.0
    }
}

fn damage_penalty(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    let prev_lives = lives(previous_ram)[0];
    let cur_lives = lives(current_ram)[0];
    if prev_lives > 0 && cur_lives < prev_lives {
        return -25.0;
    }
    let prev_meter = previous_ram[LIFE_METER_ADDR];
    let cur_meter = current_ram[LIFE_METER_ADDR];
    if prev_meter > 0 && cur_meter == 0 {
        -10.0
    } else {
        0.0
    }
}

fn terminal(_previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    lives(current_ram)[0] == 0 || current_ram[LIFE_METER_ADDR] == 0
}

fn lives(current_ram: &[u8; 0x800]) -> [u8; 4] {
    solo_lives(if current_ram[LIVES_ADDR] <= 9 {
        current_ram[LIVES_ADDR]
    } else {
        0
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::smoke_single_player_actions;

    #[test]
    fn real_rom_steps_training_actions() {
        smoke_single_player_actions(
            &SUPER_MARIO_BROS_2,
            "Super Mario Bros. 2 (USA) (Rev 1).nes",
            &["RIGHT_A_B", "LEFT_A_B", "DOWN_A"],
            |_| {},
        );
    }

    fn ram(x: u8, meter: u8, life_count: u8) -> [u8; 0x800] {
        let mut ram = [0u8; 0x800];
        ram[X_ADDR] = x;
        ram[LIFE_METER_ADDR] = meter;
        ram[LIVES_ADDR] = life_count;
        ram
    }

    #[test]
    fn reward_uses_clipped_horizontal_progress() {
        assert_eq!(x_reward(&ram(120, 31, 3), &ram(124, 31, 3)), 4.0);
        assert_eq!(x_reward(&ram(250, 31, 3), &ram(5, 31, 3)), 0.0);
        assert_eq!(x_reward(&ram(120, 31, 3), &ram(118, 31, 3)), 0.0);
    }

    #[test]
    fn terminal_and_penalty_track_health_and_lives() {
        let prev = [0u8; 0x800];
        assert!(!terminal(&prev, &ram(120, 31, 3)));
        assert!(terminal(&prev, &ram(120, 0, 3)));
        assert_eq!(damage_penalty(&ram(120, 31, 3), &ram(120, 0, 3)), -10.0);
        assert_eq!(damage_penalty(&ram(120, 31, 3), &ram(120, 31, 2)), -25.0);
    }
}
