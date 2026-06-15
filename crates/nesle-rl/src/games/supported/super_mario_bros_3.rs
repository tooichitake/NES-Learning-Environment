use nesle_common::action::{NesAction, NesButton};

use crate::games::{solo_lives, solo_reward, GameSpec};

pub static SUPER_MARIO_BROS_3: GameSpec = GameSpec {
    id: "super_mario_bros_3",
    family: "super_mario_bros_3",
    gym_id: "NESLE/SuperMarioBros3",
    display_name: "Super Mario Bros. 3",
    sha1: "6bd518e85eb46a4252af07910f61036e84b020d1",
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
    NesAction::new("UP_A", NesButton::Up as u8 | NesButton::A as u8),
];

const PLAYER_X_ADDR: usize = 0x0090;
const PLAYER_Y_ADDR: usize = 0x00a2;
const PLAYER_Y_PLUS_11_ADDR: usize = 0x00b4;
const LIVES_ADDR: usize = 0x0736;

fn reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    solo_reward(x_reward(previous_ram, current_ram) + death_penalty(previous_ram, current_ram))
}

fn x_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    if !is_level_play(previous_ram) || !is_level_play(current_ram) {
        return 0.0;
    }
    let delta = current_ram[PLAYER_X_ADDR] as i32 - previous_ram[PLAYER_X_ADDR] as i32;
    if (1..=8).contains(&delta) {
        delta as f32
    } else {
        0.0
    }
}

fn death_penalty(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    let prev_lives = lives(previous_ram)[0];
    let cur_lives = lives(current_ram)[0];
    if (prev_lives > 0 && cur_lives < prev_lives)
        || (current_ram[PLAYER_Y_PLUS_11_ADDR] >= 0xc0
            && previous_ram[PLAYER_Y_PLUS_11_ADDR] < 0xc0)
    {
        -25.0
    } else {
        0.0
    }
}

fn terminal(_previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    lives(current_ram)[0] == 0 || current_ram[PLAYER_Y_PLUS_11_ADDR] >= 0xc0
}

fn lives(current_ram: &[u8; 0x800]) -> [u8; 4] {
    solo_lives(if current_ram[LIVES_ADDR] <= 9 {
        current_ram[LIVES_ADDR]
    } else {
        0
    })
}

fn is_level_play(ram: &[u8; 0x800]) -> bool {
    lives(ram)[0] > 0
        && ram[PLAYER_X_ADDR] != 0
        && ram[PLAYER_Y_ADDR] >= 0x70
        && ram[PLAYER_Y_PLUS_11_ADDR] >= 0x80
        && ram[PLAYER_Y_PLUS_11_ADDR] < 0xc0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::smoke_single_player_actions;

    #[test]
    fn real_rom_steps_training_actions() {
        smoke_single_player_actions(
            &SUPER_MARIO_BROS_3,
            "Super Mario Bros. 3 (USA) (Rev 1).nes",
            &["RIGHT_A_B", "LEFT_A_B", "UP_A"],
            |_| {},
        );
    }

    fn ram(x: u8, y: u8, y_plus_11: u8, life_count: u8) -> [u8; 0x800] {
        let mut ram = [0u8; 0x800];
        ram[PLAYER_X_ADDR] = x;
        ram[PLAYER_Y_ADDR] = y;
        ram[PLAYER_Y_PLUS_11_ADDR] = y_plus_11;
        ram[LIVES_ADDR] = life_count;
        ram
    }

    #[test]
    fn reward_uses_level_horizontal_progress_only() {
        assert_eq!(
            x_reward(&ram(0x18, 0x80, 0x91, 4), &ram(0x1c, 0x80, 0x91, 4)),
            4.0
        );
        assert_eq!(
            x_reward(&ram(0xf0, 0x80, 0x91, 4), &ram(0x08, 0x80, 0x91, 4)),
            0.0
        );
        assert_eq!(
            x_reward(&ram(0x18, 0x27, 0x09, 4), &ram(0x1c, 0x27, 0x09, 4)),
            0.0
        );
    }

    #[test]
    fn terminal_tracks_pit_or_empty_lives() {
        let prev = [0u8; 0x800];
        assert!(!terminal(&prev, &ram(0x18, 0x80, 0x91, 4)));
        assert!(terminal(&prev, &ram(0x18, 0x80, 0xc0, 4)));
        assert!(terminal(&prev, &ram(0x18, 0x80, 0x91, 0)));
        assert_eq!(
            death_penalty(&ram(0x18, 0x80, 0x91, 4), &ram(0x18, 0x80, 0xc0, 4)),
            -25.0
        );
    }
}
