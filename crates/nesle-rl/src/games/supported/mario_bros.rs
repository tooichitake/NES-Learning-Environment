use nesle_common::action::{NesAction, NesButton};

use crate::games::{solo_lives, solo_reward, GameSpec};

pub static MARIO_BROS: GameSpec = GameSpec {
    id: "mario_bros",
    family: "mario_bros",
    gym_id: "NESLE/MarioBros",
    display_name: "Mario Bros.",
    sha1: "314b6e46e814f955b52ac954f67dab849582fe77",
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

pub const ACTIONS: [NesAction; 6] = [
    NesAction::new("NOOP", 0),
    NesAction::new("LEFT", NesButton::Left as u8),
    NesAction::new("RIGHT", NesButton::Right as u8),
    NesAction::new("A", NesButton::A as u8),
    NesAction::new("LEFT_A", NesButton::Left as u8 | NesButton::A as u8),
    NesAction::new("RIGHT_A", NesButton::Right as u8 | NesButton::A as u8),
];

const LIVES_ADDR: usize = 0x0048;
const SCORE_START: usize = 0x0095;
const MARIO_ENTITY_ADDR: usize = 0x0300;

fn reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    solo_reward(score_reward(previous_ram, current_ram) + life_penalty(previous_ram, current_ram))
}

fn score_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    let delta = score(current_ram) as i32 - score(previous_ram) as i32;
    if (1..=50_000).contains(&delta) {
        delta as f32 / 100.0
    } else {
        0.0
    }
}

fn life_penalty(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    let prev = lives(previous_ram)[0];
    let cur = lives(current_ram)[0];
    if prev > 0 && cur < prev {
        -10.0
    } else {
        0.0
    }
}

fn terminal(_previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    current_ram[MARIO_ENTITY_ADDR] != 0 && lives(current_ram)[0] == 0
}

fn lives(current_ram: &[u8; 0x800]) -> [u8; 4] {
    solo_lives(if current_ram[LIVES_ADDR] <= 9 {
        current_ram[LIVES_ADDR]
    } else {
        0
    })
}

fn score(ram: &[u8; 0x800]) -> u32 {
    bcd_pair(ram[SCORE_START]) * 10_000
        + bcd_pair(ram[SCORE_START + 1]) * 100
        + bcd_pair(ram[SCORE_START + 2])
}

fn bcd_pair(value: u8) -> u32 {
    ((value >> 4) as u32) * 10 + (value & 0x0f) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ram(score_bytes: [u8; 3], life_count: u8) -> [u8; 0x800] {
        let mut ram = [0u8; 0x800];
        ram[SCORE_START..SCORE_START + 3].copy_from_slice(&score_bytes);
        ram[LIVES_ADDR] = life_count;
        ram[MARIO_ENTITY_ADDR] = 1;
        ram
    }

    #[test]
    fn reward_uses_bcd_score_delta_and_life_loss() {
        let prev = ram([0x00, 0x01, 0x00], 2);
        let cur = ram([0x00, 0x02, 0x00], 2);
        assert_eq!(score_reward(&prev, &cur), 1.0);

        let dead = ram([0x00, 0x02, 0x00], 1);
        assert_eq!(life_penalty(&cur, &dead), -10.0);
        assert_eq!(reward(&cur, &dead)[0], -10.0);
    }

    #[test]
    fn terminal_waits_for_gameplay_lives_to_empty() {
        let prev = [0u8; 0x800];
        let mut boot = [0u8; 0x800];
        assert!(!terminal(&prev, &boot));
        boot[MARIO_ENTITY_ADDR] = 1;
        assert!(terminal(&prev, &boot));
    }
}
