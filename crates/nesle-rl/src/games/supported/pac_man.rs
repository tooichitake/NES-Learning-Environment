use nesle_common::action::{NesAction, NesButton};

use crate::games::{solo_lives, solo_reward, GameSpec};

pub static PAC_MAN: GameSpec = GameSpec {
    id: "pac_man",
    family: "pac_man",
    gym_id: "NESLE/PacMan",
    display_name: "Pac-Man",
    sha1: "ef76cebddc57b7c96cfc95b55dcc712fd5934b2c",
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

pub const ACTIONS: [NesAction; 5] = [
    NesAction::new("NOOP", 0),
    NesAction::new("UP", NesButton::Up as u8),
    NesAction::new("DOWN", NesButton::Down as u8),
    NesAction::new("LEFT", NesButton::Left as u8),
    NesAction::new("RIGHT", NesButton::Right as u8),
];

const SCORE_BASE: usize = 0x0070;
const LIVES_ADDR: usize = 0x004e;
const MODE_ADDR: usize = 0x0040;

fn reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    solo_reward(
        score(current_ram).saturating_sub(score(previous_ram)) as f32 * 0.01
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

fn death_penalty(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    let prev = previous_ram[LIVES_ADDR];
    let cur = current_ram[LIVES_ADDR];
    if cur < prev && prev != 0xff {
        -25.0
    } else {
        0.0
    }
}

fn score(ram: &[u8; 0x800]) -> u32 {
    bcd_le_score(ram, SCORE_BASE, 3)
}

fn bcd_le_score(ram: &[u8; 0x800], base: usize, len: usize) -> u32 {
    let mut multiplier = 1u32;
    let mut out = 0u32;
    for i in 0..len {
        let byte = ram[base + i];
        let lo = byte & 0x0f;
        let hi = byte >> 4;
        if lo > 9 || hi > 9 {
            return 0;
        }
        out += lo as u32 * multiplier;
        out += hi as u32 * multiplier * 10;
        multiplier *= 100;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ram_with_state(score_bytes: [u8; 3], lives_value: u8, mode: u8) -> [u8; 0x800] {
        let mut ram = [0u8; 0x800];
        ram[LIVES_ADDR] = lives_value;
        ram[MODE_ADDR] = mode;
        for (i, byte) in score_bytes.into_iter().enumerate() {
            ram[SCORE_BASE + i] = byte;
        }
        ram
    }

    #[test]
    fn reward_reads_bcd_score_and_life_loss_fixture() {
        let prev = ram_with_state([0x00, 0x00, 0x00], 3, 1);
        let mut cur = ram_with_state([0x10, 0x00, 0x00], 3, 1);
        assert!((reward(&prev, &cur)[0] - 0.10).abs() < 1e-6);

        cur[LIVES_ADDR] = 2;
        assert!(reward(&prev, &cur)[0] < -20.0);

        cur[SCORE_BASE] = 0xfa;
        assert_eq!(score(&cur), 0);
    }

    #[test]
    fn terminal_and_lives_use_mode_guard_fixture() {
        let prev = [0u8; 0x800];
        let mut ram = ram_with_state([0; 3], 2, 1);
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
        let mut ram = [0xffu8; 0x800];
        let menu = [0u8; 0x800];

        assert_eq!(reward(&ram, &ram)[0], 0.0);
        assert!(!terminal(&ram, &ram));
        assert_eq!(lives(&ram)[0], 0);

        ram[MODE_ADDR] = 1;
        assert!(terminal(&ram, &ram));

        assert_eq!(reward(&menu, &menu)[0], 0.0);
        assert!(!terminal(&menu, &menu));
        assert_eq!(lives(&menu)[0], 0);
    }
}
