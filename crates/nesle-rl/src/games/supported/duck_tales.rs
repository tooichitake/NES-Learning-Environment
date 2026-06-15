use nesle_common::action::{NesAction, NesButton};

use crate::games::{solo_lives, solo_reward, GameSpec};

pub static DUCK_TALES: GameSpec = GameSpec {
    id: "duck_tales",
    family: "duck_tales",
    gym_id: "NESLE/DuckTales",
    display_name: "DuckTales",
    sha1: "e2004d265e44e851f3c6053314af62a05fc4a595",
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

pub const ACTIONS: [NesAction; 12] = [
    NesAction::new("NOOP", 0),
    NesAction::new("RIGHT", NesButton::Right as u8),
    NesAction::new("LEFT", NesButton::Left as u8),
    NesAction::new("UP", NesButton::Up as u8),
    NesAction::new("DOWN", NesButton::Down as u8),
    NesAction::new("A", NesButton::A as u8),
    NesAction::new("B", NesButton::B as u8),
    NesAction::new("RIGHT_A", NesButton::Right as u8 | NesButton::A as u8),
    NesAction::new("LEFT_A", NesButton::Left as u8 | NesButton::A as u8),
    NesAction::new("DOWN_B", NesButton::Down as u8 | NesButton::B as u8),
    NesAction::new("RIGHT_B", NesButton::Right as u8 | NesButton::B as u8),
    NesAction::new("LEFT_B", NesButton::Left as u8 | NesButton::B as u8),
];

const MONEY_BASE: usize = 0x0700;
const ROOM_ADDR: usize = 0x0040;
const X_ADDR: usize = 0x004d;
const LIVES_ADDR: usize = 0x006a;

fn reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    let progress = if current_ram[ROOM_ADDR] == previous_ram[ROOM_ADDR] {
        let delta = current_ram[X_ADDR] as i16 - previous_ram[X_ADDR] as i16;
        if (1..=6).contains(&delta) {
            delta as f32
        } else {
            0.0
        }
    } else {
        10.0
    };
    solo_reward(
        progress
            + money(current_ram).saturating_sub(money(previous_ram)) as f32 * 0.001
            + death_penalty(previous_ram, current_ram),
    )
}

fn terminal(_previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    current_ram[LIVES_ADDR] == 0xff
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

fn money(ram: &[u8; 0x800]) -> u32 {
    let mut out = 0u32;
    for i in 0..6 {
        let d = ram[MONEY_BASE + i];
        if d > 9 {
            return 0;
        }
        out = out * 10 + d as u32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ram_with_state(money_digits: [u8; 6], room: u8, x: u8, lives_value: u8) -> [u8; 0x800] {
        let mut ram = [0u8; 0x800];
        ram[ROOM_ADDR] = room;
        ram[X_ADDR] = x;
        ram[LIVES_ADDR] = lives_value;
        for (i, digit) in money_digits.into_iter().enumerate() {
            ram[MONEY_BASE + i] = digit;
        }
        ram
    }

    #[test]
    fn reward_uses_room_progress_money_and_life_loss_fixture() {
        let prev = ram_with_state([0, 0, 0, 0, 0, 0], 0, 10, 2);
        let mut cur = ram_with_state([0, 0, 0, 0, 0, 5], 0, 14, 2);
        assert!((reward(&prev, &cur)[0] - 4.005).abs() < 1e-6);

        cur[ROOM_ADDR] = 1;
        assert!(reward(&prev, &cur)[0] >= 10.0);

        cur[LIVES_ADDR] = 1;
        assert!(reward(&prev, &cur)[0] < 0.0);

        cur[MONEY_BASE] = 0xff;
        assert_eq!(money(&cur), 0);
    }

    #[test]
    fn terminal_and_lives_use_underflow_fixture() {
        let prev = [0u8; 0x800];
        let mut ram = ram_with_state([0; 6], 0, 0, 2);
        assert!(!terminal(&prev, &ram));
        assert_eq!(lives(&ram)[0], 2);

        ram[LIVES_ADDR] = 0;
        assert!(!terminal(&prev, &ram));
        assert_eq!(lives(&ram)[0], 0);

        ram[LIVES_ADDR] = 0xff;
        assert!(terminal(&prev, &ram));
        assert_eq!(lives(&ram)[0], 0);
    }

    #[test]
    fn reward_lives_and_terminal_survive_boot_and_menu_ram() {
        let garbage = [0xffu8; 0x800];
        let menu = [0u8; 0x800];

        let _ = reward(&garbage, &garbage);
        let _ = reward(&garbage, &menu);
        let _ = terminal(&garbage, &garbage);
        assert_eq!(lives(&garbage)[0], 0);

        assert_eq!(reward(&menu, &menu)[0], 0.0);
        assert!(!terminal(&menu, &menu));
        assert_eq!(lives(&menu)[0], 0);
    }
}
