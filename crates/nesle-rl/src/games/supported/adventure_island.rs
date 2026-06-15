use nesle_common::action::{NesAction, NesButton};

use crate::games::{solo_lives, solo_reward, GameSpec};

pub static ADVENTURE_ISLAND: GameSpec = GameSpec {
    id: "adventure_island",
    family: "adventure_island",
    gym_id: "NESLE/AdventureIsland",
    display_name: "Adventure Island",
    sha1: "ac34732d64566947891cd1ece23216db32a06eae",
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

pub const ACTIONS: [NesAction; 10] = [
    NesAction::new("NOOP", 0),
    NesAction::new("RIGHT", NesButton::Right as u8),
    NesAction::new("LEFT", NesButton::Left as u8),
    NesAction::new("A", NesButton::A as u8),
    NesAction::new("B", NesButton::B as u8),
    NesAction::new("RIGHT_A", NesButton::Right as u8 | NesButton::A as u8),
    NesAction::new("RIGHT_B", NesButton::Right as u8 | NesButton::B as u8),
    NesAction::new("LEFT_A", NesButton::Left as u8 | NesButton::A as u8),
    NesAction::new("LEFT_B", NesButton::Left as u8 | NesButton::B as u8),
    NesAction::new(
        "RIGHT_AB",
        NesButton::Right as u8 | NesButton::A as u8 | NesButton::B as u8,
    ),
];

const SCROLL_ADDR: usize = 0x0000;
const PAGE_ADDR: usize = 0x0030;
const LIVES_ADDR: usize = 0x04a5;

fn reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    solo_reward(
        clipped_delta(progress(current_ram), progress(previous_ram), 32) as f32
            + death_penalty(previous_ram, current_ram),
    )
}

fn terminal(_previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    current_ram[LIVES_ADDR] == 0xff
        && (current_ram[SCROLL_ADDR] != 0xff || current_ram[PAGE_ADDR] != 0xff)
}

fn lives(current_ram: &[u8; 0x800]) -> [u8; 4] {
    let value = current_ram[LIVES_ADDR];
    solo_lives(if value <= 9 { value } else { 0 })
}

fn progress(ram: &[u8; 0x800]) -> i32 {
    ((ram[PAGE_ADDR] as i32) << 8) | ram[SCROLL_ADDR] as i32
}

fn death_penalty(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    let prev = previous_ram[LIVES_ADDR];
    let cur = current_ram[LIVES_ADDR];
    if prev <= 9 && (cur < prev || cur == 0xff) {
        -25.0
    } else {
        0.0
    }
}

fn clipped_delta(cur: i32, prev: i32, max: i32) -> i32 {
    let delta = cur - prev;
    if (1..=max).contains(&delta) {
        delta
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ram_with_state(scroll: u8, page: u8, lives_value: u8) -> [u8; 0x800] {
        let mut ram = [0u8; 0x800];
        ram[SCROLL_ADDR] = scroll;
        ram[PAGE_ADDR] = page;
        ram[LIVES_ADDR] = lives_value;
        ram
    }

    #[test]
    fn reward_uses_scroll_progress_fixture() {
        let prev = ram_with_state(10, 2, 2);
        let mut cur = ram_with_state(13, 2, 2);
        assert_eq!(reward(&prev, &cur)[0], 3.0);

        cur[SCROLL_ADDR] = 200;
        assert_eq!(clipped_delta(progress(&cur), progress(&prev), 8), 0);

        cur = ram_with_state(10, 2, 1);
        assert_eq!(death_penalty(&prev, &cur), -25.0);
        cur[LIVES_ADDR] = 0xff;
        assert_eq!(death_penalty(&prev, &cur), -25.0);
    }

    #[test]
    fn terminal_and_lives_track_life_underflow() {
        let prev = [0u8; 0x800];
        let mut ram = ram_with_state(0, 2, 2);
        assert!(!terminal(&prev, &ram));
        assert_eq!(lives(&ram)[0], 2);

        ram[LIVES_ADDR] = 1;
        assert!(!terminal(&prev, &ram));
        assert_eq!(lives(&ram)[0], 1);

        ram[LIVES_ADDR] = 0xff;
        assert!(terminal(&prev, &ram));
        assert_eq!(lives(&ram)[0], 0);

        ram[SCROLL_ADDR] = 0xff;
        ram[PAGE_ADDR] = 0xff;
        assert!(!terminal(&prev, &ram));
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
