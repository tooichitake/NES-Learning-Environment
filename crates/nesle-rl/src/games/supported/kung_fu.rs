use nesle_common::action::{NesAction, NesButton};

use crate::games::{solo_lives, solo_reward, GameSpec};

/// Kung Fu (Irem / Nintendo, 1985). The dense reward follows the side-scroller
/// training pattern: floor-direction progress, time cost, and survival cost.
/// RAM map verified against the running ROM (datacrystal + a differential
/// probe): $0531-0535 are five decimal score digits ($0531 most-significant;
/// the displayed score is this value * 10), $005C is lives (3 -> 0 = game over,
/// never 0 during play), $0051 is the game mode (1 = floor-intro banner with
/// input frozen, 2 = active play, 4 = game-over screen), $0058 is zero-based
/// current stage, $0056 is defeat count, $04A6 is Thomas' health bar, and
/// $00D4 is Thomas' screen-space X position. $0390-$0393 are the visible
/// timer digits. Floors 1/3/5 advance right-to-left; floors 2/4 advance
/// left-to-right.
pub static KUNG_FU: GameSpec = GameSpec {
    id: "kung_fu",
    family: "kung_fu",
    gym_id: "NESLE/KungFu",
    display_name: "Kung Fu",
    sha1: "b36ece8fc8330c36716a7eb19874c31fc5f14287",
    players: 1,
    four_score: false,
    mode: None,
    minimal_actions: &KUNG_FU_ACTIONS,
    reward: kung_fu_reward,
    terminal: kung_fu_terminal,
    lives: kung_fu_lives,
    in_transition: None,
    per_agent_lives_termination: false,
};

/// Move / crouch / jump + kick (B) + punch (A) -the buttons that matter for
/// Kung Fu. Up+direction and Up+attack collapse to Up in-game, so they are not
/// part of the minimal set. (full_action_space=True exposes the unified
/// 36-action set instead.)
pub const KUNG_FU_ACTIONS: [NesAction; 9] = [
    NesAction::new("NOOP", 0),
    NesAction::new("RIGHT", NesButton::Right as u8),
    NesAction::new("LEFT", NesButton::Left as u8),
    NesAction::new("B", NesButton::B as u8),
    NesAction::new("A", NesButton::A as u8),
    NesAction::new("DOWN", NesButton::Down as u8),
    NesAction::new("UP", NesButton::Up as u8),
    NesAction::new("DOWN_A", NesButton::Down as u8 | NesButton::A as u8),
    NesAction::new("DOWN_B", NesButton::Down as u8 | NesButton::B as u8),
];

#[cfg(test)]
fn kung_fu_score(ram: &[u8; 0x800]) -> u32 {
    // Five decimal digit-bytes, $0531 most-significant (the HUD shows this value * 10).
    let mut score = 0u32;
    for &addr in &[0x0531usize, 0x0532, 0x0533, 0x0534, 0x0535] {
        score = score * 10 + (ram[addr] as u32).min(9);
    }
    score
}

const GAME_MODE_ADDR: usize = 0x0051;
const STAGE_ADDR: usize = 0x0058;
const LIVES_ADDR: usize = 0x005c;
const THOMAS_X_ADDR: usize = 0x00d4;
const TIMER_DIGIT_START: usize = 0x0390;
const HEALTH_ADDR: usize = 0x04a6;
const ACTIVE_GAME_MODE: u8 = 2;
const MAX_OBSERVED_HEALTH: u8 = 0x4d;

fn kung_fu_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    let x_reward = kung_fu_x_reward(previous_ram, current_ram);
    let floor_clear_reward = kung_fu_floor_clear_reward(previous_ram, current_ram);
    let time_penalty = kung_fu_time_penalty(previous_ram, current_ram);
    let health_penalty = kung_fu_health_penalty(previous_ram, current_ram);
    let life_loss_penalty =
        if current_ram[LIVES_ADDR] < previous_ram[LIVES_ADDR] && previous_ram[LIVES_ADDR] != 0xff {
            -25.0
        } else {
            0.0
        };
    solo_reward(x_reward + floor_clear_reward + time_penalty + health_penalty + life_loss_penalty)
}

fn kung_fu_x_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    if !kung_fu_active_health_state(previous_ram) || !kung_fu_active_health_state(current_ram) {
        return 0.0;
    }
    let delta = current_ram[THOMAS_X_ADDR] as i32 - previous_ram[THOMAS_X_ADDR] as i32;
    if (-8..=8).contains(&delta) {
        delta as f32 * kung_fu_progress_direction(current_ram) * 0.05
    } else {
        0.0
    }
}

fn kung_fu_health_penalty(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    if !kung_fu_active_health_state(previous_ram) || !kung_fu_active_health_state(current_ram) {
        return 0.0;
    }
    let previous = previous_ram[HEALTH_ADDR];
    let current = current_ram[HEALTH_ADDR];
    if current < previous {
        -((previous - current) as f32) * 0.1
    } else {
        0.0
    }
}

fn kung_fu_time_penalty(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    if !kung_fu_active_health_state(previous_ram) || !kung_fu_active_health_state(current_ram) {
        return 0.0;
    }
    let previous = kung_fu_timer(previous_ram);
    let current = kung_fu_timer(current_ram);
    if current < previous && previous - current <= 20 {
        -((previous - current) as f32) * 0.01
    } else {
        0.0
    }
}

fn kung_fu_floor_clear_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    if current_ram[LIVES_ADDR] < previous_ram[LIVES_ADDR] {
        return 0.0;
    }
    let previous = kung_fu_floor(previous_ram);
    let current = kung_fu_floor(current_ram);
    if current > previous && current - previous == 1 {
        50.0
    } else {
        0.0
    }
}

fn kung_fu_timer(ram: &[u8; 0x800]) -> u16 {
    let mut timer = 0u16;
    for offset in 0..4 {
        timer = timer * 10 + (ram[TIMER_DIGIT_START + offset] as u16).min(9);
    }
    timer
}

fn kung_fu_active_health_state(ram: &[u8; 0x800]) -> bool {
    ram[GAME_MODE_ADDR] == ACTIVE_GAME_MODE && (1..=MAX_OBSERVED_HEALTH).contains(&ram[HEALTH_ADDR])
}

fn kung_fu_progress_direction(ram: &[u8; 0x800]) -> f32 {
    let floor = kung_fu_floor(ram);
    if floor % 2 == 1 {
        -1.0
    } else {
        1.0
    }
}

fn kung_fu_floor(ram: &[u8; 0x800]) -> u8 {
    ram[STAGE_ADDR].saturating_add(1)
}

fn kung_fu_terminal(_previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    // Game over = the last life is lost (lives is always 1-3 while playing).
    current_ram[LIVES_ADDR] == 0
}

fn kung_fu_lives(current_ram: &[u8; 0x800]) -> [u8; 4] {
    solo_lives(current_ram[LIVES_ADDR])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::smoke_single_player_actions;

    #[test]
    fn real_rom_steps_training_actions_and_reaches_valid_stage() {
        smoke_single_player_actions(
            &KUNG_FU,
            "Kung Fu (Japan, USA) (En).nes",
            &["LEFT", "UP", "DOWN_A", "DOWN_B"],
            |env| {
                let raw_stage = env.ram()[STAGE_ADDR];
                assert!(
                    raw_stage <= 4,
                    "Kung Fu reset should land on a valid zero-based stage via $0058, got {raw_stage}"
                );
            },
        );
    }

    #[test]
    fn minimal_actions_keep_only_required_attack_combinations() {
        let names: Vec<&str> = KUNG_FU_ACTIONS.iter().map(|action| action.name).collect();
        assert_eq!(
            names,
            vec!["NOOP", "RIGHT", "LEFT", "B", "A", "DOWN", "UP", "DOWN_A", "DOWN_B"]
        );
    }

    #[test]
    fn score_decodes_five_digits_most_significant_first() {
        let mut ram = [0u8; 0x800];
        ram[0x0534] = 3; // displayed 000300
        assert_eq!(kung_fu_score(&ram), 30);
        ram[0x0531] = 1;
        assert_eq!(kung_fu_score(&ram), 10030);
    }

    #[test]
    fn score_delta_does_not_reward_infinite_grunts() {
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        prev[GAME_MODE_ADDR] = ACTIVE_GAME_MODE;
        cur[GAME_MODE_ADDR] = ACTIVE_GAME_MODE;
        prev[HEALTH_ADDR] = 48;
        cur[HEALTH_ADDR] = 48;
        prev[0x0534] = 1; // 10
        cur[0x0534] = 4; //  40
        assert_eq!(kung_fu_reward(&prev, &cur)[0], 0.0);
        assert_eq!(kung_fu_reward(&cur, &prev)[0], 0.0);
    }

    #[test]
    fn reward_shapes_active_floor_horizontal_velocity() {
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        prev[GAME_MODE_ADDR] = ACTIVE_GAME_MODE;
        cur[GAME_MODE_ADDR] = ACTIVE_GAME_MODE;
        prev[HEALTH_ADDR] = 48;
        cur[HEALTH_ADDR] = 48;
        prev[THOMAS_X_ADDR] = 120;
        cur[THOMAS_X_ADDR] = 116;
        assert!((kung_fu_x_reward(&prev, &cur) - 0.2).abs() < 1e-6);

        cur[THOMAS_X_ADDR] = 124;
        assert!((kung_fu_x_reward(&prev, &cur) + 0.2).abs() < 1e-6);

        cur[STAGE_ADDR] = 1;
        assert!((kung_fu_x_reward(&prev, &cur) - 0.2).abs() < 1e-6);

        cur[STAGE_ADDR] = 0;
        cur[THOMAS_X_ADDR] = 240;
        assert_eq!(kung_fu_x_reward(&prev, &cur), 0.0);

        cur[GAME_MODE_ADDR] = 1;
        cur[THOMAS_X_ADDR] = 124;
        assert_eq!(kung_fu_x_reward(&prev, &cur), 0.0);
    }

    #[test]
    fn reward_penalizes_health_and_life_loss() {
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        prev[GAME_MODE_ADDR] = ACTIVE_GAME_MODE;
        cur[GAME_MODE_ADDR] = ACTIVE_GAME_MODE;
        prev[HEALTH_ADDR] = 48;
        cur[HEALTH_ADDR] = 45;
        assert!((kung_fu_reward(&prev, &cur)[0] + 0.3).abs() < 1e-6);

        prev[LIVES_ADDR] = 3;
        cur[LIVES_ADDR] = 2;
        cur[HEALTH_ADDR] = 48;
        assert_eq!(kung_fu_reward(&prev, &cur)[0], -25.0);
    }

    #[test]
    fn reward_bonuses_floor_clear_without_rewarding_death_resets() {
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        prev[LIVES_ADDR] = 3;
        cur[LIVES_ADDR] = 3;
        prev[STAGE_ADDR] = 0;
        cur[STAGE_ADDR] = 1;
        assert_eq!(kung_fu_floor_clear_reward(&prev, &cur), 50.0);
        assert_eq!(kung_fu_reward(&prev, &cur)[0], 50.0);

        cur[STAGE_ADDR] = 3;
        assert_eq!(kung_fu_floor_clear_reward(&prev, &cur), 0.0);

        cur[STAGE_ADDR] = 1;
        cur[LIVES_ADDR] = 2;
        assert_eq!(kung_fu_floor_clear_reward(&prev, &cur), 0.0);
    }

    #[test]
    fn reward_penalizes_timer_decrease() {
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        prev[GAME_MODE_ADDR] = ACTIVE_GAME_MODE;
        cur[GAME_MODE_ADDR] = ACTIVE_GAME_MODE;
        prev[HEALTH_ADDR] = 48;
        cur[HEALTH_ADDR] = 48;
        prev[TIMER_DIGIT_START..TIMER_DIGIT_START + 4].copy_from_slice(&[1, 9, 9, 9]);
        cur[TIMER_DIGIT_START..TIMER_DIGIT_START + 4].copy_from_slice(&[1, 9, 9, 8]);
        assert!((kung_fu_reward(&prev, &cur)[0] + 0.01).abs() < 1e-6);

        cur[TIMER_DIGIT_START..TIMER_DIGIT_START + 4].copy_from_slice(&[1, 9, 0, 0]);
        assert_eq!(kung_fu_time_penalty(&prev, &cur), 0.0);
    }

    #[test]
    fn terminal_only_when_lives_zero() {
        let prev = [0u8; 0x800];
        let mut ram = [0u8; 0x800];
        ram[0x005c] = 3;
        assert!(!kung_fu_terminal(&prev, &ram));
        ram[0x005c] = 1;
        assert!(!kung_fu_terminal(&prev, &ram));
        ram[0x005c] = 0;
        assert!(kung_fu_terminal(&prev, &ram));
        assert_eq!(kung_fu_lives(&ram)[0], 0);
    }
}
