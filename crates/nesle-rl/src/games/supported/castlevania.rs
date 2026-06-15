use nesle_common::action::{NesAction, NesButton};

use crate::games::{solo_lives, solo_reward, GameSpec};

/// Castlevania (Konami, 1987). A horizontal action-platformer: the dense
/// reward is forward stage progress from the camera/scroll position $002E
/// (2-byte LE) -- the analogue of SMB's world-x. RAM map verified against the
/// running ROM (datacrystal + a differential probe): walking right drives
/// $002E up monotonically within a stage (it resets at stage transitions,
/// hence the saturating delta); $0045 is health (64 max, -8/hit, 0 = death);
/// $002A is lives (decrements per death, 0 = game over); $0018 is the system
/// state (5 = playing, 0x07 = game over -- a lone death keeps it at 5).
pub static CASTLEVANIA: GameSpec = GameSpec {
    id: "castlevania",
    family: "castlevania",
    gym_id: "NESLE/Castlevania",
    display_name: "Castlevania",
    sha1: "4e3ef47de86a941c37359bb3acfa5542a0fa7876",
    players: 1,
    four_score: false,
    mode: None,
    minimal_actions: &CASTLEVANIA_ACTIONS,
    reward: castlevania_reward,
    terminal: castlevania_terminal,
    lives: castlevania_lives,
    in_transition: None,
    per_agent_lives_termination: false,
};

/// Simon's moveset: walk / jump (A) / whip (B) / walk-whip / stairs (Up/Down) /
/// crouch (Down) / kneel-whip (Down+B, only way to hit low enemies like Medusa
/// Heads + bats) / sub-weapon throw (Up+B, only way to use the cross / axe /
/// holy water / dagger). Empirically verified in-emulator: A jumps (player-Y
/// drops 32 px), B whips without changing Y, Down+B composes the two (the
/// kneel-whip sprite frame), Up+B differs from B-only on the sub-weapon
/// throw-flag. Down+A is omitted because A wins (you can't crouch + jump --
/// it's just a jump). (full_action_space=True exposes the unified 36-action
/// set instead.)
pub const CASTLEVANIA_ACTIONS: [NesAction; 13] = [
    NesAction::new("NOOP", 0),
    NesAction::new("RIGHT", NesButton::Right as u8),
    NesAction::new("LEFT", NesButton::Left as u8),
    NesAction::new("A", NesButton::A as u8),
    NesAction::new("B", NesButton::B as u8),
    NesAction::new("RIGHT_A", NesButton::Right as u8 | NesButton::A as u8),
    NesAction::new("LEFT_A", NesButton::Left as u8 | NesButton::A as u8),
    NesAction::new("RIGHT_B", NesButton::Right as u8 | NesButton::B as u8),
    NesAction::new("LEFT_B", NesButton::Left as u8 | NesButton::B as u8),
    NesAction::new("UP", NesButton::Up as u8),
    NesAction::new("DOWN", NesButton::Down as u8),
    NesAction::new("UP_B", NesButton::Up as u8 | NesButton::B as u8),
    NesAction::new("DOWN_B", NesButton::Down as u8 | NesButton::B as u8),
];

fn castlevania_view(ram: &[u8; 0x800]) -> u16 {
    // Camera/scroll position (2-byte LE): monotonic forward progress within a stage.
    (ram[0x002e] as u16) | ((ram[0x002f] as u16) << 8)
}

fn castlevania_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    // Cap stage-transition garbage: $002E/$002F briefly reads 0xFF80 entering the castle (a real scroll is <= one screen).
    let delta = castlevania_view(current_ram).saturating_sub(castlevania_view(previous_ram));
    solo_reward(if delta <= 256 { delta as f32 } else { 0.0 })
}

fn castlevania_terminal(_previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    // Game over: system state $0018 == 0x07, or all lives lost ($002A == 0). A single death keeps $0018 == 5.
    current_ram[0x0018] == 0x07 || current_ram[0x002a] == 0
}

fn castlevania_lives(current_ram: &[u8; 0x800]) -> [u8; 4] {
    solo_lives(current_ram[0x002a])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_is_little_endian_2byte() {
        let mut ram = [0u8; 0x800];
        ram[0x002e] = 0x12;
        ram[0x002f] = 0x03;
        assert_eq!(castlevania_view(&ram), 0x0312);
    }

    #[test]
    fn reward_is_saturating_forward_view_delta() {
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        prev[0x002e] = 100;
        cur[0x002e] = 150;
        assert_eq!(castlevania_reward(&prev, &cur)[0], 50.0);
        // stage reset / knockback -> clamp to 0, never negative
        assert_eq!(castlevania_reward(&cur, &prev)[0], 0.0);
    }

    #[test]
    fn reward_caps_stage_transition_garbage() {
        // Entering the castle, $002E/$002F briefly reads 0xFF80 (65408); that jump must not become a ~65k reward.
        let prev = [0u8; 0x800];
        let mut garbage = [0u8; 0x800];
        garbage[0x002e] = 0x80;
        garbage[0x002f] = 0xff;
        assert_eq!(castlevania_reward(&prev, &garbage)[0], 0.0);
        let mut cur = [0u8; 0x800];
        cur[0x002e] = 40;
        assert_eq!(castlevania_reward(&prev, &cur)[0], 40.0);
    }

    #[test]
    fn action_set_includes_kneel_whip_and_sub_weapon() {
        // The kneel-whip (DOWN+B) and sub-weapon throw (UP+B) are mechanically distinct from any single button, so the minimal set must expose them.
        let has = |name: &str, mask: u8| {
            CASTLEVANIA_ACTIONS
                .iter()
                .any(|a| a.name == name && a.mask == mask)
        };
        assert!(has("UP_B", NesButton::Up as u8 | NesButton::B as u8));
        assert!(has("DOWN_B", NesButton::Down as u8 | NesButton::B as u8));
        assert!(has("A", NesButton::A as u8));
        assert!(has("B", NesButton::B as u8));
    }

    #[test]
    fn terminal_on_game_over_state_or_zero_lives() {
        let prev = [0u8; 0x800];
        let mut ram = [0u8; 0x800];
        ram[0x002a] = 4;
        ram[0x0018] = 5;
        assert!(!castlevania_terminal(&prev, &ram));
        ram[0x0018] = 0x07;
        assert!(castlevania_terminal(&prev, &ram));
        ram[0x0018] = 5;
        ram[0x002a] = 0;
        assert!(castlevania_terminal(&prev, &ram));
        ram[0x002a] = 4;
        assert_eq!(castlevania_lives(&ram)[0], 4);
    }
}
