use nesle_common::action::{NesAction, NesButton};

use crate::games::{solo_lives, solo_reward, GameSpec};

/// Bomberman (Hudson Soft, 1985), single-player Normal Game.
///
/// RAM map basis: Data Crystal + Ragey's Bomberman reference list the stage at
/// $0058, lives at $0068, P1 tile/coordinate bytes at $0028-$002B, timer at
/// $0093, and remote detonation at $0077. The score RAM still needs a live
/// score-event probe before it should be used for reward, so reward stays on
/// verified stage, timer, and death signals.
pub static BOMBERMAN: GameSpec = GameSpec {
    id: "bomberman",
    family: "bomberman",
    gym_id: "NESLE/Bomberman",
    display_name: "Bomberman",
    sha1: "1a38860f7583e4619dacd293e80d80e7bd4ee021",
    players: 1,
    four_score: false,
    mode: None,
    minimal_actions: &BOMBERMAN_ACTIONS,
    reward: bomberman_reward,
    terminal: bomberman_terminal,
    lives: bomberman_lives,
    in_transition: None,
    per_agent_lives_termination: false,};

pub const BOMBERMAN_ACTIONS: [NesAction; 12] = [
    NesAction::new("NOOP", 0),
    NesAction::new("UP", NesButton::Up as u8),
    NesAction::new("DOWN", NesButton::Down as u8),
    NesAction::new("LEFT", NesButton::Left as u8),
    NesAction::new("RIGHT", NesButton::Right as u8),
    NesAction::new("A", NesButton::A as u8),
    NesAction::new("B", NesButton::B as u8),
    NesAction::new("UP_A", NesButton::Up as u8 | NesButton::A as u8),
    NesAction::new("DOWN_A", NesButton::Down as u8 | NesButton::A as u8),
    NesAction::new("LEFT_A", NesButton::Left as u8 | NesButton::A as u8),
    NesAction::new("RIGHT_A", NesButton::Right as u8 | NesButton::A as u8),
    NesAction::new("AB", NesButton::A as u8 | NesButton::B as u8),
];

const STAGE_ADDR: usize = 0x0058;
const LIVES_ADDR: usize = 0x0068;
const TIMER_ADDR: usize = 0x0093;

fn bomberman_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    solo_reward({
        let death_penalty = if current_ram[LIVES_ADDR] < previous_ram[LIVES_ADDR]
            && previous_ram[LIVES_ADDR] != 0xff
        {
            -25.0
        } else {
            0.0
        };
        let prev_stage = previous_ram[STAGE_ADDR];
        let cur_stage = current_ram[STAGE_ADDR];
        let stage_clear = if cur_stage > prev_stage && cur_stage - prev_stage <= 2 {
            50.0
        } else {
            0.0
        };
        death_penalty + stage_clear + bomberman_time_penalty(previous_ram, current_ram)
    })
}

fn bomberman_time_penalty(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    let previous = previous_ram[TIMER_ADDR];
    let current = current_ram[TIMER_ADDR];
    if current < previous && previous - current <= 5 {
        -((previous - current) as f32) * 0.01
    } else {
        0.0
    }
}

fn bomberman_terminal(_previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    current_ram[STAGE_ADDR] != 0xff && current_ram[LIVES_ADDR] == 0xff
}

fn bomberman_lives(current_ram: &[u8; 0x800]) -> [u8; 4] {
    solo_lives(if current_ram[LIVES_ADDR] == 0xff {
        0
    } else {
        current_ram[LIVES_ADDR]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_penalizes_life_loss_and_bonuses_stage_clear() {
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        prev[LIVES_ADDR] = 2;
        cur[LIVES_ADDR] = 1;
        assert_eq!(bomberman_reward(&prev, &cur)[0], -25.0);
        prev[LIVES_ADDR] = 2;
        cur[LIVES_ADDR] = 2;
        prev[STAGE_ADDR] = 1;
        cur[STAGE_ADDR] = 2;
        assert_eq!(bomberman_reward(&prev, &cur)[0], 50.0);
    }

    #[test]
    fn reward_penalizes_timer_decrease() {
        let mut prev = [0u8; 0x800];
        let mut cur = [0u8; 0x800];
        prev[TIMER_ADDR] = 200;
        cur[TIMER_ADDR] = 199;
        assert_eq!(bomberman_reward(&prev, &cur)[0], -0.01);

        cur[TIMER_ADDR] = 190;
        assert_eq!(bomberman_time_penalty(&prev, &cur), 0.0);
    }

    #[test]
    fn terminal_uses_life_underflow_after_init() {
        let prev = [0u8; 0x800];
        let mut ram = [0u8; 0x800];
        ram[STAGE_ADDR] = 1;
        ram[LIVES_ADDR] = 0;
        assert!(!bomberman_terminal(&prev, &ram));
        ram[LIVES_ADDR] = 0xff;
        assert!(bomberman_terminal(&prev, &ram));
        assert_eq!(bomberman_lives(&ram)[0], 0);
        ram[STAGE_ADDR] = 0xff;
        assert!(!bomberman_terminal(&prev, &ram));
    }
}
