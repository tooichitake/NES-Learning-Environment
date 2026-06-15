use nesle_common::action::{NesAction, NesButton};

use crate::games::{solo_lives, solo_reward, GameSpec};

// gym SIMPLE_MOVEMENT + DOWN (8): NOOP, RIGHT family, A, LEFT, DOWN (DOWN needed for 8-4's down-pipes).
pub const SMB1_ACTIONS: [NesAction; 8] = [
    NesAction::new("NOOP", 0),
    NesAction::new("RIGHT", NesButton::Right as u8),
    NesAction::new("RIGHT_A", NesButton::Right as u8 | NesButton::A as u8),
    NesAction::new("RIGHT_B", NesButton::Right as u8 | NesButton::B as u8),
    NesAction::new(
        "RIGHT_A_B",
        NesButton::Right as u8 | NesButton::A as u8 | NesButton::B as u8,
    ),
    NesAction::new("A", NesButton::A as u8),
    NesAction::new("LEFT", NesButton::Left as u8),
    NesAction::new("DOWN", NesButton::Down as u8),
];

pub static SUPER_MARIO_BROS: GameSpec = GameSpec {
    id: "super_mario_bros",
    family: "super_mario_bros",
    gym_id: "NESLE/SuperMarioBros",
    display_name: "Super Mario Bros.",
    sha1: "ab30029efec6ccfc5d65dfda7fbc6e6489a80805",
    players: 1,
    four_score: false,
    mode: None,
    minimal_actions: &SMB1_ACTIONS,
    reward,
    terminal,
    lives,
    in_transition: Some(in_transition),
    per_agent_lives_termination: false,
};

const X_REWARD_SCALE: f32 = 1.0;
const TIME_PENALTY_SCALE: f32 = 8.0;
const DEATH_PENALTY: f32 = -50.0;
const LEVEL_COMPLETE_BONUS: f32 = 200.0;

// gym-super-mario-bros reward: x-velocity + clock penalty + death penalty + one-off level-clear bonus.
fn reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    solo_reward(
        X_REWARD_SCALE * x_reward(previous_ram, current_ram)
            + TIME_PENALTY_SCALE * time_penalty(previous_ram, current_ram)
            + death_penalty(previous_ram, current_ram)
            + level_complete_bonus(previous_ram, current_ram),
    )
}

// Per-frame movement; magnitude > 5 is a level/death x-reset (scores 0). gym-super-mario-bros _x_reward.
fn x_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    let delta = world_x(current_ram) as i32 - world_x(previous_ram) as i32;
    if (-5..=5).contains(&delta) {
        delta as f32
    } else {
        0.0
    }
}

// Clock only ticks down, so a positive delta is a reset (-> 0); else the delta is the penalty. _time_penalty.
fn time_penalty(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    let delta = game_time(current_ram) as i32 - game_time(previous_ram) as i32;
    if delta > 0 {
        0.0
    } else {
        delta as f32
    }
}

// Fires once on the frame Mario enters the dying/dead state. gym-super-mario-bros _death_penalty.
fn death_penalty(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    if is_dead_or_dying(current_ram) && !is_dead_or_dying(previous_ram) {
        DEATH_PENALTY
    } else {
        0.0
    }
}

// One-off reward on the frame Mario reaches the flag (enters the level-clear cutscene).
fn level_complete_bonus(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> f32 {
    if flag_get(current_ram) && !flag_get(previous_ram) {
        LEVEL_COMPLETE_BONUS
    } else {
        0.0
    }
}

// Episode ends only on game over (all lives spent); the flag does not end it.
fn terminal(_previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    game_over(current_ram)
}

fn lives(current_ram: &[u8; 0x800]) -> [u8; 4] {
    solo_lives(if current_ram[0x075a] == 0xff {
        0
    } else {
        current_ram[0x075a]
    })
}

fn world_x(ram: &[u8; 0x800]) -> u16 {
    ((ram[0x006d] as u16) << 8) | ram[0x0086] as u16
}

// Time left as a 3-digit decimal at $07F8-$07FA (gym-super-mario-bros _time).
fn game_time(ram: &[u8; 0x800]) -> u16 {
    (ram[0x07f8] as u16) * 100 + (ram[0x07f9] as u16) * 10 + (ram[0x07fa] as u16)
}

// Player state $000E (0x0B dying, 0x06 dead) or y-viewport $00B5 > 1 (pit fall). _is_dying / _is_dead.
fn is_dead_or_dying(ram: &[u8; 0x800]) -> bool {
    ram[0x000e] == 0x0b || ram[0x000e] == 0x06 || ram[0x00b5] > 1
}

// $075A underflows to 0xFF when all lives are spent; $0770 guards the power-on garbage. _is_game_over.
fn game_over(ram: &[u8; 0x800]) -> bool {
    ram[0x0770] != 0xff && ram[0x075a] == 0xff
}

// Flagpole reached: $000E is 0x04/0x05 (flagpole-grab / castle-walk); drives the level-clear bonus.
fn flag_get(ram: &[u8; 0x800]) -> bool {
    matches!(ram[0x000e], 0x04 | 0x05)
}

// Auto states ($000E) the env fast-forwards: 0x00 load, 0x02 pipe, 0x03 down-pipe, 0x04 flagpole, 0x05 castle, 0x07 entering. Play/vine-climb/death are not skipped.
fn in_transition(ram: &[u8; 0x800]) -> bool {
    matches!(ram[0x000e], 0x00 | 0x02 | 0x03 | 0x04 | 0x05 | 0x07)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::smoke_single_player_actions;

    #[test]
    fn real_rom_steps_training_actions() {
        smoke_single_player_actions(
            &SUPER_MARIO_BROS,
            "super-mario-bros.nes",
            &["RIGHT_B", "RIGHT_A_B"],
            |_| {},
        );
    }

    #[test]
    fn reward_is_velocity_plus_time_and_death_penalties() {
        let mut prev = [0; 0x800];
        let mut cur = [0; 0x800];
        prev[0x0086] = 10;
        cur[0x0086] = 13;
        assert_eq!(x_reward(&prev, &cur), 3.0); // forward step within +-5
        cur[0x0086] = 200;
        assert_eq!(x_reward(&prev, &cur), 0.0); // level/death x-reset rejected
        let mut p = [0; 0x800];
        let mut c = [0; 0x800];
        p[0x07fa] = 5;
        c[0x07fa] = 3;
        assert_eq!(time_penalty(&p, &c), -2.0); // clock ticking down
        assert_eq!(reward(&p, &c)[0], -16.0);
        let alive = [0; 0x800];
        let mut dying = [0; 0x800];
        dying[0x000e] = 0x0b;
        assert_eq!(death_penalty(&alive, &dying), DEATH_PENALTY);
        assert_eq!(death_penalty(&dying, &dying), 0.0); // not repeated each frame
    }

    #[test]
    fn terminal_tracks_life_underflow() {
        let prev = [0u8; 0x800];
        let mut ram = [0; 0x800];
        assert!(!terminal(&prev, &ram));
        ram[0x000e] = 0x0b;
        assert!(!terminal(&prev, &ram));
        ram[0x000e] = 0;
        ram[0x0770] = 1;
        ram[0x075a] = 0xff;
        assert!(terminal(&prev, &ram));
        ram[0x0770] = 0xff;
        assert!(!terminal(&prev, &ram));
    }

    #[test]
    fn flag_clear_rewards_and_continues() {
        let mut normal = [0; 0x800];
        normal[0x000e] = 0x08;
        let mut flag = [0; 0x800];
        flag[0x000e] = 0x04;
        // reaching the flag does NOT end the episode -- play continues to the next level
        assert!(!terminal(&normal, &flag));
        // +bonus fires once, on the frame Mario enters the flagpole state
        assert_eq!(level_complete_bonus(&normal, &flag), LEVEL_COMPLETE_BONUS);
        assert_eq!(level_complete_bonus(&flag, &flag), 0.0);
    }

    #[test]
    fn transition_skips_dead_zone_but_not_play_or_death() {
        let state = |s: u8| {
            let mut r = [0u8; 0x800];
            r[0x000e] = s;
            r
        };
        // auto dead zones (load / pipe / down-pipe / flagpole / castle / entering) -> fast-forward
        for s in [0x00u8, 0x02, 0x03, 0x04, 0x05, 0x07] {
            assert!(in_transition(&state(s)), "0x{s:02X} should fast-forward");
        }
        // play, interactive vine-climb, and death -> NOT fast-forwarded
        for s in [0x08u8, 0x01, 0x06, 0x0b] {
            assert!(!in_transition(&state(s)), "0x{s:02X} must not be skipped");
        }
    }

    // End-to-end: a recorded 1-1 clear must fast-forward the cutscene into the next level.
    #[test]
    #[ignore = "slow: replays a full 1-1 (~14s in debug); run with `cargo test -- --ignored`"]
    fn fast_forward_skips_level_clear_cutscene() {
        let Some(path) = crate::test_support::rom_path("super-mario-bros.nes") else {
            return;
        };
        let mut env = crate::NesEnv::new(&SUPER_MARIO_BROS);
        env.set_action_repeat(4, 0.0).unwrap();
        env.load_rom_bytes(&std::fs::read(path).unwrap()).unwrap();
        env.set_start_state_id("level_1_1").unwrap();
        env.reset().unwrap();
        let (mut saw_dead_zone, mut reached_next) = (false, false);
        for &mask in include_bytes!("smb_1_1_clear.bin") {
            env.step(&[mask]).unwrap();
            let ram = env.ram();
            saw_dead_zone |= matches!(ram[0x000e], 0x00 | 0x02 | 0x03 | 0x04 | 0x05 | 0x07);
            if ram[0x075c] >= 1 {
                reached_next = true;
                break;
            }
        }
        assert!(
            reached_next,
            "must fast-forward straight into the next level"
        );
        assert!(
            !saw_dead_zone,
            "agent must never observe a cutscene/load state"
        );
    }
}
