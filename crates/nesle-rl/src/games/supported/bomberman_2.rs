use nesle_common::action::{NesAction, NesButton};

use crate::games::{solo_lives, solo_reward, GameSpec};

/// Bomberman II (Hudson Soft, 1991), NES, MMC1. The cart packs two head-to-head
/// modes that share one RAM map: VS Mode (2P, no adapter) and Battle Mode (3P,
/// needs the Four Score). One round = one episode (PettingZoo-Atari-Warlords
/// pattern, last-bomberman-standing). Sparse reward: -1 the frame a player's
/// alive flag goes 1- (bombed out), +1 to the survivor when the round ends
/// (exactly one alive AND - alive last frame). `per_agent_lives_termination`
/// is on so a dead player terminates immediately while survivors keep playing.
///
/// RAM map (verified by recorded-replay differential + set_ram poke, see
/// `.claude/skills/rust-nesle-rl-games/references/ram-maps.md`):
///   $0069 = P1 alive (1 = alive, 0 = dead)
///   $006A = P2 alive
///   $006B = P3 alive
///   $006C = P4 alive (initialised to 1; unused in 2P / 3P modes)
///   $0049 = game mode (00 Normal, 01 VS, 02 Battle, 03 Continue)
///   $0072-0075 = per-player X (P1..P4); $0078-007B = per-player Y
///
/// Reset routine drives power-on -(intro cutscene Start skip) -(disclaimer
/// Start skip) -(title "PUSH START!" Start) -mode-select menu -DOWN to
/// VS/BATTLE -A -match-select -A on "1 WIN MATCH" -gameplay first-
/// controllable frame. Both modes auto-pick 1 WIN (shortest match = 1 round =
/// 1 episode). For 3P Battle, Four Score on the spec auto-enables the adapter
/// at `load_rom_bytes`, and the cart silently uses ports 0-.
pub static BOMBERMAN_2_VS_2P: GameSpec = GameSpec {
    id: "bomberman_2_vs_2p",
    family: "bomberman_2",
    gym_id: "NESLE/Bomberman2-VS-v0",
    display_name: "Bomberman 2",
    sha1: "ac670d3e8511c52e29edff08c82b45b64c446ba1",
    players: 2,
    four_score: false,
    mode: Some("VS"),
    minimal_actions: &BOMBERMAN_2_BATTLE_ACTIONS,
    reward: bomberman_2_vs_reward,
    terminal: bomberman_2_terminal_2p,
    lives: bomberman_2_lives,
    in_transition: None,
    per_agent_lives_termination: true,
};

pub static BOMBERMAN_2_BATTLE_3P: GameSpec = GameSpec {
    id: "bomberman_2_battle_3p",
    family: "bomberman_2",
    gym_id: "NESLE/Bomberman2-Battle-v0",
    display_name: "Bomberman 2",
    sha1: "ac670d3e8511c52e29edff08c82b45b64c446ba1",
    players: 3,
    four_score: true,
    mode: Some("Battle"),
    minimal_actions: &BOMBERMAN_2_BATTLE_ACTIONS,
    reward: bomberman_2_reward_3p,
    terminal: bomberman_2_terminal_3p,
    lives: bomberman_2_lives,
    in_transition: None,
    per_agent_lives_termination: true,
};

/// Bomberman II (Hudson Soft, 1991), NES, MMC1 -single-player NORMAL MODE
/// story campaign (the 50-area dungeon-crawl, as opposed to VS_2P / BATTLE_3P
/// which live in the same file). Score-driven: dense reward = on-screen
/// score delta (10 pts per soft brick, more per enemy / item). Lives drain on
/// death; respawn until the underflow sentinel hits.
///
/// RAM map (poke-verified, see `.claude/skills/rust-nesle-rl-games/references/
/// ram-maps.md`):
///   $0049 = game-mode byte (0x00 = Normal at the first controllable frame;
///           0xff during boot before the cart init). Used here as the boot-
///           garbage guard mirroring SMB's `$0770 != 0xff` engine-state check.
///   $03D0-$03D7 = 8-byte decimal score (one digit per byte, MSB at $03D0).
///           Verified by poking each byte to 9 and watching the HUD redraw
///           "99999999" on the next frame; the cart re-reads these bytes
///           every frame to repopulate the nametable shadow at $052A-$0531.
///   $04E5 = lives counter. HUD "LEFT N" reads it directly:
///           starts at 0x02 (LEFT 2), decrements on each respawn (2 -> 1 -> 0
///           -the last on-screen value is "LEFT 0", which is the FINAL life),
///           then underflows to 0xff on the death after lives=0 (the
///           game-over banner). Triple-verified: HUD readout + poke($04E5,
///           7) -> HUD "LEFT 7" + the full natural game-over trace at the
///           death of the last life.
///   $0069 = P1 alive flag (shared with the multi-player map). 1 = alive,
///           0 = dying / waiting for respawn / on the game-over screen.
///           Not used in this spec, but worth knowing as a corroboration.
///
/// Reset routine drives power-on -> intro cutscene (Start skip) -> disclaimer
/// (Start skip) -> title "PUSH START!" (Start) -> mode-select menu (cursor
/// defaults on NORMAL MODE -*no* DOWN press for this spec) -> A to confirm
/// Normal -> AREA 1-1 banner -> first controllable frame at ~frame 2022.
pub static BOMBERMAN_2_NORMAL: GameSpec = GameSpec {
    id: "bomberman_2_normal",
    family: "bomberman_2",
    gym_id: "NESLE/Bomberman2-Normal",
    display_name: "Bomberman 2",
    sha1: "ac670d3e8511c52e29edff08c82b45b64c446ba1",
    players: 1,
    four_score: false,
    mode: Some("Normal"),
    minimal_actions: &BOMBERMAN_2_NORMAL_ACTIONS,
    reward: bomberman_2_normal_reward,
    terminal: bomberman_2_normal_terminal,
    lives: bomberman_2_normal_lives,
    in_transition: None,
    per_agent_lives_termination: false,
};

/// Minimal Bomberman action set: NOOP + 4-way move + lay-bomb (A). Grid-based,
/// so no diagonals. Composite move+A actions are intentionally omitted -- under
/// frame-skip, "drop a bomb (A)" then "move" on the next agent-step is equivalent
/// to a one-frame move+drop, and keeping a SINGLE bomb action (1/6 of the set, vs
/// 5/10 when composites are included) stops a fresh/random policy from spamming
/// bombs into immediate self-kills (the VS self-play suicide collapse). Remote-
/// detonator B stays out (power-up-conditional, unverified).
pub const BOMBERMAN_2_BATTLE_ACTIONS: [NesAction; 6] = [
    NesAction::new("NOOP", 0),
    NesAction::new("UP", NesButton::Up as u8),
    NesAction::new("DOWN", NesButton::Down as u8),
    NesAction::new("LEFT", NesButton::Left as u8),
    NesAction::new("RIGHT", NesButton::Right as u8),
    NesAction::new("A", NesButton::A as u8),
];

/// 4-way move + lay-bomb (A). Bomberman is grid-based, so diagonals are not a
/// legal move. B (remote-detonator via Detonator power-up) is power-up-
/// conditional and rare in the first areas; deferred per the multi-player
/// spec's same rationale.
pub const BOMBERMAN_2_NORMAL_ACTIONS: [NesAction; 12] = [
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

const ALIVE_BASE: usize = 0x0069;

/// Sparse last-standing reward: - to anyone who just transitioned alive->dead,
/// +1 to the survivor when the round ends. Per-agent; only `players` ports
/// count. Mutual-destruction / draws (cur_alive==0 transition) award no +1.
///
/// Menu or uninitialized RAM can hold arbitrary alive-byte values; the
/// death-transition check filters those frames, and `u32` alive counts cannot
/// overflow if every byte is 255.
fn bomberman_2_reward(prev: &[u8; 0x800], cur: &[u8; 0x800], players: usize) -> [f32; 4] {
    let mut r = [0.0f32; 4];
    let mut prev_alive: u32 = 0;
    let mut cur_alive: u32 = 0;
    for i in 0..players {
        let ap = prev[ALIVE_BASE + i];
        let ac = cur[ALIVE_BASE + i];
        prev_alive += ap as u32;
        cur_alive += ac as u32;
        if ap == 1 && ac == 0 {
            r[i] = -1.0;
        }
    }
    if prev_alive >= 2 && cur_alive == 1 {
        for i in 0..players {
            if cur[ALIVE_BASE + i] == 1 {
                r[i] += 1.0;
            }
        }
    }
    r
}

fn bomberman_2_reward_3p(prev: &[u8; 0x800], cur: &[u8; 0x800]) -> [f32; 4] {
    bomberman_2_reward(prev, cur, 3)
}

/// Round-over shared predicate: was - alive last frame AND - alive now.
/// Triggers both on a clean win (one survivor) and a mutual KO (all dead).
/// Sums use `u32` so 4 garbage bytes (255 each) can't overflow during boot
/// before the cart has initialised the alive-flag array.
fn bomberman_2_terminal(prev: &[u8; 0x800], cur: &[u8; 0x800], players: usize) -> bool {
    let prev_alive: u32 = (0..players).map(|i| prev[ALIVE_BASE + i] as u32).sum();
    let cur_alive: u32 = (0..players).map(|i| cur[ALIVE_BASE + i] as u32).sum();
    prev_alive >= 2 && cur_alive <= 1
}

fn bomberman_2_terminal_2p(prev: &[u8; 0x800], cur: &[u8; 0x800]) -> bool {
    bomberman_2_terminal(prev, cur, 2)
}

fn bomberman_2_terminal_3p(prev: &[u8; 0x800], cur: &[u8; 0x800]) -> bool {
    bomberman_2_terminal(prev, cur, 3)
}

/// Per-port alive flags from `$0069-006C`. The env's
/// `per_agent_lives_termination` reads `lives[i]==0` as the per-agent terminate
/// signal; trailing slots beyond `players` are ignored by the env loop.
fn bomberman_2_lives(cur: &[u8; 0x800]) -> [u8; 4] {
    [cur[0x0069], cur[0x006a], cur[0x006b], cur[0x006c]]
}

// VS sparse reward (BOMBERMAN_2_VS_2P): +1 to the lone survivor the frame the round
// ends, -1 to anyone bombed out (both on a mutual KO). All dense shaping (approach /
// brick / offense / exploration / anti-camp) lives in the trainer (train_bomberman.py),
// where the nametable, blast model and per-step annealing are available; the env reward
// is the canonical sparse game outcome only (Pommerman +1/-1 convention).
const VS_WIN: f32 = 1.0;
const VS_LOSS: f32 = -1.0;

fn bomberman_2_vs_reward(prev_ram: &[u8; 0x800], cur_ram: &[u8; 0x800]) -> [f32; 4] {
    let mut r = [0.0f32; 4];
    let prev_alive: u32 = (0..2).map(|i| prev_ram[ALIVE_BASE + i] as u32).sum();
    let cur_alive: u32 = (0..2).map(|i| cur_ram[ALIVE_BASE + i] as u32).sum();
    // Round over (2 -> <=1 alive): survivor +1, death / mutual KO -1.
    if prev_alive >= 2 && cur_alive <= 1 {
        for (p, slot) in r.iter_mut().take(2).enumerate() {
            *slot = if cur_ram[ALIVE_BASE + p] == 1 {
                VS_WIN
            } else {
                VS_LOSS
            };
        }
    }
    r
}

/// Decode 8-digit decimal score at $03D0-$03D7, MSB first. Each byte holds a
/// single digit (0..9); poke-verified by writing 9 to each byte and watching
/// the HUD display "99999999". Any digit > 9 (boot garbage / corruption) is
/// clamped to 9 so the score can never be negative or overflow.
fn bomberman_2_normal_score(ram: &[u8; 0x800]) -> u32 {
    let mut score = 0u32;
    for &addr in &[
        0x03D0usize,
        0x03D1,
        0x03D2,
        0x03D3,
        0x03D4,
        0x03D5,
        0x03D6,
        0x03D7,
    ] {
        score = score * 10 + (ram[addr] as u32).min(9);
    }
    score
}

/// Dense score-delta reward (the kung_fu / SMB-score pattern). Score is
/// monotonic in-game, so a `saturating_sub` guards against the boot-garbage
/// case where prev > cur (and against the wrap that would happen if the cart
/// ever clamped the score back to 0, which it doesn't observed).
fn bomberman_2_normal_reward(previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> [f32; 4] {
    solo_reward({
        let score_reward = bomberman_2_normal_score(current_ram)
            .saturating_sub(bomberman_2_normal_score(previous_ram))
            as f32
            * 0.01;
        let lives_prev = previous_ram[0x04E5];
        let lives_cur = current_ram[0x04E5];
        let death_penalty = if lives_cur < lives_prev && lives_prev != 0xff {
            -25.0
        } else {
            0.0
        };
        let area_prev = previous_ram[0x004C];
        let area_cur = current_ram[0x004C];
        let area_clear = if area_cur > area_prev && area_cur - area_prev <= 2 {
            50.0
        } else {
            0.0
        };
        score_reward + death_penalty + area_clear
    })
}

/// Game over the frame $04E5 underflows to 0xff (i.e. the player died with
/// 0 lives remaining and no respawn followed). Guarded by `$0049 != 0xff` to
/// reject the power-on garbage state where the cart hasn't initialised
/// either byte yet -mirrors SMB's `ram[0x0770] != 0xff && ram[0x075a] == 0xff`.
fn bomberman_2_normal_terminal(_previous_ram: &[u8; 0x800], current_ram: &[u8; 0x800]) -> bool {
    current_ram[0x0049] != 0xff && current_ram[0x04E5] == 0xff
}

/// Lives reads the $04E5 byte verbatim while the cart is in play. The 0xff
/// underflow sentinel is reported as 0 so the env sees a clean "0 lives"
/// signal at game over; otherwise the raw byte (0..3 in normal play) flows
/// through.
fn bomberman_2_normal_lives(current_ram: &[u8; 0x800]) -> [u8; 4] {
    solo_lives(if current_ram[0x04E5] == 0xff {
        0
    } else {
        current_ram[0x04E5]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::smoke_multiplayer_actions;

    #[test]
    fn real_rom_steps_battle_actions() {
        smoke_multiplayer_actions(
            &BOMBERMAN_2_BATTLE_3P,
            "Bomberman II (USA).nes",
            &["UP", "DOWN", "LEFT", "RIGHT", "A"],
        );
    }

    fn ram_with_alive(p1: u8, p2: u8, p3: u8) -> [u8; 0x800] {
        let mut r = [0u8; 0x800];
        r[0x0069] = p1;
        r[0x006a] = p2;
        r[0x006b] = p3;
        r[0x006c] = 1; // P4 init
        r
    }

    #[test]
    fn reward_zero_when_no_state_change() {
        let r = ram_with_alive(1, 1, 1);
        assert_eq!(bomberman_2_reward_3p(&r, &r), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn reward_minus_one_on_death_frame() {
        let prev = ram_with_alive(1, 1, 1);
        let cur = ram_with_alive(0, 1, 1);
        // P1 just died (3 -> 2 alive); round not over yet.
        assert_eq!(bomberman_2_reward_3p(&prev, &cur), [-1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn reward_round_over_gives_survivor_plus_one_and_dier_minus_one() {
        let prev = ram_with_alive(0, 1, 1);
        let cur = ram_with_alive(0, 0, 1);
        // P2 dies, P3 survives. Round-over (2 -> 1 alive).
        assert_eq!(bomberman_2_reward_3p(&prev, &cur), [0.0, -1.0, 1.0, 0.0]);
    }

    #[test]
    fn reward_mutual_destruction_gives_no_survivor_bonus() {
        let prev = ram_with_alive(0, 1, 1);
        let cur = ram_with_alive(0, 0, 0);
        // Both remaining players die same frame.
        assert_eq!(bomberman_2_reward_3p(&prev, &cur), [0.0, -1.0, -1.0, 0.0]);
    }

    #[test]
    fn terminal_fires_when_alive_drops_from_2_to_1() {
        let prev = ram_with_alive(0, 1, 1);
        let cur = ram_with_alive(0, 0, 1);
        assert!(bomberman_2_terminal_3p(&prev, &cur));
    }

    #[test]
    fn terminal_does_not_fire_mid_round() {
        let prev = ram_with_alive(1, 1, 1);
        let cur = ram_with_alive(0, 1, 1);
        assert!(!bomberman_2_terminal_3p(&prev, &cur));
    }

    #[test]
    fn terminal_fires_on_mutual_destruction() {
        let prev = ram_with_alive(0, 1, 1);
        let cur = ram_with_alive(0, 0, 0);
        assert!(bomberman_2_terminal_3p(&prev, &cur));
    }

    #[test]
    fn lives_returns_per_port_alive_bytes() {
        let cur = ram_with_alive(1, 0, 1);
        assert_eq!(bomberman_2_lives(&cur), [1, 0, 1, 1]);
    }

    fn vs_ram(p1_alive: u8, p2_alive: u8) -> [u8; 0x800] {
        let mut r = [0u8; 0x800];
        r[ALIVE_BASE] = p1_alive;
        r[ALIVE_BASE + 1] = p2_alive;
        r
    }

    #[test]
    fn vs_reward_zero_mid_round() {
        // Both alive, round not over: no env reward (all shaping lives in the trainer).
        assert_eq!(
            bomberman_2_vs_reward(&vs_ram(1, 1), &vs_ram(1, 1)),
            [0.0; 4]
        );
    }

    #[test]
    fn vs_reward_terminal_win_and_loss() {
        // P2 dies (2 -> 1 alive): survivor +1, loser -1.
        assert_eq!(
            bomberman_2_vs_reward(&vs_ram(1, 1), &vs_ram(1, 0)),
            [VS_WIN, VS_LOSS, 0.0, 0.0]
        );
    }

    #[test]
    fn vs_reward_mutual_ko_both_lose() {
        // Both die same frame (2 -> 0 alive): both -1, no survivor bonus.
        assert_eq!(
            bomberman_2_vs_reward(&vs_ram(1, 1), &vs_ram(0, 0)),
            [VS_LOSS, VS_LOSS, 0.0, 0.0]
        );
    }

    #[test]
    fn reward_and_terminal_survive_boot_garbage() {
        // Boot RAM holds arbitrary alive bytes; reward/terminal must not overflow even when all are 0xff (sum 1020 > u8::MAX).
        let mut garbage = [0u8; 0x800];
        for i in 0..4 {
            garbage[ALIVE_BASE + i] = 0xff;
        }
        // Both calls only need to return without panicking.
        let _ = bomberman_2_reward_3p(&garbage, &garbage);
        let _ = bomberman_2_terminal_3p(&garbage, &garbage);
        let _ = bomberman_2_terminal_2p(&garbage, &garbage);
    }

    fn ram_with_score_lives(digits: [u8; 8], lives: u8, mode: u8) -> [u8; 0x800] {
        let mut r = [0u8; 0x800];
        for (i, d) in digits.iter().enumerate() {
            r[0x03D0 + i] = *d;
        }
        r[0x04E5] = lives;
        r[0x0049] = mode;
        r
    }

    #[test]
    fn score_decodes_eight_digits_msb_first() {
        // "00000123" => score 123
        let ram = ram_with_score_lives([0, 0, 0, 0, 0, 1, 2, 3], 2, 0x00);
        assert_eq!(bomberman_2_normal_score(&ram), 123);
        // "10000000" => 10_000_000 (MSB at $03D0)
        let ram = ram_with_score_lives([1, 0, 0, 0, 0, 0, 0, 0], 2, 0x00);
        assert_eq!(bomberman_2_normal_score(&ram), 10_000_000);
    }

    #[test]
    fn score_clamps_garbage_digits_to_nine() {
        // Each byte > 9 (e.g. 0xff) clamps to 9 so the decoded score is bounded.
        let ram = ram_with_score_lives([0xff; 8], 2, 0x00);
        assert_eq!(bomberman_2_normal_score(&ram), 99_999_999);
    }

    #[test]
    fn reward_is_nonnegative_score_delta() {
        let prev = ram_with_score_lives([0, 0, 0, 0, 0, 0, 1, 0], 2, 0x00); // 10
        let cur = ram_with_score_lives([0, 0, 0, 0, 0, 0, 4, 0], 2, 0x00); //  40
        assert!((bomberman_2_normal_reward(&prev, &cur)[0] - 0.3).abs() < 1e-6);
        // Score doesn't decrease in-game; saturating_sub keeps the reward >= 0.
        assert_eq!(bomberman_2_normal_reward(&cur, &prev)[0], 0.0);
    }

    #[test]
    fn reward_penalizes_death_and_bonuses_area_clear() {
        let mut prev = ram_with_score_lives([0; 8], 2, 0x00);
        let mut cur = ram_with_score_lives([0; 8], 1, 0x00);
        assert_eq!(bomberman_2_normal_reward(&prev, &cur)[0], -25.0);
        prev[0x004C] = 1;
        cur[0x04E5] = 2;
        cur[0x004C] = 2;
        assert_eq!(bomberman_2_normal_reward(&prev, &cur)[0], 50.0);
    }

    #[test]
    fn terminal_fires_only_when_lives_underflows_after_init() {
        // Normal play, lives 1..3: not terminal.
        let prev = [0u8; 0x800];
        let mut ram = ram_with_score_lives([0; 8], 2, 0x00);
        assert!(!bomberman_2_normal_terminal(&prev, &ram));
        ram[0x04E5] = 0; // last on-screen life value, NOT terminal yet
        assert!(!bomberman_2_normal_terminal(&prev, &ram));
        // Underflow sentinel after death-from-zero: terminal.
        ram[0x04E5] = 0xff;
        assert!(bomberman_2_normal_terminal(&prev, &ram));
        // Boot garbage (mode == 0xff): terminal must not fire even when lives looks like 0xff.
        ram[0x0049] = 0xff;
        assert!(!bomberman_2_normal_terminal(&prev, &ram));
    }

    #[test]
    fn lives_reports_zero_at_game_over_sentinel() {
        // Normal play
        let mut ram = ram_with_score_lives([0; 8], 3, 0x00);
        assert_eq!(bomberman_2_normal_lives(&ram)[0], 3);
        ram[0x04E5] = 0;
        assert_eq!(bomberman_2_normal_lives(&ram)[0], 0);
        // Underflow sentinel reads back as 0 for a clean RL signal.
        ram[0x04E5] = 0xff;
        assert_eq!(bomberman_2_normal_lives(&ram)[0], 0);
    }

    #[test]
    fn reward_and_terminal_survive_boot_garbage_normal() {
        // Boot RAM (score/lives/mode all 0xff): reward/terminal must not panic, and terminal must be false (mode==0xff guard).
        let garbage = [0xffu8; 0x800];
        // Score is 99_999_999 both sides; delta saturates to 0.
        assert_eq!(bomberman_2_normal_reward(&garbage, &garbage)[0], 0.0);
        // Terminal: guarded by mode==0xff, must NOT fire.
        assert!(!bomberman_2_normal_terminal(&garbage, &garbage));
        // Lives: returns 0 via the 0xff clamp (boot-garbage env contract).
        assert_eq!(bomberman_2_normal_lives(&garbage)[0], 0);
    }
}
