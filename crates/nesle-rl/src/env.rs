use std::path::{Path, PathBuf};

use nesle_common::{GrayscaleFrame, IndexedFrame, NesleError, Result, RgbFrame};
use nesle_core::state::CoreState;

use crate::constants::RAM_SIZE;
use crate::games::{GameSpec, MultiPlayerValues, Ram};
use crate::interface::NesInterface;
use crate::start_state::{
    load_first_start_state_blob, load_random_start_state_blobs,
    load_start_state_blob as read_start_state_blob, load_start_state_path, StartState,
    StartStateBlob,
};

/// A sink for per-frame RGB output. `render_mode="human"` attaches one (via
/// [`NesEnv::step_rendered`]) so the env draws EVERY emulated frame (including
/// frame-skipped ones) to a window, like ALE's SDL display. The RL hot path
/// ([`NesEnv::step`]) never touches a sink, so obs/training throughput is
/// unaffected. `present` returns true if the viewer asked to close.
pub trait FrameSink {
    fn present(&mut self, rgb: &[u8]) -> bool;
}

#[derive(Debug, Clone)]
pub struct NesEnvState {
    core_state: CoreState,
    episode_frame_number: u64,
    previous_ram: Ram,
    episode_return: f32,
    last_action_masks: MultiPlayerValues<u8>,
    rng_state: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepInfo {
    pub frame_number: u64,
    pub episode_frame_number: u64,
    /// Per-port lives (controller 0..4); trailing slots `0` for games with fewer
    /// than 4 active ports. Only the first `players` entries are meaningful;
    /// single-player consumers read `lives[0]`.
    pub lives: MultiPlayerValues<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResetOutcome {
    pub info: StepInfo,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepOutcome {
    /// Per-port reward (controller 0..4); trailing slots `0`. Single-player reads
    /// `rewards[0]`.
    pub rewards: MultiPlayerValues<f32>,
    /// Per-port termination. The shared `terminal(prev, cur)` predicate ends the
    /// episode for every active port (game over). With `per_agent_lives_termination`,
    /// port `i` ALSO terminates the frame its `lives[i]` hits 0 -- so an agent can
    /// drop out while the others play on (Bomberman II Battle Mode). Inactive ports
    /// (>= `players`) are always `true`. Single-player reads `terminated[0]`.
    pub terminated: MultiPlayerValues<bool>,
    pub truncated: bool,
    pub info: StepInfo,
}

/// One NES environment driving 1..=`players` controller ports (ale-py's
/// `ALEInterface`-for-1-N model: single-player is the `players == 1` case).
/// `step(masks)` takes one action mask per active port; `rewards`/`lives`/
/// `terminated` are per-port `[_; 4]` (trailing slots zero / true). Single-player
/// passes a 1-element slice and reads index 0 -- byte-for-byte identical to the
/// pre-unification single-agent env (`act_n(&masks[..1])` sets only port 0, the
/// migrated game fns ignore `previous_ram` / wrap their scalar via `solo_*`, and
/// the sticky stream draws exactly one sample). Multi-agent self-play drives up
/// to `game.players`; >2-player games enable the Four Score adapter at load.
#[derive(Debug)]
pub struct NesEnv {
    interface: NesInterface,
    game: &'static GameSpec,
    /// Active controller ports (1..=`game.players`). Single-player façades run
    /// with `players == 1` (port 0 only); `game.players` is the max the spec drives.
    players: u8,
    episode_frame_number: u64,
    previous_ram: Ram,
    max_episode_frames: Option<u64>,
    frame_skip: usize,
    skip_transitions: bool,
    rng_state: u64,
    repeat_action_probability: f32,
    last_action_masks: MultiPlayerValues<u8>,
    episode_return: f32,
    start_state: StartState,
    start_state_cache: Vec<StartStateBlob>,
}

impl NesEnv {
    pub fn new(game: &'static GameSpec) -> Self {
        Self {
            interface: NesInterface::default(),
            game,
            players: game.players,
            episode_frame_number: 0,
            previous_ram: [0; RAM_SIZE],
            max_episode_frames: None,
            frame_skip: 1,
            repeat_action_probability: 0.0,
            rng_state: 0x4d595df4d0f33173,
            last_action_masks: [0; 4],
            episode_return: 0.0,
            skip_transitions: true,
            start_state: StartState::FirstAvailable,
            start_state_cache: Vec::new(),
        }
    }

    /// Set how many controller ports are in play (1..=`game.players`). Single-player
    /// façades call `set_players(1)` (port 0 only); multi-agent self-play uses the
    /// full count. Modes that only differ in port count (Super C 1P / 2P, Ice Hockey
    /// 1P / VS) are one spec driven here.
    pub fn set_players(&mut self, players: u8) -> Result<()> {
        if players < 1 || players > self.game.players {
            return Err(NesleError::InvalidState(format!(
                "players must be in 1..={} for {}",
                self.game.players, self.game.id
            )));
        }
        self.players = players;
        Ok(())
    }

    /// Active controller-port count (1..=`game.players`).
    pub fn players(&self) -> u8 {
        self.players
    }

    pub fn set_max_episode_frames(&mut self, max_episode_frames: Option<u64>) {
        self.max_episode_frames = max_episode_frames;
    }

    pub fn set_action_repeat(
        &mut self,
        frame_skip: usize,
        repeat_action_probability: f32,
    ) -> Result<()> {
        if frame_skip == 0 {
            return Err(NesleError::InvalidState(
                "frame_skip must be at least 1".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&repeat_action_probability) {
            return Err(NesleError::InvalidState(
                "repeat_action_probability must be between 0.0 and 1.0".to_string(),
            ));
        }
        self.frame_skip = frame_skip;
        self.repeat_action_probability = repeat_action_probability;
        Ok(())
    }

    /// Toggle the level-transition fast-forward (default on). Off = faithful per-frame
    /// stepping that shows the cutscene (server Play / Human-AI); on = the RL view.
    pub fn set_skip_transitions(&mut self, enabled: bool) {
        self.skip_transitions = enabled;
    }

    pub fn seed(&mut self, seed: u64) {
        self.rng_state = seed.max(1);
    }

    pub fn load_rom_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.start_state_cache.clear();
        self.interface.load_rom_bytes(bytes)?;
        // Force the Four Score adapter for >2-player games (their iNES header rarely declares it) so ports 2/3 read.
        self.interface.set_four_score_mode(self.game.four_score);
        Ok(())
    }

    pub fn reset(&mut self) -> Result<ResetOutcome> {
        self.reset_to_start_state()
    }

    pub fn reset_game(&mut self) -> ResetOutcome {
        self.interface.reset_game();
        self.start_episode_from_current_state();
        ResetOutcome {
            info: self.step_info(),
        }
    }

    pub fn set_start_state(&mut self, start_state: StartState) -> Result<()> {
        self.start_state = start_state;
        self.start_state_cache.clear();
        Ok(())
    }

    pub fn set_start_state_id(&mut self, start_state: &str) -> Result<()> {
        self.set_start_state(StartState::parse(start_state)?)
    }

    pub fn set_start_state_path(&mut self, path: impl Into<PathBuf>) -> Result<()> {
        self.set_start_state(StartState::Path(path.into()))
    }

    pub fn load_start_state_blob(&self, start_state: &StartState) -> Result<Vec<u8>> {
        match start_state {
            StartState::FirstAvailable => {
                load_first_start_state_blob(self.game.id).map(|blob| blob.bytes)
            }
            StartState::Id(id) => read_start_state_blob(self.game.id, id).map(|blob| blob.bytes),
            StartState::Path(path) => load_start_state_path(Path::new(path)).map(|blob| blob.bytes),
            StartState::Random => Err(NesleError::InvalidState(
                "start_state='random' selects from the game start-state directory at reset time"
                    .to_string(),
            )),
        }
    }

    pub fn reset_to_start_state(&mut self) -> Result<ResetOutcome> {
        match self.start_state.clone() {
            StartState::FirstAvailable => {
                if self.start_state_cache.is_empty() {
                    self.start_state_cache = vec![load_first_start_state_blob(self.game.id)?];
                }
                let bytes = self.start_state_cache[0].bytes.clone();
                self.restore_state_blob(bytes)?;
                Ok(ResetOutcome {
                    info: self.step_info(),
                })
            }
            StartState::Id(id) => {
                if self.start_state_cache.is_empty() {
                    self.start_state_cache = vec![read_start_state_blob(self.game.id, &id)?];
                }
                let bytes = self.start_state_cache[0].bytes.clone();
                self.restore_state_blob(bytes)?;
                Ok(ResetOutcome {
                    info: self.step_info(),
                })
            }
            StartState::Path(path) => {
                if self.start_state_cache.is_empty() {
                    self.start_state_cache = vec![load_start_state_path(Path::new(&path))?];
                }
                let bytes = self.start_state_cache[0].bytes.clone();
                self.restore_state_blob(bytes)?;
                Ok(ResetOutcome {
                    info: self.step_info(),
                })
            }
            StartState::Random => {
                if self.start_state_cache.is_empty() {
                    self.start_state_cache = load_random_start_state_blobs(self.game.id)?;
                }
                let index = self.next_random_below(self.start_state_cache.len() as u32) as usize;
                let bytes = self.start_state_cache[index].bytes.clone();
                self.restore_state_blob(bytes)?;
                Ok(ResetOutcome {
                    info: self.step_info(),
                })
            }
        }
    }

    /// Capture the cart's full core snapshot as a raw byte blob. Round-trips with
    /// `restore_state_blob`; use `clone_state` for env bookkeeping.
    pub fn save_state_blob(&self) -> Vec<u8> {
        self.interface.clone_state().into_bytes()
    }

    /// Restore the cart from a blob produced by `save_state_blob` (typically read
    /// from a disk cache). Episode bookkeeping is reset to a fresh episode start
    /// so the env doesn't carry over a previous run's frame counter.
    pub fn restore_state_blob(&mut self, blob: Vec<u8>) -> Result<()> {
        let state = CoreState::from_bytes(0, blob);
        self.interface.restore_state(&state)?;
        self.start_episode_from_current_state();
        Ok(())
    }

    /// Step one agent step (a `frame_skip` window) driving the `players` active
    /// ports. `masks[i]` is port i's action; single-player passes a 1-element
    /// slice (port 0). Per-port rewards accumulate across the window.
    /// `terminated[i]` is the shared match-over predicate OR'd with (when the spec
    /// opts into `per_agent_lives_termination`) port i's `lives[i] == 0`. Inactive
    /// ports (>= `players`) stay `true`. After the window, a single-player level
    /// transition is fast-forwarded (see `skip_transition`) when `skip_transitions`
    /// is on and the episode is still live.
    pub fn step(&mut self, masks: &[u8]) -> Result<StepOutcome> {
        let masks = self.sticky_actions(masks);
        let players = self.players as usize;
        let mut rewards = [0.0f32; 4];
        // Inactive ports start terminated; active ports are updated in-loop.
        let mut terminated = [true; 4];
        for slot in terminated.iter_mut().take(players) {
            *slot = false;
        }
        let mut truncated = false;
        let per_agent = self.game.per_agent_lives_termination;
        for _ in 0..self.frame_skip {
            let (reward, shared, lives_now, tr) = self.step_one_frame(&masks)?;
            for (acc, r) in rewards.iter_mut().zip(reward) {
                *acc += r;
            }
            for i in 0..players {
                terminated[i] = shared || (per_agent && lives_now[i] == 0);
            }
            truncated = tr;
            // Break when every active port is done (trailing slots are pre-set true).
            if terminated[..players].iter().all(|&t| t) || truncated {
                break;
            }
        }
        if self.skip_transitions
            && !terminated[..players].iter().all(|&t| t)
            && !truncated
        {
            let (t, tr) = self.skip_transition()?;
            if t {
                for slot in terminated.iter_mut().take(players) {
                    *slot = true;
                }
            }
            truncated |= tr;
        }
        self.finish_step(&masks, rewards, terminated, truncated)
    }

    /// Fast-forward a non-interactive level transition (the game's `in_transition`):
    /// advance NOOP frames until normal play resumes (or death / a frame cap), updating
    /// `previous_ram` without accruing reward, so the agent never observes the dead zone.
    /// Only single-player games declare `in_transition` (they run with `players == 1`),
    /// so the NOOP slice sets only port 0 -- byte-identical to the pre-unification path.
    fn skip_transition(&mut self) -> Result<(bool, bool)> {
        let Some(in_transition) = self.game.in_transition else {
            return Ok((false, false));
        };
        if !in_transition(&self.previous_ram) {
            return Ok((false, false));
        }
        // Cap ~20s @60fps (real flag->next-level dead zone is ~730 frames).
        const MAX_TRANSITION_FRAMES: usize = 1200;
        self.interface.set_render_enabled(true);
        let noop = [0u8; 4];
        for _ in 0..MAX_TRANSITION_FRAMES {
            let (_r, terminated, _lives, truncated) = self.step_one_frame(&noop)?;
            if terminated || truncated || !in_transition(&self.previous_ram) {
                return Ok((terminated, truncated));
            }
        }
        Ok((false, false))
    }

    /// Like [`step`](Self::step) but draws EVERY emulated frame (incl. the
    /// frame-skipped ones) to `sink` -- the ALE per-frame SDL display analogue
    /// for `render_mode="human"`. Faithful per-frame view, so (like the old single
    /// path) it does NOT fast-forward level transitions. Shares `step_one_frame`
    /// with `step`; `step` itself never references a sink, so the RL/obs hot path
    /// is unchanged.
    pub fn step_rendered(
        &mut self,
        masks: &[u8],
        sink: &mut dyn FrameSink,
    ) -> Result<StepOutcome> {
        let masks = self.sticky_actions(masks);
        let players = self.players as usize;
        let mut rewards = [0.0f32; 4];
        let mut terminated = [true; 4];
        for slot in terminated.iter_mut().take(players) {
            *slot = false;
        }
        let mut truncated = false;
        let per_agent = self.game.per_agent_lives_termination;
        for _ in 0..self.frame_skip {
            let (reward, shared, lives_now, tr) = self.step_one_frame(&masks)?;
            for (acc, r) in rewards.iter_mut().zip(reward) {
                *acc += r;
            }
            for i in 0..players {
                terminated[i] = shared || (per_agent && lives_now[i] == 0);
            }
            truncated = tr;
            sink.present(&self.interface.screen_rgb().pixels);
            if terminated[..players].iter().all(|&t| t) || truncated {
                break;
            }
        }
        self.finish_step(&masks, rewards, terminated, truncated)
    }

    /// Advance one emulated frame driving the `players` active ports with the
    /// (already sticky-resolved) masks: step, bump the episode counter, and return
    /// (per-port reward delta, shared terminal, per-port lives, truncated). Shared
    /// by `step`, `step_rendered`, and `skip_transition`. `act_n(&masks[..players])`
    /// sets only the active ports -- for `players == 1` that is exactly the old
    /// single-port `act(mask)`.
    #[inline]
    fn step_one_frame(
        &mut self,
        masks: &MultiPlayerValues<u8>,
    ) -> Result<(MultiPlayerValues<f32>, bool, MultiPlayerValues<u8>, bool)> {
        let previous_ram = self.previous_ram;
        self.interface.act_n(&masks[..self.players as usize])?;
        self.episode_frame_number = self.episode_frame_number.wrapping_add(1);
        let current_ram = *self.interface.ram();
        let reward = (self.game.reward)(&previous_ram, &current_ram);
        let shared = (self.game.terminal)(&previous_ram, &current_ram);
        let lives = (self.game.lives)(&current_ram);
        let truncated = self
            .max_episode_frames
            .is_some_and(|max_frames| self.episode_frame_number >= max_frames);
        self.previous_ram = current_ram;
        Ok((reward, shared, lives, truncated))
    }

    #[inline]
    fn finish_step(
        &mut self,
        masks: &MultiPlayerValues<u8>,
        rewards: MultiPlayerValues<f32>,
        terminated: MultiPlayerValues<bool>,
        truncated: bool,
    ) -> Result<StepOutcome> {
        self.last_action_masks = *masks;
        self.episode_return += rewards[0];
        Ok(StepOutcome {
            rewards,
            terminated,
            truncated,
            info: self.step_info(),
        })
    }

    pub fn ram(&self) -> &Ram {
        self.interface.ram()
    }

    /// Diagnostic PPU nametable (CIRAM) view -- the rendered field's tile ids.
    /// Scripted bots use it for the wall/brick/floor layout (the field is not kept
    /// as a per-tile array in CPU RAM).
    pub fn vram(&self) -> &[u8] {
        self.interface.vram()
    }

    pub fn screen_indexed(&self) -> &IndexedFrame {
        self.interface.screen_indexed()
    }

    pub fn screen_grayscale(&self) -> GrayscaleFrame {
        self.interface.screen_grayscale()
    }

    pub fn screen_grayscale_into(&self, pixels: &mut [u8]) {
        self.interface.screen_grayscale_into(pixels);
    }

    pub fn screen_rgb(&mut self) -> &RgbFrame {
        self.interface.screen_rgb()
    }

    /// Enable/disable APU audio synthesis at runtime (off by default for RL
    /// throughput; see `NesCore::set_audio_enabled`). Byte-identical RAM/CPU.
    pub fn set_audio_enabled(&mut self, enabled: bool) {
        self.interface.set_audio_enabled(enabled);
    }

    /// Enable/disable RL-only no-sprite-flicker rendering (Mesen2
    /// RemoveSpriteLimit). Byte-identical RAM/CPU/frame; only the rendered
    /// framebuffer gains the 9th+ sprites (lets the obs pipeline drop max-pool).
    pub fn set_remove_sprite_limit(&mut self, enabled: bool) {
        self.interface.set_remove_sprite_limit(enabled);
    }

    /// Enable/disable per-frame rendering (RL render-skip; see
    /// `NesCore::set_render_enabled`). Byte-identical RAM/CPU/frame_count.
    pub fn set_render_enabled(&mut self, enabled: bool) {
        self.interface.set_render_enabled(enabled);
    }

    /// Drain APU audio samples accumulated since the last call (mono f32 at
    /// `NesInterface::audio_sample_rate()` Hz). For the human / RL-observe viewers.
    pub fn drain_audio_samples(&mut self) -> Vec<f32> {
        self.interface.drain_audio_samples()
    }

    pub fn set_ram(&mut self, bytes: &[u8]) -> Result<()> {
        self.interface.set_ram(bytes)?;
        self.previous_ram = *self.interface.ram();
        Ok(())
    }

    pub fn clone_state(&self) -> NesEnvState {
        NesEnvState {
            core_state: self.interface.clone_state(),
            episode_frame_number: self.episode_frame_number,
            previous_ram: self.previous_ram,
            episode_return: self.episode_return,
            last_action_masks: self.last_action_masks,
            rng_state: self.rng_state,
        }
    }

    pub fn restore_state(&mut self, state: &NesEnvState) -> Result<()> {
        self.interface.restore_state(&state.core_state)?;
        self.episode_frame_number = state.episode_frame_number;
        self.previous_ram = state.previous_ram;
        self.episode_return = state.episode_return;
        self.last_action_masks = state.last_action_masks;
        self.rng_state = state.rng_state;
        Ok(())
    }

    pub fn game(&self) -> &'static GameSpec {
        self.game
    }

    pub(crate) fn step_info(&self) -> StepInfo {
        StepInfo {
            frame_number: self.interface.frame_count(),
            episode_frame_number: self.episode_frame_number,
            lives: (self.game.lives)(self.interface.ram()),
        }
    }

    fn start_episode_from_current_state(&mut self) {
        self.episode_frame_number = 0;
        self.previous_ram = *self.interface.ram();
        self.last_action_masks = [0; 4];
        self.episode_return = 0.0;
    }

    /// Resolve sticky-action repeat for each active port. With
    /// `repeat_action_probability == 0` (the default, and every multi-agent spec)
    /// this is a zero-RNG copy of the requested masks. Otherwise each active port
    /// draws ONE sample from the seeded LCG and, with that probability, repeats its
    /// previous mask -- so `players == 1` reproduces the single-agent sticky stream
    /// byte-for-byte (one draw, port 0 only).
    fn sticky_actions(&mut self, requested: &[u8]) -> MultiPlayerValues<u8> {
        let players = self.players as usize;
        let mut out = [0u8; 4];
        if self.repeat_action_probability <= 0.0 {
            for (i, slot) in out.iter_mut().take(players).enumerate() {
                *slot = requested.get(i).copied().unwrap_or(0);
            }
            return out;
        }
        for (i, slot) in out.iter_mut().take(players).enumerate() {
            let req = requested.get(i).copied().unwrap_or(0);
            let sample = self.next_random_unit();
            *slot = if sample < self.repeat_action_probability {
                self.last_action_masks[i]
            } else {
                req
            };
        }
        out
    }

    fn next_random_unit(&mut self) -> f32 {
        let value = self.next_random_u24();
        value as f32 / ((1u32 << 24) as f32)
    }

    /// One 24-bit draw from the env's LCG (advances `rng_state`). Shared by
    /// `next_random_unit` (sticky-action probability) and `next_random_below`
    /// (seeded no-op count), so all stochastic RL choices ride the SAME seeded
    /// stream -- reproducible from `seed()` with no Python RNG involved.
    fn next_random_u24(&mut self) -> u32 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.rng_state >> 40) as u32
    }

    /// Uniform integer in `[0, n)` from the seeded LCG (`0` when `n == 0`). Used
    /// for the gymnasium-style no-op reset count (`1 + next_random_below(noop_max)`
    /// gives `[1, noop_max]`); the 24-bit modulo bias is negligible for the small
    /// `noop_max` (<=30) RL uses.
    pub fn next_random_below(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        self.next_random_u24() % n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::load_test_rom;
    use crate::games::GameSpec;

    fn test_rom() -> Vec<u8> {
        load_test_rom()
    }

    fn zero_reward(_previous_ram: &Ram, _current_ram: &Ram) -> MultiPlayerValues<f32> {
        [0.0; 4]
    }

    fn never_terminal(_previous_ram: &Ram, _current_ram: &Ram) -> bool {
        false
    }

    fn zero_lives(_current_ram: &Ram) -> MultiPlayerValues<u8> {
        [0; 4]
    }

    static TEST_GAME: GameSpec = GameSpec {
        id: "super_mario_bros",
        family: "super_mario_bros",
        gym_id: "NESLE/TestCoreMechanics-v0",
        display_name: "Test Core Mechanics",
        sha1: "",
        players: 1,
        four_score: false,
        mode: None,
        minimal_actions: &crate::games::supported::super_mario_bros::SMB1_ACTIONS,
        reward: zero_reward,
        terminal: never_terminal,
        lives: zero_lives,
        in_transition: None,
        per_agent_lives_termination: false,
    };

    #[test]
    fn reset_and_step_return_ale_style_outcomes() {
        let mut env = NesEnv::new(&TEST_GAME);
        env.load_rom_bytes(&test_rom()).unwrap();
        env.set_max_episode_frames(Some(2));

        let reset = env.reset().unwrap();
        assert_eq!(reset.info.episode_frame_number, 0);

        let first = env.step(&[0]).unwrap();
        assert_eq!(first.rewards[0], 0.0);
        assert!(!first.terminated[0]);
        assert!(!first.truncated);

        let second = env.step(&[0]).unwrap();
        assert!(second.truncated);
        assert_eq!(second.info.episode_frame_number, 2);
    }

    #[test]
    fn state_reset_reproduces_start_state() {
        let mut env = NesEnv::new(&TEST_GAME);
        env.load_rom_bytes(&test_rom()).unwrap();

        fn run(env: &mut NesEnv) -> (u64, Ram, Vec<(u64, Ram)>) {
            let reset = env.reset().unwrap();
            let start_ram = *env.ram();
            let mut frames = Vec::new();
            for _ in 0..8 {
                let out = env.step(&[0x80]).unwrap(); // hold RIGHT
                frames.push((out.info.frame_number, *env.ram()));
            }
            (reset.info.frame_number, start_ram, frames)
        }

        let first = run(&mut env);
        let second = run(&mut env);
        assert_eq!(first.0, second.0, "reset frame_number must match");
        assert_eq!(
            first.1, second.1,
            "post-reset RAM must match the start-state asset"
        );
        assert_eq!(
            first.2, second.2,
            "replayed trajectory after restore must be identical"
        );
    }

    #[test]
    fn env_state_restores_episode_bookkeeping() {
        let mut env = NesEnv::new(&TEST_GAME);
        env.load_rom_bytes(&test_rom()).unwrap();
        env.step(&[0]).unwrap();
        let state = env.clone_state();
        env.step(&[0]).unwrap();
        assert_eq!(env.step(&[0]).unwrap().info.episode_frame_number, 3);
        env.restore_state(&state).unwrap();
        assert_eq!(env.step(&[0]).unwrap().info.episode_frame_number, 2);
    }

    #[test]
    fn frame_skip_repeats_action_and_accumulates_frames() {
        let mut env = NesEnv::new(&TEST_GAME);
        env.load_rom_bytes(&test_rom()).unwrap();
        env.set_action_repeat(4, 0.0).unwrap();
        env.reset().unwrap();

        let step = env.step(&[0]).unwrap();
        assert_eq!(step.info.episode_frame_number, 4);
    }

    #[test]
    fn max_episode_frames_truncates_inside_frame_skip() {
        let mut env = NesEnv::new(&TEST_GAME);
        env.load_rom_bytes(&test_rom()).unwrap();
        env.set_action_repeat(4, 0.0).unwrap();
        env.set_max_episode_frames(Some(2));
        env.reset().unwrap();

        let step = env.step(&[0]).unwrap();
        assert!(step.truncated);
        assert_eq!(step.info.episode_frame_number, 2);
    }

    #[test]
    fn start_state_path_restores_blob_and_repeats() {
        let mut env = NesEnv::new(&TEST_GAME);
        env.load_rom_bytes(&test_rom()).unwrap();
        env.reset().unwrap();
        env.step(&[0x80]).unwrap();
        let blob = env.save_state_blob();
        let expected_ram = *env.ram();
        let path = std::env::temp_dir().join(format!(
            "nesle-start-state-{}-{}.state",
            std::process::id(),
            env.clone_state().episode_frame_number
        ));
        std::fs::write(&path, &blob).unwrap();

        env.step(&[0]).unwrap();
        env.set_start_state_path(&path).unwrap();
        env.reset_to_start_state().unwrap();
        assert_eq!(*env.ram(), expected_ram);
        env.step(&[0]).unwrap();
        env.reset_to_start_state().unwrap();
        assert_eq!(*env.ram(), expected_ram);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn missing_start_state_reports_clear_error() {
        let mut env = NesEnv::new(&TEST_GAME);
        env.load_rom_bytes(&test_rom()).unwrap();
        env.set_start_state_id("level_99").unwrap();
        let err = env.reset_to_start_state().unwrap_err().to_string();
        assert!(err.contains("level_99.state"), "{err}");
    }

    #[test]
    fn sticky_action_can_repeat_previous_action() {
        let mut env = NesEnv::new(&TEST_GAME);
        env.load_rom_bytes(&test_rom()).unwrap();
        env.set_action_repeat(1, 1.0).unwrap();
        env.reset().unwrap();
        env.step(&[0x01]).unwrap();
        let state = env.clone_state();

        env.step(&[0x02]).unwrap();
        env.restore_state(&state).unwrap();
        env.step(&[0x03]).unwrap();
        assert_eq!(env.clone_state().last_action_masks[0], 0);
    }

    #[test]
    fn set_players_rejects_out_of_range() {
        let mut env = NesEnv::new(&TEST_GAME);
        assert!(env.set_players(1).is_ok());
        assert!(env.set_players(0).is_err());
        assert!(env.set_players(2).is_err());
        assert_eq!(env.players(), 1);
    }
}
