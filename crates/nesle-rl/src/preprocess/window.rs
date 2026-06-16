use crate::constants::{GRAY_FRAME_LEN, NES_HEIGHT, NES_WIDTH, RAM_SIZE, RGB_FRAME_LEN};
use crate::env::StepInfo;
use crate::games::MultiPlayerValues;
use nesle_common::{NesleError, Result};

use super::config::{ObsConfig, ObsKind, ObsShape};
use super::resize::{compute_obs_into, resize_area_rgb_into, ResizePlan};

/// One raw emulator frame sample fed into [`ObsWindow`]. The window never steps the
/// emulator; callers decide whether RGB/gray/RAM are available for this frame.
pub struct FrameSample<'a> {
    pub rgb: Option<&'a [u8]>,
    pub gray: Option<&'a [u8]>,
    pub ram: Option<&'a [u8]>,
    pub rewards: MultiPlayerValues<f32>,
    pub lives: MultiPlayerValues<u8>,
    pub frame_number: u64,
    pub episode_frame_number: u64,
    pub terminated: bool,
    pub truncated: bool,
}

impl<'a> FrameSample<'a> {
    pub fn single(
        gray: Option<&'a [u8]>,
        rgb: Option<&'a [u8]>,
        ram: Option<&'a [u8]>,
        reward: f32,
        info: StepInfo,
        terminated: bool,
        truncated: bool,
    ) -> Self {
        Self {
            rgb,
            gray,
            ram,
            rewards: [reward, 0.0, 0.0, 0.0],
            lives: info.lives,
            frame_number: info.frame_number,
            episode_frame_number: info.episode_frame_number,
            terminated,
            truncated,
        }
    }

    /// Multi-port variant: stores the full per-port `rewards` verbatim. `terminated`
    /// is the unit's scalar "done" (the window's boundary / life-loss logic stays
    /// scalar = the unit's done). For `players == 1`, pass `rewards == [r, 0, 0, 0]`
    /// to stay byte-identical to [`FrameSample::single`].
    #[allow(clippy::too_many_arguments)]
    pub fn multi(
        gray: Option<&'a [u8]>,
        rgb: Option<&'a [u8]>,
        ram: Option<&'a [u8]>,
        rewards: MultiPlayerValues<f32>,
        info: StepInfo,
        terminated: bool,
        truncated: bool,
    ) -> Self {
        Self {
            rgb,
            gray,
            ram,
            rewards,
            lives: info.lives,
            frame_number: info.frame_number,
            episode_frame_number: info.episode_frame_number,
            terminated,
            truncated,
        }
    }
}

pub struct ObsWindowStep {
    pub obs_step: bool,
    pub rewards: MultiPlayerValues<f32>,
    pub sample_rewards: MultiPlayerValues<f32>,
    pub lives: MultiPlayerValues<u8>,
    pub frame_number: u64,
    pub episode_frame_number: u64,
    pub terminated: bool,
    pub truncated: bool,
}

/// Shared frame-skip/max-pool/resize/reward accumulation window. It is fed by a
/// single emulator stream and can be instantiated once per viewer/agent.
pub struct ObsWindow {
    cfg: ObsConfig,
    resize: Option<ResizePlan>,
    win_idx: usize,
    win_reward: MultiPlayerValues<f32>,
    lives: MultiPlayerValues<u8>,
    frame0: Vec<u8>,
    frame1: Vec<u8>,
    rgb_frame: Vec<u8>,
    ram_frame: Vec<u8>,
    have_frame0: bool,
    have_frame1: bool,
    have_rgb: bool,
    have_ram: bool,
    obs: Vec<u8>,
    /// frame stack (oldest -> newest), `stack_num * obs.len()` bytes. Only
    /// maintained when `cfg.stack_num > 1`; for `stack_num == 1` the window serves
    /// `obs` directly (zero stacking overhead, byte-identical to the old pipeline).
    stack: Vec<u8>,
    shape: ObsShape,
    have_obs: bool,
}

impl ObsWindow {
    pub fn new(cfg: ObsConfig) -> Self {
        let resize = match cfg.obs_kind {
            ObsKind::Gray { width, height, .. } => {
                Some(ResizePlan::new(NES_WIDTH, NES_HEIGHT, width, height))
            }
            ObsKind::Rgb { width, height } => {
                Some(ResizePlan::new(NES_WIDTH, NES_HEIGHT, width, height))
            }
            ObsKind::RgbNative | ObsKind::Ram => None,
        };
        let shape = cfg.shape();
        Self {
            cfg,
            resize,
            win_idx: 0,
            win_reward: [0.0; 4],
            lives: [0; 4],
            frame0: vec![0u8; GRAY_FRAME_LEN],
            frame1: vec![0u8; GRAY_FRAME_LEN],
            rgb_frame: vec![0u8; RGB_FRAME_LEN],
            ram_frame: vec![0u8; RAM_SIZE],
            have_frame0: false,
            have_frame1: false,
            have_rgb: false,
            have_ram: false,
            obs: Vec::new(),
            stack: Vec::new(),
            shape,
            have_obs: false,
        }
    }

    pub fn config(&self) -> &ObsConfig {
        &self.cfg
    }

    pub fn observation(&self) -> &[u8] {
        if self.cfg.stack_num <= 1 {
            &self.obs
        } else {
            &self.stack
        }
    }

    pub fn shape(&self) -> ObsShape {
        self.shape
    }

    pub fn has_observation(&self) -> bool {
        self.have_obs
    }

    pub fn reset(&mut self, lives: MultiPlayerValues<u8>) {
        self.win_idx = 0;
        self.win_reward = [0.0; 4];
        self.lives = lives;
        self.have_obs = false;
        self.have_frame0 = false;
        self.have_frame1 = false;
        self.have_rgb = false;
        self.have_ram = false;
        self.obs.clear();
        self.stack.clear();
    }

    pub fn restart_window(&mut self) {
        self.win_idx = 0;
        self.win_reward = [0.0; 4];
    }

    pub fn refresh(&mut self, sample: FrameSample<'_>) -> Result<ObsWindowStep> {
        self.capture_frame(&sample)?;
        self.compute_current_obs()?;
        // reset padding: prime every stack slot with this first frame.
        self.fill_stack();
        self.win_idx = 0;
        self.win_reward = [0.0; 4];
        self.lives = sample.lives;
        Ok(ObsWindowStep {
            obs_step: true,
            rewards: [0.0; 4],
            sample_rewards: sample.rewards,
            lives: sample.lives,
            frame_number: sample.frame_number,
            episode_frame_number: sample.episode_frame_number,
            terminated: sample.terminated,
            truncated: sample.truncated,
        })
    }

    pub fn push_frame(
        &mut self,
        sample: FrameSample<'_>,
        force_boundary: bool,
    ) -> Result<ObsWindowStep> {
        self.capture_frame(&sample)?;
        for p in 0..self.win_reward.len() {
            self.win_reward[p] += sample.rewards[p];
        }
        self.win_idx += 1;

        let mut terminated = sample.terminated;
        if self.cfg.terminal_on_life_loss {
            // Only active ports hold real life counts; trailing slots are unused game RAM (Bomberman 2 VS reuses $6C as P1 tile-X).
            let active = (self.cfg.players as usize).clamp(1, self.lives.len());
            for p in 0..active {
                if self.lives[p] > 0 && sample.lives[p] < self.lives[p] {
                    terminated = true;
                    break;
                }
            }
        }
        self.lives = sample.lives;

        let boundary = force_boundary
            || self.win_idx >= self.cfg.frame_skip.max(1)
            || terminated
            || sample.truncated;

        let rewards = if boundary {
            if self.can_compute_obs() {
                self.compute_current_obs()?;
                // Roll the freshly computed frame into the stack (oldest -> newest).
                self.roll_stack();
            }
            let rewards = self.cfg.reward_clip.apply_all(self.win_reward);
            self.win_idx = 0;
            self.win_reward = [0.0; 4];
            rewards
        } else {
            sample.rewards
        };

        Ok(ObsWindowStep {
            obs_step: boundary,
            rewards,
            sample_rewards: sample.rewards,
            lives: sample.lives,
            frame_number: sample.frame_number,
            episode_frame_number: sample.episode_frame_number,
            terminated,
            truncated: sample.truncated,
        })
    }

    fn capture_frame(&mut self, sample: &FrameSample<'_>) -> Result<()> {
        if let Some(gray) = sample.gray {
            if gray.len() != GRAY_FRAME_LEN {
                return Err(NesleError::InvalidState(format!(
                    "grayscale frame has {} bytes, expected {GRAY_FRAME_LEN}",
                    gray.len()
                )));
            }
            if self.have_frame0 {
                self.frame1.copy_from_slice(&self.frame0);
                self.have_frame1 = true;
            }
            self.frame0.copy_from_slice(gray);
            self.have_frame0 = true;
        }
        if let Some(rgb) = sample.rgb {
            let expected = RGB_FRAME_LEN;
            if rgb.len() != expected {
                return Err(NesleError::InvalidState(format!(
                    "RGB frame has {} bytes, expected {expected}",
                    rgb.len()
                )));
            }
            self.rgb_frame.copy_from_slice(rgb);
            self.have_rgb = true;
        }
        if let Some(ram) = sample.ram {
            let expected = RAM_SIZE;
            if ram.len() != expected {
                return Err(NesleError::InvalidState(format!(
                    "RAM observation has {} bytes, expected {expected}",
                    ram.len()
                )));
            }
            self.ram_frame.copy_from_slice(ram);
            self.have_ram = true;
        }
        Ok(())
    }

    fn can_compute_obs(&self) -> bool {
        match self.cfg.obs_kind {
            ObsKind::Gray { .. } => self.have_frame0,
            ObsKind::Rgb { .. } => self.have_rgb,
            ObsKind::RgbNative => self.have_rgb,
            ObsKind::Ram => self.have_ram,
        }
    }

    fn compute_current_obs(&mut self) -> Result<()> {
        match self.cfg.obs_kind {
            ObsKind::Gray { maxpool, .. } => {
                if !self.have_frame0 {
                    if self.have_obs {
                        return Ok(());
                    }
                    return Err(NesleError::InvalidState(
                        "cannot compute grayscale observation before a frame is captured"
                            .to_string(),
                    ));
                }
                let prev = if maxpool && self.have_frame1 {
                    Some(self.frame1.as_slice())
                } else {
                    None
                };
                let resize = self
                    .resize
                    .as_ref()
                    .ok_or_else(|| NesleError::InvalidState("missing resize plan".to_string()))?;
                if self.obs.len() != resize.output_len() {
                    self.obs.resize(resize.output_len(), 0);
                }
                compute_obs_into(&self.frame0, prev, resize, &mut self.obs);
            }
            ObsKind::RgbNative => {
                if !self.have_rgb {
                    if self.have_obs {
                        return Ok(());
                    }
                    return Err(NesleError::InvalidState(
                        "cannot compute RGB observation before a frame is captured".to_string(),
                    ));
                }
                self.obs.clear();
                self.obs.extend_from_slice(&self.rgb_frame);
            }
            ObsKind::Rgb { .. } => {
                if !self.have_rgb {
                    if self.have_obs {
                        return Ok(());
                    }
                    return Err(NesleError::InvalidState(
                        "cannot compute RGB observation before a frame is captured".to_string(),
                    ));
                }
                let resize = self
                    .resize
                    .as_ref()
                    .ok_or_else(|| NesleError::InvalidState("missing resize plan".to_string()))?;
                let len = resize.output_len() * 3;
                if self.obs.len() != len {
                    self.obs.resize(len, 0);
                }
                resize_area_rgb_into(&self.rgb_frame, resize, &mut self.obs);
            }
            ObsKind::Ram => {
                if !self.have_ram {
                    if self.have_obs {
                        return Ok(());
                    }
                    return Err(NesleError::InvalidState(
                        "cannot compute RAM observation before RAM is captured".to_string(),
                    ));
                }
                self.obs.clear();
                self.obs.extend_from_slice(&self.ram_frame);
            }
        }
        self.have_obs = true;
        Ok(())
    }

    /// Prime every stack slot with the current single frame (`obs`). Used at reset:
    /// The stack is padded with copies of the first frame so the very first
    /// observation already has `stack_num` frames. No-op when `stack_num <= 1`.
    fn fill_stack(&mut self) {
        let n = self.cfg.stack_num.max(1);
        if n <= 1 {
            return;
        }
        let frame_len = self.obs.len();
        self.stack.clear();
        self.stack.reserve(n * frame_len);
        for _ in 0..n {
            self.stack.extend_from_slice(&self.obs);
        }
    }

    /// Shift the stack one frame toward "oldest" and append the freshly computed
    /// `obs` as the newest frame. No-op when `stack_num <= 1`. Falls back to a full
    /// fill if the stack isn't primed to the expected length yet.
    fn roll_stack(&mut self) {
        let n = self.cfg.stack_num.max(1);
        if n <= 1 {
            return;
        }
        let frame_len = self.obs.len();
        if frame_len == 0 || self.stack.len() != n * frame_len {
            self.fill_stack();
            return;
        }
        self.stack.copy_within(frame_len.., 0);
        let newest = (n - 1) * frame_len;
        self.stack[newest..].copy_from_slice(&self.obs);
    }
}
