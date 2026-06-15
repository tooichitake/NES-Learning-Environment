use crate::constants::GRAY_FRAME_LEN;
use crate::env::{NesEnv, StepInfo};
use crate::games::MultiPlayerValues;
use nesle_common::Result;

use super::config::{ObsConfig, ObsKind, ObsShape, RenderPolicy};
use super::window::{FrameSample, ObsWindow, ObsWindowStep};

/// Training stepper: owns frame stepping/render policy and delegates the
/// observation window to [`ObsWindow`].
pub struct ObsPipeline {
    cfg: ObsConfig,
    window: ObsWindow,
    gray_tmp: Vec<u8>,
    rgb_tmp: Vec<u8>,
    ram_tmp: Vec<u8>,
}

impl ObsPipeline {
    pub fn new(cfg: ObsConfig) -> Self {
        Self {
            window: ObsWindow::new(cfg.clone()),
            cfg,
            gray_tmp: vec![0u8; GRAY_FRAME_LEN],
            rgb_tmp: Vec::new(),
            ram_tmp: Vec::new(),
        }
    }

    pub fn config(&self) -> &ObsConfig {
        &self.cfg
    }

    pub fn observation(&self) -> &[u8] {
        self.window.observation()
    }

    pub fn shape(&self) -> ObsShape {
        self.window.shape()
    }

    pub fn reset_in_place(&mut self, env: &mut NesEnv, noop_max: usize) -> Result<StepInfo> {
        if self.cfg.obs_kind.needs_pixels() {
            env.set_render_enabled(true);
        }
        let noops = if noop_max > 0 {
            1 + env.next_random_below(noop_max as u32) as usize
        } else {
            0
        };
        // Noop all active ports (`&[0]` when players == 1).
        let players = env.players() as usize;
        let noop_masks = [0u8; 4];
        let mut info = env.step_info();
        for _ in 0..noops {
            let o = env.step(&noop_masks[..players])?;
            info = o.info;
            if o.terminated[..players].iter().all(|&t| t) || o.truncated {
                info = env.reset()?.info;
            }
        }
        self.window.reset(info.lives);
        self.capture_env_buffers(env, true);
        let is_gray = matches!(self.cfg.obs_kind, ObsKind::Gray { .. });
        let is_rgb = matches!(self.cfg.obs_kind, ObsKind::Rgb { .. } | ObsKind::RgbNative);
        let is_ram = matches!(self.cfg.obs_kind, ObsKind::Ram);
        let sample = FrameSample::single(
            if is_gray {
                Some(self.gray_tmp.as_slice())
            } else {
                None
            },
            if is_rgb {
                Some(self.rgb_tmp.as_slice())
            } else {
                None
            },
            if is_ram {
                Some(self.ram_tmp.as_slice())
            } else {
                None
            },
            0.0,
            info,
            false,
            false,
        );
        self.window.refresh(sample)?;
        Ok(info)
    }

    /// One preprocessed agent step. `masks` carries one controller mask per active
    /// port (`masks.len()` should be `env.players()`); the NES screen is shared, so
    /// the observation is player-count agnostic while the returned `rewards` /
    /// `terminated` are per-port.
    pub fn step_in_place(&mut self, env: &mut NesEnv, masks: &[u8]) -> Result<ObsStepMeta> {
        let fs = self.cfg.frame_skip.max(1);
        let players = env.players() as usize;
        let mut last_step = ObsWindowStep {
            obs_step: false,
            rewards: [0.0; 4],
            sample_rewards: [0.0; 4],
            lives: [0; 4],
            frame_number: 0,
            episode_frame_number: 0,
            terminated: false,
            truncated: false,
        };
        // Per-port `terminated` of the last raw env step (the window's life-loss boundary is OR'd in below).
        let mut last_terminated: MultiPlayerValues<bool> = [false; 4];
        // The `terminal_on_life_loss` contribution that the raw env outcome lacks.
        let mut life_loss_break = false;
        for t in 0..fs {
            let render = self.cfg.should_render_subframe(t, fs);
            if self.cfg.render_policy == RenderPolicy::TrainingSparse {
                env.set_render_enabled(render);
            } else {
                env.set_render_enabled(true);
            }
            let o = env.step(masks)?;
            last_terminated = o.terminated;
            // The unit is done only when every active port terminated; this scalar drives the window boundary.
            let unit_done = o.terminated[..players].iter().all(|&done| done);
            let capture = self.cfg.should_capture_subframe(t, fs);
            self.capture_env_buffers(env, capture);
            let is_gray = capture && matches!(self.cfg.obs_kind, ObsKind::Gray { .. });
            let is_rgb =
                capture && matches!(self.cfg.obs_kind, ObsKind::Rgb { .. } | ObsKind::RgbNative);
            let is_ram = capture && matches!(self.cfg.obs_kind, ObsKind::Ram);
            let sample = FrameSample::multi(
                if is_gray {
                    Some(self.gray_tmp.as_slice())
                } else {
                    None
                },
                if is_rgb {
                    Some(self.rgb_tmp.as_slice())
                } else {
                    None
                },
                if is_ram {
                    Some(self.ram_tmp.as_slice())
                } else {
                    None
                },
                o.rewards,
                o.info,
                unit_done,
                o.truncated,
            );
            last_step = self.window.push_frame(sample, t + 1 == fs)?;
            // Record the life-loss augmentation the raw env outcome does not carry.
            life_loss_break = last_step.terminated && !unit_done;
            if last_step.terminated || last_step.truncated {
                break;
            }
        }
        env.set_render_enabled(true);
        // Per-port terminated: raw env outcome with the window's life-loss boundary OR'd into active ports.
        let mut terminated = last_terminated;
        if life_loss_break {
            for slot in terminated.iter_mut().take(players) {
                *slot = true;
            }
        }
        Ok(ObsStepMeta {
            // Window-accumulated per-port rewards (`[r, 0, 0, 0]` when players == 1).
            rewards: last_step.rewards,
            terminated,
            truncated: last_step.truncated,
            info: StepInfo {
                frame_number: last_step.frame_number,
                episode_frame_number: last_step.episode_frame_number,
                lives: last_step.lives,
            },
        })
    }

    fn capture_env_buffers(&mut self, env: &mut NesEnv, capture: bool) {
        match self.cfg.obs_kind {
            ObsKind::Gray { .. } if capture => {
                env.screen_grayscale_into(&mut self.gray_tmp);
            }
            ObsKind::Rgb { .. } | ObsKind::RgbNative if capture => {
                self.rgb_tmp.clear();
                self.rgb_tmp.extend_from_slice(&env.screen_rgb().pixels);
            }
            ObsKind::Ram if capture => {
                self.ram_tmp.clear();
                self.ram_tmp.extend_from_slice(env.ram());
            }
            _ => {}
        }
    }
}

/// Metadata for an in-place preprocessed step. The observation bytes live in
/// [`ObsPipeline::observation`].
///
/// The NES screen is shared across players, so the observation is player-count
/// agnostic; only the per-port `rewards`/`terminated` differ. For `players == 1`
/// these are exactly what `NesEnv::step` returns: `rewards == [r, 0, 0, 0]`,
/// `terminated == [t, true, true, true]`.
pub struct ObsStepMeta {
    pub rewards: MultiPlayerValues<f32>,
    pub terminated: MultiPlayerValues<bool>,
    pub truncated: bool,
    pub info: StepInfo,
}
