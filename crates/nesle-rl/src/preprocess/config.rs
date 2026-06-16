use crate::constants::{NES_HEIGHT, NES_WIDTH, RAM_SIZE};
use crate::games::MultiPlayerValues;

/// Pixel-rendering policy above the emulator core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPolicy {
    /// Training hot path: disable framebuffer composition on frame-skip subframes
    /// that cannot affect the returned observation.
    TrainingSparse,
    /// Human-visible path: render every frame because a native RGB viewer is
    /// watching the same kernel state that the observation is derived from.
    HumanVisible,
}

impl RenderPolicy {
    pub fn from_render_skip(render_skip: bool) -> Self {
        if render_skip {
            Self::TrainingSparse
        } else {
            Self::HumanVisible
        }
    }
}

/// Observation representation produced by the shared preprocessing window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObsKind {
    Gray {
        width: usize,
        height: usize,
        maxpool: bool,
    },
    Rgb {
        width: usize,
        height: usize,
    },
    RgbNative,
    Ram,
}

impl ObsKind {
    pub fn gray(screen_size: usize, maxpool: bool) -> Self {
        Self::gray_shape(screen_size, screen_size, maxpool)
    }

    pub fn gray_shape(width: usize, height: usize, maxpool: bool) -> Self {
        Self::Gray {
            width,
            height,
            maxpool,
        }
    }

    pub fn rgb(screen_size: usize) -> Self {
        Self::rgb_shape(screen_size, screen_size)
    }

    pub fn rgb_shape(width: usize, height: usize) -> Self {
        Self::Rgb { width, height }
    }

    pub fn shape(&self) -> ObsShape {
        match *self {
            Self::Gray { width, height, .. } => ObsShape {
                stack: 1,
                width,
                height,
                channels: 1,
            },
            Self::Rgb { width, height } => ObsShape {
                stack: 1,
                width,
                height,
                channels: 3,
            },
            Self::RgbNative => ObsShape {
                stack: 1,
                width: NES_WIDTH,
                height: NES_HEIGHT,
                channels: 3,
            },
            Self::Ram => ObsShape {
                stack: 1,
                width: RAM_SIZE,
                height: 1,
                channels: 1,
            },
        }
    }

    pub fn needs_pixels(&self) -> bool {
        !matches!(self, Self::Ram)
    }

    pub fn maxpool(&self) -> bool {
        matches!(self, Self::Gray { maxpool: true, .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObsShape {
    /// Stacked frames (`stack_num`; `1` = no stacking). The full observation
    /// is `stack` consecutive `width*height*channels` frames, oldest -> newest.
    pub stack: usize,
    pub width: usize,
    pub height: usize,
    pub channels: u8,
}

impl ObsShape {
    pub fn len(self) -> usize {
        self.stack * self.width * self.height * self.channels as usize
    }

    pub fn is_empty(self) -> bool {
        self.stack == 0 || self.width == 0 || self.height == 0 || self.channels == 0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RewardClip {
    pub positive: f32,
    pub negative: f32,
}

impl RewardClip {
    pub const fn none() -> Self {
        Self {
            positive: 0.0,
            negative: 0.0,
        }
    }

    pub const fn symmetric(limit: f32) -> Self {
        Self {
            positive: limit,
            negative: limit,
        }
    }

    pub fn apply(self, reward: f32) -> f32 {
        let hi = if self.positive > 0.0 {
            self.positive
        } else {
            f32::INFINITY
        };
        let lo = if self.negative > 0.0 {
            -self.negative
        } else {
            f32::NEG_INFINITY
        };
        reward.clamp(lo, hi)
    }

    pub fn apply_all(self, rewards: MultiPlayerValues<f32>) -> MultiPlayerValues<f32> {
        rewards.map(|r| self.apply(r))
    }
}

/// Observation-preprocessing config shared by training and Serve.
#[derive(Debug, Clone)]
pub struct ObsConfig {
    pub frame_skip: usize,
    pub obs_kind: ObsKind,
    pub render_policy: RenderPolicy,
    pub terminal_on_life_loss: bool,
    pub repeat_action_probability: f32,
    pub noop_max: usize,
    pub reward_clip: RewardClip,
    /// `stack_num`: the observation is this many consecutive preprocessed
    /// frames (oldest -> newest). `1` (default) = no stacking — the window serves a
    /// single frame, byte-identical to the pre-stacking pipeline.
    pub stack_num: usize,
    /// Active controller ports (1..=4). `terminal_on_life_loss` only inspects this
    /// many life slots; trailing MultiPlayerValues slots are unused game RAM, not lives.
    pub players: u8,
}

impl ObsConfig {
    pub fn gray(
        frame_skip: usize,
        screen_size: usize,
        maxpool: bool,
        render_policy: RenderPolicy,
        terminal_on_life_loss: bool,
    ) -> Self {
        Self::gray_shape(
            frame_skip,
            screen_size,
            screen_size,
            maxpool,
            render_policy,
            terminal_on_life_loss,
        )
    }

    pub fn gray_shape(
        frame_skip: usize,
        width: usize,
        height: usize,
        maxpool: bool,
        render_policy: RenderPolicy,
        terminal_on_life_loss: bool,
    ) -> Self {
        Self {
            frame_skip,
            obs_kind: ObsKind::gray_shape(width, height, maxpool),
            render_policy,
            terminal_on_life_loss,
            ..Self::default()
        }
    }

    pub fn with_obs_kind(mut self, obs_kind: ObsKind) -> Self {
        self.obs_kind = obs_kind;
        self
    }

    /// Set the frame-stack depth (`1` = no stacking). Clamped to `>= 1`.
    pub fn with_stack_num(mut self, stack_num: usize) -> Self {
        self.stack_num = stack_num.max(1);
        self
    }

    pub fn shape(&self) -> ObsShape {
        let mut shape = self.obs_kind.shape();
        shape.stack = self.stack_num.max(1);
        shape
    }

    pub fn screen_size(&self) -> usize {
        match self.obs_kind {
            ObsKind::Gray { width, .. } => width,
            ObsKind::Rgb { width, .. } => width,
            ObsKind::RgbNative => NES_WIDTH,
            ObsKind::Ram => RAM_SIZE,
        }
    }

    pub fn maxpool(&self) -> bool {
        self.obs_kind.maxpool()
    }

    pub(crate) fn should_capture_subframe(&self, subframe: usize, frame_skip: usize) -> bool {
        let last = subframe + 1 == frame_skip;
        let second_last = subframe + 2 == frame_skip;
        match self.obs_kind {
            ObsKind::Gray { maxpool, .. } => last || (maxpool && second_last),
            ObsKind::Rgb { .. } | ObsKind::RgbNative | ObsKind::Ram => last,
        }
    }

    pub(crate) fn should_render_subframe(&self, subframe: usize, frame_skip: usize) -> bool {
        match self.render_policy {
            RenderPolicy::HumanVisible => true,
            RenderPolicy::TrainingSparse => {
                self.obs_kind.needs_pixels() && self.should_capture_subframe(subframe, frame_skip)
            }
        }
    }
}

impl Default for ObsConfig {
    fn default() -> Self {
        Self {
            frame_skip: 4,
            obs_kind: ObsKind::gray(84, true),
            render_policy: RenderPolicy::TrainingSparse,
            terminal_on_life_loss: false,
            repeat_action_probability: 0.0,
            noop_max: 30,
            reward_clip: RewardClip::none(),
            stack_num: 1,
            players: 1,
        }
    }
}
