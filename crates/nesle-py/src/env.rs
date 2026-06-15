#![expect(
    clippy::useless_conversion,
    reason = "PyO3 macro expansion reports PyResult returns as conversions"
)]

use crate::errors::map_error;
#[cfg(feature = "viewer")]
use crate::human_window::PyHumanWindow;
use crate::state::PyEnvState;
use nesle_rl::games::registry;
use nesle_rl::NesEnv;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

type EnvReset = (u64, u64, u8);
type EnvStep = (f32, bool, bool, u64, u64, u8);
type ObserveReset<'py> = (Bound<'py, PyBytes>, u64, u64, u8);
type ObserveStep<'py> = (Bound<'py, PyBytes>, f32, bool, bool, u64, u64, u8);
/// Multi-port (self-play) reset/step: per-port rewards/terminated/lives `[_; 4]`.
type PortsReset = (u64, u64, [u8; 4]);
type PortsStep = ([f32; 4], [bool; 4], bool, u64, u64, [u8; 4]);

#[pyclass(name = "NesEnv", unsendable)]
pub(crate) struct PyNesEnv {
    inner: NesEnv,
    obs: Option<nesle_rl::preprocess::ObsPipeline>,
}

impl PyNesEnv {
    #[allow(clippy::too_many_arguments)]
    fn configure_observation(
        &mut self,
        frame_skip: usize,
        width: usize,
        height: usize,
        maxpool: bool,
        render_skip: bool,
        terminal_on_life_loss: bool,
        stack_num: usize,
    ) -> PyResult<()> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err(
                "observation dimensions must be positive",
            ));
        }
        let render_policy = nesle_rl::preprocess::RenderPolicy::from_render_skip(render_skip);
        self.obs = Some(nesle_rl::preprocess::ObsPipeline::new(
            nesle_rl::preprocess::ObsConfig::gray_shape(
                frame_skip,
                width,
                height,
                maxpool,
                render_policy,
                terminal_on_life_loss,
            )
            .with_stack_num(stack_num),
        ));
        Ok(())
    }
}

#[pymethods]
impl PyNesEnv {
    #[new]
    #[pyo3(signature = (game_id = "super_mario_bros"))]
    fn new(game_id: &str) -> PyResult<Self> {
        let game = registry::find_game(game_id)
            .ok_or_else(|| PyValueError::new_err(format!("unknown NESLE game id: {game_id}")))?;
        let inner = NesEnv::new(game);
        Ok(Self { inner, obs: None })
    }

    fn load_rom_bytes(&mut self, bytes: &[u8]) -> PyResult<()> {
        self.inner.load_rom_bytes(bytes).map_err(map_error)
    }

    fn reset(&mut self) -> PyResult<EnvReset> {
        let outcome = self.inner.reset_to_start_state().map_err(map_error)?;
        Ok((
            outcome.info.frame_number,
            outcome.info.episode_frame_number,
            outcome.info.lives[0],
        ))
    }

    fn set_start_state(&mut self, start_state: &str) -> PyResult<()> {
        self.inner
            .set_start_state_id(start_state)
            .map_err(map_error)
    }

    fn set_start_state_path(&mut self, path: &str) -> PyResult<()> {
        self.inner.set_start_state_path(path).map_err(map_error)
    }

    /// Single-agent step (port 0). The Python single-agent façade and the gate's
    /// direct `_nesle.NesEnv(...).step(int)` calls use this; `step_ports` drives
    /// multi-agent self-play. Byte-identical to the pre-unification scalar step
    /// (`players == 1` slices `act_n` to port 0).
    fn step(&mut self, action_mask: u8) -> PyResult<EnvStep> {
        let outcome = self.inner.step(&[action_mask]).map_err(map_error)?;
        Ok((
            outcome.rewards[0],
            outcome.terminated[0],
            outcome.truncated,
            outcome.info.frame_number,
            outcome.info.episode_frame_number,
            outcome.info.lives[0],
        ))
    }

    #[pyo3(signature = (frame_skip, screen_size, maxpool, render_skip = true, terminal_on_life_loss = false, stack_num = 1))]
    fn configure_obs(
        &mut self,
        frame_skip: usize,
        screen_size: usize,
        maxpool: bool,
        render_skip: bool,
        terminal_on_life_loss: bool,
        stack_num: usize,
    ) -> PyResult<()> {
        self.configure_observation(
            frame_skip,
            screen_size,
            screen_size,
            maxpool,
            render_skip,
            terminal_on_life_loss,
            stack_num,
        )
    }

    #[pyo3(signature = (frame_skip, width, height, maxpool, render_skip = true, terminal_on_life_loss = false, stack_num = 1))]
    #[allow(clippy::too_many_arguments)]
    fn configure_obs_shape(
        &mut self,
        frame_skip: usize,
        width: usize,
        height: usize,
        maxpool: bool,
        render_skip: bool,
        terminal_on_life_loss: bool,
        stack_num: usize,
    ) -> PyResult<()> {
        self.configure_observation(
            frame_skip,
            width,
            height,
            maxpool,
            render_skip,
            terminal_on_life_loss,
            stack_num,
        )
    }

    fn observe_reset<'py>(
        &mut self,
        py: Python<'py>,
        noop_max: usize,
    ) -> PyResult<ObserveReset<'py>> {
        let pipe = self.obs.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("configure_obs() must be called before observe_reset()")
        })?;
        let info = pipe
            .reset_in_place(&mut self.inner, noop_max)
            .map_err(map_error)?;
        Ok((
            PyBytes::new_bound(py, pipe.observation()),
            info.frame_number,
            info.episode_frame_number,
            info.lives[0],
        ))
    }

    fn observe_step<'py>(&mut self, py: Python<'py>, mask: u8) -> PyResult<ObserveStep<'py>> {
        let pipe = self.obs.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("configure_obs() must be called before observe_step()")
        })?;
        let s = pipe
            .step_in_place(&mut self.inner, &[mask])
            .map_err(map_error)?;
        Ok((
            PyBytes::new_bound(py, pipe.observation()),
            s.rewards[0],
            s.terminated[0],
            s.truncated,
            s.info.frame_number,
            s.info.episode_frame_number,
            s.info.lives[0],
        ))
    }

    #[cfg(feature = "viewer")]
    fn step_human(
        &mut self,
        action_mask: u8,
        window: &Bound<'_, PyHumanWindow>,
    ) -> PyResult<EnvStep> {
        let mut win = window.borrow_mut();
        let outcome = self
            .inner
            .step_rendered(&[action_mask], &mut *win)
            .map_err(map_error)?;
        Ok((
            outcome.rewards[0],
            outcome.terminated[0],
            outcome.truncated,
            outcome.info.frame_number,
            outcome.info.episode_frame_number,
            outcome.info.lives[0],
        ))
    }

    #[pyo3(signature = (max_episode_frames = None))]
    fn set_max_episode_frames(&mut self, max_episode_frames: Option<u64>) {
        self.inner.set_max_episode_frames(max_episode_frames);
    }

    #[pyo3(signature = (frame_skip = 1, repeat_action_probability = 0.0))]
    fn set_action_repeat(
        &mut self,
        frame_skip: usize,
        repeat_action_probability: f32,
    ) -> PyResult<()> {
        self.inner
            .set_action_repeat(frame_skip, repeat_action_probability)
            .map_err(map_error)
    }

    fn seed(&mut self, seed: u64) {
        self.inner.seed(seed);
    }

    fn ram<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, self.inner.ram())
    }

    /// PPU nametable (CIRAM) bytes -- the rendered field tile ids, for scripted
    /// bots that need the wall/brick/floor layout (not stored per-tile in CPU RAM).
    fn nametable<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, self.inner.vram())
    }

    fn screen_indexed<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.screen_indexed().pixels)
    }

    fn screen_gray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        PyBytes::new_bound_with(py, 240 * 256, |bytes| {
            self.inner.screen_grayscale_into(bytes);
            Ok(())
        })
    }

    fn screen_rgb<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.screen_rgb().pixels)
    }

    fn set_ram(&mut self, bytes: &[u8]) -> PyResult<()> {
        self.inner.set_ram(bytes).map_err(map_error)
    }

    fn set_audio_enabled(&mut self, enabled: bool) {
        self.inner.set_audio_enabled(enabled);
    }

    fn set_remove_sprite_limit(&mut self, enabled: bool) {
        self.inner.set_remove_sprite_limit(enabled);
    }

    fn set_render_enabled(&mut self, enabled: bool) {
        self.inner.set_render_enabled(enabled);
    }

    fn clone_state(&self) -> PyEnvState {
        PyEnvState {
            state: self.inner.clone_state(),
        }
    }

    fn restore_state(&mut self, state: &PyEnvState) -> PyResult<()> {
        self.inner.restore_state(&state.state).map_err(map_error)
    }

    fn save_state_blob<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.save_state_blob())
    }

    fn restore_state_blob(&mut self, blob: &[u8]) -> PyResult<()> {
        self.inner
            .restore_state_blob(blob.to_vec())
            .map_err(map_error)
    }

    fn minimal_action_set(&self) -> Vec<(String, u8)> {
        self.inner
            .game()
            .minimal_actions
            .iter()
            .map(|action| (action.name.to_string(), action.mask))
            .collect()
    }

    fn full_action_set(&self) -> Vec<(String, u8)> {
        nesle_common::NES_FULL_ACTION_SET
            .iter()
            .map(|action| (action.name.to_string(), action.mask))
            .collect()
    }

    // -- multi-agent self-play (1..=game.players ports) ----------------------

    /// Set the number of active controller ports (1..=game.players). The single-
    /// agent façade leaves this at 1; the multi-player façade sets it to the mode's
    /// player count.
    fn set_players(&mut self, players: u8) -> PyResult<()> {
        self.inner.set_players(players).map_err(map_error)
    }

    fn num_players(&self) -> u8 {
        self.inner.players()
    }

    /// Reset returning per-port lives `[u8; 4]` (multi-player façade). `reset`
    /// returns the port-0 scalar for the single-agent surface.
    fn reset_ports(&mut self) -> PyResult<PortsReset> {
        let outcome = self.inner.reset_to_start_state().map_err(map_error)?;
        Ok((
            outcome.info.frame_number,
            outcome.info.episode_frame_number,
            outcome.info.lives,
        ))
    }

    /// Multi-port step: one mask per active port, per-port rewards/terminated/lives
    /// `[_; 4]`. `step` drives port 0 only (single-agent surface).
    fn step_ports(&mut self, masks: Vec<u8>) -> PyResult<PortsStep> {
        let expected = self.inner.players() as usize;
        if masks.len() != expected {
            return Err(PyValueError::new_err(format!(
                "step_ports() expects {expected} masks for this game, got {}",
                masks.len()
            )));
        }
        let outcome = self.inner.step(&masks).map_err(map_error)?;
        Ok((
            outcome.rewards,
            outcome.terminated,
            outcome.truncated,
            outcome.info.frame_number,
            outcome.info.episode_frame_number,
            outcome.info.lives,
        ))
    }
}
