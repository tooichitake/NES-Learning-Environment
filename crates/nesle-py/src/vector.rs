#![expect(
    clippy::useless_conversion,
    reason = "PyO3 macro expansion reports PyResult returns as conversions"
)]

use crate::errors::map_error;
use nesle_rl::games::registry;
use nesle_rl::preprocess::{ObsConfig, RenderPolicy};
use nesle_rl::{AutoresetMode, NesVectorEnv, VectorConfig, VectorObsMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

type VectorInfo = (u64, u64, u8);
type VectorStep = (usize, f32, bool, bool, u64, u64, u8, bool);
/// One per-UNIT multi-agent step: (env_id, rewards[4], terminated[4], truncated,
/// frame_number, episode_frame_number, lives[4], final_observation). The full
/// per-port arrays (not index 0) so the Python facade can demux K units * N ports
/// into K*N slots.
type VectorPortsStep = (usize, [f32; 4], [bool; 4], bool, u64, u64, [u8; 4], bool);

/// `async_recv()` return: (obs bytes, env_ids, rewards, terminated, truncated,
/// frame_number, episode_frame_number, lives). `obs` is `batch_size * frame_len`
/// grayscale bytes in completion order; the Python wrapper reshapes + demuxes by
/// `env_ids`.
type AsyncBatch<'py> = (
    Bound<'py, PyBytes>,
    Vec<usize>,
    Vec<f32>,
    Vec<bool>,
    Vec<bool>,
    Vec<u64>,
    Vec<u64>,
    Vec<u8>,
);

/// One worker-pool vector env exposing BOTH the synchronous Gymnasium surface
/// (`reset`/`send`/`recv`/`step` + decoupled obs readers) and the envpool-style
/// asynchronous surface (`async_reset`/`async_send`/`async_recv`). `batch_size`
/// (in the constructor) selects which is meaningful.
#[pyclass(name = "NesVectorEnv", unsendable)]
pub(crate) struct PyNesVectorEnv {
    inner: NesVectorEnv,
}

#[pymethods]
impl PyNesVectorEnv {
    #[new]
    #[pyo3(signature = (
        num_envs,
        game_id = "super_mario_bros",
        rom = None,
        obs_mode = "rgb",
        players = 1,
        batch_size = 0,
        num_threads = 0,
        frame_skip = 1,
        width = 84,
        height = 84,
        maxpool = false,
        stack_num = 1,
        terminal_on_life_loss = false,
        repeat_action_probability = 0.0,
        noop_max = 0,
        seed = 1,
        max_episode_frames = None,
        autoreset_mode = "NextStep",
        remove_sprite_limit = false,
        start_state = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        game_id: &str,
        rom: Option<&[u8]>,
        obs_mode: &str,
        players: u8,
        batch_size: usize,
        num_threads: usize,
        frame_skip: usize,
        width: usize,
        height: usize,
        maxpool: bool,
        stack_num: usize,
        terminal_on_life_loss: bool,
        repeat_action_probability: f32,
        noop_max: usize,
        seed: u64,
        max_episode_frames: Option<u64>,
        autoreset_mode: &str,
        remove_sprite_limit: bool,
        start_state: Option<&str>,
    ) -> PyResult<Self> {
        let game = registry::find_game(game_id)
            .ok_or_else(|| PyValueError::new_err(format!("unknown NESLE game id: {game_id}")))?;
        let obs_mode = parse_obs_mode(obs_mode)?;
        let obs_cfg = if obs_mode == VectorObsMode::Preprocessed {
            Some(
                ObsConfig::gray_shape(
                    frame_skip,
                    width,
                    height,
                    maxpool,
                    RenderPolicy::TrainingSparse,
                    terminal_on_life_loss,
                )
                .with_stack_num(stack_num),
            )
        } else {
            None
        };
        let config = VectorConfig {
            num_envs,
            players,
            batch_size,
            num_threads,
            obs_mode,
            obs_cfg,
            frame_skip,
            remove_sprite_limit,
            start_state: start_state.map(str::to_string),
            repeat_action_probability,
            noop_max,
            seed,
            max_episode_frames,
            autoreset_mode: parse_autoreset_mode(autoreset_mode)?,
        };
        let inner = NesVectorEnv::new(game, config, rom.unwrap_or(&[])).map_err(map_error)?;
        Ok(Self { inner })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }

    /// Controller ports each unit drives (1..=game.players).
    fn players(&self) -> u8 {
        self.inner.players()
    }

    fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }

    fn frame_len(&self) -> usize {
        self.inner.frame_len()
    }

    fn load_rom_bytes(&mut self, bytes: &[u8]) -> PyResult<()> {
        self.inner.load_rom_bytes(bytes).map_err(map_error)
    }

    fn set_remove_sprite_limit(&mut self, enabled: bool) {
        self.inner.set_remove_sprite_limit(enabled);
    }

    fn set_start_state(&mut self, start_state: &str) -> PyResult<()> {
        self.inner
            .set_start_state_id(start_state)
            .map_err(map_error)
    }

    // -- synchronous full-batch ----------------------------------------------

    fn reset(&mut self) -> PyResult<Vec<VectorInfo>> {
        Ok(self
            .inner
            .reset()
            .map_err(map_error)?
            .into_iter()
            .map(|o| {
                (
                    o.info.frame_number,
                    o.info.episode_frame_number,
                    o.info.lives[0],
                )
            })
            .collect())
    }

    fn send(&mut self, actions: Vec<u8>) -> PyResult<()> {
        self.inner.send(actions).map_err(map_error)
    }

    fn recv(&mut self, py: Python<'_>) -> PyResult<Vec<VectorStep>> {
        let steps = py.allow_threads(|| self.inner.recv()).map_err(map_error)?;
        Ok(steps.into_iter().map(vector_step_tuple).collect())
    }

    fn step(&mut self, py: Python<'_>, actions: Vec<u8>) -> PyResult<Vec<VectorStep>> {
        self.send(actions)?;
        self.recv(py)
    }

    /// Drain a full sorted batch and return per-UNIT tuples carrying the FULL per-port
    /// arrays (not index 0): `(env_id, rewards[4], terminated[4], truncated,
    /// frame_number, episode_frame_number, lives[4], final_observation)`. For
    /// multi-agent units the Python facade fans these out into K*N agent slots. The
    /// GIL is released around the inner recv (like `recv`).
    fn recv_ports(&mut self, py: Python<'_>) -> PyResult<Vec<VectorPortsStep>> {
        let steps = py.allow_threads(|| self.inner.recv()).map_err(map_error)?;
        Ok(steps.into_iter().map(vector_ports_tuple).collect())
    }

    fn ram_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        PyBytes::new_bound_with(py, self.inner.ram_batch_byte_len(), |bytes| {
            self.inner.ram_batch_into(bytes);
            Ok(())
        })
    }

    fn set_ram_batch(&mut self, bytes: &[u8]) -> PyResult<()> {
        self.inner.set_ram_batch(bytes).map_err(map_error)
    }

    fn nametable_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        PyBytes::new_bound_with(py, self.inner.vram_batch_byte_len(), |bytes| {
            self.inner.vram_batch_into(bytes);
            Ok(())
        })
    }

    fn grayscale_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        PyBytes::new_bound_with(py, self.inner.grayscale_batch_byte_len(), |bytes| {
            self.inner.grayscale_batch_into(bytes);
            Ok(())
        })
    }

    fn rgb_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        PyBytes::new_bound_with(py, self.inner.rgb_batch_byte_len(), |bytes| {
            self.inner.rgb_batch_into(bytes);
            Ok(())
        })
    }

    fn observation_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let len = self.inner.observation_batch_byte_len().map_err(map_error)?;
        PyBytes::new_bound_with(py, len, |bytes| {
            self.inner.observation_batch_into(bytes).map_err(map_error)
        })
    }

    // -- asynchronous (envpool) ----------------------------------------------

    fn async_reset(&self) -> PyResult<()> {
        self.inner.async_reset().map_err(map_error)
    }

    fn async_send(&self, py: Python<'_>, env_ids: Vec<usize>, actions: Vec<u8>) -> PyResult<()> {
        py.allow_threads(|| self.inner.async_send(&env_ids, &actions))
            .map_err(map_error)
    }

    fn async_recv<'py>(&self, py: Python<'py>) -> PyResult<AsyncBatch<'py>> {
        let batch = py
            .allow_threads(|| self.inner.async_recv())
            .map_err(map_error)?;
        let frame_len = self.inner.frame_len();
        let obs = PyBytes::new_bound_with(py, batch.len() * frame_len, |bytes| {
            for (i, c) in batch.iter().enumerate() {
                bytes[i * frame_len..(i + 1) * frame_len].copy_from_slice(&c.obs);
            }
            Ok(())
        })?;
        let env_ids = batch.iter().map(|c| c.env_id).collect();
        let rewards = batch.iter().map(|c| c.rewards[0]).collect();
        let terminated = batch.iter().map(|c| c.terminated[0]).collect();
        let truncated = batch.iter().map(|c| c.truncated).collect();
        let frame_number = batch.iter().map(|c| c.frame_number).collect();
        let episode_frame_number = batch.iter().map(|c| c.episode_frame_number).collect();
        let lives = batch.iter().map(|c| c.lives[0]).collect();
        Ok((
            obs,
            env_ids,
            rewards,
            terminated,
            truncated,
            frame_number,
            episode_frame_number,
            lives,
        ))
    }
}

fn vector_step_tuple(step: nesle_rl::VectorStep) -> VectorStep {
    (
        step.env_id,
        step.outcome.rewards[0],
        step.outcome.terminated[0],
        step.outcome.truncated,
        step.outcome.info.frame_number,
        step.outcome.info.episode_frame_number,
        step.outcome.info.lives[0],
        step.final_observation,
    )
}

fn vector_ports_tuple(step: nesle_rl::VectorStep) -> VectorPortsStep {
    (
        step.env_id,
        step.outcome.rewards,
        step.outcome.terminated,
        step.outcome.truncated,
        step.outcome.info.frame_number,
        step.outcome.info.episode_frame_number,
        step.outcome.info.lives,
        step.final_observation,
    )
}

fn parse_obs_mode(value: &str) -> PyResult<VectorObsMode> {
    match value {
        "preprocessed" => Ok(VectorObsMode::Preprocessed),
        "ram" => Ok(VectorObsMode::Ram),
        "rgb" => Ok(VectorObsMode::Rgb),
        "grayscale" => Ok(VectorObsMode::Grayscale),
        _ => Err(PyValueError::new_err(format!(
            "unsupported obs_mode: {value}"
        ))),
    }
}

fn parse_autoreset_mode(value: &str) -> PyResult<AutoresetMode> {
    match value {
        "NextStep" => Ok(AutoresetMode::NextStep),
        "SameStep" => Ok(AutoresetMode::SameStep),
        _ => Err(PyValueError::new_err(format!(
            "unsupported autoreset_mode: {value}"
        ))),
    }
}
