//! Vectorized environment — ONE worker-pool engine (ale-py `ALEVectorInterface` style).
//!
//! A pool of worker threads each pulls a `(env_id, action)` task from a lock-free
//! MPMC action queue, steps THAT env (each env is touched by one worker at a time —
//! the caller keeps a single in-flight task per env — so the per-env `Mutex` is
//! uncontended and just satisfies `Sync`), and pushes a `Completion`. `batch_size`
//! selects the consumption discipline:
//!
//! - **Synchronous** (`batch_size == num_envs`, the default): `reset`/`send`/`recv`/
//!   `step` push N tasks, drain ALL N completions, and **sort by `env_id`** → fixed
//!   env-indexed, deterministic order (mirrors the Gymnasium `VectorEnv` / SB3
//!   target). Raw observations (ram/rgb/grayscale) are read on demand AFTER the
//!   barrier by locking each idle env (`*_batch_into`); preprocessed obs are also
//!   captured inline.
//! - **Asynchronous** (`0 < batch_size < num_envs`, envpool style): `async_reset` /
//!   `async_send(env_ids, actions)` / `async_recv()` returns the FIRST `batch_size`
//!   completions (out-of-order, each tagged with `env_id`), so the extra envs keep
//!   stepping while the policy forward pass runs — env stepping and inference
//!   overlap. Preprocessed obs only (the env keeps moving, so obs is captured inline
//!   by the worker). The PyO3 layer releases the GIL around send/recv.
//!
//! Determinism: completion order depends on wall-clock timing, but the sync path
//! re-imposes `env_id` order via `drain_sorted`, and each env's outcome depends only
//! on its own `(seed, action, prior state)` — never on sibling timing — so sorted
//! output is bit-identical run-to-run and matches N independent single envs.

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{unbounded, Receiver, Sender};

use crate::autoreset::AutoresetMode;
use crate::constants::{GRAY_FRAME_LEN, RAM_SIZE, RGB_FRAME_LEN};
use crate::env::{NesEnv, ResetOutcome, StepInfo, StepOutcome};
use crate::games::{GameSpec, MultiPlayerValues};
use crate::preprocess::{ObsConfig, ObsPipeline};
use nesle_common::{NesleError, Result};

/// Which observation the vector env serves. Preprocessed runs the `ObsPipeline`
/// (gray, owns the frame-skip) and is the only mode that supports async; the raw
/// modes step the env directly (env-level frame-skip) and are read on demand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VectorObsMode {
    Preprocessed,
    Ram,
    Rgb,
    Grayscale,
}

impl VectorObsMode {
    fn is_preprocessed(self) -> bool {
        matches!(self, VectorObsMode::Preprocessed)
    }
}

#[derive(Clone)]
pub struct VectorConfig {
    pub num_envs: usize,
    /// Controller ports each worker-env (a "unit") drives (1..=`game.players`). `1`
    /// is the single-agent default. With `players == N` the K units run K parallel
    /// N-agent matches; the action stream is flat unit-major (`num_envs * players`
    /// masks).
    pub players: u8,
    /// `0` or `num_envs` → synchronous full-batch; `0 < batch_size < num_envs` → async.
    pub batch_size: usize,
    /// `0` → one worker thread per env.
    pub num_threads: usize,
    pub obs_mode: VectorObsMode,
    /// Required (and only used) for `Preprocessed`.
    pub obs_cfg: Option<ObsConfig>,
    pub frame_skip: usize,
    pub remove_sprite_limit: bool,
    pub start_state: Option<String>,
    pub repeat_action_probability: f32,
    pub noop_max: usize,
    pub seed: u64,
    pub max_episode_frames: Option<u64>,
    pub autoreset_mode: AutoresetMode,
}

impl VectorConfig {
    pub fn new(num_envs: usize) -> Self {
        Self {
            num_envs,
            players: 1,
            batch_size: 0,
            num_threads: 0,
            obs_mode: VectorObsMode::Rgb,
            obs_cfg: None,
            frame_skip: 1,
            remove_sprite_limit: false,
            start_state: None,
            repeat_action_probability: 0.0,
            noop_max: 0,
            seed: 1,
            max_episode_frames: None,
            autoreset_mode: AutoresetMode::NextStep,
        }
    }
}

/// One finished synchronous env step (or reset), env-indexed.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorStep {
    pub env_id: usize,
    pub outcome: StepOutcome,
    pub final_observation: bool,
}

/// One worker-owned env. `pipe` is `Some` only for the preprocessed obs mode.
struct WorkerEnv {
    env: NesEnv,
    pipe: Option<ObsPipeline>,
    /// Set when a `NextStep`-autoreset env finished its episode and must reset on
    /// its next task instead of stepping.
    needs_reset: bool,
}

enum Task {
    Reset {
        env_id: usize,
    },
    /// One agent step for unit `env_id`. `masks[i]` is port i's action mask;
    /// trailing ports (>= the unit's `players`) are zero and ignored by the env.
    Step {
        env_id: usize,
        masks: [u8; 4],
    },
}

/// A finished env step (or reset), ready for the consumer. `obs` is the inline
/// preprocessed observation (empty for raw modes — those read on demand). Per-port
/// `rewards`/`terminated`/`lives` (controller 0..4; trailing slots zero / true);
/// single-agent units read index 0. `final_observation` is the worker-computed
/// "episode done" (`terminated[..players].all() || truncated`) -- stored so the
/// consumer needn't know each unit's port count.
pub struct Completion {
    pub env_id: usize,
    pub obs: Vec<u8>,
    pub rewards: MultiPlayerValues<f32>,
    pub terminated: MultiPlayerValues<bool>,
    pub truncated: bool,
    pub final_observation: bool,
    pub frame_number: u64,
    pub episode_frame_number: u64,
    pub lives: MultiPlayerValues<u8>,
}

pub struct NesVectorEnv {
    num_envs: usize,
    players: u8,
    batch_size: usize,
    obs_mode: VectorObsMode,
    frame_len: usize,
    envs: Arc<Vec<Mutex<WorkerEnv>>>,
    // `Option` so `Drop` can disconnect the queue (workers then drain + exit).
    action_tx: Option<Sender<Task>>,
    completion_rx: Receiver<Completion>,
    workers: Vec<JoinHandle<()>>,
}

impl NesVectorEnv {
    pub fn new(game: &'static GameSpec, config: VectorConfig, rom: &[u8]) -> Result<Self> {
        let VectorConfig {
            num_envs,
            players,
            batch_size,
            num_threads,
            obs_mode,
            obs_cfg,
            frame_skip,
            remove_sprite_limit,
            start_state,
            repeat_action_probability,
            noop_max,
            seed,
            max_episode_frames,
            autoreset_mode,
        } = config;
        let players = players.max(1);

        if num_envs == 0 {
            return Err(NesleError::InvalidState(
                "vector env requires num_envs >= 1".to_string(),
            ));
        }
        let batch_size = if batch_size == 0 {
            num_envs
        } else {
            batch_size
        };
        if batch_size > num_envs {
            return Err(NesleError::InvalidState(
                "vector env requires batch_size <= num_envs".to_string(),
            ));
        }
        if obs_mode.is_preprocessed() && obs_cfg.is_none() {
            return Err(NesleError::InvalidState(
                "preprocessed vector env requires an obs config".to_string(),
            ));
        }
        if !obs_mode.is_preprocessed() && batch_size != num_envs {
            return Err(NesleError::InvalidState(
                "async vector env (batch_size < num_envs) requires the preprocessed obs mode"
                    .to_string(),
            ));
        }
        // Preprocessed: the ObsPipeline owns the frame-skip (env steps 1 frame/action); raw: env-level frame-skip.
        let env_repeat = if obs_mode.is_preprocessed() {
            1
        } else {
            frame_skip
        };
        let frame_len = obs_cfg.as_ref().map(|c| c.shape().len()).unwrap_or(0);

        let mut worker_envs = Vec::with_capacity(num_envs);
        for i in 0..num_envs {
            let mut env = NesEnv::new(game);
            // Propagate set_players' Result so a single-player spec with players > 1 errors clearly.
            env.set_players(players)?;
            env.set_max_episode_frames(max_episode_frames);
            env.set_action_repeat(env_repeat, repeat_action_probability)?;
            env.seed(seed.wrapping_add(i as u64).max(1));
            if !rom.is_empty() {
                env.load_rom_bytes(rom)?;
            }
            env.set_remove_sprite_limit(remove_sprite_limit);
            if let Some(state) = start_state.as_deref() {
                env.set_start_state_id(state)?;
            }
            let pipe = obs_cfg.clone().map(ObsPipeline::new);
            worker_envs.push(Mutex::new(WorkerEnv {
                env,
                pipe,
                needs_reset: false,
            }));
        }

        let envs = Arc::new(worker_envs);
        let (action_tx, action_rx) = unbounded::<Task>();
        let (completion_tx, completion_rx) = unbounded::<Completion>();

        let num_threads = if num_threads == 0 {
            num_envs
        } else {
            num_threads
        };
        let num_threads = num_threads.clamp(1, num_envs);
        let workers = (0..num_threads)
            .map(|_| {
                let envs = Arc::clone(&envs);
                let rx = action_rx.clone();
                let tx = completion_tx.clone();
                thread::spawn(move || worker_loop(&envs, &rx, &tx, noop_max, autoreset_mode))
            })
            .collect();

        Ok(Self {
            num_envs,
            players,
            batch_size,
            obs_mode,
            frame_len,
            envs,
            action_tx: Some(action_tx),
            completion_rx,
            workers,
        })
    }

    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Controller ports each unit (worker-env) drives. `1` for the single-agent
    /// default; `K` units each drive `players` ports, so the flat action stream is
    /// `num_envs * players` masks (unit-major).
    pub fn players(&self) -> u8 {
        self.players
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn frame_len(&self) -> usize {
        self.frame_len
    }

    pub fn len(&self) -> usize {
        self.num_envs
    }

    pub fn is_empty(&self) -> bool {
        self.num_envs == 0
    }

    // -- construction-time config (workers idle; locks uncontended) ----------

    pub fn load_rom_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        for m in self.envs.iter() {
            m.lock()
                .expect("env mutex poisoned")
                .env
                .load_rom_bytes(bytes)?;
        }
        Ok(())
    }

    pub fn set_remove_sprite_limit(&mut self, enabled: bool) {
        for m in self.envs.iter() {
            m.lock()
                .expect("env mutex poisoned")
                .env
                .set_remove_sprite_limit(enabled);
        }
    }

    pub fn set_start_state_id(&mut self, start_state: &str) -> Result<()> {
        for m in self.envs.iter() {
            m.lock()
                .expect("env mutex poisoned")
                .env
                .set_start_state_id(start_state)?;
        }
        Ok(())
    }

    // -- synchronous full-batch (reset/send/recv/step) -----------------------

    /// Reset every env and return env-indexed outcomes. Preprocessed envs apply the
    /// noop start + capture obs; raw envs do a plain reset.
    pub fn reset(&mut self) -> Result<Vec<ResetOutcome>> {
        let tx = self.sender()?;
        for env_id in 0..self.num_envs {
            tx.send(Task::Reset { env_id }).map_err(|_| worker_gone())?;
        }
        let batch = self.drain_sorted()?;
        Ok(batch
            .into_iter()
            .map(|c| ResetOutcome { info: info_of(&c) })
            .collect())
    }

    /// Queue actions for every unit (fixed order). The stream is flat unit-major:
    /// `actions.len()` must equal `num_envs * players`, and unit `u`'s port masks are
    /// `actions[u*players .. u*players + players]`. Errors otherwise. For
    /// `players == 1` this is one mask per unit.
    pub fn send(&mut self, actions: Vec<u8>) -> Result<()> {
        let players = self.players as usize;
        let expected = self.num_envs * players;
        if actions.len() != expected {
            return Err(NesleError::InvalidState(format!(
                "vector action count must match num_envs*players: got {}, expected {} (num_envs={}, players={})",
                actions.len(),
                expected,
                self.num_envs,
                players,
            )));
        }
        let tx = self.sender()?;
        for (env_id, ports) in actions.chunks_exact(players).enumerate() {
            tx.send(Task::Step {
                env_id,
                masks: masks_from_ports(ports),
            })
            .map_err(|_| worker_gone())?;
        }
        Ok(())
    }

    /// Drain ALL `num_envs` completions and return them env-indexed (sorted).
    pub fn recv(&mut self) -> Result<Vec<VectorStep>> {
        let batch = self.drain_sorted()?;
        Ok(batch
            .into_iter()
            .map(|c| VectorStep {
                env_id: c.env_id,
                outcome: StepOutcome {
                    rewards: c.rewards,
                    terminated: c.terminated,
                    truncated: c.truncated,
                    info: info_of(&c),
                },
                final_observation: c.final_observation,
            })
            .collect())
    }

    pub fn step(&mut self, actions: Vec<u8>) -> Result<Vec<VectorStep>> {
        self.send(actions)?;
        self.recv()
    }

    /// Block for ALL `num_envs` completions, sorted by `env_id` (determinism).
    fn drain_sorted(&self) -> Result<Vec<Completion>> {
        let mut batch = Vec::with_capacity(self.num_envs);
        for _ in 0..self.num_envs {
            batch.push(self.completion_rx.recv().map_err(|_| worker_gone())?);
        }
        batch.sort_unstable_by_key(|c| c.env_id);
        Ok(batch)
    }

    // -- decoupled obs readers (sync; read AFTER a full-batch barrier) --------
    // Sound: after a full-batch drain every worker is idle, so each env Mutex locks uncontended on post-step state.

    pub fn ram_batch_byte_len(&self) -> usize {
        self.num_envs * RAM_SIZE
    }

    pub fn ram_batch_into(&self, out: &mut [u8]) {
        assert_eq!(
            out.len(),
            self.ram_batch_byte_len(),
            "ram batch length mismatch"
        );
        for (chunk, m) in out.chunks_exact_mut(RAM_SIZE).zip(self.envs.iter()) {
            chunk.copy_from_slice(m.lock().expect("env mutex poisoned").env.ram());
        }
    }

    /// Write per-env CPU RAM (the `ram_batch_into` write counterpart): `data` is
    /// `num_envs * RAM_SIZE` bytes, one 2048-byte block per env. Call after a
    /// full-batch barrier (workers idle) so each env Mutex locks uncontended.
    pub fn set_ram_batch(&self, data: &[u8]) -> Result<()> {
        assert_eq!(
            data.len(),
            self.ram_batch_byte_len(),
            "ram batch length mismatch"
        );
        for (chunk, m) in data.chunks_exact(RAM_SIZE).zip(self.envs.iter()) {
            m.lock().expect("env mutex poisoned").env.set_ram(chunk)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn ram_batch(&self) -> Vec<Vec<u8>> {
        self.envs
            .iter()
            .map(|m| m.lock().expect("env mutex poisoned").env.ram().to_vec())
            .collect()
    }

    pub fn vram_batch_byte_len(&self) -> usize {
        let per = self
            .envs
            .first()
            .map(|m| m.lock().expect("env mutex poisoned").env.vram().len())
            .unwrap_or(0);
        self.num_envs * per
    }

    pub fn vram_batch_into(&self, out: &mut [u8]) {
        assert_eq!(
            out.len(),
            self.vram_batch_byte_len(),
            "vram batch length mismatch"
        );
        if self.num_envs == 0 {
            return;
        }
        let per = out.len() / self.num_envs;
        for (chunk, m) in out.chunks_exact_mut(per).zip(self.envs.iter()) {
            chunk.copy_from_slice(m.lock().expect("env mutex poisoned").env.vram());
        }
    }

    pub fn grayscale_batch_byte_len(&self) -> usize {
        self.num_envs * GRAY_FRAME_LEN
    }

    pub fn grayscale_batch_into(&self, out: &mut [u8]) {
        assert_eq!(
            out.len(),
            self.grayscale_batch_byte_len(),
            "grayscale batch length mismatch"
        );
        for (chunk, m) in out.chunks_exact_mut(GRAY_FRAME_LEN).zip(self.envs.iter()) {
            m.lock()
                .expect("env mutex poisoned")
                .env
                .screen_grayscale_into(chunk);
        }
    }

    #[cfg(test)]
    pub fn grayscale_batch(&self) -> Vec<Vec<u8>> {
        self.envs
            .iter()
            .map(|m| {
                let mut buf = vec![0u8; GRAY_FRAME_LEN];
                m.lock()
                    .expect("env mutex poisoned")
                    .env
                    .screen_grayscale_into(&mut buf);
                buf
            })
            .collect()
    }

    pub fn rgb_batch_byte_len(&self) -> usize {
        self.num_envs * RGB_FRAME_LEN
    }

    pub fn rgb_batch_into(&self, out: &mut [u8]) {
        assert_eq!(
            out.len(),
            self.rgb_batch_byte_len(),
            "rgb batch length mismatch"
        );
        for (chunk, m) in out.chunks_exact_mut(RGB_FRAME_LEN).zip(self.envs.iter()) {
            chunk.copy_from_slice(
                &m.lock()
                    .expect("env mutex poisoned")
                    .env
                    .screen_rgb()
                    .pixels,
            );
        }
    }

    #[cfg(test)]
    pub fn rgb_batch(&self) -> Vec<Vec<u8>> {
        self.envs
            .iter()
            .map(|m| {
                m.lock()
                    .expect("env mutex poisoned")
                    .env
                    .screen_rgb()
                    .pixels
                    .clone()
            })
            .collect()
    }

    pub fn observation_batch_byte_len(&self) -> Result<usize> {
        if !self.obs_mode.is_preprocessed() {
            return Err(NesleError::InvalidState(
                "observation_batch requires the preprocessed obs mode".to_string(),
            ));
        }
        Ok(self.num_envs * self.frame_len)
    }

    pub fn observation_batch_into(&self, out: &mut [u8]) -> Result<()> {
        let len = self.observation_batch_byte_len()?;
        assert_eq!(out.len(), len, "observation batch length mismatch");
        for (chunk, m) in out.chunks_exact_mut(self.frame_len).zip(self.envs.iter()) {
            let guard = m.lock().expect("env mutex poisoned");
            let pipe = guard.pipe.as_ref().ok_or_else(|| {
                NesleError::InvalidState("observation pipeline not configured".to_string())
            })?;
            chunk.copy_from_slice(pipe.observation());
        }
        Ok(())
    }

    // -- asynchronous (envpool) ----------------------------------------------

    /// Kick off every env: each resets (with its noop start) and its initial obs
    /// lands in the completion queue, so the first `async_recv` returns a full batch.
    pub fn async_reset(&self) -> Result<()> {
        let tx = self.sender()?;
        for env_id in 0..self.num_envs {
            tx.send(Task::Reset { env_id }).map_err(|_| worker_gone())?;
        }
        Ok(())
    }

    /// Queue actions for the given units (non-blocking). The action stream is flat
    /// unit-major: `actions.len()` must equal `env_ids.len() * players`, and the masks
    /// for `env_ids[i]` are `actions[i*players .. i*players + players]`. The caller
    /// must only send for units it has already `async_recv`d (one in-flight task per
    /// unit). For `players == 1` this is one mask per env_id.
    pub fn async_send(&self, env_ids: &[usize], actions: &[u8]) -> Result<()> {
        let tx = self.sender()?;
        let players = self.players as usize;
        if actions.len() != env_ids.len() * players {
            return Err(NesleError::InvalidState(format!(
                "async action count must match env_ids*players: got {}, expected {} (env_ids={}, players={})",
                actions.len(),
                env_ids.len() * players,
                env_ids.len(),
                players,
            )));
        }
        for (&env_id, ports) in env_ids.iter().zip(actions.chunks_exact(players)) {
            if env_id >= self.num_envs {
                return Err(NesleError::InvalidState(format!(
                    "env_id {env_id} out of range (num_envs={})",
                    self.num_envs
                )));
            }
            tx.send(Task::Step {
                env_id,
                masks: masks_from_ports(ports),
            })
            .map_err(|_| worker_gone())?;
        }
        Ok(())
    }

    /// Block until `batch_size` envs finish, returning their completions in
    /// completion order (each tagged with its `env_id`).
    pub fn async_recv(&self) -> Result<Vec<Completion>> {
        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            batch.push(self.completion_rx.recv().map_err(|_| worker_gone())?);
        }
        Ok(batch)
    }

    fn sender(&self) -> Result<&Sender<Task>> {
        self.action_tx
            .as_ref()
            .ok_or_else(|| NesleError::InvalidState("vector env is shut down".to_string()))
    }
}

impl Drop for NesVectorEnv {
    fn drop(&mut self) {
        // Disconnect the queue so each worker's recv() returns Err and its loop exits, then join.
        self.action_tx = None;
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker_gone() -> NesleError {
    NesleError::InvalidState("vector env worker pool is gone".to_string())
}

/// Pack a unit's `players` port masks into the 4-wide array the worker drives;
/// trailing inactive ports stay 0 (`NesEnv::step` only reads `masks[..players]`).
fn masks_from_ports(ports: &[u8]) -> [u8; 4] {
    let mut masks = [0u8; 4];
    masks[..ports.len()].copy_from_slice(ports);
    masks
}

fn info_of(c: &Completion) -> StepInfo {
    StepInfo {
        frame_number: c.frame_number,
        episode_frame_number: c.episode_frame_number,
        lives: c.lives,
    }
}

fn worker_loop(
    envs: &Arc<Vec<Mutex<WorkerEnv>>>,
    rx: &Receiver<Task>,
    tx: &Sender<Completion>,
    noop_max: usize,
    autoreset: AutoresetMode,
) {
    while let Ok(task) = rx.recv() {
        let completion = process_task(envs, task, noop_max, autoreset);
        if tx.send(completion).is_err() {
            break;
        }
    }
}

fn process_task(
    envs: &[Mutex<WorkerEnv>],
    task: Task,
    noop_max: usize,
    autoreset: AutoresetMode,
) -> Completion {
    match task {
        Task::Reset { env_id } => {
            let mut guard = envs[env_id].lock().expect("env mutex poisoned");
            // Deref once so `pipe` and `env` are disjoint borrows.
            let we = &mut *guard;
            we.needs_reset = false;
            let fresh = fresh_terminated(we.env.players());
            if let Some(pipe) = we.pipe.as_mut() {
                we.env.reset().expect("vector reset");
                let info = pipe
                    .reset_in_place(&mut we.env, noop_max)
                    .expect("vector reset noop");
                completion(env_id, we, [0.0; 4], fresh, false, info)
            } else {
                let info = we.env.reset().expect("vector reset").info;
                completion(env_id, we, [0.0; 4], fresh, false, info)
            }
        }
        Task::Step { env_id, masks } => {
            let mut guard = envs[env_id].lock().expect("env mutex poisoned");
            let we = &mut *guard;
            // NextStep autoreset: a previously-finished env resets on this task.
            if we.needs_reset && autoreset == AutoresetMode::NextStep {
                we.needs_reset = false;
                let fresh = fresh_terminated(we.env.players());
                let info = if let Some(pipe) = we.pipe.as_mut() {
                    we.env.reset().expect("vector autoreset");
                    pipe.reset_in_place(&mut we.env, noop_max)
                        .expect("vector autoreset noop")
                } else {
                    we.env.reset().expect("vector autoreset").info
                };
                return completion(env_id, we, [0.0; 4], fresh, false, info);
            }
            let players = we.env.players() as usize;
            if let Some(pipe) = we.pipe.as_mut() {
                // The preprocessed pipeline is N-aware: drives masks[..players], returns per-port rewards/terminated.
                let step = pipe
                    .step_in_place(&mut we.env, &masks[..players])
                    .expect("vector step");
                let rewards = step.rewards;
                let terminated = step.terminated;
                let done = terminated[..players].iter().all(|&t| t) || step.truncated;
                if done && autoreset == AutoresetMode::SameStep {
                    we.env.reset().expect("vector same-step reset");
                    let info = pipe
                        .reset_in_place(&mut we.env, noop_max)
                        .expect("vector same-step noop");
                    return completion(env_id, we, rewards, terminated, step.truncated, info);
                }
                if done {
                    we.needs_reset = true;
                }
                completion(env_id, we, rewards, terminated, step.truncated, step.info)
            } else {
                // Raw path: step(&masks[..players]) sets only the active ports.
                let outcome = we.env.step(&masks[..players]).expect("vector step");
                let done = outcome.terminated[..players].iter().all(|&t| t) || outcome.truncated;
                if done && autoreset == AutoresetMode::SameStep {
                    we.env.reset().expect("vector same-step reset");
                } else if done {
                    we.needs_reset = true;
                }
                completion(
                    env_id,
                    we,
                    outcome.rewards,
                    outcome.terminated,
                    outcome.truncated,
                    outcome.info,
                )
            }
        }
    }
}

fn completion(
    env_id: usize,
    we: &WorkerEnv,
    rewards: MultiPlayerValues<f32>,
    terminated: MultiPlayerValues<bool>,
    truncated: bool,
    info: StepInfo,
) -> Completion {
    let players = we.env.players() as usize;
    let final_observation = terminated[..players].iter().all(|&t| t) || truncated;
    Completion {
        env_id,
        obs: we
            .pipe
            .as_ref()
            .map(|p| p.observation().to_vec())
            .unwrap_or_default(),
        rewards,
        terminated,
        truncated,
        final_observation,
        frame_number: info.frame_number,
        episode_frame_number: info.episode_frame_number,
        lives: info.lives,
    }
}

/// "Nobody terminated yet" array: active ports (`< players`) are `false`, inactive
/// trailing ports are `true` (the env's resting / reset state).
fn fresh_terminated(players: u8) -> MultiPlayerValues<bool> {
    let mut t = [true; 4];
    for slot in t.iter_mut().take(players as usize) {
        *slot = false;
    }
    t
}

impl NesVectorEnv {
    pub fn ram_batch_bytes(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.ram_batch_byte_len()];
        self.ram_batch_into(&mut out);
        out
    }

    pub fn grayscale_batch_bytes(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.grayscale_batch_byte_len()];
        self.grayscale_batch_into(&mut out);
        out
    }

    pub fn rgb_batch_bytes(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.rgb_batch_byte_len()];
        self.rgb_batch_into(&mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::{GameSpec, Ram};
    use crate::preprocess::RenderPolicy;
    use crate::test_support::load_test_rom;

    fn zero_reward(_p: &Ram, _c: &Ram) -> MultiPlayerValues<f32> {
        [0.0; 4]
    }
    fn never_terminal(_p: &Ram, _c: &Ram) -> bool {
        false
    }
    fn zero_lives(_c: &Ram) -> MultiPlayerValues<u8> {
        [0; 4]
    }

    static TEST_GAME: GameSpec = GameSpec {
        id: "super_mario_bros",
        family: "super_mario_bros",
        gym_id: "NESLE/TestVector-v0",
        display_name: "Test Vector",
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

    fn raw(num_envs: usize) -> NesVectorEnv {
        NesVectorEnv::new(&TEST_GAME, VectorConfig::new(num_envs), &load_test_rom()).unwrap()
    }

    #[test]
    fn vector_send_recv_steps_each_env() {
        let mut vector = raw(3);
        let reset = vector.reset().unwrap();
        assert_eq!(reset.len(), 3);
        vector.send(vec![0, 1, 2]).unwrap();
        let steps = vector.recv().unwrap();
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[2].env_id, 2);
        assert_eq!(steps[2].outcome.info.episode_frame_number, 1);
    }

    #[test]
    fn parallel_recv_matches_independent_single_envs() {
        // Parallel stepping must match N independent single envs (same seeds + actions).
        let n = 4usize;
        let mut config = VectorConfig::new(n);
        config.frame_skip = 2;
        let (frame_skip, seed) = (config.frame_skip, config.seed);
        let mut vector = NesVectorEnv::new(&TEST_GAME, config, &load_test_rom()).unwrap();
        vector.reset().unwrap();

        let mut singles: Vec<NesEnv> = (0..n)
            .map(|i| {
                let mut e = NesEnv::new(&TEST_GAME);
                e.load_rom_bytes(&load_test_rom()).unwrap();
                e.set_action_repeat(frame_skip, 0.0).unwrap();
                e.seed(seed.wrapping_add(i as u64).max(1));
                e.reset().unwrap();
                e
            })
            .collect();

        for frame in 0..20u8 {
            let actions: Vec<u8> = (0..n).map(|i| frame.wrapping_add(i as u8) & 0x83).collect();
            let steps = vector.step(actions.clone()).unwrap();
            for (i, single) in singles.iter_mut().enumerate() {
                let s = single.step(&[actions[i]]).unwrap();
                assert_eq!(steps[i].env_id, i);
                assert_eq!(steps[i].outcome, s, "env {i} frame {frame} diverged");
            }
        }
    }

    #[test]
    fn vector_rejects_wrong_action_count() {
        let mut vector = raw(2);
        let err = vector.send(vec![0]).unwrap_err().to_string();
        assert!(err.contains("vector action count"));
    }

    #[test]
    fn next_step_autoreset_returns_reset_outcome_on_following_step() {
        let mut config = VectorConfig::new(1);
        config.max_episode_frames = Some(1);
        config.autoreset_mode = AutoresetMode::NextStep;
        let mut vector = NesVectorEnv::new(&TEST_GAME, config, &load_test_rom()).unwrap();
        vector.reset().unwrap();

        let done = vector.step(vec![0]).unwrap();
        assert!(done[0].outcome.truncated);
        assert!(done[0].final_observation);
        assert_eq!(done[0].outcome.info.episode_frame_number, 1);

        let reset = vector.step(vec![0]).unwrap();
        assert!(!reset[0].outcome.truncated);
        assert!(!reset[0].final_observation);
        assert_eq!(reset[0].outcome.info.episode_frame_number, 0);
    }

    #[test]
    fn same_step_autoreset_resets_immediately_after_final_step() {
        let mut config = VectorConfig::new(1);
        config.max_episode_frames = Some(1);
        config.autoreset_mode = AutoresetMode::SameStep;
        let mut vector = NesVectorEnv::new(&TEST_GAME, config, &load_test_rom()).unwrap();
        vector.reset().unwrap();

        let done = vector.step(vec![0]).unwrap();
        assert!(done[0].outcome.truncated);
        assert!(done[0].final_observation);
        assert_eq!(done[0].outcome.info.episode_frame_number, 1);

        let again = vector.step(vec![0]).unwrap();
        assert!(again[0].outcome.truncated);
        assert_eq!(again[0].outcome.info.episode_frame_number, 1);
    }

    #[test]
    fn vector_config_applies_frame_skip_to_each_env() {
        let mut config = VectorConfig::new(2);
        config.frame_skip = 3;
        let mut vector = NesVectorEnv::new(&TEST_GAME, config, &load_test_rom()).unwrap();
        vector.reset().unwrap();
        let steps = vector.step(vec![0, 0]).unwrap();
        assert_eq!(steps[0].outcome.info.episode_frame_number, 3);
        assert_eq!(steps[1].outcome.info.episode_frame_number, 3);
    }

    #[test]
    fn contiguous_batches_match_legacy_batch_order() {
        let mut vector = raw(2);
        vector.reset().unwrap();
        vector.step(vec![0x80, 0x00]).unwrap();

        assert_eq!(vector.ram_batch_bytes(), vector.ram_batch().concat());
        assert_eq!(
            vector.grayscale_batch_bytes(),
            vector.grayscale_batch().concat()
        );
        assert_eq!(vector.rgb_batch_bytes(), vector.rgb_batch().concat());
    }

    #[test]
    fn async_reset_then_send_recv_round_trips_all_envs() {
        let cfg = ObsConfig::gray(4, 84, false, RenderPolicy::TrainingSparse, false);
        let mut config = VectorConfig::new(6);
        config.batch_size = 3;
        config.num_threads = 2;
        config.obs_mode = VectorObsMode::Preprocessed;
        config.obs_cfg = Some(cfg);
        config.repeat_action_probability = 0.25;
        config.noop_max = 30;
        let pool = NesVectorEnv::new(&TEST_GAME, config, &load_test_rom()).unwrap();
        assert_eq!(pool.frame_len(), 84 * 84);

        pool.async_reset().unwrap();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..20 {
            let batch = pool.async_recv().unwrap();
            assert_eq!(batch.len(), 3);
            let mut ids = Vec::new();
            let mut acts = Vec::new();
            for c in &batch {
                assert!(c.env_id < 6);
                assert_eq!(c.obs.len(), 84 * 84);
                seen.insert(c.env_id);
                ids.push(c.env_id);
                acts.push(0x80u8);
            }
            pool.async_send(&ids, &acts).unwrap();
        }
        assert_eq!(seen.len(), 6, "every env should appear across the rounds");
    }

    #[test]
    fn flat_slots_multi_unit_matches_independent_multi_envs() {
        // 3 units x 2 ports: per-unit per-port outcomes must equal 3 independent 2-player NesEnvs (Super C, ROM-gated).
        use crate::games::registry;
        use crate::test_support::rom_path;

        let Some(path) = rom_path("Super C (USA).nes") else {
            return; // ROM absent -> skip (like other ROM-gated tests).
        };
        let rom = std::fs::read(path).unwrap();
        let game = registry::super_c_2p();

        let num_envs = 3usize;
        let players = 2u8;
        let mut config = VectorConfig::new(num_envs);
        config.players = players;
        config.obs_mode = VectorObsMode::Grayscale;
        config.frame_skip = 2;
        let (frame_skip, seed) = (config.frame_skip, config.seed);
        let mut vector = NesVectorEnv::new(game, config, &rom).unwrap();
        assert_eq!(vector.players(), players);
        vector.reset().unwrap();

        let mut singles: Vec<NesEnv> = (0..num_envs)
            .map(|i| {
                let mut e = NesEnv::new(game);
                e.set_players(players).unwrap();
                e.load_rom_bytes(&rom).unwrap();
                e.set_action_repeat(frame_skip, 0.0).unwrap();
                e.seed(seed.wrapping_add(i as u64).max(1));
                e.reset().unwrap();
                e
            })
            .collect();

        let n = players as usize;
        for frame in 0..12u8 {
            // Flat unit-major action stream: unit u, port p -> actions[u*players + p].
            let actions: Vec<u8> = (0..num_envs)
                .flat_map(|u| {
                    (0..n).map(move |p| {
                        let act = frame.wrapping_add((u * n + p) as u8);
                        // Stay within Super C's minimal action set indices via mask bits.
                        act & 0x83
                    })
                })
                .collect();
            vector.send(actions.clone()).unwrap();
            // recv() returns env-sorted VectorSteps with full per-port reward/terminated/lives arrays.
            let steps = vector.recv().unwrap();
            assert_eq!(steps.len(), num_envs);
            for (u, single) in singles.iter_mut().enumerate() {
                let masks = &actions[u * n..u * n + n];
                let s = single.step(masks).unwrap();
                assert_eq!(steps[u].env_id, u);
                assert_eq!(
                    steps[u].outcome.rewards, s.rewards,
                    "unit {u} frame {frame} rewards diverged"
                );
                assert_eq!(
                    steps[u].outcome.terminated, s.terminated,
                    "unit {u} frame {frame} terminated diverged"
                );
                assert_eq!(
                    steps[u].outcome.info.lives, s.info.lives,
                    "unit {u} frame {frame} lives diverged"
                );
            }
        }
    }
}
