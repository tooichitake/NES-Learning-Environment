//! Authoritative console session. One thread owns one emulator env; websocket
//! tasks communicate with it through channels. Humans and agents share the same
//! kernel, screen, rewards, and controller ports.

use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde::Serialize;
use tokio::sync::broadcast;

use nesle_rl::games::registry::{self, find_game};
use nesle_rl::preprocess::{FrameSample, ObsConfig, ObsKind, ObsWindow, RenderPolicy, RewardClip};
use nesle_rl::{NesEnv, NesInterface};

use crate::wire::{ErrorMsg, PeerInfo, PortOwner, PortsMsg, RamMsg, RecordingMsg};

use sha1::{Digest, Sha1};

/// Directory holding the packaged ROMs (`nesle/roms/`, the single ROM home shared
/// with the Python package). Override with `NESLE_ROMS_DIR`; the dev default is
/// resolved relative to this crate.
fn packaged_roms_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("NESLE_ROMS_DIR") {
        return std::path::PathBuf::from(dir);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../nesle-py/python/nesle/roms")
}

/// Find the packaged `*.nes` whose sha1 matches `sha1_hex` and return its bytes
/// (the server-side analog of Python `nesle.roms.get_rom_path`). `None` if absent.
fn resolve_packaged_rom(sha1_hex: &str) -> Option<Vec<u8>> {
    if sha1_hex.is_empty() {
        return None;
    }
    for entry in std::fs::read_dir(packaged_roms_dir()).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("nes") {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let hex: String = Sha1::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if hex == sha1_hex {
            return Some(bytes);
        }
    }
    None
}

const NES_W: u32 = 256;
const NES_H: u32 = 240;
const PORTS: usize = 4;
const REC_MAX_FRAMES: usize = 36_000;

#[derive(Clone)]
pub enum OutMsg {
    Text(String),
    Human(Arc<Vec<u8>>),
    Agent { client: u64, bytes: Arc<Vec<u8>> },
}

pub enum Command {
    Hello {
        client: u64,
        role: String,
        name: Option<String>,
        env_id: Option<String>,
    },
    Rename {
        client: u64,
        name: String,
    },
    LoadRom {
        env_id: String,
        bytes: Vec<u8>,
    },
    Action {
        client: u64,
        port: Option<usize>,
        mask: u8,
    },
    AssignPort {
        client: u64,
        target_client: Option<u64>,
        name: Option<String>,
        port: usize,
    },
    Disconnect {
        client: u64,
    },
    SetRunning(bool),
    Reset,
    SetObsSize(usize),
    SetRlMode(bool),
    SetStepMode(bool),
    Step(Vec<u8>),
    SetFrameSkip(usize),
    SetMaxpool(bool),
    SetRemoveSpriteLimit(bool),
    SetObsRgb(bool),
    SetTerminalOnLifeLoss(bool),
    SetStickyProb(f32),
    SetNoopMax(usize),
    SetClipPos(f32),
    SetClipNeg(f32),
    Record {
        on: bool,
    },
    DumpRam,
}

/// Agent-declared observation profile echoed in state meta.
#[derive(Clone, Serialize)]
struct EnvProfile {
    env_id: String,
    frame_skip: usize,
    obs_size: usize,
    maxpool: bool,
    remove_sprite_limit: bool,
    grayscale: bool,
    terminal_on_life_loss: bool,
    repeat_action_probability: f32,
    noop_max: usize,
    clip_reward: bool,
}

impl EnvProfile {
    fn standard(env_id: &str) -> Self {
        Self {
            env_id: env_id.to_string(),
            frame_skip: 4,
            obs_size: 112,
            maxpool: true,
            remove_sprite_limit: false,
            grayscale: true,
            terminal_on_life_loss: true,
            repeat_action_probability: 0.0,
            noop_max: 0,
            clip_reward: false,
        }
    }

    fn raw(env_id: &str) -> Self {
        Self {
            env_id: env_id.to_string(),
            frame_skip: 1,
            obs_size: 0,
            maxpool: false,
            remove_sprite_limit: false,
            grayscale: false,
            terminal_on_life_loss: false,
            repeat_action_probability: 0.0,
            noop_max: 0,
            clip_reward: false,
        }
    }

    fn noflicker(env_id: &str, frame_skip: usize) -> Self {
        Self {
            env_id: env_id.to_string(),
            frame_skip,
            maxpool: false,
            remove_sprite_limit: true,
            ..Self::standard(env_id)
        }
    }
}

struct EnvSelection {
    game_id: String,
    start_state: String,
    profile: Option<EnvProfile>,
    /// Effective controller-port count: the spec's native `players`.
    players: u8,
}

/// Match `env_id` against a single-agent gym family rooted at `gym_stem`
/// (`{stem}-v0` title + `{stem}-{level}-v{0..3}` + NoFrameskip). Returns the
/// backing start-state id + obs profile. Shared by genuine single-player specs and
/// the `1P` solo mode of a 2P spec.
fn match_single_agent_family(
    game_id: &str,
    gym_stem: &str,
    env_id: &str,
) -> Option<(String, EnvProfile)> {
    if env_id == format!("{gym_stem}-v0") {
        return Some(("title".to_string(), EnvProfile::raw(env_id)));
    }
    for state_id in nesle_rl::available_start_state_ids(game_id) {
        let Some(level) = nesle_rl::env_suffix_for_start_state(game_id, &state_id) else {
            continue;
        };
        let stem = format!("{gym_stem}-{level}");
        let profile = if env_id == format!("{stem}-v0") {
            EnvProfile::raw(env_id)
        } else if env_id == format!("{stem}-v1") {
            EnvProfile::standard(env_id)
        } else if env_id == format!("{stem}-v2") {
            EnvProfile::noflicker(env_id, 4)
        } else if env_id == format!("{stem}NoFrameskip-v1") {
            EnvProfile {
                frame_skip: 1,
                ..EnvProfile::standard(env_id)
            }
        } else if env_id == format!("{stem}NoFrameskip-v2") {
            EnvProfile::noflicker(env_id, 1)
        } else if env_id == format!("{stem}-v3") {
            // v3 = v2 (noflicker, frame_skip 4) + sticky actions (Machado 2018).
            EnvProfile {
                repeat_action_probability: 0.25,
                ..EnvProfile::noflicker(env_id, 4)
            }
        } else if env_id == format!("{stem}NoFrameskip-v3") {
            EnvProfile {
                repeat_action_probability: 0.25,
                ..EnvProfile::noflicker(env_id, 1)
            }
        } else {
            continue;
        };
        return Some((state_id, profile));
    }
    None
}

fn resolve_env_id(env_id: &str) -> Option<EnvSelection> {
    // A `players == 1` spec carries an RL profile (v0-v3); a multi-player spec has `profile: None` + a `-v0` gym_id.
    for game in registry::all_games() {
        if game.players == 1 {
            if let Some((start_state, profile)) =
                match_single_agent_family(game.id, game.gym_id, env_id)
            {
                return Some(EnvSelection {
                    game_id: game.id.to_string(),
                    start_state,
                    profile: Some(profile),
                    players: 1,
                });
            }
        } else {
            if env_id == game.gym_id {
                return Some(EnvSelection {
                    game_id: game.id.to_string(),
                    start_state: "title".to_string(),
                    profile: None,
                    players: game.players,
                });
            }
            if let Some(base) = game.gym_id.strip_suffix("-v0") {
                for state_id in nesle_rl::available_start_state_ids(game.id) {
                    let Some(level) = nesle_rl::env_suffix_for_start_state(game.id, &state_id)
                    else {
                        continue;
                    };
                    if env_id == format!("{base}-{level}-v0") {
                        return Some(EnvSelection {
                            game_id: game.id.to_string(),
                            start_state: state_id,
                            profile: None,
                            players: game.players,
                        });
                    }
                }
            }
        }
    }
    None
}

struct AgentObs {
    env: EnvProfile,
    window: ObsWindow,
    action_names: Vec<&'static str>,
    action_masks: Vec<u8>,
    step: u64,
    ret: f32,
    last_step_reward: f32,
}

impl AgentObs {
    fn new(env: EnvProfile, action_names: Vec<&'static str>, action_masks: Vec<u8>) -> Self {
        let window = ObsWindow::new(obs_config_from_profile(&env));
        Self {
            env,
            window,
            action_names,
            action_masks,
            step: 0,
            ret: 0.0,
            last_step_reward: 0.0,
        }
    }
}

fn obs_config_from_profile(desc: &EnvProfile) -> ObsConfig {
    let obs_kind = if desc.grayscale {
        ObsKind::gray(desc.obs_size.max(16), desc.maxpool)
    } else if desc.obs_size >= 16 {
        ObsKind::rgb(desc.obs_size)
    } else {
        ObsKind::RgbNative
    };
    ObsConfig {
        frame_skip: desc.frame_skip.max(1),
        obs_kind,
        render_policy: RenderPolicy::HumanVisible,
        terminal_on_life_loss: false,
        repeat_action_probability: desc.repeat_action_probability,
        noop_max: desc.noop_max,
        reward_clip: if desc.clip_reward {
            RewardClip::symmetric(1.0)
        } else {
            RewardClip::none()
        },
        // Server serves real-time single frames, not training stacks.
        stack_num: 1,
        players: 1,
    }
}

#[derive(Serialize, Clone)]
struct AgentView {
    id: u64,
    label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    port: Option<u8>,
    obs_w: u32,
    obs_h: u32,
    obs_channels: u8,
    reward: f32,
    ret: f32,
    lives: u8,
    mask: u8,
    step: u64,
    obs_step: bool,
    assigned: bool,
    action_names: Vec<&'static str>,
    action_masks: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    env: Option<EnvProfile>,
}

struct StepData {
    rewards: [f32; 4],
    lives: [u8; 4],
    frame: u64,
    ep_frame: u64,
    terminated: bool,
    truncated: bool,
}

#[derive(Clone)]
struct ClientInfo {
    role: String,
    name: String,
    env_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ControllerOwner {
    client: u64,
    name: String,
}

fn clean_name(name: Option<String>) -> Option<String> {
    name.map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(|value| value.chars().take(48).collect())
}

fn default_peer_name(client: u64, role: &str) -> String {
    if role == "agent" {
        format!("Agent {client}")
    } else {
        format!("Player {client}")
    }
}

#[derive(Serialize, Clone)]
struct StateMeta {
    frame: u64,
    ep_frame: u64,
    step: u64,
    rewards: [f32; 4],
    rets: [f32; 4],
    lives: [u8; 4],
    players: u8,
    owners: [Option<u64>; 4],
    terminated: bool,
    truncated: bool,
    native_w: u32,
    native_h: u32,
    obs_w: u32,
    obs_h: u32,
    obs_channels: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    env: Option<EnvProfile>,
    obs_step: bool,
    frame_skip: u32,
    audio_len: u32,
    audio_rate: u32,
    recording: bool,
    running: bool,
    masks: [u8; 4],
    // Same order as the agent obs blocks appended to the binary frame.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    agents: Vec<AgentView>,
}

#[derive(Serialize)]
struct ReadyMsg {
    t: &'static str,
    env_id: String,
    game: &'static str,
    display_name: &'static str,
    actions: Vec<&'static str>,
    action_masks: Vec<u8>,
    native_w: u32,
    native_h: u32,
    players: u8,
}

struct Console {
    machine: Option<NesEnv>,
    env_id: String,
    game_id: &'static str,
    display_name: &'static str,
    action_names: Vec<&'static str>,
    action_masks: Vec<u8>,
    players: u8,
    start_state: String,
    running: bool,
    masks: [u8; 4],
    owners: [Option<ControllerOwner>; 4],
    obs_size: usize,
    maxpool: bool,
    rets: [f32; 4],
    /// Reward accrued since the last broadcast; flushed into `rets` + the shown reward together.
    pending_rewards: [f32; 4],
    rl_mode: bool,
    step_mode: bool,
    /// Step mode only: a terminal frame was reached and is held on screen; the
    /// next Step (or Reset) starts a new episode instead of stepping, so the
    /// death reward/lives stay observable.
    pending_reset: bool,
    frame_skip: usize,
    obs_window: ObsWindow,
    remove_sprite_limit: bool,
    obs_rgb: bool,
    terminal_on_life_loss: bool,
    sticky_prob: f32,
    noop_max: usize,
    clip_pos: f32,
    clip_neg: f32,
    prev_lives: [u8; 4],
    last_ep_frame: u64,
    rl_step: u64,
    prev_applied: [u8; 4],
    rng: u32,
    recording: bool,
    rec: Vec<[u8; 4]>,
    clients: HashMap<u64, ClientInfo>,
    agents: HashMap<u64, AgentObs>,
    state_tx: broadcast::Sender<OutMsg>,
    ready: Arc<Mutex<Option<String>>>,
}

impl Console {
    fn new(state_tx: broadcast::Sender<OutMsg>, ready: Arc<Mutex<Option<String>>>) -> Self {
        Self {
            machine: None,
            env_id: String::new(),
            game_id: "",
            display_name: "",
            action_names: Vec::new(),
            action_masks: Vec::new(),
            players: 1,
            start_state: String::new(),
            running: false,
            masks: [0; 4],
            owners: std::array::from_fn(|_| None),
            obs_size: 112,
            maxpool: true,
            rets: [0.0; 4],
            pending_rewards: [0.0; 4],
            rl_mode: false,
            step_mode: false,
            pending_reset: false,
            frame_skip: 4,
            obs_window: ObsWindow::new(ObsConfig::gray(
                1,
                84,
                true,
                RenderPolicy::HumanVisible,
                false,
            )),
            remove_sprite_limit: false,
            obs_rgb: false,
            terminal_on_life_loss: true,
            sticky_prob: 0.0,
            noop_max: 0,
            clip_pos: 0.0,
            clip_neg: 0.0,
            prev_lives: [0; 4],
            last_ep_frame: 0,
            rl_step: 0,
            prev_applied: [0; 4],
            rng: 0x9E37_79B9,
            recording: false,
            rec: Vec::new(),
            clients: HashMap::new(),
            agents: HashMap::new(),
            state_tx,
            ready,
        }
    }

    fn apply(&mut self, cmd: Command) {
        match cmd {
            Command::Hello {
                client,
                role,
                name,
                env_id,
            } => self.hello(client, role, name, env_id),
            Command::Rename { client, name } => self.rename(client, name),
            Command::LoadRom { env_id, bytes } => self.load_rom(&env_id, &bytes),
            Command::Action { client, port, mask } => {
                if let Some(port) = self.action_port(client, port) {
                    self.masks[port] = mask;
                }
            }
            Command::AssignPort {
                client,
                target_client,
                name,
                port,
            } => {
                if port < PORTS && !self.is_agent(client) {
                    if let Some(target_client) = target_client {
                        if self.clients.contains_key(&target_client) {
                            let Some(name) = clean_name(name) else {
                                self.emit_error(
                                    "missing_player_name",
                                    "assign_port requires a non-empty player name",
                                );
                                return;
                            };
                            if self.player_name_in_use(
                                &name,
                                port,
                                self.is_agent(target_client).then_some(target_client),
                            ) {
                                self.emit_error(
                                    "duplicate_name",
                                    format!("player name already in use: {name}"),
                                );
                                return;
                            }
                            if self.is_agent(target_client) {
                                for owner in &mut self.owners {
                                    if owner.as_ref().map(|owner| owner.client)
                                        == Some(target_client)
                                    {
                                        *owner = None;
                                    }
                                }
                            }
                            self.owners[port] = Some(ControllerOwner {
                                client: target_client,
                                name,
                            });
                        }
                    } else {
                        self.owners[port] = None;
                    }
                    self.masks[port] = 0;
                    self.broadcast_ports();
                }
            }
            Command::Disconnect { client } => {
                self.clients.remove(&client);
                self.agents.remove(&client);
                let mut changed = false;
                for p in 0..PORTS {
                    if self.owners[p].as_ref().map(|owner| owner.client) == Some(client) {
                        self.owners[p] = None;
                        self.masks[p] = 0;
                        changed = true;
                    }
                }
                if changed {
                    self.broadcast_ports();
                }
            }
            Command::SetRunning(r) => self.running = r,
            Command::Reset => self.do_reset(),
            Command::SetObsSize(s) => {
                if (16..=256).contains(&s) {
                    self.obs_size = s;
                    self.reset_obs_windows();
                    self.maybe_refresh_obs();
                }
            }
            Command::SetRlMode(r) => {
                let changed = self.rl_mode != r;
                self.rl_mode = r;
                if let Some(e) = self.machine.as_mut() {
                    e.set_skip_transitions(r);
                }
                self.reset_obs_windows();
                if changed && self.machine.is_some() {
                    self.do_reset();
                }
            }
            Command::SetStepMode(s) => {
                self.step_mode = s;
                self.obs_window.restart_window();
            }
            Command::Step(masks) => {
                if self.machine.is_none() {
                    return;
                }
                if self.pending_reset {
                    // Previous step ended the episode; this press starts a fresh one.
                    self.reset_episode();
                    return;
                }
                // One mask per port (masks[i] -> port i); trailing ports default to NOOP.
                let mut m = [0u8; PORTS];
                for (i, &v) in masks.iter().take(PORTS).enumerate() {
                    m[i] = v;
                }
                self.masks = m;
                self.obs_window.restart_window();
                let cap = self.frame_skip.max(1) * 3 + 2;
                for _ in 0..cap {
                    if self.tick() {
                        break;
                    }
                }
            }
            Command::SetFrameSkip(f) => {
                if (1..=16).contains(&f) {
                    self.frame_skip = f;
                    self.reset_obs_windows();
                }
            }
            Command::SetMaxpool(m) => {
                self.maxpool = m;
                self.reset_obs_windows();
                self.maybe_refresh_obs();
            }
            Command::SetRemoveSpriteLimit(b) => {
                self.remove_sprite_limit = b;
                self.apply_sprite_limit();
                self.maybe_refresh_obs();
            }
            Command::SetObsRgb(b) => {
                self.obs_rgb = b;
                self.reset_obs_windows();
                self.maybe_refresh_obs();
            }
            Command::SetTerminalOnLifeLoss(b) => {
                self.terminal_on_life_loss = b;
                self.reset_obs_windows();
            }
            Command::SetStickyProb(p) => self.sticky_prob = p.clamp(0.0, 1.0),
            Command::SetNoopMax(n) => self.noop_max = n.min(60),
            Command::SetClipPos(p) => self.clip_pos = p.max(0.0),
            Command::SetClipNeg(p) => self.clip_neg = p.max(0.0),
            Command::Record { on } => {
                if on {
                    self.reset_episode();
                    self.rec.clear();
                    self.recording = true;
                } else if self.recording {
                    self.emit_recording();
                    self.recording = false;
                }
            }
            Command::DumpRam => self.emit_ram(),
        }
    }

    fn apply_sprite_limit(&mut self) {
        if let Some(e) = self.machine.as_mut() {
            e.set_remove_sprite_limit(self.remove_sprite_limit);
        }
    }

    fn maybe_refresh_obs(&mut self) {
        if self.step_mode && self.machine.is_some() {
            self.broadcast_obs_refresh();
        }
    }

    fn obs_config(&self) -> ObsConfig {
        let obs_kind = if self.obs_rgb {
            ObsKind::rgb(self.obs_size)
        } else {
            ObsKind::gray(self.obs_size, self.maxpool)
        };
        ObsConfig {
            frame_skip: if self.rl_mode {
                self.frame_skip.max(1)
            } else {
                1
            },
            obs_kind,
            render_policy: RenderPolicy::HumanVisible,
            terminal_on_life_loss: self.rl_mode && self.terminal_on_life_loss,
            repeat_action_probability: self.sticky_prob,
            noop_max: self.noop_max,
            reward_clip: RewardClip::none(),
            stack_num: 1,
            players: self.players,
        }
    }

    fn reset_obs_windows(&mut self) {
        self.obs_window = ObsWindow::new(self.obs_config());
        self.obs_window.reset(self.prev_lives);
        for agent in self.agents.values_mut() {
            agent.window = ObsWindow::new(obs_config_from_profile(&agent.env));
            agent.window.reset(self.prev_lives);
        }
    }

    /// Refresh paused Debug observations without advancing the emulator.
    fn broadcast_obs_refresh(&mut self) {
        let (rgb, gray) = match self.machine.as_mut() {
            Some(e) => (e.screen_rgb().pixels.clone(), e.screen_grayscale().pixels),
            None => return,
        };
        let sample = FrameSample {
            rgb: Some(&rgb),
            gray: Some(&gray),
            ram: None,
            rewards: [0.0; 4],
            lives: self.prev_lives,
            frame_number: self.last_ep_frame,
            episode_frame_number: self.last_ep_frame,
            terminated: false,
            truncated: false,
        };
        if self.obs_window.refresh(sample).is_err() {
            return;
        }
        let shape = self.obs_window.shape();
        let meta = StateMeta {
            frame: self.last_ep_frame,
            ep_frame: self.last_ep_frame,
            step: self.rl_step,
            rewards: [0.0; 4],
            rets: self.rets,
            lives: self.prev_lives,
            players: self.players,
            owners: self.owner_ids(),
            terminated: false,
            truncated: false,
            native_w: NES_W,
            native_h: NES_H,
            obs_w: shape.width as u32,
            obs_h: shape.height as u32,
            obs_channels: shape.channels,
            env: None,
            obs_step: true,
            frame_skip: if self.rl_mode {
                self.frame_skip.max(1) as u32
            } else {
                1
            },
            audio_len: 0,
            audio_rate: NesInterface::audio_sample_rate(),
            recording: self.recording,
            running: self.running,
            masks: self.masks,
            agents: Vec::new(),
        };
        let full = encode_state(&meta, &rgb, self.obs_window.observation(), &[], &[]);
        let _ = self.state_tx.send(OutMsg::Human(Arc::new(full)));
    }

    fn declare_env(&mut self, client: u64, env_id: &str) {
        let Some(selection) = resolve_env_id(env_id) else {
            return;
        };
        let Some(desc) = selection.profile else {
            return;
        };
        let Some(game) = find_game(&selection.game_id) else {
            return;
        };
        let action_names: Vec<&'static str> = game
            .minimal_actions
            .iter()
            .map(|action| action.name)
            .collect();
        let action_masks: Vec<u8> = game
            .minimal_actions
            .iter()
            .map(|action| action.mask)
            .collect();
        let ao = self.agents.entry(client).or_insert_with(|| {
            AgentObs::new(desc.clone(), action_names.clone(), action_masks.clone())
        });
        ao.window = ObsWindow::new(obs_config_from_profile(&desc));
        ao.window.reset(self.prev_lives);
        ao.env = desc.clone();
        ao.action_names = action_names;
        ao.action_masks = action_masks;
    }

    fn next_rng(&mut self) -> u32 {
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        self.rng
    }

    fn load_rom(&mut self, env_id: &str, bytes: &[u8]) {
        let Some(selection) = resolve_env_id(env_id) else {
            return;
        };
        if let Some(spec) = find_game(&selection.game_id) {
            // Empty bytes -> resolve the packaged ROM by sha1; a non-empty upload overrides.
            let rom: Vec<u8> = if !bytes.is_empty() {
                bytes.to_vec()
            } else if let Some(found) = resolve_packaged_rom(spec.sha1) {
                found
            } else {
                self.emit_error(
                    "rom_required",
                    format!("no packaged ROM for {env_id} -- use Upload ROM"),
                );
                return;
            };
            self.env_id = env_id.to_string();
            self.start_state = selection.start_state;
            // One env drives 1..=spec.players ports; a 2P spec's `1P` mode narrows via set_players.
            let mut e = NesEnv::new(spec);
            if selection.players != spec.players {
                let _ = e.set_players(selection.players);
            }
            if e.load_rom_bytes(&rom).is_err() {
                self.emit_error("rom_required", format!("failed to load ROM for {env_id}"));
                return;
            }
            let _ = e.set_action_repeat(1, 0.0);
            // Skip cutscenes only in Debug; Play / Agent stay faithful (no-op for multi-player).
            e.set_skip_transitions(self.rl_mode);
            e.set_audio_enabled(true);
            // Capture restored start-state lives/frame so the load broadcast reports real values, not zeros.
            let start_info = if self.start_state == "title" {
                e.reset_game().info
            } else {
                match e
                    .set_start_state_id(&self.start_state)
                    .and_then(|_| e.reset_to_start_state())
                {
                    Ok(outcome) => outcome.info,
                    Err(_) => return,
                }
            };
            let _ = e.drain_audio_samples();
            let players = selection.players;
            self.machine = Some(e);
            self.prev_lives = start_info.lives;
            self.last_ep_frame = start_info.episode_frame_number;
            self.start_loaded(
                spec.id,
                spec.display_name,
                spec.minimal_actions.iter().map(|a| a.name).collect(),
                spec.minimal_actions.iter().map(|a| a.mask).collect(),
                players,
            );
        }
    }

    fn start_loaded(
        &mut self,
        id: &'static str,
        display_name: &'static str,
        actions: Vec<&'static str>,
        action_masks: Vec<u8>,
        players: u8,
    ) {
        self.game_id = id;
        self.display_name = display_name;
        self.action_names = actions;
        self.action_masks = action_masks;
        self.players = players;
        // End-on-life-loss default mirrors the env: ALE EpisodicLife for single-agent, off for multi-player.
        self.terminal_on_life_loss = players == 1;
        self.running = true;
        self.masks = [0; 4];
        self.owners = std::array::from_fn(|_| None);
        self.rets = [0.0; 4];
        self.pending_rewards = [0.0; 4];
        self.recording = false;
        self.rec.clear();
        // prev_lives / last_ep_frame are set by load_rom from the restored start state before this runs.
        self.reset_obs_windows();
        self.prev_applied = [0; 4];
        self.apply_sprite_limit();
        self.emit_ready();
        self.broadcast_ports();
        // Broadcast the restored start-state frame WITHOUT advancing, keeping it frame-exact.
        self.broadcast_obs_refresh();
    }

    fn emit_ready(&mut self) {
        if self.game_id.is_empty() {
            return;
        }
        let ready = ReadyMsg {
            t: "ready",
            env_id: self.env_id.clone(),
            game: self.game_id,
            display_name: self.display_name,
            actions: self.action_names.clone(),
            action_masks: self.action_masks.clone(),
            native_w: NES_W,
            native_h: NES_H,
            players: self.players,
        };
        if let Ok(json) = serde_json::to_string(&ready) {
            *self.ready.lock().unwrap() = Some(json.clone());
            let _ = self.state_tx.send(OutMsg::Text(json));
        }
    }

    fn do_reset(&mut self) {
        self.reset_episode();
    }

    fn clear_episode_bookkeeping(&mut self, lives: [u8; 4], ep_frame: u64) {
        self.pending_reset = false;
        self.rets = [0.0; 4];
        self.pending_rewards = [0.0; 4];
        self.prev_lives = lives;
        self.last_ep_frame = ep_frame;
        self.rl_step = 0;
        self.masks = [0; 4];
        self.prev_applied = [0; 4];
        self.reset_obs_windows();
        for agent in self.agents.values_mut() {
            agent.step = 0;
            agent.ret = 0.0;
            agent.last_step_reward = 0.0;
        }
    }

    fn reset_episode(&mut self) {
        let noops = if self.noop_max > 0 {
            (self.next_rng() as usize) % (self.noop_max + 1)
        } else {
            0
        };
        let mut reset_lives;
        let reset_ep_frame;
        match self.machine.as_mut() {
            Some(e) => {
                let outcome = if self.start_state == "title" {
                    e.reset_game()
                } else {
                    let Ok(outcome) = e
                        .set_start_state_id(&self.start_state)
                        .and_then(|_| e.reset_to_start_state())
                    else {
                        return;
                    };
                    outcome
                };
                reset_lives = outcome.info.lives;
                let mut ep_frame = outcome.info.episode_frame_number;
                for _ in 0..noops {
                    // The env slices masks to its active ports, so a 4-wide NOOP fits 1..=4 players.
                    if let Ok(outcome) = e.step(&[0; 4]) {
                        reset_lives = outcome.info.lives;
                        ep_frame = outcome.info.episode_frame_number;
                    }
                }
                reset_ep_frame = ep_frame;
                let _ = e.drain_audio_samples();
            }
            None => return,
        }
        self.clear_episode_bookkeeping(reset_lives, reset_ep_frame);
        self.broadcast_obs_refresh();
    }

    fn hello(&mut self, client: u64, role: String, name: Option<String>, env_id: Option<String>) {
        let is_agent = role == "agent";
        let role = if is_agent { "agent" } else { "human" }.to_string();
        let name = clean_name(name).unwrap_or_else(|| default_peer_name(client, &role));
        if self.peer_name_in_use(&name, client) {
            self.emit_error(
                "duplicate_name",
                format!("peer name already in use: {name}"),
            );
            return;
        }
        self.clients.insert(
            client,
            ClientInfo {
                role: role.clone(),
                name,
                env_id: env_id.clone(),
            },
        );
        if is_agent {
            if let Some(env_id) = env_id {
                self.declare_env(client, &env_id);
            }
        }
        self.broadcast_ports();
    }

    fn rename(&mut self, client: u64, name: String) {
        let Some(name) = clean_name(Some(name)) else {
            return;
        };
        if self.peer_name_in_use(&name, client) {
            self.emit_error(
                "duplicate_name",
                format!("peer name already in use: {name}"),
            );
            return;
        }
        if let Some(info) = self.clients.get_mut(&client) {
            info.name = name;
            self.broadcast_ports();
        }
    }

    fn is_agent(&self, client: u64) -> bool {
        self.clients
            .get(&client)
            .map(|info| info.role == "agent")
            .unwrap_or(false)
    }

    fn action_port(&self, client: u64, port: Option<usize>) -> Option<usize> {
        if self.is_agent(client) {
            self.owners
                .iter()
                .position(|owner| owner.as_ref().map(|owner| owner.client) == Some(client))
        } else {
            let port = port?;
            (port < PORTS && self.owners[port].as_ref().map(|owner| owner.client) == Some(client))
                .then_some(port)
        }
    }

    fn peer_info(&self, client: u64) -> Option<PeerInfo> {
        self.clients.get(&client).map(|info| PeerInfo {
            id: client,
            role: info.role.clone(),
            name: info.name.clone(),
            env_id: info.env_id.clone(),
        })
    }

    fn owner_ids(&self) -> [Option<u64>; 4] {
        std::array::from_fn(|port| {
            let owner = self.owners[port].as_ref()?;
            Some(owner.client)
        })
    }

    fn port_owner(&self, port: usize) -> Option<PortOwner> {
        let owner = self.owners[port].as_ref()?;
        let role = self
            .clients
            .get(&owner.client)
            .map(|info| info.role.as_str())
            .unwrap_or("human");
        Some(PortOwner {
            client_id: owner.client,
            role: role.to_string(),
            name: owner.name.clone(),
        })
    }

    fn peer_name_in_use(&self, name: &str, except_client: u64) -> bool {
        self.clients
            .iter()
            .any(|(&id, info)| id != except_client && info.name.eq_ignore_ascii_case(name))
    }

    fn player_name_in_use(
        &self,
        name: &str,
        except_port: usize,
        moving_agent: Option<u64>,
    ) -> bool {
        self.owners.iter().enumerate().any(|(port, owner)| {
            let Some(owner) = owner else {
                return false;
            };
            port != except_port
                && Some(owner.client) != moving_agent
                && owner.name.eq_ignore_ascii_case(name)
        })
    }

    fn emit_error(&self, code: &'static str, message: impl Into<String>) {
        let msg = ErrorMsg {
            t: "error",
            code,
            message: message.into(),
        };
        if let Ok(text) = serde_json::to_string(&msg) {
            let _ = self.state_tx.send(OutMsg::Text(text));
        }
    }

    fn broadcast_ports(&self) {
        let mut ids: Vec<u64> = self.clients.keys().copied().collect();
        ids.sort_unstable();
        let peers: Vec<PeerInfo> = ids
            .into_iter()
            .filter_map(|client| self.peer_info(client))
            .collect();
        let owners: Vec<Option<PortOwner>> = (0..PORTS).map(|port| self.port_owner(port)).collect();
        let msg = PortsMsg {
            t: "ports",
            players: self.players,
            ports: PORTS,
            peers,
            owners,
        };
        if let Ok(text) = serde_json::to_string(&msg) {
            let _ = self.state_tx.send(OutMsg::Text(text));
        }
    }

    fn emit_recording(&self) {
        let actions = if self.players >= 2 {
            let n = (self.players as usize).min(PORTS);
            serde_json::json!(self.rec.iter().map(|m| m[..n].to_vec()).collect::<Vec<_>>())
        } else {
            serde_json::json!(self.rec.iter().map(|m| m[0]).collect::<Vec<_>>())
        };
        let msg = RecordingMsg {
            t: "recording",
            game: self.game_id,
            frameskip: 1,
            players: self.players,
            frames: self.rec.len(),
            actions,
        };
        if let Ok(s) = serde_json::to_string(&msg) {
            let _ = self.state_tx.send(OutMsg::Text(s));
        }
    }

    fn emit_ram(&self) {
        let ram: &[u8] = match self.machine.as_ref() {
            Some(e) => e.ram(),
            None => return,
        };
        let b64 = base64::engine::general_purpose::STANDARD.encode(ram);
        let msg = RamMsg {
            t: "ram",
            game: self.game_id,
            bytes: ram.len(),
            b64,
        };
        if let Ok(s) = serde_json::to_string(&msg) {
            let _ = self.state_tx.send(OutMsg::Text(s));
        }
    }

    fn tick(&mut self) -> bool {
        if self.machine.is_none() {
            return false;
        }
        let mut a = self.masks;
        if self.rl_mode && self.sticky_prob > 0.0 {
            let scale = u32::MAX as f32;
            for (p, action) in a.iter_mut().enumerate().take(PORTS) {
                if (self.next_rng() as f32 / scale) < self.sticky_prob {
                    *action = self.prev_applied[p];
                }
            }
        }
        self.prev_applied = a;
        let stepped = {
            let e = self.machine.as_mut().unwrap();
            // Drive the active ports; for a single-player game this is just port 0.
            let players = e.players() as usize;
            e.step(&a[..players]).map(|o| StepData {
                rewards: o.rewards,
                lives: o.info.lives,
                frame: o.info.frame_number,
                ep_frame: o.info.episode_frame_number,
                terminated: o.terminated[..players].iter().all(|&t| t),
                truncated: o.truncated,
            })
        };
        let mut d = match stepped {
            Ok(d) => d,
            Err(_) => {
                self.running = false;
                return false;
            }
        };
        if self.rl_mode && (self.clip_pos > 0.0 || self.clip_neg > 0.0) {
            let hi = if self.clip_pos > 0.0 {
                self.clip_pos
            } else {
                f32::INFINITY
            };
            let lo = if self.clip_neg > 0.0 {
                -self.clip_neg
            } else {
                f32::NEG_INFINITY
            };
            for r in d.rewards.iter_mut() {
                *r = r.clamp(lo, hi);
            }
        }
        self.last_ep_frame = d.ep_frame;
        if self.recording {
            self.rec.push(a);
            if self.rec.len() >= REC_MAX_FRAMES {
                self.emit_recording();
                self.recording = false;
            }
        }

        let (rgb, gray, audio) = {
            let e = self.machine.as_mut().unwrap();
            (
                e.screen_rgb().pixels.clone(),
                e.screen_grayscale().pixels,
                e.drain_audio_samples(),
            )
        };
        let global_sample = FrameSample {
            rgb: Some(&rgb),
            gray: Some(&gray),
            ram: None,
            rewards: d.rewards,
            lives: d.lives,
            frame_number: d.frame,
            episode_frame_number: d.ep_frame,
            terminated: d.terminated,
            truncated: d.truncated,
        };
        let force_global_obs =
            !self.rl_mode || (!self.step_mode && !self.obs_window.has_observation());
        let global_step = match self.obs_window.push_frame(global_sample, force_global_obs) {
            Ok(step) => step,
            Err(_) => {
                self.running = false;
                return false;
            }
        };
        d.terminated = global_step.terminated;
        d.truncated = global_step.truncated;
        self.prev_lives = global_step.lives;
        if self.rl_mode && global_step.obs_step {
            self.rl_step = self.rl_step.wrapping_add(1);
        }
        // Accrue per-step reward; flush into Return + the shown reward together each broadcast (so they always match).
        for p in 0..PORTS {
            self.pending_rewards[p] += d.rewards[p];
        }
        let step_rewards = if !self.step_mode || global_step.obs_step {
            let flushed = self.pending_rewards;
            for (rp, f) in self.rets.iter_mut().zip(flushed) {
                *rp += f;
            }
            self.pending_rewards = [0.0; 4];
            flushed
        } else {
            [0.0; 4]
        };
        let global_shape = self.obs_window.shape();

        let mut agents = std::mem::take(&mut self.agents);
        let mut ids: Vec<u64> = agents.keys().copied().collect();
        ids.sort_unstable();
        let mut agent_views: Vec<AgentView> = Vec::with_capacity(ids.len());
        let mut agent_blocks: Vec<Vec<u8>> = Vec::with_capacity(ids.len());
        let mut lean_sends: Vec<(u64, AgentView, Vec<u8>)> = Vec::new();
        for cid in ids {
            let ao = agents.get_mut(&cid).unwrap();
            let assigned_port = self
                .owners
                .iter()
                .position(|owner| owner.as_ref().map(|owner| owner.client) == Some(cid));
            let port = assigned_port.unwrap_or(0).min(PORTS - 1);
            let mut agent_rewards = [0.0; 4];
            agent_rewards[port] = d.rewards[port];
            let agent_sample = FrameSample {
                rgb: Some(&rgb),
                gray: Some(&gray),
                ram: None,
                rewards: agent_rewards,
                lives: d.lives,
                frame_number: d.frame,
                episode_frame_number: d.ep_frame,
                terminated: d.terminated,
                truncated: d.truncated,
            };
            let force_agent_obs = !ao.window.has_observation();
            let a_step = match ao.window.push_frame(agent_sample, force_agent_obs) {
                Ok(step) => step,
                Err(_) => continue,
            };
            if a_step.obs_step {
                ao.step = ao.step.wrapping_add(1);
                ao.last_step_reward = a_step.rewards[port];
                ao.ret += ao.last_step_reward;
            }
            let shape = ao.window.shape();
            let label = self
                .clients
                .get(&cid)
                .map(|info| info.name.clone())
                .unwrap_or_else(|| default_peer_name(cid, "agent"));
            let view = AgentView {
                id: cid,
                label,
                port: assigned_port.map(|p| p as u8),
                obs_w: shape.width as u32,
                obs_h: shape.height as u32,
                obs_channels: shape.channels,
                reward: ao.last_step_reward,
                ret: ao.ret,
                lives: d.lives[port],
                mask: a[port],
                step: ao.step,
                obs_step: a_step.obs_step,
                assigned: assigned_port.is_some(),
                action_names: ao.action_names.clone(),
                action_masks: ao.action_masks.clone(),
                env: Some(ao.env.clone()),
            };
            agent_views.push(view.clone());
            agent_blocks.push(ao.window.observation().to_vec());
            if a_step.obs_step {
                lean_sends.push((cid, view, ao.window.observation().to_vec()));
            }
        }
        self.agents = agents;

        let meta = StateMeta {
            frame: d.frame,
            ep_frame: d.ep_frame,
            step: self.rl_step,
            rewards: step_rewards,
            rets: self.rets,
            lives: d.lives,
            players: self.players,
            owners: self.owner_ids(),
            terminated: d.terminated,
            truncated: d.truncated,
            native_w: NES_W,
            native_h: NES_H,
            obs_w: global_shape.width as u32,
            obs_h: global_shape.height as u32,
            obs_channels: global_shape.channels,
            env: None,
            obs_step: global_step.obs_step,
            frame_skip: if self.rl_mode {
                self.frame_skip.max(1) as u32
            } else {
                1
            },
            audio_len: audio.len() as u32,
            audio_rate: NesInterface::audio_sample_rate(),
            recording: self.recording,
            running: self.running,
            masks: a,
            agents: agent_views,
        };
        let broadcasted = !self.step_mode || global_step.obs_step;
        if broadcasted {
            let full = Arc::new(encode_state(
                &meta,
                &rgb,
                self.obs_window.observation(),
                &audio,
                &agent_blocks,
            ));
            let _ = self.state_tx.send(OutMsg::Human(full));
            for (cid, view, obs) in lean_sends {
                let mut lm = meta.clone();
                lm.native_w = 0;
                lm.native_h = 0;
                lm.audio_len = 0;
                lm.obs_w = view.obs_w;
                lm.obs_h = view.obs_h;
                lm.obs_channels = view.obs_channels;
                lm.step = view.step;
                lm.rewards = [view.reward, 0.0, 0.0, 0.0];
                lm.rets = [view.ret, 0.0, 0.0, 0.0];
                lm.lives = [view.lives, 0, 0, 0];
                lm.obs_step = true;
                lm.env = view.env;
                lm.agents = Vec::new();
                let bytes = Arc::new(encode_lean(&lm, &obs));
                let _ = self.state_tx.send(OutMsg::Agent { client: cid, bytes });
            }
        }
        if d.terminated || d.truncated {
            if self.recording {
                self.emit_recording();
                self.recording = false;
            }
            if self.rl_mode {
                if self.step_mode {
                    // Hold the terminal frame so its reward/lives stay visible; next Step/Reset autoresets.
                    self.pending_reset = true;
                } else {
                    self.reset_episode();
                }
            }
        }
        broadcasted
    }
}

/// Full frame: `[meta_len][meta JSON][native RGB][global obs][audio][agent obs blocks]`.
fn encode_state(
    meta: &StateMeta,
    rgb: &[u8],
    obs: &[u8],
    audio: &[f32],
    agent_obs: &[Vec<u8>],
) -> Vec<u8> {
    let mj = serde_json::to_vec(meta).unwrap_or_default();
    let agents_len: usize = agent_obs.iter().map(|b| b.len()).sum();
    let mut buf =
        Vec::with_capacity(4 + mj.len() + rgb.len() + obs.len() + audio.len() * 4 + agents_len);
    buf.extend_from_slice(&(mj.len() as u32).to_le_bytes());
    buf.extend_from_slice(&mj);
    buf.extend_from_slice(rgb);
    buf.extend_from_slice(obs);
    for &s in audio {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    for block in agent_obs {
        buf.extend_from_slice(block);
    }
    buf
}

/// Agent frame: `[meta_len][meta JSON][obs]`.
fn encode_lean(meta: &StateMeta, obs: &[u8]) -> Vec<u8> {
    let mj = serde_json::to_vec(meta).unwrap_or_default();
    let mut buf = Vec::with_capacity(4 + mj.len() + obs.len());
    buf.extend_from_slice(&(mj.len() as u32).to_le_bytes());
    buf.extend_from_slice(&mj);
    buf.extend_from_slice(obs);
    buf
}

pub fn spawn(
    state_tx: broadcast::Sender<OutMsg>,
    ready: Arc<Mutex<Option<String>>>,
) -> mpsc::Sender<Command> {
    let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();
    std::thread::Builder::new()
        .name("nesle-console".into())
        .spawn(move || run(cmd_rx, Console::new(state_tx, ready)))
        .expect("spawn console thread");
    cmd_tx
}

fn run(cmd_rx: mpsc::Receiver<Command>, mut console: Console) {
    let frame_dur = Duration::from_micros(1_000_000 / 60);
    loop {
        loop {
            match cmd_rx.try_recv() {
                Ok(cmd) => console.apply(cmd),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return,
            }
        }
        let step_blocked = console.step_mode;
        if console.machine.is_none() || !console.running || step_blocked {
            match cmd_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(cmd) => console.apply(cmd),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
            continue;
        }
        let start = Instant::now();
        console.tick();
        let elapsed = start.elapsed();
        if elapsed < frame_dur {
            std::thread::sleep(frame_dur - elapsed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ready_payload_shape_is_stable() {
        let msg = ReadyMsg {
            t: "ready",
            env_id: "NESLE/SuperC-1P-2-v0".to_string(),
            game: "super_c_1p",
            display_name: "Super C (1P)",
            actions: vec!["NOOP", "RIGHT"],
            action_masks: vec![0, 128],
            native_w: NES_W,
            native_h: NES_H,
            players: 1,
        };
        let value = serde_json::to_value(msg).unwrap();
        assert_eq!(value["t"], "ready");
        assert_eq!(value["native_w"], 256);
        assert_eq!(value["actions"][1], "RIGHT");
        assert_eq!(value["env_id"], "NESLE/SuperC-1P-2-v0");
    }

    #[test]
    fn env_id_resolver_accepts_registered_ids_only() {
        let title = resolve_env_id("NESLE/SuperMarioBros-v0").unwrap();
        assert_eq!(title.game_id, "super_mario_bros");
        assert_eq!(title.start_state, "title");
        assert_eq!(title.profile.unwrap().env_id, "NESLE/SuperMarioBros-v0");

        let mario = resolve_env_id("NESLE/SuperMarioBros-1-1-v2").unwrap();
        assert_eq!(mario.game_id, "super_mario_bros");
        assert_eq!(mario.start_state, "level_1_1");
        let env = mario.profile.unwrap();
        assert_eq!(env.env_id, "NESLE/SuperMarioBros-1-1-v2");
        assert_eq!(env.frame_skip, 4);
        assert!(!env.maxpool);
        assert!(env.remove_sprite_limit);

        let noframeskip = resolve_env_id("NESLE/SuperMarioBros-1-1NoFrameskip-v1")
            .unwrap()
            .profile
            .unwrap();
        assert_eq!(noframeskip.env_id, "NESLE/SuperMarioBros-1-1NoFrameskip-v1");
        assert_eq!(noframeskip.frame_skip, 1);
        assert!(noframeskip.maxpool);

        let multiplayer = resolve_env_id("NESLE/SuperC-2P-2-v0").unwrap();
        assert_eq!(multiplayer.game_id, "super_c_2p");
        assert_eq!(multiplayer.start_state, "level_2");
        assert!(multiplayer.profile.is_none());
        assert!(resolve_env_id("super_c").is_none());
    }

    #[test]
    fn agent_hello_without_port_waits_for_assignment() {
        let (tx, _rx) = broadcast::channel(8);
        let ready = Arc::new(Mutex::new(None));
        let mut console = Console::new(tx, ready);
        console.hello(
            7,
            "agent".to_string(),
            Some("SB3 PPO".to_string()),
            Some("NESLE/SuperMarioBros-1-1-v2".to_string()),
        );
        assert!(console.owners.iter().all(Option::is_none));
        assert!(console.agents.contains_key(&7));

        console.hello(1, "human".to_string(), Some("Alice".to_string()), None);
        console.apply(Command::AssignPort {
            client: 1,
            target_client: Some(7),
            name: Some("SB3 PPO".to_string()),
            port: 0,
        });
        assert_eq!(
            console.owners[0].as_ref().map(|owner| owner.client),
            Some(7)
        );
        assert_eq!(console.owners[0].as_ref().unwrap().name, "SB3 PPO");
    }

    #[test]
    fn agent_hello_never_assigns_a_port() {
        let (tx, _rx) = broadcast::channel(8);
        let ready = Arc::new(Mutex::new(None));
        let mut console = Console::new(tx, ready);
        console.hello(
            7,
            "agent".to_string(),
            Some("SB3 PPO".to_string()),
            Some("NESLE/SuperMarioBros-1-1-v2".to_string()),
        );
        assert!(console.owners.iter().all(Option::is_none));
    }

    #[test]
    fn one_human_peer_can_drive_distinct_named_players() {
        let (tx, _rx) = broadcast::channel(8);
        let ready = Arc::new(Mutex::new(None));
        let mut console = Console::new(tx, ready);
        console.hello(3, "human".to_string(), Some("Alex".to_string()), None);

        console.apply(Command::AssignPort {
            client: 3,
            target_client: Some(3),
            name: Some("Alice".to_string()),
            port: 0,
        });
        console.apply(Command::AssignPort {
            client: 3,
            target_client: Some(3),
            name: Some("Bob".to_string()),
            port: 1,
        });

        let p1 = console.port_owner(0).unwrap();
        let p2 = console.port_owner(1).unwrap();
        assert_eq!(p1.client_id, 3);
        assert_eq!(p2.client_id, 3);
        assert_eq!(p1.name, "Alice");
        assert_eq!(p2.name, "Bob");
    }

    #[test]
    fn duplicate_player_name_is_rejected() {
        let (tx, _rx) = broadcast::channel(8);
        let ready = Arc::new(Mutex::new(None));
        let mut console = Console::new(tx, ready);
        console.hello(3, "human".to_string(), Some("Alex".to_string()), None);

        console.apply(Command::AssignPort {
            client: 3,
            target_client: Some(3),
            name: Some("Alice".to_string()),
            port: 0,
        });
        console.apply(Command::AssignPort {
            client: 3,
            target_client: Some(3),
            name: Some("Alice".to_string()),
            port: 1,
        });

        assert!(console.owners[1].is_none());
    }

    #[test]
    fn duplicate_peer_name_is_rejected() {
        let (tx, _rx) = broadcast::channel(8);
        let ready = Arc::new(Mutex::new(None));
        let mut console = Console::new(tx, ready);
        console.hello(3, "human".to_string(), Some("Alex".to_string()), None);
        console.hello(4, "human".to_string(), Some("Alex".to_string()), None);

        assert!(console.clients.contains_key(&3));
        assert!(!console.clients.contains_key(&4));
    }
}
