//! WebSocket client lifecycle and command mapping.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::broadcast;

use crate::protocol::ClientMsg;
use crate::session::{self, Command, OutMsg};
use crate::wire::{game_roster, WelcomeMsg};

#[derive(Clone)]
pub struct AppState {
    state_tx: broadcast::Sender<OutMsg>,
    ready: Arc<Mutex<Option<String>>>,
    cmd_tx: std::sync::mpsc::Sender<Command>,
    next_client: Arc<AtomicU64>,
}

impl AppState {
    pub fn new(state_tx: broadcast::Sender<OutMsg>) -> Self {
        let ready = Arc::new(Mutex::new(None));
        let cmd_tx = session::spawn(state_tx.clone(), ready.clone());
        Self {
            state_tx,
            ready,
            cmd_tx,
            next_client: Arc::new(AtomicU64::new(1)),
        }
    }
}

pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| client(socket, state))
}

async fn client(socket: WebSocket, state: AppState) {
    let (mut sink, mut stream) = socket.split();
    let client_id = state.next_client.fetch_add(1, Ordering::Relaxed);

    let welcome = WelcomeMsg {
        t: "welcome",
        client_id,
        games: game_roster(),
    };
    if let Ok(text) = serde_json::to_string(&welcome) {
        let _ = sink.send(Message::Text(text)).await;
    }

    let snapshot = state.ready.lock().unwrap().clone();
    if let Some(json) = snapshot {
        let _ = sink.send(Message::Text(json)).await;
    }

    let is_agent = Arc::new(AtomicBool::new(false));
    let send_task = spawn_sender(
        client_id,
        is_agent.clone(),
        state.state_tx.subscribe(),
        sink,
    );

    while let Some(Ok(msg)) = stream.next().await {
        let Message::Text(text) = msg else {
            continue;
        };
        let Ok(cm) = serde_json::from_str::<ClientMsg>(&text) else {
            continue;
        };
        handle_client_msg(client_id, &state.cmd_tx, &is_agent, cm);
    }

    let _ = state.cmd_tx.send(Command::Disconnect { client: client_id });
    send_task.abort();
}

fn spawn_sender(
    client_id: u64,
    is_agent: Arc<AtomicBool>,
    mut rx: broadcast::Receiver<OutMsg>,
    mut sink: futures_util::stream::SplitSink<WebSocket, Message>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(OutMsg::Text(text)) => {
                    if sink.send(Message::Text(text)).await.is_err() {
                        break;
                    }
                }
                Ok(OutMsg::Human(bytes)) => {
                    if !is_agent.load(Ordering::Relaxed)
                        && sink.send(Message::Binary(bytes.to_vec())).await.is_err()
                    {
                        break;
                    }
                }
                Ok(OutMsg::Agent { client, bytes }) => {
                    if client == client_id
                        && sink.send(Message::Binary(bytes.to_vec())).await.is_err()
                    {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    })
}

fn handle_client_msg(
    client_id: u64,
    cmd_tx: &std::sync::mpsc::Sender<Command>,
    is_agent: &AtomicBool,
    msg: ClientMsg,
) {
    match msg {
        ClientMsg::Hello { role, name, env_id } => {
            if role == "agent" {
                is_agent.store(true, Ordering::Relaxed);
            }
            send_cmd(
                cmd_tx,
                Command::Hello {
                    client: client_id,
                    role,
                    name,
                    env_id,
                },
            );
        }
        ClientMsg::Rename { name } => send_cmd(
            cmd_tx,
            Command::Rename {
                client: client_id,
                name,
            },
        ),
        ClientMsg::LoadRom { env_id, bytes_b64 } => {
            if let Ok(bytes) =
                base64::engine::general_purpose::STANDARD.decode(bytes_b64.as_bytes())
            {
                send_cmd(cmd_tx, Command::LoadRom { env_id, bytes });
            }
        }
        ClientMsg::Action { port, mask } => send_cmd(
            cmd_tx,
            Command::Action {
                client: client_id,
                port: port.map(usize::from),
                mask,
            },
        ),
        ClientMsg::AssignPort {
            port,
            client_id: target_client,
            name,
        } => send_cmd(
            cmd_tx,
            Command::AssignPort {
                client: client_id,
                target_client,
                name,
                port: port as usize,
            },
        ),
        ClientMsg::Settings {
            running,
            reset,
            obs_size,
            rl_mode,
            step_mode,
            frame_skip,
            maxpool,
            remove_sprite_limit,
            obs_rgb,
            terminal_on_life_loss,
            sticky_prob,
            noop_max,
            clip_pos,
            clip_neg,
        } => {
            if let Some(value) = running {
                send_cmd(cmd_tx, Command::SetRunning(value));
            }
            if reset == Some(true) {
                send_cmd(cmd_tx, Command::Reset);
            }
            if let Some(value) = obs_size {
                send_cmd(cmd_tx, Command::SetObsSize(value));
            }
            if let Some(value) = rl_mode {
                send_cmd(cmd_tx, Command::SetRlMode(value));
            }
            if let Some(value) = step_mode {
                send_cmd(cmd_tx, Command::SetStepMode(value));
            }
            if let Some(value) = frame_skip {
                send_cmd(cmd_tx, Command::SetFrameSkip(value));
            }
            if let Some(value) = maxpool {
                send_cmd(cmd_tx, Command::SetMaxpool(value));
            }
            if let Some(value) = remove_sprite_limit {
                send_cmd(cmd_tx, Command::SetRemoveSpriteLimit(value));
            }
            if let Some(value) = obs_rgb {
                send_cmd(cmd_tx, Command::SetObsRgb(value));
            }
            if let Some(value) = terminal_on_life_loss {
                send_cmd(cmd_tx, Command::SetTerminalOnLifeLoss(value));
            }
            if let Some(value) = sticky_prob {
                send_cmd(cmd_tx, Command::SetStickyProb(value));
            }
            if let Some(value) = noop_max {
                send_cmd(cmd_tx, Command::SetNoopMax(value));
            }
            if let Some(value) = clip_pos {
                send_cmd(cmd_tx, Command::SetClipPos(value));
            }
            if let Some(value) = clip_neg {
                send_cmd(cmd_tx, Command::SetClipNeg(value));
            }
        }
        ClientMsg::Step { masks } => send_cmd(cmd_tx, Command::Step(masks)),
        ClientMsg::Record { on } => send_cmd(cmd_tx, Command::Record { on }),
        ClientMsg::DumpRam => send_cmd(cmd_tx, Command::DumpRam),
    }
}

fn send_cmd(cmd_tx: &std::sync::mpsc::Sender<Command>, command: Command) {
    let _ = cmd_tx.send(command);
}
