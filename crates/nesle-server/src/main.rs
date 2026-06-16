//! `nesle-server`: authoritative browser/agent console host.
//!
//! The browser is a thin client. Emulator execution, RL semantics, controller
//! ownership, and start-state restore all stay in the server/runtime layers.

mod client;
mod protocol;
mod session;
mod wire;

use axum::routing::get;
use axum::Router;
use tokio::sync::broadcast;
use tower_http::services::ServeDir;

use client::{ws_handler, AppState};
use session::OutMsg;

#[tokio::main]
async fn main() {
    let web_dir =
        std::env::var("NESLE_WEB_DIR").unwrap_or_else(|_| "crates/nesle-server/web".to_string());
    let addr = std::env::var("NESLE_SERVER_ADDR").unwrap_or_else(|_| "127.0.0.1:8090".to_string());

    let (state_tx, _keep) = broadcast::channel::<OutMsg>(256);
    let state = AppState::new(state_tx);
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .fallback_service(ServeDir::new(&web_dir))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("bind nesle-server address");
    println!("nesle-server: http://{addr}  (ws://{addr}/ws), serving {web_dir}");
    axum::serve(listener, app).await.expect("serve");
}
