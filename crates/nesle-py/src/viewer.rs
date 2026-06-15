//! In-process SDL2 window backing gym `render_mode="human"` (the ALE
//! `display_screen` analogue). This whole module compiles only under nesle-py's
//! optional `viewer` feature (see the `#[cfg(feature = "viewer")] mod viewer;` in
//! `lib.rs`), so headless builds never link SDL2. The host owns the step loop and
//! calls `present(rgb)` per frame (exactly how ALE's `act()` drives its SDL window).

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::render::Canvas;
use sdl2::video::Window;
use sdl2::{EventPump, Sdl};

/// Native NES framebuffer dimensions.
const NES_W: u32 = 256;
const NES_H: u32 = 240;

/// A live SDL2 window that blits NES RGB frames on demand. Created once and fed
/// `present(rgb)` per step by the host (gym `render_mode="human"`).
pub struct HumanWindow {
    canvas: Canvas<Window>,
    events: EventPump,
    _sdl: Sdl,
}

impl HumanWindow {
    /// Open a window sized to `scale` x (256x240) -- an integer multiple of the
    /// framebuffer, so the picture fills it with no black borders.
    pub fn open(title: &str, scale: u32) -> Result<Self, String> {
        let sdl = sdl2::init()?;
        let video = sdl.video()?;
        let scale = scale.max(1);
        let window = video
            .window(title, NES_W * scale, NES_H * scale)
            .position_centered()
            .resizable()
            .build()
            .map_err(|e| e.to_string())?;
        let canvas = window.into_canvas().build().map_err(|e| e.to_string())?;
        let events = sdl.event_pump()?;
        Ok(Self {
            canvas,
            events,
            _sdl: sdl,
        })
    }

    /// Blit a 256x240 RGB frame and pump events. Returns `true` if the user
    /// requested close (window close button or Esc). A fresh streaming texture
    /// is built per frame -- negligible for a debug window and it sidesteps the
    /// texture/creator self-borrow.
    pub fn present(&mut self, rgb: &[u8]) -> Result<bool, String> {
        let mut closed = false;
        for event in self.events.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => closed = true,
                _ => {}
            }
        }
        let creator = self.canvas.texture_creator();
        let mut texture = creator
            .create_texture_streaming(PixelFormatEnum::RGB24, NES_W, NES_H)
            .map_err(|e| e.to_string())?;
        texture
            .update(None, rgb, (NES_W * 3) as usize)
            .map_err(|e| e.to_string())?;
        self.canvas.clear();
        self.canvas.copy(&texture, None, None)?;
        self.canvas.present();
        Ok(closed)
    }
}
