use nesle_common::{GrayscaleFrame, IndexedFrame, Result, RgbFrame};
use nesle_core::state::CoreState;
use nesle_core::{bus::WramWriteLogEntry, core::CpuTraceLogEntry, NesCore};

#[derive(Debug, Default)]
pub struct NesInterface {
    core: NesCore,
}

impl NesInterface {
    pub fn load_rom_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.core.load_rom_bytes(bytes)
    }

    pub fn reset_game(&mut self) {
        self.core.reset();
    }

    pub fn act(&mut self, action_mask: u8) -> Result<()> {
        self.core.set_controller_mask(0, action_mask);
        self.core.step_frame()
    }

    /// Step one frame driving up to 4 controller ports for multiplayer / multi-agent
    /// play: `masks[i]` is port i (port 0 = player 1 at $4016, port 1 = player 2 at
    /// $4017; ports 2/3 are multiplexed through the Four Score adapter). Only the
    /// supplied ports are set (-); the rest keep their current mask. Enable the Four
    /// Score adapter via `set_four_score_mode` for >2-player games so ports 2/3 read.
    pub fn act_n(&mut self, masks: &[u8]) -> Result<()> {
        for (port, &mask) in masks.iter().take(4).enumerate() {
            self.core.set_controller_mask(port, mask);
        }
        self.core.step_frame()
    }

    /// Debug-only: step exactly one CPU instruction (no frame boundary
    /// semantics). Returns total cycles consumed (instruction + any
    /// DMA/NMI/IRQ handler). Used by `scripts/cycle_drift_debug.py` and
    /// the trace dumper to pinpoint cycle drift vs the Mesen2 oracle.
    pub fn step_cpu_instruction(&mut self) -> Result<u16> {
        self.core.step_cpu_instruction()
    }

    pub fn frame_count(&self) -> u64 {
        self.core.frame_count()
    }

    pub fn ram(&self) -> &[u8; 0x800] {
        self.core.ram()
    }

    pub fn set_ram(&mut self, bytes: &[u8]) -> Result<()> {
        self.core.set_ram(bytes)
    }

    pub fn clear_wram_write_log(&mut self) {
        self.core.clear_wram_write_log();
    }

    pub fn wram_write_log(&self) -> &[WramWriteLogEntry] {
        self.core.wram_write_log()
    }

    pub fn clear_cpu_trace_log(&mut self) {
        self.core.clear_cpu_trace_log();
    }

    pub fn set_cpu_trace_enabled(&mut self, enabled: bool) {
        self.core.set_cpu_trace_enabled(enabled);
    }

    pub fn cpu_trace_log(&self) -> &[CpuTraceLogEntry] {
        self.core.cpu_trace_log()
    }

    /// VRAM debug accessor (CIRAM nametable).
    pub fn vram(&self) -> &[u8] {
        self.core.vram()
    }

    /// 256-byte sprite OAM debug accessor.
    pub fn oam(&self) -> &[u8; 256] {
        self.core.oam()
    }

    /// 32-byte palette RAM debug accessor.
    pub fn palette_ram(&self) -> &[u8; 32] {
        self.core.palette_ram()
    }

    /// 8KB CHR pattern table snapshot via mapper ppu_read.
    pub fn chr_data(&mut self) -> Vec<u8> {
        self.core.chr_data()
    }

    /// secondary sprite tile array (Battletoads-DD R1 diff).
    pub fn ppu_sprite_tiles(&self) -> &[nesle_core::ppu::NesSpriteInfo; 64] {
        self.core.ppu_sprite_tiles()
    }

    /// BG shift registers + fine-X + sprite count snapshot.
    pub fn ppu_shift_registers(&self) -> (u16, u16, u8, u8) {
        self.core.ppu_shift_registers()
    }

    /// CPU cycle at which sprite-0 hit first transitioned 0-
    /// in current frame (0 if not set this frame).
    pub fn ppu_sprite0_hit_first_set_clock(&self) -> u64 {
        self.core.ppu_sprite0_hit_first_set_clock()
    }

    /// full PPU state snapshot at first sprite-0 hit fire.
    pub fn ppu_sprite0_hit_debug(&self) -> nesle_core::ppu::Sprite0HitDebugSnapshot {
        self.core.ppu_sprite0_hit_debug()
    }

    /// set PPU mid-frame capture target.
    pub fn set_ppu_capture_target(&mut self, scanline: i32, cycle: u32) {
        self.core.set_ppu_capture_target(scanline, cycle);
    }

    /// read PPU mid-frame state snapshot.
    pub fn ppu_capture_snapshot(&self) -> (u8, u16, u16, u8, u8, u8, u16, u8) {
        self.core.ppu_capture_snapshot()
    }

    /// tile fetch state (tile_addr, low, high, palette).
    pub fn ppu_capture_tile_fetch(&self) -> (u16, u8, u8, u8) {
        self.core.ppu_capture_tile_fetch()
    }

    /// Sprite-0-hit localization snapshot at the capture dot.
    pub fn ppu_capture_sprite0(&self) -> nesle_core::ppu::Sprite0Capture {
        self.core.ppu_capture_sprite0()
    }

    /// MMC1 internal register snapshot (Boulder Dash R2 diff).
    pub fn mmc1_state(&self) -> Option<nesle_core::mapper::Mmc1State> {
        self.core.mmc1_state()
    }

    /// APU FrameCounter internal state snapshot.
    pub fn apu_frame_counter_state(&self) -> nesle_core::apu::FrameCounterDebugState {
        self.core.apu_frame_counter_state()
    }

    /// per-channel APU outputs [pulse1, pulse2, triangle, noise, dmc].
    #[cfg(feature = "audio-synth")]
    pub fn apu_channel_outputs(&self) -> [u8; 5] {
        self.core.apu_channel_outputs()
    }

    /// pulse channel (0/1) state for the APU period/sweep diff.
    #[cfg(feature = "audio-synth")]
    pub fn apu_pulse_state(&self, ch: usize) -> [u16; 9] {
        self.core.apu_pulse_state(ch)
    }

    /// envelope channel (0=pulse1, 1=pulse2, 2=noise) envelope +
    /// length-counter state.
    #[cfg(feature = "audio-synth")]
    pub fn apu_envelope_state(&self, ch: usize) -> [u16; 6] {
        self.core.apu_envelope_state(ch)
    }

    pub fn clone_state(&self) -> CoreState {
        self.core.clone_state()
    }

    pub fn restore_state(&mut self, state: &CoreState) -> Result<()> {
        self.core.restore_state(state)
    }

    pub fn screen_indexed(&self) -> &IndexedFrame {
        self.core.indexed_frame()
    }

    pub fn screen_grayscale(&self) -> GrayscaleFrame {
        self.core.grayscale_frame()
    }

    pub fn screen_grayscale_into(&self, pixels: &mut [u8]) {
        self.core.grayscale_frame_into(pixels);
    }

    pub fn screen_rgb(&mut self) -> &RgbFrame {
        self.core.rgb_frame()
    }

    /// Drain APU audio samples. See `Core::drain_audio_samples`.
    pub fn drain_audio_samples(&mut self) -> Vec<f32> {
        self.core.drain_audio_samples()
    }

    /// Enable/disable APU audio synthesis at runtime (ALE `sound`-style). See
    /// `NesCore::set_audio_enabled`. RAM/CPU output is byte-identical either way.
    pub fn set_audio_enabled(&mut self, enabled: bool) {
        self.core.set_audio_enabled(enabled);
    }

    /// Enable/disable RL-only no-sprite-flicker rendering. See
    /// `NesCore::set_remove_sprite_limit`. RAM/CPU/frame output is byte-identical.
    pub fn set_remove_sprite_limit(&mut self, enabled: bool) {
        self.core.set_remove_sprite_limit(enabled);
    }

    /// Force the Four Score 4-controller adapter on/off (see
    /// `NesCore::set_four_score_mode`). Needed for 4-player games whose iNES
    /// header doesn't declare the adapter.
    pub fn set_four_score_mode(&mut self, enabled: bool) {
        self.core.set_four_score_mode(enabled);
    }

    /// Enable/disable per-frame rendering (RL render-skip). See
    /// `NesCore::set_render_enabled`. RAM/CPU/frame_count are byte-identical.
    pub fn set_render_enabled(&mut self, enabled: bool) {
        self.core.set_render_enabled(enabled);
    }

    /// Audio output sample rate (Hz).
    pub fn audio_sample_rate() -> u32 {
        nesle_core::core::NesCore::audio_sample_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::load_test_rom;

    fn test_rom() -> Vec<u8> {
        load_test_rom()
    }

    #[test]
    fn interface_loads_rom_and_steps() {
        let mut interface = NesInterface::default();
        interface.load_rom_bytes(&test_rom()).unwrap();
        assert_eq!(interface.frame_count(), 0);
        interface.act(0).unwrap();
        assert_eq!(interface.frame_count(), 1);
    }

    #[test]
    fn interface_exposes_ram_and_clone_restore() {
        let mut interface = NesInterface::default();
        interface.load_rom_bytes(&test_rom()).unwrap();
        let ram = vec![0xab; 0x800];
        interface.set_ram(&ram).unwrap();
        let state = interface.clone_state();

        interface.set_ram(&vec![0xcd; 0x800]).unwrap();
        assert_eq!(interface.ram()[0], 0xcd);

        interface.restore_state(&state).unwrap();
        assert_eq!(interface.ram()[0], 0xab);
    }
}
