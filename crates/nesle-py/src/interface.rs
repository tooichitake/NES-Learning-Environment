#![expect(
    clippy::useless_conversion,
    reason = "PyO3 macro expansion reports PyResult returns as conversions"
)]

use crate::errors::map_error;
use crate::state::PyCoreState;
use nesle_rl::NesInterface;
use pyo3::prelude::*;

type SpriteTile = (u8, u8, u8, u8, u8, u8);
type Sprite0Capture = ((u8, u8, u8, u8, u8), (u8, u8, u8, u8, u8, u8, u8, u8, u8));
type Sprite0HitDebug = (
    (u8, u64, i16, u16, u8, u8, u8, u8, u8, u16),
    (u8, u8, u8, u8, u8, u8, u16, u16, u8),
);
type Mmc1Snapshot = ((u8, u8, u8, u8, u8, u8, u8, u8, u8), (u64, u16, u8, u8, u8));
type WramWrite = (u64, u16, u16, u16, u8, u8, u8, u8, u8, u8, u8, u8);
type CpuTraceEntry = (
    (u64, u16, u8, u8, u8, u8, u8, u8, u8),
    (i16, u16, u8, u8, u8),
);

#[pyclass(name = "NesInterface", unsendable)]
#[derive(Default)]
pub(crate) struct PyNesInterface {
    inner: NesInterface,
}

#[pymethods]
impl PyNesInterface {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn load_rom_bytes(&mut self, bytes: &[u8]) -> PyResult<()> {
        self.inner.load_rom_bytes(bytes).map_err(map_error)
    }

    fn reset_game(&mut self) {
        self.inner.reset_game();
    }

    fn act(&mut self, action_mask: u8) -> PyResult<()> {
        self.inner.act(action_mask).map_err(map_error)
    }

    fn act2(&mut self, mask0: u8, mask1: u8) -> PyResult<()> {
        self.inner.act_n(&[mask0, mask1]).map_err(map_error)
    }

    fn act_n(&mut self, masks: Vec<u8>) -> PyResult<()> {
        self.inner.act_n(&masks).map_err(map_error)
    }

    fn set_four_score_mode(&mut self, enabled: bool) {
        self.inner.set_four_score_mode(enabled);
    }

    fn frame_count(&self) -> u64 {
        self.inner.frame_count()
    }

    fn ram(&self) -> Vec<u8> {
        self.inner.ram().to_vec()
    }

    fn set_ram(&mut self, bytes: &[u8]) -> PyResult<()> {
        self.inner.set_ram(bytes).map_err(map_error)
    }

    fn vram(&self) -> Vec<u8> {
        self.inner.vram().to_vec()
    }

    fn oam(&self) -> Vec<u8> {
        self.inner.oam().to_vec()
    }

    fn palette_ram(&self) -> Vec<u8> {
        self.inner.palette_ram().to_vec()
    }

    fn chr_data(&mut self) -> Vec<u8> {
        self.inner.chr_data()
    }

    fn ppu_sprite_tiles(&self) -> Vec<SpriteTile> {
        self.inner
            .ppu_sprite_tiles()
            .iter()
            .map(|s| {
                (
                    u8::from(s.horizontal_mirror),
                    u8::from(s.background_priority),
                    s.sprite_x,
                    s.low_byte,
                    s.high_byte,
                    s.palette_offset,
                )
            })
            .collect()
    }

    fn ppu_shift_registers(&self) -> (u16, u16, u8, u8) {
        self.inner.ppu_shift_registers()
    }

    fn ppu_sprite0_hit_first_set_clock(&self) -> u64 {
        self.inner.ppu_sprite0_hit_first_set_clock()
    }

    fn set_ppu_capture_target(&mut self, scanline: i32, cycle: u32) {
        self.inner.set_ppu_capture_target(scanline, cycle);
    }

    fn ppu_capture_snapshot(&self) -> (u8, u16, u16, u8, u8, u8, u16, u8) {
        self.inner.ppu_capture_snapshot()
    }

    fn ppu_capture_tile_fetch(&self) -> (u16, u8, u8, u8) {
        self.inner.ppu_capture_tile_fetch()
    }

    fn ppu_capture_sprite0(&self) -> Sprite0Capture {
        let capture = self.inner.ppu_capture_sprite0();
        (
            (
                capture.primary.oam_addr,
                capture.primary.y,
                capture.primary.tile,
                capture.primary.attr,
                capture.primary.x,
            ),
            (
                capture.pipeline.x,
                capture.pipeline.low,
                capture.pipeline.high,
                capture.pipeline.visible,
                capture.pipeline.has_sprite_at_dot,
                capture.pipeline.sec_oam_y,
                capture.pipeline.sec_oam_tile,
                capture.pipeline.sec_oam_attr,
                capture.pipeline.sec_oam_x,
            ),
        )
    }

    fn ppu_sprite0_hit_debug(&self) -> Sprite0HitDebug {
        let s = self.inner.ppu_sprite0_hit_debug();
        (
            (
                u8::from(s.valid),
                s.cpu_cycle,
                s.scanline,
                s.ppu_cycle,
                s.mask,
                s.sprite_count,
                u8::from(s.sprite0_visible),
                s.sprite_bg_color,
                s.sprite_color,
                s.minimum_draw_sprite_standard_cycle,
            ),
            (
                s.sprite0_x,
                s.sprite0_low,
                s.sprite0_high,
                u8::from(s.sprite0_hm),
                u8::from(s.sprite0_bg_pri),
                s.sprite0_pal,
                s.low_bit_shift,
                s.high_bit_shift,
                s.fine_x,
            ),
        )
    }

    fn mmc1_state(&self) -> Option<Mmc1Snapshot> {
        self.inner.mmc1_state().map(|s| {
            (
                (
                    s.write_buffer,
                    s.shift_count,
                    s.control,
                    s.chr_reg0,
                    s.chr_reg1,
                    s.prg_reg,
                    s.mirroring,
                    u8::from(s.chr_mode),
                    u8::from(s.prg_mode),
                ),
                (
                    s.last_write_cycle,
                    s.last_chr_reg,
                    u8::from(s.wram_disable),
                    u8::from(s.force_wram_on),
                    u8::from(s.slot_select),
                ),
            )
        })
    }

    fn apu_frame_counter_state(&self) -> (i32, u32, u32, u8, u8, i16, i8, u8, u64) {
        let s = self.inner.apu_frame_counter_state();
        (
            s.previous_cycle,
            s.current_step,
            s.step_mode,
            u8::from(s.inhibit_irq),
            s.block_frame_counter_tick,
            s.new_value,
            s.write_delay_counter,
            u8::from(s.irq_flag),
            s.irq_flag_clear_clock,
        )
    }

    #[cfg(feature = "audio-synth")]
    fn apu_channel_outputs(&self) -> Vec<u8> {
        self.inner.apu_channel_outputs().to_vec()
    }

    #[cfg(feature = "audio-synth")]
    fn apu_pulse_state(&self, ch: usize) -> Vec<u16> {
        self.inner.apu_pulse_state(ch).to_vec()
    }

    #[cfg(feature = "audio-synth")]
    fn apu_envelope_state(&self, ch: usize) -> Vec<u16> {
        self.inner.apu_envelope_state(ch).to_vec()
    }

    fn screen_indexed(&self) -> Vec<u8> {
        self.inner.screen_indexed().pixels.clone()
    }

    fn screen_gray(&self) -> Vec<u8> {
        self.inner.screen_grayscale().pixels
    }

    fn screen_rgb(&mut self) -> Vec<u8> {
        self.inner.screen_rgb().pixels.clone()
    }

    fn clear_wram_write_log(&mut self) {
        self.inner.clear_wram_write_log();
    }

    fn wram_write_log(&self) -> Vec<WramWrite> {
        self.inner
            .wram_write_log()
            .iter()
            .map(|entry| {
                (
                    entry.cycle_count,
                    entry.pc,
                    entry.addr,
                    entry.normalized_addr,
                    entry.old_value,
                    entry.new_value,
                    entry.op_type,
                    entry.a,
                    entry.x,
                    entry.y,
                    entry.sp,
                    entry.status,
                )
            })
            .collect()
    }

    fn clear_cpu_trace_log(&mut self) {
        self.inner.clear_cpu_trace_log();
    }

    fn set_cpu_trace_enabled(&mut self, enabled: bool) {
        self.inner.set_cpu_trace_enabled(enabled);
    }

    fn cpu_trace_log(&self) -> Vec<CpuTraceEntry> {
        self.inner
            .cpu_trace_log()
            .iter()
            .map(|entry| {
                (
                    (
                        entry.cycle_count,
                        entry.pc,
                        entry.sp,
                        entry.a,
                        entry.x,
                        entry.y,
                        entry.status,
                        entry.irq_flag,
                        entry.nmi_flag,
                    ),
                    (
                        entry.ppu_scanline,
                        entry.ppu_cycle,
                        entry.ppu_status,
                        entry.ppu_sprite0_visible,
                        entry.ppu_mask,
                    ),
                )
            })
            .collect()
    }

    fn clone_state(&self) -> PyCoreState {
        PyCoreState {
            state: self.inner.clone_state(),
        }
    }

    fn restore_state(&mut self, state: &PyCoreState) -> PyResult<()> {
        self.inner.restore_state(&state.state).map_err(map_error)
    }
}
