use nesle_common::{
    AudioBuffer, FrameDims, GrayscaleFrame, IndexedFrame, NesleError, Result, RgbFrame,
};

use crate::apu::Apu;
use crate::bus::{bus_read, Bus, CpuBusContext, WramWriteLogEntry};
use crate::cartridge::{parse_cartridge_image, CartridgeImage};
use crate::cpu::{Cpu, InterruptLines};
use crate::input::ControllerPorts;
use crate::mapper::{create_mapper, Mapper, MapperImpl};
use crate::ppu::Ppu;
use crate::state::CoreState;

// Match the Mesen2 headless oracle trace cap.
const CPU_TRACE_LOG_CAPACITY: usize = 131_072;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuTraceLogEntry {
    pub cycle_count: u64,
    pub pc: u16,
    pub sp: u8,
    pub a: u8,
    pub x: u8,
    pub y: u8,
    pub status: u8,
    pub irq_flag: u8,
    pub nmi_flag: u8,
    // PPU snapshot paired with each CPU instruction for divergence diagnostics.
    pub ppu_scanline: i16,
    pub ppu_cycle: u16,
    pub ppu_status: u8,
    pub ppu_sprite0_visible: u8,
    pub ppu_mask: u8,
}

/// Snapshot version magic. Prepended to every `clone_state` output and checked
/// by `restore_state`. Bump it on serialized layout changes; old magics are
/// rejected rather than migrated.
///
/// Current `NESF` layout includes framebuffer output so restored start states
/// have valid reset-time RGB/grayscale observations. Older layout history lives
/// in `docs/testing.md` and the snapshot-magic project skill.
const SNAPSHOT_MAGIC: [u8; 4] = *b"NESF";
/// Mesen2 NES default palette for the standard 2C02 PPU
/// (`NesDefaultVideoFilter.cpp:13`, `_ppuPaletteArgb[0]`), 64 ARGB colours.
/// `build_calculated_palette` expands this to the 512-entry RGB LUT
/// (base + 7 colour-emphasis combinations).
const NES_PALETTE_2C02_ARGB: [u32; 64] = [
    0xFF666666, 0xFF002A88, 0xFF1412A7, 0xFF3B00A4, 0xFF5C007E, 0xFF6E0040, 0xFF6C0600, 0xFF561D00,
    0xFF333500, 0xFF0B4800, 0xFF005200, 0xFF004F08, 0xFF00404D, 0xFF000000, 0xFF000000, 0xFF000000,
    0xFFADADAD, 0xFF155FD9, 0xFF4240FF, 0xFF7527FE, 0xFFA01ACC, 0xFFB71E7B, 0xFFB53120, 0xFF994E00,
    0xFF6B6D00, 0xFF388700, 0xFF0C9300, 0xFF008F32, 0xFF007C8D, 0xFF000000, 0xFF000000, 0xFF000000,
    0xFFFFFEFF, 0xFF64B0FF, 0xFF9290FF, 0xFFC676FF, 0xFFF36AFF, 0xFFFE6ECC, 0xFFFE8170, 0xFFEA9E22,
    0xFFBCBE00, 0xFF88D800, 0xFF5CE430, 0xFF45E082, 0xFF48CDDE, 0xFF4F4F4F, 0xFF000000, 0xFF000000,
    0xFFFFFEFF, 0xFFC0DFFF, 0xFFD3D2FF, 0xFFE8C8FF, 0xFFFBC2FF, 0xFFFEC4EA, 0xFFFECCC5, 0xFFF7D8A5,
    0xFFE4E594, 0xFFCFEF96, 0xFFBDF4AB, 0xFFB3F3CC, 0xFFB5EBF2, 0xFFB8B8B8, 0xFF000000, 0xFF000000,
];

/// 2C02 colour index 0 -the pre-render RGB placeholder (matches an
/// all-index-0 blank frame decoded through `calculated_palette`).
const NES_2C02_INDEX0_RGB: [u8; 3] = [
    ((NES_PALETTE_2C02_ARGB[0] >> 16) & 0xFF) as u8,
    ((NES_PALETTE_2C02_ARGB[0] >> 8) & 0xFF) as u8,
    (NES_PALETTE_2C02_ARGB[0] & 0xFF) as u8,
];

#[derive(Debug)]
pub struct NesCore {
    cpu: Cpu,
    ppu: Ppu,
    apu: Apu,
    bus: Bus,
    controllers: ControllerPorts,
    interrupt: InterruptLines,
    cartridge: Option<CartridgeImage>,
    mapper: Option<MapperImpl>,
    initial_state: Option<CoreState>,
    ram: [u8; 0x800],
    frame: IndexedFrame,
    frame_color: Vec<u16>,
    calculated_palette: Box<[[u8; 3]; 512]>,
    calculated_luma: Box<[u8; 512]>,
    rgb_frame: RgbFrame,
    rgb_dirty: bool,
    audio: AudioBuffer,
    cpu_trace_log: Vec<CpuTraceLogEntry>,
    cpu_trace_enabled: bool,
}

impl Default for NesCore {
    fn default() -> Self {
        let calculated_palette = build_calculated_palette();
        let calculated_luma = build_calculated_luma(&calculated_palette);
        Self {
            cpu: Cpu::default(),
            ppu: Ppu::default(),
            apu: Apu::default(),
            bus: Bus::default(),
            controllers: ControllerPorts::default(),
            interrupt: InterruptLines::default(),
            cartridge: None,
            mapper: None,
            initial_state: None,
            ram: [0xff; 0x800],
            frame: IndexedFrame::blank_nes(),
            frame_color: vec![0; FrameDims::NES.len()],
            calculated_palette,
            calculated_luma,
            rgb_frame: default_rgb_frame(),
            rgb_dirty: false,
            audio: AudioBuffer::empty_stereo(),
            cpu_trace_log: Vec::with_capacity(CPU_TRACE_LOG_CAPACITY),
            cpu_trace_enabled: false,
        }
    }
}

impl NesCore {
    pub fn load_rom_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let cartridge = parse_cartridge_image(bytes)?;
        let mapper = create_mapper(&cartridge)?;
        let region = cartridge.region;
        // NES 2.0 byte 15 selects the Four Score input adapter; Rust does
        // not consult Mesen2's GameDB.
        self.controllers
            .set_four_score_mode(cartridge.input_device == 2);
        self.cartridge = Some(cartridge);
        self.mapper = Some(mapper);
        // Region affects CPU, mapper, PPU, and APU timing tables.
        self.apply_region(region);
        self.reset_machine_state();
        self.initial_state = Some(self.clone_state());
        Ok(())
    }

    /// Apply region timing to all timing-dependent subsystems.
    fn apply_region(&mut self, region: crate::cartridge::Region) {
        self.cpu.set_master_clock_divider(region);
        if let Some(mapper) = self.mapper.as_mut() {
            mapper.set_region(region);
        }
        self.ppu.set_region(region);
        self.apu.set_region(region);
    }

    pub fn reset(&mut self) {
        if let Some(state) = self.initial_state.clone() {
            self.restore_state(&state)
                .expect("stored initial core state should restore");
        } else {
            self.reset_machine_state();
        }
    }

    pub fn step_frame(&mut self) -> Result<()> {
        // Run CPU instructions until the PPU completes one frame.
        let start_frame = self.ppu.frame_count;
        while self.ppu.frame_count == start_frame {
            self.step_cpu_instruction_inner()?;
        }
        // Flush lazy APU state and reset its frame cursor.
        self.apu.end_frame(&mut self.interrupt);
        // Snapshot the PPU output buffer (Mesen u16: index | emphasis). Mask
        // to the 6-bit index for IndexedFrame; keep the full value for RGB.
        // RL render-skip: when rendering is disabled the PPU did not produce the
        // output buffer this frame, so skip the (pure-output) frame_color copy +
        // index extract. Emulation state was already updated in the step loop.
        if self.ppu.render_enabled {
            self.frame_color.copy_from_slice(&*self.ppu.output_buffer);
            for (dst, &src) in self.frame.pixels.iter_mut().zip(self.frame_color.iter()) {
                *dst = (src & 0x3F) as u8;
            }
        }
        self.refresh_ram_from_bus();
        self.rgb_dirty = true;
        Ok(())
    }

    pub fn step_cpu_instruction(&mut self) -> Result<u16> {
        let cycles = self.step_cpu_instruction_inner()?;
        self.refresh_ram_from_bus();
        Ok(cycles)
    }

    fn step_cpu_instruction_inner(&mut self) -> Result<u16> {
        // Mapper/APU/PPU push interrupt changes directly through context.
        let cycles_before = self.cpu.cycles;
        if self.cpu_trace_enabled {
            self.record_cpu_trace();
        }
        {
            let mapper = self.mapper.as_mut().ok_or_else(|| {
                NesleError::InvalidState("ROM must be loaded before CPU execution".to_string())
            })?;
            let mapper_has_cpu_clock_hook = mapper.has_cpu_clock_hook();
            let mapper_has_vram_addr_hook = mapper.has_vram_addr_hook();
            let mut ctx = CpuBusContext {
                bus: &mut self.bus,
                // Phase D: `mapper` is now `&mut MapperImpl`; the field type
                // is still `&mut dyn Mapper` for one more migration step, so
                // unsize coercion happens here (free at compile time).
                mapper,
                mapper_has_cpu_clock_hook,
                mapper_has_vram_addr_hook,
                ppu: &mut self.ppu,
                apu: &mut self.apu,
                controllers: &mut self.controllers,
                interrupt: &mut self.interrupt,
                cpu_cycle_count: self.cpu.cycles,
                master_clock: self.cpu.master_clock,
            };
            self.cpu.exec(&mut ctx);
        }
        let total_cycles = (self.cpu.cycles - cycles_before) as u16;
        Ok(total_cycles)
    }

    pub fn set_controller_mask(&mut self, port: usize, mask: u8) {
        self.controllers.set_mask(port, mask);
    }

    /// Force the Four Score adapter independently of cartridge metadata.
    pub fn set_four_score_mode(&mut self, enabled: bool) {
        self.controllers.set_four_score_mode(enabled);
    }

    pub fn four_score_mode(&self) -> bool {
        self.controllers.four_score_mode()
    }

    /// Number of completed frames exposed by the public NESLE API.
    pub fn frame_count(&self) -> u64 {
        (self.ppu.frame_count as u64).saturating_sub(1)
    }

    pub fn ram(&self) -> &[u8; 0x800] {
        &self.ram
    }

    pub fn clear_wram_write_log(&mut self) {
        self.bus.clear_wram_write_log();
    }

    pub fn wram_write_log(&self) -> &[WramWriteLogEntry] {
        self.bus.wram_write_log()
    }

    pub fn clear_cpu_trace_log(&mut self) {
        self.cpu_trace_log.clear();
    }

    pub fn set_cpu_trace_enabled(&mut self, enabled: bool) {
        self.cpu_trace_enabled = enabled;
    }

    pub fn cpu_trace_log(&self) -> &[CpuTraceLogEntry] {
        &self.cpu_trace_log
    }

    /// Diagnostic VRAM view (CIRAM nametable plus mapper 4-screen data).
    pub fn vram(&self) -> &[u8] {
        self.ppu.nametable_view()
    }

    /// Diagnostic 256-byte sprite OAM view.
    pub fn oam(&self) -> &[u8; 256] {
        self.ppu.oam_view()
    }

    /// Diagnostic 32-byte palette RAM view.
    pub fn palette_ram(&self) -> &[u8; 32] {
        self.ppu.palette_view()
    }

    /// Diagnostic 8KB CHR pattern-table snapshot.
    pub fn chr_data(&mut self) -> Vec<u8> {
        let mut out = Vec::with_capacity(0x2000);
        if let Some(mapper) = self.mapper.as_mut() {
            for addr in 0..0x2000u16 {
                out.push(mapper.ppu_read(addr));
            }
        } else {
            out.resize(0x2000, 0);
        }
        out
    }

    /// Diagnostic secondary sprite tile array.
    pub fn ppu_sprite_tiles(&self) -> &[crate::ppu::NesSpriteInfo; 64] {
        self.ppu.sprite_tiles()
    }

    /// Diagnostic BG shift/fine-X/sprite-count snapshot.
    pub fn ppu_shift_registers(&self) -> (u16, u16, u8, u8) {
        self.ppu.shift_registers()
    }

    /// CPU cycle when sprite-0 hit first transitioned 0->1 this frame.
    /// Returns 0 if not set.
    pub fn ppu_sprite0_hit_first_set_clock(&self) -> u64 {
        self.ppu.sprite0_hit_first_set_clock()
    }

    /// Full state snapshot at first sprite-0 hit fire of frame.
    pub fn ppu_sprite0_hit_debug(&self) -> crate::ppu::Sprite0HitDebugSnapshot {
        self.ppu.sprite0_hit_debug()
    }

    /// Set PPU mid-frame capture target.
    pub fn set_ppu_capture_target(&mut self, scanline: i32, cycle: u32) {
        self.ppu.set_ppu_capture_target(scanline, cycle);
    }

    /// Read PPU mid-frame state snapshot.
    pub fn ppu_capture_snapshot(&self) -> (u8, u16, u16, u8, u8, u8, u16, u8) {
        self.ppu.ppu_capture_snapshot()
    }

    /// Captured tile fetch state (tile_addr, low, high, palette).
    pub fn ppu_capture_tile_fetch(&self) -> (u16, u8, u8, u8) {
        self.ppu.ppu_capture_tile_fetch()
    }

    /// Sprite-0-hit localization snapshot at the capture dot.
    pub fn ppu_capture_sprite0(&self) -> crate::ppu::Sprite0Capture {
        self.ppu.ppu_capture_sprite0()
    }

    /// Optional MMC1 internal register snapshot.
    pub fn mmc1_state(&self) -> Option<crate::mapper::Mmc1State> {
        self.mapper.as_ref().and_then(|m| m.mmc1_state())
    }

    /// APU frame-counter diagnostic snapshot.
    pub fn apu_frame_counter_state(&self) -> crate::apu::FrameCounterDebugState {
        self.apu.frame_counter_state()
    }

    /// Diagnostic per-channel gated outputs.
    #[cfg(feature = "audio-synth")]
    pub fn apu_channel_outputs(&self) -> [u8; 5] {
        self.apu.channel_outputs()
    }

    /// Diagnostic pulse channel state.
    #[cfg(feature = "audio-synth")]
    pub fn apu_pulse_state(&self, ch: usize) -> [u16; 9] {
        self.apu.pulse_state(ch)
    }

    /// Diagnostic envelope and length state.
    #[cfg(feature = "audio-synth")]
    pub fn apu_envelope_state(&self, ch: usize) -> [u16; 6] {
        self.apu.envelope_state(ch)
    }

    fn record_cpu_trace(&mut self) {
        if self.cpu_trace_log.len() == CPU_TRACE_LOG_CAPACITY {
            return;
        }
        self.cpu_trace_log.push(CpuTraceLogEntry {
            cycle_count: self.cpu.cycles,
            pc: self.cpu.pc,
            sp: self.cpu.sp,
            a: self.cpu.a,
            x: self.cpu.x,
            y: self.cpu.y,
            status: self.cpu.status,
            irq_flag: self.interrupt.irq_flag,
            nmi_flag: u8::from(self.interrupt.nmi_flag),
            ppu_scanline: self.ppu.scanline,
            ppu_cycle: self.ppu.cycle,
            // PPU status byte ($2002 format) -bit 5 overflow,
            // bit 6 sprite-0 hit, bit 7 vblank. Mesen2's GetStatusByte at
            // BaseNesPpu.h encodes the same 3 bool flags. Rust's
            // `self.ppu.status` already stores the upper 3 bits in
            // $2002 layout (ppu.rs:645/687/774/1091/1256), so mask &
            // 0xE0 is the canonical readout.
            ppu_status: self.ppu.status_byte() & 0xE0,
            ppu_sprite0_visible: u8::from(self.ppu.sprite0_visible),
            ppu_mask: self.ppu.mask_byte(),
        });
    }

    pub fn set_ram(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.ram.len() {
            return Err(NesleError::InvalidState(format!(
                "RAM image must be exactly 0x800 bytes, got {}",
                bytes.len()
            )));
        }
        self.ram.copy_from_slice(bytes);
        self.bus.set_wram(self.ram);
        Ok(())
    }

    #[cold]
    pub fn clone_state(&self) -> CoreState {
        let cpu = self.cpu.snapshot_bytes();
        let bus = self.bus.snapshot_bytes();
        let ppu = self.ppu.snapshot_bytes();
        let apu = self.apu.snapshot_bytes();
        let mapper = self
            .mapper
            .as_ref()
            .map(|mapper| mapper.snapshot_bytes())
            .unwrap_or_default();
        let interrupt_bytes = [
            self.interrupt.irq_flag,
            u8::from(self.interrupt.nmi_flag),
            u8::from(self.interrupt.dmc_dma_pending),
            u8::from(self.interrupt.dmc_dma_stop),
        ];
        let mut bytes = Vec::with_capacity(
            4 + 4
                + cpu.len()
                + interrupt_bytes.len()
                + 0x800
                + 4
                + self.frame.pixels.len()
                + 4
                + self.frame_color.len() * 2
                + 4
                + bus.len()
                + 17
                + 4
                + ppu.len()
                + 4
                + apu.len()
                + 4
                + mapper.len(),
        );
        bytes.extend_from_slice(&SNAPSHOT_MAGIC);
        // PPU snapshot owns frame_count as the single source of truth.
        bytes.extend_from_slice(&(cpu.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&cpu);
        bytes.extend_from_slice(&interrupt_bytes);
        bytes.extend_from_slice(&self.ram);
        bytes.extend_from_slice(&(self.frame.pixels.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.frame.pixels);
        bytes.extend_from_slice(&((self.frame_color.len() * 2) as u32).to_le_bytes());
        for &color in &self.frame_color {
            bytes.extend_from_slice(&color.to_le_bytes());
        }
        bytes.extend_from_slice(&(bus.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&bus);
        bytes.extend_from_slice(&self.controllers.snapshot_bytes());
        bytes.extend_from_slice(&(ppu.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&ppu);
        bytes.extend_from_slice(&(apu.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&apu);
        bytes.extend_from_slice(&(mapper.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&mapper);
        CoreState::from_bytes(self.frame_count(), bytes)
    }

    #[cold]
    pub fn restore_state(&mut self, state: &CoreState) -> Result<()> {
        let raw = state.bytes();
        if raw.len() < SNAPSHOT_MAGIC.len() {
            return Err(NesleError::InvalidState(
                "core snapshot is truncated before magic header".to_string(),
            ));
        }
        // Snapshot format changes bump the magic and reject old versions
        // loudly; identify legacy magics for a clear error message.
        if raw[..SNAPSHOT_MAGIC.len()] != SNAPSHOT_MAGIC {
            let header = &raw[..SNAPSHOT_MAGIC.len()];
            let is_legacy = header.len() == 4
                && &header[..3] == b"NES"
                && matches!(
                    header[3],
                    b'3' | b'4'
                        | b'5'
                        | b'6'
                        | b'7'
                        | b'8'
                        | b'9'
                        | b'A'
                        | b'B'
                        | b'C'
                        | b'D'
                        | b'E'
                );
            if is_legacy {
                return Err(NesleError::InvalidState(format!(
                    "snapshot magic {} no longer supported; load fresh ROM",
                    std::str::from_utf8(header).unwrap_or("?")
                )));
            }
            return Err(NesleError::InvalidState(
                "core snapshot version mismatch: expected NESE magic header".to_string(),
            ));
        }
        let bytes = &raw[SNAPSHOT_MAGIC.len()..];
        if bytes.len() < 4 {
            return Err(NesleError::InvalidState(format!(
                "core snapshot is truncated: got {} bytes (after magic)",
                bytes.len()
            )));
        }
        // PPU snapshot owns frame_count; core has no separate frame payload.
        let mut offset = 0;
        let cpu_len = read_len(bytes, &mut offset)?;
        if bytes.len() < offset + cpu_len + 4 + 0x800 + 4 {
            return Err(NesleError::InvalidState(
                "core snapshot is truncated before CPU/interrupt/RAM state".to_string(),
            ));
        }
        self.cpu
            .restore_snapshot(&bytes[offset..offset + cpu_len])?;
        offset += cpu_len;
        self.interrupt.irq_flag = bytes[offset];
        self.interrupt.nmi_flag = bytes[offset + 1] != 0;
        self.interrupt.dmc_dma_pending = bytes[offset + 2] != 0;
        self.interrupt.dmc_dma_stop = bytes[offset + 3] != 0;
        offset += 4;
        self.ram.copy_from_slice(&bytes[offset..offset + 0x800]);
        offset += 0x800;
        self.bus.set_wram(self.ram);
        let frame_len = read_len(bytes, &mut offset)?;
        if frame_len != FrameDims::NES.len() || bytes.len() < offset + frame_len + 4 {
            return Err(NesleError::InvalidState(format!(
                "indexed framebuffer snapshot length must be {}, got {frame_len}",
                FrameDims::NES.len()
            )));
        }
        self.frame
            .pixels
            .copy_from_slice(&bytes[offset..offset + frame_len]);
        offset += frame_len;
        let frame_color_len = read_len(bytes, &mut offset)?;
        let expected_color_len = FrameDims::NES.len() * 2;
        if frame_color_len != expected_color_len || bytes.len() < offset + frame_color_len + 4 {
            return Err(NesleError::InvalidState(format!(
                "colour framebuffer snapshot length must be {expected_color_len}, got {frame_color_len}"
            )));
        }
        for (dst, chunk) in self
            .frame_color
            .iter_mut()
            .zip(bytes[offset..offset + frame_color_len].chunks_exact(2))
        {
            *dst = u16::from_le_bytes(chunk.try_into().unwrap());
        }
        offset += frame_color_len;
        self.rgb_dirty = true;
        let bus_len = read_len(bytes, &mut offset)?;
        if bus_len != 1 || bytes.len() < offset + bus_len + 25 {
            return Err(NesleError::InvalidState(format!(
                "Bus snapshot length must be 1 byte, got {bus_len}"
            )));
        }
        self.bus
            .restore_snapshot(bytes[offset..offset + bus_len].try_into().unwrap());
        self.bus.set_wram(self.ram);
        offset += bus_len;
        // Controller snapshot layout is fixed at 25 bytes:
        // base state + Four Score + pending input + per-latch masks.
        self.controllers
            .restore_snapshot(bytes[offset..offset + 25].try_into().unwrap());
        offset += 25;
        let ppu_len = read_len(bytes, &mut offset)?;
        if bytes.len() < offset + ppu_len + 4 {
            return Err(NesleError::InvalidState(
                "core snapshot is truncated before PPU state".to_string(),
            ));
        }
        self.ppu
            .restore_snapshot(&bytes[offset..offset + ppu_len])?;
        offset += ppu_len;
        let apu_len = read_len(bytes, &mut offset)?;
        if bytes.len() < offset + apu_len + 4 {
            return Err(NesleError::InvalidState(
                "core snapshot is truncated before APU state".to_string(),
            ));
        }
        self.apu
            .restore_snapshot(&bytes[offset..offset + apu_len])?;
        offset += apu_len;
        let mapper_len = read_len(bytes, &mut offset)?;
        if bytes.len() != offset + mapper_len {
            return Err(NesleError::InvalidState(
                "core snapshot trailing length does not match mapper state".to_string(),
            ));
        }
        if mapper_len != 0 {
            let mapper = self.mapper.as_mut().ok_or_else(|| {
                NesleError::InvalidState("cannot restore mapper state before ROM load".to_string())
            })?;
            mapper.restore_snapshot(&bytes[offset..offset + mapper_len])?;
        }
        // Restore keeps serialized timing state intact; region is fixed at
        // ROM load and is not re-applied here.
        self.cpu_trace_log.clear();
        self.cpu_trace_enabled = false;
        Ok(())
    }

    pub fn indexed_frame(&self) -> &IndexedFrame {
        &self.frame
    }

    pub fn rgb_frame(&mut self) -> &RgbFrame {
        if self.rgb_dirty {
            rgb_from_color_into(
                &self.frame_color,
                self.frame.dims,
                &self.calculated_palette,
                &mut self.rgb_frame,
            );
            self.rgb_dirty = false;
        }
        &self.rgb_frame
    }

    /// Per-pixel Rec.601 luminance of the displayed frame (`round(0.299R + 0.587G + 0.114B)`). Derived from the
    /// SAME 512-entry palette as `rgb_frame` (colour emphasis included), so it
    /// equals the luma of `rgb_frame` pixel-for-pixel and cannot drift. Single
    /// channel, built directly from the `u16` colour buffer (no RGB round-trip).
    pub fn grayscale_frame(&self) -> GrayscaleFrame {
        let mut pixels = Vec::with_capacity(self.frame_color.len());
        for &c in &self.frame_color {
            pixels.push(self.calculated_luma[(c as usize) & 0x1FF]);
        }
        GrayscaleFrame {
            dims: self.frame.dims,
            pixels,
        }
    }

    /// Fill caller-owned storage with the same Rec.601 luminance bytes returned
    /// by [`Self::grayscale_frame`], avoiding a transient frame allocation in RL
    /// observation hot paths.
    pub fn grayscale_frame_into(&self, pixels: &mut [u8]) {
        assert_eq!(
            pixels.len(),
            self.frame_color.len(),
            "grayscale destination length must equal native frame length"
        );
        for (dst, &c) in pixels.iter_mut().zip(&self.frame_color) {
            *dst = self.calculated_luma[(c as usize) & 0x1FF];
        }
    }

    pub fn audio(&self) -> &AudioBuffer {
        &self.audio
    }

    /// Drain accumulated APU audio samples (f32 mono at
    /// [`crate::apu::AUDIO_SAMPLE_RATE`]). Host (e.g. WASM frontend)
    /// should call this once per frame and pipe samples into Web Audio
    /// (or equivalent). Returns ownership of the buffer.
    pub fn drain_audio_samples(&mut self) -> Vec<f32> {
        self.apu.drain_samples()
    }

    /// Enable/disable APU audio synthesis at runtime (ALE `sound`-style toggle).
    /// Disabled = skip per-cycle channel-tick + mix + resample; emulation state
    /// ($4015 / IRQ / DMC DMA) is unaffected so RAM/CPU stays byte-identical.
    /// No effect unless built with the `audio-synth` feature. Survives reset().
    pub fn set_audio_enabled(&mut self, enabled: bool) {
        self.apu.set_audio_enabled(enabled);
    }

    pub fn audio_enabled(&self) -> bool {
        self.apu.audio_enabled()
    }

    /// Enable/disable RL-only no-sprite-flicker rendering (Mesen2
    /// RemoveSpriteLimit). See `crate::ppu::Ppu::set_remove_sprite_limit`:
    /// CPU/RAM/IRQ/frame output is byte-identical; only the rendered framebuffer
    /// gains the per-scanline 9th+ sprites. No effect on emulation determinism.
    pub fn set_remove_sprite_limit(&mut self, enabled: bool) {
        self.ppu.set_remove_sprite_limit(enabled);
    }

    /// Enable/disable per-frame rendering (RL render-skip). See
    /// `crate::ppu::Ppu::set_render_enabled`: when off, the pure pixel-output
    /// (composition + frame readout) is skipped; RAM/CPU/IRQ/frame_count are
    /// byte-identical. The host renders only the frames whose obs it consumes.
    pub fn set_render_enabled(&mut self, enabled: bool) {
        self.ppu.set_render_enabled(enabled);
    }

    /// Audio output sample rate (Hz). Constant -caller can use this
    /// to set up an AudioContext with matching sample rate.
    pub fn audio_sample_rate() -> u32 {
        crate::apu::AUDIO_SAMPLE_RATE
    }

    fn reset_machine_state(&mut self) {
        self.ppu.reset();
        self.apu.reset();
        self.bus.reset();
        self.controllers.reset();
        self.interrupt = InterruptLines::default();
        self.ram = [0xff; 0x800];
        self.frame = IndexedFrame::blank_nes();
        self.frame_color.clear();
        self.frame_color.resize(FrameDims::NES.len(), 0);
        self.rgb_frame = default_rgb_frame();
        self.rgb_dirty = false;
        self.audio = AudioBuffer::empty_stereo();
        self.cpu_trace_log.clear();
        self.cpu_trace_enabled = false;
        if let Some(mapper) = self.mapper.as_mut() {
            mapper.soft_reset();
            self.cpu.reset();
            let mapper_has_cpu_clock_hook = mapper.has_cpu_clock_hook();
            let mapper_has_vram_addr_hook = mapper.has_vram_addr_hook();
            let mut ctx = CpuBusContext {
                bus: &mut self.bus,
                mapper,
                mapper_has_cpu_clock_hook,
                mapper_has_vram_addr_hook,
                ppu: &mut self.ppu,
                apu: &mut self.apu,
                controllers: &mut self.controllers,
                interrupt: &mut self.interrupt,
                cpu_cycle_count: self.cpu.cycles,
                master_clock: self.cpu.master_clock,
            };
            let lo = bus_read(0xfffc, &mut ctx);
            let hi = bus_read(0xfffd, &mut ctx);
            self.cpu.pc = u16::from(lo) | (u16::from(hi) << 8);
            self.cpu.run_power_on_reset_warmup(&mut ctx);
        } else {
            self.cpu.reset();
        }
    }

    fn refresh_ram_from_bus(&mut self) {
        self.ram.copy_from_slice(self.bus.wram());
    }
}

fn default_rgb_frame() -> RgbFrame {
    let mut pixels = Vec::with_capacity(FrameDims::NES.len() * 3);
    for _ in 0..FrameDims::NES.len() {
        pixels.extend_from_slice(&NES_2C02_INDEX0_RGB);
    }
    RgbFrame {
        dims: FrameDims::NES,
        pixels,
    }
}

/// Decode the per-pixel Mesen2 output value (`u16`: low 6 bits = colour
/// index, bits 6-8 = emphasis) to RGB via the 512-entry palette. Mirrors
/// `NesDefaultVideoFilter::DecodePpuBuffer` (`out = _calculatedPalette[idx]`).
fn rgb_from_color_into(
    color: &[u16],
    dims: FrameDims,
    palette: &[[u8; 3]; 512],
    rgb: &mut RgbFrame,
) {
    rgb.dims = dims;
    rgb.pixels.clear();
    rgb.pixels.reserve(color.len() * 3);
    for &c in color {
        rgb.pixels.extend_from_slice(&palette[(c as usize) & 0x1FF]);
    }
}

/// Rec.601 luma LUT for the 512-entry calculated palette: `round(0.299R + 0.587G + 0.114B)` with round-half-to-even
/// (verified bit-exact against a reference implementation over a full frame). Built from the same
/// `build_calculated_palette` output so `grayscale_frame` == luma(`rgb_frame`)
/// for every pixel, colour-emphasis combinations included.
fn build_calculated_luma(palette: &[[u8; 3]; 512]) -> Box<[u8; 512]> {
    let mut luma = Box::new([0u8; 512]);
    for (i, &[r, g, b]) in palette.iter().enumerate() {
        let y = 0.299 * f64::from(r) + 0.587 * f64::from(g) + 0.114 * f64::from(b);
        luma[i] = y.round_ties_even() as u8;
    }
    luma
}

/// Build the 512-entry 2C02 RGB palette with color-emphasis variants.
fn build_calculated_palette() -> Box<[[u8; 3]; 512]> {
    let mut palette = Box::new([[0u8; 3]; 512]);
    for i in 0..64usize {
        let argb = NES_PALETTE_2C02_ARGB[i];
        let r0 = ((argb >> 16) & 0xFF) as f64;
        let g0 = ((argb >> 8) & 0xFF) as f64;
        let b0 = (argb & 0xFF) as f64;
        palette[i] = [r0 as u8, g0 as u8, b0 as u8];
        for j in 1..8usize {
            let (mut r, mut g, mut b) = (r0, g0, b0);
            if (i & 0x0F) <= 0x0D {
                if j & 0x01 != 0 {
                    g *= 0.84;
                    b *= 0.84;
                }
                if j & 0x02 != 0 {
                    r *= 0.84;
                    b *= 0.84;
                }
                if j & 0x04 != 0 {
                    r *= 0.84;
                    g *= 0.84;
                }
            }
            palette[(j << 6) | i] = [
                if r > 255.0 { 255 } else { r as u8 },
                if g > 255.0 { 255 } else { g as u8 },
                if b > 255.0 { 255 } else { b as u8 },
            ];
        }
    }
    palette
}

fn read_len(bytes: &[u8], offset: &mut usize) -> Result<usize> {
    if bytes.len() < *offset + 4 {
        return Err(NesleError::InvalidState(
            "core snapshot length prefix is truncated".to_string(),
        ));
    }
    let value = u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap()) as usize;
    *offset += 4;
    Ok(value)
}

// CPU bus access flows through `CpuBusContext`; per-cycle PPU catch-up
// preserves `$2007` pipeline timing.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculated_palette_matches_mesen2_2c02() {
        let pal = build_calculated_palette();
        // Base colours (emphasis 0) == Mesen2 2C02 `_ppuPaletteArgb[0]`.
        assert_eq!(pal[0x00], [0x66, 0x66, 0x66]);
        assert_eq!(pal[0x0F], [0, 0, 0]);
        assert_eq!(pal[0x20], [0xFF, 0xFE, 0xFF]);
        // Emphasis: GenerateFullColorPalette scales the two channels NOT being
        // intensified by 0.84 (NesDefaultVideoFilter.cpp:45-79).
        assert_eq!(pal[1 << 6], [0x66, 85, 85]); // red   -> G,B *= 0.84
        assert_eq!(pal[2 << 6], [85, 0x66, 85]); // green -> R,B *= 0.84
        assert_eq!(pal[4 << 6], [85, 85, 0x66]); // blue  -> R,G *= 0.84
        assert_eq!(pal[7 << 6], [71, 71, 71]); // R+G+B
        assert_eq!(pal[(1 << 6) | 0x20], [0xFF, 213, 214]); // near-white + red
                                                            // Emphasis is skipped for palette columns $xE/$xF.
        assert_eq!(pal[(7 << 6) | 0x0E], pal[0x0E]);
        assert_eq!(pal[(7 << 6) | 0x0F], pal[0x0F]);
    }

    #[test]
    fn calculated_luma_is_rec601_luminance_not_palette_index() {
        // The grayscale obs must be Rec.601 luminance (the standard getScreenGrayscale,
        // round(0.299R + 0.587G + 0.114B)), NOT the palette index scaled to 0..255.
        let pal = build_calculated_palette();
        let luma = build_calculated_luma(&pal);
        // Hand-computed from the known 2C02 RGB above:
        assert_eq!(luma[0x0F], 0); // black [0,0,0]
        assert_eq!(luma[0x00], 0x66); // gray [102,102,102] -> 102 (weights sum to 1)
        assert_eq!(luma[0x20], 254); // white [255,254,255] -> 254
        assert_eq!(luma[1 << 6], 90); // red-emphasis gray [102,85,85] -> 90
                                      // Sanity: index-scaling would map black ($0F=15) to 60, not 0 -> we are NOT that.
        assert_ne!(luma[0x0F], (0x0Fu16 * 255 / 63) as u8);
        // Every entry equals round-half-to-even of its palette RGB (full LUT guard).
        for (i, &[r, g, b]) in pal.iter().enumerate() {
            let y = (0.299 * f64::from(r) + 0.587 * f64::from(g) + 0.114 * f64::from(b))
                .round_ties_even() as u8;
            assert_eq!(luma[i], y, "luma entry {i}");
        }
    }

    fn nrom_rom(program: &[u8]) -> Vec<u8> {
        let mut rom = Vec::new();
        rom.extend_from_slice(b"NES\x1a");
        rom.extend_from_slice(&[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut prg = vec![0xea; 16 * 1024];
        prg[..program.len()].copy_from_slice(program);
        prg[0x3ffc] = 0x00;
        prg[0x3ffd] = 0x80;
        rom.extend_from_slice(&prg);
        rom.extend_from_slice(&vec![0; 8 * 1024]);
        rom
    }

    #[test]
    fn load_rom_reads_cpu_reset_vector() {
        let mut core = NesCore::default();
        core.load_rom_bytes(&nrom_rom(&[0xea])).unwrap();
        assert_eq!(core.cpu.pc, 0x8000);
    }

    #[test]
    fn cpu_instruction_execution_writes_wram() {
        let mut core = NesCore::default();
        core.load_rom_bytes(&nrom_rom(&[0xa9, 0x42, 0x8d, 0x00, 0x02]))
            .unwrap();

        core.step_cpu_instruction().unwrap();
        core.step_cpu_instruction().unwrap();
        assert_eq!(core.ram()[0x0200], 0x42);
    }

    #[test]
    fn cpu_trace_records_only_after_explicit_enable() {
        let mut core = NesCore::default();
        core.load_rom_bytes(&nrom_rom(&[0xea, 0xea])).unwrap();

        core.step_cpu_instruction().unwrap();
        assert!(core.cpu_trace_log().is_empty());

        core.clear_cpu_trace_log();
        core.step_cpu_instruction().unwrap();
        assert!(core.cpu_trace_log().is_empty());

        core.set_cpu_trace_enabled(true);
        core.clear_cpu_trace_log();
        core.step_cpu_instruction().unwrap();
        assert_eq!(core.cpu_trace_log().len(), 1);
    }

    #[test]
    fn clone_restore_keeps_cpu_and_ram_state() {
        let mut core = NesCore::default();
        core.load_rom_bytes(&nrom_rom(&[0xa9, 0x10, 0x8d, 0x00, 0x02, 0xe8]))
            .unwrap();
        core.step_cpu_instruction().unwrap();
        let state = core.clone_state();

        core.step_cpu_instruction().unwrap();
        core.step_cpu_instruction().unwrap();
        assert_eq!(core.ram()[0x0200], 0x10);
        assert_eq!(core.cpu.x, 1);

        core.restore_state(&state).unwrap();
        assert_eq!(core.ram()[0x0200], 0xff);
        assert_eq!(core.cpu.a, 0x10);
        assert_eq!(core.cpu.x, 0);
    }

    #[test]
    fn clone_restore_keeps_ppu_state() {
        let mut core = NesCore::default();
        core.load_rom_bytes(&nrom_rom(&[0xea])).unwrap();
        {
            let mapper = core.mapper.as_mut().unwrap();
            core.ppu
                .cpu_write_register(0x2006, 0x20, 0, mapper, &mut core.interrupt);
            core.ppu
                .cpu_write_register(0x2006, 0x00, 0, mapper, &mut core.interrupt);
            let target = core.ppu.master_clock + u64::from(core.ppu.master_clock_divider) * 3;
            core.ppu.run(target, mapper, &mut core.interrupt);
            core.ppu
                .cpu_write_register(0x2007, 0x77, 0, mapper, &mut core.interrupt);
            let target = core.ppu.master_clock + u64::from(core.ppu.master_clock_divider) * 6;
            core.ppu.run(target, mapper, &mut core.interrupt);
        }
        let state = core.clone_state();
        {
            let mapper = core.mapper.as_mut().unwrap();
            core.ppu
                .cpu_write_register(0x2006, 0x20, 0, mapper, &mut core.interrupt);
            core.ppu
                .cpu_write_register(0x2006, 0x00, 0, mapper, &mut core.interrupt);
            let target = core.ppu.master_clock + u64::from(core.ppu.master_clock_divider) * 3;
            core.ppu.run(target, mapper, &mut core.interrupt);
            core.ppu
                .cpu_write_register(0x2007, 0x22, 0, mapper, &mut core.interrupt);
            let target = core.ppu.master_clock + u64::from(core.ppu.master_clock_divider) * 6;
            core.ppu.run(target, mapper, &mut core.interrupt);
        }

        core.restore_state(&state).unwrap();
        let mapper = core.mapper.as_mut().unwrap();
        core.ppu
            .cpu_write_register(0x2006, 0x20, 0, mapper, &mut core.interrupt);
        core.ppu
            .cpu_write_register(0x2006, 0x00, 0, mapper, &mut core.interrupt);
        let target = core.ppu.master_clock + u64::from(core.ppu.master_clock_divider) * 3;
        core.ppu.run(target, mapper, &mut core.interrupt);
        assert_eq!(
            core.ppu
                .cpu_read_register(0x2007, mapper, &mut core.interrupt),
            0
        );
        let target = core.ppu.master_clock + u64::from(core.ppu.master_clock_divider) * 6;
        core.ppu.run(target, mapper, &mut core.interrupt);
        assert_eq!(
            core.ppu
                .cpu_read_register(0x2007, mapper, &mut core.interrupt),
            0x77
        );
    }

    #[test]
    fn clone_restore_keeps_framebuffer_output() {
        let mut core = NesCore::default();
        core.load_rom_bytes(&nrom_rom(&[0xea])).unwrap();
        core.frame.pixels[0] = 0x12;
        core.frame_color[0] = 0x21;
        core.rgb_dirty = true;
        let expected_rgb = core.rgb_frame().pixels[0..3].to_vec();
        let state = core.clone_state();

        core.frame.pixels[0] = 0x00;
        core.frame_color[0] = 0x00;
        core.rgb_dirty = true;
        core.restore_state(&state).unwrap();

        assert_eq!(core.frame.pixels[0], 0x12);
        assert_eq!(core.frame_color[0], 0x21);
        assert_eq!(&core.rgb_frame().pixels[0..3], expected_rgb.as_slice());
    }

    #[test]
    fn clone_restore_keeps_hidden_scheduler_state() {
        let mut core = NesCore::default();
        core.load_rom_bytes(&nrom_rom(&[0xea])).unwrap();
        core.cpu.master_clock = 1234;
        core.cpu.ppu_offset = 5;
        core.cpu.need_halt = true;
        core.cpu.sprite_dma_transfer = true;
        core.cpu.sprite_dma_offset = 0x7f;
        core.cpu.prev_need_nmi = true;
        core.interrupt.nmi_flag = true;
        core.interrupt.irq_flag = crate::cpu::IrqSource::External as u8;
        core.ppu.master_clock = 4321;
        core.ppu.scanline = 17;
        core.ppu.cycle = 99;
        core.ppu.need_state_update = true;
        core.ppu.update_vram_addr = 0x2abc;

        let state = core.clone_state();

        core.cpu.master_clock = 0;
        core.cpu.ppu_offset = 1;
        core.cpu.need_halt = false;
        core.cpu.sprite_dma_transfer = false;
        core.cpu.sprite_dma_offset = 0;
        core.cpu.prev_need_nmi = false;
        core.interrupt.nmi_flag = false;
        core.interrupt.irq_flag = 0;
        core.ppu.master_clock = 0;
        core.ppu.scanline = -1;
        core.ppu.cycle = 0;
        core.ppu.need_state_update = false;
        core.ppu.update_vram_addr = 0;

        core.restore_state(&state).unwrap();

        assert_eq!(core.cpu.master_clock, 1234);
        assert_eq!(core.cpu.ppu_offset, 5);
        assert!(core.cpu.need_halt);
        assert!(core.cpu.sprite_dma_transfer);
        assert_eq!(core.cpu.sprite_dma_offset, 0x7f);
        assert!(core.cpu.prev_need_nmi);
        assert!(core.interrupt.nmi_flag);
        assert_eq!(
            core.interrupt.irq_flag,
            crate::cpu::IrqSource::External as u8
        );
        assert_eq!(core.ppu.master_clock, 4321);
        assert_eq!(core.ppu.scanline, 17);
        assert_eq!(core.ppu.cycle, 99);
        assert!(core.ppu.need_state_update);
        assert_eq!(core.ppu.update_vram_addr, 0x2abc);
    }
}
