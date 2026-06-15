use crate::cartridge::Mirroring;
use crate::cpu::InterruptLines;
use crate::mapper::Mapper;

const PALETTE_RAM_BOOT_VALUES: [u8; 0x20] = [
    0x09, 0x01, 0x00, 0x01, 0x00, 0x02, 0x02, 0x0D, 0x08, 0x10, 0x08, 0x24, 0x00, 0x00, 0x04, 0x2C,
    0x09, 0x01, 0x34, 0x03, 0x00, 0x04, 0x00, 0x14, 0x08, 0x3A, 0x00, 0x02, 0x00, 0x20, 0x2C, 0x08,
];

/// BG tile fetch latch for the tile entering the shift registers.
#[derive(Debug, Default, Clone, Copy)]
pub struct TileInfo {
    /// Low pattern byte (fetched at cycle % 8 == 5).
    pub low_byte: u8,
    /// High pattern byte (fetched at cycle % 8 == 7).
    pub high_byte: u8,
    /// Palette offset for this tile (computed from AT byte at cycle % 8 == 3).
    pub palette_offset: u8,
    /// PPU pattern address computed from NT byte, fine Y, and BG table base.
    pub tile_addr: u16,
}

/// Sprite tile data prefetched for next-scanline pixel composition.
#[derive(Debug, Default, Clone, Copy)]
pub struct NesSpriteInfo {
    /// Screen X coordinate (sprite's leftmost pixel).
    pub sprite_x: u8,
    /// Low pattern byte (fetched during sprite tile fetch).
    pub low_byte: u8,
    /// High pattern byte (fetched during sprite tile fetch).
    pub high_byte: u8,
    /// Combined sprite palette offset: `((attr & 0x03) << 2) | 0x10`.
    pub palette_offset: u8,
    /// Sprite is horizontally mirrored (attr bit 6).
    pub horizontal_mirror: bool,
    /// Sprite is behind background (attr bit 5).
    pub background_priority: bool,
}

/// First sprite-0-hit diagnostic snapshot.
#[derive(Debug, Default, Clone, Copy)]
pub struct Sprite0HitDebugSnapshot {
    pub valid: bool,
    pub cpu_cycle: u64,
    pub scanline: i16,
    pub ppu_cycle: u16,
    pub mask: u8,
    pub sprite_count: u8,
    pub sprite0_visible: bool,
    pub sprite_bg_color: u8,
    pub sprite_color: u8,
    pub minimum_draw_sprite_standard_cycle: u16,
    pub sprite0_x: u8,
    pub sprite0_low: u8,
    pub sprite0_high: u8,
    pub sprite0_hm: bool,
    pub sprite0_bg_pri: bool,
    pub sprite0_pal: u8,
    /// Shift register state at hit moment.
    pub low_bit_shift: u16,
    pub high_bit_shift: u16,
    pub fine_x: u8,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Sprite0PrimaryCapture {
    pub oam_addr: u8,
    pub y: u8,
    pub tile: u8,
    pub attr: u8,
    pub x: u8,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Sprite0PipelineCapture {
    pub x: u8,
    pub low: u8,
    pub high: u8,
    pub visible: u8,
    pub has_sprite_at_dot: u8,
    pub sec_oam_y: u8,
    pub sec_oam_tile: u8,
    pub sec_oam_attr: u8,
    pub sec_oam_x: u8,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Sprite0Capture {
    pub primary: Sprite0PrimaryCapture,
    pub pipeline: Sprite0PipelineCapture,
}

#[derive(Debug, Clone, Copy)]
struct SpriteLoad {
    y: u8,
    tile: u8,
    attr: u8,
    x: u8,
    extra: bool,
}

#[derive(Debug, Clone)]
pub struct Ppu {
    ctrl: u8,
    mask: u8,
    status: u8,
    oam_addr: u8,
    /// Secondary OAM pointer for the `$2004` write path. Reset to 0 on
    /// VBlank entry and to `oam_addr & 0x07` on `$2003` write.
    ppu_spl: u8,
    oam: [u8; 256],
    vram_addr: u16,
    temp_addr: u16,
    fine_x: u8,
    write_latch: bool,
    data_buffer: u8,
    io_latch: u8,
    ppu_open_bus: u8,
    ppu_open_bus_decay_stamp: [u32; 8],
    nametable: [u8; 0x1000],
    palette: [u8; 32],
    upper_palette_aliases: [u8; 3],
    odd_frame_toggle: u8,

    // Master-clock timing.
    /// Absolute PPU master clock.
    pub master_clock: u64,
    /// Master clocks consumed per PPU dot. NTSC = 4, PAL = 5, Dendy = 5.
    pub master_clock_divider: u8,
    /// CPU cycle count visible to mapper PPU-address notifications.
    pub mapper_cpu_cycle_count: u64,
    // Per-dot rendering state.
    /// Signed scanline: -1 pre-render, 0..239 visible, 240 post, then VBlank.
    pub scanline: i16,
    /// PPU dot within scanline; cycle 0 is the idle/transition tick.
    pub cycle: u16,
    /// Frame counter incremented when VBlank wraps to pre-render.
    pub frame_count: u32,
    /// NMI scanline; Dendy differs from NTSC/PAL.
    pub nmi_scanline: u16,
    /// Last VBlank scanline before wrapping to pre-render.
    pub vblank_end: u16,
    /// Console region; serialized as the authoritative timing carrier.
    pub region: crate::cartridge::Region,

    /// BG tile fetch state currently being assembled.
    pub tile: TileInfo,
    /// Prefetched sprite tile data for the current scanline.
    pub sprite_tiles: [NesSpriteInfo; 64],
    /// Number of loaded sprite tiles.
    pub sprite_count: u8,
    /// Per-X sprite-pixel presence flag populated during sprite fetch.
    pub has_sprite: [bool; 257],
    /// Sprite 0 survived sprite evaluation for the current scanline.
    pub sprite0_visible: bool,
    /// Sprite 0 was copied into secondary OAM during evaluation.
    pub sprite0_added: bool,
    /// CPU cycle when sprite-0 hit first transitions 0->1 this frame.
    pub sprite0_hit_first_set_clock: u64,
    /// Full state snapshot at first sprite-0 hit of frame.
    pub sprite0_hit_debug: Sprite0HitDebugSnapshot,
    /// Mid-frame diagnostic capture target and result. Capture fires at the
    /// end of the matching tick and is read through `ppu_capture_snapshot()`.
    pub capture_target_scanline: i32,
    pub capture_target_cycle: u32,
    pub capture_valid: u8,
    pub captured_low_shift: u16,
    pub captured_high_shift: u16,
    pub captured_mask: u8,
    pub captured_prev_rendering: u8,
    pub captured_rendering: u8,
    pub captured_vram_addr: u16,
    pub captured_sprite_count: u8,
    /// Captured tile fetch state.
    pub captured_tile_addr: u16,
    pub captured_tile_low_byte: u8,
    pub captured_tile_high_byte: u8,
    pub captured_tile_palette_offset: u8,
    /// Sprite-0/OAM diagnostic capture at the target dot.
    pub captured_oam_addr: u8,
    pub captured_pri_oam0: [u8; 4], // primary OAM sprite 0 (Y,tile,attr,X)
    pub captured_st0_x: u8,         // sprite_tiles[0] X
    pub captured_st0_low: u8,       // sprite_tiles[0] low pattern byte
    pub captured_st0_high: u8,      // sprite_tiles[0] high pattern byte
    pub captured_sprite0_visible: u8,
    pub captured_has_sprite_dot: u8, // has_sprite[capture_cycle]
    pub captured_sec_oam0: [u8; 4],  // secondary OAM slot 0 (Y,tile,attr,X)

    /// BG pattern shift register, low byte.
    pub low_bit_shift: u16,
    /// BG pattern shift register, high byte.
    pub high_bit_shift: u16,

    /// Last address driven onto the PPU bus and reported to mappers.
    pub ppu_bus_address: u16,

    /// `$2002` read-on-VBlank-edge suppression flag.
    pub prevent_vbl_flag: bool,
    /// One-shot per-frame input-commit trigger at the input scanline.
    pub pending_input_commit: bool,
    /// Delayed rendering-enabled state (background OR sprites).
    pub rendering_enabled: bool,
    /// Previous-cycle delayed rendering-enabled state.
    pub prev_rendering_enabled: bool,
    /// Deferred `$2001/$2006/$2007` state update pending.
    pub need_state_update: bool,

    /// Secondary OAM populated during sprite evaluation.
    pub secondary_sprite_ram: [u8; 32],
    /// Secondary OAM write pointer during sprite evaluation.
    pub secondary_oam_addr: u8,
    /// Latched OAM byte being copied during sprite evaluation.
    pub oam_copy_buffer: u8,
    /// Current sprite overlaps the current scanline.
    pub sprite_in_range: bool,
    /// Sprite-eval OAM address high part (sprite index).
    pub sprite_addr_h: u8,
    /// Sprite-eval OAM address low part (byte within sprite).
    pub sprite_addr_l: u8,
    /// Physical OAM address of the most recently in-range sprite.
    pub last_visible_sprite_addr: u8,
    /// Scanline eval start address; brackets the RL no-flicker extra scan.
    pub first_visible_sprite_addr: u8,
    /// RL no-flicker render option; CPU-visible state remains unchanged.
    pub remove_sprite_limit: bool,
    /// RL render-output toggle; PPU timing and CPU-visible effects still run.
    pub render_enabled: bool,
    /// Counter for the sprite-overflow hardware bug.
    pub overflow_bug_counter: u8,
    /// Sprite evaluation has copied all visible sprites for this scanline.
    pub oam_copy_done: bool,
    /// Index of the next sprite tile to load.
    pub sprite_index: u32,
    /// Palette latched for the current 8-pixel BG tile.
    pub current_tile_palette: u8,
    /// Palette latched for the previous tile when fine-X spans a boundary.
    pub previous_tile_palette: u8,
    /// Minimum cycle where BG pixels are visible.
    pub minimum_draw_bg_cycle: u16,
    /// Minimum cycle where sprite pixels are visible.
    pub minimum_draw_sprite_cycle: u16,
    /// Minimum cycle where sprite-0 hit can fire.
    pub minimum_draw_sprite_standard_cycle: u16,

    /// Visible pixel buffer: 6-bit color index plus emphasis bits.
    pub output_buffer: Box<[u16; 256 * 240]>,
    /// Batched grayscale/emphasis state derived from `$2001` and region.
    last_updated_pixel: i32,
    palette_ram_mask: u8,
    intensify_color_bits: u16,

    // Delayed PPU register effects.
    /// `$2006` second-write delay before `vram_addr` is applied.
    pub update_vram_addr_delay: u8,
    /// Pending `$2006` second-write address.
    pub update_vram_addr: u16,
    /// Pending `$2007` post-access VRAM increment.
    pub need_video_ram_increment: bool,
    /// `$2007` read ignore window counter.
    pub ignore_vram_read: u32,
}

impl Default for Ppu {
    fn default() -> Self {
        Self {
            ctrl: 0,
            mask: 0,
            status: 0,
            oam_addr: 0,
            ppu_spl: 0,
            oam: [0xff; 256],
            vram_addr: 0,
            temp_addr: 0,
            fine_x: 0,
            write_latch: false,
            data_buffer: 0,
            io_latch: 0,
            ppu_open_bus: 0,
            ppu_open_bus_decay_stamp: [0; 8],
            nametable: [0xff; 0x1000],
            palette: PALETTE_RAM_BOOT_VALUES,
            upper_palette_aliases: [0; 3],
            odd_frame_toggle: 0,
            pending_input_commit: false,
            // Constructor defaults to NTSC timing.
            master_clock: 0,
            master_clock_divider: 4,
            mapper_cpu_cycle_count: 0,
            // First `tick` after reset wraps to scanline/cycle (0, 0).
            scanline: -1,
            cycle: 340,
            frame_count: 1,
            nmi_scanline: 241,
            vblank_end: 260,
            region: crate::cartridge::Region::Ntsc,
            tile: TileInfo::default(),
            sprite_tiles: [NesSpriteInfo::default(); 64],
            sprite_count: 0,
            has_sprite: [false; 257],
            sprite0_visible: false,
            sprite0_added: false,
            sprite0_hit_first_set_clock: 0,
            sprite0_hit_debug: Sprite0HitDebugSnapshot::default(),
            capture_target_scanline: -2,
            capture_target_cycle: 0,
            capture_valid: 0,
            captured_low_shift: 0,
            captured_high_shift: 0,
            captured_mask: 0,
            captured_prev_rendering: 0,
            captured_rendering: 0,
            captured_vram_addr: 0,
            captured_sprite_count: 0,
            captured_tile_addr: 0,
            captured_tile_low_byte: 0,
            captured_tile_high_byte: 0,
            captured_tile_palette_offset: 0,
            captured_oam_addr: 0,
            captured_pri_oam0: [0; 4],
            captured_st0_x: 0,
            captured_st0_low: 0,
            captured_st0_high: 0,
            captured_sprite0_visible: 0,
            captured_has_sprite_dot: 0,
            captured_sec_oam0: [0; 4],
            low_bit_shift: 0,
            high_bit_shift: 0,
            ppu_bus_address: 0,
            prevent_vbl_flag: false,
            rendering_enabled: false,
            prev_rendering_enabled: false,
            need_state_update: false,
            secondary_sprite_ram: [0xFF; 32],
            secondary_oam_addr: 0,
            oam_copy_buffer: 0,
            sprite_in_range: false,
            sprite_addr_h: 0,
            sprite_addr_l: 0,
            last_visible_sprite_addr: 0,
            first_visible_sprite_addr: 0,
            remove_sprite_limit: false,
            render_enabled: true,
            overflow_bug_counter: 0,
            oam_copy_done: false,
            sprite_index: 0,
            current_tile_palette: 0,
            previous_tile_palette: 0,
            minimum_draw_bg_cycle: 0,
            minimum_draw_sprite_cycle: 0,
            minimum_draw_sprite_standard_cycle: 0,
            output_buffer: Box::new([0; 256 * 240]),
            last_updated_pixel: -1,
            palette_ram_mask: 0x3F,
            intensify_color_bits: 0,
            update_vram_addr_delay: 0,
            update_vram_addr: 0,
            need_video_ram_increment: false,
            ignore_vram_read: 0,
        }
    }
}

impl Ppu {
    pub fn reset(&mut self) {
        self.ctrl = 0;
        self.mask = 0;
        self.status = 0;
        self.oam_addr = 0;
        self.ppu_spl = 0;
        self.oam = [0xff; 256];
        self.vram_addr = 0;
        self.temp_addr = 0;
        self.fine_x = 0;
        self.write_latch = false;
        self.data_buffer = 0;
        self.io_latch = 0;
        self.ppu_open_bus = 0;
        self.ppu_open_bus_decay_stamp = [0; 8];
        // CIRAM powers on as 0xFF to match the Mesen2 oracle configuration.
        self.nametable = [0xff; 0x1000];
        self.palette = PALETTE_RAM_BOOT_VALUES;
        self.upper_palette_aliases = [0; 3];
        self.odd_frame_toggle = 0;
        // Master-clock divider is a region property, not runtime state.
        self.master_clock = 0;
        // Per-dot state reset.
        self.scanline = -1;
        self.cycle = 340;
        self.frame_count = 1;
        self.tile = TileInfo::default();
        self.sprite_tiles = [NesSpriteInfo::default(); 64];
        self.sprite_count = 0;
        self.has_sprite = [false; 257];
        self.sprite0_visible = false;
        self.sprite0_added = false;
        self.sprite0_hit_first_set_clock = 0;
        self.sprite0_hit_debug = Sprite0HitDebugSnapshot::default();
        self.low_bit_shift = 0;
        self.high_bit_shift = 0;
        self.ppu_bus_address = 0;
        self.prevent_vbl_flag = false;
        self.rendering_enabled = false;
        self.prev_rendering_enabled = false;
        self.need_state_update = false;
        self.secondary_sprite_ram = [0xFF; 32];
        self.secondary_oam_addr = 0;
        self.oam_copy_buffer = 0;
        self.sprite_in_range = false;
        self.sprite_addr_h = 0;
        self.sprite_addr_l = 0;
        self.overflow_bug_counter = 0;
        self.oam_copy_done = false;
        self.sprite_index = 0;
        self.current_tile_palette = 0;
        self.previous_tile_palette = 0;
        self.minimum_draw_bg_cycle = 0;
        self.minimum_draw_sprite_cycle = 0;
        self.minimum_draw_sprite_standard_cycle = 0;
        self.output_buffer.fill(0);
        self.last_updated_pixel = -1;
        self.palette_ram_mask = 0x3F;
        self.intensify_color_bits = 0;
        self.update_vram_addr_delay = 0;
        self.update_vram_addr = 0;
        self.need_video_ram_increment = false;
        self.ignore_vram_read = 0;
    }

    /// per-instruction trace accessor. Returns the raw $2002
    /// status byte (upper 3 bits = SpriteOverflow/Sprite0Hit/VerticalBlank).
    pub fn status_byte(&self) -> u8 {
        self.status
    }

    /// per-instruction trace accessor. Returns the raw $2001
    /// mask byte (Grayscale/BG-mask/Sprite-mask/BG-enabled/Sprites-enabled/
    /// Intensify-R/G/B). Mirrors Mesen2 BaseNesPpu::GetMaskByte byte layout.
    pub fn mask_byte(&self) -> u8 {
        self.mask
    }

    /// Diagnostic nametable buffer view; mapper 4-screen size may differ.
    pub fn nametable_view(&self) -> &[u8] {
        &self.nametable
    }

    /// Diagnostic 256-byte sprite OAM view.
    pub fn oam_view(&self) -> &[u8; 256] {
        &self.oam
    }

    /// Diagnostic 32-byte palette RAM view.
    pub fn palette_view(&self) -> &[u8; 32] {
        &self.palette
    }

    /// Diagnostic secondary sprite tile array; first 8 are hardware-visible.
    pub fn sprite_tiles(&self) -> &[NesSpriteInfo; 64] {
        &self.sprite_tiles
    }

    /// Diagnostic BG shift/fine-X/sprite-count snapshot.
    pub fn shift_registers(&self) -> (u16, u16, u8, u8) {
        (
            self.low_bit_shift,
            self.high_bit_shift,
            self.fine_x,
            self.sprite_count,
        )
    }

    /// CPU cycle when sprite-0 hit first transitioned 0->1 this frame.
    /// Returns 0 if not yet set.
    pub fn sprite0_hit_first_set_clock(&self) -> u64 {
        self.sprite0_hit_first_set_clock
    }

    /// Full state snapshot at the first sprite-0 hit this frame.
    pub fn sprite0_hit_debug(&self) -> Sprite0HitDebugSnapshot {
        self.sprite0_hit_debug
    }

    /// Set the mid-frame diagnostic capture target.
    pub fn set_ppu_capture_target(&mut self, scanline: i32, cycle: u32) {
        self.capture_target_scanline = scanline;
        self.capture_target_cycle = cycle;
        self.capture_valid = 0;
    }

    /// Return mid-frame PPU state captured at target cycle.
    /// Returns 9-tuple: (valid, low_shift, high_shift, mask,
    /// prev_rendering, rendering, vram_addr, sprite_count, unused).
    pub fn ppu_capture_snapshot(&self) -> (u8, u16, u16, u8, u8, u8, u16, u8) {
        (
            self.capture_valid,
            self.captured_low_shift,
            self.captured_high_shift,
            self.captured_mask,
            self.captured_prev_rendering,
            self.captured_rendering,
            self.captured_vram_addr,
            self.captured_sprite_count,
        )
    }

    /// Return captured tile fetch state (tile_addr, low_byte,
    /// high_byte, palette_offset) at the capture target cycle.
    pub fn ppu_capture_tile_fetch(&self) -> (u16, u8, u8, u8) {
        (
            self.captured_tile_addr,
            self.captured_tile_low_byte,
            self.captured_tile_high_byte,
            self.captured_tile_palette_offset,
        )
    }

    /// Sprite-0-hit localization snapshot at the capture dot.
    pub fn ppu_capture_sprite0(&self) -> Sprite0Capture {
        Sprite0Capture {
            primary: Sprite0PrimaryCapture {
                oam_addr: self.captured_oam_addr,
                y: self.captured_pri_oam0[0],
                tile: self.captured_pri_oam0[1],
                attr: self.captured_pri_oam0[2],
                x: self.captured_pri_oam0[3],
            },
            pipeline: Sprite0PipelineCapture {
                x: self.captured_st0_x,
                low: self.captured_st0_low,
                high: self.captured_st0_high,
                visible: self.captured_sprite0_visible,
                has_sprite_at_dot: self.captured_has_sprite_dot,
                sec_oam_y: self.captured_sec_oam0[0],
                sec_oam_tile: self.captured_sec_oam0[1],
                sec_oam_attr: self.captured_sec_oam0[2],
                sec_oam_x: self.captured_sec_oam0[3],
            },
        }
    }

    /// Apply deferred `$2001/$2006/$2007` side effects at tick end.
    fn update_state<const VRAM_HOOK: bool, M: Mapper + ?Sized>(
        &mut self,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) {
        self.need_state_update = false;

        if self.prev_rendering_enabled != self.rendering_enabled {
            self.prev_rendering_enabled = self.rendering_enabled;
            if self.scanline < 240 && !self.prev_rendering_enabled {
                self.set_bus_address_cached::<VRAM_HOOK, M>(
                    self.vram_addr & 0x3fff,
                    mapper,
                    interrupt,
                );
                if (65..=256).contains(&self.cycle) {
                    self.oam_addr = self.oam_addr.wrapping_add(1);
                    self.sprite_addr_h = (self.oam_addr >> 2) & 0x3f;
                    self.sprite_addr_l = self.oam_addr & 0x03;
                }
            }
        }

        let raw_rendering_enabled = self.mask & 0x18 != 0;
        if self.rendering_enabled != raw_rendering_enabled {
            self.rendering_enabled = raw_rendering_enabled;
            self.need_state_update = true;
        }

        if self.update_vram_addr_delay > 0 {
            self.update_vram_addr_delay -= 1;
            if self.update_vram_addr_delay == 0 {
                // The oracle keeps the optional `$2006` scroll glitch disabled.
                self.vram_addr = self.update_vram_addr;
                // Mesen2 mirrors the delayed V update back into T.
                self.temp_addr = self.vram_addr;
                if self.scanline >= 240 || !self.is_rendering_enabled() {
                    self.set_bus_address_cached::<VRAM_HOOK, M>(
                        self.vram_addr & 0x3fff,
                        mapper,
                        interrupt,
                    );
                }
            } else {
                self.need_state_update = true;
            }
        }

        // $2007 ignore-vram-read counter (6 PPU cycles).
        if self.ignore_vram_read > 0 {
            self.ignore_vram_read -= 1;
            if self.ignore_vram_read > 0 {
                self.need_state_update = true;
            }
        }

        // `$2007` VRAM increment applies after one PPU cycle.
        if self.need_video_ram_increment {
            self.need_video_ram_increment = false;
            // Outside rendering, `$2007` increments by PPUCTRL bit 2.
            if self.scanline >= 240 || !self.is_rendering_enabled() {
                let inc = if self.ctrl & 0x04 != 0 { 32 } else { 1 };
                self.vram_addr = self.vram_addr.wrapping_add(inc) & 0x7FFF;
                // Update the bus address after the increment so mapper
                // A12 watchers see `$2007` address changes.
                self.set_bus_address_cached::<VRAM_HOOK, M>(
                    self.vram_addr & 0x3FFF,
                    mapper,
                    interrupt,
                );
            } else {
                self.inc_horizontal_scrolling();
                self.inc_vertical_scrolling();
            }
        }
    }

    /// Resolve and write the current visible pixel, including palette-0 mirroring.
    fn draw_pixel(&mut self) {
        // Render-skip sprite-0-hit fast path (RL-only). Once sprite-0 hit has
        // latched for the frame (PPUSTATUS bit 6 set), `get_pixel_color()` has
        // no remaining observable effect: its ONLY PPU-state writes are the hit
        // flag + its debug snapshot, all gated on `status & 0x40 == 0` (ppu.rs
        // ~1516). Bit 6 is cleared once per frame at pre-render (ppu.rs ~991),
        // so before the hit every dot still runs the full check (the trigger
        // dot is unchanged); after the hit the check is dead code. With
        // rendering disabled the returned pixel color is discarded too, so the
        // whole pixel resolution becomes a no-op and is skipped here -- byte-
        // identical to running it. The `render_enabled == true` path (the
        // Mesen2-gated full-render path) is deliberately NOT short-circuited,
        // so full-render alignment is untouched. SMB1's hit fires around
        // scanline ~30, so ~87% of a frame's visible dots take this fast path
        // under render-skip (the RL training default).
        if !self.render_enabled && (self.status & 0x40) != 0 {
            return;
        }
        let forced_blank_palette =
            !self.is_rendering_enabled() && (self.vram_addr & 0x3F00) == 0x3F00;
        let palette_offset = if forced_blank_palette {
            (self.vram_addr & 0x001F) as u8
        } else {
            self.get_pixel_color()
        };
        // Render-skip elides pure output after sprite-0 hit logic has run.
        if !self.render_enabled {
            return;
        }
        // Mesen2 palette-0 mirroring: if low 2 bits are 0 (transparent),
        // use universal BG color (palette[0]); else use computed offset.
        let palette_lookup = if forced_blank_palette || palette_offset & 0x03 != 0 {
            palette_offset as usize & 0x1F
        } else {
            0
        };
        let palette_idx = self.palette_read(0x3F00 | palette_lookup as u16) & 0x3F;
        let buf_idx = (self.scanline as usize) * 256 + (self.cycle as usize - 1);
        if buf_idx < self.output_buffer.len() {
            self.output_buffer[buf_idx] = palette_idx as u16;
        }
    }

    /// Batch-apply grayscale/emphasis to pixels produced since the last flush.
    fn update_grayscale_bits(&mut self) {
        if !self.render_enabled {
            // Keep masks current even when the output buffer is skipped.
            self.update_color_bit_masks();
            return;
        }
        let scan = self.scanline as i32;
        if scan < 0 || scan > self.nmi_scanline as i32 {
            self.update_color_bit_masks();
            return;
        }
        let pixel_number: i32 = if scan >= 240 {
            61439
        } else if (self.cycle as i32) < 3 {
            (scan << 8) - 1
        } else if self.cycle <= 258 {
            (scan << 8) + self.cycle as i32 - 3
        } else {
            (scan << 8) + 255
        };
        if self.palette_ram_mask == 0x3F && self.intensify_color_bits == 0 {
            // Most common case: no grayscale, no emphasis -nothing to apply.
            self.update_color_bit_masks();
            self.last_updated_pixel = pixel_number;
            return;
        }
        if self.last_updated_pixel < pixel_number {
            let mask = self.palette_ram_mask as u16;
            let intensify = self.intensify_color_bits;
            while self.last_updated_pixel < pixel_number {
                let idx = (self.last_updated_pixel + 1) as usize;
                if idx < self.output_buffer.len() {
                    self.output_buffer[idx] = (self.output_buffer[idx] & mask) | intensify;
                }
                self.last_updated_pixel += 1;
            }
        }
        self.update_color_bit_masks();
    }

    /// Derive grayscale and color-emphasis masks from `$2001` and region.
    fn update_color_bit_masks(&mut self) {
        self.palette_ram_mask = if self.mask & 0x01 != 0 { 0x30 } else { 0x3F };
        let (intensify_red, intensify_green) = match self.region {
            crate::cartridge::Region::Ntsc => (self.mask & 0x20 != 0, self.mask & 0x40 != 0),
            _ => (self.mask & 0x40 != 0, self.mask & 0x20 != 0),
        };
        let intensify_blue = self.mask & 0x80 != 0;
        self.intensify_color_bits = (if intensify_red { 0x40 } else { 0 })
            | (if intensify_green { 0x80 } else { 0 })
            | (if intensify_blue { 0x100 } else { 0 });
    }

    /// Recompute BG/sprite visibility start cycles from `$2001`.
    fn update_minimum_draw_cycles(&mut self) {
        let bg_enabled = self.mask & 0x08 != 0;
        let sprites_enabled = self.mask & 0x10 != 0;
        let bg_leftmost_shown = self.mask & 0x02 != 0;
        let sprite_leftmost_shown = self.mask & 0x04 != 0;
        self.minimum_draw_bg_cycle = if bg_enabled {
            if bg_leftmost_shown {
                0
            } else {
                8
            }
        } else {
            300
        };
        self.minimum_draw_sprite_cycle = if sprites_enabled {
            if sprite_leftmost_shown {
                0
            } else {
                8
            }
        } else {
            300
        };
        // Sprite-0 hit always uses the standard mask (Mesen2 BaseNesPpu.cpp:163).
        self.minimum_draw_sprite_standard_cycle = if sprites_enabled {
            if sprite_leftmost_shown {
                0
            } else {
                8
            }
        } else {
            300
        };
    }

    /// Advance the PPU to the target master clock.
    pub fn run<M: Mapper + ?Sized>(
        &mut self,
        target_master_clock: u64,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) {
        let mapper_has_vram_addr_hook = mapper.has_vram_addr_hook();
        self.run_with_cpu_cycle(
            target_master_clock,
            mapper,
            interrupt,
            self.mapper_cpu_cycle_count,
            mapper_has_vram_addr_hook,
        );
    }

    pub fn run_with_cpu_cycle<M: Mapper + ?Sized>(
        &mut self,
        target_master_clock: u64,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
        cpu_cycle_count: u64,
        mapper_has_vram_addr_hook: bool,
    ) {
        self.mapper_cpu_cycle_count = cpu_cycle_count;
        if mapper_has_vram_addr_hook {
            self.run_with_cpu_cycle_inner::<true, M>(target_master_clock, mapper, interrupt);
        } else {
            self.run_with_cpu_cycle_inner::<false, M>(target_master_clock, mapper, interrupt);
        }
    }

    fn run_with_cpu_cycle_inner<const VRAM_HOOK: bool, M: Mapper + ?Sized>(
        &mut self,
        target_master_clock: u64,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) {
        let divider = u64::from(self.master_clock_divider);
        // Tick before advancing master clock.
        while self.master_clock + divider <= target_master_clock {
            self.tick::<VRAM_HOOK, M>(mapper, interrupt);
            self.master_clock += divider;
        }
    }

    /// Configure master-clock divider and default VBlank timing.
    pub fn set_master_clock_divider(&mut self, divider: u8) {
        self.master_clock_divider = divider;
        // PAL/Dendy callers override the scanline values after this.
        if divider == 4 {
            self.nmi_scanline = 241;
            self.vblank_end = 260;
        } else {
            // PAL/Dendy: 50 Hz timing.
            self.nmi_scanline = 241; // PAL; Dendy = 291 (caller overrides).
            self.vblank_end = 310;
        }
    }

    /// Apply region timing; snapshots carry both region and derived fields.
    #[cold]
    pub fn set_region(&mut self, region: crate::cartridge::Region) {
        self.region = region;
        match region {
            crate::cartridge::Region::Ntsc => {
                self.set_master_clock_divider(4);
            }
            crate::cartridge::Region::Pal => {
                self.set_master_clock_divider(5);
                self.nmi_scanline = 241;
                self.vblank_end = 310;
            }
            crate::cartridge::Region::Dendy => {
                self.set_master_clock_divider(5);
                self.nmi_scanline = 291;
                self.vblank_end = 310;
            }
        }
    }

    /// Execute one PPU dot.
    fn tick<const VRAM_HOOK: bool, M: Mapper + ?Sized>(
        &mut self,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) {
        if self.cycle < 340 {
            self.cycle += 1;
            if self.scanline < 240 {
                self.process_scanline_tick::<VRAM_HOOK, M>(mapper, interrupt);
            } else if self.cycle == 1 && self.scanline == self.nmi_scanline as i16 {
                // VBlank entry.
                if !self.prevent_vbl_flag {
                    self.status |= 0x80;
                    if self.ctrl & 0x80 != 0 {
                        interrupt.set_nmi_flag();
                    }
                }
                self.prevent_vbl_flag = false;
            }
        } else {
            self.process_scanline_first_cycle::<VRAM_HOOK, M>(mapper, interrupt);
        }
        if self.need_state_update {
            self.update_state::<VRAM_HOOK, M>(mapper, interrupt);
        }
        // Mid-frame diagnostic capture trigger.
        if self.scanline as i32 == self.capture_target_scanline
            && self.cycle as u32 == self.capture_target_cycle
        {
            self.capture_valid = 1;
            self.captured_low_shift = self.low_bit_shift;
            self.captured_high_shift = self.high_bit_shift;
            self.captured_mask = self.mask;
            self.captured_prev_rendering = u8::from(self.prev_rendering_enabled);
            self.captured_rendering = u8::from(self.rendering_enabled);
            self.captured_vram_addr = self.vram_addr;
            self.captured_sprite_count = self.sprite_count;
            self.captured_tile_addr = self.tile.tile_addr;
            self.captured_tile_low_byte = self.tile.low_byte;
            self.captured_tile_high_byte = self.tile.high_byte;
            self.captured_tile_palette_offset = self.tile.palette_offset;
            // Sprite-0-hit localization state.
            self.captured_oam_addr = self.oam_addr;
            self.captured_pri_oam0 = [self.oam[0], self.oam[1], self.oam[2], self.oam[3]];
            self.captured_st0_x = self.sprite_tiles[0].sprite_x;
            self.captured_st0_low = self.sprite_tiles[0].low_byte;
            self.captured_st0_high = self.sprite_tiles[0].high_byte;
            self.captured_sprite0_visible = u8::from(self.sprite0_visible);
            self.captured_has_sprite_dot = u8::from(
                (self.cycle as usize) < self.has_sprite.len()
                    && self.has_sprite[self.cycle as usize],
            );
            self.captured_sec_oam0 = [
                self.secondary_sprite_ram[0],
                self.secondary_sprite_ram[1],
                self.secondary_sprite_ram[2],
                self.secondary_sprite_ram[3],
            ];
        }
    }

    /// Per-dot dispatch inside visible and pre-render scanlines.
    fn process_scanline_tick<const VRAM_HOOK: bool, M: Mapper + ?Sized>(
        &mut self,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) {
        let cycle = self.cycle;
        if cycle <= 256 {
            self.load_tile_info::<VRAM_HOOK, M>(mapper, interrupt);

            if self.prev_rendering_enabled && (cycle & 0x07) == 0 {
                self.inc_horizontal_scrolling();
                if cycle == 256 {
                    self.inc_vertical_scrolling();
                }
            }

            if self.scanline >= 0 {
                self.draw_pixel();
                self.shift_tile_registers();
                self.process_sprite_evaluation();
            } else if (1..9).contains(&cycle) {
                // Pre-render cycle 1 clears VBL/NMI; cycles 1-8 model the
                // OAMADDR>=8 sprite-0 corruption behavior.
                if cycle == 1 {
                    self.status &= 0x1F;
                    interrupt.clear_nmi_flag();
                }
                if self.oam_addr >= 0x08 && self.is_rendering_enabled() {
                    let src =
                        usize::from(self.oam_addr & 0xF8).wrapping_add(usize::from(cycle - 1));
                    let dst = usize::from(cycle - 1);
                    self.oam[dst & 0xFF] = self.oam[src & 0xFF];
                }
            }
        } else if (257..=320).contains(&cycle) {
            if cycle == 257 {
                self.sprite_index = 0;
                self.has_sprite = [false; 257];
                if self.prev_rendering_enabled {
                    self.vram_addr = (self.vram_addr & !0x041F) | (self.temp_addr & 0x041F);
                }
            }
            if self.is_rendering_enabled() {
                self.oam_addr = 0;
                match (cycle - 257) % 8 {
                    0 => {
                        let addr = self.get_nametable_addr();
                        let _ = self.read_vram::<VRAM_HOOK, M>(addr, mapper, interrupt);
                    }
                    2 => {
                        let addr = self.get_attribute_addr();
                        let _ = self.read_vram::<VRAM_HOOK, M>(addr, mapper, interrupt);
                    }
                    4 => self.load_sprite_tile_info::<VRAM_HOOK, M>(mapper, interrupt),
                    _ => {}
                }
                if self.scanline == -1 && (280..=304).contains(&cycle) {
                    self.vram_addr = (self.vram_addr & !0x7BE0) | (self.temp_addr & 0x7BE0);
                }
                // Extra sprites load after the 8 standard fetches, before BG prefetch.
                if cycle == 320 {
                    self.load_extra_sprites::<VRAM_HOOK, M>(mapper, interrupt);
                }
            }
        } else if (321..=336).contains(&cycle) {
            self.load_tile_info::<VRAM_HOOK, M>(mapper, interrupt);
            if cycle == 321 {
                if self.is_rendering_enabled() {
                    self.oam_copy_buffer = self.secondary_sprite_ram[0];
                }
            } else if self.prev_rendering_enabled && (cycle == 328 || cycle == 336) {
                self.low_bit_shift <<= 8;
                self.high_bit_shift <<= 8;
                self.inc_horizontal_scrolling();
            }
        } else if (cycle == 337 || cycle == 339) && self.is_rendering_enabled() {
            let nt_addr = self.get_nametable_addr();
            self.tile.tile_addr =
                u16::from(self.read_vram::<VRAM_HOOK, M>(nt_addr, mapper, interrupt));
            // Odd-frame dot skip applies only to NTSC 2C02 timing.
            if self.scanline == -1
                && cycle == 339
                && (self.frame_count & 0x01) != 0
                && self.region == crate::cartridge::Region::Ntsc
            {
                self.cycle = 340;
            }
        }
    }

    /// Cycle-340 to cycle-0 scanline transition.
    fn process_scanline_first_cycle<const VRAM_HOOK: bool, M: Mapper + ?Sized>(
        &mut self,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) {
        self.cycle = 0;
        self.scanline += 1;
        if self.scanline > self.vblank_end as i16 {
            // End of frame: wrap to pre-render scanline.
            self.scanline = -1;
            // Reset so pre-render sprite fetches use dummy tiles.
            self.sprite_count = 0;
        }
        // Input is committed once per frame at the configured input scanline.
        if self.scanline == self.nmi_scanline as i16 {
            self.pending_input_commit = true;
        }
        // Pre-render scanline clears sprite overflow and sprite-0 hit.
        if self.scanline == -1 {
            self.status &= !0x60; // clear bits 5 (overflow) + 6 (sprite-0 hit)
            self.sprite0_hit_first_set_clock = 0;
            self.sprite0_hit_debug = Sprite0HitDebugSnapshot::default();
            self.capture_valid = 0;
            // Reset the batched grayscale cursor at the pre-render line.
            self.last_updated_pixel = -1;
        }
        // Unused NT fetches still drive the PPU bus for mapper IRQ timing.
        let skipped_odd_frame_scanline_zero = self.scanline == 0 && (self.frame_count & 0x01) != 0;
        if self.scanline < 240
            && self.scanline >= 0
            && self.prev_rendering_enabled
            && !skipped_odd_frame_scanline_zero
        {
            // Use the full tile address; truncation breaks pattern-table 1.
            let bus_addr =
                (self.tile.tile_addr << 4) | (self.vram_addr >> 12) | self.bg_pattern_addr();
            self.set_bus_address_cached::<VRAM_HOOK, M>(bus_addr & 0x3FFF, mapper, interrupt);
        } else if self.scanline == 240 {
            self.set_bus_address_cached::<VRAM_HOOK, M>(self.vram_addr & 0x3FFF, mapper, interrupt);
            // Flush remaining grayscale/emphasis before frame readout.
            self.update_grayscale_bits();
            self.frame_count = self.frame_count.wrapping_add(1);
        }
    }

    /// Set the PPU bus address and notify mappers using CPU-cycle time.
    pub fn set_bus_address<M: Mapper + ?Sized>(
        &mut self,
        addr: u16,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) {
        let mapper_has_vram_addr_hook = mapper.has_vram_addr_hook();
        if mapper_has_vram_addr_hook {
            self.set_bus_address_cached::<true, M>(addr, mapper, interrupt);
        } else {
            self.set_bus_address_cached::<false, M>(addr, mapper, interrupt);
        }
    }

    fn set_bus_address_cached<const VRAM_HOOK: bool, M: Mapper + ?Sized>(
        &mut self,
        addr: u16,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) {
        self.ppu_bus_address = addr;
        if VRAM_HOOK {
            // Mapper IRQ filters consume CPU-cycle time, not PPU master clock.
            mapper.notify_vram_addr(addr, self.mapper_cpu_cycle_count, interrupt);
        }
    }

    fn read_vram<const VRAM_HOOK: bool, M: Mapper + ?Sized>(
        &mut self,
        addr: u16,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) -> u8 {
        // Nametable fetches use mirrored CIRAM, not CHR-pattern mapper reads.
        self.set_bus_address_cached::<VRAM_HOOK, M>(addr, mapper, interrupt);
        match addr & 0x3fff {
            0x0000..=0x1fff => mapper.ppu_read(addr),
            0x2000..=0x3eff => mapper
                .ppu_read_nametable(addr, &self.nametable, interrupt)
                .unwrap_or_else(|| {
                    let index = nametable_index(addr, mapper.nametable_mirroring());
                    self.nametable[index]
                }),
            0x3f00..=0x3fff => self.palette_read(addr),
            _ => 0,
        }
    }

    /// Coarse-X scroll increment at each 8-pixel BG tile boundary.
    fn inc_horizontal_scrolling(&mut self) {
        let mut addr = self.vram_addr;
        if (addr & 0x001F) == 31 {
            // Coarse X = 31: wrap to 0, toggle nametable horizontally.
            addr = (addr & !0x001F) ^ 0x0400;
        } else {
            addr += 1;
        }
        self.vram_addr = addr;
    }

    /// Fine-Y and coarse-Y scroll increment at cycle 256.
    fn inc_vertical_scrolling(&mut self) {
        let mut addr = self.vram_addr;
        if (addr & 0x7000) != 0x7000 {
            // Fine Y < 7: increment.
            addr += 0x1000;
        } else {
            // Fine Y = 7: reset, then handle coarse Y wraparound.
            addr &= !0x7000;
            let mut y = (addr & 0x03E0) >> 5;
            if y == 29 {
                // Coarse Y = 29: wrap to 0, toggle nametable vertically.
                y = 0;
                addr ^= 0x0800;
            } else if y == 31 {
                // Coarse Y = 31 (attribute table): wrap to 0, no toggle.
                y = 0;
            } else {
                y += 1;
            }
            addr = (addr & !0x03E0) | (y << 5);
        }
        self.vram_addr = addr;
    }

    /// Compute the current nametable byte fetch address.
    fn get_nametable_addr(&self) -> u16 {
        0x2000 | (self.vram_addr & 0x0FFF)
    }

    /// Compute the current attribute table byte fetch address.
    fn get_attribute_addr(&self) -> u16 {
        0x23C0
            | (self.vram_addr & 0x0C00)
            | ((self.vram_addr >> 4) & 0x38)
            | ((self.vram_addr >> 2) & 0x07)
    }

    /// Read the delayed rendering-enabled state.
    fn is_rendering_enabled(&self) -> bool {
        self.rendering_enabled
    }

    /// BG sprite-pattern address selector. Bit 4 of `$2000` selects
    /// `$0000` or `$1000` as the BG pattern table base.
    fn bg_pattern_addr(&self) -> u16 {
        if self.ctrl & 0x10 != 0 {
            0x1000
        } else {
            0x0000
        }
    }

    /// BG tile fetch state machine for visible and prefetch windows.
    fn load_tile_info<const VRAM_HOOK: bool, M: Mapper + ?Sized>(
        &mut self,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) {
        if !self.is_rendering_enabled() {
            return;
        }
        match self.cycle & 0x07 {
            1 => {
                self.previous_tile_palette = self.current_tile_palette;
                self.current_tile_palette = self.tile.palette_offset;
                self.low_bit_shift |= u16::from(self.tile.low_byte);
                self.high_bit_shift |= u16::from(self.tile.high_byte);
                let nt_addr = self.get_nametable_addr();
                let tile_index = self.read_vram::<VRAM_HOOK, M>(nt_addr, mapper, interrupt);
                self.tile.tile_addr =
                    (u16::from(tile_index) << 4) | (self.vram_addr >> 12) | self.bg_pattern_addr();
            }
            3 => {
                let at_addr = self.get_attribute_addr();
                let at_byte = self.read_vram::<VRAM_HOOK, M>(at_addr, mapper, interrupt);
                let shift = ((self.vram_addr >> 4) & 0x04) | (self.vram_addr & 0x02);
                self.tile.palette_offset = ((at_byte >> shift) & 0x03) << 2;
            }
            5 => {
                self.tile.low_byte =
                    self.read_vram::<VRAM_HOOK, M>(self.tile.tile_addr, mapper, interrupt);
            }
            7 => {
                self.tile.high_byte =
                    self.read_vram::<VRAM_HOOK, M>(self.tile.tile_addr + 8, mapper, interrupt);
            }
            _ => {}
        }
    }

    /// Shift BG pattern registers by one output pixel.
    fn shift_tile_registers(&mut self) {
        self.low_bit_shift <<= 1;
        self.high_bit_shift <<= 1;
    }

    /// Whether sprites are 8x16 instead of 8x8. Bit 5 of `$2000`.
    fn large_sprites(&self) -> bool {
        self.ctrl & 0x20 != 0
    }

    /// 8x8 sprite pattern table base. Bit 3 of `$2000` selects `$1000`
    /// vs `$0000`. (Only used for 8x8 sprites; 8x16 sprites encode the
    /// table in the tile index LSB.)
    fn sprite_pattern_addr(&self) -> u16 {
        if self.ctrl & 0x08 != 0 {
            0x1000
        } else {
            0x0000
        }
    }

    /// Enable the RL-only no-flicker render path.
    pub fn set_remove_sprite_limit(&mut self, on: bool) {
        self.remove_sprite_limit = on;
    }

    /// Enable pure framebuffer output while keeping PPU timing active.
    pub fn set_render_enabled(&mut self, on: bool) {
        self.render_enabled = on;
    }

    /// Initialize sprite evaluation at cycle 65.
    fn process_sprite_evaluation_start(&mut self) {
        self.sprite0_added = false;
        self.sprite_in_range = false;
        self.secondary_oam_addr = 0;
        self.overflow_bug_counter = 0;
        self.oam_copy_done = false;
        self.sprite_addr_h = (self.oam_addr >> 2) & 0x3F;
        self.sprite_addr_l = self.oam_addr & 0x03;
        // Bracket this scanline's visible-sprite range for RL no-flicker.
        self.first_visible_sprite_addr = self.sprite_addr_h * 4;
        self.last_visible_sprite_addr = self.first_visible_sprite_addr;
    }

    /// Finalize sprite evaluation at cycle 256.
    fn process_sprite_evaluation_end(&mut self) {
        self.sprite0_visible = self.sprite0_added;
        self.sprite_count = (self.secondary_oam_addr + 3) >> 2;
    }

    /// Per-cycle sprite evaluation for visible scanline cycles 1..256.
    fn process_sprite_evaluation(&mut self) {
        if !self.is_rendering_enabled() {
            return;
        }
        let cycle = self.cycle;
        if cycle < 65 {
            // Cycles 1..64: clear secondary OAM (one byte per even cycle).
            self.oam_copy_buffer = 0xFF;
            self.secondary_sprite_ram[((cycle - 1) >> 1) as usize] = 0xFF;
            return;
        }
        // Odd cycles read OAM, even cycles evaluate.
        if cycle & 0x01 != 0 {
            if cycle == 65 {
                self.process_sprite_evaluation_start();
            }
            // Odd cycle: read OAM byte at the eval cursor.
            self.oam_copy_buffer = self.oam[self.oam_addr as usize];
            return;
        }
        // Even cycle: evaluate and copy.
        if cycle == 256 {
            self.process_sprite_evaluation_end();
        }
        if self.oam_copy_done {
            // The optional early-2C02 sprite-eval bug is disabled in the oracle.
            self.sprite_addr_h = (self.sprite_addr_h + 1) & 0x3F;
            if self.secondary_oam_addr >= 0x20 {
                self.oam_copy_buffer =
                    self.secondary_sprite_ram[(self.secondary_oam_addr & 0x1F) as usize];
            }
            self.oam_addr = (self.sprite_addr_l & 0x03) | (self.sprite_addr_h << 2);
            return;
        }
        // Check if current Y coordinate (in copy buffer) is in range.
        let sprite_height: i16 = if self.large_sprites() { 16 } else { 8 };
        if !self.sprite_in_range
            && self.scanline >= i16::from(self.oam_copy_buffer)
            && self.scanline < i16::from(self.oam_copy_buffer) + sprite_height
        {
            self.sprite_in_range = true;
        }
        if self.secondary_oam_addr < 0x20 {
            // Copy one byte to secondary OAM.
            self.secondary_sprite_ram[self.secondary_oam_addr as usize] = self.oam_copy_buffer;
            if self.sprite_in_range {
                if cycle == 66 {
                    // Mesen2: first Y in range marks sprite-0 detection.
                    self.sprite0_added = true;
                }
                self.sprite_addr_l += 1;
                self.secondary_oam_addr += 1;
                if self.sprite_addr_l >= 4 {
                    self.sprite_addr_h = (self.sprite_addr_h + 1) & 0x3F;
                    self.sprite_addr_l = 0;
                    if self.sprite_addr_h == 0 {
                        self.oam_copy_done = true;
                    }
                }
                if (self.secondary_oam_addr & 0x03) == 0 {
                    // Mid-sprite resync for non-4-aligned OAMADDR starts.
                    self.sprite_in_range = false;
                    self.last_visible_sprite_addr =
                        self.sprite_addr_h.wrapping_sub(1).wrapping_mul(4);
                    if self.sprite_addr_l != 0 {
                        let sprite_height: i16 = if self.large_sprites() { 16 } else { 8 };
                        let in_range = self.scanline >= i16::from(self.oam_copy_buffer)
                            && self.scanline < i16::from(self.oam_copy_buffer) + sprite_height;
                        if !in_range {
                            self.sprite_addr_l = 0;
                        }
                    }
                }
            } else {
                // Not in range: skip to next sprite.
                self.sprite_addr_h = (self.sprite_addr_h + 1) & 0x3F;
                self.sprite_addr_l = 0;
                if self.sprite_addr_h == 0 {
                    self.oam_copy_done = true;
                }
            }
        } else {
            // Secondary OAM full: overflow detection.
            self.oam_copy_buffer =
                self.secondary_sprite_ram[(self.secondary_oam_addr & 0x1F) as usize];
            if self.sprite_in_range {
                // Set overflow flag (status bit 5).
                self.status |= 0x20;
                self.sprite_addr_l += 1;
                if self.sprite_addr_l == 4 {
                    self.sprite_addr_h = (self.sprite_addr_h + 1) & 0x3F;
                    self.sprite_addr_l = 0;
                }
                if self.overflow_bug_counter == 0 {
                    self.overflow_bug_counter = 3;
                } else {
                    self.overflow_bug_counter -= 1;
                    if self.overflow_bug_counter == 0 {
                        self.oam_copy_done = true;
                        self.sprite_addr_l = 0;
                    }
                }
            } else {
                // Hardware bug: increments both H and L.
                self.sprite_addr_h = (self.sprite_addr_h + 1) & 0x3F;
                self.sprite_addr_l = (self.sprite_addr_l + 1) & 0x03;
                if self.sprite_addr_h == 0 {
                    self.oam_copy_done = true;
                }
            }
        }
        self.oam_addr = (self.sprite_addr_l & 0x03) | (self.sprite_addr_h << 2);
    }

    /// Fetch one sprite pattern and mark its pixel coverage.
    fn load_sprite<const VRAM_HOOK: bool, M: Mapper + ?Sized>(
        &mut self,
        sprite: SpriteLoad,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) {
        let background_priority = sprite.attr & 0x20 != 0;
        let horizontal_mirror = sprite.attr & 0x40 != 0;
        let vertical_mirror = sprite.attr & 0x80 != 0;
        let large = self.large_sprites();
        let max_offset: i16 = if large { 15 } else { 7 };
        let line_offset = if vertical_mirror {
            max_offset - (self.scanline - i16::from(sprite.y))
        } else {
            self.scanline - i16::from(sprite.y)
        };
        // Line offset wraps through u8, including stale out-of-range fetches.
        let line_offset = u16::from(line_offset as u8);
        let tile_addr = if large {
            // 8x16 sprite: tile_index bit 0 selects $0000/$1000 base.
            let base: u16 = if sprite.tile & 0x01 != 0 {
                0x1000
            } else {
                0x0000
            };
            let tile_no = (u16::from(sprite.tile) & !0x01) << 4;
            let line = if line_offset >= 8 {
                line_offset + 8
            } else {
                line_offset
            };
            // Row is added, not ORed; stale offsets can otherwise diverge.
            (base | tile_no) + line
        } else {
            ((u16::from(sprite.tile)) << 4 | self.sprite_pattern_addr()) + line_offset
        };
        let idx = self.sprite_index as usize;
        let visible =
            (self.sprite_index < u32::from(self.sprite_count) || sprite.extra) && sprite.y < 240;
        let mut low_byte = 0;
        let mut high_byte = 0;
        if visible {
            if sprite.extra {
                // Extra sprites use side-effect-free CHR reads.
                low_byte = mapper.debug_ppu_read(tile_addr);
                high_byte = mapper.debug_ppu_read(tile_addr + 8);
            } else {
                // Real VRAM fetch (advances mapper A12 state).
                low_byte = self.read_vram::<VRAM_HOOK, M>(tile_addr, mapper, interrupt);
                high_byte = self.read_vram::<VRAM_HOOK, M>(tile_addr + 8, mapper, interrupt);
            }
        } else {
            // Dummy $FF fetch keeps mapper A12 timing aligned.
            let dummy_addr: u16 = if large {
                0x1000 | (0xFF << 4)
            } else {
                self.sprite_pattern_addr() | (0xFF << 4)
            };
            let _ = self.read_vram::<VRAM_HOOK, M>(dummy_addr, mapper, interrupt);
            let _ = self.read_vram::<VRAM_HOOK, M>(dummy_addr + 8, mapper, interrupt);
        }
        if visible {
            let info = &mut self.sprite_tiles[idx];
            info.background_priority = background_priority;
            info.horizontal_mirror = horizontal_mirror;
            info.palette_offset = ((sprite.attr & 0x03) << 2) | 0x10;
            info.low_byte = low_byte;
            info.high_byte = high_byte;
            info.sprite_x = sprite.x;
            if self.scanline >= 0 {
                let x = usize::from(sprite.x);
                for i in 0..8 {
                    let target = x + i + 1;
                    if target < 257 {
                        self.has_sprite[target] = true;
                    }
                }
            }
        }
        self.sprite_index += 1;
    }

    /// Load tile data for the current secondary-OAM sprite.
    fn load_sprite_tile_info<const VRAM_HOOK: bool, M: Mapper + ?Sized>(
        &mut self,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) {
        let base = (self.sprite_index * 4) as usize;
        let y = self.secondary_sprite_ram[base];
        let tile = self.secondary_sprite_ram[base + 1];
        let attr = self.secondary_sprite_ram[base + 2];
        let x = self.secondary_sprite_ram[base + 3];
        self.load_sprite::<VRAM_HOOK, M>(
            SpriteLoad {
                y,
                tile,
                attr,
                x,
                extra: false,
            },
            mapper,
            interrupt,
        );
    }

    /// RL no-flicker pass: render 9th+ sprites without mapper side effects.
    fn load_extra_sprites<const VRAM_HOOK: bool, M: Mapper + ?Sized>(
        &mut self,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) {
        if self.sprite_count != 8 || !self.remove_sprite_limit {
            return;
        }
        let height: i16 = if self.large_sprites() { 16 } else { 8 };
        let first = (self.first_visible_sprite_addr & 0xFC) as usize;
        let mut i = (usize::from(self.last_visible_sprite_addr) + 4) & 0xFC;
        while i != first {
            if self.sprite_index >= 64 {
                break; // sprite_tiles has 64 slots (>=64 in-range sprites is impossible)
            }
            let sprite_y = self.oam[i];
            if self.scanline >= i16::from(sprite_y) && self.scanline < i16::from(sprite_y) + height
            {
                let tile = self.oam[i + 1];
                let attr = self.oam[i + 2];
                let x = self.oam[i + 3];
                self.load_sprite::<VRAM_HOOK, M>(
                    SpriteLoad {
                        y: sprite_y,
                        tile,
                        attr,
                        x,
                        extra: true,
                    },
                    mapper,
                    interrupt,
                );
                self.sprite_count += 1;
            }
            i = (i + 4) & 0xFC;
        }
    }

    /// Compute the current visible-cycle palette index.
    fn get_pixel_color(&mut self) -> u8 {
        let offset = self.fine_x;
        let mut background_color: u8 = 0;
        let mut sprite_bg_color: u8 = 0;
        if self.cycle > self.minimum_draw_bg_cycle {
            sprite_bg_color = ((((self.low_bit_shift << offset) & 0x8000) >> 15)
                | (((self.high_bit_shift << offset) & 0x8000) >> 14))
                as u8;
            // BG enabled in mask?
            if self.mask & 0x08 != 0 {
                background_color = sprite_bg_color;
            }
        }
        let cycle_idx = self.cycle as usize;
        if self.has_sprite[cycle_idx] && self.cycle > self.minimum_draw_sprite_cycle {
            for i in 0..(self.sprite_count as usize).min(64) {
                let info = self.sprite_tiles[i];
                let shift: i32 = i32::from(self.cycle) - i32::from(info.sprite_x) - 1;
                if (0..8).contains(&shift) {
                    let sprite_color: u8 = if info.horizontal_mirror {
                        ((info.low_byte >> shift) & 0x01)
                            | (((info.high_byte >> shift) & 0x01) << 1)
                    } else {
                        (((info.low_byte << shift) & 0x80) >> 7)
                            | (((info.high_byte << shift) & 0x80) >> 6)
                    };
                    if sprite_color != 0 {
                        // Sprite-0 hit detection.
                        if i == 0
                            && sprite_bg_color != 0
                            && self.sprite0_visible
                            && self.cycle != 256
                            && self.mask & 0x08 != 0
                            && self.status & 0x40 == 0
                            && self.cycle > self.minimum_draw_sprite_standard_cycle
                        {
                            self.status |= 0x40;
                            // Capture CPU cycle at the first 0->1 transition.
                            if self.sprite0_hit_first_set_clock == 0 {
                                self.sprite0_hit_first_set_clock = self.mapper_cpu_cycle_count;
                                self.sprite0_hit_debug = Sprite0HitDebugSnapshot {
                                    valid: true,
                                    cpu_cycle: self.mapper_cpu_cycle_count,
                                    scanline: self.scanline,
                                    ppu_cycle: self.cycle,
                                    mask: self.mask,
                                    sprite_count: self.sprite_count,
                                    sprite0_visible: self.sprite0_visible,
                                    sprite_bg_color,
                                    sprite_color,
                                    minimum_draw_sprite_standard_cycle: self
                                        .minimum_draw_sprite_standard_cycle,
                                    sprite0_x: info.sprite_x,
                                    sprite0_low: info.low_byte,
                                    sprite0_high: info.high_byte,
                                    sprite0_hm: info.horizontal_mirror,
                                    sprite0_bg_pri: info.background_priority,
                                    sprite0_pal: info.palette_offset,
                                    low_bit_shift: self.low_bit_shift,
                                    high_bit_shift: self.high_bit_shift,
                                    fine_x: self.fine_x,
                                };
                            }
                        }
                        // Sprite enabled in mask + priority check.
                        if self.mask & 0x10 != 0
                            && (background_color == 0 || !info.background_priority)
                        {
                            return info.palette_offset + sprite_color;
                        }
                        break;
                    }
                }
            }
        }
        // BG pixel -apply previous-tile palette when fine_x scrolls
        // across the tile boundary.
        let palette = if (offset as u16 + ((self.cycle - 1) & 0x07)) < 8 {
            self.previous_tile_palette
        } else {
            self.current_tile_palette
        };
        palette + background_color
    }

    pub fn cpu_read_register<M: Mapper + ?Sized>(
        &mut self,
        addr: u16,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) -> u8 {
        match addr & 0x2007 {
            0x2002 => {
                let status = self.status & 0xe0;
                self.status &= 0x7f;
                self.write_latch = false;
                // `$2002` at VBlank edge suppresses VBL for the frame.
                if self.scanline == self.nmi_scanline as i16 && self.cycle == 0 {
                    self.prevent_vbl_flag = true;
                }
                interrupt.clear_nmi_flag();
                self.apply_ppu_open_bus(0x1f, status)
            }
            0x2004 => {
                let value = if self.scanline <= 239 && self.is_rendering_enabled() {
                    if (257..=320).contains(&self.cycle) {
                        let step = ((self.cycle - 257) % 8).min(3);
                        self.secondary_oam_addr = (((self.cycle - 257) / 8) * 4 + step) as u8;
                        self.oam_copy_buffer =
                            self.secondary_sprite_ram[self.secondary_oam_addr as usize];
                    }
                    self.oam_copy_buffer
                } else {
                    self.oam[usize::from(self.oam_addr)]
                };
                self.apply_ppu_open_bus(0x00, value)
            }
            0x2007 => {
                let latched = self.vram_addr;
                self.cpu_read_data_register_at(latched, mapper, interrupt)
            }
            _ => self.apply_ppu_open_bus(0xff, 0),
        }
    }

    /// Read `$2007`, applying buffered-data and post-read increment semantics.
    pub fn cpu_read_data_register_at<M: Mapper + ?Sized>(
        &mut self,
        addr: u16,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) -> u8 {
        if self.ignore_vram_read > 0 {
            return self.apply_ppu_open_bus(0xff, 0);
        }

        let addr = addr & 0x3fff;
        let value = self.ppu_read(addr, mapper, interrupt);
        let result = if addr < 0x3f00 {
            let buffered = self.data_buffer;
            self.data_buffer = value;
            self.apply_ppu_open_bus(0x00, buffered)
        } else {
            let value = if self.mask & 0x01 != 0 {
                value & 0x30
            } else {
                value
            };
            let result = value | (self.ppu_open_bus & 0xc0);
            // Palette reads refill the buffer from the latched PPU bus address.
            self.data_buffer = self.ppu_read(self.ppu_bus_address & 0x3FFF, mapper, interrupt);
            self.apply_ppu_open_bus(0xc0, result)
        };
        self.ignore_vram_read = 6;
        self.need_video_ram_increment = true;
        self.need_state_update = true;
        result
    }

    fn process_tmp_addr_scroll_glitch(&mut self, mask: u16, value: u16) {
        // The oracle keeps this optional cycle-257 scroll glitch disabled.
        const ENABLE_PPU_2000_SCROLL_GLITCH: bool = false;
        if ENABLE_PPU_2000_SCROLL_GLITCH
            && self.scanline < 240
            && self.cycle == 257
            && self.is_rendering_enabled()
        {
            self.vram_addr = (self.vram_addr & !mask) | (value & mask);
        }
    }

    fn set_ppu_open_bus(&mut self, mask: u8, mut value: u8) {
        if mask == 0xff {
            self.ppu_open_bus = value;
            self.ppu_open_bus_decay_stamp = [self.frame_count; 8];
            self.io_latch = self.ppu_open_bus;
            return;
        }
        let mut next = self.ppu_open_bus;
        let mut bit_mask = mask;
        for bit in 0..8 {
            let bit_value = 1u8 << bit;
            if bit_mask & 0x01 != 0 {
                if value & 0x01 != 0 {
                    next |= bit_value;
                } else {
                    next &= !bit_value;
                }
                self.ppu_open_bus_decay_stamp[bit] = self.frame_count;
            } else if self
                .frame_count
                .saturating_sub(self.ppu_open_bus_decay_stamp[bit])
                > 3
            {
                next &= !bit_value;
            }
            value >>= 1;
            bit_mask >>= 1;
        }
        self.ppu_open_bus = next;
        self.io_latch = self.ppu_open_bus;
    }

    fn apply_ppu_open_bus(&mut self, mask: u8, value: u8) -> u8 {
        self.set_ppu_open_bus(!mask, value);
        value | (self.ppu_open_bus & mask)
    }

    pub fn cpu_write_register<M: Mapper + ?Sized>(
        &mut self,
        addr: u16,
        value: u8,
        open_bus: u8,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) {
        self.set_ppu_open_bus(0xff, value);
        match addr & 0x2007 {
            0x2000 => {
                self.ctrl = value;
                self.temp_addr = (self.temp_addr & 0xf3ff) | ((u16::from(value) & 0x03) << 10);
                self.process_tmp_addr_scroll_glitch(0x0400, u16::from(open_bus) << 10);
                // NMI is level-sensitive across `$2000` rewrites during VBlank.
                if value & 0x80 == 0 {
                    interrupt.clear_nmi_flag();
                } else if self.status & 0x80 != 0 {
                    interrupt.set_nmi_flag();
                }
            }
            0x2001 => {
                self.mask = value;
                // `$2001` immediately updates leftmost masks.
                self.update_minimum_draw_cycles();
                // Flush prior pixels before adopting the new grayscale/emphasis bits.
                self.update_grayscale_bits();
                // Queue state recomputation only when rendering enable changes.
                let raw_rendering_enabled = (value & 0x18) != 0;
                if self.rendering_enabled != raw_rendering_enabled {
                    self.need_state_update = true;
                }
            }
            // `$2003` also updates the secondary `$2004` read pointer.
            0x2003 => {
                self.oam_addr = value;
                self.ppu_spl = value & 0x07;
            }
            // `$2004` writes are blocked during rendering.
            0x2004 => {
                if self.scanline >= 240 || !self.is_rendering_enabled() {
                    let value = if self.oam_addr & 0x03 == 0x02 {
                        value & 0xE3
                    } else {
                        value
                    };
                    self.oam[usize::from(self.oam_addr)] = value;
                    self.oam_addr = self.oam_addr.wrapping_add(1);
                } else {
                    self.oam_addr = self.oam_addr.wrapping_add(4);
                }
            }
            0x2005 => {
                if self.write_latch {
                    self.temp_addr = (self.temp_addr & 0x8fff) | ((u16::from(value) & 0x07) << 12);
                    self.temp_addr = (self.temp_addr & 0xfc1f) | ((u16::from(value) & 0xf8) << 2);
                } else {
                    self.fine_x = value & 0x07;
                    self.temp_addr = (self.temp_addr & 0xffe0) | (u16::from(value) >> 3);
                    self.process_tmp_addr_scroll_glitch(0x001f, u16::from(open_bus >> 3));
                }
                self.write_latch = !self.write_latch;
            }
            0x2006 => {
                if self.write_latch {
                    self.temp_addr = (self.temp_addr & 0xff00) | u16::from(value);
                    // `$2006` second write applies after a 3-PPU-cycle delay.
                    self.update_vram_addr = self.temp_addr;
                    self.update_vram_addr_delay = 3;
                    self.need_state_update = true;
                } else {
                    self.temp_addr = (self.temp_addr & 0x00ff) | ((u16::from(value) & 0x3f) << 8);
                    self.process_tmp_addr_scroll_glitch(0x0c00, u16::from(open_bus) << 8);
                }
                self.write_latch = !self.write_latch;
            }
            0x2007 => {
                // `$2007` targets the latched bus address; rendering
                // non-palette writes substitute the address low byte.
                let addr = self.ppu_bus_address & 0x3fff;
                let write_value =
                    if addr >= 0x3f00 || self.scanline >= 240 || !self.is_rendering_enabled() {
                        value
                    } else {
                        (self.ppu_bus_address & 0xff) as u8
                    };
                self.ppu_write(addr, write_value, mapper);
                self.need_video_ram_increment = true;
                self.need_state_update = true;
            }
            _ => {}
        }
    }

    #[cold]
    pub fn snapshot_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4096);
        bytes.extend_from_slice(&[self.ctrl, self.mask, self.status, self.oam_addr]);
        bytes.extend_from_slice(&self.oam);
        bytes.extend_from_slice(&self.vram_addr.to_le_bytes());
        bytes.extend_from_slice(&self.temp_addr.to_le_bytes());
        bytes.extend_from_slice(&[self.fine_x, u8::from(self.write_latch), self.data_buffer]);
        bytes.push(self.io_latch);
        bytes.extend_from_slice(&self.nametable);
        bytes.extend_from_slice(&self.palette);
        bytes.extend_from_slice(&self.upper_palette_aliases);
        bytes.push(self.odd_frame_toggle);
        bytes.push(self.ppu_spl);
        bytes.extend_from_slice(&self.master_clock.to_le_bytes());
        bytes.push(self.master_clock_divider);
        bytes.extend_from_slice(&self.mapper_cpu_cycle_count.to_le_bytes());
        bytes.extend_from_slice(&self.scanline.to_le_bytes());
        bytes.extend_from_slice(&self.cycle.to_le_bytes());
        bytes.extend_from_slice(&self.frame_count.to_le_bytes());
        bytes.extend_from_slice(&self.nmi_scanline.to_le_bytes());
        bytes.extend_from_slice(&self.vblank_end.to_le_bytes());
        // Region byte: 0=Ntsc, 1=Pal, 2=Dendy.
        bytes.push(match self.region {
            crate::cartridge::Region::Ntsc => 0,
            crate::cartridge::Region::Pal => 1,
            crate::cartridge::Region::Dendy => 2,
        });
        bytes.push(self.tile.low_byte);
        bytes.push(self.tile.high_byte);
        bytes.push(self.tile.palette_offset);
        bytes.extend_from_slice(&self.tile.tile_addr.to_le_bytes());
        for sprite in self.sprite_tiles {
            bytes.extend_from_slice(&[
                sprite.sprite_x,
                sprite.low_byte,
                sprite.high_byte,
                sprite.palette_offset,
                u8::from(sprite.horizontal_mirror),
                u8::from(sprite.background_priority),
            ]);
        }
        bytes.push(self.sprite_count);
        for has_sprite in self.has_sprite {
            bytes.push(u8::from(has_sprite));
        }
        bytes.extend_from_slice(&[u8::from(self.sprite0_visible), u8::from(self.sprite0_added)]);
        bytes.extend_from_slice(&self.low_bit_shift.to_le_bytes());
        bytes.extend_from_slice(&self.high_bit_shift.to_le_bytes());
        bytes.extend_from_slice(&self.ppu_bus_address.to_le_bytes());
        bytes.extend_from_slice(&[
            u8::from(self.prevent_vbl_flag),
            u8::from(self.rendering_enabled),
            u8::from(self.prev_rendering_enabled),
            u8::from(self.need_state_update),
        ]);
        bytes.extend_from_slice(&self.secondary_sprite_ram);
        bytes.extend_from_slice(&[
            self.secondary_oam_addr,
            self.oam_copy_buffer,
            u8::from(self.sprite_in_range),
            self.sprite_addr_h,
            self.sprite_addr_l,
            self.overflow_bug_counter,
            u8::from(self.oam_copy_done),
            self.last_visible_sprite_addr,
        ]);
        bytes.extend_from_slice(&self.sprite_index.to_le_bytes());
        bytes.extend_from_slice(&[self.current_tile_palette, self.previous_tile_palette]);
        bytes.extend_from_slice(&self.minimum_draw_bg_cycle.to_le_bytes());
        bytes.extend_from_slice(&self.minimum_draw_sprite_cycle.to_le_bytes());
        bytes.extend_from_slice(&self.minimum_draw_sprite_standard_cycle.to_le_bytes());
        bytes.push(self.update_vram_addr_delay);
        bytes.extend_from_slice(&self.update_vram_addr.to_le_bytes());
        bytes.push(u8::from(self.need_video_ram_increment));
        bytes.extend_from_slice(&self.ignore_vram_read.to_le_bytes());
        bytes.push(self.ppu_open_bus);
        for stamp in self.ppu_open_bus_decay_stamp {
            bytes.extend_from_slice(&stamp.to_le_bytes());
        }
        bytes
    }

    #[cold]
    pub fn restore_snapshot(&mut self, bytes: &[u8]) -> nesle_common::Result<()> {
        let mut offset = 0;
        macro_rules! take {
            ($n:expr) => {{
                let slice = &bytes[offset..offset + $n];
                offset += $n;
                slice
            }};
        }
        self.ctrl = take!(1)[0];
        self.mask = take!(1)[0];
        // Transient render masks are recomputed from restored mask + region.
        self.last_updated_pixel = -1;
        self.status = take!(1)[0];
        self.oam_addr = take!(1)[0];
        self.oam.copy_from_slice(take!(256));
        self.vram_addr = u16::from_le_bytes(take!(2).try_into().unwrap());
        self.temp_addr = u16::from_le_bytes(take!(2).try_into().unwrap());
        self.fine_x = take!(1)[0];
        self.write_latch = take!(1)[0] != 0;
        self.data_buffer = take!(1)[0];
        self.io_latch = take!(1)[0];
        self.nametable.copy_from_slice(take!(0x1000));
        self.palette.copy_from_slice(take!(32));
        self.upper_palette_aliases.copy_from_slice(take!(3));
        self.odd_frame_toggle = take!(1)[0];
        self.ppu_spl = take!(1)[0];
        self.master_clock = u64::from_le_bytes(take!(8).try_into().unwrap());
        self.master_clock_divider = take!(1)[0];
        self.mapper_cpu_cycle_count = u64::from_le_bytes(take!(8).try_into().unwrap());
        self.scanline = i16::from_le_bytes(take!(2).try_into().unwrap());
        self.cycle = u16::from_le_bytes(take!(2).try_into().unwrap());
        self.frame_count = u32::from_le_bytes(take!(4).try_into().unwrap());
        self.nmi_scanline = u16::from_le_bytes(take!(2).try_into().unwrap());
        self.vblank_end = u16::from_le_bytes(take!(2).try_into().unwrap());
        // Unknown region encodings fall back to NTSC.
        self.region = match take!(1)[0] {
            1 => crate::cartridge::Region::Pal,
            2 => crate::cartridge::Region::Dendy,
            _ => crate::cartridge::Region::Ntsc,
        };
        // Recompute color-bit masks from restored `$2001` and region.
        self.update_color_bit_masks();
        self.tile.low_byte = take!(1)[0];
        self.tile.high_byte = take!(1)[0];
        self.tile.palette_offset = take!(1)[0];
        self.tile.tile_addr = u16::from_le_bytes(take!(2).try_into().unwrap());
        for sprite in &mut self.sprite_tiles {
            sprite.sprite_x = take!(1)[0];
            sprite.low_byte = take!(1)[0];
            sprite.high_byte = take!(1)[0];
            sprite.palette_offset = take!(1)[0];
            sprite.horizontal_mirror = take!(1)[0] != 0;
            sprite.background_priority = take!(1)[0] != 0;
        }
        self.sprite_count = take!(1)[0];
        for has_sprite in &mut self.has_sprite {
            *has_sprite = take!(1)[0] != 0;
        }
        self.sprite0_visible = take!(1)[0] != 0;
        self.sprite0_added = take!(1)[0] != 0;
        self.low_bit_shift = u16::from_le_bytes(take!(2).try_into().unwrap());
        self.high_bit_shift = u16::from_le_bytes(take!(2).try_into().unwrap());
        self.ppu_bus_address = u16::from_le_bytes(take!(2).try_into().unwrap());
        self.prevent_vbl_flag = take!(1)[0] != 0;
        self.rendering_enabled = take!(1)[0] != 0;
        self.prev_rendering_enabled = take!(1)[0] != 0;
        self.need_state_update = take!(1)[0] != 0;
        self.secondary_sprite_ram.copy_from_slice(take!(32));
        self.secondary_oam_addr = take!(1)[0];
        self.oam_copy_buffer = take!(1)[0];
        self.sprite_in_range = take!(1)[0] != 0;
        self.sprite_addr_h = take!(1)[0];
        self.sprite_addr_l = take!(1)[0];
        self.overflow_bug_counter = take!(1)[0];
        self.oam_copy_done = take!(1)[0] != 0;
        self.last_visible_sprite_addr = take!(1)[0];
        self.sprite_index = u32::from_le_bytes(take!(4).try_into().unwrap());
        self.current_tile_palette = take!(1)[0];
        self.previous_tile_palette = take!(1)[0];
        self.minimum_draw_bg_cycle = u16::from_le_bytes(take!(2).try_into().unwrap());
        self.minimum_draw_sprite_cycle = u16::from_le_bytes(take!(2).try_into().unwrap());
        self.minimum_draw_sprite_standard_cycle = u16::from_le_bytes(take!(2).try_into().unwrap());
        self.update_vram_addr_delay = take!(1)[0];
        self.update_vram_addr = u16::from_le_bytes(take!(2).try_into().unwrap());
        self.need_video_ram_increment = take!(1)[0] != 0;
        self.ignore_vram_read = u32::from_le_bytes(take!(4).try_into().unwrap());
        self.ppu_open_bus = take!(1)[0];
        for stamp in &mut self.ppu_open_bus_decay_stamp {
            *stamp = u32::from_le_bytes(take!(4).try_into().unwrap());
        }
        if offset != bytes.len() {
            return Err(nesle_common::NesleError::InvalidState(format!(
                "PPU snapshot trailing length mismatch: parsed {offset} bytes, got {}",
                bytes.len()
            )));
        }
        // Restore uses conservative sprite coverage until the next eval pass.
        self.has_sprite = [true; 257];
        Ok(())
    }

    fn ppu_read<M: Mapper + ?Sized>(
        &mut self,
        addr: u16,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
    ) -> u8 {
        match addr & 0x3fff {
            0x0000..=0x1fff => mapper.ppu_read(addr),
            0x2000..=0x3eff => mapper
                .ppu_read_nametable(addr, &self.nametable, interrupt)
                .unwrap_or_else(|| {
                    let index = nametable_index(addr, mapper.nametable_mirroring());
                    self.nametable[index]
                }),
            0x3f00..=0x3fff => self.palette_read(addr),
            _ => 0,
        }
    }

    fn ppu_write<M: Mapper + ?Sized>(&mut self, addr: u16, value: u8, mapper: &mut M) {
        let addr = addr & 0x3fff;
        if addr <= 0x1fff {
            mapper.ppu_write(addr, value);
        } else if addr <= 0x3eff {
            if !mapper.ppu_write_nametable(addr, value, &mut self.nametable) {
                let index = nametable_index(addr, mapper.nametable_mirroring());
                self.nametable[index] = value;
            }
        } else {
            self.palette_write(addr, value);
        }
    }

    fn palette_read(&self, addr: u16) -> u8 {
        let mut index = usize::from((addr - 0x3f00) & 0x001f);
        if index >= 0x10 && index & 0x03 == 0 {
            index -= 0x10;
        }
        self.palette[index]
    }

    fn palette_write(&mut self, addr: u16, value: u8) {
        let value = value & 0x3f;
        let index = usize::from((addr - 0x3f00) & 0x001f);
        self.palette[index] = value;
        if index & 0x03 == 0 {
            self.palette[index ^ 0x10] = value;
        }
    }
}

fn nametable_index(addr: u16, mirroring: Mirroring) -> usize {
    let offset = usize::from((addr - 0x2000) & 0x0fff);
    let table = offset / 0x400;
    let inner = offset & 0x03ff;
    match mirroring {
        Mirroring::Horizontal => match table {
            0 | 1 => inner,
            _ => 0x400 + inner,
        },
        Mirroring::Vertical => match table {
            0 | 2 => inner,
            _ => 0x400 + inner,
        },
        Mirroring::SingleScreenLower => inner,
        Mirroring::SingleScreenUpper => 0x400 + inner,
        Mirroring::FourScreen => offset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestMapper {
        chr: [u8; 0x2000],
    }

    impl Mapper for TestMapper {
        fn mapper_id(&self) -> u16 {
            0
        }

        fn name(&self) -> &'static str {
            "TEST"
        }

        fn cpu_read(&mut self, _addr: u16) -> u8 {
            0
        }

        fn cpu_write(&mut self, _addr: u16, _value: u8, _interrupt: &mut InterruptLines) {}

        fn ppu_read(&mut self, addr: u16) -> u8 {
            self.chr[usize::from(addr & 0x1fff)]
        }

        fn debug_ppu_read(&self, addr: u16) -> u8 {
            self.chr[usize::from(addr & 0x1fff)]
        }

        fn ppu_write(&mut self, addr: u16, value: u8) {
            self.chr[usize::from(addr & 0x1fff)] = value;
        }

        fn nametable_mirroring(&self) -> Mirroring {
            Mirroring::Horizontal
        }
    }

    fn run_ppu_dots<M: Mapper + ?Sized>(
        ppu: &mut Ppu,
        mapper: &mut M,
        interrupt: &mut InterruptLines,
        dots: u64,
    ) {
        let target = ppu
            .master_clock
            .wrapping_add(dots * u64::from(ppu.master_clock_divider));
        ppu.run(target, mapper, interrupt);
    }

    #[test]
    fn remove_sprite_limit_renders_extra_sprites_per_scanline() {
        // 16 sprites all at Y=50 -> scanline 50 has 16 sprites. With the hardware
        // 8-sprite limit only 8 render (the flicker), with remove_sprite_limit the
        // 9th..16th also render. Proves load_extra_sprites is not a silent no-op.
        fn setup(remove_limit: bool) -> Ppu {
            let mut ppu = Ppu::default();
            for s in 0..16usize {
                ppu.oam[s * 4] = 50; // Y
                ppu.oam[s * 4 + 1] = 1; // tile index
                ppu.oam[s * 4 + 2] = 0; // attributes
                ppu.oam[s * 4 + 3] = (s as u8).wrapping_mul(8); // X
            }
            ppu.scanline = 50;
            // State the accurate cycle-257..320 eval would leave: 8 sprites found,
            // eval began at OAM 0, last in-range sprite at OAM addr 7*4.
            ppu.sprite_count = 8;
            ppu.sprite_index = 8;
            ppu.first_visible_sprite_addr = 0;
            ppu.last_visible_sprite_addr = 7 * 4;
            ppu.remove_sprite_limit = remove_limit;
            ppu
        }
        let mut mapper = TestMapper { chr: [0; 0x2000] };
        let mut lines = InterruptLines::default();

        let mut ppu = setup(false);
        ppu.load_extra_sprites::<false, _>(&mut mapper, &mut lines);
        assert_eq!(
            ppu.sprite_count, 8,
            "no extra sprites without remove_sprite_limit"
        );

        let mut ppu = setup(true);
        ppu.load_extra_sprites::<false, _>(&mut mapper, &mut lines);
        assert_eq!(
            ppu.sprite_count, 16,
            "remove_sprite_limit must render all 16 same-scanline sprites"
        );
    }

    #[test]
    fn writes_and_reads_vram_through_ppudata() {
        let mut ppu = Ppu::default();
        let mut mapper = TestMapper { chr: [0; 0x2000] };
        let mut lines = InterruptLines::default();
        ppu.cpu_write_register(0x2006, 0x20, 0, &mut mapper, &mut lines);
        ppu.cpu_write_register(0x2006, 0x00, 0, &mut mapper, &mut lines);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 3);
        ppu.cpu_write_register(0x2007, 0x55, 0, &mut mapper, &mut lines);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 6);

        ppu.cpu_write_register(0x2006, 0x20, 0, &mut mapper, &mut lines);
        ppu.cpu_write_register(0x2006, 0x00, 0, &mut mapper, &mut lines);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 3);
        assert_eq!(ppu.cpu_read_register(0x2007, &mut mapper, &mut lines), 0);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 6);
        assert_eq!(ppu.cpu_read_register(0x2007, &mut mapper, &mut lines), 0x55);
    }

    #[test]
    fn palette_reads_are_not_buffered() {
        let mut ppu = Ppu::default();
        let mut mapper = TestMapper { chr: [0; 0x2000] };
        let mut lines = InterruptLines::default();
        ppu.cpu_write_register(0x2006, 0x3f, 0, &mut mapper, &mut lines);
        ppu.cpu_write_register(0x2006, 0x10, 0, &mut mapper, &mut lines);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 3);
        ppu.cpu_write_register(0x2007, 0xaa, 0, &mut mapper, &mut lines);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 6);

        ppu.cpu_write_register(0x2006, 0x3f, 0, &mut mapper, &mut lines);
        ppu.cpu_write_register(0x2006, 0x00, 0, &mut mapper, &mut lines);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 3);
        assert_eq!(ppu.cpu_read_register(0x2007, &mut mapper, &mut lines), 0x2a);
    }

    /// $2007 grayscale post-processing (`if (GRAYSCALE) ret &= 0x30;`) is
    /// applied to palette $2007 reads when PPUMASK bit 0 (GRAYSCALE) is set.
    /// Hand-verification: palette[0x10] = 0x2a (after & 0x3f). With grayscale:
    /// 0x2a & 0x30 = 0x20. Without grayscale: 0x2a.
    #[test]
    fn palette_read_applies_grayscale_mask_when_ppumask_bit0_set() {
        let mut ppu = Ppu::default();
        let mut mapper = TestMapper { chr: [0; 0x2000] };
        let mut lines = InterruptLines::default();
        // Seed palette[0x10] = 0x2a.
        ppu.cpu_write_register(0x2006, 0x3f, 0, &mut mapper, &mut lines);
        ppu.cpu_write_register(0x2006, 0x10, 0, &mut mapper, &mut lines);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 3);
        ppu.cpu_write_register(0x2007, 0xaa, 0, &mut mapper, &mut lines);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 6);

        // Read 1: grayscale OFF, expect raw 0x2a.
        ppu.mask = 0x00;
        ppu.cpu_write_register(0x2006, 0x3f, 0, &mut mapper, &mut lines);
        ppu.cpu_write_register(0x2006, 0x10, 0, &mut mapper, &mut lines);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 3);
        assert_eq!(ppu.cpu_read_register(0x2007, &mut mapper, &mut lines), 0x2a);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 6);

        // Read 2: grayscale ON (bit 0), expect masked 0x2a & 0x30 = 0x20.
        ppu.mask = 0x01;
        ppu.cpu_write_register(0x2006, 0x3f, 0, &mut mapper, &mut lines);
        ppu.cpu_write_register(0x2006, 0x10, 0, &mut mapper, &mut lines);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 3);
        assert_eq!(ppu.cpu_read_register(0x2007, &mut mapper, &mut lines), 0x20);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 6);

        // Read 3: grayscale ON, palette value with low bits set should also
        // mask to grayscale column. palette[0x11] = 0x12 -0x12 & 0x30 = 0x10.
        ppu.cpu_write_register(0x2006, 0x3f, 0, &mut mapper, &mut lines);
        ppu.cpu_write_register(0x2006, 0x11, 0, &mut mapper, &mut lines);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 3);
        ppu.cpu_write_register(0x2007, 0x12, 0, &mut mapper, &mut lines);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 6);
        ppu.cpu_write_register(0x2006, 0x3f, 0, &mut mapper, &mut lines);
        ppu.cpu_write_register(0x2006, 0x11, 0, &mut mapper, &mut lines);
        run_ppu_dots(&mut ppu, &mut mapper, &mut lines, 3);
        assert_eq!(ppu.cpu_read_register(0x2007, &mut mapper, &mut lines), 0x10);
    }

    #[test]
    fn oamdata_write_outside_rendering_masks_attr_and_increments() {
        let mut ppu = Ppu::default();
        let mut mapper = TestMapper { chr: [0; 0x2000] };
        let mut lines = InterruptLines::default();
        ppu.cpu_write_register(0x2003, 0x02, 0, &mut mapper, &mut lines);
        ppu.cpu_write_register(0x2004, 0xff, 0, &mut mapper, &mut lines);

        assert_eq!(ppu.oam[0x02], 0xe3);
        assert_eq!(ppu.oam_addr, 0x03);
    }

    #[test]
    fn oamdata_write_during_rendering_only_bumps_oamaddr_high_bits() {
        let mut ppu = Ppu::default();
        let mut mapper = TestMapper { chr: [0; 0x2000] };
        let mut lines = InterruptLines::default();
        ppu.mask = 0x18;
        ppu.rendering_enabled = true;
        ppu.prev_rendering_enabled = true;
        ppu.scanline = 12;
        ppu.oam_addr = 0x21;
        ppu.oam[0x21] = 0x44;

        ppu.cpu_write_register(0x2004, 0x99, 0, &mut mapper, &mut lines);

        assert_eq!(ppu.oam[0x21], 0x44);
        assert_eq!(ppu.oam_addr, 0x25);
    }

    #[test]
    fn oamdata_read_during_sprite_fetch_returns_secondary_oam_copy_buffer() {
        let mut ppu = Ppu::default();
        let mut mapper = TestMapper { chr: [0; 0x2000] };
        let mut lines = InterruptLines::default();
        ppu.mask = 0x18;
        ppu.rendering_enabled = true;
        ppu.prev_rendering_enabled = true;
        ppu.scanline = 7;
        ppu.cycle = 257 + 8 + 5;
        ppu.secondary_sprite_ram[7] = 0x6a;

        assert_eq!(ppu.cpu_read_register(0x2004, &mut mapper, &mut lines), 0x6a);
        assert_eq!(ppu.secondary_oam_addr, 7);
    }

    #[test]
    fn status_read_clears_vblank_and_write_latch() {
        let mut ppu = Ppu::default();
        let mut mapper = TestMapper { chr: [0; 0x2000] };
        let mut lines = InterruptLines::default();
        ppu.status = 0x80;
        ppu.cpu_write_register(0x2006, 0x21, 0, &mut mapper, &mut lines);
        assert!(ppu.write_latch);
        assert_eq!(ppu.cpu_read_register(0x2002, &mut mapper, &mut lines), 0x81);
        assert_eq!(ppu.status & 0x80, 0);
        assert!(!ppu.write_latch);
    }

    // ===== Mesen2 Phase C validation tests =====
    //
    // These tests cover the per-dot PPU helpers; pure-state tests that
    // exercise individual rendering pipeline stages in isolation. The
    // helpers are hand-verified against Mesen2 source line-by-line.

    #[test]
    fn inc_horizontal_scrolling_wraps_and_toggles_nt() {
        // Coarse X = 30, nametable = 0.
        let mut ppu = Ppu {
            vram_addr: 0x001E,
            ..Ppu::default()
        };
        ppu.inc_horizontal_scrolling();
        assert_eq!(ppu.vram_addr & 0x001F, 31, "coarse X advances to 31");
        // Coarse X = 31: wrap to 0, toggle horizontal nametable bit.
        ppu.inc_horizontal_scrolling();
        assert_eq!(ppu.vram_addr & 0x001F, 0, "coarse X wraps to 0");
        assert_eq!(ppu.vram_addr & 0x0400, 0x0400, "NT bit 10 toggled");
    }

    #[test]
    fn inc_vertical_scrolling_fine_y_then_coarse_y() {
        // Fine Y = 6, coarse Y = 5.
        let mut ppu = Ppu {
            vram_addr: (5 << 5) | (6 << 12),
            ..Ppu::default()
        };
        ppu.inc_vertical_scrolling();
        assert_eq!((ppu.vram_addr >> 12) & 0x07, 7, "fine Y advances to 7");
        assert_eq!((ppu.vram_addr >> 5) & 0x1F, 5, "coarse Y unchanged");
        // Fine Y = 7: wraps to 0, coarse Y advances to 6.
        ppu.inc_vertical_scrolling();
        assert_eq!((ppu.vram_addr >> 12) & 0x07, 0, "fine Y wraps to 0");
        assert_eq!((ppu.vram_addr >> 5) & 0x1F, 6, "coarse Y advances to 6");
    }

    #[test]
    fn inc_vertical_scrolling_at_y29_wraps_and_toggles_nt() {
        // Fine Y = 7, coarse Y = 29, nametable = 0.
        let mut ppu = Ppu {
            vram_addr: (29 << 5) | (7 << 12),
            ..Ppu::default()
        };
        ppu.inc_vertical_scrolling();
        assert_eq!((ppu.vram_addr >> 12) & 0x07, 0, "fine Y wraps to 0");
        assert_eq!(
            (ppu.vram_addr >> 5) & 0x1F,
            0,
            "coarse Y wraps from 29 to 0"
        );
        assert_eq!(ppu.vram_addr & 0x0800, 0x0800, "vertical NT bit 11 toggled");
    }

    #[test]
    fn inc_vertical_scrolling_at_y31_wraps_without_nt_toggle() {
        // Coarse Y = 31 (in attribute table): wraps to 0 with NO NT toggle.
        let mut ppu = Ppu {
            vram_addr: (31 << 5) | (7 << 12),
            ..Ppu::default()
        };
        ppu.inc_vertical_scrolling();
        assert_eq!(
            (ppu.vram_addr >> 5) & 0x1F,
            0,
            "coarse Y wraps from 31 to 0"
        );
        assert_eq!(ppu.vram_addr & 0x0800, 0, "NT bit 11 NOT toggled (Y=31)");
    }

    #[test]
    fn get_nametable_addr_uses_v_low_12_bits() {
        let ppu = Ppu {
            vram_addr: 0x2ABC, // arbitrary high bits get masked
            ..Ppu::default()
        };
        assert_eq!(ppu.get_nametable_addr(), 0x2000 | 0x0ABC);
    }

    #[test]
    fn get_attribute_addr_matches_mesen2_formula() {
        // Mesen2: 0x23C0 | (v & 0x0C00) | ((v >> 4) & 0x38) | ((v >> 2) & 0x07)
        // Pick a v with coarse X = 0b10011 (19) and coarse Y = 0b11010 (26):
        // coarse X bits (0..4) = 19 = 0b10011 -> ((v>>2)&0x07) = (19>>2)&0x07 = 4
        // coarse Y bits (5..9) = 26 = 0b11010 -> ((v>>4)&0x38) = ((26<<5)>>4)&0x38
        //                       = (26<<1) & 0x38 = 52 & 0x38 = 0x30
        // NT bits = 0b10 (only bit 11 set) -> v & 0x0C00 = 0x0800
        let v = (0b10u16 << 10) | (26u16 << 5) | 19u16;
        let ppu = Ppu {
            vram_addr: v,
            ..Ppu::default()
        };
        let expected = 0x23C0 | 0x0800 | 0x30 | 0x04;
        assert_eq!(ppu.get_attribute_addr(), expected);
    }

    #[test]
    fn load_tile_info_4cycle_sequence_populates_tile() {
        // Position v so NT addr = $2042 (coarse X=2, coarse Y=2):
        let mut ppu = Ppu {
            mask: 0x08,
            rendering_enabled: true,
            prev_rendering_enabled: true,
            ctrl: 0x10,
            vram_addr: (2u16 << 5) | 2,
            ..Ppu::default()
        };
        let mut mapper = TestMapper { chr: [0; 0x2000] };
        let mut lines = InterruptLines::default();
        // NT reads (0x2000-0x3EFF) come from PPU's
        // internal nametable RAM via mirroring, NOT mapper. Plant test
        // bytes directly into PPU's nametable[]. For default Horizontal
        // mirroring (cartridge.mirroring), `nametable_index(0x2042, H)`
        // returns 0x42 (NT slot 0). `nametable_index(0x23C0, H)` returns
        // 0x3C0 (attribute area in NT slot 0).
        ppu.nametable[0x042] = 0x55;
        ppu.nametable[0x3C0] = 0xE4;

        // Cycle % 8 == 1: NT fetch + load shifts.
        ppu.cycle = 1;
        ppu.load_tile_info::<false, _>(&mut mapper, &mut lines);
        // tile_addr = (0x55 << 4) | (v >> 12 = 0) | bg_pattern_addr ($1000)
        assert_eq!(ppu.tile.tile_addr, 0x1550);

        // Cycle % 8 == 3: AT fetch + palette compute.
        // v = (2 << 5) | 2 = 0x42 = 0b0000_0000_0100_0010
        // shift = ((v >> 4) & 0x04) | (v & 0x02)
        //       = (0x04 & 0x04) | (0x02) = 4 | 2 = 6
        // AT byte = 0xE4 = 0b1110_0100; (0xE4 >> 6) & 0x03 = 0b11 = 3
        // palette_offset = 3 << 2 = 12 (0x0C)
        ppu.cycle = 3;
        ppu.load_tile_info::<false, _>(&mut mapper, &mut lines);
        assert_eq!(ppu.tile.palette_offset, 0x0C);
    }

    #[test]
    fn process_sprite_evaluation_start_initializes_state() {
        let mut ppu = Ppu {
            oam_addr: 0b1100_0011, // sprite_addr_h = 0x30, sprite_addr_l = 3
            ..Ppu::default()
        };
        ppu.process_sprite_evaluation_start();
        assert!(!ppu.sprite0_added);
        assert!(!ppu.sprite_in_range);
        assert_eq!(ppu.secondary_oam_addr, 0);
        assert_eq!(ppu.overflow_bug_counter, 0);
        assert!(!ppu.oam_copy_done);
        assert_eq!(ppu.sprite_addr_h, 0x30);
        assert_eq!(ppu.sprite_addr_l, 3);
    }

    #[test]
    fn set_bus_address_notifies_mapper() {
        // Build a tiny mapper that records its A12 edge notifications.
        #[derive(Debug)]
        struct RecordingMapper {
            last_addr: u16,
            last_cpu_cycle_count: u64,
            call_count: u32,
        }
        impl Mapper for RecordingMapper {
            fn mapper_id(&self) -> u16 {
                0
            }
            fn name(&self) -> &'static str {
                "REC"
            }
            fn cpu_read(&mut self, _addr: u16) -> u8 {
                0
            }
            fn cpu_write(&mut self, _addr: u16, _value: u8, _interrupt: &mut InterruptLines) {}
            fn ppu_read(&mut self, _addr: u16) -> u8 {
                0
            }
            fn debug_ppu_read(&self, _addr: u16) -> u8 {
                0
            }
            fn ppu_write(&mut self, _addr: u16, _value: u8) {}
            fn nametable_mirroring(&self) -> Mirroring {
                Mirroring::Horizontal
            }
            fn has_vram_addr_hook(&self) -> bool {
                true
            }
            fn notify_vram_addr(
                &mut self,
                addr: u16,
                ppu_master_clock: u64,
                _interrupt: &mut InterruptLines,
            ) {
                self.last_addr = addr;
                self.last_cpu_cycle_count = ppu_master_clock;
                self.call_count += 1;
            }
        }
        // set_bus_address now passes mapper_cpu_cycle_count
        // (Mesen2 NesConsole::GetMasterClock = _cpu->GetCycleCount()), not
        // PPU master_clock. Test asserts the new unit.
        let mut ppu = Ppu {
            mapper_cpu_cycle_count: 1234,
            ..Ppu::default()
        };
        let mut mapper = RecordingMapper {
            last_addr: 0,
            last_cpu_cycle_count: 0,
            call_count: 0,
        };
        let mut lines = InterruptLines::default();
        ppu.set_bus_address(0x1ABC, &mut mapper, &mut lines);
        assert_eq!(ppu.ppu_bus_address, 0x1ABC);
        assert_eq!(mapper.last_addr, 0x1ABC);
        assert_eq!(mapper.last_cpu_cycle_count, 1234);
        assert_eq!(mapper.call_count, 1);
    }

    #[test]
    fn run_advances_master_clock_through_tick() {
        // Ppu::run loops tick() per dot. Each call advances master_clock
        // by master_clock_divider. After running enough cycles to wrap
        // through a full frame, frame_count should increment.
        let mut ppu = Ppu::default();
        let mut mapper = TestMapper { chr: [0; 0x2000] };
        let mut lines = InterruptLines::default();
        // One full NTSC frame = 262 scanlines * 341 cycles * 4 master clocks
        // = 357,368 master clocks. Run for that much + a margin.
        let frame_master_clocks = 262 * 341 * 4;
        let start_frame = ppu.frame_count;
        ppu.run(frame_master_clocks + 100, &mut mapper, &mut lines);
        assert!(
            ppu.frame_count >= start_frame.wrapping_add(1),
            "frame_count should advance at least one full frame; \
             before={start_frame}, after={}",
            ppu.frame_count
        );
        assert!(
            ppu.master_clock <= frame_master_clocks + 100,
            "master_clock should not overshoot target"
        );
        assert!(
            ppu.master_clock + u64::from(ppu.master_clock_divider) > frame_master_clocks + 100,
            "master_clock should be within one divider of target"
        );
    }
}
