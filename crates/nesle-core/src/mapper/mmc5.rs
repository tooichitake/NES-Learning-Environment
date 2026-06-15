use crate::cartridge::{CartridgeImage, Mirroring};
use crate::cpu::{InterruptLines, IrqSource};
use crate::mapper::banking::ChrMemory;
use crate::mapper::traits::Mapper;
use crate::mapper::{restore_prg_ram, snapshot_prg_ram};
use nesle_common::{NesleError, Result};

#[derive(Debug)]
struct Mmc5Audio {
    regs: [u8; 0x16],
    pulse_enabled: u8,
    pcm_output: u8,
}

impl Mmc5Audio {
    fn new() -> Self {
        Self {
            regs: [0; 0x16],
            pulse_enabled: 0,
            pcm_output: 0,
        }
    }

    fn read(&self, addr: u16, open_bus: u8) -> u8 {
        match addr {
            0x5010 => 0,
            0x5015 => self.pulse_enabled & 0x03,
            _ => open_bus,
        }
    }

    fn write(&mut self, addr: u16, value: u8) {
        match addr {
            0x5000..=0x5007 | 0x5010 | 0x5011 | 0x5015 => {
                self.regs[usize::from(addr - 0x5000)] = value;
                if addr == 0x5011 {
                    self.pcm_output = value & 0x7f;
                } else if addr == 0x5015 {
                    self.pulse_enabled = value & 0x03;
                }
            }
            _ => {}
        }
    }

    fn snapshot_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.regs);
        bytes.push(self.pulse_enabled);
        bytes.push(self.pcm_output);
    }

    fn restore_snapshot(&mut self, bytes: &[u8], offset: &mut usize) -> Result<()> {
        if bytes.len() < *offset + 0x18 {
            return Err(NesleError::InvalidState(
                "MMC5 audio snapshot is missing bytes".to_string(),
            ));
        }
        self.regs.copy_from_slice(&bytes[*offset..*offset + 0x16]);
        *offset += 0x16;
        self.pulse_enabled = bytes[*offset] & 0x03;
        self.pcm_output = bytes[*offset + 1] & 0x7f;
        *offset += 2;
        Ok(())
    }
}

#[derive(Debug)]
pub struct Mmc5 {
    prg_rom: Vec<u8>,
    prg_ram: Vec<u8>,
    chr: ChrMemory,
    exram: [u8; 0x400],
    ppu_regs: [u8; 8],
    audio: Mmc5Audio,
    prg_mode: u8,
    chr_mode: u8,
    prg_banks: [u8; 4],
    chr_banks_a: [u16; 8],
    chr_banks_b: [u16; 4],
    chr_high_bits: u16,
    extended_ram_mode: u8,
    nametable_mapping: u8,
    fill_mode_tile: u8,
    fill_mode_color: u8,
    split_enabled: bool,
    split_right_side: bool,
    split_delimiter_tile: u8,
    split_scroll: u8,
    split_bank: u16,
    /// NT-fetch counter used to derive the vertical-split tile column.
    split_tile_number: u8,
    /// Whether current PPU fetches are inside the vertical split region.
    split_in_split_region: bool,
    /// Cached split-region tile address shared by the tile's bus fetches.
    split_tile: u16,
    extended_attribute_chr_bank: Option<u16>,
    extended_attribute_palette: u8,
    /// ExRAM-mode-1 CHR override lifetime after a nametable fetch.
    extended_attribute_fetch_counter: u8,
    /// Low 10 bits of the last ExRAM-mode-1 nametable fetch.
    ex_attr_last_nt_fetch: u16,
    ppu_in_frame: bool,
    need_in_frame: bool,
    ppu_idle_counter: u8,
    last_ppu_read_addr: u16,
    nt_read_counter: u8,
    scanline_counter: u8,
    /// Last CHR bank register write; selects CHR-A/B outside frame.
    last_chr_reg: u16,
    wram_page: u8,
    wram_mask_enable: [u8; 2],
    multiplier: [u8; 2],
    irq_scanline: u8,
    irq_enabled: bool,
    irq_status: u8,
    mirroring: Mirroring,
}

impl Mmc5 {
    pub fn new(cartridge: &CartridgeImage) -> Self {
        Self {
            prg_rom: cartridge.prg_rom.clone(),
            prg_ram: cartridge.initialized_prg_ram(64 * 1024),
            chr: ChrMemory::new(cartridge, 8 * 1024),
            exram: [0xff; 0x400],
            ppu_regs: [0; 8],
            audio: Mmc5Audio::new(),
            prg_mode: 3,
            chr_mode: 3,
            prg_banks: [0xff; 4],
            chr_banks_a: [0xff; 8],
            chr_banks_b: [0xff; 4],
            chr_high_bits: 0,
            extended_ram_mode: 0,
            nametable_mapping: 0,
            fill_mode_tile: 0,
            fill_mode_color: 0,
            split_enabled: false,
            split_right_side: false,
            split_delimiter_tile: 0,
            split_scroll: 0,
            split_bank: 0,
            split_tile_number: 0,
            split_in_split_region: false,
            split_tile: 0,
            extended_attribute_chr_bank: None,
            extended_attribute_palette: 0,
            extended_attribute_fetch_counter: 0,
            ex_attr_last_nt_fetch: 0,
            ppu_in_frame: false,
            need_in_frame: false,
            ppu_idle_counter: 0,
            last_ppu_read_addr: 0,
            nt_read_counter: 0,
            scanline_counter: 0,
            last_chr_reg: 0,
            wram_page: 0,
            wram_mask_enable: [0xff; 2],
            multiplier: [0; 2],
            irq_scanline: 0,
            irq_enabled: false,
            irq_status: 0,
            mirroring: cartridge.mirroring,
        }
    }

    fn prg_8k_offset(&self, bank: u8, addr: u16) -> usize {
        if self.prg_rom.is_empty() {
            return 0;
        }
        let banks = (self.prg_rom.len() / 0x2000).max(1);
        let bank = usize::from(bank & 0x7f) % banks;
        bank * 0x2000 + usize::from(addr & 0x1fff)
    }

    fn prg_16k_offset(&self, bank: u8, addr: u16) -> usize {
        if self.prg_rom.is_empty() {
            return 0;
        }
        let banks = (self.prg_rom.len() / 0x4000).max(1);
        let bank = usize::from((bank & 0x7f) >> 1) % banks;
        bank * 0x4000 + usize::from(addr & 0x3fff)
    }

    fn prg_32k_offset(&self, bank: u8, addr: u16) -> usize {
        if self.prg_rom.is_empty() {
            return 0;
        }
        let banks = (self.prg_rom.len() / 0x8000).max(1);
        let bank = usize::from((bank & 0x7f) >> 2) % banks;
        bank * 0x8000 + usize::from(addr & 0x7fff)
    }

    fn read_prg_rom(&self, addr: u16) -> u8 {
        let offset = self.prg_rom_offset(addr);
        self.prg_rom[offset % self.prg_rom.len()]
    }

    fn prg_rom_offset(&self, addr: u16) -> usize {
        match self.prg_mode & 0x03 {
            0 => self.prg_32k_offset(self.prg_banks[1], addr),
            1 => {
                if addr < 0xc000 {
                    self.prg_16k_offset(self.prg_banks[1], addr)
                } else {
                    self.prg_16k_offset(self.prg_banks[3], addr)
                }
            }
            2 => {
                if addr < 0xc000 {
                    self.prg_16k_offset(self.prg_banks[1], addr)
                } else if addr < 0xe000 {
                    self.prg_8k_offset(self.prg_banks[2], addr)
                } else {
                    self.prg_8k_offset(self.prg_banks[3], addr)
                }
            }
            _ => {
                let slot = usize::from((addr - 0x8000) / 0x2000);
                self.prg_8k_offset(self.prg_banks[slot], addr)
            }
        }
    }

    fn wram_index(&self, bank: u8) -> Option<usize> {
        let slot = usize::from(bank & 0x07);
        match self.prg_ram.len() / 0x2000 {
            0 => None,
            1 => (slot <= 3).then_some(0),
            2 => Some((slot & 4) >> 2),
            4 => (slot <= 3).then_some(slot & 3),
            _ => Some(slot),
        }
    }

    fn wram_offset(&self, bank: u8, addr: u16) -> Option<usize> {
        let bank = self.wram_index(bank)?;
        Some((bank * 0x2000 + usize::from(addr & 0x1fff)) % self.prg_ram.len())
    }

    fn read_wram_bank(&self, bank: u8, addr: u16) -> Option<u8> {
        self.wram_offset(bank, addr)
            .and_then(|offset| self.prg_ram.get(offset).copied())
    }

    fn prg_bank_for_8k_slot(&self, slot: usize) -> (u8, bool) {
        match self.prg_mode & 0x03 {
            0 => (self.prg_banks[1] & !3, true),
            1 => {
                if slot <= 1 {
                    (
                        (self.prg_banks[1] & !1) + slot as u8,
                        self.prg_banks[1] & 0x80 != 0,
                    )
                } else {
                    ((self.prg_banks[3] & !1) + (slot as u8 - 2), true)
                }
            }
            2 => {
                if slot <= 1 {
                    (
                        (self.prg_banks[1] & !1) + slot as u8,
                        self.prg_banks[1] & 0x80 != 0,
                    )
                } else if slot == 2 {
                    (self.prg_banks[2], self.prg_banks[2] & 0x80 != 0)
                } else {
                    (self.prg_banks[3], true)
                }
            }
            _ => {
                if slot < 3 {
                    (self.prg_banks[slot], self.prg_banks[slot] & 0x80 != 0)
                } else {
                    (self.prg_banks[3], true)
                }
            }
        }
    }

    fn read_prg_or_wram(&self, addr: u16) -> Option<u8> {
        let slot = usize::from((addr - 0x8000) / 0x2000);
        let (bank, rom) = self.prg_bank_for_8k_slot(slot);
        if rom {
            Some(self.read_prg_rom(addr))
        } else {
            self.read_wram_bank(bank, addr)
        }
    }

    fn write_protect_open(&self) -> bool {
        ((self.wram_mask_enable[0] & 0x03) | ((self.wram_mask_enable[1] & 0x03) << 2)) == 6
    }

    fn write_mapped_wram(&mut self, addr: u16, value: u8) {
        if !self.write_protect_open() {
            return;
        }
        let bank = if addr < 0x8000 {
            Some(self.wram_page)
        } else {
            let slot = usize::from((addr - 0x8000) / 0x2000);
            let (bank, rom) = self.prg_bank_for_8k_slot(slot);
            (!rom).then_some(bank)
        };
        if let Some(offset) = bank.and_then(|bank| self.wram_offset(bank, addr)) {
            self.prg_ram[offset] = value;
        }
    }

    /// Resolve CHR offset lazily from current MMC5 banking state.
    fn chr_offset(&self, addr: u16) -> usize {
        if let Some(bank) = self.extended_attribute_chr_bank {
            return ((usize::from(bank) % self.chr.bank_count(0x1000)) * 0x1000)
                + usize::from(addr & 0x0fff);
        }
        let large_sprites = (self.ppu_regs[0] & 0x20) != 0;
        let chr_a = !large_sprites
            || (self.split_tile_number >= 32 && self.split_tile_number < 40)
            || (!self.ppu_in_frame && self.last_chr_reg <= 0x5127);
        let bank = self.select_chr_bank(addr, chr_a);
        let bank_count = self.chr.bank_count(0x0400);
        ((usize::from(bank) % bank_count) * 0x0400) + usize::from(addr & 0x03ff)
    }

    /// Per-mode CHR bank selection, converting register values to 1KB pages.
    fn select_chr_bank(&self, addr: u16, chr_a: bool) -> u16 {
        match self.chr_mode & 0x03 {
            0 => {
                // CHR mode 0: one 8KB page.
                let base = if chr_a {
                    self.chr_banks_a[7]
                } else {
                    self.chr_banks_b[3]
                };
                let slot_1k = (addr >> 10) & 0x07;
                (base << 3) | slot_1k
            }
            1 => {
                // CHR mode 1: two 4KB pages.
                let base = if addr < 0x1000 {
                    if chr_a {
                        self.chr_banks_a[3]
                    } else {
                        self.chr_banks_b[3]
                    }
                } else if chr_a {
                    self.chr_banks_a[7]
                } else {
                    self.chr_banks_b[3]
                };
                let slot_1k = (addr >> 10) & 0x03;
                (base << 2) | slot_1k
            }
            2 => {
                // CHR mode 2: four 2KB pages.
                let slot = (addr >> 11) & 0x03; // 0..3
                let base = if chr_a {
                    let a_idx = match slot {
                        0 => 1,
                        1 => 3,
                        2 => 5,
                        _ => 7,
                    };
                    self.chr_banks_a[a_idx]
                } else {
                    let b_idx = if slot & 0x01 == 0 { 1 } else { 3 };
                    self.chr_banks_b[b_idx]
                };
                let slot_1k = (addr >> 10) & 0x01;
                (base << 1) | slot_1k
            }
            _ => {
                // CHR mode 3: eight 1KB pages.
                let slot = usize::from(addr / 0x0400).min(7);
                if chr_a {
                    self.chr_banks_a[slot]
                } else {
                    self.chr_banks_b[slot & 0x03]
                }
            }
        }
    }

    fn fill_nametable_value(&self, addr: u16) -> u8 {
        if (addr & 0x03ff) < 0x03c0 {
            self.fill_mode_tile
        } else {
            let color = self.fill_mode_color & 0x03;
            color | (color << 2) | (color << 4) | (color << 6)
        }
    }

    fn read_ciram_page(&self, page: u8, addr: u16, ciram: &[u8; 0x1000]) -> u8 {
        let offset = (usize::from(page & 1) * 0x400) | usize::from(addr & 0x03ff);
        ciram[offset & 0x07ff]
    }

    fn write_ciram_page(&self, page: u8, addr: u16, value: u8, ciram: &mut [u8; 0x1000]) {
        let offset = (usize::from(page & 1) * 0x400) | usize::from(addr & 0x03ff);
        ciram[offset & 0x07ff] = value;
    }

    fn nametable_source(&self, addr: u16) -> u8 {
        let slot = usize::from((addr & 0x0c00) >> 10);
        (self.nametable_mapping >> (slot * 2)) & 0x03
    }

    fn detect_scanline_start(&mut self, addr: u16, interrupt: &mut InterruptLines) {
        if self.nt_read_counter >= 2 {
            if !self.ppu_in_frame && !self.need_in_frame {
                self.need_in_frame = true;
                self.scanline_counter = 0;
            } else {
                self.scanline_counter = self.scanline_counter.wrapping_add(1);
                if self.scanline_counter == self.irq_scanline {
                    self.irq_status |= 0x80;
                    if self.irq_enabled {
                        interrupt.set_irq_source(IrqSource::External);
                    }
                }
            }
        } else if (0x2000..=0x2fff).contains(&addr) {
            // Identical NT or attribute reads can signal scanline transitions.
            if self.last_ppu_read_addr == addr {
                self.nt_read_counter = self.nt_read_counter.saturating_add(1);
                if self.nt_read_counter >= 2 {
                    // Third identical read starts the next split tile column.
                    self.split_tile_number = 0;
                }
            }
        }
        if self.last_ppu_read_addr != addr {
            self.nt_read_counter = 0;
        }
        self.last_ppu_read_addr = addr;
        // Ex-attribute CHR override expires after its three-fetch window.
        if self.extended_attribute_fetch_counter > 0 {
            self.extended_attribute_fetch_counter -= 1;
            if self.extended_attribute_fetch_counter == 0 {
                self.extended_attribute_chr_bank = None;
            }
        }
    }
}

impl Mapper for Mmc5 {
    fn mapper_id(&self) -> u16 {
        5
    }

    fn name(&self) -> &'static str {
        "MMC5"
    }

    fn cpu_read(&mut self, addr: u16) -> u8 {
        // Pure read for bus-conflict and DMC paths; no register side effects.
        match addr {
            0x6000..=0x7fff => self.read_wram_bank(self.wram_page, addr).unwrap_or(0),
            0x8000..=0xffff if !self.prg_rom.is_empty() => self.read_prg_rom(addr),
            _ => 0,
        }
    }

    fn cpu_read_open_bus(&mut self, addr: u16, open_bus: u8, interrupt: &mut InterruptLines) -> u8 {
        match addr {
            0x5c00..=0x5fff => {
                if self.extended_ram_mode <= 1 {
                    open_bus
                } else {
                    self.exram[usize::from(addr - 0x5c00)]
                }
            }
            0x5010 | 0x5015 => self.audio.read(addr, open_bus),
            0x5204 => {
                // `$5204` read returns IRQ state and acknowledges External IRQ.
                let value = (if self.ppu_in_frame { 0x40 } else { 0 }) | (self.irq_status & 0x80);
                self.irq_status &= !0x80;
                interrupt.clear_irq_source(IrqSource::External);
                value
            }
            0x5205 => self.multiplier[0].wrapping_mul(self.multiplier[1]),
            0x5206 => {
                let product = u16::from(self.multiplier[0]) * u16::from(self.multiplier[1]);
                (product >> 8) as u8
            }
            0x6000..=0x7fff => self
                .read_wram_bank(self.wram_page, addr)
                .unwrap_or(open_bus),
            0xfffa | 0xfffb => {
                // NMI-vector reads mark MMC5 end-of-frame IRQ state.
                self.ppu_in_frame = false;
                self.last_ppu_read_addr = 0;
                self.scanline_counter = 0;
                self.irq_status &= !0x80;
                interrupt.clear_irq_source(IrqSource::External);
                self.read_prg_or_wram(addr).unwrap_or(open_bus)
            }
            0x8000..=0xffff if !self.prg_rom.is_empty() => {
                self.read_prg_or_wram(addr).unwrap_or(open_bus)
            }
            _ => 0,
        }
    }

    fn cpu_write(&mut self, addr: u16, value: u8, interrupt: &mut InterruptLines) {
        match addr {
            0x5000..=0x5007 | 0x5010 | 0x5011 | 0x5015 => self.audio.write(addr, value),
            0x5100 => self.prg_mode = value & 0x03,
            0x5101 => self.chr_mode = value & 0x03,
            0x5102 => self.wram_mask_enable[0] = value,
            0x5103 => self.wram_mask_enable[1] = value,
            0x5104 => self.extended_ram_mode = value & 0x03,
            0x5105 => self.nametable_mapping = value,
            0x5106 => self.fill_mode_tile = value,
            0x5107 => self.fill_mode_color = value & 0x03,
            0x5113 => self.wram_page = value,
            0x5114..=0x5117 => self.prg_banks[usize::from(addr & 0x0003)] = value,
            0x5120..=0x5127 => {
                self.chr_banks_a[usize::from(addr & 0x0007)] =
                    u16::from(value) | (self.chr_high_bits << 8);
                // CHR-bank writes record which CHR register family is active.
                self.last_chr_reg = addr;
            }
            0x5128..=0x512b => {
                self.chr_banks_b[usize::from(addr & 0x0003)] =
                    u16::from(value) | (self.chr_high_bits << 8);
                self.last_chr_reg = addr;
            }
            0x5130 => self.chr_high_bits = u16::from(value & 0x03),
            0x5200 => {
                self.split_enabled = value & 0x80 != 0;
                self.split_right_side = value & 0x40 != 0;
                self.split_delimiter_tile = value & 0x1f;
            }
            0x5201 => self.split_scroll = value,
            0x5202 => self.split_bank = u16::from(value) | (self.chr_high_bits << 8),
            0x5203 => self.irq_scanline = value,
            0x5204 => {
                self.irq_enabled = value & 0x80 != 0;
                if !self.irq_enabled {
                    interrupt.clear_irq_source(IrqSource::External);
                } else if self.irq_status & 0x80 != 0 {
                    interrupt.set_irq_source(IrqSource::External);
                }
            }
            0x5205 => self.multiplier[0] = value,
            0x5206 => self.multiplier[1] = value,
            0x5c00..=0x5fff => {
                let value = if self.extended_ram_mode <= 1 && !self.ppu_in_frame {
                    0
                } else {
                    value
                };
                self.exram[usize::from(addr - 0x5c00)] = value;
            }
            0x6000..=0xffff => self.write_mapped_wram(addr, value),
            _ => {}
        }
    }

    fn ppu_read(&mut self, addr: u16) -> u8 {
        self.ppu_idle_counter = 0;
        // Pattern reads also reset consecutive-NT-read detection on mismatch.
        let masked = addr & 0x3fff;
        if masked != self.last_ppu_read_addr {
            self.nt_read_counter = 0;
        }
        self.last_ppu_read_addr = masked;
        // Split-region pattern fetches use the vertical-split CHR bank.
        let result = if self.extended_ram_mode <= 1
            && self.ppu_in_frame
            && self.split_enabled
            && self.split_in_split_region
            && addr < 0x2000
        {
            let scanline = if self.split_tile_number >= 41 {
                self.scanline_counter.wrapping_add(1)
            } else {
                self.scanline_counter
            };
            let vsplit_scroll = scanline.wrapping_add(self.split_scroll) % 240;
            let chr_addr = (usize::from(self.split_bank) << 12)
                + (((usize::from(addr) & !0x07) | (usize::from(vsplit_scroll) & 0x07)) & 0xfff);
            self.chr.read(chr_addr)
        } else {
            self.chr.read(self.chr_offset(addr))
        };
        // ExRAM CHR override lasts for exactly three post-nametable reads.
        if self.extended_attribute_fetch_counter > 0 {
            self.extended_attribute_fetch_counter -= 1;
            if self.extended_attribute_fetch_counter == 0 {
                self.extended_attribute_chr_bank = None;
            }
        }
        result
    }

    fn debug_ppu_read(&self, addr: u16) -> u8 {
        // Side-effect-free CHR addressing for diagnostic reads.
        if self.extended_ram_mode <= 1
            && self.ppu_in_frame
            && self.split_enabled
            && self.split_in_split_region
            && addr < 0x2000
        {
            let scanline = if self.split_tile_number >= 41 {
                self.scanline_counter.wrapping_add(1)
            } else {
                self.scanline_counter
            };
            let vsplit_scroll = scanline.wrapping_add(self.split_scroll) % 240;
            let chr_addr = (usize::from(self.split_bank) << 12)
                + (((usize::from(addr) & !0x07) | (usize::from(vsplit_scroll) & 0x07)) & 0xfff);
            self.chr.read(chr_addr)
        } else {
            self.chr.read(self.chr_offset(addr))
        }
    }

    fn ppu_write(&mut self, addr: u16, value: u8) {
        self.ppu_idle_counter = 0;
        let offset = self.chr_offset(addr);
        self.chr.write(offset, value);
    }

    fn ppu_read_nametable(
        &mut self,
        addr: u16,
        ciram: &[u8; 0x1000],
        interrupt: &mut InterruptLines,
    ) -> Option<u8> {
        // NT fetch order matters: split column and in-frame state update
        // before scanline-start detection.
        let is_nt_byte_fetch = (addr & 0x03ff) < 0x03c0;
        if is_nt_byte_fetch {
            self.split_tile_number = self.split_tile_number.saturating_add(1);
            if !self.ppu_in_frame && self.need_in_frame {
                self.need_in_frame = false;
                self.ppu_in_frame = true;
            }
        }
        self.detect_scanline_start(addr & 0x3fff, interrupt);
        self.ppu_idle_counter = 0;
        let nt_offset = usize::from(addr & 0x03ff);

        // Vertical split decisions are latched on NT fetches and reused by
        // the following attribute/pattern fetches.
        if self.extended_ram_mode <= 1 && self.ppu_in_frame && self.split_enabled {
            // Sprite-fetch phase uses the next scanline's split row.
            let scanline = if self.split_tile_number >= 41 {
                self.scanline_counter.wrapping_add(1)
            } else {
                self.scanline_counter
            };
            let vsplit_scroll = scanline.wrapping_add(self.split_scroll) % 240;
            let column = ((u16::from(self.split_tile_number)).wrapping_add(2)) % 42;
            if is_nt_byte_fetch {
                if column == 0 {
                    // Column 0 initializes split-region membership.
                    self.split_in_split_region = !self.split_right_side;
                }
                if column == u16::from(self.split_delimiter_tile) && self.split_tile_number < 42 {
                    // Delimiter column toggles split-region membership.
                    self.split_in_split_region = !self.split_in_split_region;
                } else if column > 32 {
                    // Columns past visible background leave split mode.
                    self.split_in_split_region = false;
                }
                if self.split_in_split_region {
                    // Split-region NT fetch returns the cached ExRAM tile.
                    self.split_tile = ((u16::from(vsplit_scroll) & 0xf8) << 2) | column;
                    return Some(self.exram[usize::from(self.split_tile) & 0x3ff]);
                }
            } else if self.split_in_split_region {
                // Split-region attribute fetch returns packed ExRAM palette.
                let shift = ((self.split_tile >> 4) & 0x04) | (self.split_tile & 0x02);
                let at_addr =
                    0x3c0 | ((self.split_tile & 0x0380) >> 4) | ((self.split_tile & 0x001f) >> 2);
                let palette = (self.exram[usize::from(at_addr) & 0x3ff] >> shift) & 0x03;
                return Some(palette | (palette << 2) | (palette << 4) | (palette << 6));
            }
        }

        if self.extended_ram_mode == 1 && nt_offset < 0x03c0 {
            // ExRAM mode 1 arms a three-fetch CHR/palette override.
            let ext = self.exram[nt_offset];
            self.extended_attribute_chr_bank =
                Some(u16::from(ext & 0x3f) | (self.chr_high_bits << 6));
            self.extended_attribute_palette = ext >> 6;
            self.extended_attribute_fetch_counter = 3;
            // Persist the canonical ExRAM NT-fetch address.
            self.ex_attr_last_nt_fetch = addr & 0x03FF;
        } else if self.extended_ram_mode == 1 && nt_offset >= 0x03c0 {
            let color = self.extended_attribute_palette & 0x03;
            return Some(color | (color << 2) | (color << 4) | (color << 6));
        }
        let source = self.nametable_source(addr);
        Some(match source {
            0 | 1 => self.read_ciram_page(source, addr, ciram),
            2 if self.extended_ram_mode <= 1 => self.exram[usize::from(addr & 0x03ff)],
            2 => 0,
            _ => self.fill_nametable_value(addr),
        })
    }

    fn ppu_write_nametable(&mut self, addr: u16, value: u8, ciram: &mut [u8; 0x1000]) -> bool {
        let source = self.nametable_source(addr);
        match source {
            0 | 1 => {
                self.write_ciram_page(source, addr, value, ciram);
                true
            }
            2 if self.extended_ram_mode <= 1 => {
                self.exram[usize::from(addr & 0x03ff)] = value;
                true
            }
            _ => true,
        }
    }

    fn nametable_mirroring(&self) -> Mirroring {
        self.mirroring
    }

    fn has_cpu_clock_hook(&self) -> bool {
        true
    }

    fn has_vram_addr_hook(&self) -> bool {
        true
    }

    fn ppu_register_write(&mut self, original_addr: u16, canonical_addr: u16, value: u8) {
        if (0x2000..=0x2007).contains(&original_addr) {
            self.ppu_regs[usize::from(canonical_addr & 0x0007)] = value;
            // 8x8 sprite mode resets CHR-family selection to CHR-A.
            if (canonical_addr & 0x0007) == 0 && (value & 0x20) == 0 {
                self.last_chr_reg = 0;
            }
        }
    }

    fn process_cpu_clock(&mut self, _interrupt: &mut InterruptLines) {
        // Three CPU cycles without PPU reads ends the in-frame window.
        if self.ppu_idle_counter < 3 {
            self.ppu_idle_counter += 1;
            if self.ppu_idle_counter == 3 {
                self.ppu_in_frame = false;
            }
        }
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(128 + self.prg_ram.len() + 0x400);
        bytes.push(self.prg_mode);
        bytes.push(self.chr_mode);
        bytes.extend_from_slice(&self.prg_banks);
        for bank in self.chr_banks_a {
            bytes.extend_from_slice(&bank.to_le_bytes());
        }
        for bank in self.chr_banks_b {
            bytes.extend_from_slice(&bank.to_le_bytes());
        }
        bytes.extend_from_slice(&self.chr_high_bits.to_le_bytes());
        bytes.extend_from_slice(&[
            self.extended_ram_mode,
            self.nametable_mapping,
            self.fill_mode_tile,
            self.fill_mode_color,
            u8::from(self.split_enabled),
            u8::from(self.split_right_side),
            self.split_delimiter_tile,
            self.split_scroll,
            u8::from(self.ppu_in_frame),
            u8::from(self.need_in_frame),
            self.ppu_idle_counter,
        ]);
        bytes.extend_from_slice(&self.split_bank.to_le_bytes());
        bytes.extend_from_slice(&self.ppu_regs);
        self.audio.snapshot_bytes(&mut bytes);
        bytes.extend_from_slice(
            &self
                .extended_attribute_chr_bank
                .unwrap_or(u16::MAX)
                .to_le_bytes(),
        );
        bytes.push(self.extended_attribute_palette);
        bytes.push(self.extended_attribute_fetch_counter);
        // Last ExRAM NT-fetch address, low 10 bits.
        bytes.extend_from_slice(&self.ex_attr_last_nt_fetch.to_le_bytes());
        // Vsplit runtime state is mid-scanline and must be serialized.
        bytes.push(self.split_tile_number);
        bytes.push(u8::from(self.split_in_split_region));
        bytes.extend_from_slice(&self.split_tile.to_le_bytes());
        bytes.extend_from_slice(&self.last_ppu_read_addr.to_le_bytes());
        bytes.extend_from_slice(&[self.nt_read_counter, self.scanline_counter]);
        // Last CHR register is required for post-restore CHR-A/B selection.
        bytes.extend_from_slice(&self.last_chr_reg.to_le_bytes());
        bytes.push(self.wram_page);
        bytes.extend_from_slice(&self.wram_mask_enable);
        bytes.extend_from_slice(&self.multiplier);
        bytes.extend_from_slice(&[
            self.irq_scanline,
            u8::from(self.irq_enabled),
            self.irq_status,
        ]);
        snapshot_prg_ram(&mut bytes, &self.prg_ram);
        bytes.extend_from_slice(&self.exram);
        self.chr.snapshot_bytes(&mut bytes);
        bytes
    }

    fn restore_snapshot(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() < 57 {
            return Err(NesleError::InvalidState(
                "MMC5 snapshot is missing register bytes".to_string(),
            ));
        }
        self.prg_mode = bytes[0] & 0x03;
        self.chr_mode = bytes[1] & 0x03;
        self.prg_banks.copy_from_slice(&bytes[2..6]);
        let mut offset = 6;
        for bank in &mut self.chr_banks_a {
            *bank = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            offset += 2;
        }
        for bank in &mut self.chr_banks_b {
            *bank = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            offset += 2;
        }
        self.chr_high_bits = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
        offset += 2;
        self.extended_ram_mode = bytes[offset] & 0x03;
        self.nametable_mapping = bytes[offset + 1];
        self.fill_mode_tile = bytes[offset + 2];
        self.fill_mode_color = bytes[offset + 3] & 0x03;
        // MMC5 snapshot is single-version; all fields are mandatory.
        self.split_enabled = bytes[offset + 4] != 0;
        self.split_right_side = bytes[offset + 5] != 0;
        self.split_delimiter_tile = bytes[offset + 6] & 0x1f;
        self.split_scroll = bytes[offset + 7];
        self.ppu_in_frame = bytes[offset + 8] != 0;
        self.need_in_frame = bytes[offset + 9] != 0;
        self.ppu_idle_counter = bytes[offset + 10];
        offset += 11;
        self.split_bank = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
        offset += 2;
        self.ppu_regs.copy_from_slice(&bytes[offset..offset + 8]);
        offset += 8;
        self.audio.restore_snapshot(bytes, &mut offset)?;
        let ext_bank = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
        self.extended_attribute_chr_bank = (ext_bank != u16::MAX).then_some(ext_bank);
        offset += 2;
        self.extended_attribute_palette = bytes[offset] & 0x03;
        offset += 1;
        // Ex-attribute three-fetch countdown.
        self.extended_attribute_fetch_counter = bytes[offset];
        offset += 1;
        // Ex-attribute cached NT-fetch address.
        self.ex_attr_last_nt_fetch =
            u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
        offset += 2;
        // Vsplit runtime state.
        self.split_tile_number = bytes[offset];
        self.split_in_split_region = bytes[offset + 1] != 0;
        self.split_tile = u16::from_le_bytes(bytes[offset + 2..offset + 4].try_into().unwrap());
        offset += 4;
        self.last_ppu_read_addr = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
        offset += 2;
        self.nt_read_counter = bytes[offset];
        self.scanline_counter = bytes[offset + 1];
        offset += 2;
        // Last CHR register field.
        self.last_chr_reg = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
        offset += 2;
        self.wram_page = bytes[offset];
        self.wram_mask_enable
            .copy_from_slice(&bytes[offset + 1..offset + 3]);
        offset += 3;
        self.multiplier.copy_from_slice(&bytes[offset..offset + 2]);
        offset += 2;
        self.irq_scanline = bytes[offset];
        self.irq_enabled = bytes[offset + 1] != 0;
        self.irq_status = bytes[offset + 2];
        offset += 3;
        offset = restore_prg_ram(bytes, offset, &mut self.prg_ram, "MMC5")?;
        if bytes.len() < offset + 0x400 {
            return Err(NesleError::InvalidState(
                "MMC5 snapshot is missing EXRAM".to_string(),
            ));
        }
        self.exram.copy_from_slice(&bytes[offset..offset + 0x400]);
        offset += 0x400;
        self.chr.restore_snapshot(bytes, &mut offset, "MMC5")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartridge::{CartridgeFormat, Mirroring};

    fn cartridge() -> CartridgeImage {
        CartridgeImage {
            format: CartridgeFormat::INes,
            mapper_id: 5,
            submapper: 0,
            mirroring: Mirroring::Horizontal,
            battery: false,
            region: crate::cartridge::Region::Ntsc,
            prg_rom: (0..0x4000).map(|value| value as u8).collect(),
            chr_rom: vec![0x55; 0x2000],
            trainer_data: Vec::new(),
            work_ram_size: 8 * 1024,
            save_ram_size: 0,
            prg_ram_size: 8 * 1024,
            prg_ram_unspecified: false,
            save_chr_ram_size: 0,
            chr_ram_size: 0,
            chr_ram_unspecified: false,
            input_device: 0,
        }
    }

    #[test]
    fn maps_small_prg_rom_and_exram() {
        let mut mapper = Mmc5::new(&cartridge());
        let mut lines = InterruptLines::default();
        assert_eq!(mapper.cpu_read(0x8001), 1);
        mapper.cpu_write(0x5c10, 0x77, &mut lines);
        assert_eq!(mapper.cpu_read_open_bus(0x5c10, 0x12, &mut lines), 0x12);
        mapper.cpu_write(0x5104, 0x02, &mut lines);
        mapper.cpu_write(0x5c10, 0x77, &mut lines);
        assert_eq!(mapper.cpu_read_open_bus(0x5c10, 0x12, &mut lines), 0x77);
    }

    #[test]
    fn dynamic_nametable_fill_and_exram_sources() {
        let mut mapper = Mmc5::new(&cartridge());
        let mut lines = InterruptLines::default();
        let mut ciram = [0u8; 0x1000];
        ciram[0] = 0x11;
        ciram[0x400] = 0x22;
        mapper.cpu_write(0x5106, 0x33, &mut lines);
        mapper.cpu_write(0x5107, 0x02, &mut lines);
        mapper.cpu_write(0x5105, 0b11_10_01_00, &mut lines);

        assert_eq!(
            mapper.ppu_read_nametable(0x2000, &ciram, &mut lines),
            Some(0x11)
        );
        assert_eq!(
            mapper.ppu_read_nametable(0x2400, &ciram, &mut lines),
            Some(0x22)
        );
        mapper.ppu_write_nametable(0x2801, 0x44, &mut ciram);
        assert_eq!(
            mapper.ppu_read_nametable(0x2801, &ciram, &mut lines),
            Some(0x44)
        );
        assert_eq!(
            mapper.ppu_read_nametable(0x2c00, &ciram, &mut lines),
            Some(0x33)
        );
        assert_eq!(
            mapper.ppu_read_nametable(0x2fc0, &ciram, &mut lines),
            Some(0xaa)
        );
    }
}
