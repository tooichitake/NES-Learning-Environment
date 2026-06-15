use crate::cartridge::{CartridgeImage, Mirroring};
use crate::cpu::{InterruptLines, IrqSource};
use crate::mapper::banking::{ChrMemory, PrgMemory};
use crate::mapper::traits::Mapper;
use crate::mapper::{restore_prg_ram, snapshot_prg_ram};
use nesle_common::{NesleError, Result};

/// MMC3 submapper variant; McAcc replaces the A12 IRQ counter.
#[derive(Debug)]
pub enum Mmc3Variant {
    Standard,
    /// McAcc /8 pulse counter and previous PPU address.
    McAcc {
        counter: u32,
        prev_addr: u16,
    },
}

#[derive(Debug)]
pub struct Mmc3 {
    prg: PrgMemory,
    chr: ChrMemory,
    bank_select: u8,
    bank_regs: [u8; 8],
    prg_ram: Vec<u8>,
    mirroring: Mirroring,
    four_screen: bool,
    irq_latch: u8,
    irq_counter: u8,
    irq_reload: bool,
    irq_enabled: bool,
    /// `$A001` WRAM enable/write-protect register.
    wram_protect: u8,
    /// CPU-cycle timestamp of the last PPU bus access with A12 low.
    a12_low_master_clock: u64,
    /// MMC3A IRQ wraparound behavior gate.
    force_mmc3_rev_a_irqs: bool,
    /// A12 low-phase threshold in CPU cycles.
    a12_low_threshold: u64,
    /// Submapper-specific IRQ behavior.
    variant: Mmc3Variant,
}

impl Mmc3 {
    pub fn new(cartridge: &CartridgeImage) -> Self {
        Self {
            prg: PrgMemory::new(cartridge.prg_rom.clone(), 8 * 1024),
            chr: ChrMemory::new(cartridge, 8 * 1024),
            bank_select: 0,
            bank_regs: [0, 2, 4, 5, 6, 7, 0, 1],
            prg_ram: cartridge.initialized_prg_ram(8 * 1024),
            mirroring: cartridge.mirroring,
            four_screen: cartridge.mirroring == Mirroring::FourScreen,
            irq_latch: 0,
            irq_counter: 0,
            irq_reload: false,
            irq_enabled: false,
            wram_protect: 0,
            a12_low_master_clock: 0,
            force_mmc3_rev_a_irqs: false,
            // A12 low-pass threshold in CPU cycles.
            a12_low_threshold: 3,
            // NES 2.0 submapper 3 selects the McAcc IRQ variant.
            variant: if cartridge.submapper == 3 {
                Mmc3Variant::McAcc {
                    counter: 0,
                    prev_addr: 0,
                }
            } else {
                Mmc3Variant::Standard
            },
        }
    }

    /// WRAM is writable when bit 7 (enable) is set AND bit 6
    /// (write-protect) is clear.
    fn wram_writable(&self) -> bool {
        (self.wram_protect & 0x80) != 0 && (self.wram_protect & 0x40) == 0
    }

    /// WRAM is readable when bit 7 (enable) is set. The write-protect bit
    /// doesn't gate reads.
    fn wram_readable(&self) -> bool {
        (self.wram_protect & 0x80) != 0
    }

    fn prg_bank_for(&self, addr: u16) -> usize {
        let last = self.prg.bank_count() - 1;
        let second_last = last.saturating_sub(1);
        let prg_mode = self.bank_select & 0x40 != 0;
        let bank6 = usize::from(self.bank_regs[6]);
        let bank7 = usize::from(self.bank_regs[7]);
        match (prg_mode, addr) {
            (false, 0x8000..=0x9fff) => bank6,
            (false, 0xa000..=0xbfff) => bank7,
            (false, 0xc000..=0xdfff) => second_last,
            (false, 0xe000..=0xffff) => last,
            (true, 0x8000..=0x9fff) => second_last,
            (true, 0xa000..=0xbfff) => bank7,
            (true, 0xc000..=0xdfff) => bank6,
            (true, 0xe000..=0xffff) => last,
            _ => 0,
        }
    }

    fn chr_bank_count(&self) -> usize {
        self.chr.bank_count(0x400)
    }

    fn chr_offset(&self, addr: u16) -> usize {
        let inverted = self.bank_select & 0x80 != 0;
        let slot = usize::from(addr / 0x400);
        let reg_bank = match (inverted, slot) {
            (false, 0) => usize::from(self.bank_regs[0] & !1),
            (false, 1) => usize::from(self.bank_regs[0] | 1),
            (false, 2) => usize::from(self.bank_regs[1] & !1),
            (false, 3) => usize::from(self.bank_regs[1] | 1),
            (false, 4) => usize::from(self.bank_regs[2]),
            (false, 5) => usize::from(self.bank_regs[3]),
            (false, 6) => usize::from(self.bank_regs[4]),
            (false, _) => usize::from(self.bank_regs[5]),
            (true, 0) => usize::from(self.bank_regs[2]),
            (true, 1) => usize::from(self.bank_regs[3]),
            (true, 2) => usize::from(self.bank_regs[4]),
            (true, 3) => usize::from(self.bank_regs[5]),
            (true, 4) => usize::from(self.bank_regs[0] & !1),
            (true, 5) => usize::from(self.bank_regs[0] | 1),
            (true, 6) => usize::from(self.bank_regs[1] & !1),
            (true, _) => usize::from(self.bank_regs[1] | 1),
        };
        (reg_bank % self.chr_bank_count()) * 0x400 + usize::from(addr & 0x03ff)
    }

    // A12 must stay low for at least three CPU cycles before a rising edge counts.
    fn is_a12_rising_edge(&mut self, addr: u16, cpu_cycle_count: u64) -> bool {
        if addr & 0x1000 != 0 {
            let rising = self.a12_low_master_clock > 0
                && cpu_cycle_count.saturating_sub(self.a12_low_master_clock)
                    >= self.a12_low_threshold;
            self.a12_low_master_clock = 0;
            rising
        } else {
            if self.a12_low_master_clock == 0 {
                self.a12_low_master_clock = cpu_cycle_count.max(1);
            }
            false
        }
    }

    /// IRQ counter clock step; raises External IRQ when enabled counter hits 0.
    fn clock_irq_counter(&mut self, interrupt: &mut InterruptLines) {
        let prev = self.irq_counter;
        if self.irq_counter == 0 || self.irq_reload {
            self.irq_counter = self.irq_latch;
        } else {
            self.irq_counter -= 1;
        }
        let fire = if self.force_mmc3_rev_a_irqs {
            (prev > 0 || self.irq_reload) && self.irq_counter == 0 && self.irq_enabled
        } else {
            self.irq_counter == 0 && self.irq_enabled
        };
        if fire {
            interrupt.set_irq_source(IrqSource::External);
        }
        self.irq_reload = false;
    }
}

impl Mapper for Mmc3 {
    fn mapper_id(&self) -> u16 {
        4
    }

    fn name(&self) -> &'static str {
        "MMC3"
    }

    fn cpu_read(&mut self, addr: u16) -> u8 {
        // Pure-read path. bus.rs bus-conflict + apu.rs DMC only $8000+.
        match addr {
            0x6000..=0x7fff if !self.prg_ram.is_empty() => {
                if !self.wram_readable() {
                    return 0;
                }
                self.prg_ram[usize::from(addr - 0x6000) % self.prg_ram.len()]
            }
            0x8000..=0xffff => self.prg.read(self.prg_bank_for(addr), addr),
            _ => 0,
        }
    }

    fn cpu_code_read(&self, addr: u16) -> Option<u8> {
        if addr < 0x8000 {
            return None;
        }
        Some(self.prg.read(self.prg_bank_for(addr), addr))
    }

    fn cpu_read_open_bus(
        &mut self,
        addr: u16,
        open_bus: u8,
        _interrupt: &mut InterruptLines,
    ) -> u8 {
        match addr {
            // WRAM-disabled reads return CPU open bus. NESLE does not
            // currently model boards (e.g. TQROM) that expose WRAM
            // unconditionally; those would need a per-board override.
            0x6000..=0x7fff if !self.prg_ram.is_empty() => {
                if !self.wram_readable() {
                    return open_bus;
                }
                self.prg_ram[usize::from(addr - 0x6000) % self.prg_ram.len()]
            }
            0x8000..=0xffff => self.prg.read(self.prg_bank_for(addr), addr),
            _ => open_bus,
        }
    }

    fn cpu_write(&mut self, addr: u16, value: u8, interrupt: &mut InterruptLines) {
        match addr {
            0x6000..=0x7fff if !self.prg_ram.is_empty() => {
                if !self.wram_writable() {
                    return;
                }
                let len = self.prg_ram.len();
                self.prg_ram[usize::from(addr - 0x6000) % len] = value;
            }
            0x8000..=0x9fff if addr & 1 == 0 => self.bank_select = value,
            0x8000..=0x9fff => {
                let reg_idx = usize::from(self.bank_select & 0x07);
                // MMC3.h:216-219 -R0/R1 ignore bit 0 (2KB-aligned).
                let masked = if reg_idx <= 1 { value & 0xFE } else { value };
                self.bank_regs[reg_idx] = masked;
            }
            0xa000..=0xbfff if addr & 1 == 0 && !self.four_screen => {
                self.mirroring = if value & 1 == 0 {
                    Mirroring::Vertical
                } else {
                    Mirroring::Horizontal
                };
            }
            0xa000..=0xbfff => self.wram_protect = value,
            0xc000..=0xdfff if addr & 1 == 0 => self.irq_latch = value,
            0xc000..=0xdfff => {
                // $C001 resets McAcc's pulse counter.
                if let Mmc3Variant::McAcc {
                    ref mut counter, ..
                } = self.variant
                {
                    *counter = 0;
                }
                self.irq_counter = 0;
                self.irq_reload = true;
            }
            0xe000..=0xffff if addr & 1 == 0 => {
                self.irq_enabled = false;
                interrupt.clear_irq_source(IrqSource::External);
            }
            0xe000..=0xffff => self.irq_enabled = true,
            _ => {}
        }
    }

    fn ppu_read(&mut self, addr: u16) -> u8 {
        self.chr.read(self.chr_offset(addr))
    }

    fn debug_ppu_read(&self, addr: u16) -> u8 {
        self.chr.read(self.chr_offset(addr))
    }

    fn ppu_write(&mut self, addr: u16, value: u8) {
        let offset = self.chr_offset(addr);
        self.chr.write(offset, value);
    }

    fn nametable_mirroring(&self) -> Mirroring {
        self.mirroring
    }

    fn has_vram_addr_hook(&self) -> bool {
        true
    }

    fn soft_reset(&mut self) {
        // Soft reset clears bank registers and IRQ enable/reload state.
        self.bank_select = 0;
        self.bank_regs = [0, 2, 4, 5, 6, 7, 0, 1];
        self.irq_enabled = false;
        self.irq_reload = false;
        self.irq_latch = 0;
    }

    fn notify_vram_addr(
        &mut self,
        addr: u16,
        cpu_cycle_count: u64,
        interrupt: &mut InterruptLines,
    ) {
        match self.variant {
            Mmc3Variant::Standard => {
                if self.is_a12_rising_edge(addr, cpu_cycle_count) {
                    self.clock_irq_counter(interrupt);
                }
            }
            Mmc3Variant::McAcc {
                ref mut counter,
                ref mut prev_addr,
            } => {
                // McAcc clocks the MMC3 IRQ counter on the first of each 8 A12
                // falling edges.
                let falling = (addr & 0x1000) == 0 && (*prev_addr & 0x1000) != 0;
                if falling {
                    *counter = counter.wrapping_add(1);
                    if *counter == 1 {
                        let prev = self.irq_counter;
                        if self.irq_counter == 0 || self.irq_reload {
                            self.irq_counter = self.irq_latch;
                        } else {
                            self.irq_counter -= 1;
                        }
                        let fire = if self.force_mmc3_rev_a_irqs {
                            (prev > 0 || self.irq_reload)
                                && self.irq_counter == 0
                                && self.irq_enabled
                        } else {
                            self.irq_counter == 0 && self.irq_enabled
                        };
                        if fire {
                            interrupt.set_irq_source(IrqSource::External);
                        }
                        self.irq_reload = false;
                    } else if *counter == 8 {
                        *counter = 0;
                    }
                }
                *prev_addr = addr;
            }
        }
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![self.bank_select];
        bytes.extend_from_slice(&self.bank_regs);
        bytes.push(mirroring_to_byte(self.mirroring));
        bytes.push(u8::from(self.four_screen));
        bytes.extend_from_slice(&[
            self.irq_latch,
            self.irq_counter,
            u8::from(self.irq_reload),
            u8::from(self.irq_enabled),
        ]);
        bytes.push(self.wram_protect);
        bytes.push(u8::from(self.force_mmc3_rev_a_irqs));
        bytes.extend_from_slice(&self.a12_low_master_clock.to_le_bytes());
        // McAcc appends counter + previous address before PRG RAM.
        if let Mmc3Variant::McAcc { counter, prev_addr } = &self.variant {
            bytes.extend_from_slice(&counter.to_le_bytes());
            bytes.extend_from_slice(&prev_addr.to_le_bytes());
        }
        snapshot_prg_ram(&mut bytes, &self.prg_ram);
        self.chr.snapshot_bytes(&mut bytes);
        bytes
    }

    fn restore_snapshot(&mut self, bytes: &[u8]) -> Result<()> {
        // 1 (bank_select) + 8 (bank_regs) + 1 (mirroring) + 1 (four_screen)
        // + 4 (irq_*) + 1 (wram_protect) + 1 (force_rev_a) + 8 (a12_low_clock)
        // = 25 bytes before prg_ram header (Standard) or McAcc state (McAcc).
        if bytes.len() < 25 {
            return Err(NesleError::InvalidState(
                "MMC3 snapshot is missing register bytes".to_string(),
            ));
        }
        self.bank_select = bytes[0];
        self.bank_regs.copy_from_slice(&bytes[1..9]);
        self.mirroring = byte_to_mirroring(bytes[9]);
        self.four_screen = bytes[10] != 0;
        self.irq_latch = bytes[11];
        self.irq_counter = bytes[12];
        self.irq_reload = bytes[13] != 0;
        self.irq_enabled = bytes[14] != 0;
        self.wram_protect = bytes[15];
        self.force_mmc3_rev_a_irqs = bytes[16] != 0;
        self.a12_low_master_clock = u64::from_le_bytes(bytes[17..25].try_into().unwrap());
        let mut cursor = 25;
        if let Mmc3Variant::McAcc {
            ref mut counter,
            ref mut prev_addr,
        } = self.variant
        {
            if bytes.len() < cursor + 6 {
                return Err(NesleError::InvalidState(
                    "MMC3 McAcc snapshot is missing variant state bytes".to_string(),
                ));
            }
            *counter = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            *prev_addr = u16::from_le_bytes(bytes[cursor + 4..cursor + 6].try_into().unwrap());
            cursor += 6;
        }
        let mut offset = restore_prg_ram(bytes, cursor, &mut self.prg_ram, "MMC3")?;
        self.chr.restore_snapshot(bytes, &mut offset, "MMC3")?;
        Ok(())
    }
}

fn mirroring_to_byte(mirroring: Mirroring) -> u8 {
    match mirroring {
        Mirroring::Horizontal => 0,
        Mirroring::Vertical => 1,
        Mirroring::FourScreen => 2,
        Mirroring::SingleScreenLower => 3,
        Mirroring::SingleScreenUpper => 4,
    }
}

fn byte_to_mirroring(value: u8) -> Mirroring {
    match value {
        1 => Mirroring::Vertical,
        2 => Mirroring::FourScreen,
        3 => Mirroring::SingleScreenLower,
        4 => Mirroring::SingleScreenUpper,
        _ => Mirroring::Horizontal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartridge::CartridgeFormat;

    fn cartridge() -> CartridgeImage {
        CartridgeImage {
            format: CartridgeFormat::INes,
            mapper_id: 4,
            submapper: 0,
            mirroring: Mirroring::Horizontal,
            battery: false,
            region: crate::cartridge::Region::Ntsc,
            prg_rom: (0..(8 * 0x2000)).map(|value| value as u8).collect(),
            chr_rom: vec![0; 8 * 0x400],
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

    fn bank_number_cartridge() -> CartridgeImage {
        CartridgeImage {
            prg_rom: (0..(8 * 0x2000))
                .map(|value| (value / 0x2000) as u8)
                .collect(),
            ..cartridge()
        }
    }

    #[test]
    fn wram_gated_by_protect_register() {
        let mut mapper = Mmc3::new(&cartridge());
        let mut lines = InterruptLines::default();
        assert_eq!(mapper.cpu_read(0x6000), 0);
        mapper.cpu_write(0x6000, 0x5a, &mut lines);
        assert_eq!(mapper.cpu_read(0x6000), 0);

        mapper.cpu_write(0xa001, 0x80, &mut lines);
        mapper.cpu_write(0x6000, 0x5a, &mut lines);
        assert_eq!(mapper.cpu_read(0x6000), 0x5a);

        mapper.cpu_write(0xa001, 0xc0, &mut lines);
        mapper.cpu_write(0x6000, 0xff, &mut lines);
        assert_eq!(mapper.cpu_read(0x6000), 0x5a);
    }

    #[test]
    fn cpu_code_read_tracks_prg_mode_and_skips_wram() {
        let mut mapper = Mmc3::new(&bank_number_cartridge());
        let mut lines = InterruptLines::default();

        assert_eq!(mapper.cpu_code_read(0x6000), None);
        assert_eq!(mapper.cpu_code_read(0x8000), Some(0));
        assert_eq!(mapper.cpu_code_read(0xa000), Some(1));
        assert_eq!(mapper.cpu_code_read(0xe000), Some(7));

        mapper.cpu_write(0x8000, 0x06, &mut lines);
        mapper.cpu_write(0x8001, 0x02, &mut lines);
        assert_eq!(mapper.cpu_code_read(0x8000), Some(2));
        assert_eq!(mapper.cpu_code_read(0xc000), Some(6));

        mapper.cpu_write(0x8000, 0x46, &mut lines);
        assert_eq!(mapper.cpu_code_read(0x8000), Some(6));
        assert_eq!(mapper.cpu_code_read(0xc000), Some(2));
    }

    fn step_a12(mapper: &mut Mmc3, master_clock: &mut u64, lines: &mut InterruptLines) {
        mapper.notify_vram_addr(0x0000, *master_clock, lines);
        *master_clock += 40;
        mapper.notify_vram_addr(0x1000, *master_clock, lines);
        *master_clock += 40;
    }

    #[test]
    fn clocks_irq_counter_from_a12_edges() {
        let mut mapper = Mmc3::new(&cartridge());
        let mut lines = InterruptLines::default();
        mapper.cpu_write(0xc000, 2, &mut lines);
        mapper.cpu_write(0xe001, 0, &mut lines);

        let mut clock = 4_u64;
        step_a12(&mut mapper, &mut clock, &mut lines);
        assert!(!lines.has_irq_source(IrqSource::External));
        assert_eq!(mapper.irq_counter, 2);

        step_a12(&mut mapper, &mut clock, &mut lines);
        assert!(!lines.has_irq_source(IrqSource::External));
        assert_eq!(mapper.irq_counter, 1);

        step_a12(&mut mapper, &mut clock, &mut lines);
        assert!(lines.has_irq_source(IrqSource::External));
        assert_eq!(mapper.irq_counter, 0);

        mapper.cpu_write(0xe000, 0, &mut lines);
        assert!(!lines.has_irq_source(IrqSource::External));
    }

    #[test]
    fn irq_reload_uses_latch_on_next_edge() {
        let mut mapper = Mmc3::new(&cartridge());
        let mut lines = InterruptLines::default();
        mapper.cpu_write(0xc000, 7, &mut lines);
        mapper.cpu_write(0xc001, 0, &mut lines);

        let mut clock = 4_u64;
        step_a12(&mut mapper, &mut clock, &mut lines);
        assert_eq!(mapper.irq_counter, 7);
        assert!(!mapper.irq_reload);
    }

    #[test]
    fn a12_low_filter_rejects_back_to_back_high_accesses() {
        let mut mapper = Mmc3::new(&cartridge());
        let mut lines = InterruptLines::default();
        mapper.cpu_write(0xc000, 5, &mut lines);
        mapper.cpu_write(0xe001, 0, &mut lines);

        let mut clock = 4_u64;
        mapper.notify_vram_addr(0x0000, clock, &mut lines);
        clock += 40;
        mapper.notify_vram_addr(0x1000, clock, &mut lines);
        clock += 4;
        mapper.notify_vram_addr(0x1100, clock, &mut lines);
        assert_eq!(mapper.irq_counter, 5);
    }
}
