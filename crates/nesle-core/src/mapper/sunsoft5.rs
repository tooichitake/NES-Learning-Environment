use crate::cartridge::{CartridgeImage, Mirroring};
use crate::cpu::{InterruptLines, IrqSource};
use crate::mapper::banking::{ChrMemory, PrgMemory};
use crate::mapper::traits::Mapper;
use crate::mapper::{restore_prg_ram, snapshot_prg_ram};
use nesle_common::{NesleError, Result};

#[derive(Debug)]
pub struct Sunsoft5 {
    prg: PrgMemory,
    prg_ram: Vec<u8>,
    chr: ChrMemory,
    command: u8,
    prg_banks: [u8; 4],
    chr_banks: [u8; 8],
    mirroring: Mirroring,
    irq_control: u8,
    irq_counter: i32,
}

impl Sunsoft5 {
    pub fn new(cartridge: &CartridgeImage) -> Self {
        Self {
            prg: PrgMemory::new(cartridge.prg_rom.clone(), 8 * 1024),
            prg_ram: cartridge.initialized_prg_ram(32 * 1024),
            chr: ChrMemory::new(cartridge, 8 * 1024),
            command: 0,
            prg_banks: [0; 4],
            chr_banks: [0; 8],
            mirroring: cartridge.mirroring,
            irq_control: 0,
            irq_counter: 0xffff,
        }
    }

    fn prg_ram_offset(&self, addr: u16) -> usize {
        let bank = usize::from(self.prg_banks[3] & 0x3f);
        (bank * 0x2000 + usize::from(addr & 0x1fff)) % self.prg_ram.len().max(1)
    }

    fn chr_offset(&self, addr: u16) -> usize {
        let bank_count = self.chr.bank_count(0x0400);
        let slot = usize::from(addr / 0x0400).min(7);
        let bank = usize::from(self.chr_banks[slot]) % bank_count;
        bank * 0x0400 + usize::from(addr & 0x03ff)
    }

    fn write_selected_register(&mut self, value: u8, interrupt: &mut InterruptLines) {
        match self.command & 0x0f {
            0x0..=0x7 => self.chr_banks[usize::from(self.command & 0x07)] = value,
            0x8 => self.prg_banks[3] = value,
            0x9 => self.prg_banks[0] = value,
            0xa => self.prg_banks[1] = value,
            0xb => self.prg_banks[2] = value,
            0xc => {
                self.mirroring = match value & 0x03 {
                    0 => Mirroring::Vertical,
                    1 => Mirroring::Horizontal,
                    2 => Mirroring::SingleScreenLower,
                    _ => Mirroring::SingleScreenUpper,
                };
            }
            0xd => {
                self.irq_control = value;
                interrupt.clear_irq_source(IrqSource::External);
            }
            0xe => self.irq_counter = (self.irq_counter & 0xff00) | i32::from(value),
            0xf => self.irq_counter = (self.irq_counter & 0x00ff) | (i32::from(value) << 8),
            _ => {}
        }
    }
}

impl Mapper for Sunsoft5 {
    fn mapper_id(&self) -> u16 {
        69
    }

    fn name(&self) -> &'static str {
        "Sunsoft-5/FME-7"
    }

    fn cpu_read(&mut self, addr: u16) -> u8 {
        // Pure-read path. Used by bus.rs bus-conflict + apu.rs DMC only.
        match addr {
            0x6000..=0x7fff => {
                if self.prg_banks[3] & 0xc0 == 0x40 {
                    0
                } else if self.prg_banks[3] & 0xc0 == 0xc0 && !self.prg_ram.is_empty() {
                    self.prg_ram[self.prg_ram_offset(addr)]
                } else {
                    self.prg.read(usize::from(self.prg_banks[3] & 0x3f), addr)
                }
            }
            0x8000..=0x9fff => self.prg.read(usize::from(self.prg_banks[0]), addr),
            0xa000..=0xbfff => self.prg.read(usize::from(self.prg_banks[1]), addr),
            0xc000..=0xdfff => self.prg.read(usize::from(self.prg_banks[2]), addr),
            0xe000..=0xffff => self.prg.read(self.prg.bank_count() - 1, addr),
            _ => 0,
        }
    }

    fn cpu_code_read(&self, addr: u16) -> Option<u8> {
        let bank = match addr {
            0x8000..=0x9fff => usize::from(self.prg_banks[0]),
            0xa000..=0xbfff => usize::from(self.prg_banks[1]),
            0xc000..=0xdfff => usize::from(self.prg_banks[2]),
            0xe000..=0xffff => self.prg.bank_count() - 1,
            _ => return None,
        };
        Some(self.prg.read(bank, addr))
    }

    fn cpu_read_open_bus(
        &mut self,
        addr: u16,
        open_bus: u8,
        _interrupt: &mut InterruptLines,
    ) -> u8 {
        match addr {
            0x6000..=0x7fff => {
                if self.prg_banks[3] & 0xc0 == 0x40 {
                    open_bus
                } else if self.prg_banks[3] & 0xc0 == 0xc0 && !self.prg_ram.is_empty() {
                    self.prg_ram[self.prg_ram_offset(addr)]
                } else {
                    self.prg.read(usize::from(self.prg_banks[3] & 0x3f), addr)
                }
            }
            0x8000..=0x9fff => self.prg.read(usize::from(self.prg_banks[0]), addr),
            0xa000..=0xbfff => self.prg.read(usize::from(self.prg_banks[1]), addr),
            0xc000..=0xdfff => self.prg.read(usize::from(self.prg_banks[2]), addr),
            0xe000..=0xffff => self.prg.read(self.prg.bank_count() - 1, addr),
            _ => 0,
        }
    }

    fn cpu_write(&mut self, addr: u16, value: u8, interrupt: &mut InterruptLines) {
        match addr {
            0x6000..=0x7fff if self.prg_banks[3] & 0xc0 == 0xc0 && !self.prg_ram.is_empty() => {
                let offset = self.prg_ram_offset(addr);
                self.prg_ram[offset] = value;
            }
            0x8000..=0x9fff => self.command = value & 0x0f,
            0xa000..=0xbfff => self.write_selected_register(value, interrupt),
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

    fn has_cpu_clock_hook(&self) -> bool {
        true
    }

    fn soft_reset(&mut self) {
        // Sunsoft5b.h::InitMapper -soft reset clears the command
        // register, IRQ control register, and reloads the IRQ counter to
        // power-on default (0xFFFF).
        self.command = 0;
        self.irq_control = 0;
        self.irq_counter = 0xffff;
    }

    fn process_cpu_clock(&mut self, interrupt: &mut InterruptLines) {
        let counter_enabled = self.irq_control & 0x80 != 0;
        let irq_enabled = self.irq_control & 0x01 != 0;
        if !counter_enabled {
            return;
        }
        let new_counter = (self.irq_counter as i64 - 1) & 0xFFFF;
        let underflow = new_counter == 0xFFFF;
        self.irq_counter = new_counter as i32;
        if underflow && irq_enabled {
            interrupt.set_irq_source(IrqSource::External);
        }
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(33 + self.prg_ram.len());
        bytes.push(self.command);
        bytes.extend_from_slice(&self.prg_banks);
        bytes.extend_from_slice(&self.chr_banks);
        bytes.push(match self.mirroring {
            Mirroring::Horizontal => 0,
            Mirroring::Vertical => 1,
            Mirroring::FourScreen => 2,
            Mirroring::SingleScreenLower => 3,
            Mirroring::SingleScreenUpper => 4,
        });
        bytes.push(self.irq_control);
        bytes.extend_from_slice(&self.irq_counter.to_le_bytes());
        snapshot_prg_ram(&mut bytes, &self.prg_ram);
        self.chr.snapshot_bytes(&mut bytes);
        bytes
    }

    fn restore_snapshot(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() < 24 {
            return Err(NesleError::InvalidState(
                "Sunsoft-5 snapshot is missing register bytes".to_string(),
            ));
        }
        self.command = bytes[0] & 0x0f;
        self.prg_banks.copy_from_slice(&bytes[1..5]);
        self.chr_banks.copy_from_slice(&bytes[5..13]);
        self.mirroring = match bytes[13] {
            0 => Mirroring::Horizontal,
            1 => Mirroring::Vertical,
            2 => Mirroring::FourScreen,
            3 => Mirroring::SingleScreenLower,
            4 => Mirroring::SingleScreenUpper,
            _ => {
                return Err(NesleError::InvalidState(
                    "Sunsoft-5 snapshot has invalid mirroring byte".to_string(),
                ));
            }
        };
        self.irq_control = bytes[14];
        self.irq_counter = i32::from_le_bytes(bytes[15..19].try_into().unwrap());
        let mut offset = restore_prg_ram(bytes, 19, &mut self.prg_ram, "Sunsoft-5")?;
        self.chr.restore_snapshot(bytes, &mut offset, "Sunsoft-5")?;
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
            mapper_id: 69,
            submapper: 0,
            mirroring: Mirroring::Horizontal,
            battery: false,
            region: crate::cartridge::Region::Ntsc,
            prg_rom: (0..0x20000).map(|value| (value / 0x2000) as u8).collect(),
            chr_rom: vec![0; 0x2000],
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
    fn disabled_wram_window_returns_cpu_open_bus() {
        let mut mapper = Sunsoft5::new(&cartridge());
        let mut lines = InterruptLines::default();
        mapper.cpu_write(0x8000, 0x08, &mut lines);
        mapper.cpu_write(0xa000, 0x40, &mut lines);

        assert_eq!(mapper.cpu_read_open_bus(0x6000, 0xa5, &mut lines), 0xa5);
    }

    #[test]
    fn wram_window_writes_only_when_enabled() {
        let mut mapper = Sunsoft5::new(&cartridge());
        let mut lines = InterruptLines::default();

        mapper.cpu_write(0x8000, 0x08, &mut lines);
        mapper.cpu_write(0xa000, 0x40, &mut lines);
        mapper.cpu_write(0x6000, 0x11, &mut lines);
        mapper.cpu_write(0xa000, 0xc0, &mut lines);
        assert_eq!(mapper.cpu_read(0x6000), 0xff);

        mapper.cpu_write(0x6000, 0x22, &mut lines);
        assert_eq!(mapper.cpu_read(0x6000), 0x22);
    }

    #[test]
    fn irq_control_bits_match_fme7() {
        let mut mapper = Sunsoft5::new(&cartridge());
        let mut lines = InterruptLines::default();

        mapper.irq_counter = 0;
        mapper.irq_control = 0x80;
        mapper.process_cpu_clock(&mut lines);
        assert_eq!(mapper.irq_counter, 0xffff);
        assert!(!lines.has_irq_source(IrqSource::External));

        mapper.irq_counter = 0;
        mapper.irq_control = 0x01;
        mapper.process_cpu_clock(&mut lines);
        assert_eq!(mapper.irq_counter, 0);
        assert!(!lines.has_irq_source(IrqSource::External));

        mapper.irq_counter = 0;
        mapper.irq_control = 0x81;
        mapper.process_cpu_clock(&mut lines);
        assert_eq!(mapper.irq_counter, 0xffff);
        assert!(lines.has_irq_source(IrqSource::External));
    }
}
