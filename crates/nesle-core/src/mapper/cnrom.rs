use crate::cartridge::{CartridgeImage, Mirroring};
use crate::cpu::InterruptLines;
use crate::mapper::banking::{ChrMemory, PrgMemory};
use crate::mapper::traits::Mapper;
use crate::mapper::{restore_prg_ram, snapshot_prg_ram};
use nesle_common::{NesleError, Result};

#[derive(Debug)]
pub struct Cnrom {
    prg: PrgMemory,
    prg_ram: Vec<u8>,
    chr: ChrMemory,
    chr_bank: u8,
    mirroring: Mirroring,
    has_bus_conflicts: bool,
}

impl Cnrom {
    pub fn new(cartridge: &CartridgeImage) -> Self {
        Self {
            prg: PrgMemory::new(cartridge.prg_rom.clone(), 16 * 1024),
            prg_ram: cartridge.initialized_prg_ram(0),
            chr: ChrMemory::new(cartridge, 8 * 1024),
            chr_bank: 0,
            mirroring: cartridge.mirroring,
            has_bus_conflicts: cartridge.submapper == 2,
        }
    }

    fn chr_offset(&self, addr: u16) -> usize {
        let bank_count = self.chr.bank_count(0x2000);
        ((usize::from(self.chr_bank) % bank_count) * 0x2000) + usize::from(addr & 0x1fff)
    }
}

impl Mapper for Cnrom {
    fn mapper_id(&self) -> u16 {
        3
    }

    fn name(&self) -> &'static str {
        "CNROM"
    }

    fn cpu_read(&mut self, addr: u16) -> u8 {
        if (0x6000..=0x7fff).contains(&addr) && !self.prg_ram.is_empty() {
            return self.prg_ram[usize::from(addr - 0x6000) % self.prg_ram.len()];
        }
        if addr < 0x8000 {
            return 0;
        }
        let bank = if addr < 0xc000 { 0 } else { 1 };
        self.prg.read(bank, addr)
    }

    fn cpu_code_read(&self, addr: u16) -> Option<u8> {
        if addr < 0x8000 {
            return None;
        }
        let bank = if addr < 0xc000 { 0 } else { 1 };
        Some(self.prg.read(bank, addr))
    }

    fn cpu_read_open_bus(
        &mut self,
        addr: u16,
        open_bus: u8,
        _interrupt: &mut InterruptLines,
    ) -> u8 {
        if (0x6000..=0x7fff).contains(&addr) && !self.prg_ram.is_empty() {
            return self.prg_ram[usize::from(addr - 0x6000) % self.prg_ram.len()];
        }
        if addr < 0x8000 {
            return open_bus;
        }
        let bank = if addr < 0xc000 { 0 } else { 1 };
        self.prg.read(bank, addr)
    }

    fn cpu_write(&mut self, addr: u16, value: u8, _interrupt: &mut InterruptLines) {
        if (0x6000..=0x7fff).contains(&addr) && !self.prg_ram.is_empty() {
            let len = self.prg_ram.len();
            self.prg_ram[usize::from(addr - 0x6000) % len] = value;
            return;
        }
        if addr >= 0x8000 {
            self.chr_bank = value;
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

    fn has_bus_conflicts(&self) -> bool {
        self.has_bus_conflicts
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![self.chr_bank];
        snapshot_prg_ram(&mut bytes, &self.prg_ram);
        self.chr.snapshot_bytes(&mut bytes);
        bytes
    }

    fn restore_snapshot(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Err(NesleError::InvalidState(
                "CNROM snapshot is missing bank byte".to_string(),
            ));
        }
        self.chr_bank = bytes[0];
        let mut offset = restore_prg_ram(bytes, 1, &mut self.prg_ram, "CNROM")?;
        self.chr.restore_snapshot(bytes, &mut offset, "CNROM")?;
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
            mapper_id: 3,
            submapper: 0,
            mirroring: Mirroring::Horizontal,
            battery: false,
            region: crate::cartridge::Region::Ntsc,
            prg_rom: (0..0x8000).map(|value| value as u8).collect(),
            chr_rom: (0..0x4000).map(|value| (value / 0x2000) as u8).collect(),
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
    fn switches_8k_chr_bank() {
        let mut mapper = Cnrom::new(&cartridge());
        let mut lines = InterruptLines::default();
        assert_eq!(mapper.ppu_read(0x0010), 0);
        mapper.cpu_write(0x8000, 1, &mut lines);
        assert_eq!(mapper.ppu_read(0x0010), 1);
    }
}
