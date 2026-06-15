use crate::cartridge::{CartridgeImage, Mirroring};
use crate::cpu::InterruptLines;
use crate::mapper::banking::{ChrMemory, PrgMemory};
use crate::mapper::traits::Mapper;
use crate::mapper::{restore_prg_ram, snapshot_prg_ram};
use nesle_common::Result;

#[derive(Debug)]
pub struct Nrom {
    prg: PrgMemory,
    prg_ram: Vec<u8>,
    chr: ChrMemory,
    mirroring: Mirroring,
}

impl Nrom {
    pub fn new(cartridge: &CartridgeImage) -> Self {
        // NROM: 16KB PRG mirror (NROM-128) or 32KB no-mirror (NROM-256).
        // bank_size = 16KB allows bank_count = 1 (16K) or 2 (32K) for
        // automatic mirror-vs-no-mirror via modulo.
        Self {
            prg: PrgMemory::new(cartridge.prg_rom.clone(), 16 * 1024),
            prg_ram: cartridge.initialized_prg_ram(0),
            chr: ChrMemory::new(cartridge, 8 * 1024),
            mirroring: cartridge.mirroring,
        }
    }
}

impl Mapper for Nrom {
    fn mapper_id(&self) -> u16 {
        0
    }

    fn name(&self) -> &'static str {
        "NROM"
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
        }
    }

    fn ppu_read(&mut self, addr: u16) -> u8 {
        let index = usize::from(addr & 0x1fff);
        self.chr.read(index)
    }

    fn debug_ppu_read(&self, addr: u16) -> u8 {
        self.chr.read(usize::from(addr & 0x1fff))
    }

    fn ppu_write(&mut self, addr: u16, value: u8) {
        self.chr.write(usize::from(addr & 0x1fff), value);
    }

    fn nametable_mirroring(&self) -> Mirroring {
        self.mirroring
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        snapshot_prg_ram(&mut bytes, &self.prg_ram);
        self.chr.snapshot_bytes(&mut bytes);
        bytes
    }

    fn restore_snapshot(&mut self, bytes: &[u8]) -> Result<()> {
        let mut offset = restore_prg_ram(bytes, 0, &mut self.prg_ram, "NROM")?;
        self.chr.restore_snapshot(bytes, &mut offset, "NROM")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartridge::{CartridgeFormat, Mirroring};

    fn cartridge(prg_size: usize, chr_size: usize) -> CartridgeImage {
        CartridgeImage {
            format: CartridgeFormat::INes,
            mapper_id: 0,
            submapper: 0,
            mirroring: Mirroring::Horizontal,
            battery: false,
            region: crate::cartridge::Region::Ntsc,
            prg_rom: (0..prg_size).map(|value| value as u8).collect(),
            chr_rom: (0..chr_size)
                .map(|value| (value as u8).wrapping_add(1))
                .collect(),
            trainer_data: Vec::new(),
            work_ram_size: 8 * 1024,
            save_ram_size: 0,
            prg_ram_size: 8 * 1024,
            prg_ram_unspecified: false,
            save_chr_ram_size: 0,
            chr_ram_size: if chr_size == 0 { 8 * 1024 } else { 0 },
            chr_ram_unspecified: false,
            input_device: 0,
        }
    }

    #[test]
    fn mirrors_16k_prg_rom() {
        let mut mapper = Nrom::new(&cartridge(16 * 1024, 8 * 1024));
        assert_eq!(mapper.cpu_read(0x8000), 0);
        assert_eq!(mapper.cpu_read(0xbfff), 0xff);
        assert_eq!(mapper.cpu_read(0xc000), 0);
        assert_eq!(mapper.cpu_read(0xffff), 0xff);
    }

    #[test]
    fn maps_32k_prg_rom_without_mirroring() {
        let mut mapper = Nrom::new(&cartridge(32 * 1024, 8 * 1024));
        assert_eq!(mapper.cpu_read(0x8000), 0);
        assert_eq!(mapper.cpu_read(0xc000), 0);
        assert_eq!(mapper.cpu_read(0xc001), 1);
    }

    #[test]
    fn only_chr_ram_accepts_ppu_writes() {
        let mut chr_rom = Nrom::new(&cartridge(16 * 1024, 8 * 1024));
        let before = chr_rom.ppu_read(0x0010);
        chr_rom.ppu_write(0x0010, 0xee);
        assert_eq!(chr_rom.ppu_read(0x0010), before);

        let mut chr_ram = Nrom::new(&cartridge(16 * 1024, 0));
        chr_ram.ppu_write(0x0010, 0xee);
        assert_eq!(chr_ram.ppu_read(0x0010), 0xee);
    }
}
