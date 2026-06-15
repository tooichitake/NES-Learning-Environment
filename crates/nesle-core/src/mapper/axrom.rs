use crate::cartridge::{CartridgeImage, Mirroring};
use crate::cpu::InterruptLines;
use crate::mapper::banking::{ChrMemory, PrgMemory};
use crate::mapper::traits::Mapper;
use nesle_common::{NesleError, Result};

#[derive(Debug)]
pub struct Axrom {
    prg: PrgMemory,
    chr: ChrMemory,
    bank_select: u8,
    single_screen_high: bool,
    has_bus_conflicts: bool,
}

impl Axrom {
    pub fn new(cartridge: &CartridgeImage) -> Self {
        Self {
            prg: PrgMemory::new(cartridge.prg_rom.clone(), 32 * 1024),
            chr: ChrMemory::new(cartridge, 8 * 1024),
            bank_select: 0,
            single_screen_high: false,
            has_bus_conflicts: cartridge.submapper == 2,
        }
    }
}

impl Mapper for Axrom {
    fn mapper_id(&self) -> u16 {
        7
    }

    fn name(&self) -> &'static str {
        "AxROM"
    }

    fn cpu_read(&mut self, addr: u16) -> u8 {
        if addr < 0x8000 {
            return 0;
        }
        self.prg.read(usize::from(self.bank_select), addr)
    }

    fn cpu_code_read(&self, addr: u16) -> Option<u8> {
        if addr < 0x8000 {
            return None;
        }
        Some(self.prg.read(usize::from(self.bank_select), addr))
    }

    fn cpu_read_open_bus(
        &mut self,
        addr: u16,
        open_bus: u8,
        _interrupt: &mut InterruptLines,
    ) -> u8 {
        if addr < 0x8000 {
            return open_bus;
        }
        self.prg.read(usize::from(self.bank_select), addr)
    }

    fn cpu_write(&mut self, addr: u16, value: u8, _interrupt: &mut InterruptLines) {
        if addr >= 0x8000 {
            // AXROM.h:22 -Mesen2 uses `value & 0x0F` (16 banks).
            self.bank_select = value & 0x0F;
            self.single_screen_high = value & 0x10 != 0;
        }
    }

    fn ppu_read(&mut self, addr: u16) -> u8 {
        self.chr.read(usize::from(addr & 0x1fff))
    }

    fn debug_ppu_read(&self, addr: u16) -> u8 {
        self.chr.read(usize::from(addr & 0x1fff))
    }

    fn ppu_write(&mut self, addr: u16, value: u8) {
        self.chr.write(usize::from(addr & 0x1fff), value);
    }

    fn nametable_mirroring(&self) -> Mirroring {
        if self.single_screen_high {
            Mirroring::SingleScreenUpper
        } else {
            Mirroring::SingleScreenLower
        }
    }

    fn has_bus_conflicts(&self) -> bool {
        self.has_bus_conflicts
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![self.bank_select, u8::from(self.single_screen_high)];
        self.chr.snapshot_bytes(&mut bytes);
        bytes
    }

    fn restore_snapshot(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() < 2 {
            return Err(NesleError::InvalidState(
                "AxROM snapshot is missing bank bytes".to_string(),
            ));
        }
        self.bank_select = bytes[0] & 0x0F;
        self.single_screen_high = bytes[1] != 0;
        let mut offset = 2;
        self.chr.restore_snapshot(bytes, &mut offset, "AxROM")?;
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
            mapper_id: 7,
            submapper: 0,
            mirroring: Mirroring::SingleScreenLower,
            battery: false,
            region: crate::cartridge::Region::Ntsc,
            prg_rom: (0..0x10000).map(|value| (value / 0x8000) as u8).collect(),
            chr_rom: Vec::new(),
            trainer_data: Vec::new(),
            work_ram_size: 8 * 1024,
            save_ram_size: 0,
            prg_ram_size: 8 * 1024,
            prg_ram_unspecified: false,
            save_chr_ram_size: 0,
            chr_ram_size: 8 * 1024,
            chr_ram_unspecified: false,
            input_device: 0,
        }
    }

    #[test]
    fn switches_32k_prg_bank() {
        let mut mapper = Axrom::new(&cartridge());
        let mut lines = InterruptLines::default();
        assert_eq!(mapper.cpu_read(0x8000), 0);
        mapper.cpu_write(0x8000, 1, &mut lines);
        assert_eq!(mapper.cpu_read(0x8000), 1);
        assert_eq!(mapper.cpu_read(0xffff), 1);
    }
}
