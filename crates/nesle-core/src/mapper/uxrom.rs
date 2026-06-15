use crate::cartridge::{CartridgeImage, Mirroring};
use crate::cpu::InterruptLines;
use crate::mapper::banking::{ChrMemory, PrgMemory};
use crate::mapper::traits::Mapper;
use crate::mapper::{restore_prg_ram, snapshot_prg_ram};
use nesle_common::{NesleError, Result};

#[derive(Debug)]
pub struct Uxrom {
    prg: PrgMemory,
    prg_ram: Vec<u8>,
    chr: ChrMemory,
    bank_select: u8,
    mirroring: Mirroring,
    has_bus_conflicts: bool,
}

impl Uxrom {
    pub fn new(cartridge: &CartridgeImage) -> Self {
        Self {
            prg: PrgMemory::new(cartridge.prg_rom.clone(), 16 * 1024),
            prg_ram: cartridge.initialized_prg_ram(0),
            chr: ChrMemory::new(cartridge, 8 * 1024),
            bank_select: 0,
            mirroring: cartridge.mirroring,
            has_bus_conflicts: cartridge.submapper == 2,
        }
    }
}

impl Mapper for Uxrom {
    fn mapper_id(&self) -> u16 {
        2
    }

    fn name(&self) -> &'static str {
        "UxROM"
    }

    fn cpu_read(&mut self, addr: u16) -> u8 {
        // Pure-read path (no open-bus carry-in, no IRQ side effects).
        // Used by bus.rs bus-conflict + apu.rs DMC fetch -both only
        // touch $8000-$FFFF so the unmapped-returns-0 here is fine.
        if (0x6000..=0x7fff).contains(&addr) && !self.prg_ram.is_empty() {
            return self.prg_ram[usize::from(addr - 0x6000) % self.prg_ram.len()];
        }
        if addr < 0x8000 {
            return 0;
        }
        let bank = if addr < 0xc000 {
            usize::from(self.bank_select)
        } else {
            self.prg.bank_count() - 1
        };
        self.prg.read(bank, addr)
    }

    fn cpu_code_read(&self, addr: u16) -> Option<u8> {
        if addr < 0x8000 {
            return None;
        }
        let bank = if addr < 0xc000 {
            usize::from(self.bank_select)
        } else {
            self.prg.bank_count() - 1
        };
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
            // Mesen2 BaseMapper::ReadRam falls through to `GetOpenBus()` for
            // any address not mapped by the cart -fixes Blades of Steel's
            // LDA $4092 returning $00 instead of the prior bus byte.
            return open_bus;
        }
        let bank = if addr < 0xc000 {
            usize::from(self.bank_select)
        } else {
            self.prg.bank_count() - 1
        };
        self.prg.read(bank, addr)
    }

    fn cpu_write(&mut self, addr: u16, value: u8, _interrupt: &mut InterruptLines) {
        if (0x6000..=0x7fff).contains(&addr) && !self.prg_ram.is_empty() {
            let len = self.prg_ram.len();
            self.prg_ram[usize::from(addr - 0x6000) % len] = value;
            return;
        }
        if addr >= 0x8000 {
            // Mesen2 UNROM.h:24 `SelectPrgPage(0, value)`
            // passes the raw 8-bit value; PrgMemory.read modulos by
            // bank_count internally. The previous `& 0x0f` defensive mask
            // would silently mis-bank >256KB UxROM carts (e.g. UNROM-512)
            // by truncating bank select to 4 bits.
            self.bank_select = value;
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
        self.mirroring
    }

    fn has_bus_conflicts(&self) -> bool {
        self.has_bus_conflicts
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![self.bank_select];
        snapshot_prg_ram(&mut bytes, &self.prg_ram);
        self.chr.snapshot_bytes(&mut bytes);
        bytes
    }

    fn restore_snapshot(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Err(NesleError::InvalidState(
                "UxROM snapshot is missing bank byte".to_string(),
            ));
        }
        // match F3 `cpu_write` which removed the
        // `& 0x0f` defensive mask (Mesen2 UNROM.h:24 `SelectPrgPage(0, value)`
        // passes raw 8-bit value; PrgMemory modulos by bank_count internally).
        // Without this fix, restoring a snapshot of a >256KB UxROM cart (e.g.
        // UNROM-512) would silently truncate bank_select to 4 bits and diverge
        // from the saved state.
        self.bank_select = bytes[0];
        let mut offset = restore_prg_ram(bytes, 1, &mut self.prg_ram, "UxROM")?;
        self.chr.restore_snapshot(bytes, &mut offset, "UxROM")?;
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
            mapper_id: 2,
            submapper: 0,
            mirroring: Mirroring::Horizontal,
            battery: false,
            region: crate::cartridge::Region::Ntsc,
            prg_rom: (0..0xc000).map(|value| (value / 0x4000) as u8).collect(),
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
    fn switches_low_prg_bank_and_fixes_high_bank() {
        let mut mapper = Uxrom::new(&cartridge());
        let mut lines = InterruptLines::default();
        assert_eq!(mapper.cpu_read(0x8000), 0);
        assert_eq!(mapper.cpu_read(0xc000), 2);
        mapper.cpu_write(0x8000, 1, &mut lines);
        assert_eq!(mapper.cpu_read(0x8000), 1);
        assert_eq!(mapper.cpu_read(0xc000), 2);
    }

    #[test]
    fn cpu_code_read_tracks_bank_switch_and_skips_prg_ram() {
        let mut mapper = Uxrom::new(&cartridge());
        let mut lines = InterruptLines::default();

        assert_eq!(mapper.cpu_code_read(0x6000), None);
        assert_eq!(mapper.cpu_code_read(0x8000), Some(0));
        assert_eq!(mapper.cpu_code_read(0xc000), Some(2));

        mapper.cpu_write(0x8000, 1, &mut lines);
        assert_eq!(mapper.cpu_code_read(0x8000), Some(1));
        assert_eq!(mapper.cpu_code_read(0xc000), Some(2));
    }

    #[test]
    fn cpu_code_read_state_is_per_mapper_instance() {
        let mapper_a = Uxrom::new(&cartridge());
        let mut mapper_b = Uxrom::new(&cartridge());
        let mut lines = InterruptLines::default();

        mapper_b.cpu_write(0x8000, 1, &mut lines);

        assert_eq!(mapper_a.cpu_code_read(0x8000), Some(0));
        assert_eq!(mapper_b.cpu_code_read(0x8000), Some(1));
    }
}
