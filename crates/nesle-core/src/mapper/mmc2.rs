use crate::cartridge::{CartridgeImage, Mirroring};
use crate::cpu::InterruptLines;
use crate::mapper::banking::{ChrMemory, PrgMemory};
use crate::mapper::traits::Mapper;
use crate::mapper::{restore_prg_ram, snapshot_prg_ram};
use nesle_common::{NesleError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mmc2Variant {
    Mmc2,
    Mmc4,
}

#[derive(Debug)]
pub struct Mmc2 {
    variant: Mmc2Variant,
    prg: PrgMemory,
    prg_ram: Vec<u8>,
    chr: ChrMemory,
    prg_bank: u8,
    chr_banks: [u8; 4],
    selected_chr_banks: [u8; 2],
    latch0: u8,
    latch1: u8,
    need_chr_update: bool,
    mirroring: Mirroring,
}

impl Mmc2 {
    pub fn new(cartridge: &CartridgeImage) -> Self {
        Self::with_variant(cartridge, Mmc2Variant::Mmc2)
    }

    pub fn new_mmc4(cartridge: &CartridgeImage) -> Self {
        Self::with_variant(cartridge, Mmc2Variant::Mmc4)
    }

    fn with_variant(cartridge: &CartridgeImage, variant: Mmc2Variant) -> Self {
        Self {
            variant,
            prg: PrgMemory::new(cartridge.prg_rom.clone(), 8 * 1024),
            prg_ram: cartridge.initialized_prg_ram(if cartridge.battery { 8 * 1024 } else { 0 }),
            chr: ChrMemory::new(cartridge, 8 * 1024),
            prg_bank: 0,
            chr_banks: [0; 4],
            selected_chr_banks: [0; 2],
            latch0: 1,
            latch1: 1,
            need_chr_update: false,
            mirroring: cartridge.mirroring,
        }
    }

    fn prg_bank_for(&self, addr: u16) -> usize {
        let bank_count = self.prg.bank_count();
        match self.variant {
            Mmc2Variant::Mmc2 => match addr {
                0x8000..=0x9fff => usize::from(self.prg_bank & 0x0f),
                0xa000..=0xbfff => bank_count.saturating_sub(3),
                0xc000..=0xdfff => bank_count.saturating_sub(2),
                0xe000..=0xffff => bank_count.saturating_sub(1),
                _ => 0,
            },
            Mmc2Variant::Mmc4 => match addr {
                0x8000..=0x9fff => usize::from(self.prg_bank & 0x0f) * 2,
                0xa000..=0xbfff => usize::from(self.prg_bank & 0x0f) * 2 + 1,
                0xc000..=0xdfff => bank_count.saturating_sub(2),
                0xe000..=0xffff => bank_count.saturating_sub(1),
                _ => 0,
            },
        }
    }

    fn chr_bank_count(&self) -> usize {
        self.chr.bank_count(0x1000)
    }

    fn chr_offset(&self, addr: u16) -> usize {
        let bank = if addr < 0x1000 {
            self.selected_chr_banks[0]
        } else {
            self.selected_chr_banks[1]
        };
        let bank = usize::from(bank) % self.chr_bank_count();
        bank * 0x1000 + usize::from(addr & 0x0fff)
    }

    fn select_chr_for_latches(&mut self) {
        self.selected_chr_banks[0] = self.chr_banks[usize::from(self.latch0 & 1)];
        self.selected_chr_banks[1] = self.chr_banks[2 + usize::from(self.latch1 & 1)];
    }

    fn apply_vram_latch(&mut self, addr: u16) {
        if self.need_chr_update {
            self.select_chr_for_latches();
            self.need_chr_update = false;
        }

        let addr = addr & 0x1fff;
        match addr {
            0x0fd8 if self.variant == Mmc2Variant::Mmc2 => {
                self.latch0 = 0;
                self.need_chr_update = true;
            }
            0x0fe8 if self.variant == Mmc2Variant::Mmc2 => {
                self.latch0 = 1;
                self.need_chr_update = true;
            }
            0x0fd8..=0x0fdf if self.variant == Mmc2Variant::Mmc4 => {
                self.latch0 = 0;
                self.need_chr_update = true;
            }
            0x0fe8..=0x0fef if self.variant == Mmc2Variant::Mmc4 => {
                self.latch0 = 1;
                self.need_chr_update = true;
            }
            0x1fd8..=0x1fdf => {
                self.latch1 = 0;
                self.need_chr_update = true;
            }
            0x1fe8..=0x1fef => {
                self.latch1 = 1;
                self.need_chr_update = true;
            }
            _ => {}
        }
    }
}

impl Mapper for Mmc2 {
    fn mapper_id(&self) -> u16 {
        match self.variant {
            Mmc2Variant::Mmc2 => 9,
            Mmc2Variant::Mmc4 => 10,
        }
    }

    fn name(&self) -> &'static str {
        match self.variant {
            Mmc2Variant::Mmc2 => "MMC2",
            Mmc2Variant::Mmc4 => "MMC4",
        }
    }

    fn cpu_read(&mut self, addr: u16) -> u8 {
        if (0x6000..=0x7fff).contains(&addr) && !self.prg_ram.is_empty() {
            return self.prg_ram[usize::from(addr - 0x6000) % self.prg_ram.len()];
        }
        if addr < 0x8000 {
            return 0;
        }
        self.prg.read(self.prg_bank_for(addr), addr)
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
        if (0x6000..=0x7fff).contains(&addr) && !self.prg_ram.is_empty() {
            return self.prg_ram[usize::from(addr - 0x6000) % self.prg_ram.len()];
        }
        if addr < 0x8000 {
            return open_bus;
        }
        self.prg.read(self.prg_bank_for(addr), addr)
    }

    fn cpu_write(&mut self, addr: u16, value: u8, _interrupt: &mut InterruptLines) {
        if (0x6000..=0x7fff).contains(&addr) && !self.prg_ram.is_empty() {
            let len = self.prg_ram.len();
            self.prg_ram[usize::from(addr - 0x6000) % len] = value;
            return;
        }
        match addr & 0xf000 {
            0xa000 => self.prg_bank = value & 0x0f,
            0xb000 => {
                self.chr_banks[0] = value & 0x1f;
                self.selected_chr_banks[0] = self.chr_banks[usize::from(self.latch0 & 1)];
            }
            0xc000 => {
                self.chr_banks[1] = value & 0x1f;
                self.selected_chr_banks[0] = self.chr_banks[usize::from(self.latch0 & 1)];
            }
            0xd000 => {
                self.chr_banks[2] = value & 0x1f;
                self.selected_chr_banks[1] = self.chr_banks[2 + usize::from(self.latch1 & 1)];
            }
            0xe000 => {
                self.chr_banks[3] = value & 0x1f;
                self.selected_chr_banks[1] = self.chr_banks[2 + usize::from(self.latch1 & 1)];
            }
            0xf000 => {
                self.mirroring = if value & 1 == 0 {
                    Mirroring::Vertical
                } else {
                    Mirroring::Horizontal
                };
            }
            _ => {}
        }
    }

    fn ppu_read(&mut self, addr: u16) -> u8 {
        self.apply_vram_latch(addr);
        self.chr.read(self.chr_offset(addr))
    }

    fn debug_ppu_read(&self, addr: u16) -> u8 {
        // No apply_vram_latch: a debug read must NOT toggle the MMC2 CHR latch.
        self.chr.read(self.chr_offset(addr))
    }

    fn ppu_write(&mut self, addr: u16, value: u8) {
        self.apply_vram_latch(addr);
        self.chr.write(self.chr_offset(addr), value);
    }

    fn nametable_mirroring(&self) -> Mirroring {
        self.mirroring
    }

    fn has_vram_addr_hook(&self) -> bool {
        true
    }

    fn soft_reset(&mut self) {
        // MMC2.h::InitMapper -soft reset re-initializes the PRG
        // bank register, CHR bank registers, latches, and forces a CHR
        // table update on next access.
        self.prg_bank = 0;
        self.chr_banks = [0; 4];
        self.selected_chr_banks = [0; 2];
        self.latch0 = 0;
        self.latch1 = 0;
        self.need_chr_update = true;
    }

    fn notify_vram_addr(
        &mut self,
        addr: u16,
        _cpu_cycle_count: u64,
        _interrupt: &mut InterruptLines,
    ) {
        self.apply_vram_latch(addr);
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![
            self.prg_bank,
            self.chr_banks[0],
            self.chr_banks[1],
            self.chr_banks[2],
            self.chr_banks[3],
            self.selected_chr_banks[0],
            self.selected_chr_banks[1],
            self.latch0,
            self.latch1,
            u8::from(self.need_chr_update),
            match self.mirroring {
                Mirroring::Horizontal => 0,
                Mirroring::Vertical => 1,
                Mirroring::FourScreen => 2,
                Mirroring::SingleScreenLower => 3,
                Mirroring::SingleScreenUpper => 4,
            },
        ];
        snapshot_prg_ram(&mut bytes, &self.prg_ram);
        self.chr.snapshot_bytes(&mut bytes);
        bytes
    }

    fn restore_snapshot(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() < 15 {
            return Err(NesleError::InvalidState(
                "MMC2 snapshot is missing register bytes".to_string(),
            ));
        }
        self.prg_bank = bytes[0] & 0x0f;
        self.chr_banks.copy_from_slice(&bytes[1..5]);
        for bank in &mut self.chr_banks {
            *bank &= 0x1f;
        }
        self.selected_chr_banks[0] = bytes[5] & 0x1f;
        self.selected_chr_banks[1] = bytes[6] & 0x1f;
        self.latch0 = bytes[7] & 1;
        self.latch1 = bytes[8] & 1;
        self.need_chr_update = bytes[9] != 0;
        self.mirroring = match bytes[10] {
            0 => Mirroring::Horizontal,
            1 => Mirroring::Vertical,
            2 => Mirroring::FourScreen,
            3 => Mirroring::SingleScreenLower,
            4 => Mirroring::SingleScreenUpper,
            _ => {
                return Err(NesleError::InvalidState(
                    "MMC2 snapshot has invalid mirroring byte".to_string(),
                ));
            }
        };
        let mut offset = restore_prg_ram(bytes, 11, &mut self.prg_ram, "MMC2")?;
        self.chr.restore_snapshot(bytes, &mut offset, "MMC2")?;
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
            mapper_id: 9,
            submapper: 0,
            mirroring: Mirroring::Horizontal,
            battery: false,
            region: crate::cartridge::Region::Ntsc,
            prg_rom: (0..0x10000).map(|value| (value / 0x2000) as u8).collect(),
            chr_rom: (0..0x8000).map(|value| (value / 0x1000) as u8).collect(),
            trainer_data: Vec::new(),
            work_ram_size: 0,
            save_ram_size: 0,
            prg_ram_size: 0,
            prg_ram_unspecified: false,
            save_chr_ram_size: 0,
            chr_ram_size: 0,
            chr_ram_unspecified: false,
            input_device: 0,
        }
    }

    fn cartridge_with_mapper(mapper_id: u16) -> CartridgeImage {
        CartridgeImage {
            mapper_id,
            ..cartridge()
        }
    }

    #[test]
    fn switches_low_prg_bank_and_keeps_fixed_high_banks() {
        let mut mapper = Mmc2::new(&cartridge());
        assert_eq!(mapper.cpu_read(0x8000), 0);
        assert_eq!(mapper.cpu_read(0xa000), 5);
        assert_eq!(mapper.cpu_read(0xc000), 6);
        assert_eq!(mapper.cpu_read(0xe000), 7);
        let mut lines = InterruptLines::default();
        mapper.cpu_write(0xa000, 3, &mut lines);
        assert_eq!(mapper.cpu_read(0x8000), 3);
    }

    #[test]
    fn pattern_reads_update_chr_latches() {
        let mut mapper = Mmc2::new(&cartridge());
        let mut lines = InterruptLines::default();
        mapper.cpu_write(0xb000, 2, &mut lines);
        mapper.cpu_write(0xc000, 3, &mut lines);
        mapper.cpu_write(0xd000, 4, &mut lines);
        mapper.cpu_write(0xe000, 5, &mut lines);
        assert_eq!(mapper.ppu_read(0x0000), 3);
        mapper.ppu_read(0x0fd8);
        assert_eq!(mapper.ppu_read(0x0000), 2);
        mapper.ppu_read(0x0fe8);
        assert_eq!(mapper.ppu_read(0x0000), 3);
        mapper.ppu_read(0x1fd8);
        assert_eq!(mapper.ppu_read(0x1000), 4);
        mapper.ppu_read(0x1fe8);
        assert_eq!(mapper.ppu_read(0x1000), 5);
    }

    #[test]
    fn latch_trigger_affects_next_pattern_access() {
        let mut mapper = Mmc2::new(&cartridge());
        let mut lines = InterruptLines::default();
        mapper.cpu_write(0xb000, 2, &mut lines);
        mapper.cpu_write(0xc000, 3, &mut lines);

        assert_eq!(mapper.ppu_read(0x0000), 3);
        assert_eq!(mapper.ppu_read(0x0fd8), 3);
        assert_eq!(mapper.ppu_read(0x0000), 2);
    }

    #[test]
    fn mmc2_left_latch_uses_exact_trigger_addresses() {
        let mut mapper = Mmc2::new(&cartridge());
        let mut lines = InterruptLines::default();
        mapper.cpu_write(0xb000, 2, &mut lines);
        mapper.cpu_write(0xc000, 3, &mut lines);

        assert_eq!(mapper.ppu_read(0x0000), 3);
        mapper.ppu_read(0x0fd9);
        assert_eq!(mapper.ppu_read(0x0000), 3);
        mapper.ppu_read(0x0fd8);
        assert_eq!(mapper.ppu_read(0x0000), 2);
    }

    #[test]
    fn mmc4_uses_16k_prg_banks_and_wider_left_latch_ranges() {
        let mut mapper = Mmc2::new_mmc4(&cartridge_with_mapper(10));
        assert_eq!(mapper.mapper_id(), 10);
        assert_eq!(mapper.cpu_read(0x8000), 0);
        assert_eq!(mapper.cpu_read(0xa000), 1);
        assert_eq!(mapper.cpu_read(0xc000), 6);
        assert_eq!(mapper.cpu_read(0xe000), 7);
        let mut lines = InterruptLines::default();
        mapper.cpu_write(0xa000, 2, &mut lines);
        assert_eq!(mapper.cpu_read(0x8000), 4);
        assert_eq!(mapper.cpu_read(0xa000), 5);

        mapper.cpu_write(0xb000, 2, &mut lines);
        mapper.cpu_write(0xc000, 3, &mut lines);
        assert_eq!(mapper.ppu_read(0x0000), 3);
        assert_eq!(mapper.ppu_read(0x0fd9), 3);
        assert_eq!(mapper.ppu_read(0x0000), 2);
    }
}
