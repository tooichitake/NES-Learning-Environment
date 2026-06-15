#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankWindow {
    pub source_offset: usize,
    pub size: usize,
}

use crate::cartridge::CartridgeImage;
use nesle_common::{NesleError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChrSource {
    Default = 0,
    Rom = 1,
    Ram = 2,
    NametableRam = 3,
    MapperRam = 4,
    Unmapped = 5,
}

impl ChrSource {
    fn from_byte(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Default),
            1 => Ok(Self::Rom),
            2 => Ok(Self::Ram),
            3 => Ok(Self::NametableRam),
            4 => Ok(Self::MapperRam),
            5 => Ok(Self::Unmapped),
            _ => Err(NesleError::InvalidState(format!(
                "invalid CHR source byte {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PpuAccess {
    pub read: bool,
    pub write: bool,
}

impl PpuAccess {
    pub const READ: Self = Self {
        read: true,
        write: false,
    };
    pub const READ_WRITE: Self = Self {
        read: true,
        write: true,
    };
    pub const NONE: Self = Self {
        read: false,
        write: false,
    };

    fn to_byte(self) -> u8 {
        u8::from(self.read) | (u8::from(self.write) << 1)
    }

    fn from_byte(value: u8) -> Self {
        Self {
            read: value & 0x01 != 0,
            write: value & 0x02 != 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PpuPage {
    pub source: ChrSource,
    pub source_offset: usize,
    pub access: PpuAccess,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChrMemory {
    rom: Vec<u8>,
    ram: Vec<u8>,
    pages: [PpuPage; 0x40],
}

impl ChrMemory {
    pub fn new(cartridge: &CartridgeImage, min_ram_size: usize) -> Self {
        let explicit_ram = cartridge.total_chr_ram_size();
        let ram_size = if cartridge.chr_rom.is_empty() && cartridge.chr_ram_unspecified {
            explicit_ram.max(min_ram_size)
        } else {
            explicit_ram
        };
        let mut chr = Self {
            rom: cartridge.chr_rom.clone(),
            ram: vec![0xff; ram_size],
            pages: [PpuPage {
                source: ChrSource::Default,
                source_offset: 0,
                access: PpuAccess::READ,
            }; 0x40],
        };
        chr.select_default_mapping();
        chr
    }

    fn default_access(&self) -> PpuAccess {
        if !self.rom.is_empty() {
            PpuAccess::READ
        } else if !self.ram.is_empty() {
            PpuAccess::READ_WRITE
        } else {
            PpuAccess::NONE
        }
    }

    pub fn select_default_mapping(&mut self) {
        let access = self.default_access();
        for page in 0..0x40 {
            self.pages[page] = PpuPage {
                source: ChrSource::Default,
                source_offset: page * 0x100,
                access,
            };
        }
    }

    pub fn set_ppu_mapping(
        &mut self,
        start: u16,
        end: u16,
        source: ChrSource,
        source_offset: usize,
        access: PpuAccess,
    ) {
        let first_page = usize::from((start & 0x3fff) >> 8).min(0x3f);
        let last_page = usize::from((end & 0x3fff) >> 8).min(0x3f);
        for page in first_page..=last_page {
            self.pages[page] = PpuPage {
                source,
                source_offset: source_offset + ((page - first_page) * 0x100),
                access,
            };
        }
    }

    pub fn read(&self, offset: usize) -> u8 {
        if !self.rom.is_empty() {
            self.rom[offset % self.rom.len()]
        } else if !self.ram.is_empty() {
            self.ram[offset % self.ram.len()]
        } else {
            0
        }
    }

    pub fn write(&mut self, offset: usize, value: u8) {
        if !self.ram.is_empty() && self.rom.is_empty() {
            let index = offset % self.ram.len();
            self.ram[index] = value;
        }
    }

    pub fn read_ppu(&self, addr: u16) -> u8 {
        let addr = usize::from(addr & 0x3fff);
        let page = self.pages[(addr >> 8).min(0x3f)];
        if !page.access.read {
            return 0;
        }
        let offset = page.source_offset + (addr & 0x00ff);
        match page.source {
            ChrSource::Default => self.read(offset),
            ChrSource::Rom => self
                .rom
                .get(offset % self.rom.len().max(1))
                .copied()
                .unwrap_or(0),
            ChrSource::Ram => self
                .ram
                .get(offset % self.ram.len().max(1))
                .copied()
                .unwrap_or(0),
            ChrSource::NametableRam | ChrSource::MapperRam | ChrSource::Unmapped => 0,
        }
    }

    pub fn write_ppu(&mut self, addr: u16, value: u8) {
        let addr = usize::from(addr & 0x3fff);
        let page = self.pages[(addr >> 8).min(0x3f)];
        if !page.access.write || self.ram.is_empty() {
            return;
        }
        let offset = page.source_offset + (addr & 0x00ff);
        match page.source {
            ChrSource::Default | ChrSource::Ram => {
                let index = offset % self.ram.len();
                self.ram[index] = value;
            }
            _ => {}
        }
    }

    pub fn bank_count(&self, bank_size: usize) -> usize {
        let len = if !self.rom.is_empty() {
            self.rom.len()
        } else {
            self.ram.len()
        };
        (len / bank_size).max(1)
    }

    pub fn snapshot_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&(self.ram.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.ram);
        for page in self.pages {
            bytes.push(page.source as u8);
            bytes.extend_from_slice(&(page.source_offset as u32).to_le_bytes());
            bytes.push(page.access.to_byte());
        }
    }

    pub fn restore_snapshot(&mut self, bytes: &[u8], offset: &mut usize, name: &str) -> Result<()> {
        if bytes.len() == *offset && self.ram.is_empty() {
            return Ok(());
        }
        if bytes.len() < *offset + 4 {
            return Err(NesleError::InvalidState(format!(
                "{name} CHR RAM snapshot is missing length"
            )));
        }
        let len = u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap()) as usize;
        *offset += 4;
        if len != self.ram.len() || bytes.len() < *offset + len {
            return Err(NesleError::InvalidState(format!(
                "{name} CHR RAM snapshot length must be {}, got {len}",
                self.ram.len()
            )));
        }
        self.ram.copy_from_slice(&bytes[*offset..*offset + len]);
        *offset += len;
        // Single-version snapshot format; page table is mandatory.
        if bytes.len() < *offset + (0x40 * 6) {
            return Err(NesleError::InvalidState(format!(
                "{name} CHR page-table snapshot truncated (expected {} bytes)",
                0x40 * 6
            )));
        }
        for page in &mut self.pages {
            page.source = ChrSource::from_byte(bytes[*offset])?;
            *offset += 1;
            page.source_offset =
                u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap()) as usize;
            *offset += 4;
            page.access = PpuAccess::from_byte(bytes[*offset]);
            *offset += 1;
        }
        Ok(())
    }
}

/// PRG ROM memory with bank-indexed access. Owns the ROM bytes and
/// computes offsets modulo bank_count to keep mappers free from explicit
/// `.is_empty()` / `.max(1)` defenses. The bank size is fixed at
/// construction (8KB / 16KB / 32KB per mapper); banks within a single
/// PrgMemory must be uniform-sized (MMC3 8K + MMC5 mixed-mode use
/// per-mapper logic separately).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrgMemory {
    rom: Vec<u8>,
    bank_size: usize,
    bank_count: usize,
    bank_size_mask: usize,
    bank_count_mask: Option<usize>,
    wraps_index: bool,
}

impl PrgMemory {
    pub fn new(rom: Vec<u8>, bank_size: usize) -> Self {
        assert!(bank_size > 0, "PrgMemory bank_size must be > 0");
        assert!(
            bank_size.is_power_of_two(),
            "PrgMemory bank_size must be a power of two"
        );
        let bank_count = (rom.len() / bank_size).max(1);
        let bank_count_mask = bank_count.is_power_of_two().then_some(bank_count - 1);
        let wraps_index = !rom.is_empty() && rom.len() != bank_count * bank_size;
        Self {
            rom,
            bank_size,
            bank_count,
            bank_size_mask: bank_size - 1,
            bank_count_mask,
            wraps_index,
        }
    }

    pub fn bank_count(&self) -> usize {
        self.bank_count
    }

    #[inline]
    fn normalized_bank(&self, bank: usize) -> usize {
        if let Some(mask) = self.bank_count_mask {
            bank & mask
        } else {
            bank % self.bank_count
        }
    }

    /// Read a byte from `bank` (modulo bank_count) at `addr_in_bank`
    /// (masked to `bank_size - 1`). Returns 0 if the ROM is empty.
    #[inline]
    pub fn read(&self, bank: usize, addr_in_bank: u16) -> u8 {
        if self.rom.is_empty() {
            return 0;
        }
        let bank = self.normalized_bank(bank);
        let idx = bank * self.bank_size + (usize::from(addr_in_bank) & self.bank_size_mask);
        if self.wraps_index {
            self.rom[idx % self.rom.len()]
        } else {
            self.rom[idx]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PrgMemory;

    #[test]
    fn prg_read_masks_power_of_two_bank_count() {
        let prg = PrgMemory::new((0..16).map(|value| value as u8).collect(), 4);

        assert_eq!(prg.bank_count(), 4);
        assert_eq!(prg.read(0, 0x8000), 0);
        assert_eq!(prg.read(2, 0x8000), 8);
        assert_eq!(prg.read(6, 0x8000), 8);
    }

    #[test]
    fn prg_read_modulos_non_power_of_two_bank_count() {
        let prg = PrgMemory::new((0..12).map(|value| value as u8).collect(), 4);

        assert_eq!(prg.bank_count(), 3);
        assert_eq!(prg.read(0, 0x8000), 0);
        assert_eq!(prg.read(2, 0x8000), 8);
        assert_eq!(prg.read(5, 0x8000), 8);
    }

    #[test]
    fn prg_read_preserves_partial_bank_wrap_behavior() {
        let prg = PrgMemory::new((0..3).map(|value| value as u8).collect(), 4);

        assert_eq!(prg.bank_count(), 1);
        assert_eq!(prg.read(0, 0x8002), 2);
        assert_eq!(prg.read(0, 0x8003), 0);
        assert_eq!(prg.read(7, 0x8003), 0);
    }
}
