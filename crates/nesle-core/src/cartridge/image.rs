#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartridgeFormat {
    INes,
    Nes20,
    Unif,
    Fds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mirroring {
    Horizontal,
    Vertical,
    FourScreen,
    SingleScreenLower,
    SingleScreenUpper,
}

/// Console region -drives master_clock_divider, APU frame counter step
/// table, MMC3 A12 threshold, and PPU scanline count. Mesen2 equivalent:
/// `ConsoleRegion` enum (Core/Shared/ConsoleRegion.h).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    #[default]
    Ntsc,
    Pal,
    Dendy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CartridgeImage {
    pub format: CartridgeFormat,
    pub mapper_id: u16,
    pub submapper: u8,
    pub mirroring: Mirroring,
    pub battery: bool,
    pub region: Region,
    pub prg_rom: Vec<u8>,
    pub chr_rom: Vec<u8>,
    pub trainer_data: Vec<u8>,
    pub work_ram_size: usize,
    pub save_ram_size: usize,
    pub prg_ram_size: usize,
    pub prg_ram_unspecified: bool,
    pub save_chr_ram_size: usize,
    pub chr_ram_size: usize,
    pub chr_ram_unspecified: bool,
    /// NES 2.0 header byte 15 bits 0-5 ("input device"). Mesen2 reads the
    /// equivalent from GameDB `InputType` (NesTypes.h:387 GameInputType
    /// enum). Relevant values for the supported corpus:
    ///   0 = unspecified (default for iNES 1.0)
    ///   1 = Standard NES/Famicom controllers
    ///   2 = NES Four Score / Satellite (4-player adapter)
    ///   3 = Famicom 4-player adapter
    /// Other values (Zapper, Power Pad, etc.) are out of NESLE scope.
    pub input_device: u8,
}

impl CartridgeImage {
    /// Mesen2 `RomData.Info.IsNes20`. True iff the cartridge was parsed
    /// from a NES 2.0 header (vs legacy iNES). Some mapper variants and
    /// region detection rules only apply to NES 2.0.
    pub fn is_nes20(&self) -> bool {
        matches!(self.format, CartridgeFormat::Nes20)
    }

    /// Mesen2 `RomData.Info.HasTrainer`. True iff the cartridge has the
    /// optional 512-byte trainer payload (loaded to `$7000` on power-on).
    pub fn has_trainer(&self) -> bool {
        !self.trainer_data.is_empty()
    }

    pub fn initialized_prg_ram(&self, min_size: usize) -> Vec<u8> {
        let header_size = self
            .prg_ram_size
            .max(self.work_ram_size + self.save_ram_size);
        let mapper_default = if self.prg_ram_unspecified {
            min_size
        } else {
            0
        };
        let trainer_size = if self.trainer_data.is_empty() {
            0
        } else {
            0x1000 + self.trainer_data.len()
        };
        let size = header_size.max(mapper_default).max(trainer_size);
        let mut ram = vec![0xff; size];
        if !self.trainer_data.is_empty() {
            let offset = 0x1000;
            let end = (offset + self.trainer_data.len()).min(ram.len());
            ram[offset..end].copy_from_slice(&self.trainer_data[..end - offset]);
        }
        ram
    }

    pub fn total_chr_ram_size(&self) -> usize {
        self.chr_ram_size + self.save_chr_ram_size
    }
}
