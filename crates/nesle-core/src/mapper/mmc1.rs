use crate::cartridge::{CartridgeImage, Mirroring};
use crate::cpu::InterruptLines;
use crate::mapper::banking::{ChrMemory, PrgMemory};
use crate::mapper::traits::{Mapper, Mmc1State};
use crate::mapper::{restore_prg_ram, snapshot_prg_ram};
use nesle_common::{NesleError, Result};

#[derive(Debug)]
pub struct Mmc1 {
    prg: PrgMemory,
    chr: ChrMemory,
    shift: u8,
    shift_count: u8,
    control: u8,
    chr_bank0: u8,
    chr_bank1: u8,
    prg_bank: u8,
    /// CPU-cycle counter for the MMC1 consecutive-write gate.
    cycle_count: u64,
    /// Last MMC1 register-write cycle; same-cycle writes are ignored.
    last_write_cycle: u64,
    /// Literal CHR-register write address used by `extra_reg`.
    last_chr_reg: u16,
    /// $E000 bit 4 WRAM-disable latch; ignored on MMC1A.
    wram_disable: bool,
    /// MMC1A boards keep WRAM enabled regardless of `wram_disable`.
    force_wram_on: bool,
    /// Battery-backed 8KB half on SOROM; empty otherwise.
    save_ram: Vec<u8>,
    /// Volatile WRAM: SOROM 8KB, SXROM 32KB, or plain 8KB.
    work_ram: Vec<u8>,
    /// PRG ROM byte length for SUROM 512KB banking.
    prg_size: usize,
}

impl Mmc1 {
    pub fn new(cartridge: &CartridgeImage) -> Self {
        let prg_size = cartridge.prg_rom.len();
        let save_ram_size = cartridge.save_ram_size;
        let work_ram_size = cartridge.work_ram_size;
        // MMC1A (NES 2.0 submapper 1) forces WRAM on. Other boards start
        // enabled unless a later $E000 write disables them.
        let (force_wram_on, wram_disable_init) = match cartridge.submapper {
            1 => (true, false),
            _ => (false, false),
        };
        // SOROM uses 8KB save + 8KB work; SXROM uses a 32KB WRAM pool.
        let (save_ram, work_ram) = if save_ram_size == 0x2000 && work_ram_size == 0x2000 {
            (vec![0xff; 0x2000], vec![0xff; 0x2000])
        } else if save_ram_size + work_ram_size > 0x2000 {
            (Vec::new(), vec![0xff; save_ram_size + work_ram_size])
        } else if save_ram_size + work_ram_size > 0 {
            (Vec::new(), vec![0xff; 0x2000])
        } else {
            (Vec::new(), Vec::new())
        };
        Self {
            prg: PrgMemory::new(cartridge.prg_rom.clone(), 16 * 1024),
            chr: ChrMemory::new(cartridge, 8 * 1024),
            shift: 0,
            shift_count: 0,
            // Power-on register state: control = 0x0C (PRG mode 3 =
            // fix-last 16KB bank at $C000-$FFFF), all other regs zero.
            // Banking is computed lazily on each access.
            control: 0x0c,
            chr_bank0: 0,
            chr_bank1: 0,
            prg_bank: 0,
            // Start at -1 so reset warmup leaves the write gate at cycle 7.
            cycle_count: u64::MAX,
            last_write_cycle: 0,
            // `extra_reg` depends on the literal last CHR-register address.
            last_chr_reg: 0xa000,
            wram_disable: wram_disable_init,
            force_wram_on,
            save_ram,
            work_ram,
            prg_size,
        }
    }

    /// CHR-side extension register for SUROM PRG and SXROM/SOROM WRAM banking.
    fn extra_reg(&self) -> u8 {
        if self.last_chr_reg == 0xc000 && (self.control & 0x10) != 0 {
            self.chr_bank1
        } else {
            self.chr_bank0
        }
    }

    fn prg_bank_for(&self, addr: u16) -> usize {
        let mode = (self.control >> 2) & 0x03;
        let bank = usize::from(self.prg_bank & 0x0f);
        // SUROM 512KB carts use extra_reg bit 4 to select the 256KB half.
        let prg_bank_select = if self.prg_size == 0x80000 {
            usize::from(self.extra_reg() & 0x10)
        } else {
            0
        };
        match (mode, addr) {
            // 32KB mode 0/1: low bit forced to 0 (bank pair); high half uses bank+1.
            (0 | 1, 0x8000..=0xbfff) => (bank & !1) | prg_bank_select,
            (0 | 1, 0xc000..=0xffff) => (bank & !1) | 1 | prg_bank_select,
            // Mode 2: fix bank 0 at $8000-$BFFF, switchable at $C000-$FFFF.
            (2, 0x8000..=0xbfff) => prg_bank_select,
            (2, 0xc000..=0xffff) => bank | prg_bank_select,
            // Mode 3: switch at $8000, fix the last bank inside the selected half.
            (_, 0x8000..=0xbfff) => bank | prg_bank_select,
            (_, 0xc000..=0xffff) => 0x0f | prg_bank_select,
            _ => 0,
        }
    }

    fn chr_offset(&self, addr: u16) -> usize {
        let four_k_banks = self.chr.bank_count(0x1000);
        if self.control & 0x10 == 0 {
            let bank = (usize::from(self.chr_bank0) & !1) % four_k_banks;
            bank * 0x1000 + usize::from(addr & 0x1fff)
        } else {
            let bank = if addr < 0x1000 {
                usize::from(self.chr_bank0)
            } else {
                usize::from(self.chr_bank1)
            } % four_k_banks;
            bank * 0x1000 + usize::from(addr & 0x0fff)
        }
    }

    /// `$6000-$7FFF` read path; `None` means disabled WRAM/open bus.
    fn read_ram(&self, addr: u16) -> Option<u8> {
        if self.wram_disable && !self.force_wram_on {
            return None;
        }
        let offset = usize::from(addr - 0x6000);
        if !self.save_ram.is_empty() && self.work_ram.len() == 0x2000 {
            // SOROM
            if (self.extra_reg() >> 3) & 0x01 != 0 {
                Some(self.work_ram[offset])
            } else {
                Some(self.save_ram[offset])
            }
        } else if self.work_ram.len() == 0x8000 {
            // SXROM 32KB
            let bank = usize::from((self.extra_reg() >> 2) & 0x03);
            Some(self.work_ram[bank * 0x2000 + offset])
        } else if !self.work_ram.is_empty() {
            Some(self.work_ram[offset % self.work_ram.len()])
        } else if !self.save_ram.is_empty() {
            Some(self.save_ram[offset % self.save_ram.len()])
        } else {
            None
        }
    }

    fn write_ram(&mut self, addr: u16, value: u8) {
        if self.wram_disable && !self.force_wram_on {
            return;
        }
        let offset = usize::from(addr - 0x6000);
        if !self.save_ram.is_empty() && self.work_ram.len() == 0x2000 {
            if (self.extra_reg() >> 3) & 0x01 != 0 {
                self.work_ram[offset] = value;
            } else {
                self.save_ram[offset] = value;
            }
        } else if self.work_ram.len() == 0x8000 {
            let bank = usize::from((self.extra_reg() >> 2) & 0x03);
            self.work_ram[bank * 0x2000 + offset] = value;
        } else if !self.work_ram.is_empty() {
            let len = self.work_ram.len();
            self.work_ram[offset % len] = value;
        } else if !self.save_ram.is_empty() {
            let len = self.save_ram.len();
            self.save_ram[offset % len] = value;
        }
    }
}

impl Mapper for Mmc1 {
    fn mapper_id(&self) -> u16 {
        1
    }

    fn name(&self) -> &'static str {
        "MMC1"
    }

    fn cpu_read(&mut self, addr: u16) -> u8 {
        // Pure-read path for bus conflicts and DMC fetches.
        if (0x6000..=0x7fff).contains(&addr) {
            return self.read_ram(addr).unwrap_or(0);
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
        if (0x6000..=0x7fff).contains(&addr) {
            return self.read_ram(addr).unwrap_or(open_bus);
        }
        if addr < 0x8000 {
            return open_bus;
        }
        self.prg.read(self.prg_bank_for(addr), addr)
    }

    fn cpu_write(&mut self, addr: u16, value: u8, _interrupt: &mut InterruptLines) {
        if (0x6000..=0x7fff).contains(&addr) {
            self.write_ram(addr, value);
            return;
        }
        if addr < 0x8000 {
            return;
        }
        if value & 0x80 != 0 {
            // Reset-bit writes bypass the consecutive-write gate.
            self.shift = 0;
            self.shift_count = 0;
            self.control |= 0x0c;
            self.last_write_cycle = self.cycle_count;
            return;
        }
        // Ignore same-cycle writes only; adjacent RMW writes still count.
        if self.cycle_count == self.last_write_cycle {
            self.last_write_cycle = self.cycle_count;
            return;
        }
        self.shift |= (value & 0x01) << self.shift_count;
        self.shift_count += 1;
        if self.shift_count == 5 {
            match addr {
                0x8000..=0x9fff => self.control = self.shift,
                0xa000..=0xbfff => {
                    self.chr_bank0 = self.shift;
                    // Store the literal address; `extra_reg` checks exact $C000.
                    self.last_chr_reg = addr;
                }
                0xc000..=0xdfff => {
                    self.chr_bank1 = self.shift;
                    self.last_chr_reg = addr;
                }
                0xe000..=0xffff => {
                    self.prg_bank = self.shift;
                    self.wram_disable = (self.shift & 0x10) != 0;
                }
                _ => {}
            }
            self.shift = 0;
            self.shift_count = 0;
        }
        self.last_write_cycle = self.cycle_count;
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
        match self.control & 0x03 {
            0 => Mirroring::SingleScreenLower,
            1 => Mirroring::SingleScreenUpper,
            2 => Mirroring::Vertical,
            _ => Mirroring::Horizontal,
        }
    }

    fn process_cpu_clock(&mut self, _interrupt: &mut InterruptLines) {
        self.cycle_count = self.cycle_count.wrapping_add(1);
    }

    fn has_cpu_clock_hook(&self) -> bool {
        true
    }

    fn soft_reset(&mut self) {
        // Soft reset clears the shift register and forces PRG mode 3.
        self.shift = 0;
        self.shift_count = 0;
        self.control |= 0x0c;
    }

    /// Internal MMC1 register snapshot for cross-emulator diagnostics.
    fn mmc1_state(&self) -> Option<Mmc1State> {
        let mirroring = match self.control & 0x03 {
            0 => 0, // SingleScreenLower
            1 => 1, // SingleScreenUpper
            2 => 2, // Vertical
            _ => 3, // Horizontal
        };
        Some(Mmc1State {
            write_buffer: self.shift,
            shift_count: self.shift_count,
            control: self.control,
            chr_reg0: self.chr_bank0,
            chr_reg1: self.chr_bank1,
            prg_reg: self.prg_bank,
            last_write_cycle: self.last_write_cycle,
            last_chr_reg: self.last_chr_reg,
            wram_disable: self.wram_disable,
            force_wram_on: self.force_wram_on,
            chr_mode: (self.control & 0x10) != 0,
            prg_mode: (self.control & 0x08) != 0,
            slot_select: (self.control & 0x04) != 0,
            mirroring,
        })
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![
            self.shift,
            self.shift_count,
            self.control,
            self.chr_bank0,
            self.chr_bank1,
            self.prg_bank,
        ];
        bytes.extend_from_slice(&self.last_chr_reg.to_le_bytes());
        bytes.push(u8::from(self.wram_disable));
        bytes.push(u8::from(self.force_wram_on));
        snapshot_prg_ram(&mut bytes, &self.save_ram);
        snapshot_prg_ram(&mut bytes, &self.work_ram);
        self.chr.snapshot_bytes(&mut bytes);
        bytes.extend_from_slice(&self.cycle_count.to_le_bytes());
        bytes.extend_from_slice(&self.last_write_cycle.to_le_bytes());
        bytes
    }

    fn restore_snapshot(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() < 10 {
            return Err(NesleError::InvalidState(
                "MMC1 snapshot is missing register bytes".to_string(),
            ));
        }
        self.shift = bytes[0];
        self.shift_count = bytes[1];
        self.control = bytes[2];
        self.chr_bank0 = bytes[3];
        self.chr_bank1 = bytes[4];
        self.prg_bank = bytes[5];
        self.last_chr_reg = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        self.wram_disable = bytes[8] != 0;
        self.force_wram_on = bytes[9] != 0;
        let mut offset = restore_prg_ram(bytes, 10, &mut self.save_ram, "MMC1 SaveRam")?;
        offset = restore_prg_ram(bytes, offset, &mut self.work_ram, "MMC1 WorkRam")?;
        self.chr.restore_snapshot(bytes, &mut offset, "MMC1")?;
        // Single-version snapshot format; truncation is a hard error.
        if bytes.len() < offset + 16 {
            return Err(NesleError::InvalidState(
                "MMC1 snapshot truncated: missing cycle_count fields".to_string(),
            ));
        }
        self.cycle_count = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        self.last_write_cycle =
            u64::from_le_bytes(bytes[offset + 8..offset + 16].try_into().unwrap());
        Ok(())
    }
}
