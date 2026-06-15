use std::fmt::Debug;

use crate::cartridge::{Mirroring, Region};
use crate::cpu::InterruptLines;
use nesle_common::Result;

/// Per-instance mapper interface. Mappers push IRQ changes through
/// `InterruptLines`; there are no mapper globals or TLS.
pub trait Mapper: Debug + Send {
    fn mapper_id(&self) -> u16;
    fn name(&self) -> &'static str;

    fn cpu_read(&mut self, addr: u16) -> u8;
    /// Side-effect-free CPU instruction fetch for pure PRG ROM addresses.
    /// Returning `Some(byte)` lets the CPU skip the generic bus/mapper read
    /// after it has still run the full per-cycle timing sequence. Mappers must
    /// return `None` for PRG RAM, registers, open bus, or any read with side
    /// effects.
    fn cpu_code_read(&self, _addr: u16) -> Option<u8> {
        None
    }
    /// CPU bus read with open-bus carry-in and interrupt-line access.
    /// Mappers with read-side IRQ acknowledgements override this.
    fn cpu_read_open_bus(
        &mut self,
        addr: u16,
        _open_bus: u8,
        _interrupt: &mut InterruptLines,
    ) -> u8 {
        self.cpu_read(addr)
    }
    fn cpu_write(&mut self, addr: u16, value: u8, interrupt: &mut InterruptLines);

    fn ppu_read(&mut self, addr: u16) -> u8;

    /// Side-effect-free CHR read at the current bank mapping.
    fn debug_ppu_read(&self, addr: u16) -> u8;

    fn ppu_write(&mut self, addr: u16, value: u8);
    fn ppu_read_nametable(
        &mut self,
        _addr: u16,
        _ciram: &[u8; 0x1000],
        _interrupt: &mut InterruptLines,
    ) -> Option<u8> {
        None
    }
    fn ppu_write_nametable(&mut self, _addr: u16, _value: u8, _ciram: &mut [u8; 0x1000]) -> bool {
        false
    }
    fn ppu_register_write(&mut self, _original_addr: u16, _canonical_addr: u16, _value: u8) {}

    fn nametable_mirroring(&self) -> Mirroring;

    /// Called once per CPU cycle for mappers with timers or IRQ counters.
    fn process_cpu_clock(&mut self, _interrupt: &mut InterruptLines) {}

    /// Capability query: true if the mapper actually uses `process_cpu_clock`.
    /// CPU active path skips the call when false (perf optimization).
    fn has_cpu_clock_hook(&self) -> bool {
        false
    }

    /// Called whenever the 14-bit PPU VRAM bus address changes.
    /// `cpu_cycle_count` is the CPU-cycle counter used by mapper IRQ filters.
    fn notify_vram_addr(
        &mut self,
        _addr: u16,
        _cpu_cycle_count: u64,
        _interrupt: &mut InterruptLines,
    ) {
    }

    /// Capability query: true if the mapper actually uses `notify_vram_addr`.
    /// PPU `set_bus_address` skips the call when false.
    fn has_vram_addr_hook(&self) -> bool {
        false
    }

    /// True if `$8000-$FFFF` writes go through bus conflict.
    fn has_bus_conflicts(&self) -> bool {
        false
    }

    /// Soft reset clears volatile mapper state while preserving cartridge data.
    fn soft_reset(&mut self) {}

    /// Refresh region-aware mapper timing tables. Default no-op.
    fn set_region(&mut self, _region: Region) {}

    fn snapshot_bytes(&self) -> Vec<u8> {
        Vec::new()
    }
    fn restore_snapshot(&mut self, _bytes: &[u8]) -> Result<()> {
        Ok(())
    }

    /// Optional MMC1 internal register snapshot for diagnostics.
    fn mmc1_state(&self) -> Option<Mmc1State> {
        None
    }
}

/// MMC1 internal register snapshot for cross-emulator diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mmc1State {
    pub write_buffer: u8,
    pub shift_count: u8,
    /// Reconstructed `$8000` control byte: bits 0-1 mirroring + bit 2
    /// slot_select + bit 3 prg_mode + bit 4 chr_mode.
    pub control: u8,
    pub chr_reg0: u8,
    pub chr_reg1: u8,
    pub prg_reg: u8,
    pub last_write_cycle: u64,
    pub last_chr_reg: u16,
    pub wram_disable: bool,
    pub force_wram_on: bool,
    pub chr_mode: bool,
    pub prg_mode: bool,
    pub slot_select: bool,
    /// 0=ScreenA 1=ScreenB 2=Vertical 3=Horizontal 0xFF=other.
    pub mirroring: u8,
}
