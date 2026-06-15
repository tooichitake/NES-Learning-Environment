pub mod axrom;
pub mod banking;
pub mod cnrom;
pub mod mmc1;
pub mod mmc2;
pub mod mmc3;
pub mod mmc5;
pub mod nrom;
pub mod sunsoft5;
pub mod traits;
pub mod uxrom;

use nesle_common::{NesleError, Result};

use crate::cartridge::{CartridgeImage, Mirroring, Region};
use crate::cpu::InterruptLines;

pub use traits::{Mapper, Mmc1State};

// ============================================================
// Phase D: static-dispatch mapper enum (replaces `Box<dyn Mapper>`
// on the production hot path — see plans/skills-agents-workflows-...md
// and docs/perf-baseline.md).
//
// Variants mirror the supported mapper set (SUPPORTED_MAPPERS in
// scripts/local_rom_probe.py = {0, 1, 2, 3, 4, 5, 7, 9, 69}). Mmc5 is
// the largest by far (~970 LOC, 35+ fields) so it is boxed; the rest
// stay inline so the enum size stays close to the largest non-Mmc5
// variant. The dispatch goes through `impl Mapper for MapperImpl`
// below: each method is a single `match self { ... }`, which the
// compiler with codegen-units=1 + thin-LTO can inline and devirtualize
// at the call sites in bus.rs / ppu.rs.
//
// Test mocks (TestMapper / RecordingMapper / ExecTestMapper across
// apu.rs / bus.rs / cpu.rs / ppu.rs tests) do NOT participate in
// `MapperImpl` — they stay as their own concrete types and are passed
// to hot-path functions as `&mut M` where `M: Mapper + ?Sized`. This
// keeps the production enum at nine variants and lets each test mock
// retain its own fields (`last_write`, `chr`, etc.) for assertions
// after bus operations.
// ============================================================

/// Sized enum that owns one concrete mapper. Implements [`Mapper`] via
/// static match-dispatch over each variant; this replaces the
/// `Box<dyn Mapper>` trait-object that production used to hold.
#[derive(Debug)]
pub enum MapperImpl {
    Nrom(nrom::Nrom),
    Mmc1(mmc1::Mmc1),
    Uxrom(uxrom::Uxrom),
    Cnrom(cnrom::Cnrom),
    Mmc3(mmc3::Mmc3),
    /// Boxed because `Mmc5` (35+ fields incl. ExRAM, audio, vsplit) would
    /// otherwise inflate every `MapperImpl` value to its size on the stack.
    Mmc5(Box<mmc5::Mmc5>),
    Axrom(axrom::Axrom),
    Mmc2(mmc2::Mmc2),
    Sunsoft5(sunsoft5::Sunsoft5),
}

/// Construct a [`MapperImpl`] from a cartridge image. Replaces the
/// pre-Phase-D `Box<dyn Mapper>` factory; the type alias `MapperRef`
/// and the old factory of the same name were retired alongside the
/// trait-object hot path.
pub fn create_mapper(cartridge: &CartridgeImage) -> Result<MapperImpl> {
    match cartridge.mapper_id {
        0 => Ok(MapperImpl::Nrom(nrom::Nrom::new(cartridge))),
        1 => Ok(MapperImpl::Mmc1(mmc1::Mmc1::new(cartridge))),
        2 => Ok(MapperImpl::Uxrom(uxrom::Uxrom::new(cartridge))),
        3 => Ok(MapperImpl::Cnrom(cnrom::Cnrom::new(cartridge))),
        4 => Ok(MapperImpl::Mmc3(mmc3::Mmc3::new(cartridge))),
        5 => Ok(MapperImpl::Mmc5(Box::new(mmc5::Mmc5::new(cartridge)))),
        7 => Ok(MapperImpl::Axrom(axrom::Axrom::new(cartridge))),
        9 => Ok(MapperImpl::Mmc2(mmc2::Mmc2::new(cartridge))),
        10 => Ok(MapperImpl::Mmc2(mmc2::Mmc2::new_mmc4(cartridge))),
        69 => Ok(MapperImpl::Sunsoft5(sunsoft5::Sunsoft5::new(cartridge))),
        mapper => Err(NesleError::UnsupportedMapper(mapper)),
    }
}

/// Dispatch helper: forward a trait-method call to whichever concrete
/// mapper variant `self` contains. Two-armed because `Mapper` has both
/// `&self` and `&mut self` methods, and the macro needs different match
/// keywords to thread the inner binding through to each variant.
///
/// Each variant arm uses Rust auto-deref so `Box<Mmc5>` and
/// `Box<dyn Mapper>` (test only) work identically to inline variants
/// at the source level. The compiler with codegen-units=1 + thin-LTO
/// devirtualizes the match into a jump table at the call sites that
/// matter (bus_read / bus_write / Ppu::read_vram / Ppu::set_bus_address).
macro_rules! dispatch_mapper {
    (& $self:ident => |$m:ident| $body:expr) => {
        match $self {
            Self::Nrom($m) => $body,
            Self::Mmc1($m) => $body,
            Self::Uxrom($m) => $body,
            Self::Cnrom($m) => $body,
            Self::Mmc3($m) => $body,
            Self::Mmc5($m) => $body,
            Self::Axrom($m) => $body,
            Self::Mmc2($m) => $body,
            Self::Sunsoft5($m) => $body,
        }
    };
    (&mut $self:ident => |$m:ident| $body:expr) => {
        match $self {
            Self::Nrom($m) => $body,
            Self::Mmc1($m) => $body,
            Self::Uxrom($m) => $body,
            Self::Cnrom($m) => $body,
            Self::Mmc3($m) => $body,
            Self::Mmc5($m) => $body,
            Self::Axrom($m) => $body,
            Self::Mmc2($m) => $body,
            Self::Sunsoft5($m) => $body,
        }
    };
}

impl Mapper for MapperImpl {
    fn mapper_id(&self) -> u16 {
        dispatch_mapper!(& self => |m| m.mapper_id())
    }

    fn name(&self) -> &'static str {
        dispatch_mapper!(& self => |m| m.name())
    }

    fn cpu_read(&mut self, addr: u16) -> u8 {
        dispatch_mapper!(&mut self => |m| m.cpu_read(addr))
    }

    fn cpu_code_read(&self, addr: u16) -> Option<u8> {
        dispatch_mapper!(& self => |m| m.cpu_code_read(addr))
    }

    fn cpu_read_open_bus(
        &mut self,
        addr: u16,
        open_bus: u8,
        interrupt: &mut InterruptLines,
    ) -> u8 {
        dispatch_mapper!(&mut self => |m| m.cpu_read_open_bus(addr, open_bus, interrupt))
    }

    fn cpu_write(&mut self, addr: u16, value: u8, interrupt: &mut InterruptLines) {
        dispatch_mapper!(&mut self => |m| m.cpu_write(addr, value, interrupt))
    }

    fn ppu_read(&mut self, addr: u16) -> u8 {
        dispatch_mapper!(&mut self => |m| m.ppu_read(addr))
    }

    fn debug_ppu_read(&self, addr: u16) -> u8 {
        dispatch_mapper!(& self => |m| m.debug_ppu_read(addr))
    }

    fn ppu_write(&mut self, addr: u16, value: u8) {
        dispatch_mapper!(&mut self => |m| m.ppu_write(addr, value))
    }

    fn ppu_read_nametable(
        &mut self,
        addr: u16,
        ciram: &[u8; 0x1000],
        interrupt: &mut InterruptLines,
    ) -> Option<u8> {
        dispatch_mapper!(&mut self => |m| m.ppu_read_nametable(addr, ciram, interrupt))
    }

    fn ppu_write_nametable(
        &mut self,
        addr: u16,
        value: u8,
        ciram: &mut [u8; 0x1000],
    ) -> bool {
        dispatch_mapper!(&mut self => |m| m.ppu_write_nametable(addr, value, ciram))
    }

    fn ppu_register_write(&mut self, original_addr: u16, canonical_addr: u16, value: u8) {
        dispatch_mapper!(&mut self => |m| m.ppu_register_write(original_addr, canonical_addr, value))
    }

    fn nametable_mirroring(&self) -> Mirroring {
        dispatch_mapper!(& self => |m| m.nametable_mirroring())
    }

    fn process_cpu_clock(&mut self, interrupt: &mut InterruptLines) {
        dispatch_mapper!(&mut self => |m| m.process_cpu_clock(interrupt))
    }

    fn has_cpu_clock_hook(&self) -> bool {
        dispatch_mapper!(& self => |m| m.has_cpu_clock_hook())
    }

    fn notify_vram_addr(
        &mut self,
        addr: u16,
        cpu_cycle_count: u64,
        interrupt: &mut InterruptLines,
    ) {
        dispatch_mapper!(&mut self => |m| m.notify_vram_addr(addr, cpu_cycle_count, interrupt))
    }

    fn has_vram_addr_hook(&self) -> bool {
        dispatch_mapper!(& self => |m| m.has_vram_addr_hook())
    }

    fn has_bus_conflicts(&self) -> bool {
        dispatch_mapper!(& self => |m| m.has_bus_conflicts())
    }

    fn soft_reset(&mut self) {
        dispatch_mapper!(&mut self => |m| m.soft_reset())
    }

    fn set_region(&mut self, region: Region) {
        dispatch_mapper!(&mut self => |m| m.set_region(region))
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        dispatch_mapper!(& self => |m| m.snapshot_bytes())
    }

    fn restore_snapshot(&mut self, bytes: &[u8]) -> Result<()> {
        dispatch_mapper!(&mut self => |m| m.restore_snapshot(bytes))
    }

    fn mmc1_state(&self) -> Option<Mmc1State> {
        dispatch_mapper!(& self => |m| m.mmc1_state())
    }
}

/// Serialize PRG RAM bytes prefixed with their length (4 bytes u32 LE).
/// Shared mapper snapshot helper (avoids 8x duplicated boilerplate).
pub(crate) fn snapshot_prg_ram(bytes: &mut Vec<u8>, prg_ram: &[u8]) {
    bytes.extend_from_slice(&(prg_ram.len() as u32).to_le_bytes());
    bytes.extend_from_slice(prg_ram);
}

/// Restore PRG RAM from a snapshot, verifying length matches the existing
/// buffer. Returns the new offset past the consumed bytes (4 + len).
pub(crate) fn restore_prg_ram(
    bytes: &[u8],
    offset: usize,
    prg_ram: &mut [u8],
    mapper_name: &str,
) -> Result<usize> {
    if bytes.len() < offset + 4 {
        return Err(NesleError::InvalidState(format!(
            "{mapper_name} snapshot is missing PRG RAM length"
        )));
    }
    let len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    if len != prg_ram.len() || bytes.len() < offset + 4 + len {
        return Err(NesleError::InvalidState(format!(
            "{mapper_name} PRG RAM snapshot length mismatch"
        )));
    }
    prg_ram.copy_from_slice(&bytes[offset + 4..offset + 4 + len]);
    Ok(offset + 4 + len)
}
