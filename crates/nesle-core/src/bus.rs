use crate::apu::Apu;
use crate::cpu::InterruptLines;
use crate::input::ControllerPorts;
use crate::mapper::Mapper;
use crate::ppu::Ppu;

/// Bus-side events that CPU execution consumes synchronously.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusEvent {
    OamDma(u8),
}

const WRAM_WRITE_LOG_CAPACITY: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WramWriteLogEntry {
    pub cycle_count: u64,
    pub pc: u16,
    pub addr: u16,
    pub normalized_addr: u16,
    pub old_value: u8,
    pub new_value: u8,
    pub op_type: u8,
    pub a: u8,
    pub x: u8,
    pub y: u8,
    pub sp: u8,
    pub status: u8,
}

#[derive(Debug, Clone)]
pub struct Bus {
    wram: [u8; 0x800],
    cpu_open_bus: u8,
    wram_write_log: Vec<WramWriteLogEntry>,
}

impl Default for Bus {
    fn default() -> Self {
        Self {
            wram: [0xff; 0x800],
            cpu_open_bus: 0,
            wram_write_log: Vec::with_capacity(WRAM_WRITE_LOG_CAPACITY),
        }
    }
}

impl Bus {
    pub fn reset(&mut self) {
        self.wram = [0xff; 0x800];
        self.cpu_open_bus = 0;
        self.clear_wram_write_log();
    }

    pub fn set_wram(&mut self, wram: [u8; 0x800]) {
        self.wram = wram;
    }

    pub fn wram(&self) -> &[u8; 0x800] {
        &self.wram
    }

    pub fn clear_wram_write_log(&mut self) {
        self.wram_write_log.clear();
    }

    pub fn wram_write_log(&self) -> &[WramWriteLogEntry] {
        &self.wram_write_log
    }

    pub(crate) fn record_wram_write(&mut self, entry: WramWriteLogEntry) {
        if self.wram_write_log.len() == WRAM_WRITE_LOG_CAPACITY {
            return;
        }
        self.wram_write_log.push(entry);
    }

    pub fn cpu_open_bus(&self) -> u8 {
        self.cpu_open_bus
    }

    pub fn set_cpu_open_bus(&mut self, value: u8) {
        self.cpu_open_bus = value;
    }

    pub fn snapshot_bytes(&self) -> [u8; 1] {
        [self.cpu_open_bus]
    }

    pub fn restore_snapshot(&mut self, bytes: [u8; 1]) {
        self.cpu_open_bus = bytes[0];
    }
}

/// Per-CPU-cycle bus participants used by CPU reads and writes.
///
/// Phase D: the `M` generic replaces what used to be `&'a mut dyn Mapper`.
/// Production instantiates `M = MapperImpl` (sized; static dispatch through
/// the enum's `match self`); tests instantiate `M = TestMapper` /
/// `ExecTestMapper` / etc. directly (sized; direct dispatch). Both bypass
/// the trait-object vtable. `?Sized` is retained so legacy callers that
/// still hand around `&mut dyn Mapper` continue to type-check during
/// staged migrations, and so future test helpers can pass `&mut dyn Mapper`
/// if they need erasure.
pub struct CpuBusContext<'a, M: Mapper + ?Sized> {
    pub bus: &'a mut Bus,
    pub mapper: &'a mut M,
    pub mapper_has_cpu_clock_hook: bool,
    pub mapper_has_vram_addr_hook: bool,
    pub ppu: &'a mut Ppu,
    pub apu: &'a mut Apu,
    pub controllers: &'a mut ControllerPorts,
    pub interrupt: &'a mut InterruptLines,
    pub cpu_cycle_count: u64,
    /// CPU master clock used by APU/PPU timing callers.
    pub master_clock: u64,
}

/// CPU bus read with open-bus and register side effects.
///
/// Phase E.4: `#[inline]` — entry point of the per-CPU-cycle memory
/// access path, called >1M times/sec at full bench speed. Cross-crate
/// inlining via `codegen-units=1 + thin-LTO` is usually achieved
/// automatically; the explicit hint protects against tree refactors
/// that would inadvertently break it.
#[inline]
pub fn bus_read<M: Mapper + ?Sized>(addr: u16, ctx: &mut CpuBusContext<'_, M>) -> u8 {
    // `$4015` is an internal-only read; it does not update external open bus.
    let value = match addr {
        0x0000..=0x1fff => ctx.bus.wram[usize::from(addr & 0x07ff)],
        0x2000..=0x3fff => {
            ctx.ppu
                .cpu_read_register(0x2000 | (addr & 0x0007), ctx.mapper, ctx.interrupt)
        }
        0x4000..=0x4013 | 0x4015 | 0x4018..=0x401f => {
            ctx.apu
                .cpu_read_open_bus(addr, ctx.bus.cpu_open_bus, ctx.interrupt)
        }
        // Controller reads preserve top three bits from open bus.
        0x4016 => ctx.controllers.read(0) | (ctx.bus.cpu_open_bus & 0xE0),
        0x4017 => ctx.controllers.read(1) | (ctx.bus.cpu_open_bus & 0xE0),
        0x4020..=0xffff => ctx
            .mapper
            .cpu_read_open_bus(addr, ctx.bus.cpu_open_bus, ctx.interrupt),
        // Unmapped reads return current open bus.
        _ => ctx.bus.cpu_open_bus,
    };
    if addr != 0x4015 {
        ctx.bus.cpu_open_bus = value;
    }
    value
}

/// CPU bus write. `$4014` returns an OAM-DMA event for synchronous CPU handling.
/// Phase E.4: see `bus_read` for inline rationale.
#[inline]
pub fn bus_write<M: Mapper + ?Sized>(
    addr: u16,
    value: u8,
    ctx: &mut CpuBusContext<'_, M>,
) -> Option<BusEvent> {
    let event = match addr {
        0x0000..=0x1fff => {
            ctx.bus.wram[usize::from(addr & 0x07ff)] = value;
            None
        }
        0x2000..=0x3fff => {
            let canonical_addr = 0x2000 | (addr & 0x0007);
            let open_bus = ctx.bus.cpu_open_bus;
            ctx.ppu
                .cpu_write_register(canonical_addr, value, open_bus, ctx.mapper, ctx.interrupt);
            ctx.mapper.ppu_register_write(addr, canonical_addr, value);
            None
        }
        0x4000..=0x4013 | 0x4015 | 0x4017 => {
            ctx.apu
                .cpu_write_with_cycle(addr, value, ctx.cpu_cycle_count, ctx.interrupt);
            None
        }
        0x4014 => Some(BusEvent::OamDma(value)),
        0x4016 => {
            // Controller strobe writes are deferred by CPU-cycle parity.
            ctx.controllers
                .queue_strobe_write(value, ctx.cpu_cycle_count);
            None
        }
        0x4020..=0xffff => {
            // BaseMapper.cpp::WriteRam: bus conflict (value AND rom)
            // applied centrally so mappers don't duplicate the logic.
            let final_value = if addr >= 0x8000 && ctx.mapper.has_bus_conflicts() {
                value & ctx.mapper.cpu_read(addr)
            } else {
                value
            };
            ctx.mapper.cpu_write(addr, final_value, ctx.interrupt);
            None
        }
        _ => None,
    };
    ctx.bus.cpu_open_bus = value;
    event
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartridge::Mirroring;

    #[derive(Debug)]
    struct TestMapper {
        last_write: Option<(u16, u8)>,
    }

    impl Mapper for TestMapper {
        fn mapper_id(&self) -> u16 {
            0
        }

        fn name(&self) -> &'static str {
            "TEST"
        }

        fn cpu_read(&mut self, addr: u16) -> u8 {
            (addr >> 8) as u8
        }

        fn cpu_write(&mut self, addr: u16, value: u8, _interrupt: &mut InterruptLines) {
            self.last_write = Some((addr, value));
        }

        fn ppu_read(&mut self, _addr: u16) -> u8 {
            0
        }

        fn debug_ppu_read(&self, _addr: u16) -> u8 {
            0
        }

        fn ppu_write(&mut self, _addr: u16, _value: u8) {}

        fn nametable_mirroring(&self) -> Mirroring {
            Mirroring::Horizontal
        }
    }

    /// Fixture holding all bus participants so each test can construct a
    /// `CpuBusContext` in one line.
    struct Fixture {
        bus: Bus,
        mapper: TestMapper,
        ppu: Ppu,
        apu: Apu,
        controllers: ControllerPorts,
        interrupt: InterruptLines,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                bus: Bus::default(),
                mapper: TestMapper { last_write: None },
                ppu: Ppu::default(),
                apu: Apu::default(),
                controllers: ControllerPorts::default(),
                interrupt: InterruptLines::default(),
            }
        }

        fn ctx(&mut self) -> CpuBusContext<'_, TestMapper> {
            CpuBusContext {
                bus: &mut self.bus,
                mapper: &mut self.mapper,
                mapper_has_cpu_clock_hook: false,
                mapper_has_vram_addr_hook: false,
                ppu: &mut self.ppu,
                apu: &mut self.apu,
                controllers: &mut self.controllers,
                interrupt: &mut self.interrupt,
                cpu_cycle_count: 0,
                master_clock: 0,
            }
        }
    }

    #[test]
    fn mirrors_internal_wram() {
        let mut fx = Fixture::new();
        let mut ctx = fx.ctx();
        bus_write(0x0002, 0xaa, &mut ctx);
        assert_eq!(bus_read(0x0802, &mut ctx), 0xaa);
        assert_eq!(bus_read(0x1802, &mut ctx), 0xaa);
    }

    #[test]
    fn routes_cartridge_space_to_mapper() {
        let mut fx = Fixture::new();
        {
            let mut ctx = fx.ctx();
            assert_eq!(bus_read(0x8123, &mut ctx), 0x81);
            bus_write(0x8123, 0x44, &mut ctx);
        }
        assert_eq!(fx.mapper.last_write, Some((0x8123, 0x44)));
    }

    #[test]
    fn routes_controller_strobe_and_reads() {
        let mut fx = Fixture::new();
        // set_mask writes pending; commit makes mask
        // active (PPU calls commit at scanline 241 in production).
        fx.controllers.set_mask(0, 0b0000_0011);
        fx.controllers.commit_pending_input();
        let mut ctx = fx.ctx();

        // Odd cycle parity makes each bus write flush in one pending-write
        // tick, avoiding the back-to-back overwrite path.
        ctx.cpu_cycle_count = 1;
        ctx.master_clock = 1;
        bus_write(0x4016, 1, &mut ctx);
        ctx.controllers.process_pending_write();
        bus_write(0x4016, 0, &mut ctx);
        ctx.controllers.process_pending_write();
        assert_eq!(bus_read(0x4016, &mut ctx), 1);
        assert_eq!(bus_read(0x4016, &mut ctx), 1);
        assert_eq!(bus_read(0x4016, &mut ctx), 0);
    }

    #[test]
    fn controller_reads_preserve_open_bus_high_bits() {
        let mut fx = Fixture::new();
        fx.controllers.set_mask(0, 0x01);
        fx.controllers.commit_pending_input();
        fx.controllers.write_strobe(1);
        fx.controllers.write_strobe(0);
        fx.bus.set_cpu_open_bus(0xc0);

        let mut ctx = fx.ctx();
        assert_eq!(bus_read(0x4016, &mut ctx), 0xc1);
        assert_eq!(ctx.bus.cpu_open_bus(), 0xc1);
    }

    #[test]
    fn controller_strobe_high_reloads_each_read() {
        let mut fx = Fixture::new();
        fx.controllers.set_mask(0, 0x01);
        fx.controllers.commit_pending_input();
        fx.controllers.write_strobe(1);

        assert_eq!(fx.controllers.read(0), 1);
        assert_eq!(fx.controllers.read(0), 1);
        assert_eq!(fx.controllers.read(0), 1);
    }

    #[test]
    fn writes_to_4014_return_oam_dma_event() {
        // Mesen2-aligned: $4014 write returns BusEvent::OamDma(page);
        // Cpu::memory_write consumes it synchronously and calls
        // run_dma_transfer (NesPpu.cpp:505).
        let mut fx = Fixture::new();
        let mut ctx = fx.ctx();
        let event = bus_write(0x4014, 0x03, &mut ctx);
        assert_eq!(event, Some(BusEvent::OamDma(0x03)));
    }

    #[test]
    fn snapshot_keeps_open_bus() {
        let mut bus = Bus {
            cpu_open_bus: 0x9a,
            ..Bus::default()
        };
        let snapshot = bus.snapshot_bytes();
        bus.cpu_open_bus = 0;
        bus.restore_snapshot(snapshot);
        assert_eq!(bus.cpu_open_bus(), 0x9a);
    }

    #[test]
    fn routes_apu_status_register() {
        // $4015 status read returns: frame IRQ (bit 6), DMC IRQ (bit 7),
        // channel length-counter bits (0..3), DMC active bit (4). NESLE
        // does not yet model channel length counters, so bits 0..3 are
        // always 0. After writing 0xFF to $4015 with bit 4 set, DMC
        // sample loads (size > 0) and the read returns bit 4 = 1, so
        // 0x10. Mesen2 reference: NesApu.cpp `ReadRam` for $4015.
        let mut fx = Fixture::new();
        let mut ctx = fx.ctx();
        bus_write(0x4015, 0xff, &mut ctx);
        assert_eq!(bus_read(0x4015, &mut ctx), 0x30);
    }

    #[test]
    fn apu_status_read_preserves_external_open_bus() {
        let mut fx = Fixture::new();
        fx.bus.set_cpu_open_bus(0xe5);
        let mut ctx = fx.ctx();

        assert_eq!(bus_read(0x4015, &mut ctx), 0x20);
        assert_eq!(ctx.bus.cpu_open_bus(), 0xe5);
        assert_eq!(bus_read(0x4016, &mut ctx), 0xe0);
    }

    #[test]
    fn cpu_writes_update_open_bus() {
        let mut fx = Fixture::new();
        let mut ctx = fx.ctx();

        bus_write(0x0000, 0x5a, &mut ctx);
        assert_eq!(ctx.bus.cpu_open_bus(), 0x5a);
    }
}
