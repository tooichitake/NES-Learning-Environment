use crate::bus::{bus_read, bus_write, BusEvent, CpuBusContext, WramWriteLogEntry};
use crate::mapper::Mapper;

/// CPU bus operation tag used by memory access and DMA timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOperationType {
    /// Real CPU bus read.
    Read,
    /// Real CPU bus write.
    Write,
    /// Opcode fetch; PAL DMA halt gate only accepts this access class.
    ExecOpCode,
    /// Operand fetch after the opcode byte.
    ExecOperand,
    /// Hardware dummy read with no semantic data use.
    DummyRead,
    /// Hardware dummy write, e.g. the old-value write in RMW opcodes.
    DummyWrite,
}

impl MemoryOperationType {
    pub(crate) fn tag(self) -> u8 {
        match self {
            Self::Read => 0,
            Self::Write => 1,
            Self::ExecOpCode => 2,
            Self::ExecOperand => 3,
            Self::DummyRead => 6,
            Self::DummyWrite => 7,
        }
    }
}

/// 6502 addressing mode; ordering separates value and address modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AddrMode {
    /// Special per-opcode handling such as JSR/BRK or unstable stores.
    None,
    /// Implied; still performs the post-opcode dummy read.
    Imp,
    /// Accumulator, with implied-mode dummy-read timing.
    Acc,
    /// Immediate operand byte.
    Imm,
    /// Signed branch offset.
    Rel,
    /// Zero page address.
    Zero,
    /// Zero page indexed by X, including its dummy read.
    ZeroX,
    /// Zero page indexed by Y, including its dummy read.
    ZeroY,
    /// JMP-only indirect mode, including the `$xxFF` page-wrap bug.
    Ind,
    /// `(Indirect,X)`, including its dummy read.
    IndX,
    /// `(Indirect),Y` read form.
    IndY,
    /// `(Indirect),Y` write form; always performs a dummy read.
    IndYW,
    /// Absolute 16-bit address.
    Abs,
    /// Absolute,X read form.
    AbsX,
    /// Absolute,X write form; always performs a dummy read.
    AbsXW,
    /// Absolute,Y read form.
    AbsY,
    /// Absolute,Y write form; always performs a dummy read.
    AbsYW,
}

/// 256-entry opcode -> addressing mode lookup table.
/// Mesen2 parity is pinned by tests in this module.
pub static ADDR_MODE: [AddrMode; 256] = {
    use AddrMode::*;
    [
        // 0x00 .. 0x0F
        Imp, IndX, None, IndX, Zero, Zero, Zero, Zero, Imp, Imm, Acc, Imm, Abs, Abs, Abs, Abs,
        // 0x10 .. 0x1F
        Rel, IndY, None, IndYW, ZeroX, ZeroX, ZeroX, ZeroX, Imp, AbsY, Imp, AbsYW, AbsX, AbsX,
        AbsXW, AbsXW, // 0x20 .. 0x2F  (note: 0x20 JSR uses `None` per Mesen2)
        None, IndX, None, IndX, Zero, Zero, Zero, Zero, Imp, Imm, Acc, Imm, Abs, Abs, Abs, Abs,
        // 0x30 .. 0x3F
        Rel, IndY, None, IndYW, ZeroX, ZeroX, ZeroX, ZeroX, Imp, AbsY, Imp, AbsYW, AbsX, AbsX,
        AbsXW, AbsXW, // 0x40 .. 0x4F
        Imp, IndX, None, IndX, Zero, Zero, Zero, Zero, Imp, Imm, Acc, Imm, Abs, Abs, Abs, Abs,
        // 0x50 .. 0x5F
        Rel, IndY, None, IndYW, ZeroX, ZeroX, ZeroX, ZeroX, Imp, AbsY, Imp, AbsYW, AbsX, AbsX,
        AbsXW, AbsXW, // 0x60 .. 0x6F  (note: 0x6C JMP (ind) uses `Ind`)
        Imp, IndX, None, IndX, Zero, Zero, Zero, Zero, Imp, Imm, Acc, Imm, Ind, Abs, Abs, Abs,
        // 0x70 .. 0x7F
        Rel, IndY, None, IndYW, ZeroX, ZeroX, ZeroX, ZeroX, Imp, AbsY, Imp, AbsYW, AbsX, AbsX,
        AbsXW, AbsXW, // 0x80 .. 0x8F  (NES STA-family / unofficial Imm)
        Imm, IndX, Imm, IndX, Zero, Zero, Zero, Zero, Imp, Imm, Imp, Imm, Abs, Abs, Abs, Abs,
        // 0x90 .. 0x9F  (note: 0x93 SHAZ / 0x9B TAS / 0x9C SHY / 0x9E SHX / 0x9F SHAA use `None`)
        Rel, IndYW, None, None, ZeroX, ZeroX, ZeroY, ZeroY, Imp, AbsYW, Imp, None, None, AbsXW,
        None, None, // 0xA0 .. 0xAF
        Imm, IndX, Imm, IndX, Zero, Zero, Zero, Zero, Imp, Imm, Imp, Imm, Abs, Abs, Abs, Abs,
        // 0xB0 .. 0xBF
        Rel, IndY, None, IndY, ZeroX, ZeroX, ZeroY, ZeroY, Imp, AbsY, Imp, AbsY, AbsX, AbsX, AbsY,
        AbsY, // 0xC0 .. 0xCF
        Imm, IndX, Imm, IndX, Zero, Zero, Zero, Zero, Imp, Imm, Imp, Imm, Abs, Abs, Abs, Abs,
        // 0xD0 .. 0xDF
        Rel, IndY, None, IndYW, ZeroX, ZeroX, ZeroX, ZeroX, Imp, AbsY, Imp, AbsYW, AbsX, AbsX,
        AbsXW, AbsXW, // 0xE0 .. 0xEF
        Imm, IndX, Imm, IndX, Zero, Zero, Zero, Zero, Imp, Imm, Imp, Imm, Abs, Abs, Abs, Abs,
        // 0xF0 .. 0xFF
        Rel, IndY, None, IndYW, ZeroX, ZeroX, ZeroX, ZeroX, Imp, AbsY, Imp, AbsYW, AbsX, AbsX,
        AbsXW, AbsXW,
    ]
};

// CPU bus access is centralized through `CpuBusContext` + `bus_read` /
// `bus_write`, matching Mesen2 `NesMemoryManager::Read` / `Write`.

const CARRY: u8 = 0x01;
const ZERO: u8 = 0x02;
const INTERRUPT_DISABLE: u8 = 0x04;
const DECIMAL: u8 = 0x08;
const BREAK: u8 = 0x10;
const UNUSED: u8 = 0x20;
const OVERFLOW: u8 = 0x40;
const NEGATIVE: u8 = 0x80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cpu {
    pub pc: u16,
    pub sp: u8,
    pub a: u8,
    pub x: u8,
    pub y: u8,
    pub status: u8,
    pub cycles: u64,
    /// Unofficial KIL/JAM state; permanent until reset and not serialized.
    pub jammed: bool,

    // Master-clock timing.
    /// Absolute PPU master clock advanced by CPU cycle boundaries.
    pub master_clock: u64,
    /// PPU alignment offset subtracted before PPU catch-up.
    pub ppu_offset: u8,
    /// Master clocks added at cycle start.
    pub start_clock_count: u8,
    /// Master clocks added at cycle end.
    pub end_clock_count: u8,

    // DMA state machine.
    /// Pending DMA halt and alignment work.
    pub need_halt: bool,
    /// OAM DMA in progress.
    pub sprite_dma_transfer: bool,
    /// OAM DMA source page from `$4014`.
    pub sprite_dma_offset: u8,
    /// DMC sample DMA in progress.
    pub dmc_dma_running: bool,
    /// DMC DMA was cancelled mid-transfer.
    pub abort_dmc_dma: bool,
    /// DMC DMA needs an alignment dummy read before sample fetch.
    pub need_dummy_read: bool,
    /// True while current memory access is a DMC DMA read.
    pub is_dmc_dma_read: bool,
    /// True while the CPU is executing a write cycle.
    pub cpu_write_flag: bool,

    // NMI / IRQ edge sampling.
    /// Previous-cycle IRQ dispatch gate.
    pub prev_run_irq: bool,
    /// Live IRQ flag after source mask and CPU interrupt-disable flag.
    pub run_irq: bool,
    /// Prior NMI line value for edge detection.
    pub prev_nmi_flag: bool,
    /// One-cycle-delayed NMI dispatch gate.
    pub prev_need_nmi: bool,
    /// Set when a PPU NMI rising edge is detected.
    pub need_nmi: bool,
    /// Mask applied before testing pending IRQ sources.
    pub irq_mask: u8,

    /// Addressing mode of the instruction currently executing.
    pub inst_addr_mode: AddrMode,
    /// Operand or resolved address of the instruction currently executing.
    pub operand: u16,
}

/// Mesen2-aligned IRQ source identifiers. Mirrors `IRQSource` in
/// `reference/local/mesen2/Core/NES/NesTypes.h`. Bits combine in
/// `irq_flag` so a CPU can have multiple pending sources at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IrqSource {
    /// External (mapper) IRQ -MMC3 scanline counter, FME-7 timer, etc.
    External = 0x01,
    /// APU frame counter IRQ ($4017 mode 0, every ~16.6 ms).
    FrameCounter = 0x02,
    /// APU DMC sample-end IRQ.
    Dmc = 0x04,
}

/// Producer-pushed interrupt lines owned by `NesCore`.
#[derive(Debug, Clone, Default)]
pub struct InterruptLines {
    /// Bitmask of asserted IRQ sources (External | FrameCounter | Dmc).
    pub irq_flag: u8,
    /// PPU NMI line state. Pushed by PPU at VBL begin / `$2002` read /
    /// `$2000` bit-7 transition. Sampled by CPU `end_cpu_cycle`.
    pub nmi_flag: bool,
    /// DMC DMA request. Set by APU when DMC needs the next sample byte;
    /// consumed by CPU at next `process_pending_dma`.
    pub dmc_dma_pending: bool,
    /// DMC DMA abort. Set by APU on `$4015` disable while DMC mid-fetch.
    pub dmc_dma_stop: bool,
}

impl InterruptLines {
    pub fn set_irq_source(&mut self, source: IrqSource) {
        self.irq_flag |= source as u8;
    }
    pub fn clear_irq_source(&mut self, source: IrqSource) {
        self.irq_flag &= !(source as u8);
    }
    pub fn has_irq_source(&self, source: IrqSource) -> bool {
        self.irq_flag & (source as u8) != 0
    }
    pub fn set_nmi_flag(&mut self) {
        self.nmi_flag = true;
    }
    pub fn clear_nmi_flag(&mut self) {
        self.nmi_flag = false;
    }
    pub fn request_dmc_dma(&mut self) {
        self.dmc_dma_pending = true;
    }
    pub fn request_dmc_dma_stop(&mut self) {
        self.dmc_dma_stop = true;
    }
}

// Note: Region enum is canonicalized in `crate::cartridge::Region`; the
// duplicate `ConsoleRegion` was removed in (only used internally
// here + 3 cpu tests). All callers now pass `crate::cartridge::Region`
// directly to keep one source of truth (Mesen2 single `ConsoleRegion::Type`).

impl Default for Cpu {
    fn default() -> Self {
        Self {
            pc: 0,
            sp: 0xfd,
            a: 0,
            x: 0,
            y: 0,
            status: INTERRUPT_DISABLE,
            cycles: 0,
            jammed: false,
            // Mesen2 master-clock model defaults -NTSC values from
            // NesCpu::NesCpu (NesCpu.cpp:73-76) and NesCpu::Reset
            // (NesCpu.cpp:138-156). `ppu_offset = 1` is the deterministic
            // default Mesen2 picks when `RandomizeCpuPpuAlignment` is off
            // (NesCpu.cpp:154).
            master_clock: 0,
            ppu_offset: 1,
            start_clock_count: 6,
            end_clock_count: 6,
            need_halt: false,
            sprite_dma_transfer: false,
            sprite_dma_offset: 0,
            dmc_dma_running: false,
            abort_dmc_dma: false,
            need_dummy_read: false,
            is_dmc_dma_read: false,
            cpu_write_flag: false,
            prev_run_irq: false,
            run_irq: false,
            prev_nmi_flag: false,
            prev_need_nmi: false,
            need_nmi: false,
            irq_mask: 0xFF,
            inst_addr_mode: AddrMode::None,
            operand: 0,
        }
    }
}

impl Cpu {
    pub fn reset(&mut self) {
        self.pc = 0;
        self.sp = 0xfd;
        self.a = 0;
        self.x = 0;
        self.y = 0;
        self.status = INTERRUPT_DISABLE;
        self.cycles = 0;
        self.jammed = false;
        // Mesen2 NesCpu::Reset clears the DMA + NMI/IRQ edge state
        // (NesCpu.cpp:87-116). `master_clock` is reset to 0; `ppu_offset`
        // and clock counts stay at NTSC defaults (overridden by
        // `set_master_clock_divider` if the region changes).
        self.master_clock = 0;
        self.need_halt = false;
        self.sprite_dma_transfer = false;
        self.sprite_dma_offset = 0;
        self.dmc_dma_running = false;
        self.abort_dmc_dma = false;
        self.need_dummy_read = false;
        self.is_dmc_dma_read = false;
        self.cpu_write_flag = false;
        self.prev_run_irq = false;
        self.run_irq = false;
        self.prev_nmi_flag = false;
        self.prev_need_nmi = false;
        self.need_nmi = false;
        self.irq_mask = 0xFF;
        self.inst_addr_mode = AddrMode::None;
        self.operand = 0;
    }

    #[cold]
    pub fn snapshot_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(48);
        bytes.extend_from_slice(&self.pc.to_le_bytes());
        bytes.extend_from_slice(&[self.sp, self.a, self.x, self.y, self.status]);
        bytes.extend_from_slice(&self.cycles.to_le_bytes());
        bytes.push(u8::from(self.jammed));
        bytes.extend_from_slice(&self.master_clock.to_le_bytes());
        bytes.extend_from_slice(&[
            self.ppu_offset,
            self.start_clock_count,
            self.end_clock_count,
        ]);
        bytes.extend_from_slice(&[
            u8::from(self.need_halt),
            u8::from(self.sprite_dma_transfer),
            self.sprite_dma_offset,
            u8::from(self.dmc_dma_running),
            u8::from(self.abort_dmc_dma),
            u8::from(self.need_dummy_read),
            u8::from(self.is_dmc_dma_read),
            u8::from(self.cpu_write_flag),
            u8::from(self.prev_run_irq),
            u8::from(self.run_irq),
            u8::from(self.prev_nmi_flag),
            u8::from(self.prev_need_nmi),
            u8::from(self.need_nmi),
            self.irq_mask,
            self.inst_addr_mode.snapshot_tag(),
        ]);
        bytes.extend_from_slice(&self.operand.to_le_bytes());
        bytes
    }

    #[cold]
    pub fn restore_snapshot(&mut self, bytes: &[u8]) -> nesle_common::Result<()> {
        const LEN: usize = 2 + 5 + 8 + 1 + 8 + 3 + 15 + 2;
        if bytes.len() != LEN {
            return Err(nesle_common::NesleError::InvalidState(format!(
                "CPU snapshot length must be {LEN} bytes, got {}",
                bytes.len()
            )));
        }
        let mut offset = 0;
        self.pc = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
        offset += 2;
        self.sp = bytes[offset];
        self.a = bytes[offset + 1];
        self.x = bytes[offset + 2];
        self.y = bytes[offset + 3];
        self.status = bytes[offset + 4];
        offset += 5;
        self.cycles = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
        self.jammed = bytes[offset] != 0;
        offset += 1;
        self.master_clock = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
        self.ppu_offset = bytes[offset];
        self.start_clock_count = bytes[offset + 1];
        self.end_clock_count = bytes[offset + 2];
        offset += 3;
        self.need_halt = bytes[offset] != 0;
        self.sprite_dma_transfer = bytes[offset + 1] != 0;
        self.sprite_dma_offset = bytes[offset + 2];
        self.dmc_dma_running = bytes[offset + 3] != 0;
        self.abort_dmc_dma = bytes[offset + 4] != 0;
        self.need_dummy_read = bytes[offset + 5] != 0;
        self.is_dmc_dma_read = bytes[offset + 6] != 0;
        self.cpu_write_flag = bytes[offset + 7] != 0;
        self.prev_run_irq = bytes[offset + 8] != 0;
        self.run_irq = bytes[offset + 9] != 0;
        self.prev_nmi_flag = bytes[offset + 10] != 0;
        self.prev_need_nmi = bytes[offset + 11] != 0;
        self.need_nmi = bytes[offset + 12] != 0;
        self.irq_mask = bytes[offset + 13];
        self.inst_addr_mode = AddrMode::from_snapshot_tag(bytes[offset + 14])?;
        offset += 15;
        self.operand = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
        Ok(())
    }

    /// Run Mesen2's power-on reset warmup after the reset vector has been
    /// loaded. Mesen2 `NesCpu::Reset(false)` sets `CycleCount = -1`,
    /// advances `_masterClock` by one CPU divider, then performs 8 read
    /// cycles before the first opcode fetch.
    ///
    /// Phase E.3: `#[cold]` — called once per ROM load + once per
    /// `Core::reset()`, never on the per-frame hot path.
    #[cold]
    pub fn run_power_on_reset_warmup<M: Mapper + ?Sized>(
        &mut self,
        ctx: &mut CpuBusContext<'_, M>,
    ) {
        self.cycles = u64::MAX;
        self.master_clock = 0;
        self.ppu_offset = 1;
        self.master_clock = self
            .master_clock
            .wrapping_add(u64::from(self.start_clock_count + self.end_clock_count));
        for _ in 0..8 {
            self.start_cpu_cycle(true, ctx);
            self.end_cpu_cycle(true, ctx);
        }
    }

    // Public surface for APU / PPU / Bus / Mapper interaction with CPU
    // master-clock, DMA, and interrupt state. Mirrors Mesen2 `NesCpu`.

    /// Set CPU master-clock divider constants for the given region.
    /// Mesen2 equivalent: `NesCpu::SetMasterClockDivider`
    /// (NesCpu.cpp:550-569). NTSC uses 6/6, PAL uses 8/8, Dendy uses 7/8.
    pub fn set_master_clock_divider(&mut self, region: crate::cartridge::Region) {
        match region {
            crate::cartridge::Region::Ntsc => {
                self.start_clock_count = 6;
                self.end_clock_count = 6;
            }
            crate::cartridge::Region::Pal => {
                self.start_clock_count = 8;
                self.end_clock_count = 8;
            }
            crate::cartridge::Region::Dendy => {
                self.start_clock_count = 7;
                self.end_clock_count = 8;
            }
        }
    }

    /// Begin an OAM DMA transfer (256-byte copy from page `$XX00` to PPU
    /// OAM via `$2004`). Sets up the DMA state machine; the actual
    /// transfer runs inside the next memory access via `process_pending_dma`.
    /// Mesen2 equivalent: `NesCpu::RunDMATransfer` (NesCpu.cpp:520-525).
    pub fn run_dma_transfer(&mut self, offset_value: u8) {
        self.sprite_dma_transfer = true;
        self.sprite_dma_offset = offset_value;
        self.need_halt = true;
    }

    /// Begin a DMC sample-read DMA. APU's DMC channel calls this when
    /// it needs the next sample byte. The CPU halts for 4 cycles (1
    /// halt + up to 3 alignment dummy reads + 1 read) on the next memory
    /// access. Mesen2 equivalent: `NesCpu::StartDmcTransfer`
    /// (NesCpu.cpp:527-532).
    pub fn start_dmc_transfer(&mut self) {
        self.dmc_dma_running = true;
        self.need_dummy_read = true;
        self.need_halt = true;
    }

    /// Cancel a pending or in-progress DMC DMA. If the halt cycle hasn't
    /// started yet, DMA is fully cancelled; otherwise an abort is flagged
    /// and the DMA state machine cleans up on its next tick. Mesen2
    /// equivalent: `NesCpu::StopDmcTransfer` (NesCpu.cpp:534-548).
    pub fn stop_dmc_transfer(&mut self) {
        if self.dmc_dma_running {
            if self.need_halt {
                // Halt hasn't started: cancel DMA outright.
                self.dmc_dma_running = false;
                self.need_dummy_read = false;
                self.need_halt = false;
            } else {
                // Halt already started: signal abort to the DMA state
                // machine. Cleanup happens inside `process_pending_dma`.
                self.abort_dmc_dma = true;
            }
        }
    }

    /// First half of a CPU bus cycle. Mesen2 `NesCpu::StartCpuCycle`
    /// (NesCpu.cpp:317-323). Advances master_clock, ticks PPU forward,
    /// fires mapper + APU per-cycle hooks. IRQ/NMI are pushed by the
    /// producers directly into `ctx.interrupt`.
    /// Phase E.4: `#[inline]` — called >1.79M times/sec at full bench
    /// speed (NES NTSC budget). Inlining lets the optimizer fold the
    /// `is_read` parameter and propagate `ctx` field access; without
    /// the hint the compiler still inlines in opt-level=3 + LTO, but
    /// the explicit attribute survives refactors.
    #[inline]
    pub fn start_cpu_cycle<M: Mapper + ?Sized>(
        &mut self,
        is_read: bool,
        ctx: &mut CpuBusContext<'_, M>,
    ) {
        let clocks: u64 = if is_read {
            u64::from(self.start_clock_count).wrapping_sub(1)
        } else {
            u64::from(self.start_clock_count).wrapping_add(1)
        };
        self.master_clock = self.master_clock.wrapping_add(clocks);
        self.cycles = self.cycles.wrapping_add(1);
        ctx.cpu_cycle_count = self.cycles;
        // keep ctx.master_clock in sync per cycle so
        // bus_write at $4016 (bus.rs `queue_strobe_write`) sees the
        // current Mesen2-aligned parity for the 1-2 master clock defer.
        ctx.master_clock = self.master_clock;
        let target = self.master_clock.wrapping_sub(u64::from(self.ppu_offset));
        ctx.ppu.run_with_cpu_cycle(
            target,
            &mut *ctx.mapper,
            ctx.interrupt,
            self.cycles,
            ctx.mapper_has_vram_addr_hook,
        );
        // Mesen2 commits host input inside PPU scanline-241 first-cycle
        // processing, before mapper/APU/control-manager clock hooks.
        if ctx.ppu.pending_input_commit {
            ctx.ppu.pending_input_commit = false;
            ctx.controllers.commit_pending_input();
        }
        if ctx.mapper_has_cpu_clock_hook {
            ctx.mapper.process_cpu_clock(ctx.interrupt);
        }
        ctx.apu.process_cpu_clock(ctx.interrupt);
        ctx.controllers.process_pending_write();
    }

    /// Second half of a CPU bus cycle. Mesen2 `NesCpu::EndCpuCycle`
    /// (NesCpu.cpp:294-315). Samples NMI/IRQ edges off `ctx.interrupt`.
    /// Phase E.4: see `start_cpu_cycle` for inline rationale.
    #[inline]
    pub fn end_cpu_cycle<M: Mapper + ?Sized>(
        &mut self,
        is_read: bool,
        ctx: &mut CpuBusContext<'_, M>,
    ) {
        let clocks: u64 = if is_read {
            u64::from(self.end_clock_count).wrapping_add(1)
        } else {
            u64::from(self.end_clock_count).wrapping_sub(1)
        };
        self.master_clock = self.master_clock.wrapping_add(clocks);
        let target = self.master_clock.wrapping_sub(u64::from(self.ppu_offset));
        ctx.ppu.run_with_cpu_cycle(
            target,
            &mut *ctx.mapper,
            ctx.interrupt,
            self.cycles,
            ctx.mapper_has_vram_addr_hook,
        );

        self.prev_need_nmi = self.need_nmi;
        if !self.prev_nmi_flag && ctx.interrupt.nmi_flag {
            self.need_nmi = true;
        }
        self.prev_nmi_flag = ctx.interrupt.nmi_flag;

        self.prev_run_irq = self.run_irq;
        self.run_irq =
            (ctx.interrupt.irq_flag & self.irq_mask) != 0 && (self.status & INTERRUPT_DISABLE) == 0;
    }

    /// Mesen2-aligned CPU memory read. Runs the full per-CPU-cycle
    /// sequence: optional DMA processing, then `start_cpu_cycle`, then
    /// the bus read, then `end_cpu_cycle`. Mirrors `NesCpu::MemoryRead`
    /// (NesCpu.cpp:254-268).
    /// Phase E.4: `#[inline]` — bridges `start_cpu_cycle` -> `bus_read`
    /// -> `end_cpu_cycle`. Inlining lets the cycle pair fuse with the
    /// bus access into a single hot loop body in the caller.
    #[inline]
    pub fn memory_read<M: Mapper + ?Sized>(
        &mut self,
        addr: u16,
        op: MemoryOperationType,
        ctx: &mut CpuBusContext<'_, M>,
    ) -> u8 {
        self.process_pending_dma(addr, op, ctx);
        self.start_cpu_cycle(true, ctx);
        let value = if matches!(
            op,
            MemoryOperationType::ExecOpCode | MemoryOperationType::ExecOperand
        ) {
            if let Some(value) = ctx.mapper.cpu_code_read(addr) {
                ctx.bus.set_cpu_open_bus(value);
                value
            } else {
                bus_read(addr, ctx)
            }
        } else {
            bus_read(addr, ctx)
        };
        self.end_cpu_cycle(true, ctx);
        value
    }

    /// Mesen2-aligned CPU memory write. `bus_write` returns
    /// `Some(BusEvent::OamDma(page))` on `$4014`, consumed here by
    /// `run_dma_transfer` synchronously (NesPpu.cpp:505 -    /// `Cpu::RunDMATransfer`).
    /// Phase E.4: see `memory_read` for inline rationale.
    #[inline]
    pub fn memory_write<M: Mapper + ?Sized>(
        &mut self,
        addr: u16,
        value: u8,
        op: MemoryOperationType,
        ctx: &mut CpuBusContext<'_, M>,
    ) {
        self.cpu_write_flag = true;
        self.start_cpu_cycle(false, ctx);
        if addr <= 0x1fff {
            let normalized = usize::from(addr & 0x07ff);
            let old_value = ctx.bus.wram()[normalized];
            ctx.bus.record_wram_write(WramWriteLogEntry {
                cycle_count: self.cycles,
                pc: self.pc,
                addr,
                normalized_addr: addr & 0x07ff,
                old_value,
                new_value: value,
                op_type: op.tag(),
                a: self.a,
                x: self.x,
                y: self.y,
                sp: self.sp,
                status: self.status,
            });
        }
        let event = bus_write(addr, value, ctx);
        self.end_cpu_cycle(false, ctx);
        self.cpu_write_flag = false;
        if let Some(BusEvent::OamDma(page)) = event {
            self.run_dma_transfer(page);
        }
    }

    // ===== Operand fetch (NesCpu.cpp:270-292, NesCpu.h:196-282) =====

    /// Resolve the operand for the given addressing mode. Returns the
    /// raw 16-bit operand value (an address for indexed/absolute modes,
    /// the immediate byte for `Imm`/`Rel`, 0 for `Imp`/`Acc`/`None`).
    /// Mesen2 equivalent: `NesCpu::FetchOperand` (NesCpu.cpp:270-292).
    ///
    /// For modes that include dummy reads (`ZeroX`, `ZeroY`, `IndX`,
    /// `IndYW`, `AbsXW`, `AbsYW`, and the page-cross dummy on `IndY`/
    /// `AbsX`/`AbsY` reads), the dummy reads are issued here via
    /// `memory_read(..., DummyRead)` so PPU/APU/mapper hooks see them.
    pub fn fetch_operand<M: Mapper + ?Sized>(
        &mut self,
        mode: AddrMode,
        ctx: &mut CpuBusContext<'_, M>,
    ) -> u16 {
        match mode {
            AddrMode::Imp | AddrMode::Acc => {
                // Mesen2: `DummyRead` of next-byte without PC increment.
                self.memory_read(self.pc, MemoryOperationType::DummyRead, ctx);
                0
            }
            AddrMode::Imm | AddrMode::Rel => u16::from(self.read_byte_mc(ctx)),
            AddrMode::Zero => u16::from(self.read_byte_mc(ctx)),
            AddrMode::ZeroX => {
                let zp = self.read_byte_mc(ctx);
                // Mesen2 NesCpu.h:201: dummy read of ZP address before
                // the +X offset is applied.
                self.memory_read(u16::from(zp), MemoryOperationType::DummyRead, ctx);
                u16::from(zp.wrapping_add(self.x))
            }
            AddrMode::ZeroY => {
                let zp = self.read_byte_mc(ctx);
                self.memory_read(u16::from(zp), MemoryOperationType::DummyRead, ctx);
                u16::from(zp.wrapping_add(self.y))
            }
            AddrMode::Ind => self.read_word_mc(ctx),
            AddrMode::IndX => self.fetch_indirect_x(ctx),
            AddrMode::IndY => self.fetch_indirect_y(false, ctx),
            AddrMode::IndYW => self.fetch_indirect_y(true, ctx),
            AddrMode::Abs => self.read_word_mc(ctx),
            AddrMode::AbsX => self.fetch_absolute_indexed(self.x, false, ctx),
            AddrMode::AbsXW => self.fetch_absolute_indexed(self.x, true, ctx),
            AddrMode::AbsY => self.fetch_absolute_indexed(self.y, false, ctx),
            AddrMode::AbsYW => self.fetch_absolute_indexed(self.y, true, ctx),
            AddrMode::None => 0,
        }
    }

    /// Read a byte at PC, advance PC. Mesen2 equivalent: `NesCpu::ReadByte`
    /// (NesCpu.h:84-89).
    fn read_byte_mc<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) -> u8 {
        let value = self.memory_read(self.pc, MemoryOperationType::ExecOperand, ctx);
        self.pc = self.pc.wrapping_add(1);
        value
    }

    /// Read a little-endian word starting at PC. Mesen2 equivalent:
    /// `NesCpu::ReadWord` (NesCpu.h:91-96).
    fn read_word_mc<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) -> u16 {
        let lo = self.read_byte_mc(ctx);
        let hi = self.read_byte_mc(ctx);
        u16::from(lo) | (u16::from(hi) << 8)
    }

    /// `(Indirect,X)` operand fetch. Mesen2 equivalent:
    /// `NesCpu::GetIndXAddr` (NesCpu.h:245-262), including the dummy
    /// read of the pre-indexed ZP address and the page-wrap quirk at
    /// `$FF`.
    fn fetch_indirect_x<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) -> u16 {
        let zero = self.read_byte_mc(ctx);
        self.memory_read(u16::from(zero), MemoryOperationType::DummyRead, ctx);
        let zero = zero.wrapping_add(self.x);
        if zero == 0xFF {
            // Page-wrap: high byte from $00, low byte from $FF.
            let lo = self.memory_read(0x00FF, MemoryOperationType::Read, ctx);
            let hi = self.memory_read(0x0000, MemoryOperationType::Read, ctx);
            u16::from(lo) | (u16::from(hi) << 8)
        } else {
            let lo = self.memory_read(u16::from(zero), MemoryOperationType::Read, ctx);
            let hi = self.memory_read(
                u16::from(zero.wrapping_add(1)),
                MemoryOperationType::Read,
                ctx,
            );
            u16::from(lo) | (u16::from(hi) << 8)
        }
    }

    /// `(Indirect),Y` operand fetch. The `write_form` flag forces a
    /// dummy read even when no page cross occurs (required for store
    /// opcodes). Mesen2 equivalent: `NesCpu::GetIndYAddr`
    /// (NesCpu.h:264-282).
    fn fetch_indirect_y<M: Mapper + ?Sized>(
        &mut self,
        write_form: bool,
        ctx: &mut CpuBusContext<'_, M>,
    ) -> u16 {
        let zero = self.read_byte_mc(ctx);
        let base = if zero == 0xFF {
            let lo = self.memory_read(0x00FF, MemoryOperationType::Read, ctx);
            let hi = self.memory_read(0x0000, MemoryOperationType::Read, ctx);
            u16::from(lo) | (u16::from(hi) << 8)
        } else {
            let lo = self.memory_read(u16::from(zero), MemoryOperationType::Read, ctx);
            let hi = self.memory_read(
                u16::from(zero.wrapping_add(1)),
                MemoryOperationType::Read,
                ctx,
            );
            u16::from(lo) | (u16::from(hi) << 8)
        };
        let crossed = page_crossed(base, base.wrapping_add(u16::from(self.y)));
        if crossed || write_form {
            // Mesen2 NesCpu.h:279: dummy read at the partially-indexed
            // address (subtract 0x100 if page crossed, else use the
            // straight sum).
            let dummy_addr = base
                .wrapping_add(u16::from(self.y))
                .wrapping_sub(if crossed { 0x100 } else { 0 });
            self.memory_read(dummy_addr, MemoryOperationType::DummyRead, ctx);
        }
        base.wrapping_add(u16::from(self.y))
    }

    /// `Absolute,X` and `Absolute,Y` operand fetch. The `write_form`
    /// flag forces a dummy read for store opcodes. Mesen2 equivalent:
    /// `NesCpu::GetAbsXAddr` / `GetAbsYAddr` (NesCpu.h:211-232).
    fn fetch_absolute_indexed<M: Mapper + ?Sized>(
        &mut self,
        index: u8,
        write_form: bool,
        ctx: &mut CpuBusContext<'_, M>,
    ) -> u16 {
        let base = self.read_word_mc(ctx);
        let crossed = page_crossed(base, base.wrapping_add(u16::from(index)));
        if crossed || write_form {
            let dummy_addr = base
                .wrapping_add(u16::from(index))
                .wrapping_sub(if crossed { 0x100 } else { 0 });
            self.memory_read(dummy_addr, MemoryOperationType::DummyRead, ctx);
        }
        base.wrapping_add(u16::from(index))
    }

    /// Process pending OAM/DMC DMA before the CPU memory read completes.
    pub fn process_pending_dma<M: Mapper + ?Sized>(
        &mut self,
        read_address: u16,
        op: MemoryOperationType,
        ctx: &mut CpuBusContext<'_, M>,
    ) {
        // Consume APU-pushed DMC DMA requests at memory access boundaries.
        if ctx.interrupt.dmc_dma_pending {
            self.start_dmc_transfer();
            ctx.interrupt.dmc_dma_pending = false;
        }
        if ctx.interrupt.dmc_dma_stop {
            self.stop_dmc_transfer();
            ctx.interrupt.dmc_dma_stop = false;
        }
        if !self.need_halt {
            return;
        }
        self.process_pending_dma_slow(read_address, op, ctx);
    }

    #[cold]
    fn process_pending_dma_slow<M: Mapper + ?Sized>(
        &mut self,
        read_address: u16,
        op: MemoryOperationType,
        ctx: &mut CpuBusContext<'_, M>,
    ) {
        // PAL DMA starts only on opcode fetches; NTSC path is always eligible.
        let _ = op;

        let mut prev_read_address = read_address;
        let enable_internal_reg_reads = (read_address & 0xffe0) == 0x4000;
        let is_input_read = read_address == 0x4016 || read_address == 0x4017;
        let mut skip_first_input_clock = false;
        if enable_internal_reg_reads && self.dmc_dma_running && is_input_read {
            let dmc_address = ctx.apu.dmc_read_address();
            skip_first_input_clock = (dmc_address & 0x001f) == (read_address & 0x001f);
        }
        let skip_dummy_reads = is_input_read;

        // Halt cycle: one CPU stall with a dummy read at the original address.
        self.need_halt = false;
        self.start_cpu_cycle(true, ctx);
        if self.abort_dmc_dma && is_input_read {
            // NES behavior: avoid a second separate controller clock when
            // an aborted DMC DMA is followed by a real $4016/$4017 read.
        } else if !skip_first_input_clock {
            // Dummy read of the address the CPU was about to access.
            // Mesen2 uses ::Read(addr, DmaRead) here.
            let _ = bus_read(read_address, ctx);
        }
        self.end_cpu_cycle(true, ctx);

        // Poll each cycle boundary so DMC DMA can interleave with OAM DMA.
        if ctx.interrupt.dmc_dma_pending {
            self.start_dmc_transfer();
            ctx.interrupt.dmc_dma_pending = false;
        }
        if ctx.interrupt.dmc_dma_stop {
            self.stop_dmc_transfer();
            ctx.interrupt.dmc_dma_stop = false;
        }

        // If DMC DMA was aborted and OAM is absent, processing ends here.
        if self.abort_dmc_dma {
            self.dmc_dma_running = false;
            self.abort_dmc_dma = false;
            if !self.sprite_dma_transfer {
                self.need_dummy_read = false;
                return;
            }
        }

        // DMA loop alternates get/put cycles and interleaves DMC reads.
        let mut sprite_dma_counter: u16 = 0;
        let mut sprite_read_addr: u8 = 0;
        let mut read_value: u8 = 0;

        loop {
            // New DMC requests can arrive while OAM DMA is already running.
            if ctx.interrupt.dmc_dma_pending {
                self.start_dmc_transfer();
                ctx.interrupt.dmc_dma_pending = false;
            }
            if ctx.interrupt.dmc_dma_stop {
                self.stop_dmc_transfer();
                ctx.interrupt.dmc_dma_stop = false;
            }
            if !(self.dmc_dma_running || self.sprite_dma_transfer) {
                break;
            }
            let get_cycle = (self.cycles & 0x01) == 0;
            if get_cycle {
                if self.dmc_dma_running && !self.need_halt && !self.need_dummy_read {
                    // DMC DMA read of the next sample byte.
                    self.process_cycle_setup();
                    self.start_cpu_cycle(true, ctx);
                    self.is_dmc_dma_read = true;
                    let dmc_addr = ctx.apu.dmc_read_address();
                    read_value = self.process_dma_read(
                        dmc_addr,
                        &mut prev_read_address,
                        enable_internal_reg_reads,
                        ctx,
                    );
                    self.is_dmc_dma_read = false;
                    self.end_cpu_cycle(true, ctx);
                    self.dmc_dma_running = false;
                    self.abort_dmc_dma = false;
                    ctx.apu.set_dmc_read_buffer(read_value, ctx.interrupt);
                } else if self.sprite_dma_transfer {
                    // OAM DMA read from page base + offset.
                    self.process_cycle_setup();
                    self.start_cpu_cycle(true, ctx);
                    let addr =
                        (u16::from(self.sprite_dma_offset) << 8) | u16::from(sprite_read_addr);
                    read_value = self.process_dma_read(
                        addr,
                        &mut prev_read_address,
                        enable_internal_reg_reads,
                        ctx,
                    );
                    self.end_cpu_cycle(true, ctx);
                    sprite_read_addr = sprite_read_addr.wrapping_add(1);
                    sprite_dma_counter += 1;
                } else {
                    // Idle dummy read while DMC waits for halt/dummy
                    // alignment to finish.
                    self.process_cycle_setup();
                    self.start_cpu_cycle(true, ctx);
                    if !skip_dummy_reads {
                        let _ = bus_read(read_address, ctx);
                    }
                    self.end_cpu_cycle(true, ctx);
                }
            } else {
                // "Put" cycle (odd cycle_count).
                if self.sprite_dma_transfer && (sprite_dma_counter & 0x01) == 1 {
                    // OAM DMA write to $2004.
                    self.process_cycle_setup();
                    self.start_cpu_cycle(true, ctx);
                    bus_write(0x2004, read_value, ctx);
                    self.end_cpu_cycle(true, ctx);
                    sprite_dma_counter += 1;
                    if sprite_dma_counter == 0x200 {
                        self.sprite_dma_transfer = false;
                    }
                } else {
                    // Align to next get cycle with a dummy read.
                    self.process_cycle_setup();
                    self.start_cpu_cycle(true, ctx);
                    if !skip_dummy_reads {
                        let _ = bus_read(read_address, ctx);
                    }
                    self.end_cpu_cycle(true, ctx);
                }
            }
        }
    }

    /// DMA read helper for the 2A03 internal-register glitch. Mirrors
    /// Mesen2 `NesCpu::ProcessDmaRead` (NesCpu.cpp:450-518).
    fn process_dma_read<M: Mapper + ?Sized>(
        &mut self,
        addr: u16,
        prev_read_address: &mut u16,
        enable_internal_reg_reads: bool,
        ctx: &mut CpuBusContext<'_, M>,
    ) -> u8 {
        if !enable_internal_reg_reads {
            let value = self.dma_external_read(addr, ctx);
            *prev_read_address = addr;
            return value;
        }

        let internal_addr = 0x4000 | (addr & 0x001f);
        let is_same_address = internal_addr == addr;
        let value = match internal_addr {
            0x4015 => {
                let value = bus_read(internal_addr, ctx);
                if !is_same_address {
                    let _ = self.dma_external_read(addr, ctx);
                }
                value
            }
            0x4016 | 0x4017 => {
                let mut value = if *prev_read_address == internal_addr {
                    ctx.bus.cpu_open_bus()
                } else {
                    bus_read(internal_addr, ctx)
                };

                if !is_same_address {
                    let open_bus_mask = 0xe0;
                    let external_value = self.dma_external_read(addr, ctx);
                    value = (external_value & open_bus_mask)
                        | ((value & !open_bus_mask) & (external_value & !open_bus_mask));
                }
                value
            }
            _ => bus_read(addr, ctx),
        };

        *prev_read_address = internal_addr;
        value
    }

    fn dma_external_read<M: Mapper + ?Sized>(
        &mut self,
        addr: u16,
        ctx: &mut CpuBusContext<'_, M>,
    ) -> u8 {
        if (0x4000..=0x401f).contains(&addr) {
            ctx.bus.cpu_open_bus()
        } else {
            bus_read(addr, ctx)
        }
    }

    /// Internal helper for `process_pending_dma`. Mirrors the Mesen2
    /// `processCycle` lambda (NesCpu.cpp:384-397) -resolves any pending
    /// abort/halt/dummy flags before the next StartCpuCycle.
    fn process_cycle_setup(&mut self) {
        if self.abort_dmc_dma {
            self.dmc_dma_running = false;
            self.abort_dmc_dma = false;
            self.need_dummy_read = false;
            self.need_halt = false;
        } else if self.need_halt {
            self.need_halt = false;
        } else if self.need_dummy_read {
            self.need_dummy_read = false;
        }
    }

    /// Mesen2-aligned NMI / IRQ vector dispatch. Pushes PC + flags, sets
    /// I flag, jumps to the appropriate vector. NMI takes priority over
    /// IRQ. Mesen2 equivalent: `NesCpu::IRQ` (NesCpu.cpp:183-218).
    ///
    /// Phase E.3: `#[cold]` — most opcodes finish without an interrupt
    /// dispatch (NMI ~once/frame visible; IRQ depends on mapper). The
    /// only call site is the `prev_run_irq || prev_need_nmi` branch in
    /// `Cpu::exec`; putting this body on a cold code path keeps the
    /// per-opcode hot path tighter in the I-cache.
    #[cold]
    pub fn handle_interrupt<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        // Dummy reads for opcode fetch + next-byte (Mesen2 NesCpu.cpp:194-195).
        self.memory_read(self.pc, MemoryOperationType::DummyRead, ctx);
        self.memory_read(self.pc, MemoryOperationType::DummyRead, ctx);

        // Push PC high then low (Mesen2 NesCpu::Push(uint16) helper).
        let pc = self.pc;
        self.memory_write(
            0x0100 | u16::from(self.sp),
            (pc >> 8) as u8,
            MemoryOperationType::Write,
            ctx,
        );
        self.sp = self.sp.wrapping_sub(1);
        self.memory_write(
            0x0100 | u16::from(self.sp),
            (pc & 0xFF) as u8,
            MemoryOperationType::Write,
            ctx,
        );
        self.sp = self.sp.wrapping_sub(1);

        let vector = if self.need_nmi {
            self.need_nmi = false;
            0xFFFA
        } else {
            0xFFFE
        };

        // Push status with B clear (interrupt push) but R/UNUSED set.
        // Mesen2 NesCpu.cpp:200/209 uses `PS() | PSFlags::Reserved`.
        let pushed_status = self.status | UNUSED;
        self.memory_write(
            0x0100 | u16::from(self.sp),
            pushed_status,
            MemoryOperationType::Write,
            ctx,
        );
        self.sp = self.sp.wrapping_sub(1);
        self.status |= INTERRUPT_DISABLE;

        let lo = self.memory_read(vector, MemoryOperationType::Read, ctx);
        let hi = self.memory_read(vector + 1, MemoryOperationType::Read, ctx);
        self.pc = u16::from(lo) | (u16::from(hi) << 8);
    }

    // Opcode dispatch.

    /// Execute one instruction and dispatch pending NMI/IRQ afterward.
    pub fn exec<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        if self.jammed {
            self.exec_jammed(ctx);
            return;
        }

        let opcode = self.memory_read(self.pc, MemoryOperationType::ExecOpCode, ctx);
        self.pc = self.pc.wrapping_add(1);
        self.inst_addr_mode = ADDR_MODE[opcode as usize];
        self.operand = self.fetch_operand(self.inst_addr_mode, ctx);
        self.execute_opcode(opcode, ctx);
        if self.prev_need_nmi || self.prev_run_irq {
            self.prev_need_nmi = false;
            self.prev_run_irq = false;
            self.handle_interrupt(ctx);
        }
    }

    #[cold]
    fn exec_jammed<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        self.start_cpu_cycle(true, ctx);
        self.end_cpu_cycle(true, ctx);
    }

    /// Read operand value or return the immediate byte for immediate-like modes.
    fn operand_value<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) -> u8 {
        if self.inst_addr_mode >= AddrMode::Zero {
            self.memory_read(self.operand, MemoryOperationType::Read, ctx)
        } else {
            self.operand as u8
        }
    }

    /// Stack push (byte). Mirrors `NesCpu::Push(uint8_t)` (NesCpu.h:147-150).
    fn push_op<M: Mapper + ?Sized>(&mut self, value: u8, ctx: &mut CpuBusContext<'_, M>) {
        self.memory_write(
            0x0100 | u16::from(self.sp),
            value,
            MemoryOperationType::Write,
            ctx,
        );
        self.sp = self.sp.wrapping_sub(1);
    }

    /// Stack pop (byte). Mirrors `NesCpu::Pop` (NesCpu.h:157-160).
    fn pop_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) -> u8 {
        self.sp = self.sp.wrapping_add(1);
        self.memory_read(0x0100 | u16::from(self.sp), MemoryOperationType::Read, ctx)
    }

    /// Push 16-bit word (high byte first). Mirrors `NesCpu::Push(uint16_t)`
    /// (NesCpu.h:152-155).
    fn push_u16_op<M: Mapper + ?Sized>(&mut self, value: u16, ctx: &mut CpuBusContext<'_, M>) {
        self.push_op((value >> 8) as u8, ctx);
        self.push_op(value as u8, ctx);
    }

    /// Pop 16-bit word (low byte first). Mirrors `NesCpu::PopWord`
    /// (NesCpu.h:162-167).
    fn pop_u16_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) -> u16 {
        let lo = self.pop_op(ctx);
        let hi = self.pop_op(ctx);
        u16::from(lo) | (u16::from(hi) << 8)
    }

    /// Dummy read at PC. Mirrors `NesCpu::DummyRead` (NesCpu.h:79-82).
    fn dummy_read_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        self.memory_read(self.pc, MemoryOperationType::DummyRead, ctx);
    }

    /// Set register A and update Z/N. Mirrors `NesCpu::SetA` (NesCpu.h:170).
    fn set_a(&mut self, value: u8) {
        self.a = value;
        self.set_zn(value);
    }

    /// Set register X and update Z/N. Mirrors `NesCpu::SetX` (NesCpu.h:172).
    fn set_x(&mut self, value: u8) {
        self.x = value;
        self.set_zn(value);
    }

    /// Set register Y and update Z/N. Mirrors `NesCpu::SetY` (NesCpu.h:174).
    fn set_y(&mut self, value: u8) {
        self.y = value;
        self.set_zn(value);
    }

    /// 256-way opcode dispatch. Each case calls a small helper that
    /// uses `self.operand` (pre-resolved by `fetch_operand`). Mirrors
    /// `NesCpu::_opTable` (NesCpu.cpp:23-41) -the dispatch order is
    /// hand-verified against the Mesen2 table cell-by-cell.
    fn execute_opcode<M: Mapper + ?Sized>(&mut self, opcode: u8, ctx: &mut CpuBusContext<'_, M>) {
        match opcode {
            // ----- ORA / AND / EOR (logic) -----
            0x09 | 0x05 | 0x15 | 0x01 | 0x11 | 0x0D | 0x1D | 0x19 => self.ora(ctx),
            0x29 | 0x25 | 0x35 | 0x21 | 0x31 | 0x2D | 0x3D | 0x39 => self.and(ctx),
            0x49 | 0x45 | 0x55 | 0x41 | 0x51 | 0x4D | 0x5D | 0x59 => self.eor(ctx),

            // ----- ADC / SBC -----
            0x69 | 0x65 | 0x75 | 0x61 | 0x71 | 0x6D | 0x7D | 0x79 => self.adc_op(ctx),
            // 0xEB is unofficial SBC #imm
            0xE9 | 0xEB | 0xE5 | 0xF5 | 0xE1 | 0xF1 | 0xED | 0xFD | 0xF9 => self.sbc_op(ctx),

            // ----- CMP / CPX / CPY -----
            0xC9 | 0xC5 | 0xD5 | 0xC1 | 0xD1 | 0xCD | 0xDD | 0xD9 => self.cmp_a(ctx),
            0xE0 | 0xE4 | 0xEC => self.cpx_op(ctx),
            0xC0 | 0xC4 | 0xCC => self.cpy_op(ctx),

            // ----- LDA / LDX / LDY -----
            0xA9 | 0xA5 | 0xB5 | 0xA1 | 0xB1 | 0xAD | 0xBD | 0xB9 => self.lda(ctx),
            0xA2 | 0xA6 | 0xB6 | 0xAE | 0xBE => self.ldx(ctx),
            0xA0 | 0xA4 | 0xB4 | 0xAC | 0xBC => self.ldy(ctx),

            // ----- STA / STX / STY -----
            0x85 | 0x95 | 0x81 | 0x91 | 0x8D | 0x9D | 0x99 => self.sta(ctx),
            0x86 | 0x96 | 0x8E => self.stx(ctx),
            0x84 | 0x94 | 0x8C => self.sty(ctx),

            // ----- Register transfers -----
            0xAA => self.tax(),
            0xA8 => self.tay(),
            0xBA => self.tsx(),
            0x8A => self.txa(),
            0x9A => self.txs(),
            0x98 => self.tya(),

            // ----- Stack ops -----
            0x48 => self.pha(ctx),
            0x08 => self.php(ctx),
            0x68 => self.pla(ctx),
            0x28 => self.plp(ctx),

            // ----- INC / DEC + INX/INY/DEX/DEY -----
            0xE6 | 0xF6 | 0xEE | 0xFE => self.inc_op(ctx),
            0xC6 | 0xD6 | 0xCE | 0xDE => self.dec_op(ctx),
            0xE8 => self.inx(),
            0xC8 => self.iny(),
            0xCA => self.dex(),
            0x88 => self.dey(),

            // ----- ASL / LSR / ROL / ROR -----
            0x0A => self.asl_acc(),
            0x06 | 0x16 | 0x0E | 0x1E => self.asl_mem_op(ctx),
            0x4A => self.lsr_acc(),
            0x46 | 0x56 | 0x4E | 0x5E => self.lsr_mem_op(ctx),
            0x2A => self.rol_acc(),
            0x26 | 0x36 | 0x2E | 0x3E => self.rol_mem_op(ctx),
            0x6A => self.ror_acc(),
            0x66 | 0x76 | 0x6E | 0x7E => self.ror_mem_op(ctx),

            // ----- Jumps and subroutines -----
            0x4C => self.jmp_abs(),
            0x6C => self.jmp_ind(ctx),
            0x20 => self.jsr(ctx),
            0x60 => self.rts(ctx),

            // ----- Branches -----
            0x90 => self.branch_op(!self.flag(CARRY), ctx),
            0xB0 => self.branch_op(self.flag(CARRY), ctx),
            0xF0 => self.branch_op(self.flag(ZERO), ctx),
            0x30 => self.branch_op(self.flag(NEGATIVE), ctx),
            0xD0 => self.branch_op(!self.flag(ZERO), ctx),
            0x10 => self.branch_op(!self.flag(NEGATIVE), ctx),
            0x50 => self.branch_op(!self.flag(OVERFLOW), ctx),
            0x70 => self.branch_op(self.flag(OVERFLOW), ctx),

            // ----- Flag manipulation -----
            0x18 => self.set_flag(CARRY, false),
            0xD8 => self.set_flag(DECIMAL, false),
            0x58 => self.set_flag(INTERRUPT_DISABLE, false),
            0xB8 => self.set_flag(OVERFLOW, false),
            0x38 => self.set_flag(CARRY, true),
            0xF8 => self.set_flag(DECIMAL, true),
            0x78 => self.set_flag(INTERRUPT_DISABLE, true),

            // ----- BRK / RTI / BIT / NOP -----
            0x00 => self.brk_op(ctx),
            0x40 => self.rti_op(ctx),
            0x24 | 0x2C => self.bit_op(ctx),
            0xEA => self.nop_op(ctx),

            // ----- Unofficial NOPs (single + double + triple byte) -----
            // Implied: 1A, 3A, 5A, 7A, DA, FA
            0x1A | 0x3A | 0x5A | 0x7A | 0xDA | 0xFA => self.nop_op(ctx),
            // Immediate / zp / zp,X / abs / abs,X (ADDR_MODE handles operand fetch)
            0x80 | 0x82 | 0x89 | 0xC2 | 0xE2 => self.nop_op(ctx),
            0x04 | 0x44 | 0x64 => self.nop_op(ctx),
            0x14 | 0x34 | 0x54 | 0x74 | 0xD4 | 0xF4 => self.nop_op(ctx),
            0x0C => self.nop_op(ctx),
            0x1C | 0x3C | 0x5C | 0x7C | 0xDC | 0xFC => self.nop_op(ctx),

            // ----- KIL / HLT (unofficial) -----
            0x02 | 0x12 | 0x22 | 0x32 | 0x42 | 0x52 | 0x62 | 0x72 | 0x92 | 0xB2 | 0xD2 | 0xF2 => {
                self.hlt_op()
            }

            // ----- Unofficial RMW + dual-op (SLO/SRE/RLA/RRA/DCP/ISB) -----
            0x07 | 0x17 | 0x03 | 0x13 | 0x0F | 0x1F | 0x1B => self.slo_op(ctx),
            0x47 | 0x57 | 0x43 | 0x53 | 0x4F | 0x5F | 0x5B => self.sre_op(ctx),
            0x27 | 0x37 | 0x23 | 0x33 | 0x2F | 0x3F | 0x3B => self.rla_op(ctx),
            0x67 | 0x77 | 0x63 | 0x73 | 0x6F | 0x7F | 0x7B => self.rra_op(ctx),
            0xC7 | 0xD7 | 0xC3 | 0xD3 | 0xCF | 0xDF | 0xDB => self.dcp_op(ctx),
            0xE7 | 0xF7 | 0xE3 | 0xF3 | 0xEF | 0xFF | 0xFB => self.isb_op(ctx),

            // ----- Unofficial loads + AND combinations -----
            0xA7 | 0xB7 | 0xA3 | 0xB3 | 0xAF | 0xBF => self.lax_op(ctx),
            0x87 | 0x97 | 0x83 | 0x8F => self.sax_op(ctx),
            0x0B | 0x2B => self.aac_op(ctx),
            0x4B => self.asr_op(ctx),
            0x6B => self.arr_op(ctx),
            0xAB => self.atx_op(ctx),
            0xCB => self.axs_op(ctx),

            // ----- Unofficial indexed stores (SHY/SHX/SHAA/SHAZ/TAS/ANE/LAS) -----
            0x9C => self.shy_op(ctx),
            0x9E => self.shx_op(ctx),
            0x9F => self.sha_abs_op(ctx),
            0x93 => self.sha_zp_op(ctx),
            0x9B => self.tas_op(ctx),
            0x8B => self.ane_op(ctx),
            0xBB => self.las_op(ctx),
        }
    }

    // ===== Opcode helper methods (NesCpu.h:284-801) =====
    //
    // Each method below implements one logical 6502 mnemonic using
    // `self.operand` + `self.inst_addr_mode` set by `exec`. The match
    // table in `execute_opcode` dispatches multiple opcodes to the same
    // helper when they share semantics (e.g. all `LDA` variants).

    fn ora<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.set_a(self.a | v);
    }
    fn and<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.set_a(self.a & v);
    }
    fn eor<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.set_a(self.a ^ v);
    }
    fn adc_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.adc(v);
    }
    fn sbc_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.sbc(v);
    }
    fn cmp_a<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.compare(self.a, v);
    }
    fn cpx_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.compare(self.x, v);
    }
    fn cpy_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.compare(self.y, v);
    }
    fn lda<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.set_a(v);
    }
    fn ldx<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.set_x(v);
    }
    fn ldy<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.set_y(v);
    }
    fn sta<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        self.memory_write(self.operand, self.a, MemoryOperationType::Write, ctx);
    }
    fn stx<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        self.memory_write(self.operand, self.x, MemoryOperationType::Write, ctx);
    }
    fn sty<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        self.memory_write(self.operand, self.y, MemoryOperationType::Write, ctx);
    }
    fn tax(&mut self) {
        self.set_x(self.a);
    }
    fn tay(&mut self) {
        self.set_y(self.a);
    }
    fn tsx(&mut self) {
        self.set_x(self.sp);
    }
    fn txa(&mut self) {
        self.set_a(self.x);
    }
    fn txs(&mut self) {
        // TXS does NOT update flags. Mirrors NesCpu.h:477.
        self.sp = self.x;
    }
    fn tya(&mut self) {
        self.set_a(self.y);
    }
    fn pha<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        self.push_op(self.a, ctx);
    }
    fn php<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        self.push_op(self.status | BREAK | UNUSED, ctx);
    }
    fn pla<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        // Mesen2 PLA does a dummy read at PC before pop (NesCpu.h:485-488).
        self.dummy_read_op(ctx);
        let v = self.pop_op(ctx);
        self.set_a(v);
    }
    fn plp<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        self.dummy_read_op(ctx);
        let v = self.pop_op(ctx);
        // Mesen2 SetPS masks `& 0xCF` (NesCpu.h:178) -drops BREAK + UNUSED.
        self.status = v & 0xCF;
    }
    fn inx(&mut self) {
        self.set_x(self.x.wrapping_add(1));
    }
    fn iny(&mut self) {
        self.set_y(self.y.wrapping_add(1));
    }
    fn dex(&mut self) {
        self.set_x(self.x.wrapping_sub(1));
    }
    fn dey(&mut self) {
        self.set_y(self.y.wrapping_sub(1));
    }

    fn inc_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let addr = self.operand;
        // Mesen2 NesCpu.h:327-338 -INC clears N+Z BEFORE memory read,
        // so the DummyWrite cycle observes both flags as 0. Without this,
        // wram_write_log captured during DummyWrite carries the previous
        // instruction's Z/N flag, causing oracle-comparison diffs at cycle
        // 60341 across 6 phase-full ROMs.
        self.set_flag(NEGATIVE, false);
        self.set_flag(ZERO, false);
        let v = self.memory_read(addr, MemoryOperationType::Read, ctx);
        // Mesen2 INC does a dummy write of the old value (NesCpu.h:333).
        self.memory_write(addr, v, MemoryOperationType::DummyWrite, ctx);
        let new = v.wrapping_add(1);
        self.set_zn(new);
        self.memory_write(addr, new, MemoryOperationType::Write, ctx);
    }
    fn dec_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let addr = self.operand;
        // Mesen2 NesCpu.h:340-350 -DEC clears N+Z BEFORE memory read.
        self.set_flag(NEGATIVE, false);
        self.set_flag(ZERO, false);
        let v = self.memory_read(addr, MemoryOperationType::Read, ctx);
        self.memory_write(addr, v, MemoryOperationType::DummyWrite, ctx);
        let new = v.wrapping_sub(1);
        self.set_zn(new);
        self.memory_write(addr, new, MemoryOperationType::Write, ctx);
    }

    /// ASL accumulator. Mirrors `NesCpu::ASL_Acc` (NesCpu.h:500).
    fn asl_acc(&mut self) {
        self.set_flag(CARRY, self.a & 0x80 != 0);
        let v = self.a << 1;
        self.set_a(v);
    }
    fn asl_mem_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let addr = self.operand;
        let v = self.memory_read(addr, MemoryOperationType::Read, ctx);
        self.memory_write(addr, v, MemoryOperationType::DummyWrite, ctx);
        self.set_flag(CARRY, v & 0x80 != 0);
        let new = v << 1;
        self.set_zn(new);
        self.memory_write(addr, new, MemoryOperationType::Write, ctx);
    }
    fn lsr_acc(&mut self) {
        self.set_flag(CARRY, self.a & 0x01 != 0);
        let v = self.a >> 1;
        self.set_a(v);
    }
    fn lsr_mem_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let addr = self.operand;
        let v = self.memory_read(addr, MemoryOperationType::Read, ctx);
        self.memory_write(addr, v, MemoryOperationType::DummyWrite, ctx);
        self.set_flag(CARRY, v & 0x01 != 0);
        let new = v >> 1;
        self.set_zn(new);
        self.memory_write(addr, new, MemoryOperationType::Write, ctx);
    }
    fn rol_acc(&mut self) {
        let carry_in = u8::from(self.flag(CARRY));
        self.set_flag(CARRY, self.a & 0x80 != 0);
        let v = (self.a << 1) | carry_in;
        self.set_a(v);
    }
    fn rol_mem_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let addr = self.operand;
        let v = self.memory_read(addr, MemoryOperationType::Read, ctx);
        self.memory_write(addr, v, MemoryOperationType::DummyWrite, ctx);
        let carry_in = u8::from(self.flag(CARRY));
        self.set_flag(CARRY, v & 0x80 != 0);
        let new = (v << 1) | carry_in;
        self.set_zn(new);
        self.memory_write(addr, new, MemoryOperationType::Write, ctx);
    }
    fn ror_acc(&mut self) {
        let carry_in = if self.flag(CARRY) { 0x80 } else { 0 };
        self.set_flag(CARRY, self.a & 0x01 != 0);
        let v = (self.a >> 1) | carry_in;
        self.set_a(v);
    }
    fn ror_mem_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let addr = self.operand;
        let v = self.memory_read(addr, MemoryOperationType::Read, ctx);
        self.memory_write(addr, v, MemoryOperationType::DummyWrite, ctx);
        let carry_in = if self.flag(CARRY) { 0x80 } else { 0 };
        self.set_flag(CARRY, v & 0x01 != 0);
        let new = (v >> 1) | carry_in;
        self.set_zn(new);
        self.memory_write(addr, new, MemoryOperationType::Write, ctx);
    }

    fn jmp_abs(&mut self) {
        self.pc = self.operand;
    }
    fn jmp_ind<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        // Mesen2 `GetInd` (NesCpu.h:234-243) implements the page-wrap quirk:
        // when low byte is 0xFF, the high byte is fetched from the same page.
        let ptr = self.operand;
        let lo = self.memory_read(ptr, MemoryOperationType::Read, ctx);
        let hi_addr = if ptr & 0xFF == 0xFF {
            ptr & 0xFF00
        } else {
            ptr.wrapping_add(1)
        };
        let hi = self.memory_read(hi_addr, MemoryOperationType::Read, ctx);
        self.pc = u16::from(lo) | (u16::from(hi) << 8);
    }
    fn jsr<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        // Mesen2 JSR (NesCpu.h:517-523): fetch lo, dummy read, push PC,
        // fetch hi, jump.
        let lo = self.read_byte_mc(ctx);
        self.dummy_read_op(ctx);
        let pc = self.pc;
        self.push_u16_op(pc, ctx);
        let hi = self.read_byte_mc(ctx);
        self.pc = u16::from(lo) | (u16::from(hi) << 8);
    }
    fn rts<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        // Mesen2 RTS (NesCpu.h:525-530): dummy, pop word, dummy, PC++.
        self.dummy_read_op(ctx);
        let addr = self.pop_u16_op(ctx);
        self.dummy_read_op(ctx);
        self.pc = addr.wrapping_add(1);
    }
    fn brk_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        // Mesen2 BRK (NesCpu.cpp:220-239): push PC+1, push status with
        // B|R, jump to NMI vector if hijacked, else IRQ vector. Suppress
        // pending NMI sample for nmi_and_brk test.
        let pc = self.pc.wrapping_add(1);
        self.push_u16_op(pc, ctx);
        let pushed = self.status | BREAK | UNUSED;
        let vector = if self.need_nmi {
            self.need_nmi = false;
            0xFFFA
        } else {
            0xFFFE
        };
        self.push_op(pushed, ctx);
        self.status |= INTERRUPT_DISABLE;
        let lo = self.memory_read(vector, MemoryOperationType::Read, ctx);
        let hi = self.memory_read(vector + 1, MemoryOperationType::Read, ctx);
        self.pc = u16::from(lo) | (u16::from(hi) << 8);
        self.prev_need_nmi = false;
    }
    fn rti_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        // Mesen2 RTI (NesCpu.h:574-578): dummy, pop status, pop PC.
        self.dummy_read_op(ctx);
        let s = self.pop_op(ctx);
        self.status = s & 0xCF;
        self.pc = self.pop_u16_op(ctx);
    }
    fn bit_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.bit(v);
    }
    fn nop_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        // Mesen2 NOP calls GetOperandValue (NesCpu.h:580-583) so the
        // dummy read still happens for the cycle count.
        let _ = self.operand_value(ctx);
    }
    #[cold]
    fn hlt_op(&mut self) {
        // Mesen2 HLT (NesCpu.cpp:571-593): freeze CPU. We set jammed
        // and back PC up so the next exec re-fetches the KIL byte and
        // re-enters this method. Mesen2 instead decrements PC inside
        // HLT(); functionally equivalent.
        self.jammed = true;
        self.pc = self.pc.wrapping_sub(1);
        self.prev_run_irq = false;
        self.prev_need_nmi = false;
    }

    /// Branch on condition. Reads the relative offset from `self.operand`
    /// (already fetched). Mirrors `NesCpu::BranchRelative`
    /// (NesCpu.h:432-448) including the IRQ-suppression quirk
    /// ("a taken non-page-crossing branch ignores IRQ/NMI during its
    /// last clock").
    fn branch_op<M: Mapper + ?Sized>(&mut self, take: bool, ctx: &mut CpuBusContext<'_, M>) {
        let offset = self.operand as i8;
        if !take {
            return;
        }
        // Mesen2 branch_delays_irq quirk (NesCpu.h:437-439).
        if self.run_irq && !self.prev_run_irq {
            self.run_irq = false;
        }
        self.dummy_read_op(ctx);
        let new_pc = self.pc.wrapping_add(offset as u16);
        if (self.pc & 0xFF00) != (new_pc & 0xFF00) {
            self.dummy_read_op(ctx);
        }
        self.pc = new_pc;
    }

    // ----- Unofficial / illegal opcodes -----

    fn slo_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let addr = self.operand;
        let v = self.operand_value(ctx);
        self.memory_write(addr, v, MemoryOperationType::DummyWrite, ctx);
        self.set_flag(CARRY, v & 0x80 != 0);
        let shifted = v << 1;
        let result = self.a | shifted;
        self.set_a(result);
        self.memory_write(addr, shifted, MemoryOperationType::Write, ctx);
    }
    fn sre_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let addr = self.operand;
        let v = self.operand_value(ctx);
        self.memory_write(addr, v, MemoryOperationType::DummyWrite, ctx);
        self.set_flag(CARRY, v & 0x01 != 0);
        let shifted = v >> 1;
        let result = self.a ^ shifted;
        self.set_a(result);
        self.memory_write(addr, shifted, MemoryOperationType::Write, ctx);
    }
    fn rla_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let addr = self.operand;
        let v = self.operand_value(ctx);
        self.memory_write(addr, v, MemoryOperationType::DummyWrite, ctx);
        let carry_in = u8::from(self.flag(CARRY));
        self.set_flag(CARRY, v & 0x80 != 0);
        let shifted = (v << 1) | carry_in;
        let result = self.a & shifted;
        self.set_a(result);
        self.memory_write(addr, shifted, MemoryOperationType::Write, ctx);
    }
    fn rra_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let addr = self.operand;
        let v = self.operand_value(ctx);
        self.memory_write(addr, v, MemoryOperationType::DummyWrite, ctx);
        let carry_in = if self.flag(CARRY) { 0x80 } else { 0 };
        self.set_flag(CARRY, v & 0x01 != 0);
        let shifted = (v >> 1) | carry_in;
        self.adc(shifted);
        self.memory_write(addr, shifted, MemoryOperationType::Write, ctx);
    }
    fn dcp_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let addr = self.operand;
        let v = self.operand_value(ctx);
        self.memory_write(addr, v, MemoryOperationType::DummyWrite, ctx);
        let new = v.wrapping_sub(1);
        self.compare(self.a, new);
        self.memory_write(addr, new, MemoryOperationType::Write, ctx);
    }
    fn isb_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let addr = self.operand;
        let v = self.operand_value(ctx);
        self.memory_write(addr, v, MemoryOperationType::DummyWrite, ctx);
        let new = v.wrapping_add(1);
        self.sbc(new);
        self.memory_write(addr, new, MemoryOperationType::Write, ctx);
    }
    fn lax_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.set_x(v);
        self.set_a(v);
    }
    fn sax_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let value = self.a & self.x;
        self.memory_write(self.operand, value, MemoryOperationType::Write, ctx);
    }
    fn aac_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.set_a(self.a & v);
        self.set_flag(CARRY, self.flag(NEGATIVE));
    }
    fn asr_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        self.set_flag(CARRY, false);
        self.a &= v;
        self.set_flag(CARRY, self.a & 0x01 != 0);
        let new = self.a >> 1;
        self.set_a(new);
    }
    fn arr_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        let carry_in = if self.flag(CARRY) { 0x80 } else { 0 };
        let new = ((self.a & v) >> 1) | carry_in;
        self.a = new;
        self.set_zn(new);
        self.set_flag(CARRY, new & 0x40 != 0);
        let c = u8::from(self.flag(CARRY));
        let v_bit = (new >> 5) & 0x01;
        self.set_flag(OVERFLOW, (c ^ v_bit) != 0);
    }
    fn atx_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        // Mesen2 ATX (NesCpu.h:693-700): LDA + TAX.
        let v = self.operand_value(ctx);
        self.set_a(v);
        self.set_x(self.a);
    }
    fn axs_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let v = self.operand_value(ctx);
        let computed = (self.a & self.x).wrapping_sub(v);
        self.set_flag(CARRY, (self.a & self.x) >= v);
        self.set_x(computed);
    }

    /// Indexed store quirk (SHY/SHX/SHAA/SHAZ/TAS). Mirrors
    /// `NesCpu::SyaSxaAxa` (NesCpu.h:716-745) -the address-high-byte
    /// AND quirk on page cross plus the DMA-interrupted edge case.
    fn sya_sxa_axa<M: Mapper + ?Sized>(
        &mut self,
        base_addr: u16,
        index_reg: u8,
        value_reg: u8,
        ctx: &mut CpuBusContext<'_, M>,
    ) {
        let crossed = page_crossed(base_addr, base_addr.wrapping_add(u16::from(index_reg)));
        let pre_cycles = self.cycles;
        let dummy_addr = base_addr
            .wrapping_add(u16::from(index_reg))
            .wrapping_sub(if crossed { 0x100 } else { 0 });
        self.memory_read(dummy_addr, MemoryOperationType::DummyRead, ctx);
        let had_dma = self.cycles.wrapping_sub(pre_cycles) > 1;
        let operand_addr = base_addr.wrapping_add(u16::from(index_reg));
        let mut addr_high = (operand_addr >> 8) as u8;
        let addr_low = (operand_addr & 0xFF) as u8;
        if crossed {
            addr_high &= value_reg;
        }
        let value = if had_dma {
            value_reg
        } else {
            value_reg & ((base_addr >> 8) as u8).wrapping_add(1)
        };
        let final_addr = (u16::from(addr_high) << 8) | u16::from(addr_low);
        self.memory_write(final_addr, value, MemoryOperationType::Write, ctx);
    }

    fn shy_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let base = self.read_word_mc(ctx);
        self.sya_sxa_axa(base, self.x, self.y, ctx);
    }
    fn shx_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let base = self.read_word_mc(ctx);
        self.sya_sxa_axa(base, self.y, self.x, ctx);
    }
    fn sha_abs_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        let base = self.read_word_mc(ctx);
        self.sya_sxa_axa(base, self.y, self.x & self.a, ctx);
    }
    fn sha_zp_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        // (Indirect),Y form. Mirrors NesCpu.h:762-776.
        let zero = self.read_byte_mc(ctx);
        let base = if zero == 0xFF {
            let lo = self.memory_read(0x00FF, MemoryOperationType::Read, ctx);
            let hi = self.memory_read(0x0000, MemoryOperationType::Read, ctx);
            u16::from(lo) | (u16::from(hi) << 8)
        } else {
            let lo = self.memory_read(u16::from(zero), MemoryOperationType::Read, ctx);
            let hi = self.memory_read(
                u16::from(zero.wrapping_add(1)),
                MemoryOperationType::Read,
                ctx,
            );
            u16::from(lo) | (u16::from(hi) << 8)
        };
        self.sya_sxa_axa(base, self.y, self.x & self.a, ctx);
    }
    fn tas_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        // SHA abs,Y but also SP = A & X. Mirrors NesCpu.h:778-783.
        self.sha_abs_op(ctx);
        self.sp = self.x & self.a;
    }
    fn ane_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        // Mesen2 ANE (NesCpu.h:788-792): A = (A | 0xEE) & X & imm.
        let imm = self.operand_value(ctx);
        self.set_a((self.a | 0xEE) & self.x & imm);
    }
    fn las_op<M: Mapper + ?Sized>(&mut self, ctx: &mut CpuBusContext<'_, M>) {
        // Mesen2 LAS (NesCpu.h:794-801): A = X = SP = memory & SP.
        let v = self.operand_value(ctx);
        self.set_a(v & self.sp);
        self.set_x(self.a);
        self.sp = self.a;
    }

    // Shared opcode helpers used by `exec(ctx)`.

    fn bit(&mut self, value: u8) {
        self.set_flag(ZERO, self.a & value == 0);
        self.set_flag(OVERFLOW, value & OVERFLOW != 0);
        self.set_flag(NEGATIVE, value & NEGATIVE != 0);
    }

    fn adc(&mut self, value: u8) {
        let carry = u8::from(self.flag(CARRY));
        let sum = u16::from(self.a) + u16::from(value) + u16::from(carry);
        let result = sum as u8;
        self.set_flag(CARRY, sum > 0xff);
        self.set_flag(
            OVERFLOW,
            (!(self.a ^ value) & (self.a ^ result) & 0x80) != 0,
        );
        self.a = result;
        self.set_zn(self.a);
    }

    fn sbc(&mut self, value: u8) {
        self.adc(!value);
    }

    fn compare(&mut self, register: u8, value: u8) {
        let result = register.wrapping_sub(value);
        self.set_flag(CARRY, register >= value);
        self.set_zn(result);
    }

    fn set_zn(&mut self, value: u8) {
        self.set_flag(ZERO, value == 0);
        self.set_flag(NEGATIVE, value & 0x80 != 0);
    }

    fn flag(&self, flag: u8) -> bool {
        self.status & flag != 0
    }

    fn set_flag(&mut self, flag: u8, enabled: bool) {
        if enabled {
            self.status |= flag;
        } else {
            self.status &= !flag;
        }
    }
}

impl AddrMode {
    fn snapshot_tag(self) -> u8 {
        match self {
            AddrMode::None => 0,
            AddrMode::Imp => 1,
            AddrMode::Acc => 2,
            AddrMode::Imm => 3,
            AddrMode::Rel => 4,
            AddrMode::Zero => 5,
            AddrMode::ZeroX => 6,
            AddrMode::ZeroY => 7,
            AddrMode::Ind => 8,
            AddrMode::IndX => 9,
            AddrMode::IndY => 10,
            AddrMode::IndYW => 11,
            AddrMode::Abs => 12,
            AddrMode::AbsX => 13,
            AddrMode::AbsXW => 14,
            AddrMode::AbsY => 15,
            AddrMode::AbsYW => 16,
        }
    }

    fn from_snapshot_tag(tag: u8) -> nesle_common::Result<Self> {
        let mode = match tag {
            0 => AddrMode::None,
            1 => AddrMode::Imp,
            2 => AddrMode::Acc,
            3 => AddrMode::Imm,
            4 => AddrMode::Rel,
            5 => AddrMode::Zero,
            6 => AddrMode::ZeroX,
            7 => AddrMode::ZeroY,
            8 => AddrMode::Ind,
            9 => AddrMode::IndX,
            10 => AddrMode::IndY,
            11 => AddrMode::IndYW,
            12 => AddrMode::Abs,
            13 => AddrMode::AbsX,
            14 => AddrMode::AbsXW,
            15 => AddrMode::AbsY,
            16 => AddrMode::AbsYW,
            _ => {
                return Err(nesle_common::NesleError::InvalidState(format!(
                    "invalid CPU addressing mode snapshot tag {tag}"
                )))
            }
        };
        Ok(mode)
    }
}

fn page_crossed(base: u16, addr: u16) -> bool {
    base & 0xff00 != addr & 0xff00
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Mesen2 Phase B.2 ADDR_MODE consistency tests =====
    //
    // Spot-checks against the Mesen2 reference table (NesCpu.cpp:44-62).
    // Doesn't enumerate all 256 entries (the table is hand-verified vs
    // Mesen2 line-by-line in the source), but pins the most error-prone
    // cells: the per-row "special" cells (None for SHAZ/TAS/SHY/SHX,
    // Ind for JMP indirect, the immediate vs ZP cells in row 8/A/C/E).

    #[test]
    fn addr_mode_table_jsr_and_brk_are_none() {
        // BRK (0x00) is Imp per Mesen2 (operand discarded). JSR (0x20)
        // is None because its operand fetch is non-standard
        // (lo / dummy / hi sequence with push between lo and hi).
        assert_eq!(ADDR_MODE[0x00], AddrMode::Imp);
        assert_eq!(ADDR_MODE[0x20], AddrMode::None);
    }

    #[test]
    fn addr_mode_table_jmp_indirect_uses_ind() {
        // JMP (ind) at 0x6C is the only AddrMode::Ind opcode.
        assert_eq!(ADDR_MODE[0x6C], AddrMode::Ind);
        // JMP abs at 0x4C uses AddrMode::Abs.
        assert_eq!(ADDR_MODE[0x4C], AddrMode::Abs);
    }

    #[test]
    fn addr_mode_table_indexed_store_quirks_are_none() {
        // SHAZ (0x93), TAS (0x9B), SHY (0x9C), SHX (0x9E), SHAA (0x9F):
        // Mesen2 marks these `None` because operand fetch + write logic
        // is handled inline in SyaSxaAxa (NesCpu.h:716-745).
        assert_eq!(ADDR_MODE[0x93], AddrMode::None);
        assert_eq!(ADDR_MODE[0x9B], AddrMode::None);
        assert_eq!(ADDR_MODE[0x9C], AddrMode::None);
        assert_eq!(ADDR_MODE[0x9E], AddrMode::None);
        assert_eq!(ADDR_MODE[0x9F], AddrMode::None);
    }

    #[test]
    fn addr_mode_table_lda_variants_cover_all_modes() {
        // LDA variants: imm/zp/zp,X/(zp,X)/(zp),Y/abs/abs,X/abs,Y.
        assert_eq!(ADDR_MODE[0xA9], AddrMode::Imm); // LDA #imm
        assert_eq!(ADDR_MODE[0xA5], AddrMode::Zero); // LDA zp
        assert_eq!(ADDR_MODE[0xB5], AddrMode::ZeroX); // LDA zp,X
        assert_eq!(ADDR_MODE[0xA1], AddrMode::IndX); // LDA (zp,X)
        assert_eq!(ADDR_MODE[0xB1], AddrMode::IndY); // LDA (zp),Y READ
        assert_eq!(ADDR_MODE[0xAD], AddrMode::Abs); // LDA abs
        assert_eq!(ADDR_MODE[0xBD], AddrMode::AbsX); // LDA abs,X READ
        assert_eq!(ADDR_MODE[0xB9], AddrMode::AbsY); // LDA abs,Y READ
    }

    #[test]
    fn addr_mode_table_sta_uses_write_form_for_indexed() {
        // STA variants: zp/zp,X/(zp,X)/(zp),Y/abs/abs,X/abs,Y.
        // Indexed addressing for STORE opcodes uses AbsXW/AbsYW/IndYW
        // (W = write form, includes dummy read on every access).
        assert_eq!(ADDR_MODE[0x85], AddrMode::Zero);
        assert_eq!(ADDR_MODE[0x95], AddrMode::ZeroX);
        assert_eq!(ADDR_MODE[0x81], AddrMode::IndX);
        assert_eq!(ADDR_MODE[0x91], AddrMode::IndYW);
        assert_eq!(ADDR_MODE[0x8D], AddrMode::Abs);
        assert_eq!(ADDR_MODE[0x9D], AddrMode::AbsXW);
        assert_eq!(ADDR_MODE[0x99], AddrMode::AbsYW);
    }

    #[test]
    fn addr_mode_table_rmw_uses_write_form_for_indexed() {
        // RMW opcodes (ASL/LSR/ROL/ROR/INC/DEC) use AbsXW for absolute,X
        // because they need the dummy read for the page-cross alignment.
        assert_eq!(ADDR_MODE[0x1E], AddrMode::AbsXW); // ASL abs,X
        assert_eq!(ADDR_MODE[0x5E], AddrMode::AbsXW); // LSR abs,X
        assert_eq!(ADDR_MODE[0x3E], AddrMode::AbsXW); // ROL abs,X
        assert_eq!(ADDR_MODE[0x7E], AddrMode::AbsXW); // ROR abs,X
        assert_eq!(ADDR_MODE[0xFE], AddrMode::AbsXW); // INC abs,X
        assert_eq!(ADDR_MODE[0xDE], AddrMode::AbsXW); // DEC abs,X
    }

    #[test]
    fn addr_mode_table_branches_are_relative() {
        // All 8 branch opcodes: 0x10, 0x30, 0x50, 0x70, 0x90, 0xB0,
        // 0xD0, 0xF0.
        for branch in [0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0u8] {
            assert_eq!(
                ADDR_MODE[branch as usize],
                AddrMode::Rel,
                "branch opcode 0x{branch:02X} should be Rel"
            );
        }
    }

    #[test]
    fn addr_mode_table_kil_opcodes_are_none() {
        // KIL/HLT opcodes (0x02..0xF2 in increments of 0x10, plus 0x12,
        // 0x22, 0x32...) are AddrMode::None because they trigger HLT
        // without operand fetch.
        for kil in [
            0x02, 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72, 0x92, 0xB2, 0xD2, 0xF2u8,
        ] {
            assert_eq!(
                ADDR_MODE[kil as usize],
                AddrMode::None,
                "KIL opcode 0x{kil:02X} should be None"
            );
        }
    }

    #[test]
    fn cpu_default_uses_mesen2_ntsc_clock_counts() {
        let cpu = Cpu::default();
        // Mesen2 NesCpu defaults: start_clock_count = 6, end_clock_count = 6
        // (NesCpu.cpp:74-75, NTSC).
        assert_eq!(cpu.start_clock_count, 6);
        assert_eq!(cpu.end_clock_count, 6);
        // ppu_offset = 1 (deterministic default when RandomizeCpuPpuAlignment off).
        assert_eq!(cpu.ppu_offset, 1);
        // irq_mask = 0xFF means all IRQ sources unmasked.
        assert_eq!(cpu.irq_mask, 0xFF);
        assert_eq!(cpu.master_clock, 0);
    }

    #[test]
    fn cpu_set_master_clock_divider_ntsc_pal_dendy() {
        use crate::cartridge::Region;
        let mut cpu = Cpu::default();
        cpu.set_master_clock_divider(Region::Ntsc);
        assert_eq!(cpu.start_clock_count, 6);
        assert_eq!(cpu.end_clock_count, 6);
        cpu.set_master_clock_divider(Region::Pal);
        assert_eq!(cpu.start_clock_count, 8);
        assert_eq!(cpu.end_clock_count, 8);
        cpu.set_master_clock_divider(Region::Dendy);
        assert_eq!(cpu.start_clock_count, 7);
        assert_eq!(cpu.end_clock_count, 8);
    }

    #[test]
    fn interrupt_lines_irq_source_bits_are_independent() {
        let mut lines = InterruptLines::default();
        assert!(!lines.has_irq_source(IrqSource::External));
        lines.set_irq_source(IrqSource::External);
        lines.set_irq_source(IrqSource::FrameCounter);
        assert!(lines.has_irq_source(IrqSource::External));
        assert!(lines.has_irq_source(IrqSource::FrameCounter));
        assert!(!lines.has_irq_source(IrqSource::Dmc));
        lines.clear_irq_source(IrqSource::External);
        assert!(!lines.has_irq_source(IrqSource::External));
        assert!(lines.has_irq_source(IrqSource::FrameCounter));
    }

    #[test]
    fn cpu_run_dma_transfer_sets_oam_dma_state() {
        let mut cpu = Cpu::default();
        cpu.run_dma_transfer(0x07);
        assert!(cpu.sprite_dma_transfer);
        assert_eq!(cpu.sprite_dma_offset, 0x07);
        assert!(cpu.need_halt);
    }

    #[test]
    fn cpu_start_stop_dmc_transfer_state_transitions() {
        let mut cpu = Cpu::default();
        cpu.start_dmc_transfer();
        assert!(cpu.dmc_dma_running);
        assert!(cpu.need_dummy_read);
        assert!(cpu.need_halt);
        // Stop while still in halt phase = full cancel.
        cpu.stop_dmc_transfer();
        assert!(!cpu.dmc_dma_running);
        assert!(!cpu.need_dummy_read);
        assert!(!cpu.need_halt);

        // Re-start, then advance past halt by clearing need_halt manually
        // (simulating the DMA state machine having entered the transfer
        // phase). Stop now should set abort flag instead of full cancel.
        cpu.start_dmc_transfer();
        cpu.need_halt = false;
        cpu.stop_dmc_transfer();
        assert!(cpu.dmc_dma_running, "DMA still running until aborted");
        assert!(cpu.abort_dmc_dma, "abort flag set instead of clean cancel");
    }

    // ===== Mesen2 exec(ctx) end-to-end smoke tests =====
    //
    // These tests validate the exec(ctx) path: build a real
    // `CpuBusContext` from `Bus + Mapper + Ppu + Apu + ControllerPorts`,
    // call `cpu.exec(ctx)`, observe the effects on CPU registers + WRAM
    // + cycle counter. They prove `exec(ctx)` -> `memory_read/write` ->
    // `start_cpu_cycle (master_clock + ppu.run + mapper/apu cpu_clock)` ->
    // `bus_read/bus_write` -> `end_cpu_cycle` works end-to-end.

    use crate::apu::Apu;
    use crate::bus::{Bus, CpuBusContext};
    use crate::cartridge::Mirroring;
    use crate::input::ControllerPorts;
    use crate::mapper::Mapper;
    use crate::ppu::Ppu;

    /// Test mapper that owns 32KB of PRG ROM at $8000-$FFFF + 8KB of
    /// CHR RAM. Writes to PRG ROM are silently dropped (real cartridge
    /// behavior); writes to CHR are stored.
    #[derive(Debug)]
    struct ExecTestMapper {
        prg: [u8; 0x8000],
        chr: [u8; 0x2000],
    }

    impl ExecTestMapper {
        fn new() -> Self {
            Self {
                prg: [0xEA; 0x8000], // default: NOP
                chr: [0; 0x2000],
            }
        }

        /// Plant a program at `start` (must be in $8000..=$FFFF range).
        fn plant(&mut self, start: u16, bytes: &[u8]) {
            let offset = (start - 0x8000) as usize;
            self.prg[offset..offset + bytes.len()].copy_from_slice(bytes);
        }
    }

    impl Mapper for ExecTestMapper {
        fn mapper_id(&self) -> u16 {
            0
        }
        fn name(&self) -> &'static str {
            "EXEC_TEST"
        }
        fn cpu_read(&mut self, addr: u16) -> u8 {
            if addr >= 0x8000 {
                self.prg[(addr - 0x8000) as usize]
            } else {
                0
            }
        }
        fn cpu_code_read(&self, addr: u16) -> Option<u8> {
            if addr >= 0x8000 {
                Some(self.prg[(addr - 0x8000) as usize])
            } else {
                None
            }
        }
        fn cpu_write(&mut self, _addr: u16, _value: u8, _interrupt: &mut InterruptLines) {}
        fn ppu_read(&mut self, addr: u16) -> u8 {
            self.chr[(addr & 0x1FFF) as usize]
        }
        fn debug_ppu_read(&self, addr: u16) -> u8 {
            self.chr[(addr & 0x1FFF) as usize]
        }
        fn ppu_write(&mut self, addr: u16, value: u8) {
            self.chr[(addr & 0x1FFF) as usize] = value;
        }
        fn nametable_mirroring(&self) -> Mirroring {
            Mirroring::Horizontal
        }
    }

    /// Test fixture holding all CpuBusContext components.
    struct ExecFixture {
        bus: Bus,
        mapper: ExecTestMapper,
        ppu: Ppu,
        apu: Apu,
        controllers: ControllerPorts,
        interrupt: InterruptLines,
    }

    impl ExecFixture {
        fn new() -> Self {
            Self {
                bus: Bus::default(),
                mapper: ExecTestMapper::new(),
                ppu: Ppu::default(),
                apu: Apu::default(),
                controllers: ControllerPorts::default(),
                interrupt: InterruptLines::default(),
            }
        }

        fn ctx(&mut self) -> CpuBusContext<'_, ExecTestMapper> {
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
    fn exec_lda_imm_loads_accumulator() {
        // Program: LDA #$42 at $8000.
        let mut fx = ExecFixture::new();
        fx.mapper.plant(0x8000, &[0xA9, 0x42]);
        let mut cpu = Cpu {
            pc: 0x8000,
            ..Cpu::default()
        };
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        assert_eq!(cpu.a, 0x42);
        assert_eq!(cpu.pc, 0x8002);
        // exec consumed 2 CPU cycles (opcode fetch + operand fetch).
        assert_eq!(cpu.cycles, 2);
    }

    #[test]
    fn exec_sta_abs_writes_wram() {
        // Program: STA $0200 at $8000 with A=0x55.
        let mut fx = ExecFixture::new();
        fx.mapper.plant(0x8000, &[0x8D, 0x00, 0x02]);
        let mut cpu = Cpu {
            pc: 0x8000,
            a: 0x55,
            ..Cpu::default()
        };
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        assert_eq!(cpu.pc, 0x8003);
        // STA abs: opcode + 2 operand bytes + 1 write = 4 cycles.
        assert_eq!(cpu.cycles, 4);
        assert_eq!(fx.bus.wram()[0x200], 0x55);
    }

    #[test]
    fn exec_lda_sta_round_trip_through_wram() {
        // LDA #$AA; STA $0150; LDA #$00; LDA $0150 -sanity round trip.
        let mut fx = ExecFixture::new();
        fx.mapper.plant(
            0x8000,
            &[
                0xA9, 0xAA, // LDA #$AA
                0x8D, 0x50, 0x01, // STA $0150
                0xA9, 0x00, // LDA #$00
                0xAD, 0x50, 0x01, // LDA $0150
            ],
        );
        let mut cpu = Cpu {
            pc: 0x8000,
            ..Cpu::default()
        };
        for _ in 0..4 {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        assert_eq!(cpu.a, 0xAA);
        assert_eq!(fx.bus.wram()[0x150], 0xAA);
    }

    #[test]
    fn exec_jmp_abs_sets_pc() {
        // JMP $1234 at $8000.
        let mut fx = ExecFixture::new();
        fx.mapper.plant(0x8000, &[0x4C, 0x34, 0x12]);
        let mut cpu = Cpu {
            pc: 0x8000,
            ..Cpu::default()
        };
        let mut ctx = fx.ctx();
        cpu.exec(&mut ctx);
        assert_eq!(cpu.pc, 0x1234);
        // JMP abs: opcode + 2 operand bytes = 3 cycles.
        assert_eq!(cpu.cycles, 3);
    }

    #[test]
    fn exec_jsr_rts_round_trip_preserves_pc_and_sp() {
        // At $8000: JSR $8010 ; NOP at $8003
        // At $8010: RTS
        let mut fx = ExecFixture::new();
        fx.mapper.plant(0x8000, &[0x20, 0x10, 0x80, 0xEA]); // JSR $8010 ; NOP
        fx.mapper.plant(0x8010, &[0x60]); // RTS
        let initial_sp = 0xFD;
        let mut cpu = Cpu {
            pc: 0x8000,
            sp: initial_sp,
            ..Cpu::default()
        };
        // JSR.
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        assert_eq!(cpu.pc, 0x8010);
        assert_eq!(cpu.sp, initial_sp - 2, "JSR pushed 2 bytes");
        // RTS.
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        assert_eq!(cpu.pc, 0x8003, "RTS returns to byte after JSR");
        assert_eq!(cpu.sp, initial_sp, "SP restored");
    }

    #[test]
    fn exec_branch_taken_consumes_extra_cycle() {
        // SEC ; BCS +2  (always taken)
        let mut fx = ExecFixture::new();
        fx.mapper.plant(0x8000, &[0x38, 0xB0, 0x02]);
        let mut cpu = Cpu {
            pc: 0x8000,
            ..Cpu::default()
        };
        // SEC: 2 cycles.
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        assert!(cpu.flag(CARRY));
        // BCS taken: 3 cycles (opcode + operand + dummy read for taken).
        let cycles_before = cpu.cycles;
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        assert_eq!(cpu.pc, 0x8005);
        assert_eq!(cpu.cycles - cycles_before, 3, "taken branch = 3 cycles");
    }

    #[test]
    fn exec_branch_not_taken_skips_dummy_read() {
        // CLC ; BCS +2  (not taken)
        let mut fx = ExecFixture::new();
        fx.mapper.plant(0x8000, &[0x18, 0xB0, 0x02]);
        let mut cpu = Cpu {
            pc: 0x8000,
            ..Cpu::default()
        };
        // CLC: 2 cycles.
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        let cycles_before = cpu.cycles;
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        assert_eq!(cpu.pc, 0x8003, "PC advances past branch operand");
        assert_eq!(cycles_before + 2, cpu.cycles, "not-taken branch = 2 cycles");
    }

    #[test]
    fn exec_adc_carry_chain() {
        // LDA #$FF ; CLC ; ADC #$01  ->  A = 0, C = 1
        let mut fx = ExecFixture::new();
        fx.mapper.plant(0x8000, &[0xA9, 0xFF, 0x18, 0x69, 0x01]);
        let mut cpu = Cpu {
            pc: 0x8000,
            ..Cpu::default()
        };
        for _ in 0..3 {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        assert_eq!(cpu.a, 0x00);
        assert!(cpu.flag(CARRY));
        assert!(cpu.flag(ZERO));
    }

    #[test]
    fn exec_inc_zp_rmw_writes_back() {
        // INC $80 via exec(ctx) path: seed memory directly, then exec
        // the INC opcode and verify RMW writes back to WRAM.
        let mut fx = ExecFixture::new();
        fx.bus.set_wram({
            let mut wram = [0; 0x800];
            wram[0x80] = 0x05;
            wram
        });
        fx.mapper.plant(0x8000, &[0xE6, 0x80]); // INC $80
        let mut cpu = Cpu {
            pc: 0x8000,
            ..Cpu::default()
        };
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        assert_eq!(fx.bus.wram()[0x80], 0x06);
        // INC zp: opcode + operand + read + dummy write + write = 5 cycles.
        assert_eq!(cpu.cycles, 5);
    }

    #[test]
    fn sta_4014_triggers_synchronous_oam_dma() {
        // Mesen2 NesPpu.cpp:505 - $4014 write -> Cpu::RunDMATransfer sets
        // need_halt + sprite_dma_transfer synchronously inside memory_write.
        // Actual transfer loop runs at the NEXT memory access via
        // process_pending_dma. After STA $4014 returns, the flags ARE
        // armed; the following NOP fetch drains them.
        let mut fx = ExecFixture::new();
        fx.mapper
            .plant(0x8000, &[0xA9, 0x03, 0x8D, 0x14, 0x40, 0xEA]);
        let mut cpu = Cpu {
            pc: 0x8000,
            ..Cpu::default()
        };
        // LDA #$03
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        // STA $4014: arms DMA inside memory_write.
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        assert!(cpu.sprite_dma_transfer, "DMA armed after $4014 write");
        assert!(cpu.need_halt, "halt armed after $4014 write");
        assert_eq!(cpu.sprite_dma_offset, 0x03);
        // NOP: opcode fetch triggers process_pending_dma which drains DMA.
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        assert!(
            !cpu.sprite_dma_transfer,
            "DMA drained by NOP's opcode fetch"
        );
    }

    #[test]
    fn interrupt_dmc_dma_pending_acks_at_next_memory_access() {
        // Producer-push: APU sets ctx.interrupt.dmc_dma_pending; CPU
        // consumes it at process_pending_dma entry. Mesen2 DMC calls
        // cpu->StartDmcTransfer() directly (DeltaModulationChannel.cpp).
        let mut fx = ExecFixture::new();
        fx.mapper.plant(0x8000, &[0xEA]);
        fx.interrupt.request_dmc_dma();
        let mut cpu = Cpu {
            pc: 0x8000,
            ..Cpu::default()
        };
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        let ctx = fx.ctx();
        assert!(
            !ctx.interrupt.dmc_dma_pending,
            "DMC DMA pending consumed at next memory access"
        );
    }

    #[test]
    fn exec_master_clock_advances_per_cycle() {
        // 1 cycle = 12 master clocks (NTSC: start=6 + end=6).
        let mut fx = ExecFixture::new();
        fx.mapper.plant(0x8000, &[0xEA]); // NOP (2 cycles)
        let mut cpu = Cpu {
            pc: 0x8000,
            ..Cpu::default()
        };
        let mc_before = cpu.master_clock;
        {
            let mut ctx = fx.ctx();
            cpu.exec(&mut ctx);
        }
        let delta = cpu.master_clock - mc_before;
        // NOP = 2 cycles = 24 master clocks (NTSC start+end = 12 each).
        assert_eq!(delta, 24, "2-cycle NOP = 24 master clocks");
        // PPU was synced to master_clock - ppu_offset (ppu_offset=1).
        assert!(fx.ppu.master_clock <= cpu.master_clock);
    }
}
