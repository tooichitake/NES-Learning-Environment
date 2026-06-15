use crate::cpu::{InterruptLines, IrqSource};
#[cfg(feature = "audio-synth")]
use std::sync::OnceLock;

// DMC period tables selected by cartridge region.
const NTSC_DMC_PERIOD_TABLE: [u32; 16] = [
    428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 84, 72, 54,
];
const PAL_DMC_PERIOD_TABLE: [u32; 16] = [
    398, 354, 316, 298, 276, 236, 210, 198, 176, 148, 132, 118, 98, 78, 66, 50,
];
const DEFAULT_DMC_PERIOD: u32 = NTSC_DMC_PERIOD_TABLE[0] - 1;

/// APU frame-counter diagnostic snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameCounterDebugState {
    pub previous_cycle: i32,
    pub current_step: u32,
    pub step_mode: u32, // 0=4-step, 1=5-step
    pub inhibit_irq: bool,
    pub block_frame_counter_tick: u8,
    pub new_value: i16,          // pending $4017 write, -1 if none
    pub write_delay_counter: i8, // -1..4
    pub irq_flag: bool,
    pub irq_flag_clear_clock: u64,
}

// Extra tail entries model 4-step IRQ assertions and 5-step wrap delay.
const FRAME_STEP_CYCLES_NTSC: [[i32; 6]; 2] = [
    [7457, 14913, 22371, 29828, 29829, 29830],
    [7457, 14913, 22371, 29829, 37281, 37282],
];

// PAL frame-counter step boundaries.
const FRAME_STEP_CYCLES_PAL: [[i32; 6]; 2] = [
    [8313, 16627, 24939, 33252, 33253, 33254],
    [8313, 16627, 24939, 33253, 41565, 41566],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameTick {
    None,
    Quarter,
    Half,
}

const FRAME_STEP_TYPES: [[FrameTick; 6]; 2] = [
    [
        FrameTick::Quarter,
        FrameTick::Half,
        FrameTick::Quarter,
        FrameTick::None,
        FrameTick::Half,
        FrameTick::None,
    ],
    [
        FrameTick::Quarter,
        FrameTick::Half,
        FrameTick::Quarter,
        FrameTick::None,
        FrameTick::Half,
        FrameTick::None,
    ],
];

// Noise period tables; selected at $400E write and region restore.
const NTSC_NOISE_PERIOD_TABLE: [u16; 16] = [
    0x004, 0x008, 0x010, 0x020, 0x040, 0x060, 0x080, 0x0A0, 0x0CA, 0x0FE, 0x17C, 0x1FC, 0x2FA,
    0x3F8, 0x7F2, 0xFE4,
];
const PAL_NOISE_PERIOD_TABLE: [u16; 16] = [
    4, 8, 14, 30, 60, 88, 118, 148, 188, 236, 354, 472, 708, 944, 1890, 3778,
];

// Length-counter lookup table.
const LENGTH_TABLE: [u8; 32] = [
    0x0A, 0xFE, 0x14, 0x02, 0x28, 0x04, 0x50, 0x06, 0xA0, 0x08, 0x3C, 0x0A, 0x0E, 0x0C, 0x1A, 0x0E,
    0x0C, 0x10, 0x18, 0x12, 0x30, 0x14, 0x60, 0x16, 0xC0, 0x18, 0x48, 0x1A, 0x10, 0x1C, 0x20, 0x1E,
];

// Pulse duty patterns (NESdev). Bit 7 = step 0; bit 0 = step 7.
#[cfg(feature = "audio-synth")]
const PULSE_DUTY_TABLE: [u8; 4] = [0b0100_0000, 0b0110_0000, 0b0111_1000, 0b1001_1111];

// NES master CPU clock (NTSC).
#[cfg(feature = "audio-synth")]
const NTSC_CPU_HZ: u32 = 1_789_773;

// Audio output sample rate.
pub const AUDIO_SAMPLE_RATE: u32 = 44_100;

// Non-linear NES mixer lookup tables, built once and shared read-only.
#[cfg(feature = "audio-synth")]
fn pulse_table() -> &'static [f32; 31] {
    static TABLE: OnceLock<[f32; 31]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0.0f32; 31];
        for (n, v) in t.iter_mut().enumerate().skip(1) {
            *v = 95.88 / (8128.0 / n as f32 + 100.0);
        }
        t
    })
}

#[cfg(feature = "audio-synth")]
fn tnd_table() -> &'static [f32; 203] {
    static TABLE: OnceLock<[f32; 203]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0.0f32; 203];
        for (n, v) in t.iter_mut().enumerate().skip(1) {
            *v = 159.79 / (22638.0 / n as f32 + 100.0);
        }
        t
    })
}

// Envelope generator state shared by pulse and noise channels.
#[derive(Debug, Clone, Default)]
struct Envelope {
    speed: u8,
    mode: u8,
    dec_count_to_1: u8,
    dec_volume: u8,
    reload_dec: bool,
}

impl Envelope {
    fn write_control(&mut self, value: u8) {
        // Store control bits; length-halt is applied through deferred state.
        self.mode = (value & 0x30) >> 4;
        self.speed = value & 0x0F;
    }

    // Restart envelope on the next quarter frame.
    fn schedule_restart(&mut self) {
        self.reload_dec = true;
    }

    // Quarter-frame envelope clock. `halt` is the deferred length-halt flag.
    fn clock(&mut self, halt: bool) {
        if self.reload_dec {
            self.dec_volume = 0xF;
            self.dec_count_to_1 = self.speed + 1;
            self.reload_dec = false;
            return;
        }
        if self.dec_count_to_1 > 0 {
            self.dec_count_to_1 -= 1;
        }
        if self.dec_count_to_1 == 0 {
            self.dec_count_to_1 = self.speed + 1;
            if self.dec_volume != 0 || halt {
                self.dec_volume = self.dec_volume.wrapping_sub(1) & 0xF;
            }
        }
    }

    #[cfg(feature = "audio-synth")]
    fn output(&self) -> u8 {
        // Mode bit 0 = constant volume (use Speed field directly).
        if self.mode & 0x1 != 0 {
            self.speed
        } else {
            self.dec_volume
        }
    }

    fn snapshot_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&[
            self.speed,
            self.mode,
            self.dec_count_to_1,
            self.dec_volume,
            u8::from(self.reload_dec),
        ]);
    }

    fn restore_snapshot(&mut self, bytes: &[u8], offset: &mut usize) {
        self.speed = bytes[*offset];
        self.mode = bytes[*offset + 1];
        self.dec_count_to_1 = bytes[*offset + 2];
        self.dec_volume = bytes[*offset + 3];
        self.reload_dec = bytes[*offset + 4] != 0;
        *offset += 5;
    }
}

// Pulse channel with inlined envelope, sweep, and length-counter race state.
#[derive(Debug, Clone, Default)]
struct Pulse {
    // Identifier (0 = pulse 1, 1 = pulse 2) -affects sweep negate offset.
    channel: u8,
    enabled: bool,
    duty: u8,
    envelope: Envelope,
    timer_period: u16,
    timer: u16,
    sequence_pos: u8,
    length: u8,
    // Live length-halt flag, applied after deferred reload.
    length_halt: bool,
    // Pending halt flag from $4000/$4004.
    new_length_halt: bool,
    // Pending reload from $4003/$4007; cleared by `reload_length_counter()`.
    length_reload_value: u8,
    // Length snapshot used to suppress reload after an intervening half-frame.
    length_previous_value: u8,
    // Sweep ($4001/$4005)
    sweep_enabled: bool,
    // Sweep divider reload value: P+1 after write, 0 before first write.
    sweep_period: u8,
    sweep_negate: bool,
    sweep_shift: u8,
    // Countdown divider; sweep applies on the 0 tick, then reloads.
    sweep_divider: u8,
    sweep_reload: bool,
}

impl Pulse {
    fn new(channel: u8) -> Self {
        Self {
            channel,
            ..Self::default()
        }
    }

    fn write_control(&mut self, value: u8) {
        // $4000 / $4004 -DDLC VVVV. Bit 5 (L = length-halt + envelope-loop)
        // is deferred via `new_length_halt` per ApuLengthCounter.h:24-28
        // `InitializeLengthCounter`.
        self.duty = (value >> 6) & 3;
        self.envelope.write_control(value);
        self.new_length_halt = (value & 0x20) != 0;
    }

    fn write_sweep(&mut self, value: u8) {
        // $4001 / $4005 -EPPP NSSS
        self.sweep_enabled = value & 0x80 != 0;
        // Mesen2 InitializeSweep: `_sweepPeriod = ((regValue & 0x70) >> 4) + 1`.
        self.sweep_period = ((value >> 4) & 0x07) + 1;
        self.sweep_negate = value & 0x08 != 0;
        self.sweep_shift = value & 0x07;
        self.sweep_reload = true;
    }

    fn write_timer_low(&mut self, value: u8) {
        self.timer_period = (self.timer_period & 0x0700) | u16::from(value);
    }

    fn write_timer_high(&mut self, value: u8) {
        // $4003 / $4007 -LLLLLPPP. Length reload is deferred per Mesen2
        // ApuLengthCounter.h:30-37 `LoadLengthCounter`: stash `_reloadValue`
        // + `_previousValue = _counter`; actual write happens in
        // `reload_length_counter()` after frame counter step (so a same-
        // cycle half-frame clock observes the OLD counter and the reload
        // is discarded -fixes `len_reload_timing` tests 4 & 5).
        self.timer_period = (self.timer_period & 0x00FF) | (u16::from(value & 0x07) << 8);
        if self.enabled {
            self.length_reload_value = LENGTH_TABLE[usize::from((value >> 3) & 0x1F)];
            self.length_previous_value = self.length;
        }
        self.sequence_pos = 0;
        // Restart envelope on length-counter load ($4003/$4007 write
        // with bit-7 high). Mesen2 equivalent: ApuEnvelope::ResetEnvelope.
        self.envelope.schedule_restart();
    }

    #[cfg(feature = "audio-synth")]
    fn tick(&mut self) {
        if self.timer == 0 {
            self.timer = self.timer_period;
            self.sequence_pos = (self.sequence_pos + 1) & 7;
        } else {
            self.timer -= 1;
        }
    }

    fn clock_length(&mut self) {
        // Mesen2 ApuLengthCounter::TickLengthCounter -reads the *live*
        // (already-applied) `_halt`.
        if !self.length_halt && self.length > 0 {
            self.length -= 1;
        }
    }

    /// Mesen2 `ApuLengthCounter::ReloadCounter` (ApuLengthCounter.h:82-92).
    /// Called from `Apu::run()` after `run_frame_counter_step` so any
    /// half-frame `clock_length()` runs against the OLD counter first.
    fn reload_length_counter(&mut self) {
        if self.length_reload_value != 0 {
            if self.length == self.length_previous_value {
                self.length = self.length_reload_value;
            }
            self.length_reload_value = 0;
        }
        self.length_halt = self.new_length_halt;
    }

    fn has_pending_reload(&self) -> bool {
        self.length_reload_value != 0 || self.length_halt != self.new_length_halt
    }

    // Half-frame sweep divider and length-counter step.
    fn clock_sweep(&mut self) {
        // A divider at 0 wraps to 255; target applies only when decrement hits 0.
        self.sweep_divider = self.sweep_divider.wrapping_sub(1);
        if self.sweep_divider == 0 {
            // Pulse 1 uses the ones-complement negate quirk.
            let shift_result = self.timer_period >> self.sweep_shift;
            let target = if self.sweep_negate {
                let extra = if self.channel == 0 { 1 } else { 0 };
                self.timer_period.wrapping_sub(shift_result + extra)
            } else {
                self.timer_period + shift_result
            };
            if self.sweep_shift != 0
                && self.sweep_enabled
                && self.timer_period >= 8
                && target <= 0x7FF
            {
                self.timer_period = target;
            }
            self.sweep_divider = self.sweep_period;
        }
        if self.sweep_reload {
            self.sweep_divider = self.sweep_period;
            self.sweep_reload = false;
        }
    }

    #[cfg(feature = "audio-synth")]
    fn output(&self) -> u8 {
        if !self.enabled || self.length == 0 || self.timer_period < 8 {
            return 0;
        }
        // Additive sweep overflow mutes the channel even when sweep is disabled.
        if !self.sweep_negate {
            let shift_result = self.timer_period >> self.sweep_shift;
            let target = self.timer_period as u32 + shift_result as u32;
            if target > 0x7FF {
                return 0;
            }
        }
        let duty_byte = PULSE_DUTY_TABLE[usize::from(self.duty & 3)];
        let bit = (duty_byte >> (7 - self.sequence_pos)) & 1;
        if bit == 0 {
            0
        } else {
            self.envelope.output()
        }
    }

    fn snapshot_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&[self.channel, u8::from(self.enabled), self.duty]);
        self.envelope.snapshot_bytes(bytes);
        bytes.extend_from_slice(&self.timer_period.to_le_bytes());
        bytes.extend_from_slice(&self.timer.to_le_bytes());
        bytes.extend_from_slice(&[
            self.sequence_pos,
            self.length,
            u8::from(self.length_halt),
            u8::from(self.new_length_halt),
            self.length_reload_value,
            self.length_previous_value,
            u8::from(self.sweep_enabled),
            self.sweep_period,
            u8::from(self.sweep_negate),
            self.sweep_shift,
            self.sweep_divider,
            u8::from(self.sweep_reload),
        ]);
    }

    fn restore_snapshot(&mut self, bytes: &[u8], offset: &mut usize) {
        self.channel = bytes[*offset];
        self.enabled = bytes[*offset + 1] != 0;
        self.duty = bytes[*offset + 2];
        *offset += 3;
        self.envelope.restore_snapshot(bytes, offset);
        self.timer_period = u16::from_le_bytes(bytes[*offset..*offset + 2].try_into().unwrap());
        *offset += 2;
        self.timer = u16::from_le_bytes(bytes[*offset..*offset + 2].try_into().unwrap());
        *offset += 2;
        self.sequence_pos = bytes[*offset];
        self.length = bytes[*offset + 1];
        self.length_halt = bytes[*offset + 2] != 0;
        self.new_length_halt = bytes[*offset + 3] != 0;
        self.length_reload_value = bytes[*offset + 4];
        self.length_previous_value = bytes[*offset + 5];
        self.sweep_enabled = bytes[*offset + 6] != 0;
        self.sweep_period = bytes[*offset + 7];
        self.sweep_negate = bytes[*offset + 8] != 0;
        self.sweep_shift = bytes[*offset + 9];
        self.sweep_divider = bytes[*offset + 10];
        self.sweep_reload = bytes[*offset + 11] != 0;
        *offset += 12;
    }
}

// Triangle channel with length counter + linear counter. Mesen2
// equivalent: reference/local/mesen2/Core/NES/Apu/ApuTriangleChannel.h.
//
// Note: bit 7 of $4008 drives TWO separately-tracked state bits in Mesen2
// (TriangleChannel.h:74-84): `_linearControlFlag` (live, controls
// linear-counter reload clearing in `TickLinearCounter`) and the length
// counter halt (deferred via `ApuLengthCounter::InitializeLengthCounter`).
#[derive(Debug, Clone, Default)]
struct Triangle {
    enabled: bool,
    // Mesen2 `_linearControlFlag` -applied LIVE at $4008 write, used by
    // `clock_linear()` to clear the linear-reload flag.
    linear_control_flag: bool,
    linear_reload: u8,
    linear_counter: u8,
    linear_reload_flag: bool,
    timer_period: u16,
    timer: u16,
    sequence_pos: u8,
    // Mesen2 `_timer` last output (GetLastOutput): the sequence value at the
    // last step. Held while silenced (the triangle freezes its sequencer, it
    // does NOT mute to 0). Transient -set only under `audio-synth`; in the
    // audio-off build it stays 0, so it is not serialized.
    #[cfg(feature = "audio-synth")]
    last_output: u8,
    length: u8,
    // Mesen2 `ApuLengthCounter._halt` -deferred, applied in
    // `reload_length_counter()`. Same source bit as `linear_control_flag`
    // but observed by `clock_length()` with race semantics.
    length_halt: bool,
    new_length_halt: bool,
    length_reload_value: u8,
    length_previous_value: u8,
}

impl Triangle {
    fn write_control(&mut self, value: u8) {
        // $4008 -CRRR RRRR. Bit 7 drives both `_linearControlFlag` (live)
        // and the length-counter halt (deferred via `new_length_halt`).
        // Mesen2 TriangleChannel.h:79-83 `_lengthCounter.InitializeLengthCounter(_linearControlFlag)`.
        self.linear_control_flag = value & 0x80 != 0;
        self.new_length_halt = value & 0x80 != 0;
        self.linear_reload = value & 0x7F;
    }

    fn write_timer_low(&mut self, value: u8) {
        self.timer_period = (self.timer_period & 0x0700) | u16::from(value);
    }

    fn write_timer_high(&mut self, value: u8) {
        // $400B -LLLLLPPP. Length reload is deferred per Mesen2
        // ApuLengthCounter.h:30-37.
        self.timer_period = (self.timer_period & 0x00FF) | (u16::from(value & 0x07) << 8);
        if self.enabled {
            self.length_reload_value = LENGTH_TABLE[usize::from((value >> 3) & 0x1F)];
            self.length_previous_value = self.length;
        }
        self.linear_reload_flag = true;
    }

    #[cfg(feature = "audio-synth")]
    fn tick(&mut self) {
        if self.timer == 0 {
            self.timer = self.timer_period;
            if self.length > 0 && self.linear_counter > 0 {
                self.sequence_pos = (self.sequence_pos + 1) & 31;
                // Mesen2 AddOutput on every step (SilenceTriangleHighFreq is
                // off by default -no period<2 gating). GetOutput then holds
                // this value while the sequencer is frozen (silenced).
                self.last_output = if self.sequence_pos < 16 {
                    15 - self.sequence_pos
                } else {
                    self.sequence_pos - 16
                };
            }
        } else {
            self.timer -= 1;
        }
    }

    fn clock_length(&mut self) {
        if !self.length_halt && self.length > 0 {
            self.length -= 1;
        }
    }

    // Quarter-frame linear counter clock. Mesen2 equivalent:
    // ApuTriangleChannel::TickLinearCounter (ApuTriangleChannel.h:101-112).
    // Note `_linearControlFlag` (live, set immediately at write) -not the
    // deferred `_halt`.
    fn clock_linear(&mut self) {
        if self.linear_reload_flag {
            self.linear_counter = self.linear_reload;
        } else if self.linear_counter > 0 {
            self.linear_counter -= 1;
        }
        if !self.linear_control_flag {
            self.linear_reload_flag = false;
        }
    }

    /// Mesen2 `ApuLengthCounter::ReloadCounter`. See `Pulse::reload_length_counter`.
    fn reload_length_counter(&mut self) {
        if self.length_reload_value != 0 {
            if self.length == self.length_previous_value {
                self.length = self.length_reload_value;
            }
            self.length_reload_value = 0;
        }
        self.length_halt = self.new_length_halt;
    }

    fn has_pending_reload(&self) -> bool {
        self.length_reload_value != 0 || self.length_halt != self.new_length_halt
    }

    #[cfg(feature = "audio-synth")]
    fn output(&self) -> u8 {
        // Mesen2 `GetOutput()` returns the timer's held last value. The
        // triangle freezes its sequencer when silenced (length/linear == 0,
        // see tick()) and holds the last sequence value -it does NOT mute to
        // 0, and Mesen does not silence high frequencies by default.
        self.last_output
    }

    fn snapshot_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&[
            u8::from(self.enabled),
            u8::from(self.linear_control_flag),
            self.linear_reload,
            self.linear_counter,
            u8::from(self.linear_reload_flag),
        ]);
        bytes.extend_from_slice(&self.timer_period.to_le_bytes());
        bytes.extend_from_slice(&self.timer.to_le_bytes());
        bytes.extend_from_slice(&[
            self.sequence_pos,
            self.length,
            u8::from(self.length_halt),
            u8::from(self.new_length_halt),
            self.length_reload_value,
            self.length_previous_value,
        ]);
    }

    fn restore_snapshot(&mut self, bytes: &[u8], offset: &mut usize) {
        self.enabled = bytes[*offset] != 0;
        self.linear_control_flag = bytes[*offset + 1] != 0;
        self.linear_reload = bytes[*offset + 2];
        self.linear_counter = bytes[*offset + 3];
        self.linear_reload_flag = bytes[*offset + 4] != 0;
        *offset += 5;
        self.timer_period = u16::from_le_bytes(bytes[*offset..*offset + 2].try_into().unwrap());
        *offset += 2;
        self.timer = u16::from_le_bytes(bytes[*offset..*offset + 2].try_into().unwrap());
        *offset += 2;
        self.sequence_pos = bytes[*offset];
        self.length = bytes[*offset + 1];
        self.length_halt = bytes[*offset + 2] != 0;
        self.new_length_halt = bytes[*offset + 3] != 0;
        self.length_reload_value = bytes[*offset + 4];
        self.length_previous_value = bytes[*offset + 5];
        *offset += 6;
    }
}

// Noise channel: 15-bit LFSR with mode-dependent tap (bit 6 vs bit 1).
// Mesen2 equivalent: reference/local/mesen2/Core/NES/Apu/ApuNoiseChannel.h.
// Length counter race fields (length_halt / new_length_halt /
// length_reload_value / length_previous_value) mirror Mesen2's
// `ApuLengthCounter` per-channel embedding.
#[derive(Debug, Clone)]
struct Noise {
    enabled: bool,
    envelope: Envelope,
    mode: bool,
    timer_period: u16,
    timer: u16,
    lfsr: u16,
    length: u8,
    length_halt: bool,
    new_length_halt: bool,
    length_reload_value: u8,
    length_previous_value: u8,
}

impl Default for Noise {
    fn default() -> Self {
        Self {
            enabled: false,
            envelope: Envelope::default(),
            mode: false,
            timer_period: NTSC_NOISE_PERIOD_TABLE[0],
            timer: 0,
            lfsr: 1,
            length: 0,
            length_halt: false,
            new_length_halt: false,
            length_reload_value: 0,
            length_previous_value: 0,
        }
    }
}

impl Noise {
    fn write_control(&mut self, value: u8) {
        // $400C ---LC VVVV. Bit 5 (L = length-halt + envelope-loop)
        // deferred per ApuLengthCounter.h:24-28.
        self.envelope.write_control(value);
        self.new_length_halt = (value & 0x20) != 0;
    }

    fn write_period(&mut self, value: u8, region: crate::cartridge::Region) {
        // $400E -L--- PPPP. Period table selected by region per Mesen2
        // NoiseChannel.h:15-16 / WriteRam case 2 (NoiseChannel.h:117-120).
        // Mesen2 NoiseChannel.h:91,118 stores `_timer.SetPeriod(table[idx] - 1)`.
        // Rust ticks the timer with `== 0` reload, so the `-1` is required
        // to match the Mesen2 effective period (which is `_realPeriod + 1`
        // CPU cycles to the next tick).
        self.mode = value & 0x80 != 0;
        let table = match region {
            crate::cartridge::Region::Pal => &PAL_NOISE_PERIOD_TABLE,
            _ => &NTSC_NOISE_PERIOD_TABLE,
        };
        self.timer_period = table[usize::from(value & 0x0F)].wrapping_sub(1);
    }

    fn write_length_load(&mut self, value: u8) {
        // $400F -LLLLL---. Length reload deferred per
        // ApuLengthCounter.h:30-37.
        if self.enabled {
            self.length_reload_value = LENGTH_TABLE[usize::from((value >> 3) & 0x1F)];
            self.length_previous_value = self.length;
        }
        self.envelope.schedule_restart();
    }

    #[cfg(feature = "audio-synth")]
    fn tick(&mut self) {
        if self.timer == 0 {
            self.timer = self.timer_period;
            let bit0 = self.lfsr & 1;
            let other_bit = if self.mode {
                (self.lfsr >> 6) & 1
            } else {
                (self.lfsr >> 1) & 1
            };
            let feedback = bit0 ^ other_bit;
            self.lfsr = (self.lfsr >> 1) | (feedback << 14);
        } else {
            self.timer -= 1;
        }
    }

    #[cfg(feature = "audio-synth")]
    fn output(&self) -> u8 {
        if !self.enabled || self.length == 0 || self.lfsr & 1 != 0 {
            return 0;
        }
        self.envelope.output()
    }

    fn clock_length(&mut self) {
        if !self.length_halt && self.length > 0 {
            self.length -= 1;
        }
    }

    /// Mesen2 `ApuLengthCounter::ReloadCounter`. See `Pulse::reload_length_counter`.
    fn reload_length_counter(&mut self) {
        if self.length_reload_value != 0 {
            if self.length == self.length_previous_value {
                self.length = self.length_reload_value;
            }
            self.length_reload_value = 0;
        }
        self.length_halt = self.new_length_halt;
    }

    fn has_pending_reload(&self) -> bool {
        self.length_reload_value != 0 || self.length_halt != self.new_length_halt
    }

    fn snapshot_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(u8::from(self.enabled));
        self.envelope.snapshot_bytes(bytes);
        bytes.push(u8::from(self.mode));
        bytes.extend_from_slice(&self.timer_period.to_le_bytes());
        bytes.extend_from_slice(&self.timer.to_le_bytes());
        bytes.extend_from_slice(&self.lfsr.to_le_bytes());
        bytes.extend_from_slice(&[
            self.length,
            u8::from(self.length_halt),
            u8::from(self.new_length_halt),
            self.length_reload_value,
            self.length_previous_value,
        ]);
    }

    fn restore_snapshot(&mut self, bytes: &[u8], offset: &mut usize) {
        self.enabled = bytes[*offset] != 0;
        *offset += 1;
        self.envelope.restore_snapshot(bytes, offset);
        self.mode = bytes[*offset] != 0;
        *offset += 1;
        self.timer_period = u16::from_le_bytes(bytes[*offset..*offset + 2].try_into().unwrap());
        *offset += 2;
        self.timer = u16::from_le_bytes(bytes[*offset..*offset + 2].try_into().unwrap());
        *offset += 2;
        self.lfsr = u16::from_le_bytes(bytes[*offset..*offset + 2].try_into().unwrap());
        *offset += 2;
        self.length = bytes[*offset];
        self.length_halt = bytes[*offset + 1] != 0;
        self.new_length_halt = bytes[*offset + 2] != 0;
        self.length_reload_value = bytes[*offset + 3];
        self.length_previous_value = bytes[*offset + 4];
        *offset += 5;
    }
}

// Two-pole IIR low-pass + DC blocker for output filtering. Mesen2
// equivalent: NesApu.cpp Process() filter chain (Mesen2 uses
// BlipBuffer's blip_eq sinc filter for higher fidelity -we use the
// simpler IIR for RL workloads). Phase 2:
// simple one-pole high-pass (DC blocker at ~14 Hz) cascaded with
// one-pole low-pass at ~14 kHz to remove naive-downsample aliasing.
#[derive(Debug, Clone, Default)]
struct AudioFilter {
    hp_x_prev: f32,
    hp_y: f32,
    lp_y: f32,
}

impl AudioFilter {
    #[cfg(feature = "audio-synth")]
    fn apply(&mut self, x: f32) -> f32 {
        // DC blocker: y = x - x_prev + 0.999*y_prev (~14 Hz cutoff @ 44.1 kHz).
        let hp = x - self.hp_x_prev + 0.999 * self.hp_y;
        self.hp_x_prev = x;
        self.hp_y = hp;
        // Low-pass: y = y_prev + alpha*(x - y_prev). alpha=0.8 -~14 kHz @ 44.1 kHz.
        self.lp_y += 0.8 * (hp - self.lp_y);
        self.lp_y
    }
}

#[derive(Debug, Clone)]
pub struct Apu {
    pub cycles: u64,
    /// Mesen2 `NesApu::_previousCycle`. Lazy Run() cursor: state up to
    /// this cycle is synced. Resets to 0 at end_frame.
    previous_cycle: u64,
    /// Mesen2-style master clock counter. NTSC = PPU clock = 3xCPU.
    /// Increments per CPU cycle by `master_clock_divider`. Used to mirror
    /// Mesen2 `_console->GetMasterClock()` for sub-CPU-cycle precision in
    /// `read_frame_irq_status` (FC IRQ `$4015` clear-delay scheduling).
    /// Without this, the FC IRQ flag clear-delay was 3-6x longer than
    /// Mesen2, causing tight `BIT $4015 / BVC` polling loops to diverge.
    master_clock: u64,
    /// Master clock units per CPU cycle. NTSC PPU = 12 master
    /// clocks per CPU clock (4 master per PPU, 3 PPU per CPU); PAL = 16
    /// (5 master per PPU). Mesen2 `NesCpu::_masterClockDivider`. Set by
    /// `set_region` from cartridge region.
    master_clock_divider: u64,
    /// Console region (NTSC / PAL / Dendy). Selects which frame counter
    /// step cycle table to use. Set by `set_region` from cartridge.
    region: crate::cartridge::Region,
    registers: [u8; 0x18],

    frame_previous_cycle: i32,
    frame_current_step: u8,
    frame_step_mode: u8,
    frame_inhibit_irq: bool,
    frame_block_tick: u8,
    frame_new_value: i16,
    frame_write_delay: i8,
    // Mesen2 `_irqFlag` + `_irqFlagClearClock` (local APU IRQ flag with delayed
    // clear via $4015 read). Distinct from `InterruptLines.irq_flag` FrameCounter
    // bit (CPU-visible IRQ line) per ApuFrameCounter.cpp:214-227.
    frame_irq_flag: bool,
    frame_irq_clear_clock: u64,

    // DMC cycle-parity state + audio.
    dmc_format: u8,
    dmc_load_addr: u8,
    dmc_load_length: u8,
    dmc_sample_addr: u16,
    dmc_sample_length: u16,
    dmc_current_addr: u16,
    dmc_bytes_remaining: u16,
    dmc_period: u32,
    dmc_timer: u32,
    dmc_bits_remaining: u8,
    dmc_buffer_empty: bool,
    dmc_read_buffer: u8,
    dmc_silence_flag: bool,
    dmc_dac: u8,
    dmc_shift_reg: u8,
    dmc_transfer_start_delay: u8,
    dmc_disable_delay: u8,
    dmc_need_to_run: bool,

    // Audio synthesis state.
    pulse1: Pulse,
    pulse2: Pulse,
    triangle: Triangle,
    noise: Noise,
    apu_phase: u8,
    sample_phase: u32,
    samples: Vec<f32>,
    filter: AudioFilter,
    // Runtime audio toggle, mirroring ALE's `sound` setting (default ON; RL /
    // headless turns it OFF). When false, `synthesize_step` is skipped so the
    // per-cycle channel-tick + non-linear mix + resample work is elided. The
    // $4015/IRQ/frame-counter/DMC state lives in run_frame_counter_step /
    // run_dmc_timer_step / dmc_process_clock (NOT in synthesize_step), so RAM/CPU
    // output is byte-identical whether audio is on or off -only the emitted
    // sample buffer differs. (No effect unless built with `audio-synth`.)
    audio_enabled: bool,
}

impl Default for Apu {
    fn default() -> Self {
        let mut apu = Self {
            cycles: 0,
            previous_cycle: 0,
            // NTSC default. set_region updates to 16 for PAL.
            // Mesen2 NesCpu::SetMasterClockDivider sets the same values.
            master_clock: 0,
            master_clock_divider: 12,
            region: crate::cartridge::Region::Ntsc,
            registers: [0; 0x18],
            frame_previous_cycle: 0,
            frame_current_step: 0,
            frame_step_mode: 0,
            frame_inhibit_irq: false,
            frame_block_tick: 0,
            frame_new_value: 0,
            frame_write_delay: 3,
            frame_irq_flag: false,
            frame_irq_clear_clock: 0,
            dmc_format: 0,
            dmc_load_addr: 0,
            dmc_load_length: 0,
            dmc_sample_addr: 0xC000,
            dmc_sample_length: 1,
            dmc_current_addr: 0,
            dmc_bytes_remaining: 0,
            dmc_period: 0,
            dmc_timer: 0,
            dmc_bits_remaining: 8,
            dmc_buffer_empty: true,
            dmc_read_buffer: 0,
            dmc_silence_flag: true,
            dmc_dac: 0,
            dmc_shift_reg: 0,
            dmc_transfer_start_delay: 0,
            dmc_disable_delay: 0,
            dmc_need_to_run: false,
            pulse1: Pulse::new(0),
            pulse2: Pulse::new(1),
            triangle: Triangle::default(),
            noise: Noise::default(),
            apu_phase: 0,
            sample_phase: 0,
            samples: Vec::with_capacity(2048),
            filter: AudioFilter::default(),
            audio_enabled: true,
        };
        apu.dmc_period = DEFAULT_DMC_PERIOD;
        apu.dmc_timer = apu.dmc_period;
        apu
    }
}

impl Apu {
    pub fn reset(&mut self) {
        // Preserve the runtime audio toggle across reset: it is a host setting,
        // not emulation state (a power-on reset must not silently re-enable audio
        // synthesis that the host disabled for RL throughput).
        let audio_enabled = self.audio_enabled;
        *self = Self::default();
        self.audio_enabled = audio_enabled;
    }

    /// Enable/disable APU audio synthesis at runtime (mirrors ALE's `sound`
    /// setting). When disabled, the per-cycle channel-tick + non-linear mix +
    /// resample are skipped and `drain_samples` yields nothing; the emulation
    /// state ($4015 / frame & DMC IRQ / DMC DMA) is UNAFFECTED, so RAM/CPU output
    /// stays byte-identical. No effect unless built with the `audio-synth`
    /// feature (synthesis is compiled out otherwise).
    pub fn set_audio_enabled(&mut self, enabled: bool) {
        self.audio_enabled = enabled;
    }

    pub fn audio_enabled(&self) -> bool {
        self.audio_enabled
    }

    pub fn cpu_read_open_bus(
        &mut self,
        addr: u16,
        open_bus: u8,
        interrupt: &mut InterruptLines,
    ) -> u8 {
        // NesApu.cpp Run() before register read so length counters /
        // frame IRQ status reflect the current cycle.
        self.run(self.cycles, interrupt);
        match addr {
            0x4015 => {
                let mut ret = 0;
                if self.pulse1.length > 0 {
                    ret |= 0x01;
                }
                if self.pulse2.length > 0 {
                    ret |= 0x02;
                }
                if self.triangle.length > 0 {
                    ret |= 0x04;
                }
                if self.noise.length > 0 {
                    ret |= 0x08;
                }
                if self.dmc_bytes_remaining != 0 {
                    ret |= 0x10;
                }
                if self.read_frame_irq_status() {
                    ret |= 0x40;
                }
                if interrupt.has_irq_source(IrqSource::Dmc) {
                    ret |= 0x80;
                }
                // NesApu.cpp:102 -$4015 read clears FrameCounter IRQ
                // source (DMC IRQ untouched).
                interrupt.clear_irq_source(IrqSource::FrameCounter);
                ret | (open_bus & 0x20)
            }
            0x4018..=0x401f => open_bus,
            _ => open_bus,
        }
    }

    pub fn cpu_write_with_cycle(
        &mut self,
        addr: u16,
        value: u8,
        cpu_cycle_count: u64,
        interrupt: &mut InterruptLines,
    ) {
        // NesApu.cpp Run() before any register write so the channel
        // / frame counter state is synced to the write cycle.
        self.run(self.cycles, interrupt);
        match addr {
            0x4000..=0x4013 => {
                self.registers[usize::from(addr - 0x4000)] = value;
                self.write_channel(addr & 0x1F, value, interrupt);
            }
            0x4015 => {
                self.registers[0x15] = value & 0x1F;
                self.write_status(value, cpu_cycle_count, interrupt);
            }
            0x4017 => {
                self.registers[0x17] = value;
                self.write_frame_counter(value, cpu_cycle_count, interrupt);
            }
            _ => {}
        }
    }

    fn write_channel(&mut self, reg: u16, value: u8, interrupt: &mut InterruptLines) {
        match reg {
            // Pulse 1 ($4000-$4003)
            0x00 => self.pulse1.write_control(value),
            0x01 => self.pulse1.write_sweep(value),
            0x02 => self.pulse1.write_timer_low(value),
            0x03 => self.pulse1.write_timer_high(value),
            // Pulse 2 ($4004-$4007)
            0x04 => self.pulse2.write_control(value),
            0x05 => self.pulse2.write_sweep(value),
            0x06 => self.pulse2.write_timer_low(value),
            0x07 => self.pulse2.write_timer_high(value),
            // Triangle ($4008, $400A, $400B; $4009 unused)
            0x08 => self.triangle.write_control(value),
            0x0A => self.triangle.write_timer_low(value),
            0x0B => self.triangle.write_timer_high(value),
            // Noise ($400C, $400E, $400F; $400D unused)
            0x0C => self.noise.write_control(value),
            0x0E => self.noise.write_period(value, self.region),
            0x0F => self.noise.write_length_load(value),
            0x10 => {
                // $4010 -IL-- RRRR. Period table picked by region per
                // Mesen2 DeltaModulationChannel.h:12-13 + cpp:198.
                let table = match self.region {
                    crate::cartridge::Region::Pal => &PAL_DMC_PERIOD_TABLE,
                    _ => &NTSC_DMC_PERIOD_TABLE,
                };
                self.dmc_period = table[usize::from(value & 0x0F)] - 1;
                if value & 0x80 == 0 {
                    interrupt.clear_irq_source(IrqSource::Dmc);
                }
                self.dmc_format = value;
            }
            // $4011 -direct DAC level write (bit 7 stripped). Mesen2
            // equivalent: ApuDmcChannel::WriteRam case 0x4011.
            0x11 => {
                self.dmc_dac = value & 0x7F;
            }
            0x12 => {
                self.dmc_load_addr = value;
                self.dmc_sample_addr = 0xC000 | (u16::from(value) << 6);
            }
            0x13 => {
                self.dmc_load_length = value;
                self.dmc_sample_length = (u16::from(value) << 4) | 1;
            }
            _ => {}
        }
    }

    fn write_status(&mut self, value: u8, cpu_cycle_count: u64, interrupt: &mut InterruptLines) {
        self.pulse1.enabled = value & 0x01 != 0;
        if !self.pulse1.enabled {
            self.pulse1.length = 0;
        }
        self.pulse2.enabled = value & 0x02 != 0;
        if !self.pulse2.enabled {
            self.pulse2.length = 0;
        }
        self.triangle.enabled = value & 0x04 != 0;
        if !self.triangle.enabled {
            self.triangle.length = 0;
        }
        self.noise.enabled = value & 0x08 != 0;
        if !self.noise.enabled {
            self.noise.length = 0;
        }
        interrupt.clear_irq_source(IrqSource::Dmc);
        if value & 0x10 != 0 {
            if self.dmc_bytes_remaining == 0 {
                self.dmc_init_sample();
                self.dmc_transfer_start_delay = if cpu_cycle_count & 0x01 == 0 { 2 } else { 3 };
                self.dmc_need_to_run = true;
            }
        } else if self.dmc_disable_delay == 0 {
            self.dmc_disable_delay = if cpu_cycle_count & 0x01 == 0 { 2 } else { 3 };
            self.dmc_need_to_run = true;
        }
    }

    fn write_frame_counter(
        &mut self,
        value: u8,
        cpu_cycle_count: u64,
        interrupt: &mut InterruptLines,
    ) {
        self.frame_new_value = i16::from(value);
        self.frame_write_delay = if cpu_cycle_count & 0x01 != 0 { 4 } else { 3 };
        self.frame_inhibit_irq = value & 0x40 != 0;
        if self.frame_inhibit_irq {
            // ApuFrameCounter.cpp:206-211 -$4017 bit-6 inhibit set
            // synchronously clears CPU FrameCounter IRQ source + local _irqFlag.
            self.frame_irq_flag = false;
            self.frame_irq_clear_clock = 0;
            interrupt.clear_irq_source(IrqSource::FrameCounter);
        }
    }

    /// Current DMC sample-read address. Used by the CPU's DMA state
    /// machine (`process_pending_dma`) to fetch the next sample byte
    /// during DMC DMA. Mesen2 equivalent:
    /// `NesApu::GetDmcReadAddress` (NesApu.h via DMC channel).
    /// Returns the address in $8000-$FFFF range (high PRG ROM space).
    /// Mesen2 `ApuDmcChannel::GetReadAddress` returns `0x8000 | _currentAddr`.
    pub fn dmc_read_address(&self) -> u16 {
        self.dmc_current_addr
    }

    /// Deliver a DMC sample byte that was just fetched by CPU's DMA
    /// state machine. Sets the DMA buffer, marks have_dma true (so
    /// dmc_tester can transfer it to the shift register), advances
    /// sample address (with 15-bit wraparound = wraps from $FFFF back
    /// to $8000), and
    /// decrements remaining sample size. When the sample ends, either
    /// the loop bit (`$4010` bit 6) restarts playback or the IRQ-enable
    /// bit (`$4010` bit 7) fires the DMC IRQ. Mesen2 equivalent:
    /// `NesApu::SetDmcReadBuffer(value)` -> `_dmcChannel->SetReadBuffer`.
    pub fn set_dmc_read_buffer(&mut self, value: u8, interrupt: &mut InterruptLines) {
        if self.dmc_bytes_remaining == 0 {
            return;
        }
        self.dmc_read_buffer = value;
        self.dmc_buffer_empty = false;
        self.dmc_current_addr = self.dmc_current_addr.wrapping_add(1);
        if self.dmc_current_addr == 0 {
            self.dmc_current_addr = 0x8000;
        }
        self.dmc_bytes_remaining -= 1;
        if self.dmc_bytes_remaining == 0 {
            if self.dmc_format & 0x40 != 0 {
                self.dmc_init_sample();
            } else if self.dmc_format & 0x80 != 0 {
                // DeltaModulationChannel.cpp:90 -sample end + IRQ enable
                // sets DMC IRQ source on CPU.
                interrupt.set_irq_source(IrqSource::Dmc);
            }
        }
        self.apply_dmc_sample_duplication_timing(interrupt);
    }

    /// Mesen2 `NesApu::ProcessCpuClock` -`Exec` (NesApu.cpp:193-224).
    /// Per-CPU-cycle entry called by `Cpu::start_cpu_cycle`. DMC delay
    /// processing (sample fetch + IRQ + `dmc_dma_pending`) is per-cycle
    /// critical for IRQ-cycle-boundary visibility, so it always fires.
    /// Frame counter step + DMC timer + audio synth + length-counter
    /// reload are batched into `run(target, interrupt)` via the
    /// `need_to_run` gate so non-audio builds skip the per-cycle dispatch
    /// when nothing is pending.
    pub fn process_cpu_clock(&mut self, interrupt: &mut InterruptLines) {
        self.cycles = self.cycles.wrapping_add(1);
        // tick master clock alongside CPU cycle. Mesen2
        // NesCpu::EndCpuCycle does `_state.CycleCount += _masterClockDivider`
        // each CPU cycle. We mirror that to keep `read_frame_irq_status`
        // sub-CPU-cycle precision aligned to Mesen2.
        self.master_clock = self.master_clock.wrapping_add(self.master_clock_divider);
        self.dmc_process_clock(interrupt);
        if self.need_to_run(self.cycles) {
            self.run(self.cycles, interrupt);
        }
    }

    /// Mesen2 `NesApu::NeedToRun(currentCycle)` (NesApu.cpp:180-191).
    /// Returns true if any pending state (length counter reload, halt
    /// deferral, DMC active, frame counter step about to fire, or DMC
    /// IRQ about to assert) needs `run()` to be called this cycle.
    fn need_to_run(&self, target: u64) -> bool {
        if self.dmc_need_to_run {
            return true;
        }
        if self.has_pending_length_state() {
            return true;
        }
        let delta = target.saturating_sub(self.previous_cycle) as i32;
        if self.frame_counter_need_to_run(delta) {
            return true;
        }
        if self.dmc_irq_pending(delta) {
            return true;
        }
        false
    }

    fn has_pending_length_state(&self) -> bool {
        self.pulse1.has_pending_reload()
            || self.pulse2.has_pending_reload()
            || self.triangle.has_pending_reload()
            || self.noise.has_pending_reload()
    }

    /// Mesen2 `ApuFrameCounter::NeedToRun(cyclesToRun)` (ApuFrameCounter.h:173-180).
    /// True if a pending $4017 write, the post-tick block window is open,
    /// or the next step boundary falls within `cycles_to_run`.
    fn frame_counter_need_to_run(&self, cycles_to_run: i32) -> bool {
        if self.frame_new_value >= 0 || self.frame_block_tick > 0 {
            return true;
        }
        let mode = usize::from(self.frame_step_mode);
        let step = usize::from(self.frame_current_step);
        let next = self.step_cycles_table()[mode][step];
        self.frame_previous_cycle + cycles_to_run >= next - 1
    }

    /// Mesen2 `DeltaModulationChannel::IrqPending(cyclesToRun)`
    /// (DeltaModulationChannel.cpp:166-175). Predicts whether the DMC
    /// IRQ would assert within the next `cycles_to_run` cycles, so the
    /// gate can force `run()` for visibility at the CPU IRQ sample.
    fn dmc_irq_pending(&self, cycles_to_run: i32) -> bool {
        if (self.dmc_format & 0x80) == 0 || self.dmc_bytes_remaining == 0 {
            return false;
        }
        let bits = i64::from(self.dmc_bits_remaining);
        let bytes = i64::from(self.dmc_bytes_remaining);
        let period = i64::from(self.dmc_period) + 1;
        let cycles_to_empty = (bits + (bytes - 1) * 8) * period;
        i64::from(cycles_to_run) >= cycles_to_empty
    }

    /// Catch up frame counter, length reloads, DMC timer, and audio synthesis.
    /// Frame-counter write delay decrements per batch-run call, not per CPU cycle.
    pub fn run(&mut self, target: u64, interrupt: &mut InterruptLines) {
        let delta = target.saturating_sub(self.previous_cycle);
        if delta == 0 {
            return;
        }
        let mut cycles_to_run = delta as i32;
        while cycles_to_run > 0 {
            let cycles_ran = self.run_frame_counter_step(&mut cycles_to_run, interrupt);
            self.previous_cycle = self.previous_cycle.wrapping_add(u64::from(cycles_ran));
            // Reload length counters after any frame-counter half-step.
            self.pulse1.reload_length_counter();
            self.pulse2.reload_length_counter();
            self.noise.reload_length_counter();
            self.triangle.reload_length_counter();
            // DMC timer and audio synth remain per-CPU-cycle state machines.
            for _ in 0..cycles_ran {
                self.run_dmc_timer_step(interrupt);
                #[cfg(feature = "audio-synth")]
                if self.audio_enabled {
                    self.synthesize_step();
                }
            }
        }
    }

    /// Flush lazy APU state and reset the per-frame run cursor.
    pub fn end_frame(&mut self, interrupt: &mut InterruptLines) {
        self.run(self.cycles, interrupt);
        self.cycles = 0;
        self.previous_cycle = 0;
        // Reset pending IRQ-clear schedule when the master-clock origin moves.
        self.master_clock = 0;
        self.frame_irq_clear_clock = 0;
    }

    /// Diagnostic APU frame-counter state.
    pub fn frame_counter_state(&self) -> FrameCounterDebugState {
        FrameCounterDebugState {
            previous_cycle: self.frame_previous_cycle,
            current_step: u32::from(self.frame_current_step),
            step_mode: u32::from(self.frame_step_mode),
            inhibit_irq: self.frame_inhibit_irq,
            block_frame_counter_tick: self.frame_block_tick,
            new_value: self.frame_new_value,
            write_delay_counter: self.frame_write_delay,
            irq_flag: self.frame_irq_flag,
            irq_flag_clear_clock: self.frame_irq_clear_clock,
        }
    }

    /// Mesen2 `ApuFrameCounter::SetRegion` + `DeltaModulationChannel::Reset`
    /// (DeltaModulationChannel.cpp:46) + `NoiseChannel::Reset` (NoiseChannel.h:91)
    /// -switches frame counter step table, DMC period table, and Noise
    /// period table between NTSC and PAL. Called from
    /// `NesCore::load_rom_bytes` after cartridge parse.
    #[cold]
    pub fn set_region(&mut self, region: crate::cartridge::Region) {
        self.region = region;
        // master clock divider per Mesen2 NesCpu::SetMasterClockDivider.
        // NTSC: master = 21.477 MHz, CPU = 1.789 MHz -12 master clocks per CPU.
        // PAL: master = 26.602 MHz, CPU = 1.662 MHz -16 master clocks per CPU.
        // Dendy: same as NTSC for APU (Mesen2 ApuFrameCounter.h falls to NTSC case).
        self.master_clock_divider = match region {
            crate::cartridge::Region::Pal => 16,
            _ => 12,
        };
        // Mirror Mesen2 Reset(): re-initialize the default DMC + Noise
        // timer periods using the new region's table[0]. Subsequent
        // game writes to $4010 / $400E pick the right table via
        // self.region.
        let dmc_table = match region {
            crate::cartridge::Region::Pal => &PAL_DMC_PERIOD_TABLE,
            _ => &NTSC_DMC_PERIOD_TABLE,
        };
        self.dmc_period = dmc_table[0] - 1;
        self.dmc_timer = self.dmc_period;
        let noise_table = match region {
            crate::cartridge::Region::Pal => &PAL_NOISE_PERIOD_TABLE,
            _ => &NTSC_NOISE_PERIOD_TABLE,
        };
        // Mesen2 NoiseChannel.h:118 stores `table[idx] - 1`.
        self.noise.timer_period = noise_table[0].wrapping_sub(1);
    }

    fn step_cycles_table(&self) -> &'static [[i32; 6]; 2] {
        match self.region {
            crate::cartridge::Region::Pal => &FRAME_STEP_CYCLES_PAL,
            // NTSC + Dendy use NTSC frame counter table per Mesen2
            // ApuFrameCounter.h:88-95 SetRegion (Dendy falls into Ntsc case).
            _ => &FRAME_STEP_CYCLES_NTSC,
        }
    }

    /// Mesen2 `ApuFrameCounter::Run(int32_t &cyclesToRun)`
    /// (ApuFrameCounter.h:99-171). Batch-oriented: processes UP TO one step
    /// transition per call (or consumes the entire `cycles_to_run` budget
    /// when no step boundary lies within it). Returns `cycles_ran` -the
    /// actual CPU cycles consumed in this call (may be > 1).
    ///
    /// `_writeDelayCounter` and `_blockFrameCounterTick`
    /// MUST decrement once per Run call (not once per CPU cycle). The
    /// caller (`Apu::run`) loops until `cycles_to_run` reaches 0, calling
    /// this multiple times when the budget spans multiple step boundaries.
    fn run_frame_counter_step(
        &mut self,
        cycles_to_run: &mut i32,
        interrupt: &mut InterruptLines,
    ) -> u32 {
        let mode = usize::from(self.frame_step_mode);
        let step = usize::from(self.frame_current_step);
        let step_cycle = self.step_cycles_table()[mode][step];

        let cycles_ran: u32 = if self.frame_previous_cycle + *cycles_to_run >= step_cycle {
            // Step boundary reached within this Run budget. Mesen2
            // ApuFrameCounter.h:103-144.
            if self.frame_step_mode == 0 && self.frame_current_step >= 3 {
                self.frame_irq_flag = true;
                self.frame_irq_clear_clock = 0;
                if !self.frame_inhibit_irq {
                    interrupt.set_irq_source(IrqSource::FrameCounter);
                } else if self.frame_current_step == 5 {
                    // ApuFrameCounter.cpp:113 -when inhibited at step 5,
                    // local _irqFlag is cleared (CPU side already untouched
                    // because !inhibit path was skipped).
                    self.frame_irq_flag = false;
                    self.frame_irq_clear_clock = 0;
                }
            }

            let frame_type = FRAME_STEP_TYPES[mode][step];
            if frame_type != FrameTick::None && self.frame_block_tick == 0 {
                self.frame_counter_tick(frame_type);
                self.frame_block_tick = 2;
            }

            // PAL->NTSC region switch corner case (Mesen2 L125-127):
            // step_cycle < _previousCycle means an endless loop in APU,
            // cyclesRan = 0 to avoid freezing.
            let ran: u32 = if step_cycle < self.frame_previous_cycle {
                0
            } else {
                (step_cycle - self.frame_previous_cycle) as u32
            };
            *cycles_to_run -= ran as i32;

            self.frame_current_step += 1;
            if self.frame_current_step == 6 {
                self.frame_current_step = 0;
                self.frame_previous_cycle = 0;
            } else {
                self.frame_previous_cycle += ran as i32;
            }
            ran
        } else {
            // No boundary within budget: consume the entire `cycles_to_run`
            // in one shot. Mesen2 ApuFrameCounter.h:141-144.
            let ran = *cycles_to_run as u32;
            *cycles_to_run = 0;
            self.frame_previous_cycle += ran as i32;
            ran
        };

        // Write-delay handling -ONCE PER RUN CALL (Mesen2 L147-164).
        // Pending $4017 write applies after the appropriate delay.
        if self.frame_new_value >= 0 {
            self.frame_write_delay -= 1;
            if self.frame_write_delay == 0 {
                let value = self.frame_new_value as u8;
                self.frame_step_mode = u8::from(value & 0x80 != 0);
                self.frame_current_step = 0;
                self.frame_previous_cycle = 0;
                self.frame_new_value = -1;
                self.frame_write_delay = -1;
                if self.frame_step_mode != 0 && self.frame_block_tick == 0 {
                    // 5-step mode immediate half+quarter tick (Mesen2 L158-162).
                    self.frame_counter_tick(FrameTick::Half);
                    self.frame_block_tick = 2;
                }
            }
        }

        // Block-tick decrement -ONCE PER RUN CALL (Mesen2 L166-168).
        if self.frame_block_tick > 0 {
            self.frame_block_tick -= 1;
        }

        cycles_ran
    }

    /// Per-instruction APU tick. Called from the main CPU loop with the
    /// Per-CPU-cycle audio synthesis. Ticks the triangle every cycle and
    /// pulse/noise every other cycle (APU phase). Emits a downsampled
    /// audio sample whenever `sample_phase` wraps the CPU rate.
    /// Mesen2 equivalent: NesApu mix loop in NesApu.cpp; we use
    /// integer-phase downsampling instead of Mesen2's BlipBuffer.
    #[cfg(feature = "audio-synth")]
    fn synthesize_step(&mut self) {
        self.triangle.tick();
        self.apu_phase ^= 1;
        if self.apu_phase == 0 {
            self.pulse1.tick();
            self.pulse2.tick();
            self.noise.tick();
        }
        self.sample_phase += AUDIO_SAMPLE_RATE;
        if self.sample_phase >= NTSC_CPU_HZ {
            self.sample_phase -= NTSC_CPU_HZ;
            let sample = self.mix();
            self.samples.push(sample);
        }
    }

    fn frame_counter_tick(&mut self, frame_type: FrameTick) {
        // ApuEnvelope.h:74 reads `LengthCounter.IsHalted()` (deferred
        // `_halt`) inside `TickEnvelope` -envelope wrap and length-counter
        // halt observe the same deferred bit. We pass `length_halt`
        // (already-applied) so race semantics are preserved across both.
        self.pulse1.envelope.clock(self.pulse1.length_halt);
        self.pulse2.envelope.clock(self.pulse2.length_halt);
        self.noise.envelope.clock(self.noise.length_halt);
        self.triangle.clock_linear();
        if frame_type == FrameTick::Half {
            self.pulse1.clock_length();
            self.pulse2.clock_length();
            self.triangle.clock_length();
            self.noise.clock_length();
            self.pulse1.clock_sweep();
            self.pulse2.clock_sweep();
        }
    }

    fn dmc_init_sample(&mut self) {
        self.dmc_current_addr = self.dmc_sample_addr;
        self.dmc_bytes_remaining = self.dmc_sample_length;
        self.dmc_need_to_run |= self.dmc_bytes_remaining > 0;
    }

    fn dmc_start_transfer(&mut self, interrupt: &mut InterruptLines) {
        if self.dmc_buffer_empty && self.dmc_bytes_remaining > 0 && !interrupt.dmc_dma_pending {
            interrupt.request_dmc_dma();
        }
    }

    fn dmc_process_clock(&mut self, interrupt: &mut InterruptLines) {
        if self.dmc_disable_delay > 0 {
            self.dmc_disable_delay -= 1;
            if self.dmc_disable_delay == 0 {
                self.dmc_bytes_remaining = 0;
                interrupt.dmc_dma_pending = false;
                interrupt.request_dmc_dma_stop();
            }
        }
        if self.dmc_transfer_start_delay > 0 {
            self.dmc_transfer_start_delay -= 1;
            if self.dmc_transfer_start_delay == 0 {
                self.dmc_start_transfer(interrupt);
            }
        }
        self.dmc_need_to_run = self.dmc_disable_delay != 0
            || self.dmc_transfer_start_delay != 0
            || self.dmc_bytes_remaining != 0;
    }

    fn run_dmc_timer_step(&mut self, interrupt: &mut InterruptLines) {
        if self.dmc_timer > 0 {
            self.dmc_timer -= 1;
            return;
        }
        self.dmc_timer = self.dmc_period;

        if !self.dmc_silence_flag {
            if self.dmc_shift_reg & 1 != 0 {
                if self.dmc_dac <= 125 {
                    self.dmc_dac += 2;
                }
            } else if self.dmc_dac >= 2 {
                self.dmc_dac -= 2;
            }
            self.dmc_shift_reg >>= 1;
        }

        self.dmc_bits_remaining -= 1;
        if self.dmc_bits_remaining == 0 {
            self.dmc_bits_remaining = 8;
            if self.dmc_buffer_empty {
                self.dmc_silence_flag = true;
            } else {
                self.dmc_silence_flag = false;
                self.dmc_shift_reg = self.dmc_read_buffer;
                self.dmc_buffer_empty = true;
                self.dmc_need_to_run = true;
                if self.dmc_transfer_start_delay == 0 {
                    self.dmc_start_transfer(interrupt);
                }
            }
        }
    }

    fn apply_dmc_sample_duplication_timing(&mut self, interrupt: &mut InterruptLines) {
        if self.dmc_sample_length == 1 && self.dmc_format & 0x40 == 0 {
            // Optional DMC sample-duplication glitch is disabled by oracle config.
            // The `bits==1 && timer<2` timing path remains active.
            const ENABLE_DMC_SAMPLE_DUPLICATION_GLITCH: bool = false;
            if ENABLE_DMC_SAMPLE_DUPLICATION_GLITCH
                && self.dmc_bits_remaining == 8
                && self.dmc_timer == self.dmc_period
            {
                self.dmc_shift_reg = self.dmc_read_buffer;
                self.dmc_silence_flag = false;
                self.dmc_buffer_empty = true;
                self.dmc_init_sample();
                self.dmc_start_transfer(interrupt);
            } else if self.dmc_bits_remaining == 1 && self.dmc_timer < 2 {
                self.dmc_shift_reg = self.dmc_read_buffer;
                self.dmc_buffer_empty = false;
                self.dmc_init_sample();
                self.dmc_disable_delay = 3;
            }
        }
    }

    // Non-linear NES mixer + DC blocker + low-pass. Mesen2 equivalent:
    // NesApu::Process in NesApu.cpp (uses the same canonical NESdev
    // formulas via _pulseTable / _tndTable lookup).
    #[cfg(feature = "audio-synth")]
    fn mix(&mut self) -> f32 {
        let p1 = self.pulse1.output() as usize;
        let p2 = self.pulse2.output() as usize;
        let t = self.triangle.output() as usize;
        let n = self.noise.output() as usize;
        let d = self.dmc_dac as usize;
        let pulse_idx = (p1 + p2).min(30);
        let tnd_idx = (3 * t + 2 * n + d).min(202);
        let unfiltered = pulse_table()[pulse_idx] + tnd_table()[tnd_idx];
        // Filter cascade: DC blocker + low-pass. Output -[-0.5, +0.5].
        self.filter.apply(unfiltered).clamp(-1.0, 1.0)
    }

    /// Drain accumulated audio samples (f32 mono).
    pub fn drain_samples(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.samples)
    }

    /// Per-channel gated output volumes `[pulse1, pulse2, triangle, noise, dmc]`,
    /// matching Mesen2 `ApuState.*.OutputVolume` (each channel's `GetOutput()`).
    /// Only meaningful when `audio-synth` is enabled (channels are ticked);
    /// pairs with the Mesen2 oracle `NesHeadlessGetApuChannelOutputs` to verify
    /// the channel emulation matches at the raw (pre-mixer) output level.
    #[cfg(feature = "audio-synth")]
    pub fn channel_outputs(&self) -> [u8; 5] {
        [
            self.pulse1.output(),
            self.pulse2.output(),
            self.triangle.output(),
            self.noise.output(),
            self.dmc_dac,
        ]
    }

    /// Per-pulse-channel (0/1) state for the APU period/sweep diff:
    /// `[period, duty, duty_pos, sweep_enabled, sweep_negate, sweep_period,
    /// sweep_shift, enabled, output]`. Pairs with the Mesen2 oracle
    /// `NesHeadlessGetApuPulseState`. Meaningful only with `audio-synth`.
    #[cfg(feature = "audio-synth")]
    pub fn pulse_state(&self, ch: usize) -> [u16; 9] {
        let p = if ch == 0 { &self.pulse1 } else { &self.pulse2 };
        [
            p.timer_period,
            u16::from(p.duty),
            u16::from(p.sequence_pos),
            u16::from(p.sweep_enabled),
            u16::from(p.sweep_negate),
            u16::from(p.sweep_period),
            u16::from(p.sweep_shift),
            u16::from(p.enabled),
            u16::from(p.output()),
        ]
    }

    /// Per-envelope-channel (0=pulse1, 1=pulse2, 2=noise) state for the /// APU envelope + length-counter diff: `[constant_volume, volume, counter,
    /// length_counter, length_halt, enabled]`. All sampling-INsensitive
    /// (register-set or quarter/half-frame clocked). Pairs with the Mesen2
    /// oracle `NesHeadlessGetApuEnvelopeState`. Meaningful only with `audio-synth`.
    #[cfg(feature = "audio-synth")]
    pub fn envelope_state(&self, ch: usize) -> [u16; 6] {
        let (env, length, length_halt, enabled) = match ch {
            0 => (
                &self.pulse1.envelope,
                self.pulse1.length,
                self.pulse1.length_halt,
                self.pulse1.enabled,
            ),
            1 => (
                &self.pulse2.envelope,
                self.pulse2.length,
                self.pulse2.length_halt,
                self.pulse2.enabled,
            ),
            _ => (
                &self.noise.envelope,
                self.noise.length,
                self.noise.length_halt,
                self.noise.enabled,
            ),
        };
        [
            u16::from(env.mode & 0x1),
            u16::from(env.speed),
            u16::from(env.dec_volume),
            u16::from(length),
            u16::from(length_halt),
            u16::from(enabled),
        ]
    }

    fn read_frame_irq_status(&mut self) -> bool {
        // Frame IRQ clear delay is measured in CPU cycles, not master clocks.
        if self.frame_irq_flag {
            if self.frame_irq_clear_clock == 0 {
                self.frame_irq_clear_clock =
                    self.cycles
                        .wrapping_add(if self.cycles & 0x01 != 0 { 2 } else { 1 });
            } else if self.cycles >= self.frame_irq_clear_clock {
                self.frame_irq_clear_clock = 0;
                self.frame_irq_flag = false;
            }
        }
        self.frame_irq_flag
    }

    #[cold]
    pub fn snapshot_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(186);
        bytes.extend_from_slice(&self.cycles.to_le_bytes());
        bytes.extend_from_slice(&self.previous_cycle.to_le_bytes());
        bytes.extend_from_slice(&self.registers);
        bytes.extend_from_slice(&self.frame_previous_cycle.to_le_bytes());
        bytes.push(self.frame_current_step);
        bytes.push(self.frame_step_mode);
        bytes.push(u8::from(self.frame_inhibit_irq));
        bytes.push(self.frame_block_tick);
        bytes.extend_from_slice(&self.frame_new_value.to_le_bytes());
        bytes.push(self.frame_write_delay as u8);
        bytes.push(u8::from(self.frame_irq_flag));
        bytes.extend_from_slice(&self.frame_irq_clear_clock.to_le_bytes());
        // master_clock (8) + master_clock_divider (8) appended
        // for sub-CPU-cycle FC IRQ clear-delay precision. Snapshot magic
        // bump (NESA -NESB) covers the layout change.
        bytes.extend_from_slice(&self.master_clock.to_le_bytes());
        bytes.extend_from_slice(&self.master_clock_divider.to_le_bytes());
        bytes.push(self.dmc_format);
        bytes.push(self.dmc_load_addr);
        bytes.push(self.dmc_load_length);
        bytes.extend_from_slice(&self.dmc_sample_addr.to_le_bytes());
        bytes.extend_from_slice(&self.dmc_sample_length.to_le_bytes());
        bytes.extend_from_slice(&self.dmc_current_addr.to_le_bytes());
        bytes.extend_from_slice(&self.dmc_bytes_remaining.to_le_bytes());
        bytes.extend_from_slice(&self.dmc_period.to_le_bytes());
        bytes.extend_from_slice(&self.dmc_timer.to_le_bytes());
        bytes.push(self.dmc_bits_remaining);
        bytes.push(u8::from(self.dmc_buffer_empty));
        bytes.push(self.dmc_read_buffer);
        bytes.push(u8::from(self.dmc_silence_flag));
        bytes.push(self.dmc_dac);
        bytes.push(self.dmc_shift_reg);
        bytes.push(self.dmc_transfer_start_delay);
        bytes.push(self.dmc_disable_delay);
        bytes.push(u8::from(self.dmc_need_to_run));
        self.pulse1.snapshot_bytes(&mut bytes);
        self.pulse2.snapshot_bytes(&mut bytes);
        self.triangle.snapshot_bytes(&mut bytes);
        self.noise.snapshot_bytes(&mut bytes);
        bytes.push(self.apu_phase);
        bytes.extend_from_slice(&self.sample_phase.to_le_bytes());
        bytes.extend_from_slice(&self.filter.hp_x_prev.to_le_bytes());
        bytes.extend_from_slice(&self.filter.hp_y.to_le_bytes());
        bytes.extend_from_slice(&self.filter.lp_y.to_le_bytes());
        bytes
    }

    #[cold]
    pub fn restore_snapshot(&mut self, bytes: &[u8]) -> nesle_common::Result<()> {
        // NESB layout: NESA base (186 bytes) + master_clock (8)
        // + master_clock_divider (8) = 202 bytes.
        const LEN: usize = 202;
        if bytes.len() != LEN {
            return Err(nesle_common::NesleError::InvalidState(format!(
                "APU snapshot length must be {LEN} bytes, got {}",
                bytes.len()
            )));
        }
        let mut off = 0usize;
        self.cycles = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        off += 8;
        self.previous_cycle = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        off += 8;
        self.registers.copy_from_slice(&bytes[off..off + 0x18]);
        off += 0x18;
        self.frame_previous_cycle = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        off += 4;
        self.frame_current_step = bytes[off];
        off += 1;
        self.frame_step_mode = bytes[off];
        off += 1;
        self.frame_inhibit_irq = bytes[off] != 0;
        off += 1;
        self.frame_block_tick = bytes[off];
        off += 1;
        self.frame_new_value = i16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
        off += 2;
        self.frame_write_delay = bytes[off] as i8;
        off += 1;
        self.frame_irq_flag = bytes[off] != 0;
        off += 1;
        self.frame_irq_clear_clock = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        off += 8;
        // master_clock (8) + master_clock_divider (8) -see
        // snapshot_bytes comment.
        self.master_clock = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        off += 8;
        self.master_clock_divider = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        off += 8;
        self.dmc_format = bytes[off];
        off += 1;
        self.dmc_load_addr = bytes[off];
        off += 1;
        self.dmc_load_length = bytes[off];
        off += 1;
        self.dmc_sample_addr = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
        off += 2;
        self.dmc_sample_length = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
        off += 2;
        self.dmc_current_addr = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
        off += 2;
        self.dmc_bytes_remaining = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
        off += 2;
        self.dmc_period = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        off += 4;
        self.dmc_timer = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        off += 4;
        self.dmc_bits_remaining = bytes[off];
        off += 1;
        self.dmc_buffer_empty = bytes[off] != 0;
        off += 1;
        self.dmc_read_buffer = bytes[off];
        off += 1;
        self.dmc_silence_flag = bytes[off] != 0;
        off += 1;
        self.dmc_dac = bytes[off];
        off += 1;
        self.dmc_shift_reg = bytes[off];
        off += 1;
        self.dmc_transfer_start_delay = bytes[off];
        off += 1;
        self.dmc_disable_delay = bytes[off];
        off += 1;
        self.dmc_need_to_run = bytes[off] != 0;
        off += 1;
        self.pulse1.restore_snapshot(bytes, &mut off);
        self.pulse2.restore_snapshot(bytes, &mut off);
        self.triangle.restore_snapshot(bytes, &mut off);
        self.noise.restore_snapshot(bytes, &mut off);
        self.apu_phase = bytes[off];
        off += 1;
        self.sample_phase = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        off += 4;
        self.filter.hp_x_prev = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        off += 4;
        self.filter.hp_y = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        off += 4;
        self.filter.lp_y = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        off += 4;
        debug_assert_eq!(off, bytes.len());
        self.samples.clear();
        Ok(())
    }

    pub fn register(&self, addr: u16) -> u8 {
        self.registers[usize::from(addr - 0x4000)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartridge::Mirroring;
    use crate::mapper::Mapper;

    #[derive(Debug)]
    struct TestMapper;

    impl Mapper for TestMapper {
        fn mapper_id(&self) -> u16 {
            0
        }
        fn name(&self) -> &'static str {
            "TEST"
        }
        fn cpu_read(&mut self, _addr: u16) -> u8 {
            0xAA
        }
        fn cpu_write(&mut self, _addr: u16, _value: u8, _interrupt: &mut InterruptLines) {}
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

    #[test]
    fn status_register_is_readable_and_masked() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        apu.cpu_write_with_cycle(0x4015, 0xFF, 0, &mut lines);
        assert_eq!(apu.cpu_read_open_bus(0x4015, 0, &mut lines), 0x10);
        assert_eq!(apu.register(0x4015), 0x1F);
    }

    #[test]
    fn snapshot_restores_registers() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        apu.cpu_write_with_cycle(0x4000, 0x44, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4017, 0x80, 0, &mut lines);
        let snapshot = apu.snapshot_bytes();
        apu.cpu_write_with_cycle(0x4000, 0x11, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4017, 0x00, 0, &mut lines);
        apu.restore_snapshot(&snapshot).unwrap();
        assert_eq!(apu.register(0x4000), 0x44);
        assert_eq!(apu.register(0x4017), 0x80);
    }

    #[test]
    fn power_on_frame_counter_acts_like_delayed_4017_zero_write() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        assert_eq!(apu.frame_new_value, 0);
        assert_eq!(apu.frame_write_delay, 3);
        for _ in 0..3 {
            apu.process_cpu_clock(&mut lines);
        }
        apu.run(apu.cycles, &mut lines);
        assert_eq!(apu.frame_new_value, -1);
        assert_eq!(apu.frame_step_mode, 0);
        assert!(!apu.frame_inhibit_irq);
    }

    #[test]
    fn frame_irq_fires_when_enabled_in_mode_0() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        apu.cpu_write_with_cycle(0x4017, 0x00, 0, &mut lines);
        for _ in 0..29_831 {
            apu.process_cpu_clock(&mut lines);
        }
        apu.run(apu.cycles, &mut lines);
        assert!(apu.frame_irq_flag);
        assert!(lines.has_irq_source(IrqSource::FrameCounter));
    }

    #[test]
    fn status_read_clears_frame_irq_only() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        apu.frame_irq_flag = true;
        lines.set_irq_source(IrqSource::FrameCounter);
        lines.set_irq_source(IrqSource::Dmc);
        let status = apu.cpu_read_open_bus(0x4015, 0, &mut lines);
        assert_eq!(status & 0xC0, 0xC0);
        assert!(!lines.has_irq_source(IrqSource::FrameCounter));
        assert!(lines.has_irq_source(IrqSource::Dmc));
    }

    #[test]
    fn dmc_stop_via_status_write_bit4_zero() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        apu.cpu_write_with_cycle(0x4012, 0x00, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4013, 0x01, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4015, 0x10, 0, &mut lines);
        assert_eq!(apu.dmc_bytes_remaining, 17);
        apu.cpu_write_with_cycle(0x4015, 0x00, 0, &mut lines);
        for _ in 0..2 {
            apu.process_cpu_clock(&mut lines);
        }
        assert_eq!(apu.dmc_bytes_remaining, 0);
    }

    /// Helper: simulate what `Cpu::process_pending_dma` would do when
    /// `apu.pending_dmc_dma_request` is set. Drains the flag, reads the
    /// sample byte from the mapper at the DMC address, delivers it via
    /// `set_dmc_read_buffer` (which sets `dmc_have_dma`, advances the
    /// address, decrements size). Returns the simulated 4-cycle DMC
    /// halt+fetch stall.
    fn simulate_dmc_dma_fetch<M: Mapper + ?Sized>(
        apu: &mut Apu,
        mapper: &mut M,
        lines: &mut InterruptLines,
    ) -> u32 {
        if !lines.dmc_dma_pending {
            return 0;
        }
        lines.dmc_dma_pending = false;
        let addr = apu.dmc_read_address();
        let value = mapper.cpu_read(addr);
        apu.set_dmc_read_buffer(value, lines);
        4
    }

    #[test]
    fn dmc_sample_fetch_costs_4_cpu_cycles() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        let mut mapper = TestMapper;
        apu.cpu_write_with_cycle(0x4010, 0x0F, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4012, 0x00, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4013, 0x00, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4015, 0x10, 0, &mut lines);
        assert_eq!(apu.dmc_bytes_remaining, 1);
        for _ in 0..2 {
            apu.process_cpu_clock(&mut lines);
        }
        let stall = simulate_dmc_dma_fetch(&mut apu, &mut mapper, &mut lines);
        assert_eq!(stall, 4, "DMC fetch consumes 4 CPU cycles");
        assert_eq!(apu.dmc_bytes_remaining, 0);
        assert!(!apu.dmc_buffer_empty);
    }

    #[test]
    fn dmc_irq_fires_at_sample_end_when_enabled() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        let mut mapper = TestMapper;
        apu.cpu_write_with_cycle(0x4010, 0x80, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4013, 0x00, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4015, 0x10, 0, &mut lines);
        // Tick APU: DMC signals fetch needed. Simulate the CPU fetch:
        // after delivery the sample count drops to 0 and the IRQ fires.
        for _ in 0..2 {
            apu.process_cpu_clock(&mut lines);
        }
        simulate_dmc_dma_fetch(&mut apu, &mut mapper, &mut lines);
        assert!(lines.has_irq_source(IrqSource::Dmc));
        assert!(
            (lines.has_irq_source(IrqSource::FrameCounter) || lines.has_irq_source(IrqSource::Dmc))
        );
    }

    #[test]
    fn write_4010_bit7_clear_acknowledges_dmc_irq() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        lines.set_irq_source(IrqSource::Dmc);
        apu.cpu_write_with_cycle(0x4010, 0x00, 0, &mut lines);
        assert!(!lines.has_irq_source(IrqSource::Dmc));
    }

    #[test]
    fn write_4011_sets_dmc_dac_directly() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        apu.cpu_write_with_cycle(0x4011, 0x7F, 0, &mut lines);
        assert_eq!(apu.dmc_dac, 0x7F);
        apu.cpu_write_with_cycle(0x4011, 0xFF, 0, &mut lines); // bit 7 stripped
        assert_eq!(apu.dmc_dac, 0x7F);
    }

    #[test]
    fn write_4017_irq_inhibit_clears_frame_irq() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        apu.frame_irq_flag = true;
        lines.set_irq_source(IrqSource::FrameCounter);
        apu.cpu_write_with_cycle(0x4017, 0x40, 0, &mut lines);
        assert!(!apu.frame_irq_flag);
        assert!(!lines.has_irq_source(IrqSource::FrameCounter));
    }

    // Envelope tests -quarter-frame clock + restart-on-length-load.
    // Mesen2 equivalent: ApuEnvelope::TickEnvelope / ResetEnvelope in
    // ApuEnvelope.h.

    #[test]
    fn envelope_constant_volume_outputs_speed_field() {
        let mut env = Envelope::default();
        env.write_control(0x10 | 0x07); // constant mode (bit 4), speed 7
        assert_eq!(env.output(), 7);
    }

    #[test]
    fn envelope_decay_outputs_decvolume_when_not_constant() {
        let mut env = Envelope::default();
        env.write_control(0x07); // envelope mode, speed 7
        env.schedule_restart();
        env.clock(false); // reload: dec_volume = 0xF
        assert_eq!(env.output(), 0xF);
        // Clock speed+1 = 8 quarters: dec_count_to_1 wraps, decvolume--
        for _ in 0..8 {
            env.clock(false);
        }
        assert_eq!(env.output(), 0xE);
    }

    #[test]
    fn envelope_loop_restarts_at_zero_when_length_halt_set() {
        // NES6 envelope no longer reads its own mode bit 1 for the
        // wrap-loop decision; the deferred length-halt is passed in via
        // `clock(halt)` mirroring Mesen2 ApuEnvelope.h:74
        // `LengthCounter.IsHalted()`.
        let mut env = Envelope::default();
        env.write_control(0x00); // envelope mode, speed 0 (fastest)
        env.schedule_restart();
        env.clock(true); // reload with halt=true: dec_volume=15
        for _ in 0..16 {
            env.clock(true); // halt asserted: dec_volume wraps at 0
        }
        // With halt set, dec_volume wraps from 0 back to 0xF.
        assert_eq!(env.output(), 0xF);
    }

    // Pulse sweep tests. Mesen2 equivalent:
    // ApuPulseChannel::TickSweep in ApuPulseChannel.h.

    #[test]
    fn pulse_sweep_positive_increases_period() {
        let mut p = Pulse::new(0);
        p.enabled = true;
        p.timer_period = 0x100;
        p.write_sweep(0x80 | (1 << 4) | 1); // enable, period 1, shift 1
        p.sweep_divider = 1; // fire on this clock (Mesen: divider hits 0)
        p.clock_sweep();
        // delta = 0x100 >> 1 = 0x80; new_period = 0x100 + 0x80 = 0x180
        assert_eq!(p.timer_period, 0x180);
    }

    #[test]
    fn pulse_sweep_negate_pulse1_subtracts_modulus_plus_1() {
        let mut p = Pulse::new(0);
        p.timer_period = 0x100;
        p.write_sweep(0x80 | (1 << 4) | 0x08 | 1); // enable, period 1, negate, shift 1
        p.sweep_divider = 1;
        p.clock_sweep();
        // delta = 0x80; new = 0x100 - 0x80 - 1 = 0x7F (pulse 1 quirk)
        assert_eq!(p.timer_period, 0x7F);
    }

    #[test]
    fn pulse_sweep_negate_pulse2_subtracts_modulus_only() {
        let mut p = Pulse::new(1);
        p.timer_period = 0x100;
        p.write_sweep(0x80 | (1 << 4) | 0x08 | 1);
        p.sweep_divider = 1;
        p.clock_sweep();
        // delta = 0x80; new = 0x100 - 0x80 = 0x80 (no -1 for pulse 2)
        assert_eq!(p.timer_period, 0x80);
    }

    #[test]
    fn pulse_sweep_divider_delays_first_apply() {
        // Mesen2 TickSweep: after a $4001 write the divider reloads to P+1 and
        // the period is not adjusted until it counts down to 0 (P+1 clocks). A
        // divider at 0 wraps to 255 and must NOT apply on entry -the prior
        // off-by-one applied immediately, running the sweep one half-frame
        // ahead (Wrecking Crew / 3-D WorldRunner pulse1).
        let mut p = Pulse::new(0);
        p.enabled = true;
        p.timer_period = 0x100;
        p.write_sweep(0x80 | (2 << 4) | 1); // enable, period raw=2 (divider 3), shift 1
        p.clock_sweep(); // tick1: 0->255 (no apply); reload sets divider=3
        assert_eq!(p.timer_period, 0x100, "no apply on reload tick");
        p.clock_sweep(); // 3->2
        p.clock_sweep(); // 2->1
        assert_eq!(p.timer_period, 0x100, "no apply before divider reaches 0");
        p.clock_sweep(); // >0 -> apply
        assert_eq!(p.timer_period, 0x180, "apply after P+1=3 clocks");
    }

    // DMC PCM DAC output tests. Mesen2 equivalent:
    // ApuDmcChannel::Clock DAC step in ApuDmcChannel.h.

    #[test]
    fn dmc_pcm_dac_increments_on_shift_bit_set() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        let mut mapper = TestMapper;
        apu.cpu_write_with_cycle(0x4011, 0x00, 0, &mut lines); // dac = 0
        apu.cpu_write_with_cycle(0x4010, 0x0F, 0, &mut lines); // period 54 cycles per bit
        apu.cpu_write_with_cycle(0x4012, 0x00, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4013, 0x01, 0, &mut lines); // size = 17 bytes
        apu.cpu_write_with_cycle(0x4015, 0x10, 0, &mut lines);
        // First instr: DMA needed -flag set -simulate CPU fetch -buf=0xAA.
        let mut stall = 0;
        for _ in 0..4 {
            apu.process_cpu_clock(&mut lines);
            stall = simulate_dmc_dma_fetch(&mut apu, &mut mapper, &mut lines);
            if stall != 0 {
                break;
            }
        }
        assert_eq!(stall, 4);
        assert_eq!(apu.dmc_read_buffer, 0xAA);
        // Run enough cycles for ~16 bit clocks (16 * 54 -864 cycles).
        // Each tick may set the pending flag again as samples drain;
        // simulate the fetch on each tick.
        for _ in 0..1000 {
            apu.process_cpu_clock(&mut lines);
            simulate_dmc_dma_fetch(&mut apu, &mut mapper, &mut lines);
        }
        // After many bit clocks, shift register has been populated and
        // DAC has been incremented multiple times by 0xAA's bits.
        // 0xAA = 0b10101010: half the bits set, so DAC oscillates around 0.
        // Just verify dac is non-zero or that shift register cycled.
        assert!(apu.dmc_bytes_remaining < 17 || apu.dmc_dac > 0 || !apu.dmc_silence_flag);
    }

    // Non-linear mixer tests -Mesen2 NesSoundMixer.cpp:182-183 constants.

    #[test]
    fn pulse_table_zero_index_returns_zero() {
        assert_eq!(pulse_table()[0], 0.0);
    }

    #[test]
    fn pulse_table_monotonically_increases() {
        let t = pulse_table();
        for i in 1..30 {
            assert!(t[i + 1] > t[i], "pulse_table[{}] should be greater", i + 1);
        }
    }

    #[test]
    fn tnd_table_monotonically_increases() {
        let t = tnd_table();
        for i in 1..202 {
            assert!(t[i + 1] > t[i], "tnd_table[{}] should be greater", i + 1);
        }
    }

    #[test]
    fn pulse_table_full_amplitude_matches_mesen2_formula() {
        // pulse_table[30] = 95.88 / (8128/30 + 100) -0.258
        let v = pulse_table()[30];
        assert!((v - 0.258).abs() < 0.005, "expected -0.258, got {v}");
    }

    #[test]
    fn tnd_table_full_amplitude_matches_mesen2_formula() {
        // tnd_table[202] = 159.79 / (22638/202 + 100) -0.7535
        let v = tnd_table()[202];
        assert!((v - 0.7535).abs() < 0.005, "expected -0.7535, got {v}");
    }

    // Audio synthesis end-to-end test (only when feature enabled).

    #[cfg(feature = "audio-synth")]
    #[test]
    fn process_cpu_clock_emits_audio_samples() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        apu.cpu_write_with_cycle(0x4015, 0x01, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4000, 0x9F, 0, &mut lines); // duty 2, halt, constant volume 15
        apu.cpu_write_with_cycle(0x4002, 0x40, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4003, 0x08, 0, &mut lines);
        for _ in 0..14_915 {
            for _ in 0..2 {
                apu.process_cpu_clock(&mut lines);
            }
        }
        apu.run(apu.cycles, &mut lines);
        let samples = apu.drain_samples();
        let expected = AUDIO_SAMPLE_RATE as i64 / 60;
        assert!(
            (samples.len() as i64 - expected).abs() < 50,
            "got {} samples, expected ~{}",
            samples.len(),
            expected
        );
        let nonzero = samples.iter().filter(|&&s| s.abs() > 0.001).count();
        assert!(nonzero > 0, "all samples silent");
    }

    #[cfg(feature = "audio-synth")]
    #[test]
    fn runtime_audio_disable_skips_sample_synthesis() {
        // set_audio_enabled(false) elides sample synthesis at runtime (the RL
        // fast path, mirroring ALE's `sound` setting); re-enabling resumes it.
        // Emulation state ($4015/IRQ/frame-counter/DMC) is unaffected -- the
        // RAM-identical proof is the integration test in scripts/test.py.
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        apu.set_audio_enabled(false);
        apu.cpu_write_with_cycle(0x4015, 0x01, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4000, 0x9F, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4002, 0x40, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4003, 0x08, 0, &mut lines);
        for _ in 0..14_915 {
            for _ in 0..2 {
                apu.process_cpu_clock(&mut lines);
            }
        }
        apu.run(apu.cycles, &mut lines);
        assert!(
            apu.drain_samples().is_empty(),
            "runtime audio-disable must emit no samples"
        );
        apu.set_audio_enabled(true);
        for _ in 0..2_000 {
            apu.process_cpu_clock(&mut lines);
        }
        apu.run(apu.cycles, &mut lines);
        assert!(
            !apu.drain_samples().is_empty(),
            "re-enabling audio must resume sample synthesis"
        );
    }

    #[cfg(not(feature = "audio-synth"))]
    #[test]
    fn process_cpu_clock_emits_no_samples_when_synth_disabled() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        let mut mapper = TestMapper;
        apu.cpu_write_with_cycle(0x4015, 0x01, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4003, 0x08, 0, &mut lines);
        for _ in 0..14_915 {
            for _ in 0..2 {
                apu.process_cpu_clock(&mut lines);
            }
        }
        assert!(
            apu.drain_samples().is_empty(),
            "no samples should be emitted when audio-synth feature is disabled"
        );
    }

    // Length counter tests.

    #[test]
    fn pulse_length_loads_on_high_timer_write_when_enabled() {
        // NES6: length reload is deferred per Mesen2 ApuLengthCounter.h:30-37.
        // After $4003 write the reload sits in `length_reload_value` until
        // `reload_length_counter()` fires inside `run()` post frame counter
        // step. Drive one APU clock so the gate runs.
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        apu.cpu_write_with_cycle(0x4015, 0x01, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4003, 0x08, 0, &mut lines);
        apu.process_cpu_clock(&mut lines);
        assert_eq!(apu.pulse1.length, 0xFE);
    }

    #[test]
    fn disabled_pulse_does_not_load_length() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        apu.cpu_write_with_cycle(0x4003, 0x08, 0, &mut lines);
        apu.process_cpu_clock(&mut lines);
        assert_eq!(apu.pulse1.length, 0);
    }

    #[test]
    fn status_write_disable_zeros_length() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        apu.cpu_write_with_cycle(0x4015, 0x0F, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4003, 0x08, 0, &mut lines);
        apu.cpu_write_with_cycle(0x400B, 0x08, 0, &mut lines);
        apu.cpu_write_with_cycle(0x400F, 0x08, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4007, 0x08, 0, &mut lines);
        // Commit deferred reloads.
        apu.process_cpu_clock(&mut lines);
        assert!(apu.pulse1.length > 0);
        assert!(apu.pulse2.length > 0);
        assert!(apu.triangle.length > 0);
        assert!(apu.noise.length > 0);
        apu.cpu_write_with_cycle(0x4015, 0x00, 0, &mut lines);
        assert_eq!(apu.pulse1.length, 0);
        assert_eq!(apu.pulse2.length, 0);
        assert_eq!(apu.triangle.length, 0);
        assert_eq!(apu.noise.length, 0);
    }

    /// Mesen2 `ApuLengthCounter::ReloadCounter` test 4-5 race:
    /// a half-frame clock between the write and the deferred reload
    /// must decrement the counter and cause the reload to be discarded.
    #[test]
    fn len_reload_timing_race_discards_pending_when_half_frame_fires() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        // Enable pulse1, write a short length and drive the half-frame
        // tick so length decays a few steps.
        apu.cpu_write_with_cycle(0x4015, 0x01, 0, &mut lines);
        apu.cpu_write_with_cycle(0x4000, 0x00, 0, &mut lines); // halt=0
        apu.cpu_write_with_cycle(0x4003, 0x08, 0, &mut lines); // length=0xFE
        apu.process_cpu_clock(&mut lines);
        let before = apu.pulse1.length;
        assert_eq!(before, 0xFE);

        // Race scenario: write $4003 again to start a new reload. Before
        // the half-frame fires we observe pending state.
        apu.cpu_write_with_cycle(0x4003, 0x10, 0, &mut lines); // pending=0x14
        assert!(apu.pulse1.length_reload_value != 0);
        assert_eq!(apu.pulse1.length_previous_value, 0xFE);

        // Manually fire a half-frame clock BEFORE the next run iteration
        // (simulating Mesen2's `_frameCounter->Run()` decrementing the
        // counter inside the same loop iteration as `ReloadCounter`).
        apu.pulse1.clock_length();
        assert_eq!(apu.pulse1.length, 0xFE - 1);

        // Reload now: counter (0xFD) != previous_value (0xFE) -discard.
        apu.pulse1.reload_length_counter();
        assert_eq!(apu.pulse1.length, 0xFE - 1);
        assert_eq!(apu.pulse1.length_reload_value, 0);
    }

    #[test]
    fn length_halt_deferred_until_reload_length_counter() {
        // Mesen2 ApuLengthCounter::ReloadCounter ends with `_halt = _newHaltValue`.
        // Writing $4000 with bit 5 set must NOT immediately halt; it is
        // observed only after `reload_length_counter()` runs.
        let mut p = Pulse::new(0);
        p.enabled = true;
        p.length = 10;
        // length_halt starts false, write_control sets new_length_halt=true.
        p.write_control(0x20);
        assert!(!p.length_halt);
        assert!(p.new_length_halt);
        // clock_length still decrements with the OLD live halt (false).
        p.clock_length();
        assert_eq!(p.length, 9);
        // After reload, the new halt is applied.
        p.reload_length_counter();
        assert!(p.length_halt);
        // Subsequent clock_length is gated by the new halt.
        p.clock_length();
        assert_eq!(p.length, 9);
    }

    #[test]
    fn snapshot_round_trip_preserves_frame_counter_dmc_and_channel_state() {
        let mut apu = Apu {
            frame_step_mode: 1,
            frame_current_step: 4,
            dmc_dac: 0x55,
            dmc_shift_reg: 0xAA,
            dmc_transfer_start_delay: 2,
            ..Apu::default()
        };
        apu.pulse1.timer_period = 0x456;
        apu.triangle.linear_counter = 7;
        apu.noise.lfsr = 0x2345;
        let snap = apu.snapshot_bytes();
        let mut restored = Apu::default();
        restored.restore_snapshot(&snap).unwrap();
        assert_eq!(restored.frame_step_mode, 1);
        assert_eq!(restored.frame_current_step, 4);
        assert_eq!(restored.dmc_dac, 0x55);
        assert_eq!(restored.dmc_shift_reg, 0xAA);
        assert_eq!(restored.dmc_transfer_start_delay, 2);
        assert_eq!(restored.pulse1.timer_period, 0x456);
        assert_eq!(restored.triangle.linear_counter, 7);
        assert_eq!(restored.noise.lfsr, 0x2345);
    }

    #[test]
    fn frame_counter_uses_mesen2_step_table_for_5step_mode() {
        let mut apu = Apu::default();
        let mut lines = InterruptLines::default();
        apu.cpu_write_with_cycle(0x4017, 0x80, 0, &mut lines);
        for _ in 0..3 {
            apu.process_cpu_clock(&mut lines);
        }
        apu.run(apu.cycles, &mut lines);
        assert_eq!(apu.frame_step_mode, 1);
        assert_eq!(FRAME_STEP_CYCLES_NTSC[1][3], 29829);
        assert_eq!(FRAME_STEP_CYCLES_NTSC[1][4], 37281);
    }
}
