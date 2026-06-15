#[derive(Debug, Clone, Copy, Default)]
pub struct ControllerPorts {
    masks: [u8; 4],
    /// Per-latch button snapshot. `read()` returns captured latch bits,
    /// not live masks; mirrors Mesen2 `NesController::RefreshStateBuffer`.
    latch_masks: [u8; 4],
    shift: [u8; 4],
    strobe: bool,
    pending_value: u8,
    pending_delay: u8,
    pending_write: bool,
    /// Deferred input capture. Host masks become visible at scanline 241
    /// (`NesConfig::InputScanline`), matching Mesen2 `UpdateInputState`.
    pending_masks: [u8; 4],
    /// FourScore adapter mode, derived from cartridge metadata at ROM load.
    four_score_mode: bool,
    /// FourScore port-read countdown before signature bytes are emitted.
    sig_counter: [u8; 2],
    /// FourScore fixed identification shift registers.
    signature: [u8; 2],
}

impl ControllerPorts {
    pub fn reset(&mut self) {
        self.masks = [0; 4];
        self.pending_masks = [0; 4];
        self.latch_masks = [0; 4];
        self.shift = [0; 4];
        self.strobe = false;
        self.pending_value = 0;
        self.pending_delay = 0;
        self.pending_write = false;
        self.sig_counter = [0; 2];
        self.signature = [0; 2];
        // `four_score_mode` is NOT reset -it's a cartridge-derived
        // attribute set once at load time.
    }

    /// Configure FourScore adapter mode from cartridge metadata.
    pub fn set_four_score_mode(&mut self, enabled: bool) {
        self.four_score_mode = enabled;
    }

    pub fn four_score_mode(&self) -> bool {
        self.four_score_mode
    }

    /// Clear impossible D-pad pairs for deterministic NES input.
    fn filter_invalid_input(mask: u8) -> u8 {
        let mut m = mask;
        if m & 0x30 == 0x30 {
            m &= !0x30;
        }
        if m & 0xC0 == 0xC0 {
            m &= !0xC0;
        }
        m
    }

    pub fn set_mask(&mut self, port: usize, mask: u8) {
        // PPU commits pending masks at the input scanline.
        let mask = Self::filter_invalid_input(mask);
        if let Some(slot) = self.pending_masks.get_mut(port) {
            *slot = mask;
        }
    }

    pub fn set_masks(&mut self, masks: [u8; 4]) {
        // Same input-scanline deferral as `set_mask`.
        self.pending_masks = masks.map(Self::filter_invalid_input);
    }

    pub fn masks(&self) -> &[u8; 4] {
        &self.masks
    }

    /// Commit pending input at the PPU input-capture scanline.
    pub fn commit_pending_input(&mut self) {
        self.masks = self.pending_masks;
    }

    pub fn snapshot_bytes(&self) -> [u8; 25] {
        let mut bytes = [0; 25];
        bytes[..4].copy_from_slice(&self.masks);
        bytes[4..8].copy_from_slice(&self.shift);
        bytes[8] = u8::from(self.strobe);
        bytes[9] = self.pending_value;
        bytes[10] = self.pending_delay;
        bytes[11] = u8::from(self.pending_write);
        // FourScore state bytes 12-16.
        bytes[12] = u8::from(self.four_score_mode);
        bytes[13] = self.sig_counter[0];
        bytes[14] = self.sig_counter[1];
        bytes[15] = self.signature[0];
        bytes[16] = self.signature[1];
        // pending_masks layout bytes 17-20.
        bytes[17..21].copy_from_slice(&self.pending_masks);
        // latch_masks layout bytes 21-24.
        bytes[21..25].copy_from_slice(&self.latch_masks);
        bytes
    }

    pub fn restore_snapshot(&mut self, bytes: [u8; 25]) {
        self.masks.copy_from_slice(&bytes[..4]);
        self.shift.copy_from_slice(&bytes[4..8]);
        self.strobe = bytes[8] != 0;
        self.pending_value = bytes[9];
        self.pending_delay = bytes[10];
        self.pending_write = bytes[11] != 0;
        self.four_score_mode = bytes[12] != 0;
        self.sig_counter = [bytes[13], bytes[14]];
        self.signature = [bytes[15], bytes[16]];
        self.pending_masks.copy_from_slice(&bytes[17..21]);
        self.latch_masks.copy_from_slice(&bytes[21..25]);
    }

    pub fn write_strobe(&mut self, value: u8) {
        self.pending_write = false;
        self.pending_delay = 0;
        self.apply_strobe(value);
    }

    pub fn queue_strobe_write(&mut self, value: u8, cpu_cycle: u64) {
        // Strobe writes are delayed by CPU-cycle parity; newer writes replace
        // pending ones before they fire.
        self.pending_value = value;
        self.pending_delay = if cpu_cycle & 0x01 != 0 { 1 } else { 2 };
        self.pending_write = true;
    }

    pub fn process_pending_write(&mut self) {
        if !self.pending_write {
            return;
        }
        self.pending_delay = self.pending_delay.saturating_sub(1);
        if self.pending_delay == 0 {
            self.pending_write = false;
            self.apply_strobe(self.pending_value);
        }
    }

    fn apply_strobe(&mut self, value: u8) {
        let next = value & 0x01 != 0;
        if self.strobe && !next {
            self.latch();
        }
        self.strobe = next;
    }

    pub fn read(&mut self, port: usize) -> u8 {
        if port >= 2 {
            return 0;
        }
        if self.strobe {
            self.latch();
        }
        if self.four_score_mode {
            // Four Score multiplexes primary, secondary, then signature bits.
            if self.sig_counter[port] > 0 {
                // Decrement before selecting source port.
                self.sig_counter[port] -= 1;
                let post = self.sig_counter[port];
                // Reads 1-8 use primary port; reads 9-16 use port+2.
                let source = if post < 8 { port + 2 } else { port };
                // Each logical controller has its own latched shift cursor.
                let read_bit = self.shift[source];
                // use snapshot, not live masks.
                let v = if read_bit < 8 {
                    (self.latch_masks[source] >> read_bit) & 0x01
                } else {
                    0
                };
                self.shift[source] = read_bit.saturating_add(1);
                v
            } else {
                let v = self.signature[port] & 0x01;
                self.signature[port] = (self.signature[port] >> 1) | 0x80;
                v
            }
        } else {
            // Standard NES controller reads return 1 after the 8 latched bits.
            let read_bit = self.shift[port];
            // use snapshot, not live masks.
            let v = if read_bit < 8 {
                (self.latch_masks[port] >> read_bit) & 0x01
            } else {
                1
            };
            self.shift[port] = read_bit.saturating_add(1);
            v
        }
    }

    fn latch(&mut self) {
        self.shift = [0; 4];
        // Reads consume a snapshot captured at latch time, not live masks.
        self.latch_masks = self.masks;
        if self.four_score_mode {
            // Reset Four Score counter and signatures on every latch.
            self.sig_counter = [16, 16];
            self.signature = [0x08, 0x04];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifts_controller_bits_after_latch() {
        let mut ports = ControllerPorts::default();
        // set_mask writes pending; commit makes the
        // mask active (production caller is PPU at scanline 241).
        ports.set_mask(0, 0b1000_0101);
        ports.commit_pending_input();
        ports.write_strobe(1);
        ports.write_strobe(0);

        let bits: Vec<u8> = (0..8).map(|_| ports.read(0)).collect();
        assert_eq!(bits, vec![1, 0, 1, 0, 0, 0, 0, 1]);
        // Standard controller reads return 1 after the 8 latched bits.
        let after_eight: Vec<u8> = (0..12).map(|_| ports.read(0)).collect();
        assert_eq!(after_eight, vec![1; 12]);
    }

    #[test]
    fn four_score_reads_all_four_controllers() {
        // Four Score: $4016 serves controllers 1 then 3; $4017 serves 2 then 4.
        let mut ports = ControllerPorts::default();
        ports.set_four_score_mode(true);
        // Distinct masks; avoid the U+D / L+R interlock (filter_invalid_input).
        ports.set_masks([0x0F, 0x03, 0x50, 0x88]); // c1, c2, c3, c4
        ports.commit_pending_input();
        ports.write_strobe(1);
        ports.write_strobe(0);

        // $4016: controller 1 (0x0F) LSB-first, then controller 3 (0x50) LSB-first.
        let p0: Vec<u8> = (0..16).map(|_| ports.read(0)).collect();
        let bits = |m: u8| (0..8).map(move |i| (m >> i) & 1);
        let expect0: Vec<u8> = bits(0x0F).chain(bits(0x50)).collect();
        assert_eq!(p0, expect0, "$4016 must serialize controllers 1 then 3");

        // $4017: controller 2 (0x03) then controller 4 (0x88).
        let p1: Vec<u8> = (0..16).map(|_| ports.read(1)).collect();
        let expect1: Vec<u8> = bits(0x03).chain(bits(0x88)).collect();
        assert_eq!(p1, expect1, "$4017 must serialize controllers 2 then 4");

        // Reads 17+ emit the Four Score signature (0x08 on $4016, 0x04 on $4017).
        let sig0: Vec<u8> = (0..8).map(|_| ports.read(0)).collect();
        assert_eq!(
            sig0,
            vec![0, 0, 0, 1, 0, 0, 0, 0],
            "$4016 signature 0x08 LSB-first"
        );
        let sig1: Vec<u8> = (0..8).map(|_| ports.read(1)).collect();
        assert_eq!(
            sig1,
            vec![0, 0, 1, 0, 0, 0, 0, 0],
            "$4017 signature 0x04 LSB-first"
        );
    }

    #[test]
    fn falling_strobe_edge_restarts_serial_reads() {
        let mut ports = ControllerPorts::default();
        ports.set_mask(0, 0x01);
        ports.commit_pending_input();
        ports.write_strobe(1);
        ports.write_strobe(0);
        assert_eq!(ports.read(0), 1);
        assert_eq!(ports.read(0), 0);
        ports.write_strobe(1);
        ports.write_strobe(0);
        assert_eq!(ports.read(0), 1);
    }
}
