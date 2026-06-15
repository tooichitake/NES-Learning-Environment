/// NES controller button bit values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NesButton {
    A = 0x01,
    B = 0x02,
    Select = 0x04,
    Start = 0x08,
    Up = 0x10,
    Down = 0x20,
    Left = 0x40,
    Right = 0x80,
}

/// Named action mask used by ALE-style legal action sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NesAction {
    /// Stable display/API name.
    pub name: &'static str,
    /// Bitwise OR of [`NesButton`] values.
    pub mask: u8,
}

impl NesAction {
    pub const fn new(name: &'static str, mask: u8) -> Self {
        Self { name, mask }
    }
}

/// Borrowed legal action table.
pub type ActionSet = &'static [NesAction];

/// Unified NES gameplay action set.
///
/// This is the ALE-style full action space used by general policies: 9 D-pad
/// states times {none, A, B, A+B}. Select and Start are excluded because they
/// are menu/pause controls rather than gameplay actions.
pub const NES_FULL_ACTION_SET: [NesAction; 36] = [
    NesAction::new("NOOP", 0x00),
    NesAction::new("A", 0x01),
    NesAction::new("B", 0x02),
    NesAction::new("AB", 0x03),
    NesAction::new("UP", 0x10),
    NesAction::new("UP_A", 0x11),
    NesAction::new("UP_B", 0x12),
    NesAction::new("UP_AB", 0x13),
    NesAction::new("DOWN", 0x20),
    NesAction::new("DOWN_A", 0x21),
    NesAction::new("DOWN_B", 0x22),
    NesAction::new("DOWN_AB", 0x23),
    NesAction::new("LEFT", 0x40),
    NesAction::new("LEFT_A", 0x41),
    NesAction::new("LEFT_B", 0x42),
    NesAction::new("LEFT_AB", 0x43),
    NesAction::new("RIGHT", 0x80),
    NesAction::new("RIGHT_A", 0x81),
    NesAction::new("RIGHT_B", 0x82),
    NesAction::new("RIGHT_AB", 0x83),
    NesAction::new("UPLEFT", 0x50),
    NesAction::new("UPLEFT_A", 0x51),
    NesAction::new("UPLEFT_B", 0x52),
    NesAction::new("UPLEFT_AB", 0x53),
    NesAction::new("UPRIGHT", 0x90),
    NesAction::new("UPRIGHT_A", 0x91),
    NesAction::new("UPRIGHT_B", 0x92),
    NesAction::new("UPRIGHT_AB", 0x93),
    NesAction::new("DOWNLEFT", 0x60),
    NesAction::new("DOWNLEFT_A", 0x61),
    NesAction::new("DOWNLEFT_B", 0x62),
    NesAction::new("DOWNLEFT_AB", 0x63),
    NesAction::new("DOWNRIGHT", 0xA0),
    NesAction::new("DOWNRIGHT_A", 0xA1),
    NesAction::new("DOWNRIGHT_B", 0xA2),
    NesAction::new("DOWNRIGHT_AB", 0xA3),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nes_full_action_set_is_unified_36() {
        assert_eq!(NES_FULL_ACTION_SET.len(), 36);
        assert_eq!(NES_FULL_ACTION_SET[0].mask, 0, "index 0 must be NOOP");
        let select_start = NesButton::Select as u8 | NesButton::Start as u8;
        let up_down = NesButton::Up as u8 | NesButton::Down as u8;
        let left_right = NesButton::Left as u8 | NesButton::Right as u8;
        for a in NES_FULL_ACTION_SET {
            assert_eq!(
                a.mask & select_start,
                0,
                "{} must not press Select/Start",
                a.name
            );
            assert_ne!(
                a.mask & up_down,
                up_down,
                "{} presses opposing Up+Down",
                a.name
            );
            assert_ne!(
                a.mask & left_right,
                left_right,
                "{} presses opposing Left+Right",
                a.name
            );
        }
    }
}
