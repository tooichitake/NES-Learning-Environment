#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Ntsc,
    Pal,
    Dendy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunSettings {
    pub region: Region,
    pub display_enabled: bool,
    pub sound_enabled: bool,
}

impl Default for RunSettings {
    fn default() -> Self {
        Self {
            region: Region::Ntsc,
            display_enabled: false,
            sound_enabled: false,
        }
    }
}
