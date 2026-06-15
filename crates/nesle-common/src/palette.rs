#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Palette {
    rgb: Vec<u8>,
}

impl Palette {
    pub fn from_rgb_entries(rgb: Vec<u8>) -> Self {
        Self { rgb }
    }

    pub fn entries(&self) -> usize {
        self.rgb.len() / 3
    }

    pub fn rgb(&self) -> &[u8] {
        &self.rgb
    }
}
