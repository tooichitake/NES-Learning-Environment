/// Pixel dimensions for frame buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameDims {
    pub width: usize,
    pub height: usize,
}

impl FrameDims {
    /// Native NES visible framebuffer dimensions.
    pub const NES: Self = Self {
        width: 256,
        height: 240,
    };

    /// Number of pixels in one single-channel frame.
    pub const fn len(self) -> usize {
        self.width * self.height
    }

    /// True when either dimension is zero.
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// Palette-index framebuffer, one byte per NES pixel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedFrame {
    pub dims: FrameDims,
    pub pixels: Vec<u8>,
}

/// RGB framebuffer, three bytes per pixel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbFrame {
    pub dims: FrameDims,
    pub pixels: Vec<u8>,
}

/// Grayscale framebuffer, one byte per pixel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrayscaleFrame {
    pub dims: FrameDims,
    pub pixels: Vec<u8>,
}

impl IndexedFrame {
    /// Blank native-size indexed frame using the transparent/background sentinel.
    pub fn blank_nes() -> Self {
        Self {
            dims: FrameDims::NES,
            pixels: vec![0x80; FrameDims::NES.len()],
        }
    }
}
