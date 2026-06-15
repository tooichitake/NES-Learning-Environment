#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioBuffer {
    samples: Vec<i16>,
    channels: u16,
}

impl AudioBuffer {
    pub fn stereo(samples: Vec<i16>) -> Self {
        Self {
            samples,
            channels: 2,
        }
    }

    pub fn empty_stereo() -> Self {
        Self::stereo(Vec::new())
    }

    pub fn samples(&self) -> &[i16] {
        &self.samples
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }
}
