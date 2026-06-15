#[derive(Debug, Clone)]
pub struct CoreState {
    frame_count: u64,
    bytes: Vec<u8>,
}

impl CoreState {
    /// Reconstruct a state wrapper around a serialized core snapshot.
    pub fn from_bytes(frame_count: u64, bytes: Vec<u8>) -> Self {
        Self { frame_count, bytes }
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume and return the serialized snapshot bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}
