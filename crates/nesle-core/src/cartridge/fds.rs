use nesle_common::{NesleError, Result};

use super::CartridgeImage;

const FDS_MAGIC: &[u8; 4] = b"FDS\x1a";
const FDS_HEADER_LEN: usize = 16;

/// Validate FDS magic and reject clearly; FDS hardware is outside NESLE scope.
pub fn parse_fds(bytes: &[u8]) -> Result<CartridgeImage> {
    if bytes.len() < FDS_HEADER_LEN || &bytes[..4] != FDS_MAGIC {
        return Err(NesleError::InvalidRom(
            "FDS cartridge: missing or truncated header".to_string(),
        ));
    }
    Err(NesleError::InvalidRom(
        "FDS (Famicom Disk System) cartridges are not supported. \
         NESLE targets standard NES mappers {0,1,2,3,4,5,7,9,10,69}; \
         FDS requires a separate disk controller, external BIOS ROM, \
         and runtime disk-side swapping. Use a full-featured emulator \
         (Mesen2, FCEUX) for FDS playback."
            .to_string(),
    ))
}
