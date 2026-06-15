pub mod fds;
pub mod image;
pub mod ines;
pub mod unif;

use nesle_common::{NesleError, Result};

pub use image::{CartridgeFormat, CartridgeImage, Mirroring, Region};

pub fn parse_cartridge_image(bytes: &[u8]) -> Result<CartridgeImage> {
    if bytes.starts_with(b"NES\x1a") {
        return ines::parse_ines(bytes);
    }
    if bytes.starts_with(b"UNIF") {
        return unif::parse_unif(bytes);
    }
    if bytes.starts_with(b"FDS\x1a") {
        return fds::parse_fds(bytes);
    }
    Err(NesleError::InvalidRom(
        "unsupported NES cartridge format".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_supported_cartridge_magic_to_format_parsers() {
        let mut rom = Vec::new();
        rom.extend_from_slice(b"NES\x1a");
        rom.extend_from_slice(&[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        rom.extend(std::iter::repeat_n(0xea, 16 * 1024));
        rom.extend(std::iter::repeat_n(0, 8 * 1024));
        assert_eq!(
            parse_cartridge_image(&rom).unwrap().format,
            CartridgeFormat::INes
        );

        // UNIF + FDS dispatch: both now have real parsers ().
        // A truncated UNIF header must still be routed to the UNIF
        // parser, whose error message identifies the format.
        let err = parse_cartridge_image(b"UNIFstub").unwrap_err().to_string();
        assert!(err.contains("UNIF"));

        let err = parse_cartridge_image(b"FDS\x1astub")
            .unwrap_err()
            .to_string();
        assert!(err.contains("FDS"));
    }

    #[test]
    fn rejects_unknown_rom_magic_before_core_load() {
        let err = parse_cartridge_image(b"not a cartridge")
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported NES cartridge format"));
    }
}
