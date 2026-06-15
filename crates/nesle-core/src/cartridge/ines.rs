use nesle_common::{NesleError, Result};

use super::{CartridgeFormat, CartridgeImage, Mirroring, Region};

const INES_MAGIC: &[u8; 4] = b"NES\x1a";
const HEADER_LEN: usize = 16;
const PRG_BANK: usize = 16 * 1024;
const CHR_BANK: usize = 8 * 1024;

pub fn parse_ines(bytes: &[u8]) -> Result<CartridgeImage> {
    if bytes.len() < HEADER_LEN || &bytes[..4] != INES_MAGIC {
        return Err(NesleError::InvalidRom("missing iNES header".to_string()));
    }

    let flags6 = bytes[6];
    let flags7 = bytes[7];
    let format = if flags7 & 0x0C == 0x08 {
        CartridgeFormat::Nes20
    } else {
        CartridgeFormat::INes
    };
    let flags8 = bytes[8];
    let flags9 = bytes[9];
    let mapper_id = match format {
        CartridgeFormat::INes => ((flags7 as u16) & 0xF0) | ((flags6 as u16) >> 4),
        CartridgeFormat::Nes20 => {
            ((flags6 as u16) >> 4) | ((flags7 as u16) & 0xF0) | (((flags8 as u16) & 0x0F) << 8)
        }
        _ => unreachable!(),
    };
    let submapper = match format {
        CartridgeFormat::Nes20 => flags8 >> 4,
        _ => 0,
    };
    let (prg_len, chr_len, chr_banks) = match format {
        CartridgeFormat::INes => {
            // iNES PRG count 0 encodes 256 banks.
            let prg_raw = bytes[4] as usize;
            let prg = if prg_raw == 0 { 256 } else { prg_raw };
            (
                prg * PRG_BANK,
                bytes[5] as usize * CHR_BANK,
                bytes[5] as usize,
            )
        }
        CartridgeFormat::Nes20 => {
            let prg_len = nes20_rom_size(bytes[4], flags9 & 0x0F, PRG_BANK)?;
            let chr_len = nes20_rom_size(bytes[5], flags9 >> 4, CHR_BANK)?;
            (prg_len, chr_len, chr_len / CHR_BANK)
        }
        _ => unreachable!(),
    };
    let mirroring = if flags6 & 0x08 != 0 {
        Mirroring::FourScreen
    } else if flags6 & 0x01 != 0 {
        Mirroring::Vertical
    } else {
        Mirroring::Horizontal
    };
    // NES 2.0 byte 12 and iNES flags9 select the default region.
    let region = match format {
        CartridgeFormat::Nes20 if bytes.len() > 12 => match bytes[12] & 0x03 {
            0 | 2 => Region::Ntsc,
            1 => Region::Pal,
            _ => Region::Dendy,
        },
        CartridgeFormat::INes if flags9 & 0x01 != 0 => Region::Pal,
        _ => Region::Ntsc,
    };
    // NES 2.0 byte 15 is the default expansion device; reserved values
    // map to unspecified.
    let input_device = match format {
        CartridgeFormat::Nes20 if bytes.len() > 15 && bytes[15] < 0x2E => bytes[15],
        _ => 0,
    };
    let trainer_len = if flags6 & 0x04 != 0 { 512 } else { 0 };
    let prg_start = HEADER_LEN + trainer_len;
    let chr_start = prg_start + prg_len;
    let end = chr_start + chr_len;
    if bytes.len() < end {
        return Err(NesleError::InvalidRom(
            "iNES payload is truncated".to_string(),
        ));
    }

    let (work_ram_size, save_ram_size, prg_ram_unspecified) = prg_ram_sizes(format, bytes);
    let (chr_ram_size, save_chr_ram_size, chr_ram_unspecified) =
        chr_ram_sizes(format, bytes, chr_banks);

    Ok(CartridgeImage {
        format,
        mapper_id,
        submapper,
        mirroring,
        battery: flags6 & 0x02 != 0,
        region,
        prg_rom: bytes[prg_start..chr_start].to_vec(),
        chr_rom: bytes[chr_start..end].to_vec(),
        trainer_data: bytes[HEADER_LEN..HEADER_LEN + trainer_len].to_vec(),
        work_ram_size,
        save_ram_size,
        prg_ram_size: work_ram_size + save_ram_size,
        prg_ram_unspecified,
        save_chr_ram_size,
        chr_ram_size,
        chr_ram_unspecified,
        input_device,
    })
}

fn nes20_rom_size(count: u8, upper_nibble: u8, unit: usize) -> Result<usize> {
    if upper_nibble == 0x0F {
        let exponent = usize::from(count >> 2);
        let multiplier = usize::from(count & 0x03) * 2 + 1;
        if exponent >= usize::BITS as usize {
            return Err(NesleError::InvalidRom(
                "NES 2.0 exponent ROM-size encoding overflows usize".to_string(),
            ));
        }
        multiplier.checked_shl(exponent as u32).ok_or_else(|| {
            NesleError::InvalidRom("NES 2.0 exponent ROM-size encoding overflows usize".to_string())
        })
    } else {
        Ok((usize::from(count) | (usize::from(upper_nibble) << 8)) * unit)
    }
}

fn ram_shift_size(shift: u8) -> usize {
    if shift == 0 {
        0
    } else {
        64usize << shift
    }
}

fn prg_ram_sizes(format: CartridgeFormat, header: &[u8]) -> (usize, usize, bool) {
    match format {
        CartridgeFormat::INes => {
            // iNES 1.0 has no reliable PRG-RAM size; defer to mapper defaults.
            (0, 0, true)
        }
        CartridgeFormat::Nes20 => {
            let byte = header[10];
            (
                ram_shift_size(byte & 0x0F),
                ram_shift_size(byte >> 4),
                false,
            )
        }
        _ => (0, 0, false),
    }
}

fn chr_ram_sizes(format: CartridgeFormat, header: &[u8], chr_banks: usize) -> (usize, usize, bool) {
    match format {
        CartridgeFormat::INes => {
            if chr_banks == 0 {
                (CHR_BANK, 0, true)
            } else {
                (0, 0, true)
            }
        }
        CartridgeFormat::Nes20 => {
            let byte = header[11];
            (
                ram_shift_size(byte & 0x0F),
                ram_shift_size(byte >> 4),
                false,
            )
        }
        _ => (0, 0, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ines_header() -> Vec<u8> {
        let mut rom = Vec::new();
        rom.extend_from_slice(INES_MAGIC);
        rom.extend_from_slice(&[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        rom
    }

    #[test]
    fn parses_legacy_ines_mapper_and_default_ram() {
        let mut rom = ines_header();
        rom[6] = 0x21;
        rom[7] = 0x40;
        rom.extend(std::iter::repeat_n(0xaa, PRG_BANK));
        rom.extend(std::iter::repeat_n(0xbb, CHR_BANK));

        let image = parse_ines(&rom).unwrap();
        assert_eq!(image.format, CartridgeFormat::INes);
        assert_eq!(image.mapper_id, 0x42);
        assert_eq!(image.mirroring, Mirroring::Vertical);
        // iNES 1.0 PRG RAM size is unspecified; mapper default supplies it.
        assert_eq!(image.prg_ram_size, 0);
        assert_eq!(image.work_ram_size, 0);
        assert_eq!(image.save_ram_size, 0);
        assert!(image.prg_ram_unspecified);
        assert_eq!(image.initialized_prg_ram(8 * 1024).len(), 8 * 1024);
        assert_eq!(image.chr_ram_size, 0);
    }

    #[test]
    fn parses_nes20_mapper_submapper_and_ram_sizes() {
        let mut rom = ines_header();
        rom[4] = 2;
        rom[5] = 0;
        rom[6] = 0x30;
        rom[7] = 0x98;
        rom[8] = 0x51;
        rom[10] = 0x76;
        rom[11] = 0x54;
        rom.extend(std::iter::repeat_n(0xaa, PRG_BANK * 2));

        let image = parse_ines(&rom).unwrap();
        assert_eq!(image.format, CartridgeFormat::Nes20);
        assert_eq!(image.mapper_id, 0x193);
        assert_eq!(image.submapper, 5);
        assert_eq!(image.prg_ram_size, (64usize << 6) + (64usize << 7));
        assert_eq!(image.work_ram_size, 64usize << 6);
        assert_eq!(image.save_ram_size, 64usize << 7);
        assert_eq!(image.chr_ram_size, 64usize << 4);
        assert_eq!(image.save_chr_ram_size, 64usize << 5);
    }

    #[test]
    fn parses_nes20_exponent_multiplier_rom_sizes() {
        let mut rom = ines_header();
        rom[4] = 0x16;
        rom[5] = 0x10;
        rom[7] = 0x08;
        rom[9] = 0xFF;
        let prg_len = 5usize << 5;
        let chr_len = 1usize << 4;
        rom.extend(std::iter::repeat_n(0xaa, prg_len));
        rom.extend(std::iter::repeat_n(0xbb, chr_len));

        let image = parse_ines(&rom).unwrap();
        assert_eq!(image.prg_rom.len(), prg_len);
        assert_eq!(image.chr_rom.len(), chr_len);
    }

    #[test]
    fn keeps_trainer_data_for_mapper_ram_initialization() {
        let mut rom = ines_header();
        rom[6] = 0x04;
        rom.extend(std::iter::repeat_n(0x7a, 512));
        rom.extend(std::iter::repeat_n(0xaa, PRG_BANK));
        rom.extend(std::iter::repeat_n(0xbb, CHR_BANK));

        let image = parse_ines(&rom).unwrap();
        assert_eq!(image.trainer_data.len(), 512);
        let ram = image.initialized_prg_ram(8 * 1024);
        assert_eq!(&ram[0x1000..0x1200], image.trainer_data.as_slice());
    }
}
