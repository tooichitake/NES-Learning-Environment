use nesle_common::{NesleError, Result};

use super::{CartridgeFormat, CartridgeImage, Mirroring, Region};

const UNIF_MAGIC: &[u8; 4] = b"UNIF";
const HEADER_LEN: usize = 32;

/// Supported UNIF board-name to mapper-id lookup.
const BOARD_MAPPINGS: &[(&str, u16)] = &[
    ("NROM", 0),
    ("NROM-128", 0),
    ("NROM-256", 0),
    ("RROM", 0),
    ("RROM-128", 0),
    ("SAROM", 1),
    ("SBROM", 1),
    ("SCROM", 1),
    ("SEROM", 1),
    ("SGROM", 1),
    ("SKROM", 1),
    ("SL1ROM", 1),
    ("SLROM", 1),
    ("SNROM", 1),
    ("SOROM", 1),
    ("SXROM", 1),
    ("SUROM", 1),
    ("UNROM", 2),
    ("UOROM", 2),
    ("UNROM-512", 2),
    ("CNROM", 3),
    ("HKROM", 4), // HKROM is actually MMC6 (mapper 4 variant) per Mesen2
    ("TBROM", 4),
    ("TEROM", 4),
    ("TFROM", 4),
    ("TGROM", 4),
    ("TKROM", 4),
    ("TKSROM", 4),
    ("TLROM", 4),
    ("TLSROM", 4),
    ("TQROM", 4),
    ("TR1ROM", 4),
    ("TSROM", 4),
    ("TVROM", 4),
    ("EKROM", 5),
    ("ELROM", 5),
    ("ETROM", 5),
    ("EWROM", 5),
    ("ANROM", 7),
    ("AMROM", 7),
    ("AN1ROM", 7),
    ("AOROM", 7),
    ("PNROM", 9),
    ("PEEOROM", 9),
    ("FJROM", 10),
    ("FKROM", 10),
    ("JLROM", 69),
    ("JSROM", 69),
];

/// Parse the supported UNIF chunk subset into a cartridge image.
pub fn parse_unif(bytes: &[u8]) -> Result<CartridgeImage> {
    if bytes.len() < HEADER_LEN || &bytes[..4] != UNIF_MAGIC {
        return Err(NesleError::InvalidRom(
            "UNIF cartridge: missing or truncated header".to_string(),
        ));
    }

    let mut prg_chunks: [Vec<u8>; 16] = Default::default();
    let mut chr_chunks: [Vec<u8>; 16] = Default::default();
    let mut mapper_name = String::new();
    let mut mirroring = Mirroring::Horizontal;
    let mut battery = false;
    let mut region = Region::Ntsc;

    let mut pos = HEADER_LEN;
    while pos + 8 <= bytes.len() {
        let fourcc = std::str::from_utf8(&bytes[pos..pos + 4])
            .map_err(|_| NesleError::InvalidRom("UNIF chunk id is not ASCII".to_string()))?;
        let length = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        pos += 8;
        let chunk_end = pos.checked_add(length).ok_or_else(|| {
            NesleError::InvalidRom("UNIF chunk length overflows usize".to_string())
        })?;
        if chunk_end > bytes.len() {
            return Err(NesleError::InvalidRom(
                "UNIF chunk payload extends past end-of-file".to_string(),
            ));
        }
        let payload = &bytes[pos..chunk_end];

        if fourcc == "MAPR" {
            mapper_name = parse_mapper_name(payload);
        } else if let Some(idx) = parse_prg_chunk_index(fourcc) {
            prg_chunks[idx] = payload.to_vec();
        } else if let Some(idx) = parse_chr_chunk_index(fourcc) {
            chr_chunks[idx] = payload.to_vec();
        } else if fourcc == "MIRR" && !payload.is_empty() {
            mirroring = match payload[0] {
                0 => Mirroring::Horizontal,
                1 => Mirroring::Vertical,
                2 => Mirroring::SingleScreenLower,
                3 => Mirroring::SingleScreenUpper,
                4 => Mirroring::FourScreen,
                _ => Mirroring::Horizontal,
            };
        } else if fourcc == "BATR" && !payload.is_empty() {
            battery = payload[0] != 0;
        } else if fourcc == "TVCI" && !payload.is_empty() {
            region = match payload[0] {
                1 => Region::Pal,
                _ => Region::Ntsc,
            };
        }
        // Other chunks are accepted and ignored.

        pos = chunk_end;
    }

    if mapper_name.is_empty() {
        return Err(NesleError::InvalidRom(
            "UNIF cartridge: missing MAPR (board name) chunk".to_string(),
        ));
    }
    let mapper_id = resolve_mapper(&mapper_name).ok_or_else(|| {
        NesleError::InvalidRom(format!(
            "UNIF cartridge: unsupported board name `{mapper_name}`"
        ))
    })?;

    let mut prg_rom = Vec::new();
    for chunk in &prg_chunks {
        prg_rom.extend_from_slice(chunk);
    }
    let mut chr_rom = Vec::new();
    for chunk in &chr_chunks {
        chr_rom.extend_from_slice(chunk);
    }

    if prg_rom.is_empty() {
        return Err(NesleError::InvalidRom(
            "UNIF cartridge: no PRG ROM chunks (PRG0..PRGF all empty)".to_string(),
        ));
    }

    // UNIF has no formal RAM-size declaration; use 8KB defaults.
    let chr_ram_size = if chr_rom.is_empty() { 8 * 1024 } else { 0 };
    let work_ram_size = 8 * 1024;
    let save_ram_size = if battery { 8 * 1024 } else { 0 };

    Ok(CartridgeImage {
        format: CartridgeFormat::Unif,
        mapper_id,
        submapper: 0,
        mirroring,
        battery,
        region,
        prg_rom,
        chr_rom,
        trainer_data: Vec::new(),
        work_ram_size,
        save_ram_size,
        prg_ram_size: work_ram_size + save_ram_size,
        prg_ram_unspecified: true,
        save_chr_ram_size: 0,
        chr_ram_size,
        chr_ram_unspecified: chr_ram_size > 0,
        input_device: 0,
    })
}

/// Normalize the null-terminated UNIF board name.
fn parse_mapper_name(payload: &[u8]) -> String {
    let mut s: String = payload
        .iter()
        .take_while(|&&b| b != 0)
        .filter(|&&b| b != b' ' && b != b'\t')
        .map(|&b| b as char)
        .collect();
    for prefix in &["NES-", "UNL-", "HVC-", "BTL-", "BMC-"] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.to_string();
            break;
        }
    }
    s
}

/// Parse a `PRGx` chunk id where `x` is hex.
fn parse_prg_chunk_index(fourcc: &str) -> Option<usize> {
    let bytes = fourcc.as_bytes();
    if bytes.len() != 4 || &bytes[..3] != b"PRG" {
        return None;
    }
    let last = bytes[3] as char;
    last.to_digit(16).map(|d| d as usize)
}

/// Parse a `CHRx` chunk id where `x` is hex.
fn parse_chr_chunk_index(fourcc: &str) -> Option<usize> {
    let bytes = fourcc.as_bytes();
    if bytes.len() != 4 || &bytes[..3] != b"CHR" {
        return None;
    }
    let last = bytes[3] as char;
    last.to_digit(16).map(|d| d as usize)
}

fn resolve_mapper(name: &str) -> Option<u16> {
    BOARD_MAPPINGS
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| *v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_unif(mapper: &str, prg: &[u8], chr: &[u8], mirr: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(UNIF_MAGIC);
        buf.extend(std::iter::repeat_n(0, HEADER_LEN - 4));
        // MAPR chunk
        buf.extend_from_slice(b"MAPR");
        let mut mapr = mapper.as_bytes().to_vec();
        mapr.push(0);
        buf.extend_from_slice(&(mapr.len() as u32).to_le_bytes());
        buf.extend_from_slice(&mapr);
        // PRG0
        buf.extend_from_slice(b"PRG0");
        buf.extend_from_slice(&(prg.len() as u32).to_le_bytes());
        buf.extend_from_slice(prg);
        // CHR0 (optional)
        if !chr.is_empty() {
            buf.extend_from_slice(b"CHR0");
            buf.extend_from_slice(&(chr.len() as u32).to_le_bytes());
            buf.extend_from_slice(chr);
        }
        // MIRR
        buf.extend_from_slice(b"MIRR");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.push(mirr);
        buf
    }

    #[test]
    fn parses_minimal_nrom_unif() {
        let rom = build_unif("NROM-256", &[0xea; 32 * 1024], &[0; 8 * 1024], 1);
        let img = parse_unif(&rom).unwrap();
        assert_eq!(img.mapper_id, 0);
        assert_eq!(img.format, CartridgeFormat::Unif);
        assert_eq!(img.prg_rom.len(), 32 * 1024);
        assert_eq!(img.chr_rom.len(), 8 * 1024);
        assert_eq!(img.mirroring, Mirroring::Vertical);
        assert_eq!(img.chr_ram_size, 0);
    }

    #[test]
    fn unif_default_chr_ram_8k_when_chr_rom_absent() {
        let rom = build_unif("UNROM", &[0xea; 32 * 1024], &[], 0);
        let img = parse_unif(&rom).unwrap();
        assert_eq!(img.mapper_id, 2);
        assert_eq!(img.chr_rom.len(), 0);
        assert_eq!(img.chr_ram_size, 8 * 1024);
        assert!(img.chr_ram_unspecified);
    }

    #[test]
    fn unif_mapper_name_prefix_strip() {
        let rom = build_unif("NES-TLROM", &[0xea; 32 * 1024], &[0; 8 * 1024], 1);
        let img = parse_unif(&rom).unwrap();
        assert_eq!(img.mapper_id, 4);
    }

    #[test]
    fn unif_rejects_unknown_mapper() {
        let rom = build_unif("MADE-UP-BOARD", &[0xea; 8 * 1024], &[], 0);
        let err = parse_unif(&rom).unwrap_err().to_string();
        assert!(err.contains("unsupported board name"));
    }

    #[test]
    fn unif_rejects_truncated_header() {
        let err = parse_unif(b"UNIFstub").unwrap_err().to_string();
        assert!(err.contains("UNIF"));
    }
}
