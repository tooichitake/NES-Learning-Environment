use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum NesleError {
    InvalidRom(String),
    UnsupportedMapper(u16),
    InvalidState(String),
    Io(std::io::Error),
}

impl Display for NesleError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRom(msg) => write!(f, "invalid ROM: {msg}"),
            Self::UnsupportedMapper(id) => write!(f, "unsupported mapper: {id}"),
            Self::InvalidState(msg) => write!(f, "invalid state: {msg}"),
            Self::Io(err) => Display::fmt(err, f),
        }
    }
}

impl std::error::Error for NesleError {}

impl From<std::io::Error> for NesleError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, NesleError>;
