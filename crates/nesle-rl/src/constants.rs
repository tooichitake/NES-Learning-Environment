pub const RAM_SIZE: usize = 0x800;
pub const NES_WIDTH: usize = 256;
pub const NES_HEIGHT: usize = 240;
pub const GRAY_FRAME_LEN: usize = NES_WIDTH * NES_HEIGHT;
pub const RGB_FRAME_LEN: usize = GRAY_FRAME_LEN * 3;
pub const MAX_PLAYERS: usize = 4;
