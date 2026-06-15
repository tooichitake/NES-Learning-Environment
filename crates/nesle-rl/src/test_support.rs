use std::fs;
use std::path::{Path, PathBuf};

use crate::games::GameSpec;
use crate::NesEnv;

pub(crate) fn rom_path(name: &str) -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root
        .join("crates")
        .join("nesle-py")
        .join("python")
        .join("nesle")
        .join("roms")
        .join(name);
    path.exists().then_some(path)
}

pub(crate) fn load_test_rom() -> Vec<u8> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("crates/nesle-py/python/nesle/roms/super-mario-bros.nes");
    fs::read(&path)
        .unwrap_or_else(|_| panic!("test ROM must be present at {}", path.display()))
}

pub(crate) fn action_mask(game: &GameSpec, name: &str) -> u8 {
    game.minimal_actions
        .iter()
        .find(|action| action.name == name)
        .unwrap_or_else(|| panic!("{} missing action {name}", game.id))
        .mask
}

pub(crate) fn smoke_single_player_actions(
    game: &'static GameSpec,
    rom_name: &str,
    actions: &[&str],
    after_reset: impl FnOnce(&NesEnv),
) {
    let Some(path) = rom_path(rom_name) else {
        return;
    };
    let rom = fs::read(path).unwrap();
    let mut env = NesEnv::new(game);
    env.set_players(1).unwrap();
    env.set_action_repeat(2, 0.0).unwrap();
    env.load_rom_bytes(&rom).unwrap();
    env.reset().unwrap();
    after_reset(&env);
    for &name in actions {
        let outcome = env.step(&[action_mask(game, name)]).unwrap();
        assert!(
            outcome.info.episode_frame_number > 0,
            "{} action {name} did not advance",
            game.id
        );
    }
}

pub(crate) fn smoke_multiplayer_actions(
    game: &'static GameSpec,
    rom_name: &str,
    actions: &[&str],
) {
    let Some(path) = rom_path(rom_name) else {
        return;
    };
    let rom = fs::read(path).unwrap();
    let mut env = NesEnv::new(game);
    env.set_action_repeat(2, 0.0).unwrap();
    env.load_rom_bytes(&rom).unwrap();
    env.reset().unwrap();
    for &name in actions {
        let mask = action_mask(game, name);
        let masks = vec![mask; game.players as usize];
        let outcome = env.step(&masks).unwrap();
        assert!(
            outcome.info.episode_frame_number > 0,
            "{} action {name} did not advance",
            game.id
        );
    }
}
