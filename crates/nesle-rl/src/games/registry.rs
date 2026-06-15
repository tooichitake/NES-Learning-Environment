use super::GameSpec;

/// Every game spec (single- and multi-player unified: a spec drives 1..=`players`
/// ports). Game order is stable for the metadata / asset audit.
static ALL_GAMES: [&GameSpec; 20] = [
    &super::supported::super_mario_bros::SUPER_MARIO_BROS,
    &super::supported::super_mario_bros_2::SUPER_MARIO_BROS_2,
    &super::supported::super_mario_bros_3::SUPER_MARIO_BROS_3,
    &super::supported::mario_bros::MARIO_BROS,
    &super::supported::kung_fu::KUNG_FU,
    &super::supported::castlevania::CASTLEVANIA,
    &super::supported::bomberman_2::BOMBERMAN_2_NORMAL,
    &super::supported::bomberman::BOMBERMAN,
    &super::supported::adventure_island::ADVENTURE_ISLAND,
    &super::supported::pac_man::PAC_MAN,
    &super::supported::duck_tales::DUCK_TALES,
    &super::supported::mega_man_2::MEGA_MAN_2,
    &super::supported::super_c::SUPER_C_1P,
    &super::supported::super_c::SUPER_C_2P,
    &super::supported::ice_hockey::ICE_HOCKEY_1P,
    &super::supported::ice_hockey::ICE_HOCKEY_2P,
    &super::supported::bomberman_2::BOMBERMAN_2_VS_2P,
    &super::supported::bomberman_2::BOMBERMAN_2_BATTLE_3P,
    &super::supported::r_c_pro_am_2::R_C_PRO_AM_2,
    &super::supported::roundball_2on2::ROUNDBALL_2ON2,
];

pub fn super_mario_bros() -> &'static GameSpec {
    &super::supported::super_mario_bros::SUPER_MARIO_BROS
}

pub fn super_mario_bros_2() -> &'static GameSpec {
    &super::supported::super_mario_bros_2::SUPER_MARIO_BROS_2
}

pub fn super_mario_bros_3() -> &'static GameSpec {
    &super::supported::super_mario_bros_3::SUPER_MARIO_BROS_3
}

pub fn mario_bros() -> &'static GameSpec {
    &super::supported::mario_bros::MARIO_BROS
}

pub fn kung_fu() -> &'static GameSpec {
    &super::supported::kung_fu::KUNG_FU
}

pub fn castlevania() -> &'static GameSpec {
    &super::supported::castlevania::CASTLEVANIA
}

pub fn bomberman_2_normal() -> &'static GameSpec {
    &super::supported::bomberman_2::BOMBERMAN_2_NORMAL
}

pub fn bomberman() -> &'static GameSpec {
    &super::supported::bomberman::BOMBERMAN
}

pub fn adventure_island() -> &'static GameSpec {
    &super::supported::adventure_island::ADVENTURE_ISLAND
}

pub fn pac_man() -> &'static GameSpec {
    &super::supported::pac_man::PAC_MAN
}

pub fn duck_tales() -> &'static GameSpec {
    &super::supported::duck_tales::DUCK_TALES
}

pub fn mega_man_2() -> &'static GameSpec {
    &super::supported::mega_man_2::MEGA_MAN_2
}

pub fn super_c_1p() -> &'static GameSpec {
    &super::supported::super_c::SUPER_C_1P
}

pub fn super_c_2p() -> &'static GameSpec {
    &super::supported::super_c::SUPER_C_2P
}

pub fn ice_hockey_1p() -> &'static GameSpec {
    &super::supported::ice_hockey::ICE_HOCKEY_1P
}

pub fn ice_hockey_2p() -> &'static GameSpec {
    &super::supported::ice_hockey::ICE_HOCKEY_2P
}

pub fn bomberman_2_vs_2p() -> &'static GameSpec {
    &super::supported::bomberman_2::BOMBERMAN_2_VS_2P
}

pub fn bomberman_2_battle_3p() -> &'static GameSpec {
    &super::supported::bomberman_2::BOMBERMAN_2_BATTLE_3P
}

pub fn r_c_pro_am_2() -> &'static GameSpec {
    &super::supported::r_c_pro_am_2::R_C_PRO_AM_2
}

pub fn roundball_2on2() -> &'static GameSpec {
    &super::supported::roundball_2on2::ROUNDBALL_2ON2
}

pub fn all_games() -> &'static [&'static GameSpec] {
    &ALL_GAMES
}

pub fn find_game(id: &str) -> Option<&'static GameSpec> {
    all_games()
        .iter()
        .copied()
        .find(|game| game.id == id || game.gym_id == id)
}
