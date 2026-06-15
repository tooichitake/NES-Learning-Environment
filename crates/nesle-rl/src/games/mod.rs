pub mod registry;
pub mod spec;
pub mod supported;

pub use spec::{
    solo_lives, solo_reward, GameSpec, LivesFn, MultiPlayerValues, Ram, RewardFn, TerminalFn,
    TransitionFn,
};
