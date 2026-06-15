#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum AutoresetMode {
    #[default]
    NextStep,
    SameStep,
}
