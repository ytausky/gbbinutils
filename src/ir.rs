#[derive(Clone, Debug, PartialEq)]
pub enum AluOperand {
    A,
    B,
    #[cfg(test)] C,
    #[cfg(test)] D,
    #[cfg(test)] E,
    #[cfg(test)] H,
    #[cfg(test)] L,
    DerefHl,
}
