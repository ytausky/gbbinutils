#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Alu(AluOperand),
    Reg16(Reg16),
}

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

#[derive(Clone, Debug, PartialEq)]
pub enum Reg16 {
    Bc,
}
