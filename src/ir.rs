#[derive(Debug, PartialEq)]
pub enum Instruction {
    Halt,
    LdAluAlu(AluOperand, AluOperand),
    Nop,
    Push(Reg16),
    Stop,
    Xor(AluOperand),
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
