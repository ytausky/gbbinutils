#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mnemonic {
    Halt,
    Include,
    Ld,
    Nop,
    Push,
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Register {
    A,
    B,
    C,
    D,
    E,
    H,
    L,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegisterPair {
    Bc,
}
