#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mnemonic {
    Halt,
    Ld,
    Nop,
    Push,
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Register {
    A,
    B,
    #[cfg(test)]
    C,
    #[cfg(test)]
    D,
    #[cfg(test)]
    E,
    #[cfg(test)]
    H,
    #[cfg(test)]
    L,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegisterPair {
    Bc,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Keyword {
    A,
    B,
    Bc,
    Endm,
    Halt,
    Include,
    Ld,
    Macro,
    Nop,
    Push,
    Stop,
}
