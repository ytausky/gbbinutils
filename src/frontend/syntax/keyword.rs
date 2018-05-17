#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
    And,
    Charmap,
    Cp,
    Db,
    Dec,
    Dw,
    Halt,
    Include,
    Jp,
    Jr,
    Ld,
    Nop,
    Push,
    Stop,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand {
    A,
    B,
    Bc,
    C,
    D,
    E,
    H,
    Hl,
    L,
    Nc,
    Nz,
    Z,
}
