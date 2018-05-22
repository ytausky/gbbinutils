#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
    Add,
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
    De,
    E,
    H,
    Hl,
    L,
    Nc,
    Nz,
    Sp,
    Z,
}
