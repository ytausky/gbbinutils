#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
    Directive(Directive),
    Mnemonic(Mnemonic),
}

impl From<Directive> for Command {
    fn from(directive: Directive) -> Self {
        Command::Directive(directive)
    }
}

impl From<Mnemonic> for Command {
    fn from(mnemonic: Mnemonic) -> Self {
        Command::Mnemonic(mnemonic)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Directive {
    Db,
    Ds,
    Dw,
    Include,
    Org,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mnemonic {
    Adc,
    Add,
    And,
    Bit,
    Call,
    Cp,
    Cpl,
    Daa,
    Dec,
    Di,
    Ei,
    Halt,
    Inc,
    Jp,
    Jr,
    Ld,
    Ldhl,
    Nop,
    Or,
    Pop,
    Push,
    Res,
    Ret,
    Reti,
    Rl,
    Rla,
    Rlc,
    Rlca,
    Rr,
    Rra,
    Rrc,
    Rrca,
    Rst,
    Sbc,
    Set,
    Sla,
    Sra,
    Srl,
    Stop,
    Sub,
    Swap,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand {
    A,
    Af,
    B,
    Bc,
    C,
    D,
    De,
    E,
    H,
    Hl,
    Hld,
    Hli,
    L,
    Nc,
    Nz,
    Sp,
    Z,
}
