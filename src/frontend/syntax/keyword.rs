#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
    Directive(DirectiveKeyword),
    Mnemonic(MnemonicKeyword),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DirectiveKeyword {
    Db,
    Dw,
    Include,
    Org,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MnemonicKeyword {
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
pub enum OperandKeyword {
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
