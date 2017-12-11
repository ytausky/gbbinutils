#[derive(Debug, PartialEq)]
pub enum AsmItem {
    #[cfg(test)]
    Instruction(Instruction),
}

#[derive(Debug, PartialEq)]
pub struct Instruction {
    pub mnemonic: String,
    pub operands: Vec<Operand>,
}

impl Instruction {
    #[cfg(test)]
    pub fn new(mnemonic: &str, operands: &[Operand]) -> Instruction {
        Instruction {
            mnemonic: mnemonic.to_owned(),
            operands: operands.iter().cloned().collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand {
    #[cfg(test)]
    Register(Register),
    #[cfg(test)]
    RegisterPair(RegisterPair),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Register {
    #[cfg(test)]
    A,
    #[cfg(test)]
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
    #[cfg(test)]
    Bc,
}

#[cfg(test)]
pub const A: Operand = Operand::Register(Register::A);
#[cfg(test)]
pub const B: Operand = Operand::Register(Register::B);
#[cfg(test)]
pub const C: Operand = Operand::Register(Register::C);
#[cfg(test)]
pub const D: Operand = Operand::Register(Register::D);
#[cfg(test)]
pub const E: Operand = Operand::Register(Register::E);
#[cfg(test)]
pub const H: Operand = Operand::Register(Register::H);
#[cfg(test)]
pub const L: Operand = Operand::Register(Register::L);
