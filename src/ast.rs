pub trait Section {
    fn add_instruction(&mut self, instruction: Instruction);
    fn add_label(&mut self, label: &str);
}

#[derive(Clone, Debug, PartialEq)]
pub enum AsmItem<'a> {
    Include(&'a str),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Instruction {
    pub mnemonic: Mnemonic,
    pub operands: Vec<Operand>,
}

impl Instruction {
    pub fn new(mnemonic: Mnemonic, operands: &[Operand]) -> Instruction {
        Instruction {
            mnemonic: mnemonic,
            operands: operands.iter().cloned().collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand {
    Register(Register),
    RegisterPair(RegisterPair),
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

#[cfg(test)]
pub const BC: Operand = Operand::RegisterPair(RegisterPair::Bc);

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
