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
    Alu(AluOperand),
    RegisterPair(RegisterPair),
}

#[derive(Clone, Copy, Debug, PartialEq)]
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

#[cfg(test)]
pub const A: Operand = Operand::Alu(AluOperand::A);
#[cfg(test)]
pub const B: Operand = Operand::Alu(AluOperand::B);
#[cfg(test)]
pub const C: Operand = Operand::Alu(AluOperand::C);
#[cfg(test)]
pub const D: Operand = Operand::Alu(AluOperand::D);
#[cfg(test)]
pub const E: Operand = Operand::Alu(AluOperand::E);
#[cfg(test)]
pub const H: Operand = Operand::Alu(AluOperand::H);
#[cfg(test)]
pub const L: Operand = Operand::Alu(AluOperand::L);

#[cfg(test)]
pub const BC: Operand = Operand::RegisterPair(RegisterPair::Bc);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mnemonic {
    Halt,
    Ld,
    Nop,
    Push,
    Stop,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegisterPair {
    Bc,
}
