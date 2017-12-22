use keyword;

#[derive(Debug, PartialEq)]
pub enum AsmItem<'a> {
    Include(&'a str),
    Instruction(Instruction),
}

#[derive(Debug, PartialEq)]
pub struct Instruction {
    pub mnemonic: keyword::Mnemonic,
    pub operands: Vec<Operand>,
}

impl Instruction {
    pub fn new(mnemonic: keyword::Mnemonic, operands: &[Operand]) -> Instruction {
        Instruction {
            mnemonic: mnemonic,
            operands: operands.iter().cloned().collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand {
    Register(keyword::Register),
    RegisterPair(keyword::RegisterPair),
}

#[cfg(test)]
pub const A: Operand = Operand::Register(keyword::Register::A);
#[cfg(test)]
pub const B: Operand = Operand::Register(keyword::Register::B);
#[cfg(test)]
pub const C: Operand = Operand::Register(keyword::Register::C);
#[cfg(test)]
pub const D: Operand = Operand::Register(keyword::Register::D);
#[cfg(test)]
pub const E: Operand = Operand::Register(keyword::Register::E);
#[cfg(test)]
pub const H: Operand = Operand::Register(keyword::Register::H);
#[cfg(test)]
pub const L: Operand = Operand::Register(keyword::Register::L);

#[cfg(test)]
pub const BC: Operand = Operand::RegisterPair(keyword::RegisterPair::Bc);
