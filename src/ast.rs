use ir;

use std::marker::PhantomData;

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

#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Alu(ir::AluOperand),
    RegisterPair(RegisterPair),
}

#[cfg(test)]
pub const A: Operand = Operand::Alu(ir::AluOperand::A);
#[cfg(test)]
pub const B: Operand = Operand::Alu(ir::AluOperand::B);
#[cfg(test)]
pub const C: Operand = Operand::Alu(ir::AluOperand::C);
#[cfg(test)]
pub const D: Operand = Operand::Alu(ir::AluOperand::D);
#[cfg(test)]
pub const E: Operand = Operand::Alu(ir::AluOperand::E);
#[cfg(test)]
pub const H: Operand = Operand::Alu(ir::AluOperand::H);
#[cfg(test)]
pub const L: Operand = Operand::Alu(ir::AluOperand::L);

#[cfg(test)]
pub const BC: Operand = Operand::RegisterPair(RegisterPair::Bc);

#[derive(Clone, Debug, PartialEq)]
pub enum Mnemonic {
    Halt,
    Ld,
    Nop,
    Push,
    Stop,
    Xor,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RegisterPair {
    Bc,
}

#[derive(Debug, PartialEq)]
pub enum Expression<T> {
    Atom(T),
    Deref(Box<Expression<T>>),
}

pub trait ExprFactory {
    type Terminal;
    type Expr;
    fn from_atom(&mut self, atom: Self::Terminal) -> Self::Expr;
    fn apply_deref(&mut self, expr: Self::Expr) -> Self::Expr;
}

pub struct ExprBuilder<T>(PhantomData<T>);

impl<T> ExprBuilder<T> {
    pub fn new() -> ExprBuilder<T> {
        ExprBuilder(PhantomData)
    }
}

impl<T> ExprFactory for ExprBuilder<T> {
    type Terminal = T;
    type Expr = Expression<Self::Terminal>;

    fn from_atom(&mut self, atom: Self::Terminal) -> Self::Expr {
        Expression::Atom(atom)
    }

    fn apply_deref(&mut self, expr: Self::Expr) -> Self::Expr {
        Expression::Deref(Box::new(expr))
    }
}
