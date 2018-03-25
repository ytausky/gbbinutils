use ir::*;

use std::marker::PhantomData;

pub trait Section {
    fn add_instruction(&mut self, instruction: Instruction);
    fn add_label(&mut self, label: &str);
}

#[derive(Clone, Debug, PartialEq)]
pub enum AsmItem<'a> {
    Include(&'a str),
}

#[cfg(test)]
pub const A: Operand = Operand::Alu(AluOperand::A);
#[cfg(test)]
pub const B: Operand = Operand::Alu(AluOperand::B);

#[cfg(test)]
pub const BC: Operand = Operand::Reg16(Reg16::Bc);

#[derive(Clone, Debug, PartialEq)]
pub enum Mnemonic {
    Halt,
    Ld,
    Nop,
    Push,
    Stop,
    Xor,
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
