use frontend::ast;

use frontend::ast::Expression;
use frontend::syntax::{Keyword, Token};

use ir::{AluOperand, Reg16, Instruction};

pub fn reduce_include<'a>(mut arguments: Vec<Expression<Token<'a>>>) -> ast::AsmItem<'a> {
    assert_eq!(arguments.len(), 1);
    let path = arguments.pop().unwrap();
    match path {
        Expression::Atom(Token::QuotedString(path_str)) => include(path_str),
        _ => panic!(),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Alu(AluOperand),
    Reg16(Reg16),
}

pub fn interpret_instruction<'a, I>(mnemonic: Keyword, operands: I) -> Instruction
where
    I: Iterator<Item = Expression<Token<'a>>>,
{
    instruction(to_mnemonic(mnemonic), operands.map(interpret_as_operand))
}

fn interpret_as_operand<'a>(expr: Expression<Token<'a>>) -> Operand {
    match expr {
        Expression::Atom(Token::Keyword(keyword)) => interpret_as_keyword_operand(keyword),
        Expression::Deref(address_specifier) => interpret_as_deref_operand(*address_specifier),
        _ => panic!(),
    }
}

fn interpret_as_keyword_operand(keyword: Keyword) -> Operand {
    match keyword {
        Keyword::A => Operand::Alu(AluOperand::A),
        Keyword::B => Operand::Alu(AluOperand::B),
        Keyword::Bc => Operand::Reg16(Reg16::Bc),
        _ => panic!(),
    }
}

fn interpret_as_deref_operand<'a>(addr: Expression<Token<'a>>) -> Operand {
    match addr {
        Expression::Atom(Token::Keyword(Keyword::Hl)) => Operand::Alu(AluOperand::DerefHl),
        _ => panic!(),
    }
}

#[derive(Debug, PartialEq)]
pub enum Mnemonic {
    Halt,
    Ld,
    Nop,
    Push,
    Stop,
    Xor,
}

fn to_mnemonic(keyword: Keyword) -> Mnemonic {
    match keyword {
        Keyword::Halt => Mnemonic::Halt,
        Keyword::Ld => Mnemonic::Ld,
        Keyword::Nop => Mnemonic::Nop,
        Keyword::Push => Mnemonic::Push,
        Keyword::Stop => Mnemonic::Stop,
        Keyword::Xor => Mnemonic::Xor,
        _ => panic!(),
    }
}

pub fn instruction<I>(mnemonic: Mnemonic, mut operands: I) -> Instruction
where
    I: Iterator<Item = Operand>,
{
    use self::Mnemonic::*;
    match mnemonic {
        Halt => Instruction::Halt,
        Ld => analyze_ld(operands),
        Nop => Instruction::Nop,
        Push => match operands.next() {
            Some(Operand::Reg16(src)) => Instruction::Push(src),
            _ => panic!(),
        },
        Stop => Instruction::Stop,
        Xor => match operands.next() {
            Some(Operand::Alu(src)) => Instruction::Xor(src),
            _ => panic!(),
        },
    }
}

fn analyze_ld<I: Iterator<Item = Operand>>(mut operands: I) -> Instruction {
    let dest = operands.next().unwrap();
    let src = operands.next().unwrap();
    assert_eq!(operands.next(), None);
    match (dest, src) {
        (Operand::Alu(dest), Operand::Alu(src)) => Instruction::LdAluAlu(dest, src),
        _ => panic!(),
    }
}

pub fn include(path: &str) -> ast::AsmItem {
    ast::AsmItem::Include(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyze_push_bc() {
        let actual = interpret_instruction(Keyword::Push, vec![Expression::Atom(Token::Keyword(Keyword::Bc))].into_iter());
        assert_eq!(actual, Instruction::Push(Reg16::Bc))
    }
}
