use frontend::ast;

use frontend::ast::Expression;
use frontend::syntax::{Keyword, Token};

use ir::{AluOperand, Instruction, Reg16};

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
    I: IntoIterator<Item = Expression<Token<'a>>>,
{
    instruction(
        to_mnemonic(mnemonic),
        operands.into_iter().map(interpret_as_operand),
    )
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

    use self::Keyword::*;

    fn atom<'a>(keyword: Keyword) -> Expression<Token<'a>> {
        Expression::Atom(Token::Keyword(keyword))
    }

    fn deref<'a>(expr: Expression<Token<'a>>) -> Expression<Token<'a>> {
        Expression::Deref(Box::new(expr))
    }

    #[test]
    fn interpret_push_bc() {
        assert_eq!(
            interpret_instruction(Keyword::Push, Some(atom(Bc))),
            Instruction::Push(Reg16::Bc)
        )
    }

    #[test]
    fn interpret_nop() {
        interpret_nullary_instruction(Keyword::Nop, Instruction::Nop)
    }

    #[test]
    fn interpret_halt() {
        interpret_nullary_instruction(Keyword::Halt, Instruction::Halt)
    }

    #[test]
    fn interpret_stop() {
        interpret_nullary_instruction(Keyword::Stop, Instruction::Stop)
    }

    fn interpret_nullary_instruction(mnemonic: Keyword, expected: Instruction) {
        assert_eq!(interpret_instruction(mnemonic, None), expected)
    }

    #[test]
    fn interpret_ld_a_a() {
        assert_eq!(
            interpret_instruction(Keyword::Ld, vec![atom(A), atom(A)]),
            Instruction::LdAluAlu(AluOperand::A, AluOperand::A)
        )
    }

    #[test]
    fn interpret_ld_a_b() {
        assert_eq!(
            interpret_instruction(Keyword::Ld, vec![atom(A), atom(B)]),
            Instruction::LdAluAlu(AluOperand::A, AluOperand::B)
        )
    }

    #[test]
    fn interpret_xor_a() {
        assert_eq!(
            interpret_instruction(Keyword::Xor, Some(atom(A))),
            Instruction::Xor(AluOperand::A)
        )
    }

    #[test]
    fn interpret_xor_deref_hl() {
        assert_eq!(
            interpret_instruction(Keyword::Xor, Some(deref(atom(Hl)))),
            Instruction::Xor(AluOperand::DerefHl)
        )
    }
}
