use frontend::ast;

use frontend::ast::Expression;
use frontend::syntax::{Keyword, Token};

use ir::*;

pub fn reduce_include<'a>(mut arguments: Vec<Expression<Token<'a>>>) -> ast::AsmItem<'a> {
    assert_eq!(arguments.len(), 1);
    let path = arguments.pop().unwrap();
    match path {
        Expression::Atom(Token::QuotedString(path_str)) => include(path_str),
        _ => panic!(),
    }
}

#[derive(Debug, PartialEq)]
pub enum Operand {
    Alu(AluOperand),
    Condition(Condition),
    DerefImm16(Expr),
    Imm16(Expr),
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
        Expression::Atom(Token::Identifier(ident)) => {
            Operand::Imm16(Expr::Symbol(ident.to_string()))
        }
        Expression::Deref(address_specifier) => interpret_as_deref_operand(*address_specifier),
        _ => panic!(),
    }
}

fn interpret_as_keyword_operand(keyword: Keyword) -> Operand {
    match keyword {
        Keyword::A => Operand::Alu(AluOperand::A),
        Keyword::B => Operand::Alu(AluOperand::B),
        Keyword::Bc => Operand::Reg16(Reg16::Bc),
        Keyword::Z => Operand::Condition(Condition::Z),
        _ => panic!(),
    }
}

fn interpret_as_deref_operand<'a>(addr: Expression<Token<'a>>) -> Operand {
    match addr {
        Expression::Atom(Token::Keyword(Keyword::Hl)) => Operand::Alu(AluOperand::DerefHl),
        Expression::Atom(Token::Identifier(ident)) => {
            Operand::DerefImm16(Expr::Symbol(ident.to_string()))
        }
        _ => panic!(),
    }
}

#[derive(Debug, PartialEq)]
enum Mnemonic {
    Alu(AluOperation),
    Jr,
    Ld,
    Nullary(Instruction),
    Push,
}

fn to_mnemonic(keyword: Keyword) -> Mnemonic {
    match keyword {
        Keyword::And => Mnemonic::Alu(AluOperation::And),
        Keyword::Halt => Mnemonic::Nullary(Instruction::Halt),
        Keyword::Jr => Mnemonic::Jr,
        Keyword::Ld => Mnemonic::Ld,
        Keyword::Nop => Mnemonic::Nullary(Instruction::Nop),
        Keyword::Push => Mnemonic::Push,
        Keyword::Stop => Mnemonic::Nullary(Instruction::Stop),
        Keyword::Xor => Mnemonic::Alu(AluOperation::Xor),
        _ => panic!(),
    }
}

fn instruction<I>(mnemonic: Mnemonic, mut operands: I) -> Instruction
where
    I: Iterator<Item = Operand>,
{
    use self::Mnemonic::*;
    match mnemonic {
        Alu(operation) => interpret_alu_instruction(operation, operands),
        Jr => match (operands.next().unwrap(), operands.next().unwrap()) {
            (Operand::Condition(condition), Operand::Imm16(expr)) => {
                Instruction::Jr(condition, expr)
            }
            _ => panic!(),
        },
        Ld => analyze_ld(operands),
        Nullary(instruction) => instruction,
        Push => match operands.next() {
            Some(Operand::Reg16(src)) => Instruction::Push(src),
            _ => panic!(),
        },
    }
}

fn interpret_alu_instruction<I>(operation: AluOperation, mut operands: I) -> Instruction
where
    I: Iterator<Item = Operand>,
{
    match operands.next() {
        Some(Operand::Alu(src)) => Instruction::Alu(operation, src),
        _ => panic!(),
    }
}

fn analyze_ld<I: Iterator<Item = Operand>>(mut operands: I) -> Instruction {
    let dest = operands.next().unwrap();
    let src = operands.next().unwrap();
    assert_eq!(operands.next(), None);
    match (dest, src) {
        (Operand::Alu(dest), Operand::Alu(src)) => Instruction::LdAluAlu(dest, src),
        (Operand::Alu(AluOperand::A), src) => interpret_ld_a(src, Direction::IntoA),
        (dest, Operand::Alu(AluOperand::A)) => interpret_ld_a(dest, Direction::FromA),
        _ => panic!(),
    }
}

fn interpret_ld_a(other: Operand, direction: Direction) -> Instruction {
    match other {
        Operand::DerefImm16(expr) => Instruction::LdDerefImm16(expr, direction),
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
    fn interpret_and_a() {
        assert_eq!(
            interpret_instruction(Keyword::And, Some(atom(A))),
            Instruction::Alu(AluOperation::And, AluOperand::A)
        )
    }

    #[test]
    fn interpret_xor_a() {
        assert_eq!(
            interpret_instruction(Keyword::Xor, Some(atom(A))),
            Instruction::Alu(AluOperation::Xor, AluOperand::A)
        )
    }

    #[test]
    fn interpret_xor_deref_hl() {
        assert_eq!(
            interpret_instruction(Keyword::Xor, Some(deref(atom(Hl)))),
            Instruction::Alu(AluOperation::Xor, AluOperand::DerefHl)
        )
    }

    #[test]
    fn interpret_ld_deref_symbol_a() {
        let ident = "ident";
        assert_eq!(
            interpret_instruction(
                Keyword::Ld,
                vec![deref(Expression::Atom(Token::Identifier(ident))), atom(A)]
            ),
            Instruction::LdDerefImm16(Expr::Symbol(ident.to_string()), Direction::FromA)
        )
    }

    #[test]
    fn interpret_ld_a_deref_symbol() {
        let ident = "ident";
        assert_eq!(
            interpret_instruction(
                Keyword::Ld,
                vec![atom(A), deref(Expression::Atom(Token::Identifier(ident)))]
            ),
            Instruction::LdDerefImm16(Expr::Symbol(ident.to_string()), Direction::IntoA)
        )
    }

    #[test]
    fn interpret_jr_z_symbol() {
        let ident = "ident";
        assert_eq!(
            interpret_instruction(
                Keyword::Jr,
                vec![atom(Z), Expression::Atom(Token::Identifier(ident))]
            ),
            Instruction::Jr(Condition::Z, Expr::Symbol(ident.to_string()))
        )
    }
}
