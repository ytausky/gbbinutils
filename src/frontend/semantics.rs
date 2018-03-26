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
    Const(Expr),
    Deref(Expr),
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
            Operand::Const(Expr::Symbol(ident.to_string()))
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
        Keyword::C => Operand::Alu(AluOperand::C),
        Keyword::D => Operand::Alu(AluOperand::D),
        Keyword::E => Operand::Alu(AluOperand::E),
        Keyword::H => Operand::Alu(AluOperand::H),
        Keyword::L => Operand::Alu(AluOperand::L),
        Keyword::Z => Operand::Condition(Condition::Z),
        _ => panic!(),
    }
}

fn interpret_as_deref_operand<'a>(addr: Expression<Token<'a>>) -> Operand {
    match addr {
        Expression::Atom(Token::Keyword(Keyword::Hl)) => Operand::Alu(AluOperand::DerefHl),
        Expression::Atom(Token::Identifier(ident)) => {
            Operand::Deref(Expr::Symbol(ident.to_string()))
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
        Keyword::Cp => Mnemonic::Alu(AluOperation::Cp),
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
            (Operand::Condition(condition), Operand::Const(expr)) => {
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
        Some(Operand::Const(expr)) => Instruction::AluImm8(operation, expr),
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
        Operand::Deref(expr) => Instruction::LdDerefImm16(expr, direction),
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
    fn interpret_legal_instructions() {
        let nullary_instructions = vec![
            (Keyword::Halt, Instruction::Halt),
            (Keyword::Nop, Instruction::Nop),
            (Keyword::Stop, Instruction::Stop),
        ].into_iter()
            .map(|(mnemonic, instruction)| ((mnemonic, vec![]), instruction));
        let instructions = vec![
            (
                (Keyword::Push, vec![atom(Bc)]),
                Instruction::Push(Reg16::Bc),
            ),
        ];
        test_instruction_interpretation(nullary_instructions);
        test_instruction_interpretation(generate_ld_instruction_descriptors());
        test_instruction_interpretation(generate_alu_instruction_descriptors());
        test_instruction_interpretation(instructions)
    }

    type InstructionDescriptor = ((Keyword, Vec<Expression<Token<'static>>>), Instruction);

    fn generate_ld_instruction_descriptors() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        for (dest_expr, dest_operand) in alu_operands() {
            for (src_expr, src_operand) in alu_operands() {
                match (dest_operand, src_operand) {
                    (AluOperand::DerefHl, AluOperand::DerefHl) => (),
                    _ => descriptors.push((
                        (Keyword::Ld, vec![dest_expr.clone(), src_expr]),
                        Instruction::LdAluAlu(dest_operand, src_operand),
                    )),
                }
            }
        }
        descriptors
    }

    fn generate_alu_instruction_descriptors() -> Vec<InstructionDescriptor> {
        let alu_operations = [
            (Keyword::And, AluOperation::And),
            (Keyword::Cp, AluOperation::Cp),
            (Keyword::Xor, AluOperation::Xor),
        ];
        let mut descriptors = Vec::new();
        for &(mnemonic, operation) in alu_operations.iter() {
            for (expr, operand) in alu_operands() {
                descriptors.push(((mnemonic, vec![expr]), Instruction::Alu(operation, operand)))
            }
        }
        descriptors
    }

    fn alu_operands() -> Vec<(Expression<Token<'static>>, AluOperand)> {
        vec![
            (atom(A), AluOperand::A),
            (atom(B), AluOperand::B),
            (atom(C), AluOperand::C),
            (atom(D), AluOperand::D),
            (atom(E), AluOperand::E),
            (atom(H), AluOperand::H),
            (atom(L), AluOperand::L),
            (deref(atom(Hl)), AluOperand::DerefHl),
        ]
    }

    fn test_instruction_interpretation<'a, DII, OII>(descriptors: DII)
    where
        DII: IntoIterator<Item = ((Keyword, OII), Instruction)>,
        OII: IntoIterator<Item = Expression<Token<'a>>>,
    {
        for ((mnemonic, operands), expected) in descriptors {
            assert_eq!(interpret_instruction(mnemonic, operands), expected)
        }
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

    #[test]
    fn interpret_cp_const() {
        let ident = "ident";
        assert_eq!(
            interpret_instruction(
                Keyword::Cp,
                Some(Expression::Atom(Token::Identifier(ident)))
            ),
            Instruction::AluImm8(AluOperation::Cp, Expr::Symbol(ident.to_string()))
        )
    }
}
