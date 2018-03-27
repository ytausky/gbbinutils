use diagnostics;
use frontend::ast;

use frontend::ast::SynExpr;
use frontend::syntax::{Keyword, Token};

use ir::*;

pub fn reduce_include<'a>(mut arguments: Vec<SynExpr<Token<'a>>>) -> ast::AsmItem<'a> {
    assert_eq!(arguments.len(), 1);
    let path = arguments.pop().unwrap();
    match path {
        SynExpr::Atom(Token::QuotedString(path_str)) => include(path_str),
        _ => panic!(),
    }
}

#[derive(Debug, PartialEq)]
pub enum Operand {
    Simple(SimpleOperand),
    Condition(Condition),
    Const(Expr),
    Deref(Expr),
    Reg16(Reg16),
}

pub type InterpretationResult = Result<Instruction, diagnostics::Error>;

pub fn interpret_instruction<'a, I>(mnemonic: Keyword, operands: I) -> InterpretationResult
where
    I: IntoIterator<Item = SynExpr<Token<'a>>>,
{
    instruction(
        to_mnemonic(mnemonic),
        operands.into_iter().map(interpret_as_operand),
    )
}

fn interpret_as_operand<'a>(expr: SynExpr<Token<'a>>) -> Operand {
    match expr {
        SynExpr::Atom(Token::Keyword(keyword)) => interpret_as_keyword_operand(keyword),
        SynExpr::Atom(Token::Identifier(ident)) => Operand::Const(Expr::Symbol(ident.to_string())),
        SynExpr::Atom(Token::Number(number)) => Operand::Const(Expr::Literal(number)),
        SynExpr::Deref(address_specifier) => interpret_as_deref_operand(*address_specifier),
        _ => panic!(),
    }
}

fn interpret_as_keyword_operand(keyword: Keyword) -> Operand {
    match keyword {
        Keyword::A => Operand::Simple(SimpleOperand::A),
        Keyword::B => Operand::Simple(SimpleOperand::B),
        Keyword::Bc => Operand::Reg16(Reg16::Bc),
        Keyword::C => Operand::Simple(SimpleOperand::C),
        Keyword::D => Operand::Simple(SimpleOperand::D),
        Keyword::E => Operand::Simple(SimpleOperand::E),
        Keyword::H => Operand::Simple(SimpleOperand::H),
        Keyword::L => Operand::Simple(SimpleOperand::L),
        Keyword::Nz => Operand::Condition(Condition::Nz),
        Keyword::Z => Operand::Condition(Condition::Z),
        _ => panic!(),
    }
}

fn interpret_as_deref_operand<'a>(addr: SynExpr<Token<'a>>) -> Operand {
    match addr {
        SynExpr::Atom(Token::Keyword(Keyword::Hl)) => Operand::Simple(SimpleOperand::DerefHl),
        SynExpr::Atom(Token::Identifier(ident)) => Operand::Deref(Expr::Symbol(ident.to_string())),
        _ => panic!(),
    }
}

#[derive(Debug, PartialEq)]
enum Mnemonic {
    Alu(AluOperation),
    Dec,
    Jr,
    Ld,
    Nullary(Instruction),
    Push,
}

fn to_mnemonic(keyword: Keyword) -> Mnemonic {
    match keyword {
        Keyword::And => Mnemonic::Alu(AluOperation::And),
        Keyword::Cp => Mnemonic::Alu(AluOperation::Cp),
        Keyword::Dec => Mnemonic::Dec,
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

fn instruction<I>(mnemonic: Mnemonic, mut operands: I) -> InterpretationResult
where
    I: Iterator<Item = Operand>,
{
    use self::Mnemonic::*;
    match mnemonic {
        Alu(operation) => interpret_alu_instruction(operation, operands),
        Dec => match operands.next() {
            Some(Operand::Simple(operand)) => Ok(Instruction::Dec(operand)),
            _ => panic!(),
        },
        Jr => interpret_jr_instruction(operands),
        Ld => analyze_ld(operands),
        Nullary(instruction) => interpret_nullary_instruction(instruction, operands),
        Push => match operands.next() {
            Some(Operand::Reg16(src)) => Ok(Instruction::Push(src)),
            _ => panic!(),
        },
    }
}

fn interpret_alu_instruction<I>(operation: AluOperation, mut operands: I) -> InterpretationResult
where
    I: Iterator<Item = Operand>,
{
    match operands.next() {
        Some(Operand::Simple(src)) => Ok(Instruction::Alu(operation, AluSource::Simple(src))),
        Some(Operand::Const(expr)) => Ok(Instruction::Alu(operation, AluSource::Immediate(expr))),
        _ => panic!(),
    }
}

fn interpret_jr_instruction<I: Iterator<Item = Operand>>(mut operands: I) -> InterpretationResult {
    match operands.next() {
        Some(Operand::Condition(condition)) => match operands.next() {
            Some(Operand::Const(expr)) => Ok(Instruction::Jr(Some(condition), expr)),
            _ => panic!(),
        },
        Some(Operand::Const(expr)) => Ok(Instruction::Jr(None, expr)),
        _ => panic!(),
    }
}

fn interpret_nullary_instruction<I: Iterator>(
    instruction: Instruction,
    operands: I,
) -> InterpretationResult {
    match operands.count() {
        0 => Ok(instruction),
        n => Err(diagnostics::Error::OperandCount(0, n)),
    }
}

fn analyze_ld<I: Iterator<Item = Operand>>(mut operands: I) -> InterpretationResult {
    let dest = operands.next().unwrap();
    let src = operands.next().unwrap();
    assert_eq!(operands.next(), None);
    match (dest, src) {
        (Operand::Simple(dest), Operand::Simple(src)) => {
            Ok(Instruction::Ld(LdKind::Simple(dest, src)))
        }
        (Operand::Simple(SimpleOperand::A), src) => interpret_ld_a(src, Direction::IntoA),
        (dest, Operand::Simple(SimpleOperand::A)) => interpret_ld_a(dest, Direction::FromA),
        _ => panic!(),
    }
}

fn interpret_ld_a(other: Operand, direction: Direction) -> InterpretationResult {
    match other {
        Operand::Deref(expr) => Ok(Instruction::Ld(LdKind::ImmediateAddr(expr, direction))),
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

    fn atom(keyword: Keyword) -> SynExpr<Token<'static>> {
        SynExpr::Atom(Token::Keyword(keyword))
    }

    fn deref(expr: SynExpr<Token<'static>>) -> SynExpr<Token<'static>> {
        SynExpr::Deref(Box::new(expr))
    }

    impl From<AluOperation> for Keyword {
        fn from(alu_operation: AluOperation) -> Self {
            match alu_operation {
                AluOperation::And => Keyword::And,
                AluOperation::Cp => Keyword::Cp,
                AluOperation::Xor => Keyword::Xor,
            }
        }
    }

    impl From<SimpleOperand> for SynExpr<Token<'static>> {
        fn from(alu_operand: SimpleOperand) -> Self {
            match alu_operand {
                SimpleOperand::A => atom(A),
                SimpleOperand::B => atom(B),
                SimpleOperand::C => atom(C),
                SimpleOperand::D => atom(D),
                SimpleOperand::E => atom(E),
                SimpleOperand::H => atom(H),
                SimpleOperand::L => atom(L),
                SimpleOperand::DerefHl => deref(atom(Hl)),
            }
        }
    }

    impl From<Condition> for SynExpr<Token<'static>> {
        fn from(condition: Condition) -> Self {
            match condition {
                Condition::Nz => atom(Nz),
                Condition::Z => atom(Z),
            }
        }
    }

    #[test]
    fn interpret_ld_deref_symbol_a() {
        let ident = "ident";
        assert_eq!(
            interpret_instruction(
                Keyword::Ld,
                vec![deref(SynExpr::Atom(Token::Identifier(ident))), atom(A)]
            ),
            Ok(Instruction::Ld(LdKind::ImmediateAddr(
                Expr::Symbol(ident.to_string()),
                Direction::FromA
            )))
        )
    }

    #[test]
    fn interpret_ld_a_deref_symbol() {
        let ident = "ident";
        assert_eq!(
            interpret_instruction(
                Keyword::Ld,
                vec![atom(A), deref(SynExpr::Atom(Token::Identifier(ident)))]
            ),
            Ok(Instruction::Ld(LdKind::ImmediateAddr(
                Expr::Symbol(ident.to_string()),
                Direction::IntoA
            )))
        )
    }

    #[test]
    fn interpret_cp_symbol() {
        let ident = "ident";
        test_cp_const(Token::Identifier(ident), Expr::Symbol(ident.to_string()))
    }

    #[test]
    fn interpret_cp_literal() {
        let literal = 0x50;
        test_cp_const(Token::Number(literal), Expr::Literal(literal))
    }

    fn test_cp_const(atom: Token<'static>, expr: Expr) {
        assert_eq!(
            interpret_instruction(Keyword::Cp, Some(SynExpr::Atom(atom))),
            Ok(Instruction::Alu(
                AluOperation::Cp,
                AluSource::Immediate(expr)
            ))
        )
    }

    #[test]
    fn interpret_unconditional_jr() {
        let ident = "ident";
        assert_eq!(
            interpret_instruction(Keyword::Jr, Some(SynExpr::Atom(Token::Identifier(ident)))),
            Ok(Instruction::Jr(None, Expr::Symbol(ident.to_string())))
        )
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
        test_instruction_interpretation(generate_condition_jr_instruction_descriptors());
        test_instruction_interpretation(generate_dec_instruction_descriptors());
        test_instruction_interpretation(instructions)
    }

    type InstructionDescriptor = ((Keyword, Vec<SynExpr<Token<'static>>>), Instruction);

    fn generate_ld_instruction_descriptors() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        for &dest in SIMPLE_OPERANDS.iter() {
            for &src in SIMPLE_OPERANDS.iter() {
                descriptors.extend(generate_ld_alu_alu(dest, src))
            }
        }
        descriptors
    }

    fn generate_ld_alu_alu(
        dest: SimpleOperand,
        src: SimpleOperand,
    ) -> Option<InstructionDescriptor> {
        match (dest, src) {
            (SimpleOperand::DerefHl, SimpleOperand::DerefHl) => None,
            _ => Some((
                (Keyword::Ld, vec![SynExpr::from(dest), SynExpr::from(src)]),
                Instruction::Ld(LdKind::Simple(dest, src)),
            )),
        }
    }

    fn generate_alu_instruction_descriptors() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        for &operation in ALU_OPERATIONS.iter() {
            for &operand in SIMPLE_OPERANDS.iter() {
                descriptors.push(generate_simple_alu_instruction(operation, operand))
            }
        }
        descriptors
    }

    fn generate_simple_alu_instruction(
        operation: AluOperation,
        operand: SimpleOperand,
    ) -> InstructionDescriptor {
        (
            (Keyword::from(operation), vec![SynExpr::from(operand)]),
            Instruction::Alu(operation, AluSource::Simple(operand)),
        )
    }

    fn generate_condition_jr_instruction_descriptors() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        for &condition in CONDITIONS.iter() {
            descriptors.push(generate_condition_jr(condition))
        }
        descriptors
    }

    fn generate_condition_jr(condition: Condition) -> InstructionDescriptor {
        let ident = "ident";
        (
            (
                Keyword::Jr,
                vec![
                    SynExpr::from(condition),
                    SynExpr::Atom(Token::Identifier(ident)),
                ],
            ),
            Instruction::Jr(Some(condition), Expr::Symbol(ident.to_string())),
        )
    }

    fn generate_dec_instruction_descriptors() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        for &operand in SIMPLE_OPERANDS.iter() {
            descriptors.push(generate_dec(operand))
        }
        descriptors
    }

    fn generate_dec(operand: SimpleOperand) -> InstructionDescriptor {
        (
            (Keyword::Dec, vec![SynExpr::from(operand)]),
            Instruction::Dec(operand),
        )
    }

    const ALU_OPERATIONS: [AluOperation; 3] =
        [AluOperation::And, AluOperation::Cp, AluOperation::Xor];

    const SIMPLE_OPERANDS: [SimpleOperand; 8] = [
        SimpleOperand::A,
        SimpleOperand::B,
        SimpleOperand::C,
        SimpleOperand::D,
        SimpleOperand::E,
        SimpleOperand::H,
        SimpleOperand::L,
        SimpleOperand::DerefHl,
    ];

    const CONDITIONS: [Condition; 2] = [Condition::Nz, Condition::Z];

    fn test_instruction_interpretation<'a, DII, OII>(descriptors: DII)
    where
        DII: IntoIterator<Item = ((Keyword, OII), Instruction)>,
        OII: IntoIterator<Item = SynExpr<Token<'a>>>,
    {
        for ((mnemonic, operands), expected) in descriptors {
            assert_eq!(interpret_instruction(mnemonic, operands), Ok(expected))
        }
    }

    use diagnostics;

    #[test]
    fn test_nop_a() {
        assert_eq!(
            interpret_instruction(Keyword::Nop, vec![atom(A)]),
            Err(diagnostics::Error::OperandCount(0, 1))
        )
    }
}
