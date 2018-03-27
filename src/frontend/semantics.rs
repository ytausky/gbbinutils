use diagnostics;

use ir::*;
use frontend::syntax::{Keyword, StrToken, SynExpr};

pub struct CommandAnalyzer;

impl CommandAnalyzer {
    pub fn new() -> CommandAnalyzer {
        CommandAnalyzer {}
    }

    pub fn analyze_instruction<'a, I>(&mut self, mnemonic: Keyword, operands: I) -> AnalysisResult
    where
        I: IntoIterator<Item = SynExpr<StrToken<'a>>>,
    {
        use self::Mnemonic::*;
        let mut operands = operands.into_iter().map(analyze_operand);
        match to_mnemonic(mnemonic) {
            Alu(operation) => analyze_alu_instruction(operation, operands),
            Dec => match operands.next() {
                Some(Operand::Simple(operand)) => Ok(Instruction::Dec(operand)),
                _ => panic!(),
            },
            Jr => analyze_jr_instruction(operands),
            Ld => analyze_ld(operands),
            Nullary(instruction) => analyze_nullary_instruction(instruction, operands),
            Push => match operands.next() {
                Some(Operand::Reg16(src)) => Ok(Instruction::Push(src)),
                _ => panic!(),
            },
        }
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

pub type AnalysisResult = Result<Instruction, diagnostics::Error>;

fn analyze_operand<'a>(expr: SynExpr<StrToken<'a>>) -> Operand {
    match expr {
        SynExpr::Atom(StrToken::Keyword(keyword)) => analyze_keyword_operand(keyword),
        SynExpr::Atom(StrToken::Identifier(ident)) => {
            Operand::Const(Expr::Symbol(ident.to_string()))
        }
        SynExpr::Atom(StrToken::Number(number)) => Operand::Const(Expr::Literal(number)),
        SynExpr::Deref(address_specifier) => analyze_deref_operand(*address_specifier),
        _ => panic!(),
    }
}

fn analyze_keyword_operand(keyword: Keyword) -> Operand {
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

fn analyze_deref_operand<'a>(addr: SynExpr<StrToken<'a>>) -> Operand {
    match addr {
        SynExpr::Atom(StrToken::Keyword(Keyword::Hl)) => Operand::Simple(SimpleOperand::DerefHl),
        SynExpr::Atom(StrToken::Identifier(ident)) => {
            Operand::Deref(Expr::Symbol(ident.to_string()))
        }
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

fn analyze_alu_instruction<I>(operation: AluOperation, mut operands: I) -> AnalysisResult
where
    I: Iterator<Item = Operand>,
{
    match operands.next() {
        Some(Operand::Simple(src)) => Ok(Instruction::Alu(operation, AluSource::Simple(src))),
        Some(Operand::Const(expr)) => Ok(Instruction::Alu(operation, AluSource::Immediate(expr))),
        _ => panic!(),
    }
}

fn analyze_jr_instruction<I: Iterator<Item = Operand>>(mut operands: I) -> AnalysisResult {
    match operands.next() {
        Some(Operand::Condition(condition)) => match operands.next() {
            Some(Operand::Const(expr)) => Ok(Instruction::Jr(Some(condition), expr)),
            _ => panic!(),
        },
        Some(Operand::Const(expr)) => Ok(Instruction::Jr(None, expr)),
        _ => panic!(),
    }
}

fn analyze_nullary_instruction<I: Iterator>(
    instruction: Instruction,
    operands: I,
) -> AnalysisResult {
    match operands.count() {
        0 => Ok(instruction),
        n => Err(diagnostics::Error::OperandCount(0, n)),
    }
}

fn analyze_ld<I: Iterator<Item = Operand>>(mut operands: I) -> AnalysisResult {
    let dest = operands.next().unwrap();
    let src = operands.next().unwrap();
    assert_eq!(operands.next(), None);
    match (dest, src) {
        (Operand::Simple(dest), Operand::Simple(src)) => {
            Ok(Instruction::Ld(LdKind::Simple(dest, src)))
        }
        (Operand::Simple(SimpleOperand::A), src) => analyze_ld_a(src, Direction::IntoA),
        (dest, Operand::Simple(SimpleOperand::A)) => analyze_ld_a(dest, Direction::FromA),
        _ => panic!(),
    }
}

fn analyze_ld_a(other: Operand, direction: Direction) -> AnalysisResult {
    match other {
        Operand::Deref(expr) => Ok(Instruction::Ld(LdKind::ImmediateAddr(expr, direction))),
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use self::Keyword::*;

    fn atom(keyword: Keyword) -> SynExpr<StrToken<'static>> {
        SynExpr::from(StrToken::Keyword(keyword))
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

    impl From<SimpleOperand> for SynExpr<StrToken<'static>> {
        fn from(alu_operand: SimpleOperand) -> Self {
            match alu_operand {
                SimpleOperand::A => atom(A),
                SimpleOperand::B => atom(B),
                SimpleOperand::C => atom(C),
                SimpleOperand::D => atom(D),
                SimpleOperand::E => atom(E),
                SimpleOperand::H => atom(H),
                SimpleOperand::L => atom(L),
                SimpleOperand::DerefHl => atom(Hl).deref(),
            }
        }
    }

    impl From<Condition> for SynExpr<StrToken<'static>> {
        fn from(condition: Condition) -> Self {
            match condition {
                Condition::Nz => atom(Nz),
                Condition::Z => atom(Z),
            }
        }
    }

    #[test]
    fn analyze_ld_deref_symbol_a() {
        let ident = "ident";
        assert_eq!(
            analyze(
                Keyword::Ld,
                vec![SynExpr::from(StrToken::Identifier(ident)).deref(), atom(A)]
            ),
            Ok(Instruction::Ld(LdKind::ImmediateAddr(
                Expr::Symbol(ident.to_string()),
                Direction::FromA
            )))
        )
    }

    #[test]
    fn analyze_ld_a_deref_symbol() {
        let ident = "ident";
        assert_eq!(
            analyze(
                Keyword::Ld,
                vec![atom(A), SynExpr::from(StrToken::Identifier(ident)).deref()]
            ),
            Ok(Instruction::Ld(LdKind::ImmediateAddr(
                Expr::Symbol(ident.to_string()),
                Direction::IntoA
            )))
        )
    }

    #[test]
    fn analyze_cp_symbol() {
        let ident = "ident";
        test_cp_const_analysis(StrToken::Identifier(ident), Expr::Symbol(ident.to_string()))
    }

    #[test]
    fn analyze_cp_literal() {
        let literal = 0x50;
        test_cp_const_analysis(StrToken::Number(literal), Expr::Literal(literal))
    }

    fn test_cp_const_analysis(atom: StrToken<'static>, expr: Expr) {
        assert_eq!(
            analyze(Keyword::Cp, Some(SynExpr::Atom(atom))),
            Ok(Instruction::Alu(
                AluOperation::Cp,
                AluSource::Immediate(expr)
            ))
        )
    }

    #[test]
    fn analyze_unconditional_jr() {
        let ident = "ident";
        assert_eq!(
            analyze(
                Keyword::Jr,
                Some(SynExpr::Atom(StrToken::Identifier(ident)))
            ),
            Ok(Instruction::Jr(None, Expr::Symbol(ident.to_string())))
        )
    }

    #[test]
    fn analyze_legal_instructions() {
        test_instruction_analysis(describe_legal_instructions());
    }

    type InstructionDescriptor = ((Keyword, Vec<SynExpr<StrToken<'static>>>), Instruction);

    fn describe_legal_instructions() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        descriptors.extend(describe_nullary_instructions());
        descriptors.extend(describe_ld_simple_instructions());
        descriptors.extend(describe_alu_simple_instructions());
        descriptors.extend(describe_jr_conditional_instuctions());
        descriptors.extend(describe_dec_instructions());
        descriptors.push((
            (Keyword::Push, vec![atom(Bc)]),
            Instruction::Push(Reg16::Bc),
        ));
        descriptors
    }

    fn describe_nullary_instructions() -> Vec<InstructionDescriptor> {
        vec![
            (Keyword::Halt, Instruction::Halt),
            (Keyword::Nop, Instruction::Nop),
            (Keyword::Stop, Instruction::Stop),
        ].into_iter()
            .map(|(mnemonic, instruction)| ((mnemonic, vec![]), instruction))
            .collect()
    }

    fn describe_ld_simple_instructions() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        for &dest in SIMPLE_OPERANDS.iter() {
            for &src in SIMPLE_OPERANDS.iter() {
                descriptors.extend(describe_ld_simple(dest, src))
            }
        }
        descriptors
    }

    fn describe_ld_simple(
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

    fn describe_alu_simple_instructions() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        for &operation in ALU_OPERATIONS.iter() {
            for &operand in SIMPLE_OPERANDS.iter() {
                descriptors.push(describe_alu_simple(operation, operand))
            }
        }
        descriptors
    }

    fn describe_alu_simple(
        operation: AluOperation,
        operand: SimpleOperand,
    ) -> InstructionDescriptor {
        (
            (Keyword::from(operation), vec![SynExpr::from(operand)]),
            Instruction::Alu(operation, AluSource::Simple(operand)),
        )
    }

    fn describe_jr_conditional_instuctions() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        for &condition in CONDITIONS.iter() {
            descriptors.push(describe_jr_conditional(condition))
        }
        descriptors
    }

    fn describe_jr_conditional(condition: Condition) -> InstructionDescriptor {
        let ident = "ident";
        (
            (
                Keyword::Jr,
                vec![
                    SynExpr::from(condition),
                    SynExpr::Atom(StrToken::Identifier(ident)),
                ],
            ),
            Instruction::Jr(Some(condition), Expr::Symbol(ident.to_string())),
        )
    }

    fn describe_dec_instructions() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        for &operand in SIMPLE_OPERANDS.iter() {
            descriptors.push(describe_dec(operand))
        }
        descriptors
    }

    fn describe_dec(operand: SimpleOperand) -> InstructionDescriptor {
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

    fn test_instruction_analysis(descriptors: Vec<InstructionDescriptor>) {
        for ((mnemonic, operands), expected) in descriptors {
            assert_eq!(analyze(mnemonic, operands), Ok(expected))
        }
    }

    fn analyze<I>(mnemonic: Keyword, operands: I) -> AnalysisResult
    where
        I: IntoIterator<Item = SynExpr<StrToken<'static>>>,
    {
        let mut analyzer = CommandAnalyzer::new();
        analyzer.analyze_instruction(mnemonic, operands)
    }

    use diagnostics;

    #[test]
    fn analyze_nop_a() {
        assert_eq!(
            analyze(Keyword::Nop, vec![atom(A)]),
            Err(diagnostics::Error::OperandCount(0, 1))
        )
    }
}
