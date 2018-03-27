use diagnostics;

use ir::*;
use frontend::ExprFactory;
use frontend::syntax::{Keyword, SynExpr, Token, TokenKind};

struct OperandAnalyzer<EF> {
    expr_factory: EF,
}

impl<EF: ExprFactory> OperandAnalyzer<EF> {
    fn new(expr_factory: EF) -> OperandAnalyzer<EF> {
        OperandAnalyzer { expr_factory }
    }

    fn analyze_operand(&mut self, expr: SynExpr<EF::Token>) -> Operand {
        match expr {
            SynExpr::Atom(token) => self.analyze_atom_operand(token),
            SynExpr::Deref(expr) => self.analyze_deref_operand(*expr),
        }
    }

    fn analyze_atom_operand(&mut self, token: EF::Token) -> Operand {
        match token.kind() {
            TokenKind::Keyword(keyword) => analyze_keyword_operand(keyword),
            TokenKind::Identifier | TokenKind::Number => {
                Operand::Const(self.expr_factory.mk_atom(token))
            }
            _ => panic!(),
        }
    }

    fn analyze_deref_operand(&mut self, expr: SynExpr<EF::Token>) -> Operand {
        if let SynExpr::Atom(token) = expr {
            match token.kind() {
                TokenKind::Keyword(Keyword::Hl) => Operand::Simple(SimpleOperand::DerefHl),
                TokenKind::Identifier => Operand::Deref(self.expr_factory.mk_atom(token)),
                _ => panic!(),
            }
        } else {
            panic!()
        }
    }
}

pub struct CommandAnalyzer<EF> {
    operand_analyzer: OperandAnalyzer<EF>,
}

impl<EF: ExprFactory> CommandAnalyzer<EF> {
    pub fn new(expr_factory: EF) -> CommandAnalyzer<EF> {
        CommandAnalyzer {
            operand_analyzer: OperandAnalyzer::new(expr_factory),
        }
    }

    pub fn analyze_instruction<I>(&mut self, mnemonic: Keyword, operands: I) -> AnalysisResult
    where
        I: IntoIterator<Item = SynExpr<EF::Token>>,
    {
        Analysis::new(
            mnemonic,
            operands
                .into_iter()
                .map(|x| self.operand_analyzer.analyze_operand(x)),
        ).run()
    }
}

struct Analysis<I> {
    mnemonic: Keyword,
    operands: I,
}

impl<'a, I: Iterator<Item = Operand>> Analysis<I> {
    fn new(mnemonic: Keyword, operands: I) -> Analysis<I> {
        Analysis { mnemonic, operands }
    }

    fn run(mut self) -> AnalysisResult {
        use self::Mnemonic::*;
        match to_mnemonic(self.mnemonic) {
            Alu(operation) => self.analyze_alu_instruction(operation),
            Dec => match self.operands.next() {
                Some(Operand::Simple(operand)) => Ok(Instruction::Dec(operand)),
                _ => panic!(),
            },
            Jr => self.analyze_jr_instruction(),
            Ld => self.analyze_ld(),
            Nullary(instruction) => self.analyze_nullary_instruction(instruction),
            Push => match self.operands.next() {
                Some(Operand::Reg16(src)) => Ok(Instruction::Push(src)),
                _ => panic!(),
            },
        }
    }

    fn analyze_alu_instruction(&mut self, operation: AluOperation) -> AnalysisResult {
        match self.operands.next() {
            Some(Operand::Simple(src)) => Ok(Instruction::Alu(operation, AluSource::Simple(src))),
            Some(Operand::Const(expr)) => {
                Ok(Instruction::Alu(operation, AluSource::Immediate(expr)))
            }
            _ => panic!(),
        }
    }

    fn analyze_jr_instruction(&mut self) -> AnalysisResult {
        match self.operands.next() {
            Some(Operand::Condition(condition)) => match self.operands.next() {
                Some(Operand::Const(expr)) => Ok(Instruction::Jr(Some(condition), expr)),
                _ => panic!(),
            },
            Some(Operand::Const(expr)) => Ok(Instruction::Jr(None, expr)),
            _ => panic!(),
        }
    }

    fn analyze_nullary_instruction(&mut self, instruction: Instruction) -> AnalysisResult {
        match self.operands.by_ref().count() {
            0 => Ok(instruction),
            n => Err(diagnostics::Error::OperandCount(0, n)),
        }
    }

    fn analyze_ld(&mut self) -> AnalysisResult {
        let dest = self.operands.next().unwrap();
        let src = self.operands.next().unwrap();
        assert_eq!(self.operands.next(), None);
        match (dest, src) {
            (Operand::Simple(dest), Operand::Simple(src)) => {
                Ok(Instruction::Ld(LdKind::Simple(dest, src)))
            }
            (Operand::Simple(SimpleOperand::A), src) => analyze_ld_a(src, Direction::IntoA),
            (dest, Operand::Simple(SimpleOperand::A)) => analyze_ld_a(dest, Direction::FromA),
            _ => panic!(),
        }
    }
}

fn analyze_ld_a(other: Operand, direction: Direction) -> AnalysisResult {
    match other {
        Operand::Deref(expr) => Ok(Instruction::Ld(LdKind::ImmediateAddr(expr, direction))),
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

pub type AnalysisResult = Result<Instruction, diagnostics::Error>;

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

#[cfg(test)]
mod tests {
    use super::*;

    use self::Keyword::*;

    enum TestToken {
        Identifier(&'static str),
        Keyword(Keyword),
        Number(isize),
    }

    impl Token for TestToken {
        fn kind(&self) -> TokenKind {
            match *self {
                TestToken::Identifier(_) => TokenKind::Identifier,
                TestToken::Keyword(keyword) => TokenKind::Keyword(keyword),
                TestToken::Number(_) => TokenKind::Number,
            }
        }
    }

    fn atom(keyword: Keyword) -> SynExpr<TestToken> {
        SynExpr::from(TestToken::Keyword(keyword))
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

    impl From<SimpleOperand> for SynExpr<TestToken> {
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

    impl From<Condition> for SynExpr<TestToken> {
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
                vec![SynExpr::from(TestToken::Identifier(ident)).deref(), atom(A)]
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
                vec![atom(A), SynExpr::from(TestToken::Identifier(ident)).deref()]
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
        test_cp_const_analysis(TestToken::Identifier(ident), Expr::Symbol(ident.to_string()))
    }

    #[test]
    fn analyze_cp_literal() {
        let literal = 0x50;
        test_cp_const_analysis(TestToken::Number(literal), Expr::Literal(literal))
    }

    fn test_cp_const_analysis(atom: TestToken, expr: Expr) {
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
                Some(SynExpr::Atom(TestToken::Identifier(ident)))
            ),
            Ok(Instruction::Jr(None, Expr::Symbol(ident.to_string())))
        )
    }

    #[test]
    fn analyze_legal_instructions() {
        test_instruction_analysis(describe_legal_instructions());
    }

    type InstructionDescriptor = ((Keyword, Vec<SynExpr<TestToken>>), Instruction);

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
                    SynExpr::Atom(TestToken::Identifier(ident)),
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
        I: IntoIterator<Item = SynExpr<TestToken>>,
    {
        let mut analyzer = CommandAnalyzer::new(TestExprFactory::new());
        analyzer.analyze_instruction(mnemonic, operands)
    }

    struct TestExprFactory;

    impl TestExprFactory {
        fn new() -> TestExprFactory {
            TestExprFactory {}
        }
    }

    impl ExprFactory for TestExprFactory {
        type Token = TestToken;

        fn mk_atom(&mut self, token: Self::Token) -> Expr {
            match token {
                TestToken::Identifier(ident) => Expr::Symbol(ident.to_string()),
                TestToken::Number(number) => Expr::Literal(number),
                _ => panic!(),
            }
        }
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
