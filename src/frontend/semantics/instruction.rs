use std::fmt::Debug;

use backend::*;
use diagnostics::{Diagnostic, Message};
use frontend::ExprFactory;
use frontend::syntax::{keyword, Literal, SynExpr};

struct OperandAnalyzer<'a, EF: 'a> {
    expr_factory: &'a mut EF,
}

enum OperandAnalysisContext {
    Branch,
    Other,
}

impl<'a, EF: 'a + ExprFactory> OperandAnalyzer<'a, EF> {
    fn new(expr_factory: &'a mut EF) -> OperandAnalyzer<'a, EF> {
        OperandAnalyzer { expr_factory }
    }

    fn analyze_operand<R>(
        &mut self,
        expr: SynExpr<String, R>,
        context: &OperandAnalysisContext,
    ) -> Operand<R> {
        match expr {
            SynExpr::Deref(expr) => self.analyze_deref_operand(*expr),
            SynExpr::Ident(ident) => self.analyze_ident_operand(ident),
            SynExpr::Literal(literal) => self.analyze_literal_operand(literal, context),
        }
    }

    fn analyze_ident_operand<R>(&mut self, ident: (String, R)) -> Operand<R> {
        Operand::Const(self.expr_factory.mk_symbol(ident))
    }

    fn analyze_literal_operand<R>(
        &mut self,
        literal: (Literal<String>, R),
        context: &OperandAnalysisContext,
    ) -> Operand<R> {
        match literal.0 {
            Literal::Operand(operand) => analyze_keyword_operand((operand, literal.1), context),
            Literal::Number(_) => Operand::Const(self.expr_factory.mk_literal(literal)),
            _ => panic!(),
        }
    }

    fn analyze_deref_operand<R>(&mut self, expr: SynExpr<String, R>) -> Operand<R> {
        match expr {
            SynExpr::Ident(ident) => Operand::Deref(self.expr_factory.mk_symbol(ident)),
            SynExpr::Literal((Literal::Operand(keyword::Operand::Hl), token_ref)) => {
                Operand::Simple(SimpleOperand::DerefHl, token_ref)
            }
            _ => panic!(),
        }
    }
}

pub struct CommandAnalyzer<'a, EF: 'a> {
    operand_analyzer: OperandAnalyzer<'a, EF>,
}

impl<'a, EF: ExprFactory> CommandAnalyzer<'a, EF> {
    pub fn new(expr_factory: &'a mut EF) -> CommandAnalyzer<'a, EF> {
        CommandAnalyzer {
            operand_analyzer: OperandAnalyzer::new(expr_factory),
        }
    }

    pub fn analyze_instruction<I, R>(
        &mut self,
        mnemonic: (keyword::Command, R),
        operands: I,
    ) -> AnalysisResult<R>
    where
        I: IntoIterator<Item = SynExpr<String, R>>,
        R: Debug + PartialEq,
    {
        let (mnemonic, mnemonic_ref) = (to_mnemonic(mnemonic.0), mnemonic.1);
        let context = match mnemonic {
            Mnemonic::Branch(_) => OperandAnalysisContext::Branch,
            _ => OperandAnalysisContext::Other,
        };
        Analysis::new(
            operands
                .into_iter()
                .map(|x| self.operand_analyzer.analyze_operand(x, &context)),
        ).run((mnemonic, mnemonic_ref))
    }
}

struct Analysis<I> {
    operands: I,
}

impl<'a, R: Debug + PartialEq, I: Iterator<Item = Operand<R>>> Analysis<I> {
    fn new(operands: I) -> Analysis<I> {
        Analysis { operands }
    }

    fn run(mut self, mnemonic: (Mnemonic, R)) -> AnalysisResult<R> {
        use self::Mnemonic::*;
        match mnemonic.0 {
            Alu(operation) => self.analyze_alu_instruction((operation, mnemonic.1)),
            Dec => match self.operands.next() {
                Some(Operand::Simple(operand, _)) => Ok(Instruction::Dec(operand)),
                _ => panic!(),
            },
            Branch(branch) => self.analyze_branch(branch),
            Ld => self.analyze_ld(),
            Nullary(instruction) => self.analyze_nullary_instruction((instruction, mnemonic.1)),
            Push => match self.operands.next() {
                Some(Operand::Reg16(src, _)) => Ok(Instruction::Push(src)),
                _ => panic!(),
            },
        }
    }

    fn analyze_alu_instruction(&mut self, (operation, _): (AluOperation, R)) -> AnalysisResult<R> {
        match self.operands.next() {
            Some(Operand::Simple(src, _)) => {
                Ok(Instruction::Alu(operation, AluSource::Simple(src)))
            }
            Some(Operand::Const(expr)) => {
                Ok(Instruction::Alu(operation, AluSource::Immediate(expr)))
            }
            _ => panic!(),
        }
    }

    fn analyze_branch(&mut self, branch: BranchKind) -> AnalysisResult<R> {
        let first_operand = self.operands.next();
        let (condition, target) = if let Some(Operand::Condition(condition, _)) = first_operand {
            (Some(condition), analyze_branch_target(self.operands.next()))
        } else {
            (None, analyze_branch_target(first_operand))
        };
        Ok(Instruction::Branch(mk_branch(branch, target), condition))
    }

    fn analyze_nullary_instruction(&mut self, mnemonic: (NullaryMnemonic, R)) -> AnalysisResult<R> {
        match self.operands.by_ref().count() {
            0 => Ok(mnemonic.0.into()),
            n => Err(Diagnostic::new(
                Message::OperandCount {
                    actual: n,
                    expected: 0,
                },
                mnemonic.1,
            )),
        }
    }

    fn analyze_ld(&mut self) -> AnalysisResult<R> {
        let dest = self.operands.next().unwrap();
        let src = self.operands.next().unwrap();
        assert_eq!(self.operands.next(), None);
        match (dest, src) {
            (Operand::Simple(dest, _), Operand::Simple(src, _)) => {
                Ok(Instruction::Ld(LdKind::Simple(dest, src)))
            }
            (Operand::Simple(SimpleOperand::A, _), src) => analyze_ld_a(src, Direction::IntoA),
            (dest, Operand::Simple(SimpleOperand::A, _)) => analyze_ld_a(dest, Direction::FromA),
            (Operand::Reg16(reg16, _), Operand::Const(expr)) => {
                Ok(Instruction::Ld(LdKind::Immediate16(reg16, expr)))
            }
            _ => panic!(),
        }
    }
}

fn analyze_branch_target<R>(target: Option<Operand<R>>) -> Option<Expr<R>> {
    match target {
        Some(Operand::Const(expr)) => Some(expr),
        None => None,
        _ => panic!(),
    }
}

fn mk_branch<R>(kind: BranchKind, target: Option<Expr<R>>) -> Branch<R> {
    match (kind, target) {
        (BranchKind::Jp, Some(expr)) => Branch::Jp(expr),
        (BranchKind::Jr, Some(expr)) => Branch::Jr(expr),
        _ => panic!(),
    }
}

fn analyze_ld_a<R>(other: Operand<R>, direction: Direction) -> AnalysisResult<R> {
    match other {
        Operand::Deref(expr) => Ok(Instruction::Ld(LdKind::ImmediateAddr(expr, direction))),
        _ => panic!(),
    }
}

#[derive(Debug, PartialEq)]
pub enum Operand<R> {
    Simple(SimpleOperand, R),
    Condition(Condition, R),
    Const(Expr<R>),
    Deref(Expr<R>),
    Reg16(Reg16, R),
}

pub type AnalysisResult<R> = Result<Instruction<R>, Diagnostic<R>>;

fn analyze_keyword_operand<R>(
    keyword: (keyword::Operand, R),
    context: &OperandAnalysisContext,
) -> Operand<R> {
    use frontend::syntax::keyword::Operand::*;
    match keyword.0 {
        A => Operand::Simple(SimpleOperand::A, keyword.1),
        B => Operand::Simple(SimpleOperand::B, keyword.1),
        Bc => Operand::Reg16(Reg16::Bc, keyword.1),
        C => match *context {
            OperandAnalysisContext::Branch => Operand::Condition(Condition::C, keyword.1),
            OperandAnalysisContext::Other => Operand::Simple(SimpleOperand::C, keyword.1),
        },
        D => Operand::Simple(SimpleOperand::D, keyword.1),
        E => Operand::Simple(SimpleOperand::E, keyword.1),
        H => Operand::Simple(SimpleOperand::H, keyword.1),
        Hl => Operand::Reg16(Reg16::Hl, keyword.1),
        L => Operand::Simple(SimpleOperand::L, keyword.1),
        Nc => Operand::Condition(Condition::Nc, keyword.1),
        Nz => Operand::Condition(Condition::Nz, keyword.1),
        Z => Operand::Condition(Condition::Z, keyword.1),
    }
}

#[derive(Debug, PartialEq)]
enum Mnemonic {
    Alu(AluOperation),
    Dec,
    Branch(BranchKind),
    Ld,
    Nullary(NullaryMnemonic),
    Push,
}

#[derive(Debug, PartialEq)]
enum NullaryMnemonic {
    Halt,
    Nop,
    Stop,
}

impl<R> From<NullaryMnemonic> for Instruction<R> {
    fn from(nullary_mnemonic: NullaryMnemonic) -> Instruction<R> {
        match nullary_mnemonic {
            NullaryMnemonic::Halt => Instruction::Halt,
            NullaryMnemonic::Nop => Instruction::Nop,
            NullaryMnemonic::Stop => Instruction::Stop,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BranchKind {
    Jp,
    Jr,
}

fn to_mnemonic(command: keyword::Command) -> Mnemonic {
    use frontend::syntax::keyword::Command::*;
    match command {
        And => Mnemonic::Alu(AluOperation::And),
        Cp => Mnemonic::Alu(AluOperation::Cp),
        Dec => Mnemonic::Dec,
        Halt => Mnemonic::Nullary(NullaryMnemonic::Halt),
        Jp => Mnemonic::Branch(BranchKind::Jp),
        Jr => Mnemonic::Branch(BranchKind::Jr),
        Ld => Mnemonic::Ld,
        Nop => Mnemonic::Nullary(NullaryMnemonic::Nop),
        Push => Mnemonic::Push,
        Stop => Mnemonic::Nullary(NullaryMnemonic::Stop),
        Xor => Mnemonic::Alu(AluOperation::Xor),
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use self::keyword::{Command, Operand::*};

    fn literal(keyword: keyword::Operand) -> SynExpr<String, ()> {
        SynExpr::Literal((Literal::Operand(keyword), ()))
    }

    impl From<AluOperation> for Command {
        fn from(alu_operation: AluOperation) -> Self {
            match alu_operation {
                AluOperation::And => Command::And,
                AluOperation::Cp => Command::Cp,
                AluOperation::Xor => Command::Xor,
            }
        }
    }

    impl From<BranchKind> for Command {
        fn from(branch: BranchKind) -> Self {
            match branch {
                BranchKind::Jp => Command::Jp,
                BranchKind::Jr => Command::Jr,
            }
        }
    }

    impl From<SimpleOperand> for SynExpr<String, ()> {
        fn from(alu_operand: SimpleOperand) -> Self {
            match alu_operand {
                SimpleOperand::A => literal(A),
                SimpleOperand::B => literal(B),
                SimpleOperand::C => literal(C),
                SimpleOperand::D => literal(D),
                SimpleOperand::E => literal(E),
                SimpleOperand::H => literal(H),
                SimpleOperand::L => literal(L),
                SimpleOperand::DerefHl => literal(Hl).deref(),
            }
        }
    }

    impl From<Reg16> for SynExpr<String, ()> {
        fn from(reg16: Reg16) -> Self {
            match reg16 {
                Reg16::Bc => literal(Bc),
                Reg16::Hl => literal(Hl),
            }
        }
    }

    impl From<Condition> for SynExpr<String, ()> {
        fn from(condition: Condition) -> Self {
            match condition {
                Condition::C => literal(C),
                Condition::Nc => literal(Nc),
                Condition::Nz => literal(Nz),
                Condition::Z => literal(Z),
            }
        }
    }

    #[test]
    fn analyze_ld_deref_symbol_a() {
        let ident = "ident";
        assert_eq!(
            analyze(
                Command::Ld,
                vec![SynExpr::Ident((ident.to_string(), ())).deref(), literal(A)]
            ),
            Ok(Instruction::Ld(LdKind::ImmediateAddr(
                Expr::Symbol(ident.to_string(), ()),
                Direction::FromA,
            )))
        )
    }

    #[test]
    fn analyze_ld_a_deref_symbol() {
        let ident = "ident";
        assert_eq!(
            analyze(
                Command::Ld,
                vec![literal(A), SynExpr::Ident((ident.to_string(), ())).deref()]
            ),
            Ok(Instruction::Ld(LdKind::ImmediateAddr(
                Expr::Symbol(ident.to_string(), ()),
                Direction::IntoA,
            )))
        )
    }

    #[test]
    fn analyze_cp_symbol() {
        let ident = "ident";
        test_cp_const_analysis(
            SynExpr::Ident((ident.to_string(), ())),
            Expr::Symbol(ident.to_string(), ()),
        )
    }

    #[test]
    fn analyze_cp_literal() {
        let n = 0x50;
        let literal = SynExpr::Literal((Literal::Number(n), ()));
        test_cp_const_analysis(literal, Expr::Literal(n, ()))
    }

    fn test_cp_const_analysis(parsed: SynExpr<String, ()>, expr: Expr<()>) {
        assert_eq!(
            analyze(Command::Cp, Some(parsed)),
            Ok(Instruction::Alu(
                AluOperation::Cp,
                AluSource::Immediate(expr)
            ))
        )
    }

    #[test]
    fn analyze_legal_instructions() {
        test_instruction_analysis(describe_legal_instructions());
    }

    type InstructionDescriptor = ((Command, Vec<SynExpr<String, ()>>), Instruction<()>);

    fn describe_legal_instructions() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        descriptors.extend(describe_nullary_instructions());
        descriptors.extend(describe_ld_simple_instructions());
        descriptors.extend(describe_ld_reg16_immediate_instructions());
        descriptors.extend(describe_alu_simple_instructions());
        descriptors.extend(describe_branch_instuctions());
        descriptors.extend(describe_dec_instructions());
        descriptors.push((
            (Command::Push, vec![literal(Bc)]),
            Instruction::Push(Reg16::Bc),
        ));
        descriptors
    }

    fn describe_nullary_instructions() -> Vec<InstructionDescriptor> {
        vec![
            (Command::Halt, Instruction::Halt),
            (Command::Nop, Instruction::Nop),
            (Command::Stop, Instruction::Stop),
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
                (Command::Ld, vec![SynExpr::from(dest), SynExpr::from(src)]),
                Instruction::Ld(LdKind::Simple(dest, src)),
            )),
        }
    }

    fn describe_ld_reg16_immediate_instructions() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        for &dest in REG16.iter() {
            descriptors.push(describe_ld_reg16_immediate(dest))
        }
        descriptors
    }

    fn describe_ld_reg16_immediate(dest: Reg16) -> InstructionDescriptor {
        let value = "value";
        (
            (
                Command::Ld,
                vec![SynExpr::from(dest), SynExpr::Ident((value.to_string(), ()))],
            ),
            Instruction::Ld(LdKind::Immediate16(
                dest,
                Expr::Symbol(value.to_string(), ()),
            )),
        )
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
            (Command::from(operation), vec![SynExpr::from(operand)]),
            Instruction::Alu(operation, AluSource::Simple(operand)),
        )
    }

    fn describe_branch_instuctions() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        for &kind in BRANCHES.iter() {
            descriptors.push(describe_branch(kind, None));
            for &condition in CONDITIONS.iter() {
                descriptors.push(describe_branch(kind, Some(condition)))
            }
        }
        descriptors
    }

    fn describe_branch(branch: BranchKind, condition: Option<Condition>) -> InstructionDescriptor {
        let ident = "ident";
        let mut operands = Vec::new();
        if let Some(condition) = condition {
            operands.push(SynExpr::from(condition))
        };
        operands.push(SynExpr::Ident((ident.to_string(), ())));
        (
            (Command::from(branch), operands),
            Instruction::Branch(
                mk_branch(branch, Some(Expr::Symbol(ident.to_string(), ()))),
                condition,
            ),
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
            (Command::Dec, vec![SynExpr::from(operand)]),
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

    const REG16: [Reg16; 2] = [Reg16::Bc, Reg16::Hl];

    const BRANCHES: [BranchKind; 2] = [BranchKind::Jp, BranchKind::Jr];

    const CONDITIONS: [Condition; 4] = [Condition::C, Condition::Nc, Condition::Nz, Condition::Z];

    fn test_instruction_analysis(descriptors: Vec<InstructionDescriptor>) {
        for ((mnemonic, operands), expected) in descriptors {
            assert_eq!(analyze(mnemonic, operands), Ok(expected))
        }
    }

    fn analyze<I>(mnemonic: Command, operands: I) -> AnalysisResult<()>
    where
        I: IntoIterator<Item = SynExpr<String, ()>>,
    {
        use frontend::StrExprFactory;
        let mut expr_factory = StrExprFactory::new();
        let mut analyzer = CommandAnalyzer::new(&mut expr_factory);
        analyzer.analyze_instruction((mnemonic, ()), operands)
    }

    #[test]
    fn analyze_nop_a() {
        assert_eq!(
            analyze(Command::Nop, vec![literal(A)]),
            Err(Diagnostic::new(
                Message::OperandCount {
                    actual: 1,
                    expected: 0,
                },
                (),
            ))
        )
    }
}
