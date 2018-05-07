use backend::*;
use diagnostics::{Diagnostic, Message};
use frontend::ExprFactory;
use frontend::syntax::{keyword, token, Atom, SynExpr, Token};

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

    fn analyze_operand(
        &mut self,
        expr: SynExpr<Token>,
        context: &OperandAnalysisContext,
    ) -> Operand {
        match expr {
            SynExpr::Atom(token) => self.analyze_atom_operand(token, context),
            SynExpr::Deref(expr) => self.analyze_deref_operand(*expr),
        }
    }

    fn analyze_atom_operand(&mut self, token: Token, context: &OperandAnalysisContext) -> Operand {
        match token {
            token::Atom(Atom::Operand(operand)) => analyze_keyword_operand(operand, context),
            token::Atom(Atom::Ident(_)) | token::Atom(Atom::Number(_)) => {
                Operand::Const(self.expr_factory.mk_atom(token))
            }
            _ => panic!(),
        }
    }

    fn analyze_deref_operand(&mut self, expr: SynExpr<Token>) -> Operand {
        if let SynExpr::Atom(token) = expr {
            match token {
                token::Atom(Atom::Operand(keyword::Operand::Hl)) => {
                    Operand::Simple(SimpleOperand::DerefHl)
                }
                token::Atom(Atom::Ident(_)) => Operand::Deref(self.expr_factory.mk_atom(token)),
                _ => panic!(),
            }
        } else {
            panic!()
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

    pub fn analyze_instruction<I>(
        &mut self,
        mnemonic: keyword::Command,
        operands: I,
    ) -> AnalysisResult<()>
    where
        I: IntoIterator<Item = SynExpr<Token>>,
    {
        let mnemonic = to_mnemonic(mnemonic);
        let context = match mnemonic {
            Mnemonic::Branch(_) => OperandAnalysisContext::Branch,
            _ => OperandAnalysisContext::Other,
        };
        Analysis::new(
            operands
                .into_iter()
                .map(|x| self.operand_analyzer.analyze_operand(x, &context)),
        ).run(mnemonic)
    }
}

struct Analysis<I> {
    operands: I,
}

impl<'a, I: Iterator<Item = Operand>> Analysis<I> {
    fn new(operands: I) -> Analysis<I> {
        Analysis { operands }
    }

    fn run(mut self, mnemonic: Mnemonic) -> AnalysisResult<()> {
        use self::Mnemonic::*;
        match mnemonic {
            Alu(operation) => self.analyze_alu_instruction(operation),
            Dec => match self.operands.next() {
                Some(Operand::Simple(operand)) => Ok(Instruction::Dec(operand)),
                _ => panic!(),
            },
            Branch(branch) => self.analyze_branch(branch),
            Ld => self.analyze_ld(),
            Nullary(instruction) => self.analyze_nullary_instruction(instruction),
            Push => match self.operands.next() {
                Some(Operand::Reg16(src)) => Ok(Instruction::Push(src)),
                _ => panic!(),
            },
        }
    }

    fn analyze_alu_instruction(&mut self, operation: AluOperation) -> AnalysisResult<()> {
        match self.operands.next() {
            Some(Operand::Simple(src)) => Ok(Instruction::Alu(operation, AluSource::Simple(src))),
            Some(Operand::Const(expr)) => {
                Ok(Instruction::Alu(operation, AluSource::Immediate(expr)))
            }
            _ => panic!(),
        }
    }

    fn analyze_branch(&mut self, branch: BranchKind) -> AnalysisResult<()> {
        let first_operand = self.operands.next();
        let (condition, target) = if let Some(Operand::Condition(condition)) = first_operand {
            (Some(condition), analyze_branch_target(self.operands.next()))
        } else {
            (None, analyze_branch_target(first_operand))
        };
        Ok(Instruction::Branch(mk_branch(branch, target), condition))
    }

    fn analyze_nullary_instruction(&mut self, instruction: Instruction) -> AnalysisResult<()> {
        match self.operands.by_ref().count() {
            0 => Ok(instruction),
            n => Err(Diagnostic {
                message: Message::OperandCount {
                    actual: n,
                    expected: 0,
                },
                highlight: None,
            }),
        }
    }

    fn analyze_ld(&mut self) -> AnalysisResult<()> {
        let dest = self.operands.next().unwrap();
        let src = self.operands.next().unwrap();
        assert_eq!(self.operands.next(), None);
        match (dest, src) {
            (Operand::Simple(dest), Operand::Simple(src)) => {
                Ok(Instruction::Ld(LdKind::Simple(dest, src)))
            }
            (Operand::Simple(SimpleOperand::A), src) => analyze_ld_a(src, Direction::IntoA),
            (dest, Operand::Simple(SimpleOperand::A)) => analyze_ld_a(dest, Direction::FromA),
            (Operand::Reg16(reg16), Operand::Const(expr)) => {
                Ok(Instruction::Ld(LdKind::Immediate16(reg16, expr)))
            }
            _ => panic!(),
        }
    }
}

fn analyze_branch_target(target: Option<Operand>) -> Option<Expr> {
    match target {
        Some(Operand::Const(expr)) => Some(expr),
        None => None,
        _ => panic!(),
    }
}

fn mk_branch(kind: BranchKind, target: Option<Expr>) -> Branch {
    match (kind, target) {
        (BranchKind::Jp, Some(expr)) => Branch::Jp(expr),
        (BranchKind::Jr, Some(expr)) => Branch::Jr(expr),
        _ => panic!(),
    }
}

fn analyze_ld_a(other: Operand, direction: Direction) -> AnalysisResult<()> {
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

pub type AnalysisResult<R> = Result<Instruction, Diagnostic<R>>;

fn analyze_keyword_operand(keyword: keyword::Operand, context: &OperandAnalysisContext) -> Operand {
    use frontend::syntax::keyword::Operand::*;
    match keyword {
        A => Operand::Simple(SimpleOperand::A),
        B => Operand::Simple(SimpleOperand::B),
        Bc => Operand::Reg16(Reg16::Bc),
        C => match *context {
            OperandAnalysisContext::Branch => Operand::Condition(Condition::C),
            OperandAnalysisContext::Other => Operand::Simple(SimpleOperand::C),
        },
        D => Operand::Simple(SimpleOperand::D),
        E => Operand::Simple(SimpleOperand::E),
        H => Operand::Simple(SimpleOperand::H),
        Hl => Operand::Reg16(Reg16::Hl),
        L => Operand::Simple(SimpleOperand::L),
        Nc => Operand::Condition(Condition::Nc),
        Nz => Operand::Condition(Condition::Nz),
        Z => Operand::Condition(Condition::Z),
    }
}

#[derive(Debug, PartialEq)]
enum Mnemonic {
    Alu(AluOperation),
    Dec,
    Branch(BranchKind),
    Ld,
    Nullary(Instruction),
    Push,
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
        Halt => Mnemonic::Nullary(Instruction::Halt),
        Jp => Mnemonic::Branch(BranchKind::Jp),
        Jr => Mnemonic::Branch(BranchKind::Jr),
        Ld => Mnemonic::Ld,
        Nop => Mnemonic::Nullary(Instruction::Nop),
        Push => Mnemonic::Push,
        Stop => Mnemonic::Nullary(Instruction::Stop),
        Xor => Mnemonic::Alu(AluOperation::Xor),
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use self::keyword::{Command, Operand::*};

    type TestToken = Token;

    fn atom(keyword: keyword::Operand) -> SynExpr<TestToken> {
        SynExpr::from(token::Atom(Atom::Operand(keyword)))
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

    impl From<Reg16> for SynExpr<TestToken> {
        fn from(reg16: Reg16) -> Self {
            match reg16 {
                Reg16::Bc => atom(Bc),
                Reg16::Hl => atom(Hl),
            }
        }
    }

    impl From<Condition> for SynExpr<TestToken> {
        fn from(condition: Condition) -> Self {
            match condition {
                Condition::C => atom(C),
                Condition::Nc => atom(Nc),
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
                Command::Ld,
                vec![
                    SynExpr::from(token::Atom(Atom::Ident(ident.to_string()))).deref(),
                    atom(A),
                ]
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
                Command::Ld,
                vec![
                    atom(A),
                    SynExpr::from(token::Atom(Atom::Ident(ident.to_string()))).deref(),
                ]
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
        test_cp_const_analysis(
            token::Atom(Atom::Ident(ident.to_string())),
            Expr::Symbol(ident.to_string()),
        )
    }

    #[test]
    fn analyze_cp_literal() {
        let literal = 0x50;
        test_cp_const_analysis(token::Atom(Atom::Number(literal)), Expr::Literal(literal))
    }

    fn test_cp_const_analysis(atom: TestToken, expr: Expr) {
        assert_eq!(
            analyze(Command::Cp, Some(SynExpr::Atom(atom))),
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

    type InstructionDescriptor = ((Command, Vec<SynExpr<TestToken>>), Instruction);

    fn describe_legal_instructions() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        descriptors.extend(describe_nullary_instructions());
        descriptors.extend(describe_ld_simple_instructions());
        descriptors.extend(describe_ld_reg16_immediate_instructions());
        descriptors.extend(describe_alu_simple_instructions());
        descriptors.extend(describe_branch_instuctions());
        descriptors.extend(describe_dec_instructions());
        descriptors.push((
            (Command::Push, vec![atom(Bc)]),
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
                vec![
                    SynExpr::from(dest),
                    SynExpr::from(token::Atom(Atom::Ident(value.to_string()))),
                ],
            ),
            Instruction::Ld(LdKind::Immediate16(dest, Expr::Symbol(value.to_string()))),
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
        operands.push(SynExpr::Atom(token::Atom(Atom::Ident(ident.to_string()))));
        (
            (Command::from(branch), operands),
            Instruction::Branch(
                mk_branch(branch, Some(Expr::Symbol(ident.to_string()))),
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
        I: IntoIterator<Item = SynExpr<TestToken>>,
    {
        use frontend::StrExprFactory;
        let mut expr_factory = StrExprFactory::new();
        let mut analyzer = CommandAnalyzer::new(&mut expr_factory);
        analyzer.analyze_instruction(mnemonic, operands)
    }

    #[test]
    fn analyze_nop_a() {
        assert_eq!(
            analyze(Command::Nop, vec![atom(A)]),
            Err(Diagnostic {
                message: Message::OperandCount {
                    actual: 1,
                    expected: 0,
                },
                highlight: None,
            })
        )
    }
}
