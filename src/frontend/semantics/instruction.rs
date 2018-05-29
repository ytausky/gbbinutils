use std::fmt::Debug;

use diagnostics::{Diagnostic, Message};
use frontend::ExprFactory;
use frontend::semantics::operand::{AtomKind, Context, Operand, OperandAnalyzer, OperandCounter};
use frontend::syntax::{keyword, ParsedExpr};
use instruction::*;

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
        I: IntoIterator<Item = ParsedExpr<String, R>>,
        R: Clone + Debug + PartialEq,
    {
        let (mnemonic, mnemonic_ref) = (to_mnemonic(mnemonic.0), mnemonic.1);
        let context = match mnemonic {
            Mnemonic::Branch(_) => Context::Branch,
            Mnemonic::Stack(_) => Context::Stack,
            _ => Context::Other,
        };
        Analysis::new(
            operands
                .into_iter()
                .map(|x| self.operand_analyzer.analyze_operand(x, &context)),
        ).run((mnemonic, mnemonic_ref))
    }
}

struct Analysis<I> {
    operands: OperandCounter<I>,
}

impl<'a, R: Clone + Debug + PartialEq, I: Iterator<Item = Operand<R>>> Analysis<I> {
    fn new(operands: I) -> Analysis<I> {
        Analysis {
            operands: OperandCounter::new(operands),
        }
    }

    fn run(mut self, (mnemonic, range): (Mnemonic, R)) -> AnalysisResult<R> {
        use self::Mnemonic::*;
        let instruction = match mnemonic {
            Alu(AluOperation::Add, explicit_a) => {
                self.analyze_add_instruction(range.clone(), explicit_a)
            }
            Alu(operation, explicit_a) => {
                let first_operand = self.operands.next();
                self.analyze_alu_instruction((operation, range.clone()), explicit_a, first_operand)
            }
            IncDec(mode) => self.analyze_inc_dec(mode),
            Branch(branch) => self.analyze_branch(branch),
            Ld => self.analyze_ld(),
            Ldh => self.analyze_ldh(),
            Nullary(instruction) => Ok(instruction.into()),
            Stack(operation) => self.analyze_stack_operation(operation),
        }?;
        let expected = self.operands.seen();
        let extra = self.operands.count();
        let actual = expected + extra;
        if actual == expected {
            Ok(instruction)
        } else {
            Err(Diagnostic::new(
                Message::OperandCount { actual, expected },
                range,
            ))
        }
    }

    fn analyze_add_instruction(
        &mut self,
        operation_ref: R,
        explicit_a: ExplicitA,
    ) -> AnalysisResult<R> {
        match self.operands.next() {
            Some(Operand::Atom(AtomKind::Reg16(reg16), range)) => {
                self.analyze_add_hl_instruction((reg16, range))
            }
            operand => self.analyze_alu_instruction(
                (AluOperation::Add, operation_ref),
                explicit_a,
                operand,
            ),
        }
    }

    fn analyze_add_hl_instruction(&mut self, target: (Reg16, R)) -> AnalysisResult<R> {
        assert_eq!(target.0, Reg16::Hl);
        match self.operands.next() {
            Some(Operand::Atom(AtomKind::Reg16(src), _)) => Ok(Instruction::AddHl(src)),
            _ => panic!(),
        }
    }

    fn analyze_alu_instruction(
        &mut self,
        (operation, _): (AluOperation, R),
        explicit_a: ExplicitA,
        first_operand: Option<Operand<R>>,
    ) -> AnalysisResult<R> {
        let next_operand = if explicit_a == ExplicitA::Required {
            match first_operand {
                Some(Operand::Atom(AtomKind::Simple(SimpleOperand::A), _)) => (),
                _ => panic!(),
            };
            self.operands.next()
        } else {
            first_operand
        };
        match next_operand {
            Some(Operand::Atom(AtomKind::Simple(src), _)) => {
                Ok(Instruction::Alu(operation, AluSource::Simple(src)))
            }
            Some(Operand::Const(expr)) => {
                Ok(Instruction::Alu(operation, AluSource::Immediate(expr)))
            }
            _ => panic!(),
        }
    }

    fn analyze_branch(&mut self, branch: BranchKind) -> AnalysisResult<R> {
        let (condition, target) = self.collect_condition_and_target();
        match (branch, condition, target) {
            (BranchKind::Jp, None, Some(TargetSelector::DerefHl)) => Ok(Instruction::JpDerefHl),
            (BranchKind::Jp, Some((_, condition_ref)), Some(TargetSelector::DerefHl)) => {
                Err(Diagnostic::new(Message::AlwaysUnconditional, condition_ref))
            }
            (_, condition, target) => Ok(Instruction::Branch(
                mk_branch(branch, target),
                condition.map(|(condition, _)| condition),
            )),
        }
    }

    fn collect_condition_and_target(
        &mut self,
    ) -> (Option<(Condition, R)>, Option<TargetSelector<R>>) {
        let first_operand = self.operands.next();
        if let Some(Operand::Atom(AtomKind::Condition(condition), range)) = first_operand {
            (
                Some((condition, range)),
                analyze_branch_target(self.operands.next()),
            )
        } else {
            (None, analyze_branch_target(first_operand))
        }
    }

    fn analyze_ld(&mut self) -> AnalysisResult<R> {
        let dest = self.operands.next().unwrap();
        let src = self.operands.next().unwrap();
        match (dest, src) {
            (Operand::Atom(AtomKind::Simple(dest), _), Operand::Atom(AtomKind::Simple(src), _)) => {
                Ok(Instruction::Ld(LdKind::Simple(dest, src)))
            }
            (Operand::Atom(AtomKind::Simple(dest), _), Operand::Const(expr)) => {
                Ok(Instruction::Ld(LdKind::Immediate8(dest, expr)))
            }
            (Operand::Atom(AtomKind::Simple(SimpleOperand::A), _), src) => {
                analyze_ld_a(src, Direction::IntoA)
            }
            (dest, Operand::Atom(AtomKind::Simple(SimpleOperand::A), _)) => {
                analyze_ld_a(dest, Direction::FromA)
            }
            (Operand::Atom(AtomKind::Reg16(dest), _), Operand::Const(expr)) => {
                Ok(Instruction::Ld(LdKind::Immediate16(dest, expr)))
            }
            _ => panic!(),
        }
    }

    fn analyze_ldh(&mut self) -> AnalysisResult<R> {
        let dest = self.operands.next().unwrap();
        let src = self.operands.next().unwrap();
        match (dest, src) {
            (Operand::Deref(expr), Operand::Atom(AtomKind::Simple(SimpleOperand::A), _)) => {
                Ok(Instruction::Ldh(expr, Direction::FromA))
            }
            (Operand::Atom(AtomKind::Simple(SimpleOperand::A), _), Operand::Deref(expr)) => {
                Ok(Instruction::Ldh(expr, Direction::IntoA))
            }
            _ => panic!(),
        }
    }

    fn analyze_stack_operation(&mut self, operation: StackOperation) -> AnalysisResult<R> {
        let reg_pair = match self.operands.next() {
            Some(Operand::Atom(AtomKind::RegPair(reg_pair), _)) => reg_pair,
            _ => panic!(),
        };
        let instruction_ctor = match operation {
            StackOperation::Push => Instruction::Push,
            StackOperation::Pop => Instruction::Pop,
        };
        Ok(instruction_ctor(reg_pair))
    }

    fn analyze_inc_dec(&mut self, mode: IncDec) -> AnalysisResult<R> {
        match self.operands.next() {
            Some(Operand::Atom(AtomKind::Simple(operand), _)) => {
                Ok(Instruction::IncDec8(mode, operand))
            }
            Some(Operand::Atom(AtomKind::Reg16(operand), _)) => {
                Ok(Instruction::IncDec16(mode, operand))
            }
            _ => panic!(),
        }
    }
}

fn analyze_branch_target<R>(target: Option<Operand<R>>) -> Option<TargetSelector<R>> {
    match target {
        Some(Operand::Const(expr)) => Some(TargetSelector::Expr(expr)),
        Some(Operand::Atom(AtomKind::Simple(SimpleOperand::DerefHl), _)) => {
            Some(TargetSelector::DerefHl)
        }
        None => None,
        _ => panic!(),
    }
}

fn mk_branch<R>(kind: BranchKind, target: Option<TargetSelector<R>>) -> Branch<R> {
    match (kind, target) {
        (BranchKind::Call, Some(TargetSelector::Expr(expr))) => Branch::Call(expr),
        (BranchKind::Jp, Some(TargetSelector::Expr(expr))) => Branch::Jp(expr),
        (BranchKind::Jr, Some(TargetSelector::Expr(expr))) => Branch::Jr(expr),
        _ => panic!(),
    }
}

fn analyze_ld_a<R>(other: Operand<R>, direction: Direction) -> AnalysisResult<R> {
    match other {
        Operand::Deref(expr) => Ok(Instruction::Ld(LdKind::ImmediateAddr(expr, direction))),
        _ => panic!(),
    }
}

pub type AnalysisResult<R> = Result<Instruction<R>, Diagnostic<R>>;

#[derive(Debug, PartialEq)]
enum Mnemonic {
    Alu(AluOperation, ExplicitA),
    Branch(BranchKind),
    IncDec(IncDec),
    Ld,
    Ldh,
    Nullary(NullaryMnemonic),
    Stack(StackOperation),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ExplicitA {
    Required,
    NotAllowed,
}

#[derive(Debug, PartialEq)]
enum NullaryMnemonic {
    Halt,
    Nop,
    Reti,
    Stop,
}

#[derive(Debug, PartialEq)]
enum StackOperation {
    Push,
    Pop,
}

impl<R> From<NullaryMnemonic> for Instruction<R> {
    fn from(nullary_mnemonic: NullaryMnemonic) -> Instruction<R> {
        match nullary_mnemonic {
            NullaryMnemonic::Halt => Instruction::Halt,
            NullaryMnemonic::Nop => Instruction::Nop,
            NullaryMnemonic::Reti => Instruction::Reti,
            NullaryMnemonic::Stop => Instruction::Stop,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BranchKind {
    Call,
    Jp,
    Jr,
}

enum TargetSelector<R> {
    DerefHl,
    Expr(Expr<R>),
}

fn to_mnemonic(command: keyword::Command) -> Mnemonic {
    use frontend::syntax::keyword::Command::*;
    match command {
        Add => Mnemonic::Alu(AluOperation::Add, ExplicitA::Required),
        And => Mnemonic::Alu(AluOperation::And, ExplicitA::NotAllowed),
        Call => Mnemonic::Branch(BranchKind::Call),
        Cp => Mnemonic::Alu(AluOperation::Cp, ExplicitA::NotAllowed),
        Dec => Mnemonic::IncDec(IncDec::Dec),
        Halt => Mnemonic::Nullary(NullaryMnemonic::Halt),
        Inc => Mnemonic::IncDec(IncDec::Inc),
        Jp => Mnemonic::Branch(BranchKind::Jp),
        Jr => Mnemonic::Branch(BranchKind::Jr),
        Ld => Mnemonic::Ld,
        Ldh => Mnemonic::Ldh,
        Nop => Mnemonic::Nullary(NullaryMnemonic::Nop),
        Pop => Mnemonic::Stack(StackOperation::Pop),
        Push => Mnemonic::Stack(StackOperation::Push),
        Reti => Mnemonic::Nullary(NullaryMnemonic::Reti),
        Stop => Mnemonic::Nullary(NullaryMnemonic::Stop),
        Xor => Mnemonic::Alu(AluOperation::Xor, ExplicitA::NotAllowed),
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frontend::syntax::Literal;

    use self::keyword::{Command, Operand::*};

    #[derive(Clone, Debug, PartialEq)]
    enum Marking {
        Normal,
        Special,
    }

    impl Default for Marking {
        fn default() -> Self {
            Marking::Normal
        }
    }

    trait Mark {
        fn mark(self) -> Self;
    }

    type Input = ParsedExpr<String, Marking>;

    impl<S: ::frontend::syntax::StringRef> Mark for ParsedExpr<S, Marking> {
        fn mark(self) -> Self {
            match self {
                ParsedExpr::Ident((ident, _)) => ParsedExpr::Ident((ident, Marking::Special)),
                ParsedExpr::Deref(expr) => ParsedExpr::Deref(Box::new(expr.mark())),
                ParsedExpr::Literal((literal, _)) => {
                    ParsedExpr::Literal((literal, Marking::Special))
                }
            }
        }
    }

    fn literal(keyword: keyword::Operand) -> Input {
        ParsedExpr::Literal((Literal::Operand(keyword), Marking::Normal))
    }

    fn symbol(ident: &str) -> Expr<Marking> {
        Expr::Symbol(ident.to_string(), Marking::default())
    }

    trait ToMarked<T> {
        fn to_marked(self) -> (T, Marking);
    }

    impl ToMarked<Command> for Command {
        fn to_marked(self) -> (Command, Marking) {
            (self, Marking::default())
        }
    }

    impl ToMarked<Command> for (Command, Marking) {
        fn to_marked(self) -> (Command, Marking) {
            self
        }
    }

    impl From<AluOperation> for Command {
        fn from(alu_operation: AluOperation) -> Self {
            match alu_operation {
                AluOperation::Add => Command::Add,
                AluOperation::And => Command::And,
                AluOperation::Cp => Command::Cp,
                AluOperation::Xor => Command::Xor,
            }
        }
    }

    impl From<BranchKind> for Command {
        fn from(branch: BranchKind) -> Self {
            match branch {
                BranchKind::Call => Command::Call,
                BranchKind::Jp => Command::Jp,
                BranchKind::Jr => Command::Jr,
            }
        }
    }

    impl From<IncDec> for Command {
        fn from(mode: IncDec) -> Self {
            match mode {
                IncDec::Inc => Command::Inc,
                IncDec::Dec => Command::Dec,
            }
        }
    }

    impl From<SimpleOperand> for Input {
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

    impl From<Reg16> for ParsedExpr<String, Marking> {
        fn from(reg16: Reg16) -> Self {
            match reg16 {
                Reg16::Bc => literal(Bc),
                Reg16::De => literal(De),
                Reg16::Hl => literal(Hl),
                Reg16::Sp => literal(Sp),
            }
        }
    }

    impl From<RegPair> for ParsedExpr<String, Marking> {
        fn from(reg_pair: RegPair) -> Self {
            literal(match reg_pair {
                RegPair::Bc => Bc,
                RegPair::De => De,
                RegPair::Hl => Hl,
                RegPair::Af => Af,
            })
        }
    }

    impl From<Condition> for ParsedExpr<String, Marking> {
        fn from(condition: Condition) -> Self {
            match condition {
                Condition::C => literal(C),
                Condition::Nc => literal(Nc),
                Condition::Nz => literal(Nz),
                Condition::Z => literal(Z),
            }
        }
    }

    impl<'a> From<&'a str> for ParsedExpr<String, Marking> {
        fn from(ident: &'a str) -> Self {
            ParsedExpr::Ident((ident.to_string(), Marking::Normal))
        }
    }

    impl From<i32> for ParsedExpr<String, Marking> {
        fn from(n: i32) -> Self {
            ParsedExpr::Literal((Literal::Number(n), Marking::Normal))
        }
    }

    impl From<i32> for Expr<Marking> {
        fn from(n: i32) -> Self {
            Expr::Literal(n, Marking::default())
        }
    }

    #[test]
    fn analyze_jp_deref_hl() {
        analyze(Command::Jp, vec![literal(Hl).deref()]).expect_instruction(Instruction::JpDerefHl)
    }

    #[test]
    fn analyze_ld_deref_symbol_a() {
        let ident = "ident";
        analyze(
            Command::Ld,
            vec![ParsedExpr::from(ident).deref(), literal(A)],
        ).expect_instruction(Instruction::Ld(LdKind::ImmediateAddr(
            symbol(ident),
            Direction::FromA,
        )))
    }

    #[test]
    fn analyze_ld_a_deref_symbol() {
        let ident = "ident";
        analyze(
            Command::Ld,
            vec![literal(A), ParsedExpr::from(ident).deref()],
        ).expect_instruction(Instruction::Ld(LdKind::ImmediateAddr(
            symbol(ident),
            Direction::IntoA,
        )))
    }

    #[test]
    fn analyze_ldh_from_a() {
        let index = 0xcc;
        analyze(
            Command::Ldh,
            vec![ParsedExpr::from(index).deref(), literal(A)],
        ).expect_instruction(Instruction::Ldh(index.into(), Direction::FromA))
    }

    #[test]
    fn analyze_ldh_into_a() {
        let index = 0xcc;
        analyze(
            Command::Ldh,
            vec![literal(A), ParsedExpr::from(index).deref()],
        ).expect_instruction(Instruction::Ldh(index.into(), Direction::IntoA))
    }

    #[test]
    fn analyze_cp_symbol() {
        let ident = "ident";
        test_cp_const_analysis(ident.into(), symbol(ident))
    }

    #[test]
    fn analyze_cp_literal() {
        let n = 0x50;
        test_cp_const_analysis(n.into(), n.into())
    }

    fn test_cp_const_analysis(parsed: ParsedExpr<String, Marking>, expr: Expr<Marking>) {
        analyze(Command::Cp, Some(parsed)).expect_instruction(Instruction::Alu(
            AluOperation::Cp,
            AluSource::Immediate(expr),
        ))
    }

    #[test]
    fn analyze_legal_instructions() {
        test_instruction_analysis(describe_legal_instructions());
    }

    type InstructionDescriptor = ((Command, Vec<Input>), Instruction<Marking>);

    fn describe_legal_instructions() -> Vec<InstructionDescriptor> {
        let mut descriptors = Vec::new();
        descriptors.extend(describe_nullary_instructions());
        descriptors.extend(describe_ld_simple_simple_instructions());
        descriptors.extend(describe_ld_simple_immediate_instructions());
        descriptors.extend(describe_ld_reg16_immediate_instructions());
        descriptors.extend(describe_alu_simple_instructions());
        descriptors.extend(describe_add_hl_reg16_instructions());
        descriptors.extend(describe_branch_instuctions());
        descriptors.extend(describe_inc_dec8_instructions());
        descriptors.extend(describe_inc_dec16_instructions());
        descriptors.extend(describe_push_pop_instructions());
        descriptors
    }

    fn describe_push_pop_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        REG_PAIRS.iter().flat_map(|&reg_pair| {
            vec![
                (
                    (Command::Push, vec![reg_pair.into()]),
                    Instruction::Push(reg_pair),
                ),
                (
                    (Command::Pop, vec![reg_pair.into()]),
                    Instruction::Pop(reg_pair),
                ),
            ]
        })
    }

    fn describe_nullary_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        [
            (Command::Halt, Instruction::Halt),
            (Command::Nop, Instruction::Nop),
            (Command::Reti, Instruction::Reti),
            (Command::Stop, Instruction::Stop),
        ].iter()
            .map(|(mnemonic, instruction)| ((*mnemonic, vec![]), instruction.clone()))
    }

    fn describe_ld_simple_simple_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        SIMPLE_OPERANDS.iter().flat_map(|&dest| {
            SIMPLE_OPERANDS
                .iter()
                .flat_map(move |&src| describe_ld_simple_simple(dest, src))
        })
    }

    fn describe_ld_simple_simple(
        dest: SimpleOperand,
        src: SimpleOperand,
    ) -> Option<InstructionDescriptor> {
        match (dest, src) {
            (SimpleOperand::DerefHl, SimpleOperand::DerefHl) => None,
            _ => Some((
                (Command::Ld, vec![dest.into(), src.into()]),
                Instruction::Ld(LdKind::Simple(dest, src)),
            )),
        }
    }

    fn describe_ld_simple_immediate_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        SIMPLE_OPERANDS
            .iter()
            .map(|&dest| describe_ld_simple_immediate(dest))
    }

    fn describe_ld_simple_immediate(dest: SimpleOperand) -> InstructionDescriptor {
        let n = 0x12;
        (
            (Command::Ld, vec![ParsedExpr::from(dest), n.into()]),
            Instruction::Ld(LdKind::Immediate8(dest, n.into())),
        )
    }

    fn describe_ld_reg16_immediate_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        REG16.iter().map(|&dest| describe_ld_reg16_immediate(dest))
    }

    fn describe_ld_reg16_immediate(dest: Reg16) -> InstructionDescriptor {
        let value = "value";
        (
            (Command::Ld, vec![ParsedExpr::from(dest), value.into()]),
            Instruction::Ld(LdKind::Immediate16(dest, symbol(value))),
        )
    }

    fn describe_alu_simple_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        SIMPLE_OPERANDS.iter().flat_map(|&operand| {
            let with_a = ALU_OPERATIONS_WITH_A
                .iter()
                .map(move |&operation| describe_alu_simple_with_a(operation, operand));
            let without_a = ALU_OPERATIONS_WITHOUT_A
                .iter()
                .map(move |&operation| describe_alu_simple_without_a(operation, operand));
            with_a.chain(without_a)
        })
    }

    fn describe_alu_simple_with_a(
        operation: AluOperation,
        operand: SimpleOperand,
    ) -> InstructionDescriptor {
        (
            (
                operation.into(),
                vec![SimpleOperand::A.into(), operand.into()],
            ),
            Instruction::Alu(operation, AluSource::Simple(operand)),
        )
    }

    fn describe_alu_simple_without_a(
        operation: AluOperation,
        operand: SimpleOperand,
    ) -> InstructionDescriptor {
        (
            (Command::from(operation), vec![ParsedExpr::from(operand)]),
            Instruction::Alu(operation, AluSource::Simple(operand)),
        )
    }

    fn describe_add_hl_reg16_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        REG16.iter().map(|&reg16| describe_add_hl_reg16(reg16))
    }

    fn describe_add_hl_reg16(reg16: Reg16) -> InstructionDescriptor {
        (
            (Command::Add, vec![Reg16::Hl.into(), reg16.into()]),
            Instruction::AddHl(reg16),
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
            operands.push(ParsedExpr::from(condition))
        };
        operands.push(ident.into());
        (
            (Command::from(branch), operands),
            Instruction::Branch(
                mk_branch(branch, Some(TargetSelector::Expr(symbol(ident)))),
                condition,
            ),
        )
    }

    fn describe_inc_dec8_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        INC_DEC.iter().flat_map(|&mode| {
            SIMPLE_OPERANDS.iter().map(move |&operand| {
                (
                    (mode.into(), vec![operand.into()]),
                    Instruction::IncDec8(mode, operand),
                )
            })
        })
    }

    fn describe_inc_dec16_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        INC_DEC.iter().flat_map(|&mode| {
            REG16.iter().map(move |&reg16| {
                (
                    (mode.into(), vec![reg16.into()]),
                    Instruction::IncDec16(mode, reg16),
                )
            })
        })
    }

    const ALU_OPERATIONS_WITH_A: &[AluOperation] = &[AluOperation::Add];

    const ALU_OPERATIONS_WITHOUT_A: &[AluOperation] =
        &[AluOperation::And, AluOperation::Cp, AluOperation::Xor];

    const SIMPLE_OPERANDS: &[SimpleOperand] = &[
        SimpleOperand::A,
        SimpleOperand::B,
        SimpleOperand::C,
        SimpleOperand::D,
        SimpleOperand::E,
        SimpleOperand::H,
        SimpleOperand::L,
        SimpleOperand::DerefHl,
    ];

    const REG16: &[Reg16] = &[Reg16::Bc, Reg16::De, Reg16::Hl, Reg16::Sp];

    const REG_PAIRS: &[RegPair] = &[RegPair::Bc, RegPair::De, RegPair::Hl, RegPair::Af];

    const BRANCHES: &[BranchKind] = &[BranchKind::Call, BranchKind::Jp, BranchKind::Jr];

    const CONDITIONS: [Condition; 4] = [Condition::C, Condition::Nc, Condition::Nz, Condition::Z];

    const INC_DEC: &[IncDec] = &[IncDec::Inc, IncDec::Dec];

    fn test_instruction_analysis(descriptors: Vec<InstructionDescriptor>) {
        for ((mnemonic, operands), expected) in descriptors {
            analyze(mnemonic, operands).expect_instruction(expected)
        }
    }

    struct Result(AnalysisResult<Marking>);

    impl Result {
        fn expect_instruction(self, expected: Instruction<Marking>) {
            assert_eq!(self.0, Ok(expected))
        }

        fn expect_diagnostic(self, message: Message) {
            assert_eq!(self.0, Err(Diagnostic::new(message, Marking::Special)))
        }
    }

    fn analyze<C, I>(mnemonic: C, operands: I) -> Result
    where
        C: ToMarked<Command>,
        I: IntoIterator<Item = ParsedExpr<String, Marking>>,
    {
        use frontend::StrExprFactory;
        let mut expr_factory = StrExprFactory::new();
        let mut analyzer = CommandAnalyzer::new(&mut expr_factory);
        Result(analyzer.analyze_instruction(mnemonic.to_marked(), operands))
    }

    #[test]
    fn analyze_nop_a() {
        analyze((Command::Nop, Marking::Special), vec![literal(A)]).expect_diagnostic(
            Message::OperandCount {
                actual: 1,
                expected: 0,
            },
        )
    }

    #[test]
    fn analyze_add_a_a_a() {
        analyze(
            (Command::Add, Marking::Special),
            vec![A, A, A].into_iter().map(|a| literal(a)),
        ).expect_diagnostic(Message::OperandCount {
            actual: 3,
            expected: 2,
        })
    }

    #[test]
    fn analyze_jp_c_deref_hl() {
        analyze(
            Command::Jp,
            vec![literal(C).mark(), SimpleOperand::DerefHl.into()],
        ).expect_diagnostic(Message::AlwaysUnconditional)
    }
}
