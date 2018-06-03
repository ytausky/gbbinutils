use diagnostics::{Diagnostic, Message, Source, SourceInterval};
use frontend::semantics::operand::{AtomKind, Context, Operand, OperandAnalyzer, OperandCounter};
use frontend::syntax::{keyword, ParsedExpr};
use frontend::ExprFactory;
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
        R: SourceInterval,
    {
        let (mnemonic, mnemonic_ref) = (to_mnemonic(mnemonic.0), mnemonic.1);
        let context = match mnemonic {
            Mnemonic::Branch(_) => Context::Branch,
            Mnemonic::Stack(_) => Context::Stack,
            _ => Context::Other,
        };
        Analysis::new(
            (mnemonic, mnemonic_ref),
            operands
                .into_iter()
                .map(|x| self.operand_analyzer.analyze_operand(x, &context)),
        ).run()
    }
}

struct Analysis<R, I> {
    mnemonic: (Mnemonic, R),
    operands: OperandCounter<I>,
}

impl<'a, R: SourceInterval, I: Iterator<Item = Operand<R>>> Analysis<R, I> {
    fn new(mnemonic: (Mnemonic, R), operands: I) -> Analysis<R, I> {
        Analysis {
            mnemonic,
            operands: OperandCounter::new(operands),
        }
    }

    fn run(mut self) -> AnalysisResult<R> {
        let instruction = self.analyze_mnemonic()?;
        self.check_for_unexpected_operands()?;
        Ok(instruction)
    }

    fn analyze_mnemonic(&mut self) -> AnalysisResult<R> {
        use self::Mnemonic::*;
        match self.mnemonic.0 {
            Alu(AluOperation::Add, explicit_a) => self.analyze_add_instruction(explicit_a),
            Alu(operation, explicit_a) => {
                let first_operand = self.operands.next();
                self.analyze_alu_instruction(operation, explicit_a, first_operand)
            }
            IncDec(mode) => self.analyze_inc_dec(mode),
            Branch(branch) => self.analyze_branch(branch),
            Ld => self.analyze_ld(LdHint::Normal),
            Ldh => self.analyze_ld(LdHint::Ldh),
            Nullary(instruction) => Ok(instruction.into()),
            Stack(operation) => self.analyze_stack_operation(operation),
        }
    }

    fn check_for_unexpected_operands(self) -> Result<(), Diagnostic<R>> {
        let expected = self.operands.seen();
        let extra = self.operands.count();
        let actual = expected + extra;
        if actual == expected {
            Ok(())
        } else {
            Err(Diagnostic::new(
                Message::OperandCount { actual, expected },
                self.mnemonic.1,
            ))
        }
    }

    fn analyze_add_instruction(&mut self, explicit_a: ExplicitA) -> AnalysisResult<R> {
        match self.operands.next() {
            Some(Operand::Atom(AtomKind::Reg16(reg16), range)) => {
                self.analyze_add_reg16_instruction((reg16, range))
            }
            operand => self.analyze_alu_instruction(AluOperation::Add, explicit_a, operand),
        }
    }

    fn analyze_add_reg16_instruction(&mut self, target: (Reg16, R)) -> AnalysisResult<R> {
        match target.0 {
            Reg16::Hl => self.analyze_add_hl_instruction(),
            _ => Err(Diagnostic::new(Message::DestMustBeHl, target.1)),
        }
    }

    fn analyze_add_hl_instruction(&mut self) -> AnalysisResult<R> {
        match self.expect_operand(2)? {
            Operand::Atom(AtomKind::Reg16(src), _) => Ok(Instruction::AddHl(src)),
            Operand::Atom(_, interval) => {
                Err(Diagnostic::new(Message::IncompatibleOperand, interval))
            }
            _ => panic!(),
        }
    }

    fn analyze_alu_instruction(
        &mut self,
        operation: AluOperation,
        explicit_a: ExplicitA,
        first_operand: Option<Operand<R>>,
    ) -> AnalysisResult<R> {
        let first_operand = expect_operand(first_operand, 0, 2, self.mnemonic.1.clone())?;
        let src = if explicit_a == ExplicitA::Required {
            let second_operand = self.expect_operand(2)?;
            match first_operand {
                Operand::Atom(AtomKind::Simple(SimpleOperand::A), _) => Ok(()),
                operand => Err(Diagnostic::new(
                    Message::DestMustBeA,
                    operand.source_interval(),
                )),
            }?;
            second_operand
        } else {
            first_operand
        };
        match src {
            Operand::Atom(AtomKind::Simple(src), _) => {
                Ok(Instruction::Alu(operation, AluSource::Simple(src)))
            }
            Operand::Const(expr) => Ok(Instruction::Alu(operation, AluSource::Immediate(expr))),
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

    fn analyze_ld(&mut self, hint: LdHint) -> AnalysisResult<R> {
        let dest = self.operands.next().unwrap();
        let src = self.operands.next().unwrap();
        match (dest, src) {
            (Operand::Atom(AtomKind::Simple(dest), _), Operand::Atom(AtomKind::Simple(src), _)) => {
                Ok(Instruction::Ld(Ld::Simple(dest, src)))
            }
            (Operand::Atom(AtomKind::Simple(dest), _), Operand::Const(expr)) => {
                Ok(Instruction::Ld(Ld::Immediate8(dest, expr)))
            }
            (Operand::Atom(AtomKind::Simple(SimpleOperand::A), _), src) => {
                analyze_special_ld(src, Direction::IntoA, hint)
            }
            (dest, Operand::Atom(AtomKind::Simple(SimpleOperand::A), _)) => {
                analyze_special_ld(dest, Direction::FromA, hint)
            }
            (Operand::Atom(AtomKind::Reg16(dest), _), Operand::Const(expr)) => {
                Ok(Instruction::Ld(Ld::Immediate16(dest, expr)))
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

    fn expect_operand(&mut self, out_of: usize) -> Result<Operand<R>, Diagnostic<R>> {
        expect_operand(
            self.operands.next(),
            self.operands.seen(),
            out_of,
            self.mnemonic.1.clone(),
        )
    }
}

fn expect_operand<R>(
    operand: Option<Operand<R>>,
    actual: usize,
    expected: usize,
    range: R,
) -> Result<Operand<R>, Diagnostic<R>> {
    operand.ok_or_else(|| Diagnostic::new(Message::OperandCount { actual, expected }, range))
}

enum LdHint {
    Normal,
    Ldh,
}

fn analyze_branch_target<R>(target: Option<Operand<R>>) -> Option<TargetSelector<R>> {
    target.map(|target| match target {
        Operand::Const(expr) => TargetSelector::Expr(expr),
        Operand::Atom(AtomKind::Simple(SimpleOperand::DerefHl), _) => TargetSelector::DerefHl,
        _ => panic!(),
    })
}

fn mk_branch<R>(kind: BranchKind, target: Option<TargetSelector<R>>) -> Branch<R> {
    match (kind, target) {
        (BranchKind::Call, Some(TargetSelector::Expr(expr))) => Branch::Call(expr),
        (BranchKind::Jp, Some(TargetSelector::Expr(expr))) => Branch::Jp(expr),
        (BranchKind::Jr, Some(TargetSelector::Expr(expr))) => Branch::Jr(expr),
        (BranchKind::Ret, None) => Branch::Ret,
        _ => panic!(),
    }
}

fn analyze_special_ld<R>(
    other: Operand<R>,
    direction: Direction,
    hint: LdHint,
) -> AnalysisResult<R> {
    Ok(Instruction::Ld(Ld::Special(
        match other {
            Operand::Deref(expr) => match hint {
                LdHint::Normal => SpecialLd::InlineAddr(expr),
                LdHint::Ldh => SpecialLd::InlineIndex(expr),
            },
            Operand::Atom(AtomKind::DerefC, _) => SpecialLd::RegIndex,
            _ => panic!(),
        },
        direction,
    )))
}

pub type AnalysisResult<R> = Result<Instruction<R>, Diagnostic<R>>;

#[derive(Debug, PartialEq)]
enum Mnemonic {
    Alu(AluOperation, ExplicitA),
    Branch(BranchKind),
    IncDec(IncDec),
    Ld,
    Ldh,
    Nullary(Nullary),
    Stack(StackOperation),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ExplicitA {
    Required,
    NotAllowed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum StackOperation {
    Push,
    Pop,
}

impl<R> From<Nullary> for Instruction<R> {
    fn from(nullary: Nullary) -> Instruction<R> {
        Instruction::Nullary(nullary)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BranchKind {
    Call,
    Jp,
    Jr,
    Ret,
}

#[cfg(test)]
impl BranchKind {
    fn has_target(&self) -> bool {
        *self != BranchKind::Ret
    }
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
        Daa => Mnemonic::Nullary(Nullary::Daa),
        Dec => Mnemonic::IncDec(IncDec::Dec),
        Di => Mnemonic::Nullary(Nullary::Di),
        Ei => Mnemonic::Nullary(Nullary::Ei),
        Halt => Mnemonic::Nullary(Nullary::Halt),
        Inc => Mnemonic::IncDec(IncDec::Inc),
        Jp => Mnemonic::Branch(BranchKind::Jp),
        Jr => Mnemonic::Branch(BranchKind::Jr),
        Ld => Mnemonic::Ld,
        Ldh => Mnemonic::Ldh,
        Nop => Mnemonic::Nullary(Nullary::Nop),
        Pop => Mnemonic::Stack(StackOperation::Pop),
        Push => Mnemonic::Stack(StackOperation::Push),
        Ret => Mnemonic::Branch(BranchKind::Ret),
        Reti => Mnemonic::Nullary(Nullary::Reti),
        Stop => Mnemonic::Nullary(Nullary::Stop),
        Xor => Mnemonic::Alu(AluOperation::Xor, ExplicitA::NotAllowed),
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frontend::syntax::{ExprNode, Literal};

    use self::keyword::{Command, Operand::*};

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Marking {
        Normal,
        Special,
    }

    impl SourceInterval for Marking {
        fn extend(&self, _: &Self) -> Self {
            *self
        }
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
        fn mark(mut self) -> Self {
            self.interval = Marking::Special;
            self
        }
    }

    impl From<ExprNode<String, Marking>> for Input {
        fn from(node: ExprNode<String, Marking>) -> Self {
            ParsedExpr {
                node,
                interval: Marking::default(),
            }
        }
    }

    impl From<Literal<String>> for Input {
        fn from(literal: Literal<String>) -> Input {
            ExprNode::Literal(literal).into()
        }
    }

    fn literal(keyword: keyword::Operand) -> Input {
        Literal::Operand(keyword).into()
    }

    fn symbol(ident: &str) -> Expr<Marking> {
        Expr::Symbol(ident.to_string(), Marking::default())
    }

    fn deref(expr: Input) -> Input {
        ParsedExpr {
            node: ExprNode::Deref(Box::new(expr)),
            interval: Marking::default(),
        }
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
                BranchKind::Ret => Command::Ret,
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
                SimpleOperand::DerefHl => deref(literal(Hl)),
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

    impl<'a> From<&'a str> for Input {
        fn from(ident: &'a str) -> Self {
            ExprNode::Ident(ident.to_string()).into()
        }
    }

    impl From<i32> for ParsedExpr<String, Marking> {
        fn from(n: i32) -> Self {
            ExprNode::Literal(Literal::Number(n)).into()
        }
    }

    impl From<i32> for Expr<Marking> {
        fn from(n: i32) -> Self {
            Expr::Literal(n, Marking::default())
        }
    }

    #[test]
    fn analyze_jp_deref_hl() {
        analyze(Command::Jp, vec![deref(literal(Hl))]).expect_instruction(Instruction::JpDerefHl)
    }

    #[test]
    fn analyze_ld_deref_symbol_a() {
        let ident = "ident";
        analyze(Command::Ld, vec![deref(ident.into()), literal(A)]).expect_instruction(
            Instruction::Ld(Ld::Special(
                SpecialLd::InlineAddr(symbol(ident)),
                Direction::FromA,
            )),
        )
    }

    #[test]
    fn analyze_ld_a_deref_symbol() {
        let ident = "ident";
        analyze(Command::Ld, vec![literal(A), deref(ident.into())]).expect_instruction(
            Instruction::Ld(Ld::Special(
                SpecialLd::InlineAddr(symbol(ident)),
                Direction::IntoA,
            )),
        )
    }

    #[test]
    fn analyze_ld_deref_c_a() {
        analyze(Command::Ld, vec![deref(literal(C)), literal(A)]).expect_instruction(
            Instruction::Ld(Ld::Special(SpecialLd::RegIndex, Direction::FromA)),
        )
    }

    #[test]
    fn analyze_ld_a_deref_c() {
        analyze(Command::Ld, vec![literal(A), deref(literal(C))]).expect_instruction(
            Instruction::Ld(Ld::Special(SpecialLd::RegIndex, Direction::IntoA)),
        )
    }

    #[test]
    fn analyze_ldh_from_a() {
        let index = 0xcc;
        analyze(Command::Ldh, vec![deref(index.into()), literal(A)]).expect_instruction(
            Instruction::Ld(Ld::Special(
                SpecialLd::InlineIndex(index.into()),
                Direction::FromA,
            )),
        )
    }

    #[test]
    fn analyze_ldh_into_a() {
        let index = 0xcc;
        analyze(Command::Ldh, vec![literal(A), deref(index.into())]).expect_instruction(
            Instruction::Ld(Ld::Special(
                SpecialLd::InlineIndex(index.into()),
                Direction::IntoA,
            )),
        )
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
        let mut descriptors: Vec<InstructionDescriptor> = Vec::new();
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
            (Command::Daa, Nullary::Daa),
            (Command::Di, Nullary::Di),
            (Command::Ei, Nullary::Ei),
            (Command::Halt, Nullary::Halt),
            (Command::Nop, Nullary::Nop),
            (Command::Reti, Nullary::Reti),
            (Command::Stop, Nullary::Stop),
        ].iter()
            .map(|(mnemonic, nullary)| ((*mnemonic, vec![]), Instruction::Nullary(nullary.clone())))
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
                Instruction::Ld(Ld::Simple(dest, src)),
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
            Instruction::Ld(Ld::Immediate8(dest, n.into())),
        )
    }

    fn describe_ld_reg16_immediate_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        REG16.iter().map(|&dest| describe_ld_reg16_immediate(dest))
    }

    fn describe_ld_reg16_immediate(dest: Reg16) -> InstructionDescriptor {
        let value = "value";
        (
            (Command::Ld, vec![ParsedExpr::from(dest), value.into()]),
            Instruction::Ld(Ld::Immediate16(dest, symbol(value))),
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
        if branch.has_target() {
            operands.push(ident.into());
        };
        (
            (Command::from(branch), operands),
            Instruction::Branch(
                mk_branch(
                    branch,
                    if branch.has_target() {
                        Some(TargetSelector::Expr(symbol(ident)))
                    } else {
                        None
                    },
                ),
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

    const BRANCHES: &[BranchKind] = &[
        BranchKind::Call,
        BranchKind::Jp,
        BranchKind::Jr,
        BranchKind::Ret,
    ];

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

    #[test]
    fn analyze_add() {
        analyze((Command::Add, Marking::Special), Vec::new()).expect_diagnostic(
            Message::OperandCount {
                actual: 0,
                expected: 2,
            },
        )
    }

    #[test]
    fn analyze_add_a() {
        analyze((Command::Add, Marking::Special), vec![literal(A)]).expect_diagnostic(
            Message::OperandCount {
                actual: 1,
                expected: 2,
            },
        )
    }

    #[test]
    fn analyze_add_b_a() {
        analyze(Command::Add, vec![literal(B).mark(), literal(A)])
            .expect_diagnostic(Message::DestMustBeA)
    }

    #[test]
    fn analyze_add_bc_de() {
        analyze(Command::Add, vec![literal(Bc).mark(), literal(De)])
            .expect_diagnostic(Message::DestMustBeHl)
    }

    #[test]
    fn analyze_add_hl_af() {
        analyze(Command::Add, vec![literal(Hl), literal(Af).mark()])
            .expect_diagnostic(Message::IncompatibleOperand)
    }

    #[test]
    fn analyze_add_hl() {
        analyze((Command::Add, Marking::Special), vec![literal(Hl)]).expect_diagnostic(
            Message::OperandCount {
                actual: 1,
                expected: 2,
            },
        )
    }
}
