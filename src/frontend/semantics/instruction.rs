use super::Expr;
use diagnostics::{InternalDiagnostic, Message};
use frontend::semantics::operand::{self, AtomKind, Context, Operand, OperandCounter};
use frontend::syntax::keyword as kw;
use instruction::*;
use span::{Source, Span};
use std::iter;
use Width;

pub fn analyze_instruction<Id: Into<String>, I, S>(
    mnemonic: (kw::Mnemonic, S),
    operands: I,
) -> AnalysisResult<S>
where
    I: IntoIterator<Item = Expr<Id, S>>,
    S: Span,
{
    let mnemonic: (Mnemonic, S) = (mnemonic.0.into(), mnemonic.1);
    let context = mnemonic.0.context();
    Analysis::new(
        mnemonic,
        operands
            .into_iter()
            .map(|x| operand::analyze_operand(x, context)),
    ).run()
}

struct Analysis<R, I> {
    mnemonic: (Mnemonic, R),
    operands: OperandCounter<I>,
}

impl<'a, S: Span, I: Iterator<Item = Result<Operand<S>, InternalDiagnostic<S>>>> Analysis<S, I> {
    fn new(mnemonic: (Mnemonic, S), operands: I) -> Analysis<S, I> {
        Analysis {
            mnemonic,
            operands: OperandCounter::new(operands),
        }
    }

    fn run(mut self) -> AnalysisResult<S> {
        let instruction = self.analyze_mnemonic()?;
        self.operands
            .check_for_unexpected_operands(self.mnemonic.1)?;
        Ok(instruction)
    }

    fn analyze_mnemonic(&mut self) -> AnalysisResult<S> {
        use self::Mnemonic::*;
        match self.mnemonic.0 {
            Alu(AluOperation::Add) => self.analyze_add_instruction(),
            Alu(operation) => {
                let first_operand = self.next_operand_out_of(operation.expected_operands())?;
                self.analyze_alu_instruction(operation, first_operand)
            }
            Bit(operation) => self.analyze_bit_operation(operation),
            IncDec(mode) => self.analyze_inc_dec(mode),
            Branch(branch) => self.analyze_branch(branch),
            Ld => self.analyze_ld(),
            Ldhl => self.analyze_ldhl(),
            Misc(operation) => self.analyze_misc(operation),
            Nullary(instruction) => Ok(instruction.into()),
            Rst => self.analyze_rst(),
            Stack(operation) => self.analyze_stack_operation(operation),
        }
    }

    fn analyze_add_instruction(&mut self) -> AnalysisResult<S> {
        match self.next_operand_out_of(2)? {
            Operand::Atom(AtomKind::Reg16(reg16), range) => {
                self.analyze_add_reg16_instruction((reg16, range))
            }
            operand => self.analyze_alu_instruction(AluOperation::Add, operand),
        }
    }

    fn analyze_add_reg16_instruction(&mut self, target: (Reg16, S)) -> AnalysisResult<S> {
        match target.0 {
            Reg16::Hl => self.analyze_add_hl_instruction(),
            _ => Err(InternalDiagnostic::new(
                Message::DestMustBeHl,
                iter::empty(),
                target.1,
            )),
        }
    }

    fn analyze_add_hl_instruction(&mut self) -> AnalysisResult<S> {
        match self.next_operand_out_of(2)? {
            Operand::Atom(AtomKind::Reg16(src), _) => Ok(Instruction::AddHl(src)),
            operand => Err(InternalDiagnostic::new(
                Message::IncompatibleOperand,
                iter::empty(),
                operand.span(),
            )),
        }
    }

    fn analyze_alu_instruction(
        &mut self,
        operation: AluOperation,
        first_operand: Operand<S>,
    ) -> AnalysisResult<S> {
        let src = if operation.implicit_dest() {
            first_operand
        } else {
            let second_operand = self.next_operand_out_of(2)?;
            first_operand
                .expect_specific_atom(AtomKind::Simple(SimpleOperand::A), Message::DestMustBeA)?;
            second_operand
        };
        match src {
            Operand::Atom(AtomKind::Simple(src), _) => {
                Ok(Instruction::Alu(operation, AluSource::Simple(src)))
            }
            Operand::Const(expr) => Ok(Instruction::Alu(operation, AluSource::Immediate(expr))),
            src => Err(InternalDiagnostic::new(
                Message::IncompatibleOperand,
                iter::empty(),
                src.span(),
            )),
        }
    }

    fn analyze_bit_operation(&mut self, operation: BitOperation) -> AnalysisResult<S> {
        let bit_number = self.next_operand_out_of(2)?;
        let operand = self.next_operand_out_of(2)?;
        let expr = if let Operand::Const(expr) = bit_number {
            expr
        } else {
            return Err(InternalDiagnostic::new(
                Message::MustBeBit,
                iter::once(self.mnemonic.1.clone()),
                bit_number.span(),
            ));
        };
        Ok(Instruction::Bit(operation, expr, operand.expect_simple()?))
    }

    fn analyze_branch(&mut self, branch: BranchKind) -> AnalysisResult<S> {
        let (condition, target) = self.collect_condition_and_target()?;
        match (branch, condition, target) {
            (BranchKind::Jp, None, Some(TargetSelector::DerefHl)) => Ok(Instruction::JpDerefHl),
            (BranchKind::Jp, Some((_, condition_ref)), Some(TargetSelector::DerefHl)) => Err(
                InternalDiagnostic::new(Message::AlwaysUnconditional, iter::empty(), condition_ref),
            ),
            (_, condition, target) => Ok(Instruction::Branch(
                mk_branch((branch, self.mnemonic.1.clone()), target)?,
                condition.map(|(condition, _)| condition),
            )),
        }
    }

    fn collect_condition_and_target(
        &mut self,
    ) -> Result<CondtitionTargetPair<S>, InternalDiagnostic<S>> {
        let first_operand = self.operands.next()?;
        Ok(
            if let Some(Operand::Atom(AtomKind::Condition(condition), range)) = first_operand {
                (
                    Some((condition, range)),
                    analyze_branch_target(self.operands.next()?),
                )
            } else {
                (None, analyze_branch_target(first_operand))
            },
        )
    }

    fn analyze_ld(&mut self) -> AnalysisResult<S> {
        let dest = self.next_operand_out_of(2)?;
        let src = self.next_operand_out_of(2)?;
        match (dest.into_ld_operand()?, src.into_ld_operand()?) {
            (LdOperand::Byte(dest), LdOperand::Byte(src)) => self.analyze_8_bit_ld(dest, src),
            (LdOperand::Byte(dest), LdOperand::Const(src)) => {
                self.analyze_8_bit_ld(dest, ByteOperand::Const(src))
            }
            (LdOperand::Word(dest), LdOperand::Word(src)) => self.analyze_16_bit_ld(dest, src),
            (LdOperand::Word(dest), LdOperand::Const(src)) => {
                self.analyze_16_bit_ld(dest, WordOperand::Const(src))
            }
            (LdOperand::Const(dest), src) => Err(InternalDiagnostic::new(
                Message::IllegalOperands,
                iter::once(self.mnemonic.1.clone()),
                dest.span().extend(&src.span()),
            )),
            (LdOperand::Byte(dest), LdOperand::Word(src)) => Err(InternalDiagnostic::new(
                Message::LdWidthMismatch { src: Width::Word },
                vec![src.span(), dest.span()],
                dest.span().extend(&src.span()),
            )),
            (LdOperand::Word(dest), LdOperand::Byte(src)) => Err(InternalDiagnostic::new(
                Message::LdWidthMismatch { src: Width::Byte },
                vec![src.span(), dest.span()],
                dest.span().extend(&src.span()),
            )),
        }
        /*(dest, src) => Err(InternalDiagnostic::new(
                Message::IllegalOperands,
                iter::once(self.mnemonic.1.clone()),
                dest.span().extend(&src.span()),
            )),*/
    }

    fn analyze_8_bit_ld(&mut self, dest: ByteOperand<S>, src: ByteOperand<S>) -> AnalysisResult<S> {
        match (dest, src) {
            (ByteOperand::Simple(dest, _), ByteOperand::Simple(src, _)) => {
                Ok(Instruction::Ld(Ld::Simple(dest, src)))
            }
            (ByteOperand::Simple(dest, _), ByteOperand::Const(expr)) => {
                Ok(Instruction::Ld(Ld::Immediate8(dest, expr)))
            }
            (ByteOperand::Simple(SimpleOperand::A, _), src) => {
                analyze_special_ld(src, Direction::IntoA)
            }
            (dest, ByteOperand::Simple(SimpleOperand::A, _)) => {
                analyze_special_ld(dest, Direction::FromA)
            }
            _ => panic!(),
        }
    }

    fn analyze_16_bit_ld(
        &mut self,
        dest: WordOperand<S>,
        src: WordOperand<S>,
    ) -> AnalysisResult<S> {
        match (dest, src) {
            (WordOperand::Reg16(Reg16::Sp, _), WordOperand::Reg16(Reg16::Hl, _)) => {
                Ok(Instruction::Ld(Ld::SpHl))
            }
            (WordOperand::Reg16(dest, _), WordOperand::Const(expr)) => {
                Ok(Instruction::Ld(Ld::Immediate16(dest, expr)))
            }
            _ => panic!(),
        }
    }

    fn analyze_ldhl(&mut self) -> AnalysisResult<S> {
        let src = self.next_operand_out_of(2)?;
        let offset = self.next_operand_out_of(2)?;
        src.expect_specific_atom(AtomKind::Reg16(Reg16::Sp), Message::SrcMustBeSp)?;
        Ok(Instruction::Ldhl(offset.expect_const()?))
    }

    fn analyze_misc(&mut self, operation: MiscOperation) -> AnalysisResult<S> {
        let operand = self.next_operand_out_of(1)?;
        Ok(Instruction::Misc(operation, operand.expect_simple()?))
    }

    fn analyze_stack_operation(&mut self, operation: StackOperation) -> AnalysisResult<S> {
        let reg_pair = self.next_operand_out_of(1)?.expect_reg_pair()?;
        let instruction_ctor = match operation {
            StackOperation::Push => Instruction::Push,
            StackOperation::Pop => Instruction::Pop,
        };
        Ok(instruction_ctor(reg_pair))
    }

    fn analyze_inc_dec(&mut self, mode: IncDec) -> AnalysisResult<S> {
        match self.next_operand_out_of(1)? {
            Operand::Atom(AtomKind::Simple(operand), _) => Ok(Instruction::IncDec8(mode, operand)),
            Operand::Atom(AtomKind::Reg16(operand), _) => Ok(Instruction::IncDec16(mode, operand)),
            _ => panic!(),
        }
    }

    fn analyze_rst(&mut self) -> AnalysisResult<S> {
        Ok(Instruction::Rst(
            self.next_operand_out_of(1)?.expect_const()?,
        ))
    }

    fn next_operand_out_of(&mut self, out_of: usize) -> Result<Operand<S>, InternalDiagnostic<S>> {
        let actual = self.operands.seen();
        self.operands.next()?.ok_or_else(|| {
            InternalDiagnostic::new(
                Message::OperandCount {
                    actual,
                    expected: out_of,
                },
                iter::empty(),
                self.mnemonic.1.clone(),
            )
        })
    }
}

impl<S: Span> Operand<S> {
    fn into_ld_operand(self) -> Result<LdOperand<S>, InternalDiagnostic<S>> {
        match self {
            Operand::Const(expr) => Ok(LdOperand::Const(expr)),
            Operand::Deref(expr) => Ok(LdOperand::Byte(ByteOperand::Deref(expr))),
            Operand::Atom(kind, span) => match kind {
                AtomKind::Condition(_) => panic!(),
                AtomKind::Simple(simple) => Ok(LdOperand::Byte(ByteOperand::Simple(simple, span))),
                AtomKind::DerefC => Ok(LdOperand::Byte(ByteOperand::DerefC(span))),
                AtomKind::DerefPtrReg(ptr_reg) => {
                    Ok(LdOperand::Byte(ByteOperand::DerefPtrReg(ptr_reg, span)))
                }
                AtomKind::Reg16(reg16) => Ok(LdOperand::Word(WordOperand::Reg16(reg16, span))),
                AtomKind::RegPair(_) => panic!(),
            },
        }
    }

    fn expect_specific_atom(
        self,
        expected: AtomKind,
        message: Message,
    ) -> Result<(), InternalDiagnostic<S>> {
        match self {
            Operand::Atom(ref actual, _) if *actual == expected => Ok(()),
            operand => operand.error(message),
        }
    }

    fn expect_simple(self) -> Result<SimpleOperand, InternalDiagnostic<S>> {
        match self {
            Operand::Atom(AtomKind::Simple(simple), _) => Ok(simple),
            operand => operand.error(Message::RequiresSimpleOperand),
        }
    }

    fn expect_const(self) -> Result<RelocExpr<S>, InternalDiagnostic<S>> {
        match self {
            Operand::Const(expr) => Ok(expr),
            operand => operand.error(Message::MustBeConst),
        }
    }

    fn expect_reg_pair(self) -> Result<RegPair, InternalDiagnostic<S>> {
        match self {
            Operand::Atom(AtomKind::RegPair(reg_pair), _) => Ok(reg_pair),
            operand => operand.error(Message::RequiresRegPair),
        }
    }

    fn error<T>(self, message: Message) -> Result<T, InternalDiagnostic<S>> {
        Err(InternalDiagnostic::new(message, iter::empty(), self.span()))
    }
}

enum LdOperand<S> {
    Byte(ByteOperand<S>),
    Const(RelocExpr<S>),
    Word(WordOperand<S>),
}

enum ByteOperand<S> {
    Const(RelocExpr<S>),
    Deref(RelocExpr<S>),
    DerefC(S),
    DerefPtrReg(PtrReg, S),
    Simple(SimpleOperand, S),
}

enum WordOperand<S> {
    Const(RelocExpr<S>),
    Reg16(Reg16, S),
}

impl<S: Span> Source for LdOperand<S> {
    type Span = S;
    fn span(&self) -> Self::Span {
        match self {
            LdOperand::Byte(byte) => byte.span(),
            LdOperand::Const(expr) => expr.span(),
            LdOperand::Word(word) => word.span(),
        }
    }
}

impl<S: Span> Source for ByteOperand<S> {
    type Span = S;
    fn span(&self) -> Self::Span {
        use self::ByteOperand::*;
        match self {
            Const(expr) | Deref(expr) => expr.span(),
            DerefC(span) | DerefPtrReg(_, span) | Simple(_, span) => span.clone(),
        }
    }
}

impl<S: Span> Source for WordOperand<S> {
    type Span = S;
    fn span(&self) -> Self::Span {
        use self::WordOperand::*;
        match self {
            Const(expr) => expr.span(),
            Reg16(_, span) => span.clone(),
        }
    }
}

type CondtitionTargetPair<S> = (Option<(Condition, S)>, Option<TargetSelector<S>>);

fn analyze_branch_target<S>(target: Option<Operand<S>>) -> Option<TargetSelector<S>> {
    target.map(|target| match target {
        Operand::Const(expr) => TargetSelector::Expr(expr),
        Operand::Atom(AtomKind::Simple(SimpleOperand::DerefHl), _) => TargetSelector::DerefHl,
        _ => panic!(),
    })
}

fn mk_branch<S>(
    kind: (BranchKind, S),
    target: Option<TargetSelector<S>>,
) -> Result<Branch<S>, InternalDiagnostic<S>> {
    match (kind.0, target) {
        (BranchKind::Call, Some(TargetSelector::Expr(expr))) => Ok(Branch::Call(expr)),
        (BranchKind::Jp, Some(TargetSelector::Expr(expr))) => Ok(Branch::Jp(expr)),
        (BranchKind::Jp, None) => Err(InternalDiagnostic::new(
            Message::MissingTarget,
            iter::empty(),
            kind.1,
        )),
        (BranchKind::Jr, Some(TargetSelector::Expr(expr))) => Ok(Branch::Jr(expr)),
        (BranchKind::Ret, None) => Ok(Branch::Ret),
        _ => panic!(),
    }
}

fn analyze_special_ld<S>(other: ByteOperand<S>, direction: Direction) -> AnalysisResult<S> {
    Ok(Instruction::Ld(Ld::Special(
        match other {
            ByteOperand::Deref(expr) => SpecialLd::InlineAddr(expr),
            ByteOperand::DerefC(_) => SpecialLd::RegIndex,
            ByteOperand::DerefPtrReg(ptr_reg, _) => SpecialLd::DerefPtrReg(ptr_reg),
            _ => panic!(),
        },
        direction,
    )))
}

pub type AnalysisResult<S> = Result<Instruction<S>, InternalDiagnostic<S>>;

#[derive(Debug, PartialEq)]
enum Mnemonic {
    Alu(AluOperation),
    Bit(BitOperation),
    Branch(BranchKind),
    IncDec(IncDec),
    Ld,
    Ldhl,
    Misc(MiscOperation),
    Nullary(Nullary),
    Rst,
    Stack(StackOperation),
}

impl Mnemonic {
    fn context(&self) -> Context {
        match *self {
            Mnemonic::Branch(_) => Context::Branch,
            Mnemonic::Stack(_) => Context::Stack,
            _ => Context::Other,
        }
    }
}

impl AluOperation {
    fn expected_operands(self) -> usize {
        if self.implicit_dest() {
            1
        } else {
            2
        }
    }

    fn implicit_dest(self) -> bool {
        use instruction::AluOperation::*;
        match self {
            Add | Adc | Sbc => false,
            _ => true,
        }
    }
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
    Expr(RelocExpr<R>),
}

impl From<kw::Mnemonic> for Mnemonic {
    fn from(keyword: kw::Mnemonic) -> Self {
        use self::kw::Mnemonic::*;
        match keyword {
            Adc => Mnemonic::Alu(AluOperation::Adc),
            Add => Mnemonic::Alu(AluOperation::Add),
            And => Mnemonic::Alu(AluOperation::And),
            Bit => Mnemonic::Bit(BitOperation::Bit),
            Call => Mnemonic::Branch(BranchKind::Call),
            Cp => Mnemonic::Alu(AluOperation::Cp),
            Cpl => Mnemonic::Nullary(Nullary::Cpl),
            Daa => Mnemonic::Nullary(Nullary::Daa),
            Dec => Mnemonic::IncDec(IncDec::Dec),
            Di => Mnemonic::Nullary(Nullary::Di),
            Ei => Mnemonic::Nullary(Nullary::Ei),
            Halt => Mnemonic::Nullary(Nullary::Halt),
            Inc => Mnemonic::IncDec(IncDec::Inc),
            Jp => Mnemonic::Branch(BranchKind::Jp),
            Jr => Mnemonic::Branch(BranchKind::Jr),
            Ld => Mnemonic::Ld,
            Ldhl => Mnemonic::Ldhl,
            Nop => Mnemonic::Nullary(Nullary::Nop),
            Or => Mnemonic::Alu(AluOperation::Or),
            Pop => Mnemonic::Stack(StackOperation::Pop),
            Push => Mnemonic::Stack(StackOperation::Push),
            Res => Mnemonic::Bit(BitOperation::Res),
            Ret => Mnemonic::Branch(BranchKind::Ret),
            Reti => Mnemonic::Nullary(Nullary::Reti),
            Rl => Mnemonic::Misc(MiscOperation::Rl),
            Rla => Mnemonic::Nullary(Nullary::Rla),
            Rlc => Mnemonic::Misc(MiscOperation::Rlc),
            Rlca => Mnemonic::Nullary(Nullary::Rlca),
            Rr => Mnemonic::Misc(MiscOperation::Rr),
            Rra => Mnemonic::Nullary(Nullary::Rra),
            Rrc => Mnemonic::Misc(MiscOperation::Rrc),
            Rrca => Mnemonic::Nullary(Nullary::Rrca),
            Rst => Mnemonic::Rst,
            Sbc => Mnemonic::Alu(AluOperation::Sbc),
            Set => Mnemonic::Bit(BitOperation::Set),
            Sla => Mnemonic::Misc(MiscOperation::Sla),
            Sra => Mnemonic::Misc(MiscOperation::Sra),
            Srl => Mnemonic::Misc(MiscOperation::Srl),
            Stop => Mnemonic::Nullary(Nullary::Stop),
            Sub => Mnemonic::Alu(AluOperation::Sub),
            Swap => Mnemonic::Misc(MiscOperation::Swap),
            Xor => Mnemonic::Alu(AluOperation::Xor),
        }
    }
}

#[cfg(test)]
mod tests {
    use self::kw::Operand::*;
    use super::*;
    use frontend::semantics::ExprVariant;
    use frontend::syntax::Literal;
    use std::cmp;

    type Input = Expr<String, ()>;

    impl From<ExprVariant<String, ()>> for Input {
        fn from(variant: ExprVariant<String, ()>) -> Self {
            Expr { variant, span: () }
        }
    }

    impl From<Literal<String>> for Input {
        fn from(literal: Literal<String>) -> Input {
            ExprVariant::Literal(literal).into()
        }
    }

    fn literal(keyword: kw::Operand) -> Input {
        Literal::Operand(keyword).into()
    }

    fn number(n: i32, span: impl Into<TokenSpan>) -> RelocExpr<TokenSpan> {
        RelocExpr::Literal(n, span.into())
    }

    fn symbol(ident: &str, span: impl Into<TokenSpan>) -> RelocExpr<TokenSpan> {
        RelocExpr::Symbol(ident.to_string(), span.into())
    }

    fn deref(expr: Input) -> Input {
        Expr {
            variant: ExprVariant::Parentheses(Box::new(expr)),
            span: (),
        }
    }

    impl From<AluOperation> for kw::Mnemonic {
        fn from(alu_operation: AluOperation) -> Self {
            match alu_operation {
                AluOperation::Add => kw::Mnemonic::Add,
                AluOperation::Adc => kw::Mnemonic::Adc,
                AluOperation::Sub => kw::Mnemonic::Sub,
                AluOperation::Sbc => kw::Mnemonic::Sbc,
                AluOperation::And => kw::Mnemonic::And,
                AluOperation::Xor => kw::Mnemonic::Xor,
                AluOperation::Or => kw::Mnemonic::Or,
                AluOperation::Cp => kw::Mnemonic::Cp,
            }
        }
    }

    impl From<BitOperation> for kw::Mnemonic {
        fn from(operation: BitOperation) -> Self {
            match operation {
                BitOperation::Bit => kw::Mnemonic::Bit,
                BitOperation::Set => kw::Mnemonic::Set,
                BitOperation::Res => kw::Mnemonic::Res,
            }
        }
    }

    impl From<MiscOperation> for kw::Mnemonic {
        fn from(operation: MiscOperation) -> Self {
            match operation {
                MiscOperation::Rlc => kw::Mnemonic::Rlc,
                MiscOperation::Rrc => kw::Mnemonic::Rrc,
                MiscOperation::Rl => kw::Mnemonic::Rl,
                MiscOperation::Rr => kw::Mnemonic::Rr,
                MiscOperation::Sla => kw::Mnemonic::Sla,
                MiscOperation::Sra => kw::Mnemonic::Sra,
                MiscOperation::Swap => kw::Mnemonic::Swap,
                MiscOperation::Srl => kw::Mnemonic::Srl,
            }
        }
    }

    impl From<BranchKind> for kw::Mnemonic {
        fn from(branch: BranchKind) -> Self {
            match branch {
                BranchKind::Call => kw::Mnemonic::Call,
                BranchKind::Jp => kw::Mnemonic::Jp,
                BranchKind::Jr => kw::Mnemonic::Jr,
                BranchKind::Ret => kw::Mnemonic::Ret,
            }
        }
    }

    impl From<IncDec> for kw::Mnemonic {
        fn from(mode: IncDec) -> Self {
            match mode {
                IncDec::Inc => kw::Mnemonic::Inc,
                IncDec::Dec => kw::Mnemonic::Dec,
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

    impl From<PtrReg> for kw::Operand {
        fn from(ptr_reg: PtrReg) -> Self {
            match ptr_reg {
                PtrReg::Bc => Bc,
                PtrReg::De => De,
                PtrReg::Hli => Hli,
                PtrReg::Hld => Hld,
            }
        }
    }

    impl From<Reg16> for kw::Operand {
        fn from(reg16: Reg16) -> Self {
            match reg16 {
                Reg16::Bc => Bc,
                Reg16::De => De,
                Reg16::Hl => Hl,
                Reg16::Sp => Sp,
            }
        }
    }

    impl<T> From<T> for Input
    where
        kw::Operand: From<T>,
    {
        fn from(src: T) -> Self {
            literal(src.into())
        }
    }

    impl From<RegPair> for Input {
        fn from(reg_pair: RegPair) -> Self {
            literal(match reg_pair {
                RegPair::Bc => Bc,
                RegPair::De => De,
                RegPair::Hl => Hl,
                RegPair::Af => Af,
            })
        }
    }

    impl From<Condition> for Input {
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
            ExprVariant::Ident(ident.to_string()).into()
        }
    }

    impl From<i32> for Input {
        fn from(n: i32) -> Self {
            ExprVariant::Literal(Literal::Number(n)).into()
        }
    }

    impl From<i32> for RelocExpr<()> {
        fn from(n: i32) -> Self {
            RelocExpr::Literal(n, ())
        }
    }

    #[test]
    fn analyze_jp_deref_hl() {
        analyze(kw::Mnemonic::Jp, vec![deref(literal(Hl))])
            .expect_instruction(Instruction::JpDerefHl)
    }

    #[test]
    fn analyze_ld_deref_symbol_a() {
        let ident = "ident";
        analyze(kw::Mnemonic::Ld, vec![deref(ident.into()), literal(A)]).expect_instruction(
            Instruction::Ld(Ld::Special(
                SpecialLd::InlineAddr(symbol(ident, TokenId::Operand(0, 1))),
                Direction::FromA,
            )),
        )
    }

    #[test]
    fn analyze_ld_a_deref_symbol() {
        let ident = "ident";
        analyze(kw::Mnemonic::Ld, vec![literal(A), deref(ident.into())]).expect_instruction(
            Instruction::Ld(Ld::Special(
                SpecialLd::InlineAddr(symbol(ident, TokenId::Operand(1, 1))),
                Direction::IntoA,
            )),
        )
    }

    #[test]
    fn analyze_ld_deref_c_a() {
        analyze(kw::Mnemonic::Ld, vec![deref(literal(C)), literal(A)]).expect_instruction(
            Instruction::Ld(Ld::Special(SpecialLd::RegIndex, Direction::FromA)),
        )
    }

    #[test]
    fn analyze_ld_a_deref_c() {
        analyze(kw::Mnemonic::Ld, vec![literal(A), deref(literal(C))]).expect_instruction(
            Instruction::Ld(Ld::Special(SpecialLd::RegIndex, Direction::IntoA)),
        )
    }

    #[test]
    fn analyze_cp_symbol() {
        let ident = "ident";
        test_cp_const_analysis(ident.into(), symbol(ident, TokenId::Operand(0, 0)))
    }

    #[test]
    fn analyze_cp_literal() {
        let n = 0x50;
        test_cp_const_analysis(n.into(), number(n, TokenId::Operand(0, 0)))
    }

    fn test_cp_const_analysis(parsed: Input, expr: RelocExpr<TokenSpan>) {
        analyze(kw::Mnemonic::Cp, Some(parsed)).expect_instruction(Instruction::Alu(
            AluOperation::Cp,
            AluSource::Immediate(expr),
        ))
    }

    #[test]
    fn analyze_rst() {
        let n = 3;
        analyze(kw::Mnemonic::Rst, vec![n.into()])
            .expect_instruction(Instruction::Rst(number(n, TokenId::Operand(0, 0))))
    }

    #[test]
    fn analyze_legal_instructions() {
        test_instruction_analysis(describe_legal_instructions());
    }

    type InstructionDescriptor = ((kw::Mnemonic, Vec<Input>), Instruction<TokenSpan>);

    fn describe_legal_instructions() -> Vec<InstructionDescriptor> {
        let mut descriptors: Vec<InstructionDescriptor> = Vec::new();
        descriptors.extend(describe_nullary_instructions());
        descriptors.extend(describe_ld_simple_simple_instructions());
        descriptors.extend(describe_ld_simple_immediate_instructions());
        descriptors.extend(describe_ld_reg16_immediate_instructions());
        descriptors.extend(describe_ld_deref_reg16_instructions());
        descriptors.extend(describe_alu_simple_instructions());
        descriptors.extend(describe_add_hl_reg16_instructions());
        descriptors.extend(describe_bit_operation_instructions());
        descriptors.extend(describe_branch_instuctions());
        descriptors.extend(describe_inc_dec8_instructions());
        descriptors.extend(describe_inc_dec16_instructions());
        descriptors.extend(describe_push_pop_instructions());
        descriptors.extend(describe_misc_operation_instructions());
        descriptors.push((
            (kw::Mnemonic::Ld, vec![Reg16::Sp.into(), Reg16::Hl.into()]),
            Instruction::Ld(Ld::SpHl),
        ));
        descriptors.push((
            (kw::Mnemonic::Ldhl, vec![Reg16::Sp.into(), 0x42.into()]),
            Instruction::Ldhl(number(0x42, TokenId::Operand(1, 0))),
        ));
        descriptors
    }

    fn describe_push_pop_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        REG_PAIRS.iter().flat_map(|&reg_pair| {
            vec![
                (
                    (kw::Mnemonic::Push, vec![reg_pair.into()]),
                    Instruction::Push(reg_pair),
                ),
                (
                    (kw::Mnemonic::Pop, vec![reg_pair.into()]),
                    Instruction::Pop(reg_pair),
                ),
            ]
        })
    }

    fn describe_nullary_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        [
            (kw::Mnemonic::Cpl, Nullary::Cpl),
            (kw::Mnemonic::Daa, Nullary::Daa),
            (kw::Mnemonic::Di, Nullary::Di),
            (kw::Mnemonic::Ei, Nullary::Ei),
            (kw::Mnemonic::Halt, Nullary::Halt),
            (kw::Mnemonic::Nop, Nullary::Nop),
            (kw::Mnemonic::Reti, Nullary::Reti),
            (kw::Mnemonic::Rla, Nullary::Rla),
            (kw::Mnemonic::Rlca, Nullary::Rlca),
            (kw::Mnemonic::Rra, Nullary::Rra),
            (kw::Mnemonic::Rrca, Nullary::Rrca),
            (kw::Mnemonic::Stop, Nullary::Stop),
        ]
            .iter()
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
                (kw::Mnemonic::Ld, vec![dest.into(), src.into()]),
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
            (kw::Mnemonic::Ld, vec![Expr::from(dest), n.into()]),
            Instruction::Ld(Ld::Immediate8(dest, number(n, TokenId::Operand(1, 0)))),
        )
    }

    fn describe_ld_reg16_immediate_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        REG16.iter().map(|&dest| describe_ld_reg16_immediate(dest))
    }

    fn describe_ld_reg16_immediate(dest: Reg16) -> InstructionDescriptor {
        let value = "value";
        (
            (kw::Mnemonic::Ld, vec![Expr::from(dest), value.into()]),
            Instruction::Ld(Ld::Immediate16(dest, symbol(value, TokenId::Operand(1, 0)))),
        )
    }

    fn describe_ld_deref_reg16_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        PTR_REGS
            .iter()
            .flat_map(|&addr| describe_ld_deref_ptr_reg(addr))
    }

    fn describe_ld_deref_ptr_reg(ptr_reg: PtrReg) -> impl Iterator<Item = InstructionDescriptor> {
        vec![
            (
                (
                    kw::Mnemonic::Ld,
                    vec![deref(Expr::from(ptr_reg)), literal(A)],
                ),
                Instruction::Ld(Ld::Special(
                    SpecialLd::DerefPtrReg(ptr_reg),
                    Direction::FromA,
                )),
            ),
            (
                (
                    kw::Mnemonic::Ld,
                    vec![literal(A), deref(Expr::from(ptr_reg))],
                ),
                Instruction::Ld(Ld::Special(
                    SpecialLd::DerefPtrReg(ptr_reg),
                    Direction::IntoA,
                )),
            ),
        ].into_iter()
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
            (kw::Mnemonic::from(operation), vec![Expr::from(operand)]),
            Instruction::Alu(operation, AluSource::Simple(operand)),
        )
    }

    fn describe_add_hl_reg16_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        REG16.iter().map(|&reg16| describe_add_hl_reg16(reg16))
    }

    fn describe_add_hl_reg16(reg16: Reg16) -> InstructionDescriptor {
        (
            (kw::Mnemonic::Add, vec![Reg16::Hl.into(), reg16.into()]),
            Instruction::AddHl(reg16),
        )
    }

    fn describe_bit_operation_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        BIT_OPERATIONS.iter().flat_map(|&operation| {
            SIMPLE_OPERANDS
                .iter()
                .map(move |&operand| describe_bit_operation(operation, operand))
        })
    }

    fn describe_bit_operation(
        operation: BitOperation,
        operand: SimpleOperand,
    ) -> InstructionDescriptor {
        let bit_number = 4;
        (
            (operation.into(), vec![bit_number.into(), operand.into()]),
            Instruction::Bit(
                operation,
                number(bit_number, TokenId::Operand(0, 0)),
                operand,
            ),
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
        let mut has_condition = false;
        if let Some(condition) = condition {
            operands.push(Expr::from(condition));
            has_condition = true;
        };
        if branch.has_target() {
            operands.push(ident.into());
        };
        (
            (kw::Mnemonic::from(branch), operands),
            Instruction::Branch(
                mk_branch(
                    (branch, TokenId::Mnemonic.into()),
                    if branch.has_target() {
                        Some(TargetSelector::Expr(symbol(
                            ident,
                            TokenId::Operand(if has_condition { 1 } else { 0 }, 0),
                        )))
                    } else {
                        None
                    },
                ).unwrap(),
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

    fn describe_misc_operation_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        MISC_OPERATIONS.iter().flat_map(|&operation| {
            SIMPLE_OPERANDS.iter().map(move |&operand| {
                (
                    (operation.into(), vec![operand.into()]),
                    Instruction::Misc(operation, operand),
                )
            })
        })
    }

    const ALU_OPERATIONS_WITH_A: &[AluOperation] =
        &[AluOperation::Add, AluOperation::Adc, AluOperation::Sbc];

    const ALU_OPERATIONS_WITHOUT_A: &[AluOperation] = &[
        AluOperation::Sub,
        AluOperation::And,
        AluOperation::Xor,
        AluOperation::Or,
        AluOperation::Cp,
    ];

    const BIT_OPERATIONS: &[BitOperation] =
        &[BitOperation::Bit, BitOperation::Set, BitOperation::Res];

    const MISC_OPERATIONS: &[MiscOperation] = &[
        MiscOperation::Rlc,
        MiscOperation::Rrc,
        MiscOperation::Rl,
        MiscOperation::Rr,
        MiscOperation::Sla,
        MiscOperation::Sra,
        MiscOperation::Swap,
        MiscOperation::Srl,
    ];

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

    const PTR_REGS: &[PtrReg] = &[PtrReg::Bc, PtrReg::De, PtrReg::Hli, PtrReg::Hld];

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

    struct Result(AnalysisResult<TokenSpan>);

    impl Result {
        fn expect_instruction(self, expected: Instruction<TokenSpan>) {
            assert_eq!(self.0, Ok(expected))
        }

        fn expect_diagnostic(self, diagnostic: impl Into<ExpectedDiagnostic>) {
            let expected = diagnostic.into();
            assert_eq!(
                self.0,
                Err(InternalDiagnostic::new(
                    expected.message,
                    expected.spans,
                    expected.highlight.unwrap(),
                ))
            )
        }
    }

    fn analyze<I>(mnemonic: kw::Mnemonic, operands: I) -> Result
    where
        I: IntoIterator<Item = Input>,
    {
        Result(analyze_instruction(
            (mnemonic, TokenId::Mnemonic.into()),
            operands.into_iter().enumerate().map(add_token_spans),
        ))
    }

    fn add_token_spans((i, operand): (usize, Input)) -> Expr<String, TokenSpan> {
        add_token_spans_recursive(i, 0, operand).1
    }

    fn add_token_spans_recursive(
        i: usize,
        mut j: usize,
        expr: Expr<String, ()>,
    ) -> (usize, Expr<String, TokenSpan>) {
        let mut span: TokenSpan = TokenId::Operand(i, j).into();
        let variant = match expr.variant {
            ExprVariant::Parentheses(expr) => {
                let (new_j, inner) = add_token_spans_recursive(i, j + 1, *expr);
                j = new_j;
                span = span.extend(&TokenId::Operand(i, j).into());
                ExprVariant::Parentheses(Box::new(inner))
            }
            ExprVariant::Ident(ident) => ExprVariant::Ident(ident),
            ExprVariant::Literal(literal) => ExprVariant::Literal(literal),
        };
        (j + 1, Expr { variant, span })
    }

    struct ExpectedDiagnostic {
        message: Message,
        spans: Vec<TokenSpan>,
        highlight: Option<TokenSpan>,
    }

    impl ExpectedDiagnostic {
        fn new(message: Message) -> Self {
            ExpectedDiagnostic {
                message,
                spans: Vec::new(),
                highlight: None,
            }
        }

        fn with_spans<I>(mut self, spans: I) -> Self
        where
            I: IntoIterator,
            I::Item: Into<TokenSpan>,
        {
            self.spans.extend(spans.into_iter().map(Into::into));
            self
        }

        fn with_highlight(mut self, highlight: impl Into<TokenSpan>) -> Self {
            self.highlight = Some(highlight.into());
            self
        }
    }

    impl From<Message> for ExpectedDiagnostic {
        fn from(message: Message) -> Self {
            ExpectedDiagnostic::new(message).with_highlight(TokenId::Mnemonic)
        }
    }

    #[test]
    fn analyze_nop_a() {
        analyze(kw::Mnemonic::Nop, vec![literal(A)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 1,
                expected: 0,
            }).with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_add_a_a_a() {
        analyze(
            kw::Mnemonic::Add,
            vec![A, A, A].into_iter().map(|a| literal(a)),
        ).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 3,
                expected: 2,
            }).with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_jp_c_deref_hl() {
        analyze(
            kw::Mnemonic::Jp,
            vec![literal(C), SimpleOperand::DerefHl.into()],
        ).expect_diagnostic(
            ExpectedDiagnostic::new(Message::AlwaysUnconditional)
                .with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_add() {
        analyze(kw::Mnemonic::Add, Vec::new()).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 0,
                expected: 2,
            }).with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_add_a() {
        analyze(kw::Mnemonic::Add, vec![literal(A)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 1,
                expected: 2,
            }).with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_add_b_a() {
        analyze(kw::Mnemonic::Add, vec![literal(B), literal(A)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::DestMustBeA).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_add_bc_de() {
        analyze(kw::Mnemonic::Add, vec![literal(Bc), literal(De)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::DestMustBeHl).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_add_hl_af() {
        analyze(kw::Mnemonic::Add, vec![literal(Hl), literal(Af)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::IncompatibleOperand)
                .with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_add_hl() {
        analyze(kw::Mnemonic::Add, vec![literal(Hl)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 1,
                expected: 2,
            }).with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_ld() {
        analyze(kw::Mnemonic::Ld, vec![]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 0,
                expected: 2,
            }).with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_ld_a() {
        analyze(kw::Mnemonic::Ld, vec![literal(A)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 1,
                expected: 2,
            }).with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_push() {
        analyze(kw::Mnemonic::Push, vec![]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 0,
                expected: 1,
            }).with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_inc() {
        analyze(kw::Mnemonic::Inc, vec![]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 0,
                expected: 1,
            }).with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_jp_z() {
        analyze(kw::Mnemonic::Jp, vec![literal(Z)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::MissingTarget).with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_ld_const_const() {
        analyze(kw::Mnemonic::Ld, vec![2.into(), 4.into()]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::IllegalOperands)
                .with_spans(iter::once(TokenId::Mnemonic))
                .with_highlight(
                    TokenSpan::from(TokenId::Operand(0, 0)).extend(&TokenId::Operand(1, 0).into()),
                ),
        )
    }

    #[test]
    fn analyze_ld_a_bc() {
        analyze(kw::Mnemonic::Ld, vec![literal(A), literal(Bc)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::LdWidthMismatch { src: Width::Word })
                .with_spans(vec![TokenId::Operand(1, 0), TokenId::Operand(0, 0)])
                .with_highlight(
                    TokenSpan::from(TokenId::Operand(0, 0)).extend(&TokenId::Operand(1, 0).into()),
                ),
        )
    }

    #[test]
    fn analyze_ld_bc_a() {
        analyze(kw::Mnemonic::Ld, vec![literal(Bc), literal(A)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::LdWidthMismatch { src: Width::Byte })
                .with_spans(vec![TokenId::Operand(1, 0), TokenId::Operand(0, 0)])
                .with_highlight(
                    TokenSpan::from(TokenId::Operand(0, 0)).extend(&TokenId::Operand(1, 0).into()),
                ),
        )
    }

    #[test]
    fn analyze_add_hl_const() {
        analyze(kw::Mnemonic::Add, vec![literal(Hl), 2.into()]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::IncompatibleOperand)
                .with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_add_a_bc_deref() {
        analyze(kw::Mnemonic::Add, vec![literal(A), deref(literal(Bc))]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::IncompatibleOperand).with_highlight(TokenSpan {
                first: TokenId::Operand(1, 0),
                last: TokenId::Operand(1, 2),
            }),
        )
    }

    #[test]
    fn analyze_bit_a_b() {
        analyze(kw::Mnemonic::Bit, vec![literal(A), literal(B)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::MustBeBit)
                .with_spans(iter::once(TokenId::Mnemonic))
                .with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_bit_7_bc() {
        analyze(kw::Mnemonic::Bit, vec![7.into(), literal(Bc)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::RequiresSimpleOperand)
                .with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_ldhl_bc_7() {
        analyze(kw::Mnemonic::Ldhl, vec![literal(Bc), 7.into()]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::SrcMustBeSp).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_ldhl_sp_a() {
        analyze(kw::Mnemonic::Ldhl, vec![literal(Sp), literal(A)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::MustBeConst).with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_swap_bc() {
        analyze(kw::Mnemonic::Swap, vec![literal(Bc)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::RequiresSimpleOperand)
                .with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_push_a() {
        analyze(kw::Mnemonic::Push, vec![literal(A)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::RequiresRegPair)
                .with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_rst_a() {
        analyze(kw::Mnemonic::Rst, vec![literal(A)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::MustBeConst).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    enum TokenId {
        Mnemonic,
        Operand(usize, usize),
    }

    #[derive(Clone, Debug, PartialEq)]
    struct TokenSpan {
        first: TokenId,
        last: TokenId,
    }

    impl From<TokenId> for TokenSpan {
        fn from(id: TokenId) -> Self {
            TokenSpan {
                first: id,
                last: id,
            }
        }
    }

    impl Span for TokenSpan {
        fn extend(&self, other: &Self) -> Self {
            TokenSpan {
                first: cmp::min(self.first, other.first),
                last: cmp::max(self.last, other.last),
            }
        }
    }
}
