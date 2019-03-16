use self::branch::*;

use super::operand::{AtomKind, Context, Operand, OperandCounter};

use crate::diag::span::Source;
use crate::diag::*;
use crate::model::*;
use crate::syntax::keyword as kw;

mod branch;
mod ld;

pub(super) fn analyze_instruction<I, V, D, S>(
    mnemonic: (kw::Mnemonic, S),
    operands: I,
    diagnostics: &mut D,
) -> Result<Instruction<V>, ()>
where
    I: IntoIterator<Item = Result<Operand<V>, ()>>,
    V: Source<Span = S>,
    D: DownstreamDiagnostics<S>,
    S: Clone,
{
    let mnemonic: (Mnemonic, _) = (mnemonic.0.into(), mnemonic.1);
    Analysis::new(mnemonic, operands.into_iter(), diagnostics).run()
}

struct Analysis<'a, I, D: 'a, S> {
    mnemonic: (Mnemonic, S),
    operands: OperandCounter<I>,
    diagnostics: &'a mut D,
}

impl<'a, I, D, S> DelegateDiagnostics<S> for Analysis<'a, I, D, S>
where
    D: DownstreamDiagnostics<S> + 'a,
{
    type Delegate = D;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.diagnostics
    }
}

impl<'a, I, D, S> EmitDiagnostic<S, D::Stripped> for Analysis<'a, I, D, S>
where
    D: DownstreamDiagnostics<S> + 'a,
{
    fn emit_diagnostic(&mut self, diagnostic: impl Into<CompactDiagnostic<S, D::Stripped>>) {
        self.diagnostics.emit_diagnostic(diagnostic)
    }
}

impl<'a, I, V, D, S> Analysis<'a, I, D, S>
where
    I: Iterator<Item = Result<Operand<V>, ()>>,
    V: Source<Span = S>,
    D: DownstreamDiagnostics<S>,
    S: Clone,
{
    fn new(mnemonic: (Mnemonic, S), operands: I, diagnostics: &'a mut D) -> Analysis<'a, I, D, S> {
        Analysis {
            mnemonic,
            operands: OperandCounter::new(operands),
            diagnostics,
        }
    }

    fn run(mut self) -> Result<Instruction<V>, ()> {
        let instruction = self.analyze_mnemonic()?;
        self.check_for_unexpected_operands()?;
        Ok(instruction)
    }

    fn analyze_mnemonic(&mut self) -> Result<Instruction<V>, ()> {
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

    fn analyze_add_instruction(&mut self) -> Result<Instruction<V>, ()> {
        match self.next_operand_out_of(2)? {
            Operand::Atom(AtomKind::Reg16(reg16), range) => {
                self.analyze_add_reg16_instruction((reg16, range))
            }
            operand => self.analyze_alu_instruction(AluOperation::Add, operand),
        }
    }

    fn analyze_add_reg16_instruction(&mut self, target: (Reg16, S)) -> Result<Instruction<V>, ()> {
        match target.0 {
            Reg16::Hl => self.analyze_add_hl_instruction(),
            _ => {
                self.emit_diagnostic(Message::DestMustBeHl.at(target.1));
                Err(())
            }
        }
    }

    fn analyze_add_hl_instruction(&mut self) -> Result<Instruction<V>, ()> {
        match self.next_operand_out_of(2)? {
            Operand::Atom(AtomKind::Reg16(src), _) => Ok(Instruction::AddHl(src)),
            operand => {
                self.emit_diagnostic(Message::IncompatibleOperand.at(operand.span()));
                Err(())
            }
        }
    }

    fn analyze_alu_instruction(
        &mut self,
        operation: AluOperation,
        first_operand: Operand<V>,
    ) -> Result<Instruction<V>, ()> {
        let src = if operation.implicit_dest() {
            first_operand
        } else {
            let second_operand = self.next_operand_out_of(2)?;
            first_operand.expect_specific_atom(
                AtomKind::Simple(SimpleOperand::A),
                Message::DestMustBeA,
                self.diagnostics,
            )?;
            second_operand
        };
        match src {
            Operand::Atom(AtomKind::Simple(src), _) => {
                Ok(Instruction::Alu(operation, AluSource::Simple(src)))
            }
            Operand::Const(expr) => Ok(Instruction::Alu(operation, AluSource::Immediate(expr))),
            src => {
                self.emit_diagnostic(Message::IncompatibleOperand.at(src.span()));
                Err(())
            }
        }
    }

    fn analyze_bit_operation(&mut self, operation: BitOperation) -> Result<Instruction<V>, ()> {
        let bit_number = self.next_operand_out_of(2)?;
        let operand = self.next_operand_out_of(2)?;
        let expr = if let Operand::Const(expr) = bit_number {
            expr
        } else {
            let stripped = self.diagnostics.strip_span(&self.mnemonic.1);
            self.emit_diagnostic(Message::MustBeBit { mnemonic: stripped }.at(bit_number.span()));
            return Err(());
        };
        Ok(Instruction::Bit(
            operation,
            expr,
            operand.expect_simple(self.diagnostics)?,
        ))
    }

    fn analyze_ldhl(&mut self) -> Result<Instruction<V>, ()> {
        let src = self.next_operand_out_of(2)?;
        let offset = self.next_operand_out_of(2)?;
        src.expect_specific_atom(
            AtomKind::Reg16(Reg16::Sp),
            Message::SrcMustBeSp,
            self.diagnostics,
        )?;
        Ok(Instruction::Ldhl(offset.expect_const(self.diagnostics)?))
    }

    fn analyze_misc(&mut self, operation: MiscOperation) -> Result<Instruction<V>, ()> {
        let operand = self.next_operand_out_of(1)?;
        Ok(Instruction::Misc(
            operation,
            operand.expect_simple(self.diagnostics)?,
        ))
    }

    fn analyze_stack_operation(&mut self, operation: StackOperation) -> Result<Instruction<V>, ()> {
        let reg_pair = self
            .next_operand_out_of(1)?
            .expect_reg_pair(self.diagnostics)?;
        let instruction_ctor = match operation {
            StackOperation::Push => Instruction::Push,
            StackOperation::Pop => Instruction::Pop,
        };
        Ok(instruction_ctor(reg_pair))
    }

    fn analyze_inc_dec(&mut self, mode: IncDec) -> Result<Instruction<V>, ()> {
        match self.next_operand_out_of(1)? {
            Operand::Atom(AtomKind::Simple(operand), _) => Ok(Instruction::IncDec8(mode, operand)),
            Operand::Atom(AtomKind::Reg16(operand), _) => Ok(Instruction::IncDec16(mode, operand)),
            operand => {
                self.emit_diagnostic(Message::OperandCannotBeIncDec(mode).at(operand.span()));
                Err(())
            }
        }
    }

    fn analyze_rst(&mut self) -> Result<Instruction<V>, ()> {
        Ok(Instruction::Rst(
            self.next_operand_out_of(1)?
                .expect_const(self.diagnostics)?,
        ))
    }

    fn next_operand_out_of(&mut self, out_of: usize) -> Result<Operand<V>, ()> {
        let actual = self.operands.seen();
        self.next_operand()?.ok_or_else(|| {
            self.emit_diagnostic(
                Message::OperandCount {
                    actual,
                    expected: out_of,
                }
                .at(self.mnemonic.1.clone()),
            );
        })
    }

    fn next_operand(&mut self) -> Result<Option<Operand<V>>, ()> {
        self.operands
            .next()
            .map_or(Ok(None), |result| result.map(Some))
    }

    fn check_for_unexpected_operands(self) -> Result<(), ()> {
        let expected = self.operands.seen();
        let extra = self.operands.count();
        let actual = expected + extra;
        if actual == expected {
            Ok(())
        } else {
            self.diagnostics
                .emit_diagnostic(Message::OperandCount { actual, expected }.at(self.mnemonic.1));
            Err(())
        }
    }
}

impl<V: Source> Operand<V> {
    fn expect_specific_atom<D>(
        self,
        expected: AtomKind,
        message: Message<D::Stripped>,
        diagnostics: &mut D,
    ) -> Result<(), ()>
    where
        D: DownstreamDiagnostics<V::Span>,
    {
        match self {
            Operand::Atom(ref actual, _) if *actual == expected => Ok(()),
            operand => operand.error(message, diagnostics),
        }
    }

    fn expect_simple<D>(self, diagnostics: &mut D) -> Result<SimpleOperand, ()>
    where
        D: DownstreamDiagnostics<V::Span>,
    {
        match self {
            Operand::Atom(AtomKind::Simple(simple), _) => Ok(simple),
            operand => operand.error(Message::RequiresSimpleOperand, diagnostics),
        }
    }

    fn expect_const<D>(self, diagnostics: &mut D) -> Result<V, ()>
    where
        D: DownstreamDiagnostics<V::Span>,
    {
        match self {
            Operand::Const(expr) => Ok(expr),
            operand => operand.error(Message::MustBeConst, diagnostics),
        }
    }

    fn expect_reg_pair<D>(self, diagnostics: &mut D) -> Result<RegPair, ()>
    where
        D: DownstreamDiagnostics<V::Span>,
    {
        match self {
            Operand::Atom(AtomKind::RegPair(reg_pair), _) => Ok(reg_pair),
            operand => operand.error(Message::RequiresRegPair, diagnostics),
        }
    }

    fn error<T, D>(self, message: Message<D::Stripped>, diagnostics: &mut D) -> Result<T, ()>
    where
        D: DownstreamDiagnostics<V::Span>,
    {
        diagnostics.emit_diagnostic(message.at(self.span()));
        Err(())
    }
}

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

impl kw::Mnemonic {
    pub fn context(self) -> Context {
        Mnemonic::from(self).context()
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
        use self::AluOperation::*;
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

impl<V: Source> From<Nullary> for Instruction<V> {
    fn from(nullary: Nullary) -> Instruction<V> {
        Instruction::Nullary(nullary)
    }
}

impl From<kw::Mnemonic> for Mnemonic {
    fn from(keyword: kw::Mnemonic) -> Self {
        use self::kw::Mnemonic::*;
        match keyword {
            Adc => Mnemonic::Alu(AluOperation::Adc),
            Add => Mnemonic::Alu(AluOperation::Add),
            And => Mnemonic::Alu(AluOperation::And),
            Bit => Mnemonic::Bit(BitOperation::Bit),
            Call => Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Call)),
            Cp => Mnemonic::Alu(AluOperation::Cp),
            Cpl => Mnemonic::Nullary(Nullary::Cpl),
            Daa => Mnemonic::Nullary(Nullary::Daa),
            Dec => Mnemonic::IncDec(IncDec::Dec),
            Di => Mnemonic::Nullary(Nullary::Di),
            Ei => Mnemonic::Nullary(Nullary::Ei),
            Halt => Mnemonic::Nullary(Nullary::Halt),
            Inc => Mnemonic::IncDec(IncDec::Inc),
            Jp => Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Jp)),
            Jr => Mnemonic::Branch(BranchKind::Explicit(ExplicitBranch::Jr)),
            Ld => Mnemonic::Ld,
            Ldhl => Mnemonic::Ldhl,
            Nop => Mnemonic::Nullary(Nullary::Nop),
            Or => Mnemonic::Alu(AluOperation::Or),
            Pop => Mnemonic::Stack(StackOperation::Pop),
            Push => Mnemonic::Stack(StackOperation::Push),
            Res => Mnemonic::Bit(BitOperation::Res),
            Ret => Mnemonic::Branch(BranchKind::Implicit(ImplicitBranch::Ret)),
            Reti => Mnemonic::Branch(BranchKind::Implicit(ImplicitBranch::Reti)),
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
    pub use self::kw::Operand::*;
    pub use crate::analysis::semantics::{TokenId, TokenSpan};
    pub(crate) use crate::diag::Message;
    pub use crate::span::{MergeSpans, Span};

    use super::*;
    use crate::analysis::semantics::*;
    use crate::analysis::{Ident, Literal};
    use crate::expr::{Expr, ExprVariant};
    use crate::model::{RelocAtom, RelocExpr};
    use crate::syntax::Mnemonic;
    use std::cmp;

    type Input = SemanticExpr<String, ()>;

    impl From<SemanticExprVariant<String, ()>> for Input {
        fn from(variant: SemanticExprVariant<String, ()>) -> Self {
            SemanticExpr { variant, span: () }
        }
    }

    impl From<Literal<String>> for Input {
        fn from(literal: Literal<String>) -> Input {
            ExprVariant::Atom(SemanticAtom::Literal(literal)).into()
        }
    }

    pub(super) fn literal(keyword: kw::Operand) -> Input {
        Literal::Operand(keyword).into()
    }

    pub fn number(n: i32, span: impl Into<TokenSpan>) -> RelocExpr<Ident<String>, TokenSpan> {
        RelocExpr::from_atom(n, span.into())
    }

    pub fn symbol(ident: &str, span: impl Into<TokenSpan>) -> RelocExpr<Ident<String>, TokenSpan> {
        RelocExpr::from_atom(RelocAtom::Symbol(ident.into()), span.into())
    }

    pub(super) fn deref(expr: Input) -> Input {
        Expr {
            variant: ExprVariant::Unary(SemanticUnary::Parentheses, Box::new(expr)),
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
            Expr::from_atom(SemanticAtom::Ident(ident.into()), ())
        }
    }

    impl From<i32> for Input {
        fn from(n: i32) -> Self {
            Literal::Number(n).into()
        }
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

    fn test_cp_const_analysis(parsed: Input, expr: RelocExpr<Ident<String>, TokenSpan>) {
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

    pub(super) type InstructionDescriptor = (
        (kw::Mnemonic, Vec<Input>),
        Instruction<RelocExpr<Ident<String>, TokenSpan>>,
    );

    fn describe_legal_instructions() -> Vec<InstructionDescriptor> {
        let mut descriptors: Vec<InstructionDescriptor> = Vec::new();
        descriptors.extend(describe_nullary_instructions());
        descriptors.extend(describe_alu_simple_instructions());
        descriptors.extend(describe_add_hl_reg16_instructions());
        descriptors.extend(describe_bit_operation_instructions());
        descriptors.extend(describe_inc_dec8_instructions());
        descriptors.extend(describe_inc_dec16_instructions());
        descriptors.extend(describe_push_pop_instructions());
        descriptors.extend(describe_misc_operation_instructions());
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
            (kw::Mnemonic::Rla, Nullary::Rla),
            (kw::Mnemonic::Rlca, Nullary::Rlca),
            (kw::Mnemonic::Rra, Nullary::Rra),
            (kw::Mnemonic::Rrca, Nullary::Rrca),
            (kw::Mnemonic::Stop, Nullary::Stop),
        ]
        .iter()
        .map(|(mnemonic, nullary)| ((*mnemonic, vec![]), Instruction::Nullary(nullary.clone())))
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

    pub const SIMPLE_OPERANDS: &[SimpleOperand] = &[
        SimpleOperand::A,
        SimpleOperand::B,
        SimpleOperand::C,
        SimpleOperand::D,
        SimpleOperand::E,
        SimpleOperand::H,
        SimpleOperand::L,
        SimpleOperand::DerefHl,
    ];

    pub const REG16: &[Reg16] = &[Reg16::Bc, Reg16::De, Reg16::Hl, Reg16::Sp];

    const REG_PAIRS: &[RegPair] = &[RegPair::Bc, RegPair::De, RegPair::Hl, RegPair::Af];

    const INC_DEC: &[IncDec] = &[IncDec::Inc, IncDec::Dec];

    pub(super) fn test_instruction_analysis(descriptors: Vec<InstructionDescriptor>) {
        for ((mnemonic, operands), expected) in descriptors {
            analyze(mnemonic, operands).expect_instruction(expected)
        }
    }

    pub struct AnalysisResult(
        Result<Instruction<RelocExpr<Ident<String>, TokenSpan>>, Vec<DiagnosticsEvent<TokenSpan>>>,
    );

    impl AnalysisResult {
        pub fn expect_instruction(
            self,
            expected: Instruction<RelocExpr<Ident<String>, TokenSpan>>,
        ) {
            assert_eq!(self.0, Ok(expected))
        }

        pub fn expect_diagnostic(self, diagnostic: impl Into<ExpectedDiagnostic>) {
            let expected = diagnostic.into();
            assert_eq!(
                self.0,
                Err(vec![DiagnosticsEvent::EmitDiagnostic(
                    expected.message.at(expected.highlight.unwrap()).into()
                )])
            )
        }
    }

    pub(super) fn analyze<I>(mnemonic: kw::Mnemonic, operands: I) -> AnalysisResult
    where
        I: IntoIterator<Item = Input>,
    {
        use crate::analysis::semantics::operand::analyze_operand;
        use crate::analysis::session::MockSession;
        use std::cell::RefCell;

        let log = RefCell::new(Vec::new());
        let mut session = MockSession::new(&log);
        let operands: Vec<_> = operands
            .into_iter()
            .enumerate()
            .map(add_token_spans)
            .map(|op| analyze_operand(op, Mnemonic::from(mnemonic).context(), &mut session))
            .collect();
        let result = analyze_instruction(
            (mnemonic, TokenId::Mnemonic.into()),
            operands,
            session.diagnostics(),
        );
        AnalysisResult(result.map_err(|_| log.into_inner()))
    }

    fn add_token_spans((i, operand): (usize, Input)) -> SemanticExpr<String, TokenSpan> {
        add_token_spans_recursive(i, 0, operand).1
    }

    fn add_token_spans_recursive(
        i: usize,
        mut j: usize,
        expr: SemanticExpr<String, ()>,
    ) -> (usize, SemanticExpr<String, TokenSpan>) {
        let mut span: TokenSpan = TokenId::Operand(i, j).into();
        let variant = match expr.variant {
            ExprVariant::Unary(SemanticUnary::Parentheses, expr) => {
                let (new_j, inner) = add_token_spans_recursive(i, j + 1, *expr);
                j = new_j;
                span = TokenSpan::merge(&span, &TokenId::Operand(i, j).into());
                ExprVariant::Unary(SemanticUnary::Parentheses, Box::new(inner))
            }
            ExprVariant::Binary(_, _, _) => panic!(),
            ExprVariant::Atom(SemanticAtom::Ident(ident)) => {
                ExprVariant::Atom(SemanticAtom::Ident(ident))
            }
            ExprVariant::Atom(SemanticAtom::Literal(literal)) => {
                ExprVariant::Atom(SemanticAtom::Literal(literal))
            }
        };
        (j + 1, Expr { variant, span })
    }

    pub struct ExpectedDiagnostic {
        message: Message<TokenSpan>,
        highlight: Option<TokenSpan>,
    }

    impl ExpectedDiagnostic {
        pub fn new(message: Message<TokenSpan>) -> Self {
            ExpectedDiagnostic {
                message,
                highlight: None,
            }
        }

        pub fn with_highlight(mut self, highlight: impl Into<TokenSpan>) -> Self {
            self.highlight = Some(highlight.into());
            self
        }
    }

    impl From<Message<TokenSpan>> for ExpectedDiagnostic {
        fn from(message: Message<TokenSpan>) -> Self {
            ExpectedDiagnostic::new(message).with_highlight(TokenId::Mnemonic)
        }
    }

    #[test]
    fn analyze_nop_a() {
        analyze(kw::Mnemonic::Nop, vec![literal(A)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 1,
                expected: 0,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_add_a_a_a() {
        analyze(
            kw::Mnemonic::Add,
            vec![A, A, A].into_iter().map(|a| literal(a)),
        )
        .expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 3,
                expected: 2,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_add() {
        analyze(kw::Mnemonic::Add, Vec::new()).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 0,
                expected: 2,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_add_a() {
        analyze(kw::Mnemonic::Add, vec![literal(A)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 1,
                expected: 2,
            })
            .with_highlight(TokenId::Mnemonic),
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
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_push() {
        analyze(kw::Mnemonic::Push, vec![]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 0,
                expected: 1,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_inc() {
        analyze(kw::Mnemonic::Inc, vec![]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 0,
                expected: 1,
            })
            .with_highlight(TokenId::Mnemonic),
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
            ExpectedDiagnostic::new(Message::MustBeBit {
                mnemonic: TokenId::Mnemonic.into(),
            })
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

    #[test]
    fn analyze_inc_7() {
        analyze(kw::Mnemonic::Inc, vec![7.into()]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCannotBeIncDec(IncDec::Inc))
                .with_highlight(TokenId::Operand(0, 0)),
        )
    }

    impl From<TokenId> for TokenSpan {
        fn from(id: TokenId) -> Self {
            TokenSpan {
                first: id,
                last: id,
            }
        }
    }

    impl TokenSpan {
        pub fn merge(left: &TokenSpan, right: &TokenSpan) -> TokenSpan {
            TokenSpan {
                first: cmp::min(left.first, right.first),
                last: cmp::max(left.last, right.last),
            }
        }
    }
}
