use self::mnemonic::{Mnemonic, StackOperation};
use self::operand::{AtomKind, Context, Operand, OperandCounter};

use crate::diag::span::Source;
use crate::diag::*;
use crate::object::builder::*;

pub(in crate::analyze::semantics) mod mnemonic;
pub(in crate::analyze::semantics) mod operand;

mod branch;
mod ld;

pub(in crate::analyze::semantics) fn analyze_instruction<I, V, D, S>(
    mnemonic: (&Mnemonic, S),
    operands: I,
    diagnostics: &mut D,
) -> Result<CpuInstr<V>, ()>
where
    I: IntoIterator<Item = Result<Operand<V, S>, ()>>,
    V: Source<Span = S>,
    D: Diagnostics<S>,
    S: Clone,
{
    let mnemonic: (&Mnemonic, _) = (mnemonic.0, mnemonic.1);
    Analysis::new(mnemonic, operands.into_iter(), diagnostics).run()
}

struct Analysis<'a, 'b, I, D: 'b, S> {
    mnemonic: (&'a Mnemonic, S),
    operands: OperandCounter<I>,
    diagnostics: &'b mut D,
}

impl<'a, 'b, I, D, S> EmitDiag<S, D::Stripped> for Analysis<'a, 'b, I, D, S>
where
    D: Diagnostics<S> + 'b,
{
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<S, D::Stripped>>) {
        self.diagnostics.emit_diag(diag)
    }
}

impl<'a, 'b, I, V, D, S> Analysis<'a, 'b, I, D, S>
where
    I: Iterator<Item = Result<Operand<V, S>, ()>>,
    V: Source<Span = S>,
    D: Diagnostics<S>,
    S: Clone,
{
    fn new(mnemonic: (&'a Mnemonic, S), operands: I, diagnostics: &'b mut D) -> Self {
        Analysis {
            mnemonic,
            operands: OperandCounter::new(operands),
            diagnostics,
        }
    }

    fn run(mut self) -> Result<CpuInstr<V>, ()> {
        let instruction = self.analyze_mnemonic()?;
        self.check_for_unexpected_operands()?;
        Ok(instruction)
    }

    fn analyze_mnemonic(&mut self) -> Result<CpuInstr<V>, ()> {
        use self::Mnemonic::*;
        match self.mnemonic.0 {
            Alu(AluOperation::Add) => self.analyze_add_instruction(),
            Alu(operation) => {
                let first_operand = self.next_operand_of(operation.expected_operands())?;
                self.analyze_alu_instruction(*operation, first_operand)
            }
            Bit(operation) => self.analyze_bit_operation(*operation),
            IncDec(mode) => self.analyze_inc_dec(*mode),
            Branch(branch) => self.analyze_branch(*branch),
            Ld => self.analyze_ld(),
            Ldhl => self.analyze_ldhl(),
            Misc(operation) => self.analyze_misc(*operation),
            Nullary(instruction) => Ok((*instruction).into()),
            Rst => self.analyze_rst(),
            Stack(operation) => self.analyze_stack_operation(*operation),
        }
    }

    fn analyze_add_instruction(&mut self) -> Result<CpuInstr<V>, ()> {
        match self.next_operand_of(2)? {
            Operand::Atom(AtomKind::Reg16(reg16), range) => {
                self.analyze_add_reg16_instruction((reg16, range))
            }
            operand => self.analyze_alu_instruction(AluOperation::Add, operand),
        }
    }

    fn analyze_add_reg16_instruction(&mut self, target: (Reg16, S)) -> Result<CpuInstr<V>, ()> {
        match target.0 {
            Reg16::Hl => self.analyze_add_hl_instruction(),
            _ => {
                self.emit_diag(Message::DestMustBeHl.at(target.1));
                Err(())
            }
        }
    }

    fn analyze_add_hl_instruction(&mut self) -> Result<CpuInstr<V>, ()> {
        match self.next_operand_of(2)? {
            Operand::Atom(AtomKind::Reg16(src), _) => Ok(CpuInstr::AddHl(src)),
            operand => {
                self.emit_diag(Message::IncompatibleOperand.at(operand.span()));
                Err(())
            }
        }
    }

    fn analyze_alu_instruction(
        &mut self,
        operation: AluOperation,
        first_operand: Operand<V, S>,
    ) -> Result<CpuInstr<V>, ()> {
        let src = if operation.implicit_dest() {
            first_operand
        } else {
            let second_operand = self.next_operand_of(2)?;
            first_operand.expect_specific_atom(
                AtomKind::Simple(SimpleOperand::A),
                Message::DestMustBeA,
                self.diagnostics,
            )?;
            second_operand
        };
        match src {
            Operand::Atom(AtomKind::Simple(src), _) => {
                Ok(CpuInstr::Alu(operation, AluSource::Simple(src)))
            }
            Operand::Const(expr) => Ok(CpuInstr::Alu(operation, AluSource::Immediate(expr))),
            src => {
                self.emit_diag(Message::IncompatibleOperand.at(src.span()));
                Err(())
            }
        }
    }

    fn analyze_bit_operation(&mut self, operation: BitOperation) -> Result<CpuInstr<V>, ()> {
        let bit_number = self.next_operand_of(2)?;
        let operand = self.next_operand_of(2)?;
        let expr = if let Operand::Const(expr) = bit_number {
            expr
        } else {
            let stripped = self.diagnostics.strip_span(&self.mnemonic.1);
            self.emit_diag(Message::MustBeBit { mnemonic: stripped }.at(bit_number.span()));
            return Err(());
        };
        Ok(CpuInstr::Bit(
            operation,
            expr,
            operand.expect_simple(self.diagnostics)?,
        ))
    }

    fn analyze_ldhl(&mut self) -> Result<CpuInstr<V>, ()> {
        let src = self.next_operand_of(2)?;
        let offset = self.next_operand_of(2)?;
        src.expect_specific_atom(
            AtomKind::Reg16(Reg16::Sp),
            Message::SrcMustBeSp,
            self.diagnostics,
        )?;
        Ok(CpuInstr::Ldhl(offset.expect_const(self.diagnostics)?))
    }

    fn analyze_misc(&mut self, operation: MiscOperation) -> Result<CpuInstr<V>, ()> {
        let operand = self.next_operand_of(1)?;
        Ok(CpuInstr::Misc(
            operation,
            operand.expect_simple(self.diagnostics)?,
        ))
    }

    fn analyze_stack_operation(&mut self, operation: StackOperation) -> Result<CpuInstr<V>, ()> {
        let reg_pair = self.next_operand_of(1)?.expect_reg_pair(self.diagnostics)?;
        let instruction_ctor = match operation {
            StackOperation::Push => CpuInstr::Push,
            StackOperation::Pop => CpuInstr::Pop,
        };
        Ok(instruction_ctor(reg_pair))
    }

    fn analyze_inc_dec(&mut self, mode: IncDec) -> Result<CpuInstr<V>, ()> {
        match self.next_operand_of(1)? {
            Operand::Atom(AtomKind::Simple(operand), _) => Ok(CpuInstr::IncDec8(mode, operand)),
            Operand::Atom(AtomKind::Reg16(operand), _) => Ok(CpuInstr::IncDec16(mode, operand)),
            operand => {
                self.emit_diag(Message::OperandCannotBeIncDec(mode).at(operand.span()));
                Err(())
            }
        }
    }

    fn analyze_rst(&mut self) -> Result<CpuInstr<V>, ()> {
        Ok(CpuInstr::Rst(
            self.next_operand_of(1)?.expect_const(self.diagnostics)?,
        ))
    }

    fn next_operand_of(&mut self, out_of: usize) -> Result<Operand<V, S>, ()> {
        let actual = self.operands.seen();
        self.next_operand()?.ok_or_else(|| {
            self.emit_diag(
                Message::OperandCount {
                    actual,
                    expected: out_of,
                }
                .at(self.mnemonic.1.clone()),
            );
        })
    }

    fn next_operand(&mut self) -> Result<Option<Operand<V, S>>, ()> {
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
                .emit_diag(Message::OperandCount { actual, expected }.at(self.mnemonic.1));
            Err(())
        }
    }
}

impl<V: Source> Operand<V, V::Span> {
    fn expect_specific_atom<D>(
        self,
        expected: AtomKind,
        message: Message<D::Stripped>,
        diagnostics: &mut D,
    ) -> Result<(), ()>
    where
        D: Diagnostics<V::Span>,
    {
        match self {
            Operand::Atom(ref actual, _) if *actual == expected => Ok(()),
            operand => operand.error(message, diagnostics),
        }
    }

    fn expect_simple<D>(self, diagnostics: &mut D) -> Result<SimpleOperand, ()>
    where
        D: Diagnostics<V::Span>,
    {
        match self {
            Operand::Atom(AtomKind::Simple(simple), _) => Ok(simple),
            operand => operand.error(Message::RequiresSimpleOperand, diagnostics),
        }
    }

    fn expect_const<D>(self, diagnostics: &mut D) -> Result<V, ()>
    where
        D: Diagnostics<V::Span>,
    {
        match self {
            Operand::Const(expr) => Ok(expr),
            operand => operand.error(Message::MustBeConst, diagnostics),
        }
    }

    fn expect_reg_pair<D>(self, diagnostics: &mut D) -> Result<RegPair, ()>
    where
        D: Diagnostics<V::Span>,
    {
        match self {
            Operand::Atom(AtomKind::RegPair(reg_pair), _) => Ok(reg_pair),
            operand => operand.error(Message::RequiresRegPair, diagnostics),
        }
    }

    fn error<T, D>(self, message: Message<D::Stripped>, diagnostics: &mut D) -> Result<T, ()>
    where
        D: Diagnostics<V::Span>,
    {
        diagnostics.emit_diag(message.at(self.span()));
        Err(())
    }
}

impl Mnemonic {
    pub fn context(&self) -> Context {
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
        use self::AluOperation::*;
        match self {
            Add | Adc | Sbc => false,
            _ => true,
        }
    }
}

impl<V: Source> From<Nullary> for CpuInstr<V> {
    fn from(nullary: Nullary) -> CpuInstr<V> {
        CpuInstr::Nullary(nullary)
    }
}

#[cfg(test)]
mod tests {
    pub use crate::analyze::semantics::arg::OperandSymbol::*;
    pub(crate) use crate::diag::Message;
    pub(crate) use crate::object::builder::mock::MockSymbolId;
    pub use crate::span::{MergeSpans, SpanSource};

    use self::operand::tests::Event;

    use super::mnemonic::*;
    use super::*;

    use crate::analyze::semantics::arg::*;
    use crate::analyze::semantics::mock::MockExprBuilder;
    use crate::analyze::Literal;
    use crate::expr::Atom;

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    pub(super) enum TokenId {
        Mnemonic,
        Operand(usize, usize),
    }

    pub(super) type TokenSpan = MockSpan<TokenId>;

    type Expr = crate::expr::Expr<MockSymbolId, TokenSpan>;
    type Input = Arg<MockSymbolId, String, ()>;

    impl From<ArgVariant<MockSymbolId, String, ()>> for Input {
        fn from(variant: ArgVariant<MockSymbolId, String, ()>) -> Self {
            Arg { variant, span: () }
        }
    }

    impl From<Literal<String>> for Input {
        fn from(literal: Literal<String>) -> Input {
            ArgVariant::Atom(ArgAtom::Literal(literal)).into()
        }
    }

    pub(super) fn literal(symbol: OperandSymbol) -> Input {
        Arg::from_atom(ArgAtom::OperandSymbol(symbol), ())
    }

    pub(super) fn number(n: i32, span: impl Into<TokenSpan>) -> Expr {
        Expr::from_atom(n.into(), span.into())
    }

    pub(super) fn name(symbol: MockSymbolId, span: impl Into<TokenSpan>) -> Expr {
        Expr::from_atom(Atom::Name(symbol), span.into())
    }

    pub(super) fn deref(expr: impl Into<Input>) -> Input {
        Arg {
            variant: ArgVariant::Unary(ArgUnaryOp::Parentheses, Box::new(expr.into())),
            span: (),
        }
    }

    impl From<AluOperation> for Mnemonic {
        fn from(alu_operation: AluOperation) -> Self {
            match alu_operation {
                AluOperation::Add => ADD,
                AluOperation::Adc => ADC,
                AluOperation::Sub => SUB,
                AluOperation::Sbc => SBC,
                AluOperation::And => AND,
                AluOperation::Xor => XOR,
                AluOperation::Or => OR,
                AluOperation::Cp => CP,
            }
        }
    }

    impl From<BitOperation> for Mnemonic {
        fn from(operation: BitOperation) -> Self {
            match operation {
                BitOperation::Bit => BIT,
                BitOperation::Set => SET,
                BitOperation::Res => RES,
            }
        }
    }

    impl From<MiscOperation> for Mnemonic {
        fn from(operation: MiscOperation) -> Self {
            match operation {
                MiscOperation::Rlc => RLC,
                MiscOperation::Rrc => RRC,
                MiscOperation::Rl => RL,
                MiscOperation::Rr => RR,
                MiscOperation::Sla => SLA,
                MiscOperation::Sra => SRA,
                MiscOperation::Swap => SWAP,
                MiscOperation::Srl => SRL,
            }
        }
    }

    impl From<IncDec> for Mnemonic {
        fn from(mode: IncDec) -> Self {
            match mode {
                IncDec::Inc => INC,
                IncDec::Dec => DEC,
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

    impl From<PtrReg> for OperandSymbol {
        fn from(ptr_reg: PtrReg) -> Self {
            match ptr_reg {
                PtrReg::Bc => Bc,
                PtrReg::De => De,
                PtrReg::Hli => Hli,
                PtrReg::Hld => Hld,
            }
        }
    }

    impl From<Reg16> for OperandSymbol {
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
        OperandSymbol: From<T>,
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

    impl From<MockSymbolId> for Input {
        fn from(ident: MockSymbolId) -> Self {
            Arg::from_atom(ArgAtom::Ident(ident), ())
        }
    }

    impl From<i32> for Input {
        fn from(n: i32) -> Self {
            Literal::Number(n).into()
        }
    }

    #[test]
    fn analyze_cp_symbol() {
        let ident = MockSymbolId(42);
        test_cp_const_analysis(ident.into(), name(ident, TokenId::Operand(0, 0)))
    }

    #[test]
    fn analyze_cp_literal() {
        let n = 0x50;
        test_cp_const_analysis(n.into(), number(n, TokenId::Operand(0, 0)))
    }

    fn test_cp_const_analysis(parsed: Input, expr: Expr) {
        analyze(CP, Some(parsed))
            .expect_instruction(CpuInstr::Alu(AluOperation::Cp, AluSource::Immediate(expr)))
    }

    #[test]
    fn analyze_rst() {
        let n = 3;
        analyze(RST, vec![n.into()])
            .expect_instruction(CpuInstr::Rst(number(n, TokenId::Operand(0, 0))))
    }

    #[test]
    fn analyze_legal_instructions() {
        test_instruction_analysis(describe_legal_instructions());
    }

    pub(super) type InstructionDescriptor = ((Mnemonic, Vec<Input>), CpuInstr<Expr>);

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
            (LDHL, vec![Reg16::Sp.into(), 0x42.into()]),
            CpuInstr::Ldhl(number(0x42, TokenId::Operand(1, 0))),
        ));
        descriptors
    }

    fn describe_push_pop_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        REG_PAIRS.iter().flat_map(|&reg_pair| {
            vec![
                ((PUSH, vec![reg_pair.into()]), CpuInstr::Push(reg_pair)),
                ((POP, vec![reg_pair.into()]), CpuInstr::Pop(reg_pair)),
            ]
        })
    }

    fn describe_nullary_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        [
            Nullary::Cpl,
            Nullary::Daa,
            Nullary::Di,
            Nullary::Ei,
            Nullary::Halt,
            Nullary::Nop,
            Nullary::Rla,
            Nullary::Rlca,
            Nullary::Rra,
            Nullary::Rrca,
            Nullary::Stop,
        ]
        .iter()
        .map(|&nullary| ((nullary.into(), vec![]), nullary.into()))
    }

    impl From<Nullary> for Mnemonic {
        fn from(nullary: Nullary) -> Self {
            Mnemonic::Nullary(nullary)
        }
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
            CpuInstr::Alu(operation, AluSource::Simple(operand)),
        )
    }

    fn describe_alu_simple_without_a(
        operation: AluOperation,
        operand: SimpleOperand,
    ) -> InstructionDescriptor {
        (
            (operation.into(), vec![operand.into()]),
            CpuInstr::Alu(operation, AluSource::Simple(operand)),
        )
    }

    fn describe_add_hl_reg16_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        REG16.iter().map(|&reg16| describe_add_hl_reg16(reg16))
    }

    fn describe_add_hl_reg16(reg16: Reg16) -> InstructionDescriptor {
        (
            (ADD, vec![Reg16::Hl.into(), reg16.into()]),
            CpuInstr::AddHl(reg16),
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
            CpuInstr::Bit(
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
                    CpuInstr::IncDec8(mode, operand),
                )
            })
        })
    }

    fn describe_inc_dec16_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        INC_DEC.iter().flat_map(|&mode| {
            REG16.iter().map(move |&reg16| {
                (
                    (mode.into(), vec![reg16.into()]),
                    CpuInstr::IncDec16(mode, reg16),
                )
            })
        })
    }

    fn describe_misc_operation_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        MISC_OPERATIONS.iter().flat_map(|&operation| {
            SIMPLE_OPERANDS.iter().map(move |&operand| {
                (
                    (operation.into(), vec![operand.into()]),
                    CpuInstr::Misc(operation, operand),
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

    pub(super) struct AnalysisResult(InnerAnalysisResult);

    type InnerAnalysisResult = Result<CpuInstr<Expr>, Vec<Event<TokenSpan>>>;

    impl AnalysisResult {
        pub fn expect_instruction(self, expected: CpuInstr<Expr>) {
            assert_eq!(self.0, Ok(expected))
        }

        pub fn expect_diag(self, diag: impl Into<ExpectedDiag>) {
            let expected = diag.into();
            assert_eq!(
                self.0,
                Err(vec![DiagnosticsEvent::EmitDiag(
                    expected.message.at(expected.highlight.unwrap()).into()
                )
                .into()])
            )
        }
    }

    pub(super) fn analyze<I>(mnemonic: Mnemonic, operands: I) -> AnalysisResult
    where
        I: IntoIterator<Item = Input>,
    {
        use super::operand::analyze_operand;
        use crate::analyze::reentrancy::MockSourceComponents;

        let mut result = None;
        let log = crate::log::with_log(|log| {
            let operands: Vec<_> = operands
                .into_iter()
                .enumerate()
                .map(add_token_spans)
                .map(|op| {
                    analyze_operand(
                        op,
                        mnemonic.context(),
                        MockExprBuilder::with_log(log.clone()),
                    )
                    .0
                })
                .collect();
            let mut session = MockSourceComponents::with_log(log);
            result = Some(analyze_instruction(
                (&mnemonic, TokenId::Mnemonic.into()),
                operands,
                &mut session,
            ));
        });
        AnalysisResult(result.unwrap().map_err(|_| log))
    }

    fn add_token_spans((i, operand): (usize, Input)) -> Arg<MockSymbolId, String, TokenSpan> {
        add_token_spans_recursive(i, 0, operand).1
    }

    fn add_token_spans_recursive(
        i: usize,
        mut j: usize,
        expr: Arg<MockSymbolId, String, ()>,
    ) -> (usize, Arg<MockSymbolId, String, TokenSpan>) {
        let mut span: TokenSpan = TokenId::Operand(i, j).into();
        let variant = match expr.variant {
            ArgVariant::Unary(ArgUnaryOp::Parentheses, expr) => {
                let (new_j, inner) = add_token_spans_recursive(i, j + 1, *expr);
                j = new_j;
                span = TokenSpan::merge(span.clone(), TokenId::Operand(i, j));
                ArgVariant::Unary(ArgUnaryOp::Parentheses, Box::new(inner))
            }
            ArgVariant::Binary(_, _, _) => panic!(),
            ArgVariant::Atom(atom) => ArgVariant::Atom(atom),
            ArgVariant::FnCall(..) => panic!(),
        };
        (j + 1, Arg { variant, span })
    }

    pub(super) struct ExpectedDiag {
        message: Message<TokenSpan>,
        highlight: Option<TokenSpan>,
    }

    impl ExpectedDiag {
        pub fn new(message: Message<TokenSpan>) -> Self {
            ExpectedDiag {
                message,
                highlight: None,
            }
        }

        pub fn with_highlight(mut self, highlight: impl Into<TokenSpan>) -> Self {
            self.highlight = Some(highlight.into());
            self
        }
    }

    impl From<Message<TokenSpan>> for ExpectedDiag {
        fn from(message: Message<TokenSpan>) -> Self {
            ExpectedDiag::new(message).with_highlight(TokenId::Mnemonic)
        }
    }

    #[test]
    fn analyze_nop_a() {
        analyze(NOP, vec![literal(A)]).expect_diag(
            ExpectedDiag::new(Message::OperandCount {
                actual: 1,
                expected: 0,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_add_a_a_a() {
        analyze(ADD, vec![A, A, A].into_iter().map(literal)).expect_diag(
            ExpectedDiag::new(Message::OperandCount {
                actual: 3,
                expected: 2,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_add() {
        analyze(ADD, Vec::new()).expect_diag(
            ExpectedDiag::new(Message::OperandCount {
                actual: 0,
                expected: 2,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_add_a() {
        analyze(ADD, vec![literal(A)]).expect_diag(
            ExpectedDiag::new(Message::OperandCount {
                actual: 1,
                expected: 2,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_add_b_a() {
        analyze(ADD, vec![literal(B), literal(A)]).expect_diag(
            ExpectedDiag::new(Message::DestMustBeA).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_add_bc_de() {
        analyze(ADD, vec![literal(Bc), literal(De)]).expect_diag(
            ExpectedDiag::new(Message::DestMustBeHl).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_add_hl_af() {
        analyze(ADD, vec![literal(Hl), literal(Af)]).expect_diag(
            ExpectedDiag::new(Message::IncompatibleOperand).with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_add_hl() {
        analyze(ADD, vec![literal(Hl)]).expect_diag(
            ExpectedDiag::new(Message::OperandCount {
                actual: 1,
                expected: 2,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_push() {
        analyze(PUSH, vec![]).expect_diag(
            ExpectedDiag::new(Message::OperandCount {
                actual: 0,
                expected: 1,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_inc() {
        analyze(INC, vec![]).expect_diag(
            ExpectedDiag::new(Message::OperandCount {
                actual: 0,
                expected: 1,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_add_hl_const() {
        analyze(ADD, vec![literal(Hl), 2.into()]).expect_diag(
            ExpectedDiag::new(Message::IncompatibleOperand).with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_add_a_bc_deref() {
        analyze(ADD, vec![literal(A), deref(literal(Bc))]).expect_diag(
            ExpectedDiag::new(Message::IncompatibleOperand).with_highlight(TokenSpan::merge(
                TokenId::Operand(1, 0),
                TokenId::Operand(1, 2),
            )),
        )
    }

    #[test]
    fn analyze_bit_a_b() {
        analyze(BIT, vec![literal(A), literal(B)]).expect_diag(
            ExpectedDiag::new(Message::MustBeBit {
                mnemonic: TokenId::Mnemonic.into(),
            })
            .with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_bit_7_bc() {
        analyze(BIT, vec![7.into(), literal(Bc)]).expect_diag(
            ExpectedDiag::new(Message::RequiresSimpleOperand)
                .with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_ldhl_bc_7() {
        analyze(LDHL, vec![literal(Bc), 7.into()]).expect_diag(
            ExpectedDiag::new(Message::SrcMustBeSp).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_ldhl_sp_a() {
        analyze(LDHL, vec![literal(Sp), literal(A)]).expect_diag(
            ExpectedDiag::new(Message::MustBeConst).with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_swap_bc() {
        analyze(SWAP, vec![literal(Bc)]).expect_diag(
            ExpectedDiag::new(Message::RequiresSimpleOperand)
                .with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_push_a() {
        analyze(PUSH, vec![literal(A)]).expect_diag(
            ExpectedDiag::new(Message::RequiresRegPair).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_rst_a() {
        analyze(RST, vec![literal(A)]).expect_diag(
            ExpectedDiag::new(Message::MustBeConst).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_inc_7() {
        analyze(INC, vec![7.into()]).expect_diag(
            ExpectedDiag::new(Message::OperandCannotBeIncDec(IncDec::Inc))
                .with_highlight(TokenId::Operand(0, 0)),
        )
    }
}
