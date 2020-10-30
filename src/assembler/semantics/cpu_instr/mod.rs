use self::operand::{AtomKind, Context, Operand, OperandCounter};

use crate::assembler::keywords::*;
use crate::assembler::session::Backend;
use crate::diagnostics::*;
use crate::expr::Expr;
use crate::object::{Fragment, SymbolId, Width};
use crate::span::Source;
use crate::IncDec;

pub mod operand;

mod branch;
mod ld;

pub(super) fn analyze_instruction<I, D, S>(mnemonic: (&Mnemonic, S), operands: I, session: &mut D)
where
    I: IntoIterator<Item = Result<Operand<S>, ()>>,
    D: Backend<S> + Diagnostics<S>,
    S: Clone,
{
    let mnemonic: (&Mnemonic, _) = (mnemonic.0, mnemonic.1);
    Analysis::new(mnemonic, operands.into_iter(), session).run()
}

struct Analysis<'a, 'b, I, D: 'b, S> {
    mnemonic: (&'a Mnemonic, S),
    operands: OperandCounter<I>,
    session: &'b mut D,
}

impl<'a, 'b, I, D, S> EmitDiag<S, D::Stripped> for Analysis<'a, 'b, I, D, S>
where
    D: Diagnostics<S> + 'b,
{
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<S, D::Stripped>>) {
        self.session.emit_diag(diag)
    }
}

impl<'a, 'b, I, D, S> Analysis<'a, 'b, I, D, S>
where
    I: Iterator<Item = Result<Operand<S>, ()>>,
    D: Backend<S> + Diagnostics<S>,
    S: Clone,
{
    fn new(mnemonic: (&'a Mnemonic, S), operands: I, session: &'b mut D) -> Self {
        Analysis {
            mnemonic,
            operands: OperandCounter::new(operands),
            session,
        }
    }

    fn run(mut self) {
        if self.analyze_mnemonic().is_ok() {
            self.check_for_unexpected_operands()
        }
    }

    fn analyze_mnemonic(&mut self) -> Result<(), ()> {
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
            Nullary(opcode) => {
                self.session.emit_fragment(Fragment::Byte(*opcode));
                Ok(())
            }
            Rst => self.analyze_rst(),
            Stack(operation) => self.analyze_stack_operation(*operation),
            Stop => {
                self.session.emit_fragment(Fragment::Byte(0x10));
                self.session.emit_fragment(Fragment::Byte(0x00));
                Ok(())
            }
        }
    }

    fn analyze_add_instruction(&mut self) -> Result<(), ()> {
        match self.next_operand_of(2)? {
            Operand::Atom(AtomKind::Reg16(reg16), range) => {
                self.analyze_add_reg16_instruction((reg16, range))
            }
            operand => self.analyze_alu_instruction(AluOperation::Add, operand),
        }
    }

    fn analyze_add_reg16_instruction(&mut self, target: (Reg16, S)) -> Result<(), ()> {
        match target.0 {
            Reg16::Hl => self.analyze_add_hl_instruction(),
            Reg16::Sp => self.analyze_add_sp_e(),
            _ => {
                self.emit_diag(Message::DestMustBeHl.at(target.1));
                Err(())
            }
        }
    }

    fn analyze_add_hl_instruction(&mut self) -> Result<(), ()> {
        match self.next_operand_of(2)? {
            Operand::Atom(AtomKind::Reg16(src), _) => {
                self.session
                    .emit_fragment(Fragment::Byte(0x09 | encode_reg16(src)));
                Ok(())
            }
            operand => {
                self.emit_diag(Message::IncompatibleOperand.at(operand.span()));
                Err(())
            }
        }
    }

    fn analyze_add_sp_e(&mut self) -> Result<(), ()> {
        match self.next_operand_of(2)? {
            Operand::Const(expr) => {
                self.session.emit_fragment(Fragment::Byte(0xe8));
                self.session
                    .emit_fragment(Fragment::Immediate(expr, Width::Byte));
                Ok(())
            }
            _ => todo!(),
        }
    }

    fn analyze_alu_instruction(
        &mut self,
        operation: AluOperation,
        first_operand: Operand<S>,
    ) -> Result<(), ()> {
        let src = if operation.implicit_dest() {
            first_operand
        } else {
            let second_operand = self.next_operand_of(2)?;
            first_operand.expect_specific_atom(
                AtomKind::Simple(M::A),
                Message::DestMustBeA,
                self.session,
            )?;
            second_operand
        };
        match src {
            Operand::Atom(AtomKind::Simple(src), _) => {
                self.session.emit_fragment(Fragment::Byte(
                    0b10_000_000 | encode_alu_operation(operation) | src.encode(),
                ));
                Ok(())
            }
            Operand::Const(expr) => {
                self.session.emit_fragment(Fragment::Byte(
                    0b11_000_110 | encode_alu_operation(operation),
                ));
                self.session
                    .emit_fragment(Fragment::Immediate(expr, Width::Byte));
                Ok(())
            }
            src => {
                self.emit_diag(Message::IncompatibleOperand.at(src.span()));
                Err(())
            }
        }
    }

    fn analyze_bit_operation(&mut self, operation: BitOperation) -> Result<(), ()> {
        let bit_number = self.next_operand_of(2)?;
        let operand = self.next_operand_of(2)?;
        let expr = if let Operand::Const(expr) = bit_number {
            expr
        } else {
            let stripped = self.session.strip_span(&self.mnemonic.1);
            self.emit_diag(Message::MustBeBit { mnemonic: stripped }.at(bit_number.span()));
            return Err(());
        };
        let operand = operand.expect_simple(self.session)?;
        self.session.emit_fragment(Fragment::Byte(0xcb));
        self.session.emit_fragment(Fragment::Embedded(
            encode_bit_operation(operation) | operand.encode(),
            expr,
        ));
        Ok(())
    }

    fn analyze_ldhl(&mut self) -> Result<(), ()> {
        let src = self.next_operand_of(2)?;
        let offset = self.next_operand_of(2)?.expect_const(self.session)?;
        src.expect_specific_atom(
            AtomKind::Reg16(Reg16::Sp),
            Message::SrcMustBeSp,
            self.session,
        )?;
        self.session.emit_fragment(Fragment::Byte(0xf8));
        self.session
            .emit_fragment(Fragment::Immediate(offset, Width::Byte));
        Ok(())
    }

    fn analyze_misc(&mut self, operation: MiscOperation) -> Result<(), ()> {
        let operand = self.next_operand_of(1)?.expect_simple(self.session)?;
        self.session.emit_fragment(Fragment::Byte(0xcb));
        self.session
            .emit_fragment(Fragment::Byte(operation.encode() | operand.encode()));
        Ok(())
    }

    fn analyze_stack_operation(&mut self, operation: StackOperation) -> Result<(), ()> {
        let reg_pair = self.next_operand_of(1)?.expect_reg_pair(self.session)?;
        let opcode = match operation {
            StackOperation::Push => 0xc5,
            StackOperation::Pop => 0xc1,
        } | (encode_reg_pair(reg_pair) << 4);
        self.session.emit_fragment(Fragment::Byte(opcode));
        Ok(())
    }

    fn analyze_inc_dec(&mut self, mode: IncDec) -> Result<(), ()> {
        match self.next_operand_of(1)? {
            Operand::Atom(AtomKind::Simple(operand), _) => {
                self.session.emit_fragment(Fragment::Byte(
                    0b00_000_100 | encode_inc_dec(mode) | (operand.encode() << 3),
                ));
                Ok(())
            }
            Operand::Atom(AtomKind::Reg16(operand), _) => {
                self.session.emit_fragment(Fragment::Byte(
                    0x03 | (encode_inc_dec(mode) << 3) | encode_reg16(operand),
                ));
                Ok(())
            }
            operand => {
                self.emit_diag(Message::OperandCannotBeIncDec(mode).at(operand.span()));
                Err(())
            }
        }
    }

    fn analyze_rst(&mut self) -> Result<(), ()> {
        let operand = self.next_operand_of(1)?.expect_const(self.session)?;
        self.session
            .emit_fragment(Fragment::Embedded(0b11_000_111, operand));
        Ok(())
    }

    fn next_operand_of(&mut self, out_of: usize) -> Result<Operand<S>, ()> {
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

    fn next_operand(&mut self) -> Result<Option<Operand<S>>, ()> {
        self.operands
            .next()
            .map_or(Ok(None), |result| result.map(Some))
    }

    fn check_for_unexpected_operands(self) {
        let expected = self.operands.seen();
        let extra = self.operands.count();
        let actual = expected + extra;
        if actual != expected {
            self.session
                .emit_diag(Message::OperandCount { actual, expected }.at(self.mnemonic.1));
        }
    }
}

impl MiscOperation {
    pub fn encode(self) -> u8 {
        use self::MiscOperation::*;
        (match self {
            Rlc => 0b000,
            Rrc => 0b001,
            Rl => 0b010,
            Rr => 0b011,
            Sla => 0b100,
            Sra => 0b101,
            Swap => 0b110,
            Srl => 0b111,
        }) << 3
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum M {
    A,
    B,
    C,
    D,
    E,
    H,
    L,
    DerefHl,
}

impl M {
    pub fn encode(self) -> u8 {
        use self::M::*;
        match self {
            B => 0b000,
            C => 0b001,
            D => 0b010,
            E => 0b011,
            H => 0b100,
            L => 0b101,
            DerefHl => 0b110,
            A => 0b111,
        }
    }
}

fn encode_alu_operation(operation: AluOperation) -> u8 {
    use self::AluOperation::*;
    (match operation {
        Add => 0b000,
        Adc => 0b001,
        Sub => 0b010,
        Sbc => 0b011,
        And => 0b100,
        Xor => 0b101,
        Or => 0b110,
        Cp => 0b111,
    }) << 3
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegPair {
    Bc,
    De,
    Hl,
    Af,
}

fn encode_reg_pair(reg_pair: RegPair) -> u8 {
    use self::RegPair::*;
    match reg_pair {
        Bc => 0b00,
        De => 0b01,
        Hl => 0b10,
        Af => 0b11,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reg16 {
    Bc,
    De,
    Hl,
    Sp,
}

fn encode_reg16(reg16: Reg16) -> u8 {
    use self::Reg16::*;
    (match reg16 {
        Bc => 0b00,
        De => 0b01,
        Hl => 0b10,
        Sp => 0b11,
    }) << 4
}

fn encode_bit_operation(operation: BitOperation) -> u8 {
    use self::BitOperation::*;
    (match operation {
        Bit => 0b01,
        Set => 0b11,
        Res => 0b10,
    }) << 6
}

fn encode_inc_dec(mode: IncDec) -> u8 {
    use self::IncDec::*;
    match mode {
        Inc => 0,
        Dec => 1,
    }
}

impl<S: Clone> Operand<S> {
    fn expect_specific_atom<D>(
        self,
        expected: AtomKind,
        message: Message<D::Stripped>,
        diagnostics: &mut D,
    ) -> Result<(), ()>
    where
        D: Diagnostics<S>,
    {
        match self {
            Operand::Atom(ref actual, _) if *actual == expected => Ok(()),
            operand => operand.error(message, diagnostics),
        }
    }

    fn expect_simple<D>(self, diagnostics: &mut D) -> Result<M, ()>
    where
        D: Diagnostics<S>,
    {
        match self {
            Operand::Atom(AtomKind::Simple(simple), _) => Ok(simple),
            operand => operand.error(Message::RequiresSimpleOperand, diagnostics),
        }
    }

    fn expect_const<D>(self, diagnostics: &mut D) -> Result<Expr<SymbolId, S>, ()>
    where
        D: Diagnostics<S>,
    {
        match self {
            Operand::Const(expr) => Ok(expr),
            operand => operand.error(Message::MustBeConst, diagnostics),
        }
    }

    fn expect_reg_pair<D>(self, diagnostics: &mut D) -> Result<RegPair, ()>
    where
        D: Diagnostics<S>,
    {
        match self {
            Operand::Atom(AtomKind::RegPair(reg_pair), _) => Ok(reg_pair),
            operand => operand.error(Message::RequiresRegPair, diagnostics),
        }
    }

    fn error<T, D>(self, message: Message<D::Stripped>, diagnostics: &mut D) -> Result<T, ()>
    where
        D: Diagnostics<S>,
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
        !matches!(self, Add | Adc | Sbc)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Condition {
    C,
    Nc,
    Nz,
    Z,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PtrReg {
    Bc,
    De,
    Hli,
    Hld,
}

#[cfg(test)]
mod tests {
    pub use crate::assembler::keywords::OperandKeyword::*;
    pub(crate) use crate::diagnostics::Message;
    pub(crate) use crate::object::Fragment;
    pub(crate) use crate::span::{Spanned, WithSpan};

    use self::operand::tests::Event;

    use super::*;

    use crate::assembler::semantics::*;
    use crate::assembler::syntax::Literal;
    use crate::expr::Atom;
    use crate::object::SymbolId;

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    pub(super) enum TokenId {
        Mnemonic,
        Operand(usize, usize),
    }

    pub(super) type TokenSpan = MockSpan<TokenId>;

    type Expr<S> = crate::expr::Expr<SymbolId, S>;
    type Input = Arg<()>;

    impl From<Literal> for Input {
        fn from(literal: Literal) -> Input {
            match literal {
                Literal::Number(n) => Arg::Bare(BareArg::Const(Expr::from_atom(n.into(), ()))),
                Literal::String(_string) => unimplemented!(),
            }
        }
    }

    pub(super) fn literal(symbol: OperandKeyword) -> Input {
        Arg::Bare(BareArg::OperandKeyword(symbol, ()))
    }

    pub(super) fn number(n: i32, span: impl Into<TokenSpan>) -> Expr<TokenSpan> {
        Expr::from_atom(n.into(), span.into())
    }

    pub(super) fn name(symbol: SymbolId, span: impl Into<TokenSpan>) -> Expr<TokenSpan> {
        Expr::from_atom(Atom::Name(symbol), span.into())
    }

    pub(super) fn deref_symbol(symbol: impl Into<OperandKeyword>) -> Input {
        Arg::Deref(BareArg::OperandKeyword(symbol.into(), ()), ())
    }

    pub(super) fn deref_ident(ident: SymbolId) -> Input {
        Arg::Deref(BareArg::Const(Expr::from_atom(Atom::Name(ident), ())), ())
    }

    impl From<M> for Input {
        fn from(alu_operand: M) -> Self {
            match alu_operand {
                M::A => literal(A),
                M::B => literal(B),
                M::C => literal(C),
                M::D => literal(D),
                M::E => literal(E),
                M::H => literal(H),
                M::L => literal(L),
                M::DerefHl => deref_symbol(Hl),
            }
        }
    }

    impl From<PtrReg> for OperandKeyword {
        fn from(ptr_reg: PtrReg) -> Self {
            match ptr_reg {
                PtrReg::Bc => Bc,
                PtrReg::De => De,
                PtrReg::Hli => Hli,
                PtrReg::Hld => Hld,
            }
        }
    }

    impl From<Reg16> for OperandKeyword {
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
        OperandKeyword: From<T>,
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

    impl From<SymbolId> for Input {
        fn from(ident: SymbolId) -> Self {
            Arg::Bare(BareArg::Const(Expr::from_atom(Atom::Name(ident), ())))
        }
    }

    impl From<i32> for Input {
        fn from(n: i32) -> Self {
            Literal::Number(n).into()
        }
    }

    #[test]
    fn analyze_rst() {
        let n = 3;
        analyze(RST, vec![n.into()]).expect_fragments(vec![Fragment::Embedded(
            0b11_000_111,
            number(n, TokenId::Operand(0, 0)),
        )])
    }

    #[test]
    fn ldhl_sp_0x42() {
        analyze(LDHL, vec![Reg16::Sp.into(), 0x42.into()]).expect_fragments(vec![
            Fragment::Byte(0xf8),
            Fragment::Immediate(number(0x42, TokenId::Operand(1, 0)), Width::Byte),
        ])
    }

    #[test]
    fn push_bc() {
        analyze(PUSH, vec![RegPair::Bc.into()]).expect_fragments(vec![Fragment::Byte(0xc5)])
    }

    #[test]
    fn push_de() {
        analyze(PUSH, vec![RegPair::De.into()]).expect_fragments(vec![Fragment::Byte(0xd5)])
    }

    #[test]
    fn push_hl() {
        analyze(PUSH, vec![RegPair::Hl.into()]).expect_fragments(vec![Fragment::Byte(0xe5)])
    }

    #[test]
    fn push_af() {
        analyze(PUSH, vec![RegPair::Af.into()]).expect_fragments(vec![Fragment::Byte(0xf5)])
    }

    #[test]
    fn pop_bc() {
        analyze(POP, vec![RegPair::Bc.into()]).expect_fragments(vec![Fragment::Byte(0xc1)])
    }

    #[test]
    fn pop_de() {
        analyze(POP, vec![RegPair::De.into()]).expect_fragments(vec![Fragment::Byte(0xd1)])
    }

    #[test]
    fn pop_hl() {
        analyze(POP, vec![RegPair::Hl.into()]).expect_fragments(vec![Fragment::Byte(0xe1)])
    }

    #[test]
    fn pop_af() {
        analyze(POP, vec![RegPair::Af.into()]).expect_fragments(vec![Fragment::Byte(0xf1)])
    }

    #[test]
    fn cpl() {
        analyze(CPL, vec![]).expect_fragments(vec![Fragment::Byte(0x2f)])
    }

    #[test]
    fn daa() {
        analyze(DAA, vec![]).expect_fragments(vec![Fragment::Byte(0x27)])
    }

    #[test]
    fn di() {
        analyze(DI, vec![]).expect_fragments(vec![Fragment::Byte(0xf3)])
    }

    #[test]
    fn ei() {
        analyze(EI, vec![]).expect_fragments(vec![Fragment::Byte(0xfb)])
    }

    #[test]
    fn halt() {
        analyze(HALT, vec![]).expect_fragments(vec![Fragment::Byte(0x76)])
    }

    #[test]
    fn nop() {
        analyze(NOP, vec![]).expect_fragments(vec![Fragment::Byte(0x00)])
    }

    #[test]
    fn rla() {
        analyze(RLA, vec![]).expect_fragments(vec![Fragment::Byte(0x17)])
    }

    #[test]
    fn rlca() {
        analyze(RLCA, vec![]).expect_fragments(vec![Fragment::Byte(0x07)])
    }

    #[test]
    fn rra() {
        analyze(RRA, vec![]).expect_fragments(vec![Fragment::Byte(0x1f)])
    }

    #[test]
    fn rrca() {
        analyze(RRCA, vec![]).expect_fragments(vec![Fragment::Byte(0x0f)])
    }

    #[test]
    fn stop() {
        analyze(STOP, vec![]).expect_fragments(vec![Fragment::Byte(0x10), Fragment::Byte(0x00)])
    }

    #[test]
    fn add_a_a() {
        analyze(ADD, vec![M::A.into(), M::A.into()]).expect_fragments(vec![Fragment::Byte(0x87)])
    }

    #[test]
    fn add_a_b() {
        analyze(ADD, vec![M::A.into(), M::B.into()]).expect_fragments(vec![Fragment::Byte(0x80)])
    }

    #[test]
    fn add_a_c() {
        analyze(ADD, vec![M::A.into(), M::C.into()]).expect_fragments(vec![Fragment::Byte(0x81)])
    }

    #[test]
    fn add_a_d() {
        analyze(ADD, vec![M::A.into(), M::D.into()]).expect_fragments(vec![Fragment::Byte(0x82)])
    }

    #[test]
    fn add_a_e() {
        analyze(ADD, vec![M::A.into(), M::E.into()]).expect_fragments(vec![Fragment::Byte(0x83)])
    }

    #[test]
    fn add_a_h() {
        analyze(ADD, vec![M::A.into(), M::H.into()]).expect_fragments(vec![Fragment::Byte(0x84)])
    }

    #[test]
    fn add_a_l() {
        analyze(ADD, vec![M::A.into(), M::L.into()]).expect_fragments(vec![Fragment::Byte(0x85)])
    }

    #[test]
    fn add_a_deref_hl() {
        analyze(ADD, vec![M::A.into(), M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0x86)])
    }

    #[test]
    fn add_a_5() {
        analyze(ADD, vec![M::A.into(), 5.into()]).expect_fragments(vec![
            Fragment::Byte(0xc6),
            Fragment::Immediate(number(5, TokenId::Operand(1, 0)), Width::Byte),
        ])
    }

    #[test]
    fn adc_a_a() {
        analyze(ADC, vec![M::A.into(), M::A.into()]).expect_fragments(vec![Fragment::Byte(0x8f)])
    }

    #[test]
    fn adc_a_b() {
        analyze(ADC, vec![M::A.into(), M::B.into()]).expect_fragments(vec![Fragment::Byte(0x88)])
    }

    #[test]
    fn adc_a_c() {
        analyze(ADC, vec![M::A.into(), M::C.into()]).expect_fragments(vec![Fragment::Byte(0x89)])
    }

    #[test]
    fn adc_a_d() {
        analyze(ADC, vec![M::A.into(), M::D.into()]).expect_fragments(vec![Fragment::Byte(0x8a)])
    }

    #[test]
    fn adc_a_e() {
        analyze(ADC, vec![M::A.into(), M::E.into()]).expect_fragments(vec![Fragment::Byte(0x8b)])
    }

    #[test]
    fn adc_a_h() {
        analyze(ADC, vec![M::A.into(), M::H.into()]).expect_fragments(vec![Fragment::Byte(0x8c)])
    }

    #[test]
    fn adc_a_l() {
        analyze(ADC, vec![M::A.into(), M::L.into()]).expect_fragments(vec![Fragment::Byte(0x8d)])
    }

    #[test]
    fn adc_a_deref_hl() {
        analyze(ADC, vec![M::A.into(), M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0x8e)])
    }

    #[test]
    fn adc_a_5() {
        analyze(ADC, vec![M::A.into(), 5.into()]).expect_fragments(vec![
            Fragment::Byte(0xce),
            Fragment::Immediate(number(5, TokenId::Operand(1, 0)), Width::Byte),
        ])
    }

    #[test]
    fn sub_a() {
        analyze(SUB, vec![M::A.into()]).expect_fragments(vec![Fragment::Byte(0x97)])
    }

    #[test]
    fn sub_b() {
        analyze(SUB, vec![M::B.into()]).expect_fragments(vec![Fragment::Byte(0x90)])
    }

    #[test]
    fn sub_c() {
        analyze(SUB, vec![M::C.into()]).expect_fragments(vec![Fragment::Byte(0x91)])
    }

    #[test]
    fn sub_d() {
        analyze(SUB, vec![M::D.into()]).expect_fragments(vec![Fragment::Byte(0x92)])
    }

    #[test]
    fn sub_e() {
        analyze(SUB, vec![M::E.into()]).expect_fragments(vec![Fragment::Byte(0x93)])
    }

    #[test]
    fn sub_h() {
        analyze(SUB, vec![M::H.into()]).expect_fragments(vec![Fragment::Byte(0x94)])
    }

    #[test]
    fn sub_l() {
        analyze(SUB, vec![M::L.into()]).expect_fragments(vec![Fragment::Byte(0x95)])
    }

    #[test]
    fn sub_deref_hl() {
        analyze(SUB, vec![M::DerefHl.into()]).expect_fragments(vec![Fragment::Byte(0x96)])
    }

    #[test]
    fn sub_5() {
        analyze(SUB, vec![5.into()]).expect_fragments(vec![
            Fragment::Byte(0xd6),
            Fragment::Immediate(number(5, TokenId::Operand(0, 0)), Width::Byte),
        ])
    }

    #[test]
    fn sbc_a_a() {
        analyze(SBC, vec![M::A.into(), M::A.into()]).expect_fragments(vec![Fragment::Byte(0x9f)])
    }

    #[test]
    fn sbc_a_b() {
        analyze(SBC, vec![M::A.into(), M::B.into()]).expect_fragments(vec![Fragment::Byte(0x98)])
    }

    #[test]
    fn sbc_a_c() {
        analyze(SBC, vec![M::A.into(), M::C.into()]).expect_fragments(vec![Fragment::Byte(0x99)])
    }

    #[test]
    fn sbc_a_d() {
        analyze(SBC, vec![M::A.into(), M::D.into()]).expect_fragments(vec![Fragment::Byte(0x9a)])
    }

    #[test]
    fn sbc_a_e() {
        analyze(SBC, vec![M::A.into(), M::E.into()]).expect_fragments(vec![Fragment::Byte(0x9b)])
    }

    #[test]
    fn sbc_a_h() {
        analyze(SBC, vec![M::A.into(), M::H.into()]).expect_fragments(vec![Fragment::Byte(0x9c)])
    }

    #[test]
    fn sbc_a_l() {
        analyze(SBC, vec![M::A.into(), M::L.into()]).expect_fragments(vec![Fragment::Byte(0x9d)])
    }

    #[test]
    fn sbc_a_deref_hl() {
        analyze(SBC, vec![M::A.into(), M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0x9e)])
    }

    #[test]
    fn sbc_a_5() {
        analyze(SBC, vec![M::A.into(), 5.into()]).expect_fragments(vec![
            Fragment::Byte(0xde),
            Fragment::Immediate(number(5, TokenId::Operand(1, 0)), Width::Byte),
        ])
    }

    #[test]
    fn and_a() {
        analyze(AND, vec![M::A.into()]).expect_fragments(vec![Fragment::Byte(0xa7)])
    }

    #[test]
    fn and_b() {
        analyze(AND, vec![M::B.into()]).expect_fragments(vec![Fragment::Byte(0xa0)])
    }

    #[test]
    fn and_c() {
        analyze(AND, vec![M::C.into()]).expect_fragments(vec![Fragment::Byte(0xa1)])
    }

    #[test]
    fn and_d() {
        analyze(AND, vec![M::D.into()]).expect_fragments(vec![Fragment::Byte(0xa2)])
    }

    #[test]
    fn and_e() {
        analyze(AND, vec![M::E.into()]).expect_fragments(vec![Fragment::Byte(0xa3)])
    }

    #[test]
    fn and_h() {
        analyze(AND, vec![M::H.into()]).expect_fragments(vec![Fragment::Byte(0xa4)])
    }

    #[test]
    fn and_l() {
        analyze(AND, vec![M::L.into()]).expect_fragments(vec![Fragment::Byte(0xa5)])
    }

    #[test]
    fn and_deref_hl() {
        analyze(AND, vec![M::DerefHl.into()]).expect_fragments(vec![Fragment::Byte(0xa6)])
    }

    #[test]
    fn and_5() {
        analyze(AND, vec![5.into()]).expect_fragments(vec![
            Fragment::Byte(0xe6),
            Fragment::Immediate(number(5, TokenId::Operand(0, 0)), Width::Byte),
        ])
    }

    #[test]
    fn xor_a() {
        analyze(XOR, vec![M::A.into()]).expect_fragments(vec![Fragment::Byte(0xaf)])
    }

    #[test]
    fn xor_b() {
        analyze(XOR, vec![M::B.into()]).expect_fragments(vec![Fragment::Byte(0xa8)])
    }

    #[test]
    fn xor_c() {
        analyze(XOR, vec![M::C.into()]).expect_fragments(vec![Fragment::Byte(0xa9)])
    }

    #[test]
    fn xor_d() {
        analyze(XOR, vec![M::D.into()]).expect_fragments(vec![Fragment::Byte(0xaa)])
    }

    #[test]
    fn xor_e() {
        analyze(XOR, vec![M::E.into()]).expect_fragments(vec![Fragment::Byte(0xab)])
    }

    #[test]
    fn xor_h() {
        analyze(XOR, vec![M::H.into()]).expect_fragments(vec![Fragment::Byte(0xac)])
    }

    #[test]
    fn xor_l() {
        analyze(XOR, vec![M::L.into()]).expect_fragments(vec![Fragment::Byte(0xad)])
    }

    #[test]
    fn xor_deref_hl() {
        analyze(XOR, vec![M::DerefHl.into()]).expect_fragments(vec![Fragment::Byte(0xae)])
    }

    #[test]
    fn xor_5() {
        analyze(XOR, vec![5.into()]).expect_fragments(vec![
            Fragment::Byte(0xee),
            Fragment::Immediate(number(5, TokenId::Operand(0, 0)), Width::Byte),
        ])
    }

    #[test]
    fn or_a() {
        analyze(OR, vec![M::A.into()]).expect_fragments(vec![Fragment::Byte(0xb7)])
    }

    #[test]
    fn or_b() {
        analyze(OR, vec![M::B.into()]).expect_fragments(vec![Fragment::Byte(0xb0)])
    }

    #[test]
    fn or_c() {
        analyze(OR, vec![M::C.into()]).expect_fragments(vec![Fragment::Byte(0xb1)])
    }

    #[test]
    fn or_d() {
        analyze(OR, vec![M::D.into()]).expect_fragments(vec![Fragment::Byte(0xb2)])
    }

    #[test]
    fn or_e() {
        analyze(OR, vec![M::E.into()]).expect_fragments(vec![Fragment::Byte(0xb3)])
    }

    #[test]
    fn or_h() {
        analyze(OR, vec![M::H.into()]).expect_fragments(vec![Fragment::Byte(0xb4)])
    }

    #[test]
    fn or_l() {
        analyze(OR, vec![M::L.into()]).expect_fragments(vec![Fragment::Byte(0xb5)])
    }

    #[test]
    fn or_deref_hl() {
        analyze(OR, vec![M::DerefHl.into()]).expect_fragments(vec![Fragment::Byte(0xb6)])
    }

    #[test]
    fn or_5() {
        analyze(OR, vec![5.into()]).expect_fragments(vec![
            Fragment::Byte(0xf6),
            Fragment::Immediate(number(5, TokenId::Operand(0, 0)), Width::Byte),
        ])
    }

    #[test]
    fn cp_a() {
        analyze(CP, vec![M::A.into()]).expect_fragments(vec![Fragment::Byte(0xbf)])
    }

    #[test]
    fn cp_b() {
        analyze(CP, vec![M::B.into()]).expect_fragments(vec![Fragment::Byte(0xb8)])
    }

    #[test]
    fn cp_c() {
        analyze(CP, vec![M::C.into()]).expect_fragments(vec![Fragment::Byte(0xb9)])
    }

    #[test]
    fn cp_d() {
        analyze(CP, vec![M::D.into()]).expect_fragments(vec![Fragment::Byte(0xba)])
    }

    #[test]
    fn cp_e() {
        analyze(CP, vec![M::E.into()]).expect_fragments(vec![Fragment::Byte(0xbb)])
    }

    #[test]
    fn cp_h() {
        analyze(CP, vec![M::H.into()]).expect_fragments(vec![Fragment::Byte(0xbc)])
    }

    #[test]
    fn cp_l() {
        analyze(CP, vec![M::L.into()]).expect_fragments(vec![Fragment::Byte(0xbd)])
    }

    #[test]
    fn cp_deref_hl() {
        analyze(CP, vec![M::DerefHl.into()]).expect_fragments(vec![Fragment::Byte(0xbe)])
    }

    #[test]
    fn cp_5() {
        analyze(CP, vec![5.into()]).expect_fragments(vec![
            Fragment::Byte(0xfe),
            Fragment::Immediate(number(5, TokenId::Operand(0, 0)), Width::Byte),
        ])
    }

    #[test]
    fn add_hl_bc() {
        analyze(ADD, vec![Reg16::Hl.into(), Reg16::Bc.into()])
            .expect_fragments(vec![Fragment::Byte(0x09)])
    }

    #[test]
    fn add_hl_de() {
        analyze(ADD, vec![Reg16::Hl.into(), Reg16::De.into()])
            .expect_fragments(vec![Fragment::Byte(0x19)])
    }

    #[test]
    fn add_hl_hl() {
        analyze(ADD, vec![Reg16::Hl.into(), Reg16::Hl.into()])
            .expect_fragments(vec![Fragment::Byte(0x29)])
    }

    #[test]
    fn add_hl_sp() {
        analyze(ADD, vec![Reg16::Hl.into(), Reg16::Sp.into()])
            .expect_fragments(vec![Fragment::Byte(0x39)])
    }

    #[test]
    fn bit_4_a() {
        analyze(BIT, vec![4.into(), M::A.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b01_000_111, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn bit_4_b() {
        analyze(BIT, vec![4.into(), M::B.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b01_000_000, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn bit_4_c() {
        analyze(BIT, vec![4.into(), M::C.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b01_000_001, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn bit_4_d() {
        analyze(BIT, vec![4.into(), M::D.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b01_000_010, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn bit_4_e() {
        analyze(BIT, vec![4.into(), M::E.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b01_000_011, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn bit_4_h() {
        analyze(BIT, vec![4.into(), M::H.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b01_000_100, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn bit_4_l() {
        analyze(BIT, vec![4.into(), M::L.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b01_000_101, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn bit_4_deref_hl() {
        analyze(BIT, vec![4.into(), M::DerefHl.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b01_000_110, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn set_4_a() {
        analyze(SET, vec![4.into(), M::A.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b11_000_111, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn set_4_b() {
        analyze(SET, vec![4.into(), M::B.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b11_000_000, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn set_4_c() {
        analyze(SET, vec![4.into(), M::C.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b11_000_001, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn set_4_d() {
        analyze(SET, vec![4.into(), M::D.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b11_000_010, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn set_4_e() {
        analyze(SET, vec![4.into(), M::E.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b11_000_011, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn set_4_h() {
        analyze(SET, vec![4.into(), M::H.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b11_000_100, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn set_4_l() {
        analyze(SET, vec![4.into(), M::L.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b11_000_101, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn set_4_deref_hl() {
        analyze(SET, vec![4.into(), M::DerefHl.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b11_000_110, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn res_4_a() {
        analyze(RES, vec![4.into(), M::A.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b10_000_111, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn res_4_b() {
        analyze(RES, vec![4.into(), M::B.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b10_000_000, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn res_4_c() {
        analyze(RES, vec![4.into(), M::C.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b10_000_001, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn res_4_d() {
        analyze(RES, vec![4.into(), M::D.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b10_000_010, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn res_4_e() {
        analyze(RES, vec![4.into(), M::E.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b10_000_011, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn res_4_h() {
        analyze(RES, vec![4.into(), M::H.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b10_000_100, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn res_4_l() {
        analyze(RES, vec![4.into(), M::L.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b10_000_101, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn res_4_deref_hl() {
        analyze(RES, vec![4.into(), M::DerefHl.into()]).expect_fragments(vec![
            Fragment::Byte(0xcb),
            Fragment::Embedded(0b10_000_110, number(4, TokenId::Operand(0, 0))),
        ])
    }

    #[test]
    fn inc_a() {
        analyze(INC, vec![M::A.into()]).expect_fragments(vec![Fragment::Byte(0x3c)])
    }

    #[test]
    fn inc_b() {
        analyze(INC, vec![M::B.into()]).expect_fragments(vec![Fragment::Byte(0x04)])
    }

    #[test]
    fn inc_c() {
        analyze(INC, vec![M::C.into()]).expect_fragments(vec![Fragment::Byte(0x0c)])
    }

    #[test]
    fn inc_d() {
        analyze(INC, vec![M::D.into()]).expect_fragments(vec![Fragment::Byte(0x14)])
    }

    #[test]
    fn inc_e() {
        analyze(INC, vec![M::E.into()]).expect_fragments(vec![Fragment::Byte(0x1c)])
    }

    #[test]
    fn inc_h() {
        analyze(INC, vec![M::H.into()]).expect_fragments(vec![Fragment::Byte(0x24)])
    }

    #[test]
    fn inc_l() {
        analyze(INC, vec![M::L.into()]).expect_fragments(vec![Fragment::Byte(0x2c)])
    }

    #[test]
    fn inc_deref_hl() {
        analyze(INC, vec![M::DerefHl.into()]).expect_fragments(vec![Fragment::Byte(0x34)])
    }

    #[test]
    fn dec_a() {
        analyze(DEC, vec![M::A.into()]).expect_fragments(vec![Fragment::Byte(0x3d)])
    }

    #[test]
    fn dec_b() {
        analyze(DEC, vec![M::B.into()]).expect_fragments(vec![Fragment::Byte(0x05)])
    }

    #[test]
    fn dec_c() {
        analyze(DEC, vec![M::C.into()]).expect_fragments(vec![Fragment::Byte(0x0d)])
    }

    #[test]
    fn dec_d() {
        analyze(DEC, vec![M::D.into()]).expect_fragments(vec![Fragment::Byte(0x15)])
    }

    #[test]
    fn dec_e() {
        analyze(DEC, vec![M::E.into()]).expect_fragments(vec![Fragment::Byte(0x1d)])
    }

    #[test]
    fn dec_h() {
        analyze(DEC, vec![M::H.into()]).expect_fragments(vec![Fragment::Byte(0x25)])
    }

    #[test]
    fn dec_l() {
        analyze(DEC, vec![M::L.into()]).expect_fragments(vec![Fragment::Byte(0x2d)])
    }

    #[test]
    fn dec_deref_hl() {
        analyze(DEC, vec![M::DerefHl.into()]).expect_fragments(vec![Fragment::Byte(0x35)])
    }

    #[test]
    fn inc_bc() {
        analyze(INC, vec![Reg16::Bc.into()]).expect_fragments(vec![Fragment::Byte(0x03)])
    }

    #[test]
    fn inc_de() {
        analyze(INC, vec![Reg16::De.into()]).expect_fragments(vec![Fragment::Byte(0x13)])
    }

    #[test]
    fn inc_hl() {
        analyze(INC, vec![Reg16::Hl.into()]).expect_fragments(vec![Fragment::Byte(0x23)])
    }

    #[test]
    fn inc_sp() {
        analyze(INC, vec![Reg16::Sp.into()]).expect_fragments(vec![Fragment::Byte(0x33)])
    }

    #[test]
    fn dec_bc() {
        analyze(DEC, vec![Reg16::Bc.into()]).expect_fragments(vec![Fragment::Byte(0x0b)])
    }

    #[test]
    fn dec_de() {
        analyze(DEC, vec![Reg16::De.into()]).expect_fragments(vec![Fragment::Byte(0x1b)])
    }

    #[test]
    fn dec_hl() {
        analyze(DEC, vec![Reg16::Hl.into()]).expect_fragments(vec![Fragment::Byte(0x2b)])
    }

    #[test]
    fn dec_sp() {
        analyze(DEC, vec![Reg16::Sp.into()]).expect_fragments(vec![Fragment::Byte(0x3b)])
    }

    #[test]
    fn rlc_a() {
        analyze(RLC, vec![M::A.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x07)])
    }

    #[test]
    fn rlc_b() {
        analyze(RLC, vec![M::B.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x00)])
    }

    #[test]
    fn rlc_c() {
        analyze(RLC, vec![M::C.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x01)])
    }

    #[test]
    fn rlc_d() {
        analyze(RLC, vec![M::D.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x02)])
    }

    #[test]
    fn rlc_e() {
        analyze(RLC, vec![M::E.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x03)])
    }

    #[test]
    fn rlc_h() {
        analyze(RLC, vec![M::H.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x04)])
    }

    #[test]
    fn rlc_l() {
        analyze(RLC, vec![M::L.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x05)])
    }

    #[test]
    fn rlc_deref_hl() {
        analyze(RLC, vec![M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x06)])
    }

    #[test]
    fn rrc_a() {
        analyze(RRC, vec![M::A.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x0f)])
    }

    #[test]
    fn rrc_b() {
        analyze(RRC, vec![M::B.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x08)])
    }

    #[test]
    fn rrc_c() {
        analyze(RRC, vec![M::C.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x09)])
    }

    #[test]
    fn rrc_d() {
        analyze(RRC, vec![M::D.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x0a)])
    }

    #[test]
    fn rrc_e() {
        analyze(RRC, vec![M::E.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x0b)])
    }

    #[test]
    fn rrc_h() {
        analyze(RRC, vec![M::H.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x0c)])
    }

    #[test]
    fn rrc_l() {
        analyze(RRC, vec![M::L.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x0d)])
    }

    #[test]
    fn rrc_deref_hl() {
        analyze(RRC, vec![M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x0e)])
    }

    #[test]
    fn rl_a() {
        analyze(RL, vec![M::A.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x17)])
    }

    #[test]
    fn rl_b() {
        analyze(RL, vec![M::B.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x10)])
    }

    #[test]
    fn rl_c() {
        analyze(RL, vec![M::C.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x11)])
    }

    #[test]
    fn rl_d() {
        analyze(RL, vec![M::D.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x12)])
    }

    #[test]
    fn rl_e() {
        analyze(RL, vec![M::E.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x13)])
    }

    #[test]
    fn rl_h() {
        analyze(RL, vec![M::H.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x14)])
    }

    #[test]
    fn rl_l() {
        analyze(RL, vec![M::L.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x15)])
    }

    #[test]
    fn rl_deref_hl() {
        analyze(RL, vec![M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x16)])
    }

    #[test]
    fn rr_a() {
        analyze(RR, vec![M::A.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x1f)])
    }

    #[test]
    fn rr_b() {
        analyze(RR, vec![M::B.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x18)])
    }

    #[test]
    fn rr_c() {
        analyze(RR, vec![M::C.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x19)])
    }

    #[test]
    fn rr_d() {
        analyze(RR, vec![M::D.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x1a)])
    }

    #[test]
    fn rr_e() {
        analyze(RR, vec![M::E.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x1b)])
    }

    #[test]
    fn rr_h() {
        analyze(RR, vec![M::H.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x1c)])
    }

    #[test]
    fn rr_l() {
        analyze(RR, vec![M::L.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x1d)])
    }

    #[test]
    fn rr_deref_hl() {
        analyze(RR, vec![M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x1e)])
    }

    #[test]
    fn sla_a() {
        analyze(SLA, vec![M::A.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x27)])
    }

    #[test]
    fn sla_b() {
        analyze(SLA, vec![M::B.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x20)])
    }

    #[test]
    fn sla_c() {
        analyze(SLA, vec![M::C.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x21)])
    }

    #[test]
    fn sla_d() {
        analyze(SLA, vec![M::D.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x22)])
    }

    #[test]
    fn sla_e() {
        analyze(SLA, vec![M::E.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x23)])
    }

    #[test]
    fn sla_h() {
        analyze(SLA, vec![M::H.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x24)])
    }

    #[test]
    fn sla_l() {
        analyze(SLA, vec![M::L.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x25)])
    }

    #[test]
    fn sla_deref_hl() {
        analyze(SLA, vec![M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x26)])
    }

    #[test]
    fn sra_a() {
        analyze(SRA, vec![M::A.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x2f)])
    }

    #[test]
    fn sra_b() {
        analyze(SRA, vec![M::B.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x28)])
    }

    #[test]
    fn sra_c() {
        analyze(SRA, vec![M::C.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x29)])
    }

    #[test]
    fn sra_d() {
        analyze(SRA, vec![M::D.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x2a)])
    }

    #[test]
    fn sra_e() {
        analyze(SRA, vec![M::E.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x2b)])
    }

    #[test]
    fn sra_h() {
        analyze(SRA, vec![M::H.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x2c)])
    }

    #[test]
    fn sra_l() {
        analyze(SRA, vec![M::L.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x2d)])
    }

    #[test]
    fn sra_deref_hl() {
        analyze(SRA, vec![M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x2e)])
    }

    #[test]
    fn swap_a() {
        analyze(SWAP, vec![M::A.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x37)])
    }

    #[test]
    fn swap_b() {
        analyze(SWAP, vec![M::B.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x30)])
    }

    #[test]
    fn swap_c() {
        analyze(SWAP, vec![M::C.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x31)])
    }

    #[test]
    fn swap_d() {
        analyze(SWAP, vec![M::D.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x32)])
    }

    #[test]
    fn swap_e() {
        analyze(SWAP, vec![M::E.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x33)])
    }

    #[test]
    fn swap_h() {
        analyze(SWAP, vec![M::H.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x34)])
    }

    #[test]
    fn swap_l() {
        analyze(SWAP, vec![M::L.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x35)])
    }

    #[test]
    fn swap_deref_hl() {
        analyze(SWAP, vec![M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x36)])
    }

    #[test]
    fn srl_a() {
        analyze(SRL, vec![M::A.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x3f)])
    }

    #[test]
    fn srl_b() {
        analyze(SRL, vec![M::B.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x38)])
    }

    #[test]
    fn srl_c() {
        analyze(SRL, vec![M::C.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x39)])
    }

    #[test]
    fn srl_d() {
        analyze(SRL, vec![M::D.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x3a)])
    }

    #[test]
    fn srl_e() {
        analyze(SRL, vec![M::E.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x3b)])
    }

    #[test]
    fn srl_h() {
        analyze(SRL, vec![M::H.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x3c)])
    }

    #[test]
    fn srl_l() {
        analyze(SRL, vec![M::L.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x3d)])
    }

    #[test]
    fn srl_deref_hl() {
        analyze(SRL, vec![M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0xcb), Fragment::Byte(0x3e)])
    }

    #[test]
    fn add_sp_5() {
        analyze(ADD, vec![Reg16::Sp.into(), 5.into()]).expect_fragments(vec![
            Fragment::Byte(0xe8),
            Fragment::Immediate(number(5, TokenId::Operand(1, 0)), Width::Byte),
        ])
    }

    pub(super) struct AnalysisResult(Vec<Event<TokenSpan>>);

    impl AnalysisResult {
        pub fn expect_fragments(self, expected: Vec<Fragment<Expr<TokenSpan>>>) {
            assert_eq!(
                self.0,
                expected
                    .into_iter()
                    .map(|fragment| Event::EmitFragment { fragment })
                    .collect::<Vec<_>>()
            )
        }

        pub fn expect_diag(self, diag: impl Into<ExpectedDiag>) {
            let expected = diag.into();
            assert_eq!(
                self.0,
                vec![Event::EmitDiag {
                    diag: expected.message.at(expected.highlight.unwrap()).into()
                }]
            )
        }
    }

    pub(super) fn analyze<I>(mnemonic: Mnemonic, operands: I) -> AnalysisResult
    where
        I: IntoIterator<Item = Input>,
    {
        use super::operand::analyze_operand;

        let mut fixture = TestFixture::new();
        let mut session = fixture.session();
        let operands: Vec<_> = operands
            .into_iter()
            .enumerate()
            .map(add_token_spans)
            .map(|op| analyze_operand(op, mnemonic.context(), &mut session))
            .collect();
        analyze_instruction(
            (&mnemonic, TokenId::Mnemonic.into()),
            operands,
            &mut session,
        );
        AnalysisResult(session.log().to_vec())
    }

    fn add_token_spans((i, operand): (usize, Input)) -> Arg<TokenSpan> {
        match operand {
            Arg::Bare(BareArg::Const(value)) => Arg::Bare(BareArg::Const(crate::expr::Expr(
                value
                    .0
                    .into_iter()
                    .map(|Spanned { item, .. }| item.with_span(TokenId::Operand(i, 0).into()))
                    .collect(),
            ))),
            Arg::Bare(BareArg::OperandKeyword(symbol, ())) => Arg::Bare(BareArg::OperandKeyword(
                symbol,
                TokenId::Operand(i, 0).into(),
            )),
            Arg::Deref(BareArg::Const(value), ()) => Arg::Deref(
                BareArg::Const(crate::expr::Expr(
                    value
                        .0
                        .into_iter()
                        .map(|Spanned { item, .. }| item.with_span(TokenId::Operand(i, 1).into()))
                        .collect(),
                )),
                TokenSpan::merge(TokenId::Operand(i, 0), TokenId::Operand(i, 2)),
            ),
            Arg::Deref(BareArg::OperandKeyword(symbol, ()), ()) => Arg::Deref(
                BareArg::OperandKeyword(symbol, TokenId::Operand(i, 1).into()),
                TokenSpan::merge(TokenId::Operand(i, 0), TokenId::Operand(i, 2)),
            ),
            _ => unimplemented!(),
        }
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
    fn analyze_add_a_a_a() {
        assert_eq!(
            analyze(ADD, vec![A, A, A].into_iter().map(literal)).0,
            vec![
                Event::EmitFragment {
                    fragment: Fragment::Byte(0x87)
                },
                Event::EmitDiag {
                    diag: Message::OperandCount {
                        actual: 3,
                        expected: 2,
                    }
                    .at(TokenId::Mnemonic.into())
                    .into()
                }
            ]
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
        analyze(ADD, vec![literal(A), deref_symbol(Bc)]).expect_diag(
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
