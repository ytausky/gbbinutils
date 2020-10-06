use super::operand::AtomKind;
use super::{Analysis, Expr, Fragment, Operand, Width};

use crate::assembler::session::builder::*;
use crate::diagnostics::{Diagnostics, EmitDiag, Message};
use crate::span::{Source, SpanSource};

impl<'a, 'b, I, D, S> Analysis<'a, 'b, I, D, S>
where
    I: Iterator<Item = Result<Operand<D::SymbolId, S>, ()>>,
    D: Backend<S> + Diagnostics<S>,
    S: Clone,
{
    pub fn analyze_ld(&mut self) -> Result<(), ()> {
        let dest = self.next_operand_of(2)?;
        let src = self.next_operand_of(2)?;
        match (
            dest.into_ld_dest(self.session)?,
            src.into_ld_src(self.session)?,
        ) {
            (LdDest::Byte(dest), LdOperand::Other(LdDest::Byte(src))) => {
                self.analyze_8_bit_ld(dest, src)
            }
            (LdDest::Byte(dest), LdOperand::Const(src)) => {
                self.analyze_8_bit_ld(dest, LdOperand::Const(src))
            }
            (LdDest::Word(dest), LdOperand::Other(LdDest::Word(src))) => {
                self.analyze_16_bit_ld(dest, src)
            }
            (LdDest::Word(dest), LdOperand::Const(src)) => {
                self.analyze_16_bit_ld(dest, LdOperand::Const(src))
            }
            (LdDest::Byte(dest), LdOperand::Other(LdDest::Word(src))) => {
                self.diagnose_ld_width_mismatch(&dest, &src)
            }
            (LdDest::Word(dest), LdOperand::Other(LdDest::Byte(src))) => {
                self.diagnose_ld_width_mismatch(&dest, &src)
            }
            (LdDest::DerefExpr(expr), LdOperand::Other(LdDest::Word(src))) => {
                self.analyze_16_bit_ld(LdDest16::DerefNn(expr), src)
            }
            (LdDest::DerefExpr(expr), LdOperand::Other(LdDest::Byte(src))) => {
                self.analyze_8_bit_ld(LdDest8::Special(LdSpecial::Deref(expr)), src)
            }
            (LdDest::Byte(dest), LdOperand::Other(LdDest::DerefExpr(expr))) => self
                .analyze_8_bit_ld(
                    dest,
                    LdOperand::Other(LdDest8::Special(LdSpecial::Deref(expr))),
                ),
            _ => todo!(),
        }
    }

    fn diagnose_ld_width_mismatch(
        &mut self,
        dest: &impl Source<Span = S>,
        src: &(impl Source<Span = S> + DataWidth),
    ) -> Result<(), ()> {
        let session = &mut self.session;
        let diagnostic = Message::LdWidthMismatch {
            src_width: src.width(),
            src: session.strip_span(&src.span()),
            dest: session.strip_span(&dest.span()),
        }
        .at(session.merge_spans(&dest.span(), &src.span()));
        session.emit_diag(diagnostic);
        Err(())
    }

    fn analyze_8_bit_ld(
        &mut self,
        dest: LdDest8<D::SymbolId, S>,
        src: impl Into<LdOperand<D::SymbolId, LdDest8<D::SymbolId, S>, S>>,
    ) -> Result<(), ()> {
        match (dest, src.into()) {
            (
                LdDest8::Simple(M::DerefHl, dest),
                LdOperand::Other(LdDest8::Simple(M::DerefHl, src)),
            ) => {
                let session = &mut self.session;
                let diagnostic = Message::LdDerefHlDerefHl {
                    mnemonic: session.strip_span(&self.mnemonic.1),
                    dest: session.strip_span(&dest),
                    src: session.strip_span(&src),
                }
                .at(session.merge_spans(&self.mnemonic.1, &src));
                self.emit_diag(diagnostic);
                Err(())
            }
            (LdDest8::Simple(dest, _), LdOperand::Other(LdDest8::Simple(src, _))) => {
                self.session.emit_fragment(Fragment::Byte(
                    0b01_000_000 | (dest.encode() << 3) | src.encode(),
                ));
                Ok(())
            }
            (LdDest8::Simple(dest, _), LdOperand::Const(expr)) => {
                self.session
                    .emit_fragment(Fragment::Byte(0x06 | (dest.encode() << 3)));
                self.session
                    .emit_fragment(Fragment::Immediate(expr, Width::Byte));
                Ok(())
            }
            (LdDest8::Special(dest), src) => {
                src.expect_a(self.session)?;
                self.analyze_special_ld(dest, Direction::FromA)
            }
            (dest, LdOperand::Other(LdDest8::Special(src))) => {
                dest.expect_a(self.session)?;
                self.analyze_special_ld(src, Direction::IntoA)
            }
        }
    }

    fn analyze_16_bit_ld(
        &mut self,
        dest: LdDest16<D::SymbolId, S>,
        src: impl Into<LdOperand<D::SymbolId, LdDest16<D::SymbolId, S>, S>>,
    ) -> Result<(), ()> {
        match (dest, src.into()) {
            (LdDest16::Reg16(Reg16::Sp, _), LdOperand::Other(LdDest16::Reg16(Reg16::Hl, _))) => {
                self.session.emit_fragment(Fragment::Byte(0xf9));
                Ok(())
            }
            (LdDest16::Reg16(_, dest_span), LdOperand::Other(LdDest16::Reg16(_, src_span))) => {
                let session = &mut self.session;
                let merged_span = session.merge_spans(&dest_span, &src_span);
                session.emit_diag(Message::LdSpHlOperands.at(merged_span));
                Err(())
            }
            (LdDest16::Reg16(dest, _), LdOperand::Const(expr)) => {
                self.session
                    .emit_fragment(Fragment::Byte(0x01 | encode_reg16(dest)));
                self.session
                    .emit_fragment(Fragment::Immediate(expr, Width::Word));
                Ok(())
            }
            (LdDest16::DerefNn(nn), LdOperand::Other(LdDest16::Reg16(Reg16::Sp, _))) => {
                self.session.emit_fragment(Fragment::Byte(0x8));
                self.session
                    .emit_fragment(Fragment::Immediate(nn, Width::Word));
                Ok(())
            }
            _ => todo!(),
        }
    }

    fn analyze_special_ld(
        &mut self,
        other: LdSpecial<D::SymbolId, S>,
        direction: Direction,
    ) -> Result<(), ()> {
        match other {
            LdSpecial::DerefPtrReg(ptr_reg, _) => self.session.emit_fragment(Fragment::Byte(
                0x02 | encode_ptr_reg(ptr_reg) | (encode_direction(direction) >> 1),
            )),
            LdSpecial::DerefC(_) => self
                .session
                .emit_fragment(Fragment::Byte(0xe2 | encode_direction(direction))),
            LdSpecial::Deref(expr) => self.session.emit_fragment(Fragment::LdInlineAddr(
                0xe0 | encode_direction(direction),
                expr,
            )),
        };
        Ok(())
    }
}

impl<N, S: Clone> Operand<N, S> {
    fn into_ld_dest<D>(self, diagnostics: &mut D) -> Result<LdDest<N, S>, ()>
    where
        D: Diagnostics<S>,
    {
        match self {
            Operand::Deref(expr) => Ok(LdDest::DerefExpr(expr)),
            Operand::Atom(kind, span) => match kind {
                AtomKind::Condition(_) => Err(Message::ConditionOutsideBranch.at(span)),
                AtomKind::Simple(simple) => Ok(LdDest::Byte(LdDest8::Simple(simple, span))),
                AtomKind::DerefC => Ok(LdDest::Byte(LdDest8::Special(LdSpecial::DerefC(span)))),
                AtomKind::DerefPtrReg(ptr_reg) => Ok(LdDest::Byte(LdDest8::Special(
                    LdSpecial::DerefPtrReg(ptr_reg, span),
                ))),
                AtomKind::Reg16(reg16) => Ok(LdDest::Word(LdDest16::Reg16(reg16, span))),
                AtomKind::RegPair(reg_pair) => {
                    assert_eq!(reg_pair, RegPair::Af);
                    Err(Message::AfOutsideStackOperation.at(span))
                }
            },
            Operand::Const(expr) => Err(Message::DestCannotBeConst.at(expr.span())),
        }
        .map_err(|diagnostic| {
            diagnostics.emit_diag(diagnostic);
        })
    }

    fn into_ld_src<D>(self, diagnostics: &mut D) -> Result<LdOperand<N, LdDest<N, S>, S>, ()>
    where
        D: Diagnostics<S>,
    {
        match self {
            Operand::Const(expr) => Ok(LdOperand::Const(expr)),
            operand => Ok(LdOperand::Other(operand.into_ld_dest(diagnostics)?)),
        }
    }
}

enum LdOperand<N, T, S> {
    Const(Expr<N, S>),
    Other(T),
}

impl<N, S> From<LdDest8<N, S>> for LdOperand<N, LdDest8<N, S>, S> {
    fn from(dest: LdDest8<N, S>) -> Self {
        LdOperand::Other(dest)
    }
}

impl<N, S> From<LdDest16<N, S>> for LdOperand<N, LdDest16<N, S>, S> {
    fn from(dest: LdDest16<N, S>) -> Self {
        LdOperand::Other(dest)
    }
}

enum LdDest<N, S> {
    Byte(LdDest8<N, S>),
    Word(LdDest16<N, S>),
    DerefExpr(Expr<N, S>),
}

enum LdDest8<N, S> {
    Simple(M, S),
    Special(LdSpecial<N, S>),
}

enum LdSpecial<N, S> {
    Deref(Expr<N, S>),
    DerefC(S),
    DerefPtrReg(PtrReg, S),
}

enum LdDest16<N, S> {
    Reg16(Reg16, S),
    DerefNn(Expr<N, S>),
}

trait DataWidth {
    fn width(&self) -> Width;
}

impl<N, S> DataWidth for LdDest8<N, S> {
    fn width(&self) -> Width {
        Width::Byte
    }
}

impl<N, S> DataWidth for LdDest16<N, S> {
    fn width(&self) -> Width {
        Width::Word
    }
}

impl<N, S: Clone> LdOperand<N, LdDest8<N, S>, S> {
    fn expect_a<D>(self, diagnostics: &mut D) -> Result<(), ()>
    where
        D: Diagnostics<S>,
    {
        match self {
            LdOperand::Const(expr) => diagnose_not_a(expr.span(), diagnostics),
            LdOperand::Other(other) => other.expect_a(diagnostics),
        }
    }
}

impl<N, S: Clone> LdDest8<N, S> {
    fn expect_a<D>(self, diagnostics: &mut D) -> Result<(), ()>
    where
        D: Diagnostics<S>,
    {
        match self {
            LdDest8::Simple(M::A, _) => Ok(()),
            operand => diagnose_not_a(operand.span(), diagnostics),
        }
    }
}

fn diagnose_not_a<T, D: EmitDiag<S, T>, S>(span: S, diagnostics: &mut D) -> Result<(), ()> {
    diagnostics.emit_diag(Message::OnlySupportedByA.at(span));
    Err(())
}

impl<N, T: Source<Span = S>, S: Clone> SpanSource for LdOperand<N, T, S> {
    type Span = S;
}

impl<N, T: Source<Span = S>, S: Clone> Source for LdOperand<N, T, S> {
    fn span(&self) -> Self::Span {
        match self {
            LdOperand::Const(expr) => expr.span(),
            LdOperand::Other(operand) => operand.span(),
        }
    }
}

impl<N, S: Clone> SpanSource for LdDest8<N, S> {
    type Span = S;
}

impl<N, S: Clone> Source for LdDest8<N, S> {
    fn span(&self) -> Self::Span {
        use self::LdDest8::*;
        match self {
            Simple(_, span) => span.clone(),
            Special(special) => special.span(),
        }
    }
}

impl<N, S: Clone> SpanSource for LdSpecial<N, S> {
    type Span = S;
}

impl<N, S: Clone> Source for LdSpecial<N, S> {
    fn span(&self) -> Self::Span {
        use self::LdSpecial::*;
        match self {
            Deref(expr) => expr.span(),
            DerefC(span) | DerefPtrReg(_, span) => span.clone(),
        }
    }
}

impl<N, S: Clone> SpanSource for LdDest16<N, S> {
    type Span = S;
}

impl<N, S: Clone> Source for LdDest16<N, S> {
    fn span(&self) -> Self::Span {
        match self {
            LdDest16::Reg16(_, span) => span.clone(),
            LdDest16::DerefNn(nn) => nn.span(),
        }
    }
}

fn encode_direction(direction: Direction) -> u8 {
    match direction {
        Direction::FromA => 0x00,
        Direction::IntoA => 0x10,
    }
}

fn encode_ptr_reg(ptr_reg: PtrReg) -> u8 {
    use self::PtrReg::*;
    (match ptr_reg {
        Bc => 0b00,
        De => 0b01,
        Hli => 0b10,
        Hld => 0b11,
    }) << 4
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

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;

    use crate::assembler::keywords::LD;
    use crate::diagnostics::Merge;

    #[test]
    fn ld_a_a() {
        analyze(LD, vec![M::A.into(), M::A.into()]).expect_fragments(vec![Fragment::Byte(0x7f)])
    }

    #[test]
    fn ld_a_b() {
        analyze(LD, vec![M::A.into(), M::B.into()]).expect_fragments(vec![Fragment::Byte(0x78)])
    }

    #[test]
    fn ld_a_c() {
        analyze(LD, vec![M::A.into(), M::C.into()]).expect_fragments(vec![Fragment::Byte(0x79)])
    }

    #[test]
    fn ld_a_d() {
        analyze(LD, vec![M::A.into(), M::D.into()]).expect_fragments(vec![Fragment::Byte(0x7a)])
    }

    #[test]
    fn ld_a_e() {
        analyze(LD, vec![M::A.into(), M::E.into()]).expect_fragments(vec![Fragment::Byte(0x7b)])
    }

    #[test]
    fn ld_a_h() {
        analyze(LD, vec![M::A.into(), M::H.into()]).expect_fragments(vec![Fragment::Byte(0x7c)])
    }

    #[test]
    fn ld_a_l() {
        analyze(LD, vec![M::A.into(), M::L.into()]).expect_fragments(vec![Fragment::Byte(0x7d)])
    }

    #[test]
    fn ld_b_a() {
        analyze(LD, vec![M::B.into(), M::A.into()]).expect_fragments(vec![Fragment::Byte(0x47)])
    }

    #[test]
    fn ld_b_b() {
        analyze(LD, vec![M::B.into(), M::B.into()]).expect_fragments(vec![Fragment::Byte(0x40)])
    }

    #[test]
    fn ld_b_c() {
        analyze(LD, vec![M::B.into(), M::C.into()]).expect_fragments(vec![Fragment::Byte(0x41)])
    }

    #[test]
    fn ld_b_d() {
        analyze(LD, vec![M::B.into(), M::D.into()]).expect_fragments(vec![Fragment::Byte(0x42)])
    }

    #[test]
    fn ld_b_e() {
        analyze(LD, vec![M::B.into(), M::E.into()]).expect_fragments(vec![Fragment::Byte(0x43)])
    }

    #[test]
    fn ld_b_h() {
        analyze(LD, vec![M::B.into(), M::H.into()]).expect_fragments(vec![Fragment::Byte(0x44)])
    }

    #[test]
    fn ld_b_l() {
        analyze(LD, vec![M::B.into(), M::L.into()]).expect_fragments(vec![Fragment::Byte(0x45)])
    }

    #[test]
    fn ld_c_a() {
        analyze(LD, vec![M::C.into(), M::A.into()]).expect_fragments(vec![Fragment::Byte(0x4f)])
    }

    #[test]
    fn ld_c_b() {
        analyze(LD, vec![M::C.into(), M::B.into()]).expect_fragments(vec![Fragment::Byte(0x48)])
    }

    #[test]
    fn ld_c_c() {
        analyze(LD, vec![M::C.into(), M::C.into()]).expect_fragments(vec![Fragment::Byte(0x49)])
    }

    #[test]
    fn ld_c_d() {
        analyze(LD, vec![M::C.into(), M::D.into()]).expect_fragments(vec![Fragment::Byte(0x4a)])
    }

    #[test]
    fn ld_c_e() {
        analyze(LD, vec![M::C.into(), M::E.into()]).expect_fragments(vec![Fragment::Byte(0x4b)])
    }

    #[test]
    fn ld_c_h() {
        analyze(LD, vec![M::C.into(), M::H.into()]).expect_fragments(vec![Fragment::Byte(0x4c)])
    }

    #[test]
    fn ld_c_l() {
        analyze(LD, vec![M::C.into(), M::L.into()]).expect_fragments(vec![Fragment::Byte(0x4d)])
    }

    #[test]
    fn ld_d_a() {
        analyze(LD, vec![M::D.into(), M::A.into()]).expect_fragments(vec![Fragment::Byte(0x57)])
    }

    #[test]
    fn ld_d_b() {
        analyze(LD, vec![M::D.into(), M::B.into()]).expect_fragments(vec![Fragment::Byte(0x50)])
    }

    #[test]
    fn ld_d_c() {
        analyze(LD, vec![M::D.into(), M::C.into()]).expect_fragments(vec![Fragment::Byte(0x51)])
    }

    #[test]
    fn ld_d_d() {
        analyze(LD, vec![M::D.into(), M::D.into()]).expect_fragments(vec![Fragment::Byte(0x52)])
    }

    #[test]
    fn ld_d_e() {
        analyze(LD, vec![M::D.into(), M::E.into()]).expect_fragments(vec![Fragment::Byte(0x53)])
    }

    #[test]
    fn ld_d_h() {
        analyze(LD, vec![M::D.into(), M::H.into()]).expect_fragments(vec![Fragment::Byte(0x54)])
    }

    #[test]
    fn ld_d_l() {
        analyze(LD, vec![M::D.into(), M::L.into()]).expect_fragments(vec![Fragment::Byte(0x55)])
    }

    #[test]
    fn ld_e_a() {
        analyze(LD, vec![M::E.into(), M::A.into()]).expect_fragments(vec![Fragment::Byte(0x5f)])
    }

    #[test]
    fn ld_e_b() {
        analyze(LD, vec![M::E.into(), M::B.into()]).expect_fragments(vec![Fragment::Byte(0x58)])
    }

    #[test]
    fn ld_e_c() {
        analyze(LD, vec![M::E.into(), M::C.into()]).expect_fragments(vec![Fragment::Byte(0x59)])
    }

    #[test]
    fn ld_e_d() {
        analyze(LD, vec![M::E.into(), M::D.into()]).expect_fragments(vec![Fragment::Byte(0x5a)])
    }

    #[test]
    fn ld_e_e() {
        analyze(LD, vec![M::E.into(), M::E.into()]).expect_fragments(vec![Fragment::Byte(0x5b)])
    }

    #[test]
    fn ld_e_h() {
        analyze(LD, vec![M::E.into(), M::H.into()]).expect_fragments(vec![Fragment::Byte(0x5c)])
    }

    #[test]
    fn ld_e_l() {
        analyze(LD, vec![M::E.into(), M::L.into()]).expect_fragments(vec![Fragment::Byte(0x5d)])
    }

    #[test]
    fn ld_h_a() {
        analyze(LD, vec![M::H.into(), M::A.into()]).expect_fragments(vec![Fragment::Byte(0x67)])
    }

    #[test]
    fn ld_h_b() {
        analyze(LD, vec![M::H.into(), M::B.into()]).expect_fragments(vec![Fragment::Byte(0x60)])
    }

    #[test]
    fn ld_h_c() {
        analyze(LD, vec![M::H.into(), M::C.into()]).expect_fragments(vec![Fragment::Byte(0x61)])
    }

    #[test]
    fn ld_h_d() {
        analyze(LD, vec![M::H.into(), M::D.into()]).expect_fragments(vec![Fragment::Byte(0x62)])
    }

    #[test]
    fn ld_h_e() {
        analyze(LD, vec![M::H.into(), M::E.into()]).expect_fragments(vec![Fragment::Byte(0x63)])
    }

    #[test]
    fn ld_h_h() {
        analyze(LD, vec![M::H.into(), M::H.into()]).expect_fragments(vec![Fragment::Byte(0x64)])
    }

    #[test]
    fn ld_h_l() {
        analyze(LD, vec![M::H.into(), M::L.into()]).expect_fragments(vec![Fragment::Byte(0x65)])
    }

    #[test]
    fn ld_l_a() {
        analyze(LD, vec![M::L.into(), M::A.into()]).expect_fragments(vec![Fragment::Byte(0x6f)])
    }

    #[test]
    fn ld_l_b() {
        analyze(LD, vec![M::L.into(), M::B.into()]).expect_fragments(vec![Fragment::Byte(0x68)])
    }

    #[test]
    fn ld_l_c() {
        analyze(LD, vec![M::L.into(), M::C.into()]).expect_fragments(vec![Fragment::Byte(0x69)])
    }

    #[test]
    fn ld_l_d() {
        analyze(LD, vec![M::L.into(), M::D.into()]).expect_fragments(vec![Fragment::Byte(0x6a)])
    }

    #[test]
    fn ld_l_e() {
        analyze(LD, vec![M::L.into(), M::E.into()]).expect_fragments(vec![Fragment::Byte(0x6b)])
    }

    #[test]
    fn ld_l_h() {
        analyze(LD, vec![M::L.into(), M::H.into()]).expect_fragments(vec![Fragment::Byte(0x6c)])
    }

    #[test]
    fn ld_l_l() {
        analyze(LD, vec![M::L.into(), M::L.into()]).expect_fragments(vec![Fragment::Byte(0x6d)])
    }

    #[test]
    fn ld_a_n() {
        analyze(LD, vec![M::A.into(), 0x42.into()]).expect_fragments(vec![
            Fragment::Byte(0x3e),
            Fragment::Immediate(number(0x42, TokenId::Operand(1, 0)), Width::Byte),
        ])
    }

    #[test]
    fn ld_b_n() {
        analyze(LD, vec![M::B.into(), 0x42.into()]).expect_fragments(vec![
            Fragment::Byte(0x06),
            Fragment::Immediate(number(0x42, TokenId::Operand(1, 0)), Width::Byte),
        ])
    }

    #[test]
    fn ld_c_n() {
        analyze(LD, vec![M::C.into(), 0x42.into()]).expect_fragments(vec![
            Fragment::Byte(0x0e),
            Fragment::Immediate(number(0x42, TokenId::Operand(1, 0)), Width::Byte),
        ])
    }

    #[test]
    fn ld_d_n() {
        analyze(LD, vec![M::D.into(), 0x42.into()]).expect_fragments(vec![
            Fragment::Byte(0x16),
            Fragment::Immediate(number(0x42, TokenId::Operand(1, 0)), Width::Byte),
        ])
    }

    #[test]
    fn ld_e_n() {
        analyze(LD, vec![M::E.into(), 0x42.into()]).expect_fragments(vec![
            Fragment::Byte(0x1e),
            Fragment::Immediate(number(0x42, TokenId::Operand(1, 0)), Width::Byte),
        ])
    }

    #[test]
    fn ld_h_n() {
        analyze(LD, vec![M::H.into(), 0x42.into()]).expect_fragments(vec![
            Fragment::Byte(0x26),
            Fragment::Immediate(number(0x42, TokenId::Operand(1, 0)), Width::Byte),
        ])
    }

    #[test]
    fn ld_l_n() {
        analyze(LD, vec![M::L.into(), 0x42.into()]).expect_fragments(vec![
            Fragment::Byte(0x2e),
            Fragment::Immediate(number(0x42, TokenId::Operand(1, 0)), Width::Byte),
        ])
    }

    #[test]
    fn ld_a_deref_hl() {
        analyze(LD, vec![M::A.into(), M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0x7e)])
    }

    #[test]
    fn ld_b_deref_hl() {
        analyze(LD, vec![M::B.into(), M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0x46)])
    }

    #[test]
    fn ld_c_deref_hl() {
        analyze(LD, vec![M::C.into(), M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0x4e)])
    }

    #[test]
    fn ld_d_deref_hl() {
        analyze(LD, vec![M::D.into(), M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0x56)])
    }

    #[test]
    fn ld_e_deref_hl() {
        analyze(LD, vec![M::E.into(), M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0x5e)])
    }

    #[test]
    fn ld_h_deref_hl() {
        analyze(LD, vec![M::H.into(), M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0x66)])
    }

    #[test]
    fn ld_l_deref_hl() {
        analyze(LD, vec![M::L.into(), M::DerefHl.into()])
            .expect_fragments(vec![Fragment::Byte(0x6e)])
    }

    #[test]
    fn ld_deref_hl_a() {
        analyze(LD, vec![M::DerefHl.into(), M::A.into()])
            .expect_fragments(vec![Fragment::Byte(0x77)])
    }

    #[test]
    fn ld_deref_hl_b() {
        analyze(LD, vec![M::DerefHl.into(), M::B.into()])
            .expect_fragments(vec![Fragment::Byte(0x70)])
    }

    #[test]
    fn ld_deref_hl_c() {
        analyze(LD, vec![M::DerefHl.into(), M::C.into()])
            .expect_fragments(vec![Fragment::Byte(0x71)])
    }

    #[test]
    fn ld_deref_hl_d() {
        analyze(LD, vec![M::DerefHl.into(), M::D.into()])
            .expect_fragments(vec![Fragment::Byte(0x72)])
    }

    #[test]
    fn ld_deref_hl_e() {
        analyze(LD, vec![M::DerefHl.into(), M::E.into()])
            .expect_fragments(vec![Fragment::Byte(0x73)])
    }

    #[test]
    fn ld_deref_hl_h() {
        analyze(LD, vec![M::DerefHl.into(), M::H.into()])
            .expect_fragments(vec![Fragment::Byte(0x74)])
    }

    #[test]
    fn ld_deref_hl_l() {
        analyze(LD, vec![M::DerefHl.into(), M::L.into()])
            .expect_fragments(vec![Fragment::Byte(0x75)])
    }

    #[test]
    fn ld_deref_hl_n() {
        analyze(LD, vec![M::DerefHl.into(), 0x42.into()]).expect_fragments(vec![
            Fragment::Byte(0x36),
            Fragment::Immediate(number(0x42, TokenId::Operand(1, 0)), Width::Byte),
        ])
    }

    #[test]
    fn ld_a_deref_bc() {
        analyze(LD, vec![M::A.into(), deref_symbol(Bc)])
            .expect_fragments(vec![Fragment::Byte(0x0a)])
    }

    #[test]
    fn ld_a_deref_de() {
        analyze(LD, vec![M::A.into(), deref_symbol(De)])
            .expect_fragments(vec![Fragment::Byte(0x1a)])
    }

    #[test]
    fn ld_a_deref_c() {
        analyze(LD, vec![literal(A), deref_symbol(C)]).expect_fragments(vec![Fragment::Byte(0xf2)])
    }

    #[test]
    fn ld_deref_c_a() {
        analyze(LD, vec![deref_symbol(C), literal(A)]).expect_fragments(vec![Fragment::Byte(0xe2)])
    }

    #[test]
    fn ld_a_deref_expr() {
        analyze(LD, vec![literal(A), deref_ident(MockSymbolId(7))]).expect_fragments(vec![
            Fragment::LdInlineAddr(0xf0, name(MockSymbolId(7), TokenId::Operand(1, 1))),
        ])
    }

    #[test]
    fn ld_deref_expr_a() {
        analyze(LD, vec![deref_ident(MockSymbolId(7)), literal(A)]).expect_fragments(vec![
            Fragment::LdInlineAddr(0xe0, name(MockSymbolId(7), TokenId::Operand(0, 1))),
        ])
    }

    #[test]
    fn ld_a_deref_hli() {
        analyze(LD, vec![literal(A), deref_symbol(Hli)])
            .expect_fragments(vec![Fragment::Byte(0x2a)])
    }

    #[test]
    fn ld_a_deref_hld() {
        analyze(LD, vec![literal(A), deref_symbol(Hld)])
            .expect_fragments(vec![Fragment::Byte(0x3a)])
    }

    #[test]
    fn ld_deref_bc_a() {
        analyze(LD, vec![deref_symbol(Bc), literal(A)]).expect_fragments(vec![Fragment::Byte(0x02)])
    }

    #[test]
    fn ld_deref_de_a() {
        analyze(LD, vec![deref_symbol(De), literal(A)]).expect_fragments(vec![Fragment::Byte(0x12)])
    }

    #[test]
    fn ld_deref_hli_a() {
        analyze(LD, vec![deref_symbol(Hli), literal(A)])
            .expect_fragments(vec![Fragment::Byte(0x22)])
    }

    #[test]
    fn ld_deref_hld_a() {
        analyze(LD, vec![deref_symbol(Hld), literal(A)])
            .expect_fragments(vec![Fragment::Byte(0x32)])
    }

    #[test]
    fn ld_bc_nn() {
        analyze(LD, vec![literal(Bc), 0x1234.into()]).expect_fragments(vec![
            Fragment::Byte(0x01),
            Fragment::Immediate(number(0x1234, TokenId::Operand(1, 0)), Width::Word),
        ])
    }

    #[test]
    fn ld_de_nn() {
        analyze(LD, vec![literal(De), 0x1234.into()]).expect_fragments(vec![
            Fragment::Byte(0x11),
            Fragment::Immediate(number(0x1234, TokenId::Operand(1, 0)), Width::Word),
        ])
    }

    #[test]
    fn ld_hl_nn() {
        analyze(LD, vec![literal(Hl), 0x1234.into()]).expect_fragments(vec![
            Fragment::Byte(0x21),
            Fragment::Immediate(number(0x1234, TokenId::Operand(1, 0)), Width::Word),
        ])
    }

    #[test]
    fn ld_sp_nn() {
        analyze(LD, vec![literal(Sp), 0x1234.into()]).expect_fragments(vec![
            Fragment::Byte(0x31),
            Fragment::Immediate(number(0x1234, TokenId::Operand(1, 0)), Width::Word),
        ])
    }

    #[test]
    fn ld_sp_hl() {
        analyze(LD, vec![literal(Sp), literal(Hl)]).expect_fragments(vec![Fragment::Byte(0xf9)])
    }

    #[test]
    fn ld_deref_nn_sp() {
        analyze(LD, vec![deref_ident(MockSymbolId(0)), literal(Sp)]).expect_fragments(vec![
            Fragment::Byte(0x08),
            Fragment::Immediate(name(MockSymbolId(0), TokenId::Operand(0, 1)), Width::Word),
        ])
    }

    // Test errors

    #[test]
    fn analyze_ld() {
        analyze(LD, vec![]).expect_diag(
            ExpectedDiag::new(Message::OperandCount {
                actual: 0,
                expected: 2,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_ld_a() {
        analyze(LD, vec![literal(A)]).expect_diag(
            ExpectedDiag::new(Message::OperandCount {
                actual: 1,
                expected: 2,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_ld_const_const() {
        analyze(LD, vec![2.into(), 4.into()]).expect_diag(
            ExpectedDiag::new(Message::DestCannotBeConst).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_ld_a_bc() {
        analyze(LD, vec![literal(A), literal(Bc)]).expect_diag(
            ExpectedDiag::new(Message::LdWidthMismatch {
                src_width: Width::Word,
                src: TokenId::Operand(1, 0).into(),
                dest: TokenId::Operand(0, 0).into(),
            })
            .with_highlight(TokenSpan::merge(
                TokenId::Operand(0, 0),
                TokenId::Operand(1, 0),
            )),
        )
    }

    #[test]
    fn analyze_ld_bc_a() {
        analyze(LD, vec![literal(Bc), literal(A)]).expect_diag(
            ExpectedDiag::new(Message::LdWidthMismatch {
                src_width: Width::Byte,
                src: TokenId::Operand(1, 0).into(),
                dest: TokenId::Operand(0, 0).into(),
            })
            .with_highlight(TokenSpan::merge(
                TokenId::Operand(0, 0),
                TokenId::Operand(1, 0),
            )),
        )
    }

    #[test]
    fn analyze_ld_deref_c_b() {
        analyze(LD, vec![deref_symbol(C), literal(B)]).expect_diag(
            ExpectedDiag::new(Message::OnlySupportedByA).with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_ld_deref_c_4() {
        analyze(LD, vec![deref_symbol(C), 4.into()]).expect_diag(
            ExpectedDiag::new(Message::OnlySupportedByA).with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_ld_hl_sp() {
        analyze(LD, vec![literal(Hl), literal(Sp)]).expect_diag(
            ExpectedDiag::new(Message::LdSpHlOperands).with_highlight(TokenSpan::merge(
                TokenId::Operand(0, 0),
                TokenId::Operand(1, 0),
            )),
        )
    }

    #[test]
    fn analyze_ld_a_z() {
        analyze(LD, vec![literal(A), literal(Z)]).expect_diag(
            ExpectedDiag::new(Message::ConditionOutsideBranch)
                .with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_ld_sp_af() {
        analyze(LD, vec![literal(Sp), literal(Af)]).expect_diag(
            ExpectedDiag::new(Message::AfOutsideStackOperation)
                .with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_ld_deref_hl_deref_hl() {
        let src = TokenSpan::merge(TokenId::Operand(1, 0), TokenId::Operand(1, 2));
        analyze(LD, vec![deref_symbol(Hl), deref_symbol(Hl)]).expect_diag(
            ExpectedDiag::new(Message::LdDerefHlDerefHl {
                mnemonic: TokenId::Mnemonic.into(),
                dest: TokenSpan::merge(TokenId::Operand(0, 0), TokenId::Operand(0, 2)),
                src: src.clone(),
            })
            .with_highlight(TokenSpan::merge(TokenId::Mnemonic, src)),
        )
    }
}
