use super::{Analysis, Operand};

use crate::analyze::semantics::instr_line::builtin_instr::operand::AtomKind;
use crate::diag::span::{Source, SpanSource};
use crate::diag::{Diagnostics, EmitDiag, Message};
use crate::object::builder::*;

impl<'a, I, V, D, S> Analysis<'a, I, D, S>
where
    I: Iterator<Item = Result<Operand<V, S>, ()>>,
    V: Source<Span = S>,
    D: Diagnostics<S>,
    S: Clone,
{
    pub fn analyze_ld(&mut self) -> Result<CpuInstr<V>, ()> {
        let dest = self.next_operand_of(2)?;
        let src = self.next_operand_of(2)?;
        match (
            dest.into_ld_dest(self.diagnostics)?,
            src.into_ld_src(self.diagnostics)?,
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
        }
    }

    fn diagnose_ld_width_mismatch(
        &mut self,
        dest: &impl Source<Span = S>,
        src: &(impl Source<Span = S> + DataWidth),
    ) -> Result<CpuInstr<V>, ()> {
        let diagnostics = &mut self.diagnostics;
        let diagnostic = Message::LdWidthMismatch {
            src_width: src.width(),
            src: diagnostics.strip_span(&src.span()),
            dest: diagnostics.strip_span(&dest.span()),
        }
        .at(diagnostics.merge_spans(&dest.span(), &src.span()));
        diagnostics.emit_diag(diagnostic);
        Err(())
    }

    fn analyze_8_bit_ld(
        &mut self,
        dest: LdDest8<V>,
        src: impl Into<LdOperand<V, LdDest8<V>>>,
    ) -> Result<CpuInstr<V>, ()> {
        match (dest, src.into()) {
            (
                LdDest8::Simple(SimpleOperand::DerefHl, dest),
                LdOperand::Other(LdDest8::Simple(SimpleOperand::DerefHl, src)),
            ) => {
                let diagnostics = &mut self.diagnostics;
                let diagnostic = Message::LdDerefHlDerefHl {
                    mnemonic: diagnostics.strip_span(&self.mnemonic.1),
                    dest: diagnostics.strip_span(&dest),
                    src: diagnostics.strip_span(&src),
                }
                .at(diagnostics.merge_spans(&self.mnemonic.1, &src));
                self.emit_diag(diagnostic);
                Err(())
            }
            (LdDest8::Simple(dest, _), LdOperand::Other(LdDest8::Simple(src, _))) => {
                Ok(CpuInstr::Ld(Ld::Simple(dest, src)))
            }
            (LdDest8::Simple(dest, _), LdOperand::Const(expr)) => {
                Ok(CpuInstr::Ld(Ld::Immediate8(dest, expr)))
            }
            (LdDest8::Special(dest), src) => {
                src.expect_a(self.diagnostics)?;
                analyze_special_ld(dest, Direction::FromA)
            }
            (dest, LdOperand::Other(LdDest8::Special(src))) => {
                dest.expect_a(self.diagnostics)?;
                analyze_special_ld(src, Direction::IntoA)
            }
        }
    }

    fn analyze_16_bit_ld(
        &mut self,
        dest: LdDest16<S>,
        src: impl Into<LdOperand<V, LdDest16<S>>>,
    ) -> Result<CpuInstr<V>, ()> {
        match (dest, src.into()) {
            (LdDest16::Reg16(Reg16::Sp, _), LdOperand::Other(LdDest16::Reg16(Reg16::Hl, _))) => {
                Ok(CpuInstr::Ld(Ld::SpHl))
            }
            (LdDest16::Reg16(_, dest_span), LdOperand::Other(LdDest16::Reg16(_, src_span))) => {
                let diagnostics = &mut self.diagnostics;
                let merged_span = diagnostics.merge_spans(&dest_span, &src_span);
                diagnostics.emit_diag(Message::LdSpHlOperands.at(merged_span));
                Err(())
            }
            (LdDest16::Reg16(dest, _), LdOperand::Const(expr)) => {
                Ok(CpuInstr::Ld(Ld::Immediate16(dest, expr)))
            }
        }
    }
}

fn analyze_special_ld<V: Source>(
    other: LdSpecial<V>,
    direction: Direction,
) -> Result<CpuInstr<V>, ()> {
    Ok(CpuInstr::Ld(Ld::Special(
        match other {
            LdSpecial::Deref(expr) => SpecialLd::InlineAddr(expr),
            LdSpecial::DerefC(_) => SpecialLd::RegIndex,
            LdSpecial::DerefPtrReg(ptr_reg, _) => SpecialLd::DerefPtrReg(ptr_reg),
        },
        direction,
    )))
}

impl<V: Source> Operand<V, V::Span> {
    fn into_ld_dest<D>(self, diagnostics: &mut D) -> Result<LdDest<V>, ()>
    where
        D: Diagnostics<V::Span>,
    {
        match self {
            Operand::Deref(expr) => Ok(LdDest::Byte(LdDest8::Special(LdSpecial::Deref(expr)))),
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

    fn into_ld_src<D>(self, diagnostics: &mut D) -> Result<LdOperand<V, LdDest<V>>, ()>
    where
        D: Diagnostics<V::Span>,
    {
        match self {
            Operand::Const(expr) => Ok(LdOperand::Const(expr)),
            operand => Ok(LdOperand::Other(operand.into_ld_dest(diagnostics)?)),
        }
    }
}

enum LdOperand<V, T> {
    Const(V),
    Other(T),
}

impl<V: Source> From<LdDest8<V>> for LdOperand<V, LdDest8<V>> {
    fn from(dest: LdDest8<V>) -> Self {
        LdOperand::Other(dest)
    }
}

impl<V, S> From<LdDest16<S>> for LdOperand<V, LdDest16<S>> {
    fn from(dest: LdDest16<S>) -> Self {
        LdOperand::Other(dest)
    }
}

enum LdDest<V: SpanSource> {
    Byte(LdDest8<V>),
    Word(LdDest16<V::Span>),
}

enum LdDest8<V: SpanSource> {
    Simple(SimpleOperand, V::Span),
    Special(LdSpecial<V>),
}

enum LdSpecial<V: SpanSource> {
    Deref(V),
    DerefC(V::Span),
    DerefPtrReg(PtrReg, V::Span),
}

enum LdDest16<S> {
    Reg16(Reg16, S),
}

trait DataWidth {
    fn width(&self) -> Width;
}

impl<V: Source> DataWidth for LdDest8<V> {
    fn width(&self) -> Width {
        Width::Byte
    }
}

impl<S> DataWidth for LdDest16<S> {
    fn width(&self) -> Width {
        Width::Word
    }
}

impl<V: Source> LdOperand<V, LdDest8<V>> {
    fn expect_a<D>(self, diagnostics: &mut D) -> Result<(), ()>
    where
        D: Diagnostics<V::Span>,
    {
        match self {
            LdOperand::Const(expr) => diagnose_not_a(expr.span(), diagnostics),
            LdOperand::Other(other) => other.expect_a(diagnostics),
        }
    }
}

impl<V: Source> LdDest8<V> {
    fn expect_a<D>(self, diagnostics: &mut D) -> Result<(), ()>
    where
        D: Diagnostics<V::Span>,
    {
        match self {
            LdDest8::Simple(SimpleOperand::A, _) => Ok(()),
            operand => diagnose_not_a(operand.span(), diagnostics),
        }
    }
}

fn diagnose_not_a<T, D: EmitDiag<S, T>, S>(span: S, diagnostics: &mut D) -> Result<(), ()> {
    diagnostics.emit_diag(Message::OnlySupportedByA.at(span));
    Err(())
}

impl<V: Source, T: Source<Span = V::Span>> SpanSource for LdOperand<V, T> {
    type Span = V::Span;
}

impl<V: Source, T: Source<Span = V::Span>> Source for LdOperand<V, T> {
    fn span(&self) -> Self::Span {
        match self {
            LdOperand::Const(expr) => expr.span(),
            LdOperand::Other(operand) => operand.span(),
        }
    }
}

impl<V: SpanSource> SpanSource for LdDest8<V> {
    type Span = V::Span;
}

impl<V: Source> Source for LdDest8<V> {
    fn span(&self) -> Self::Span {
        use self::LdDest8::*;
        match self {
            Simple(_, span) => span.clone(),
            Special(special) => special.span(),
        }
    }
}

impl<V: SpanSource> SpanSource for LdSpecial<V> {
    type Span = V::Span;
}

impl<V: Source> Source for LdSpecial<V> {
    fn span(&self) -> Self::Span {
        use self::LdSpecial::*;
        match self {
            Deref(expr) => expr.span(),
            DerefC(span) | DerefPtrReg(_, span) => span.clone(),
        }
    }
}

impl<S: Clone> SpanSource for LdDest16<S> {
    type Span = S;
}

impl<S: Clone> Source for LdDest16<S> {
    fn span(&self) -> Self::Span {
        match self {
            LdDest16::Reg16(_, span) => span.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analyze::semantics::instr_line::builtin_instr::cpu_instr::mnemonic::LD;
    use crate::analyze::semantics::instr_line::builtin_instr::cpu_instr::tests::*;
    use crate::diag::Merge;

    #[test]
    fn analyze_ld_deref_symbol_a() {
        let ident = "ident";
        analyze(LD, vec![deref(ident), literal(A)]).expect_instruction(CpuInstr::Ld(Ld::Special(
            SpecialLd::InlineAddr(name(ident, TokenId::Operand(0, 1))),
            Direction::FromA,
        )))
    }

    #[test]
    fn analyze_ld_a_deref_symbol() {
        let ident = "ident";
        analyze(LD, vec![literal(A), deref(ident)]).expect_instruction(CpuInstr::Ld(Ld::Special(
            SpecialLd::InlineAddr(name(ident, TokenId::Operand(1, 1))),
            Direction::IntoA,
        )))
    }

    #[test]
    fn analyze_ld_deref_c_a() {
        analyze(LD, vec![deref(literal(C)), literal(A)]).expect_instruction(CpuInstr::Ld(
            Ld::Special(SpecialLd::RegIndex, Direction::FromA),
        ))
    }

    #[test]
    fn analyze_ld_a_deref_c() {
        analyze(LD, vec![literal(A), deref(literal(C))]).expect_instruction(CpuInstr::Ld(
            Ld::Special(SpecialLd::RegIndex, Direction::IntoA),
        ))
    }

    #[test]
    fn analyze_legal_ld_instructions() {
        test_instruction_analysis(describe_legal_ld_instructions());
    }

    fn describe_legal_ld_instructions() -> Vec<InstructionDescriptor> {
        let mut descriptors: Vec<InstructionDescriptor> = Vec::new();
        descriptors.extend(describe_ld_simple_simple_instructions());
        descriptors.extend(describe_ld_simple_immediate_instructions());
        descriptors.extend(describe_ld_reg16_immediate_instructions());
        descriptors.extend(describe_ld_deref_reg16_instructions());
        descriptors.push((
            (LD, vec![Reg16::Sp.into(), Reg16::Hl.into()]),
            CpuInstr::Ld(Ld::SpHl),
        ));
        descriptors
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
                (LD, vec![dest.into(), src.into()]),
                CpuInstr::Ld(Ld::Simple(dest, src)),
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
            (LD, vec![dest.into(), n.into()]),
            CpuInstr::Ld(Ld::Immediate8(dest, number(n, TokenId::Operand(1, 0)))),
        )
    }

    fn describe_ld_reg16_immediate_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        REG16.iter().map(|&dest| describe_ld_reg16_immediate(dest))
    }

    fn describe_ld_reg16_immediate(dest: Reg16) -> InstructionDescriptor {
        let value = "value";
        (
            (LD, vec![dest.into(), value.into()]),
            CpuInstr::Ld(Ld::Immediate16(dest, name(value, TokenId::Operand(1, 0)))),
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
                (LD, vec![deref(ptr_reg), literal(A)]),
                CpuInstr::Ld(Ld::Special(
                    SpecialLd::DerefPtrReg(ptr_reg),
                    Direction::FromA,
                )),
            ),
            (
                (LD, vec![literal(A), deref(ptr_reg)]),
                CpuInstr::Ld(Ld::Special(
                    SpecialLd::DerefPtrReg(ptr_reg),
                    Direction::IntoA,
                )),
            ),
        ]
        .into_iter()
    }

    const PTR_REGS: &[PtrReg] = &[PtrReg::Bc, PtrReg::De, PtrReg::Hli, PtrReg::Hld];

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
        analyze(LD, vec![deref(literal(C)), literal(B)]).expect_diag(
            ExpectedDiag::new(Message::OnlySupportedByA).with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_ld_deref_c_4() {
        analyze(LD, vec![deref(literal(C)), 4.into()]).expect_diag(
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
        analyze(LD, vec![deref(literal(Hl)), deref(literal(Hl))]).expect_diag(
            ExpectedDiag::new(Message::LdDerefHlDerefHl {
                mnemonic: TokenId::Mnemonic.into(),
                dest: TokenSpan::merge(TokenId::Operand(0, 0), TokenId::Operand(0, 2)),
                src: src.clone(),
            })
            .with_highlight(TokenSpan::merge(TokenId::Mnemonic, src)),
        )
    }
}
