use super::{Analysis, AnalysisResult, Operand};
use crate::backend::Width;
use crate::diagnostics::{CompactDiagnostic, Message};
use crate::frontend::semantics::operand::AtomKind;
use crate::instruction::{Direction, Instruction, Ld, PtrReg, Reg16, SimpleOperand, SpecialLd};
use crate::span::{MergeSpans, Source, Span};
use std::fmt::Debug;

impl<'a, I, V, M> Analysis<'a, I, M>
where
    I: Iterator<Item = Result<Operand<V>, CompactDiagnostic<M::Span>>>,
    V: Source<Span = M::Span>,
    M: MergeSpans,
{
    pub fn analyze_ld(&mut self) -> AnalysisResult<V> {
        let dest = self.next_operand_out_of(2)?;
        let src = self.next_operand_out_of(2)?;
        match (dest.into_ld_dest()?, src.into_ld_src()?) {
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
                Err(CompactDiagnostic::new(
                    Message::LdWidthMismatch {
                        src_width: Width::Word,
                        src: src.span(),
                        dest: dest.span(),
                    },
                    self.spans.merge_spans(&dest.span(), &src.span()),
                ))
            }
            (LdDest::Word(dest), LdOperand::Other(LdDest::Byte(src))) => {
                Err(CompactDiagnostic::new(
                    Message::LdWidthMismatch {
                        src_width: Width::Byte,
                        src: src.span(),
                        dest: dest.span(),
                    },
                    self.spans.merge_spans(&dest.span(), &src.span()),
                ))
            }
        }
    }

    fn analyze_8_bit_ld(
        &mut self,
        dest: LdDest8<V>,
        src: impl Into<LdOperand<V, LdDest8<V>>>,
    ) -> AnalysisResult<V> {
        match (dest, src.into()) {
            (
                LdDest8::Simple(SimpleOperand::DerefHl, dest),
                LdOperand::Other(LdDest8::Simple(SimpleOperand::DerefHl, src)),
            ) => Err(CompactDiagnostic::new(
                Message::LdDerefHlDerefHl {
                    mnemonic: self.mnemonic.1.clone(),
                    dest,
                    src: src.clone(),
                },
                self.spans.merge_spans(&self.mnemonic.1, &src),
            )),
            (LdDest8::Simple(dest, _), LdOperand::Other(LdDest8::Simple(src, _))) => {
                Ok(Instruction::Ld(Ld::Simple(dest, src)))
            }
            (LdDest8::Simple(dest, _), LdOperand::Const(expr)) => {
                Ok(Instruction::Ld(Ld::Immediate8(dest, expr)))
            }
            (LdDest8::Special(dest), src) => {
                src.expect_a()?;
                analyze_special_ld(dest, Direction::FromA)
            }
            (dest, LdOperand::Other(LdDest8::Special(src))) => {
                dest.expect_a()?;
                analyze_special_ld(src, Direction::IntoA)
            }
        }
    }

    fn analyze_16_bit_ld(
        &mut self,
        dest: LdDest16<M::Span>,
        src: impl Into<LdOperand<V, LdDest16<M::Span>>>,
    ) -> AnalysisResult<V> {
        match (dest, src.into()) {
            (LdDest16::Reg16(Reg16::Sp, _), LdOperand::Other(LdDest16::Reg16(Reg16::Hl, _))) => {
                Ok(Instruction::Ld(Ld::SpHl))
            }
            (LdDest16::Reg16(_, dest_span), LdOperand::Other(LdDest16::Reg16(_, src_span))) => {
                Err(CompactDiagnostic::new(
                    Message::LdSpHlOperands,
                    self.spans.merge_spans(&dest_span, &src_span),
                ))
            }
            (LdDest16::Reg16(dest, _), LdOperand::Const(expr)) => {
                Ok(Instruction::Ld(Ld::Immediate16(dest, expr)))
            }
        }
    }
}

fn analyze_special_ld<V: Source>(other: LdSpecial<V>, direction: Direction) -> AnalysisResult<V> {
    Ok(Instruction::Ld(Ld::Special(
        match other {
            LdSpecial::Deref(expr) => SpecialLd::InlineAddr(expr),
            LdSpecial::DerefC(_) => SpecialLd::RegIndex,
            LdSpecial::DerefPtrReg(ptr_reg, _) => SpecialLd::DerefPtrReg(ptr_reg),
        },
        direction,
    )))
}

impl<V: Source> Operand<V> {
    fn into_ld_dest(self) -> Result<LdDest<V>, CompactDiagnostic<V::Span>> {
        match self {
            Operand::Deref(expr) => Ok(LdDest::Byte(LdDest8::Special(LdSpecial::Deref(expr)))),
            Operand::Atom(kind, span) => match kind {
                AtomKind::Condition(_) => Err(CompactDiagnostic::new(
                    Message::ConditionOutsideBranch,
                    span,
                )),
                AtomKind::Simple(simple) => Ok(LdDest::Byte(LdDest8::Simple(simple, span))),
                AtomKind::DerefC => Ok(LdDest::Byte(LdDest8::Special(LdSpecial::DerefC(span)))),
                AtomKind::DerefPtrReg(ptr_reg) => Ok(LdDest::Byte(LdDest8::Special(
                    LdSpecial::DerefPtrReg(ptr_reg, span),
                ))),
                AtomKind::Reg16(reg16) => Ok(LdDest::Word(LdDest16::Reg16(reg16, span))),
                AtomKind::RegPair(reg_pair) => {
                    use crate::instruction::RegPair;
                    assert_eq!(reg_pair, RegPair::Af);
                    Err(CompactDiagnostic::new(
                        Message::AfOutsideStackOperation,
                        span,
                    ))
                }
            },
            Operand::Const(expr) => Err(CompactDiagnostic::new(
                Message::DestCannotBeConst,
                expr.span(),
            )),
        }
    }

    fn into_ld_src(self) -> Result<LdOperand<V, LdDest<V>>, CompactDiagnostic<V::Span>> {
        match self {
            Operand::Const(expr) => Ok(LdOperand::Const(expr)),
            operand => Ok(LdOperand::Other(operand.into_ld_dest()?)),
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

enum LdDest<V: Source> {
    Byte(LdDest8<V>),
    Word(LdDest16<V::Span>),
}

enum LdDest8<V: Source> {
    Simple(SimpleOperand, V::Span),
    Special(LdSpecial<V>),
}

enum LdSpecial<V: Source> {
    Deref(V),
    DerefC(V::Span),
    DerefPtrReg(PtrReg, V::Span),
}

enum LdDest16<S> {
    Reg16(Reg16, S),
}

impl<V: Source> LdOperand<V, LdDest8<V>> {
    fn expect_a(self) -> Result<(), CompactDiagnostic<V::Span>> {
        match self {
            LdOperand::Const(expr) => Err(diagnose_not_a(expr.span())),
            LdOperand::Other(other) => other.expect_a(),
        }
    }
}

impl<V: Source> LdDest8<V> {
    fn expect_a(self) -> Result<(), CompactDiagnostic<V::Span>> {
        match self {
            LdDest8::Simple(SimpleOperand::A, _) => Ok(()),
            operand => Err(diagnose_not_a(operand.span())),
        }
    }
}

fn diagnose_not_a<S>(span: S) -> CompactDiagnostic<S> {
    CompactDiagnostic::new(Message::OnlySupportedByA, span)
}

impl<V: Span, T: Span<Span = V::Span>> Span for LdOperand<V, T> {
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

impl<V: Source> Span for LdDest8<V> {
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

impl<V: Source> Span for LdSpecial<V> {
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

impl<S: Clone + Debug + PartialEq> Span for LdDest16<S> {
    type Span = S;
}

impl<S: Clone + Debug + PartialEq> Source for LdDest16<S> {
    fn span(&self) -> Self::Span {
        match self {
            LdDest16::Reg16(_, span) => span.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::Width;
    use crate::expr::Expr;
    use crate::frontend::semantics::instruction::tests::*;
    use crate::frontend::syntax::keyword::Mnemonic;
    use crate::instruction::{Direction, Instruction, Ld, PtrReg, Reg16, SimpleOperand, SpecialLd};

    #[test]
    fn analyze_ld_deref_symbol_a() {
        let ident = "ident";
        analyze(Mnemonic::Ld, vec![deref(ident.into()), literal(A)]).expect_instruction(
            Instruction::Ld(Ld::Special(
                SpecialLd::InlineAddr(symbol(ident, TokenId::Operand(0, 1))),
                Direction::FromA,
            )),
        )
    }

    #[test]
    fn analyze_ld_a_deref_symbol() {
        let ident = "ident";
        analyze(Mnemonic::Ld, vec![literal(A), deref(ident.into())]).expect_instruction(
            Instruction::Ld(Ld::Special(
                SpecialLd::InlineAddr(symbol(ident, TokenId::Operand(1, 1))),
                Direction::IntoA,
            )),
        )
    }

    #[test]
    fn analyze_ld_deref_c_a() {
        analyze(Mnemonic::Ld, vec![deref(literal(C)), literal(A)]).expect_instruction(
            Instruction::Ld(Ld::Special(SpecialLd::RegIndex, Direction::FromA)),
        )
    }

    #[test]
    fn analyze_ld_a_deref_c() {
        analyze(Mnemonic::Ld, vec![literal(A), deref(literal(C))]).expect_instruction(
            Instruction::Ld(Ld::Special(SpecialLd::RegIndex, Direction::IntoA)),
        )
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
            (Mnemonic::Ld, vec![Reg16::Sp.into(), Reg16::Hl.into()]),
            Instruction::Ld(Ld::SpHl),
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
                (Mnemonic::Ld, vec![dest.into(), src.into()]),
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
            (Mnemonic::Ld, vec![Expr::from(dest), n.into()]),
            Instruction::Ld(Ld::Immediate8(dest, number(n, TokenId::Operand(1, 0)))),
        )
    }

    fn describe_ld_reg16_immediate_instructions() -> impl Iterator<Item = InstructionDescriptor> {
        REG16.iter().map(|&dest| describe_ld_reg16_immediate(dest))
    }

    fn describe_ld_reg16_immediate(dest: Reg16) -> InstructionDescriptor {
        let value = "value";
        (
            (Mnemonic::Ld, vec![Expr::from(dest), value.into()]),
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
                (Mnemonic::Ld, vec![deref(Expr::from(ptr_reg)), literal(A)]),
                Instruction::Ld(Ld::Special(
                    SpecialLd::DerefPtrReg(ptr_reg),
                    Direction::FromA,
                )),
            ),
            (
                (Mnemonic::Ld, vec![literal(A), deref(Expr::from(ptr_reg))]),
                Instruction::Ld(Ld::Special(
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
        analyze(Mnemonic::Ld, vec![]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 0,
                expected: 2,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_ld_a() {
        analyze(Mnemonic::Ld, vec![literal(A)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OperandCount {
                actual: 1,
                expected: 2,
            })
            .with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_ld_const_const() {
        analyze(Mnemonic::Ld, vec![2.into(), 4.into()]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::DestCannotBeConst)
                .with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_ld_a_bc() {
        analyze(Mnemonic::Ld, vec![literal(A), literal(Bc)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::LdWidthMismatch {
                src_width: Width::Word,
                src: TokenId::Operand(1, 0).into(),
                dest: TokenId::Operand(0, 0).into(),
            })
            .with_highlight(TokenSpan::merge(
                &TokenSpan::from(TokenId::Operand(0, 0)),
                &TokenId::Operand(1, 0).into(),
            )),
        )
    }

    #[test]
    fn analyze_ld_bc_a() {
        analyze(Mnemonic::Ld, vec![literal(Bc), literal(A)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::LdWidthMismatch {
                src_width: Width::Byte,
                src: TokenId::Operand(1, 0).into(),
                dest: TokenId::Operand(0, 0).into(),
            })
            .with_highlight(TokenSpan::merge(
                &TokenSpan::from(TokenId::Operand(0, 0)),
                &TokenId::Operand(1, 0).into(),
            )),
        )
    }

    #[test]
    fn analyze_ld_deref_c_b() {
        analyze(Mnemonic::Ld, vec![deref(literal(C)), literal(B)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OnlySupportedByA)
                .with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_ld_deref_c_4() {
        analyze(Mnemonic::Ld, vec![deref(literal(C)), 4.into()]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::OnlySupportedByA)
                .with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_ld_hl_sp() {
        analyze(Mnemonic::Ld, vec![literal(Hl), literal(Sp)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::LdSpHlOperands).with_highlight(TokenSpan::merge(
                &TokenSpan::from(TokenId::Operand(0, 0)),
                &TokenId::Operand(1, 0).into(),
            )),
        )
    }

    #[test]
    fn analyze_ld_a_z() {
        analyze(Mnemonic::Ld, vec![literal(A), literal(Z)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::ConditionOutsideBranch)
                .with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_ld_sp_af() {
        analyze(Mnemonic::Ld, vec![literal(Sp), literal(Af)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::AfOutsideStackOperation)
                .with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_ld_deref_hl_deref_hl() {
        analyze(Mnemonic::Ld, vec![deref(literal(Hl)), deref(literal(Hl))]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::LdDerefHlDerefHl {
                mnemonic: TokenId::Mnemonic.into(),
                dest: TokenSpan::merge(
                    &TokenSpan::from(TokenId::Operand(0, 0)),
                    &TokenId::Operand(0, 2).into(),
                ),
                src: TokenSpan::merge(
                    &TokenSpan::from(TokenId::Operand(1, 0)),
                    &TokenId::Operand(1, 2).into(),
                ),
            })
            .with_highlight(TokenSpan::merge(
                &TokenSpan::from(TokenId::Mnemonic),
                &TokenId::Operand(1, 2).into(),
            )),
        )
    }
}
