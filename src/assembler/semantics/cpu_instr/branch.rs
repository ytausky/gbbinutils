use super::{Analysis, AtomKind, Condition, Expr, Fragment, Operand, Width, M};

use crate::assembler::keywords::{BranchKind, ExplicitBranch, ImplicitBranch};
use crate::assembler::session::builder::Backend;
use crate::diagnostics::{Diagnostics, EmitDiag, Message};
use crate::expr::{Atom, BinOp, ExprOp};
use crate::span::WithSpan;
use crate::span::{Source, SpanSource};

impl<'a, 'b, I, D, S> Analysis<'a, 'b, I, D, S>
where
    I: Iterator<Item = Result<Operand<D::SymbolId, S>, ()>>,
    D: Backend<S> + Diagnostics<S>,
    S: Clone,
{
    pub fn analyze_branch(&mut self, branch: BranchKind) -> Result<(), ()> {
        let (condition, target) = self.collect_branch_operands()?;
        let variant = analyze_branch_variant((branch, &self.mnemonic.1), target, self.session)?;
        match variant {
            BranchVariant::Unconditional(branch) => match condition {
                None => {
                    self.session.emit_fragment(Fragment::Byte(match branch {
                        UnconditionalBranch::JpDerefHl => 0xe9,
                        UnconditionalBranch::Reti => 0xd9,
                    }));
                    Ok(())
                }
                Some((_, condition_span)) => {
                    self.emit_diag(Message::AlwaysUnconditional.at(condition_span));
                    Err(())
                }
            },
            BranchVariant::PotentiallyConditional(branch) => {
                match branch {
                    Branch::Explicit(ExplicitBranch::Jp, target) => {
                        self.session.emit_fragment(Fragment::Byte(match condition {
                            None => 0xc3,
                            Some((condition, _)) => 0xc2 | encode_condition(condition),
                        }));
                        self.session
                            .emit_fragment(Fragment::Immediate(target, Width::Word))
                    }
                    Branch::Explicit(ExplicitBranch::Jr, mut target) => {
                        self.session.emit_fragment(Fragment::Byte(match condition {
                            None => 0x18,
                            Some((condition, _)) => 0x20 | encode_condition(condition),
                        }));
                        let span = target.span();
                        target
                            .0
                            .push(ExprOp::Atom(Atom::Location).with_span(span.clone()));
                        target.0.push(ExprOp::Binary(BinOp::Minus).with_span(span));
                        self.session
                            .emit_fragment(Fragment::Immediate(target, Width::Byte))
                    }
                    Branch::Explicit(ExplicitBranch::Call, target) => {
                        self.session.emit_fragment(Fragment::Byte(match condition {
                            None => 0xcd,
                            Some((condition, _)) => 0xc4 | encode_condition(condition),
                        }));
                        self.session
                            .emit_fragment(Fragment::Immediate(target, Width::Word))
                    }
                    Branch::Ret => self.session.emit_fragment(Fragment::Byte(match condition {
                        None => 0xc9,
                        Some((condition, _)) => 0xc0 | encode_condition(condition),
                    })),
                }
                Ok(())
            }
        }
    }

    fn collect_branch_operands(&mut self) -> Result<BranchOperands<D::SymbolId, S>, ()> {
        let first_operand = self.next_operand()?;
        Ok(
            if let Some(Operand::Atom(AtomKind::Condition(condition), range)) = first_operand {
                (
                    Some((condition, range)),
                    analyze_branch_target(self.next_operand()?, self.session)?,
                )
            } else {
                (None, analyze_branch_target(first_operand, self.session)?)
            },
        )
    }
}

fn encode_condition(condition: Condition) -> u8 {
    use self::Condition::*;
    (match condition {
        Nz => 0b00,
        Z => 0b01,
        Nc => 0b10,
        C => 0b11,
    }) << 3
}

type BranchOperands<N, S> = (Option<(Condition, S)>, Option<BranchTarget<N, S>>);

enum BranchTarget<N, S> {
    DerefHl(S),
    Expr(Expr<N, S>),
}

impl<N, S: Clone> SpanSource for BranchTarget<N, S> {
    type Span = S;
}

impl<N, S: Clone> Source for BranchTarget<N, S> {
    fn span(&self) -> Self::Span {
        match self {
            BranchTarget::DerefHl(span) => span.clone(),
            BranchTarget::Expr(expr) => expr.span(),
        }
    }
}

fn analyze_branch_target<N, D, S>(
    target: Option<Operand<N, S>>,
    diagnostics: &mut D,
) -> Result<Option<BranchTarget<N, S>>, ()>
where
    D: Diagnostics<S>,
    S: Clone,
{
    let target = match target {
        Some(target) => target,
        None => return Ok(None),
    };
    match target {
        Operand::Const(expr) => Ok(Some(BranchTarget::Expr(expr))),
        Operand::Atom(AtomKind::Simple(M::DerefHl), span) => Ok(Some(BranchTarget::DerefHl(span))),
        operand => {
            diagnostics.emit_diag(Message::CannotBeUsedAsTarget.at(operand.span()));
            Err(())
        }
    }
}

enum BranchVariant<N, S> {
    PotentiallyConditional(Branch<N, S>),
    Unconditional(UnconditionalBranch),
}

#[derive(Clone, Debug, PartialEq)]
enum Branch<N, S> {
    Explicit(ExplicitBranch, Expr<N, S>),
    Ret,
}

enum UnconditionalBranch {
    JpDerefHl,
    Reti,
}

fn analyze_branch_variant<N, D, S>(
    kind: (BranchKind, &S),
    target: Option<BranchTarget<N, S>>,
    diagnostics: &mut D,
) -> Result<BranchVariant<N, S>, ()>
where
    D: Diagnostics<S>,
    S: Clone,
{
    match (kind.0, target) {
        (BranchKind::Explicit(ExplicitBranch::Jp), Some(BranchTarget::DerefHl(_))) => {
            Ok(BranchVariant::Unconditional(UnconditionalBranch::JpDerefHl))
        }
        (BranchKind::Explicit(_), Some(BranchTarget::DerefHl(span))) => {
            Err(Message::RequiresConstantTarget {
                mnemonic: diagnostics.strip_span(&kind.1),
            }
            .at(span))
        }
        (BranchKind::Explicit(branch), Some(BranchTarget::Expr(expr))) => Ok(
            BranchVariant::PotentiallyConditional(Branch::Explicit(branch, expr)),
        ),
        (BranchKind::Implicit(ImplicitBranch::Ret), None) => {
            Ok(BranchVariant::PotentiallyConditional(Branch::Ret))
        }
        (BranchKind::Implicit(ImplicitBranch::Reti), None) => {
            Ok(BranchVariant::Unconditional(UnconditionalBranch::Reti))
        }
        (BranchKind::Explicit(_), None) => Err(Message::MissingTarget.at(kind.1.clone())),
        (BranchKind::Implicit(_), Some(target)) => {
            Err(Message::CannotSpecifyTarget.at(target.span()))
        }
    }
    .map_err(|diagnostic| {
        diagnostics.emit_diag(diagnostic);
    })
}

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;

    use crate::assembler::keywords::*;
    use crate::diagnostics::Merge;
    use crate::expr::{Atom, BinOp, Expr, ExprOp};

    #[test]
    fn jp_nn() {
        let nn = MockSymbolId(7);
        analyze(JP, vec![nn.into()]).expect_fragments(vec![
            Fragment::Byte(0xc3),
            Fragment::Immediate(name(nn, TokenId::Operand(0, 0)), Width::Word),
        ])
    }

    #[test]
    fn jp_nz_nn() {
        let nn = MockSymbolId(7);
        analyze(JP, vec![Condition::Nz.into(), nn.into()]).expect_fragments(vec![
            Fragment::Byte(0xc2),
            Fragment::Immediate(name(nn, TokenId::Operand(1, 0)), Width::Word),
        ])
    }

    #[test]
    fn jp_z_nn() {
        let nn = MockSymbolId(7);
        analyze(JP, vec![Condition::Z.into(), nn.into()]).expect_fragments(vec![
            Fragment::Byte(0xca),
            Fragment::Immediate(name(nn, TokenId::Operand(1, 0)), Width::Word),
        ])
    }

    #[test]
    fn jp_nc_nn() {
        let nn = MockSymbolId(7);
        analyze(JP, vec![Condition::Nc.into(), nn.into()]).expect_fragments(vec![
            Fragment::Byte(0xd2),
            Fragment::Immediate(name(nn, TokenId::Operand(1, 0)), Width::Word),
        ])
    }

    #[test]
    fn jp_c_nn() {
        let nn = MockSymbolId(7);
        analyze(JP, vec![Condition::C.into(), nn.into()]).expect_fragments(vec![
            Fragment::Byte(0xda),
            Fragment::Immediate(name(nn, TokenId::Operand(1, 0)), Width::Word),
        ])
    }

    #[test]
    fn jr_e() {
        let nn = MockSymbolId(7);
        analyze(JR, vec![nn.into()]).expect_fragments(vec![
            Fragment::Byte(0x18),
            Fragment::Immediate(
                Expr(vec![
                    ExprOp::Atom(Atom::Name(nn)).with_span(TokenId::Operand(0, 0).into()),
                    ExprOp::Atom(Atom::Location).with_span(TokenId::Operand(0, 0).into()),
                    ExprOp::Binary(BinOp::Minus).with_span(TokenId::Operand(0, 0).into()),
                ]),
                Width::Byte,
            ),
        ])
    }

    #[test]
    fn jr_nz_e() {
        let nn = MockSymbolId(7);
        analyze(JR, vec![Condition::Nz.into(), nn.into()]).expect_fragments(vec![
            Fragment::Byte(0x20),
            Fragment::Immediate(
                Expr(vec![
                    ExprOp::Atom(Atom::Name(nn)).with_span(TokenId::Operand(1, 0).into()),
                    ExprOp::Atom(Atom::Location).with_span(TokenId::Operand(1, 0).into()),
                    ExprOp::Binary(BinOp::Minus).with_span(TokenId::Operand(1, 0).into()),
                ]),
                Width::Byte,
            ),
        ])
    }

    #[test]
    fn jr_z_e() {
        let nn = MockSymbolId(7);
        analyze(JR, vec![Condition::Z.into(), nn.into()]).expect_fragments(vec![
            Fragment::Byte(0x28),
            Fragment::Immediate(
                Expr(vec![
                    ExprOp::Atom(Atom::Name(nn)).with_span(TokenId::Operand(1, 0).into()),
                    ExprOp::Atom(Atom::Location).with_span(TokenId::Operand(1, 0).into()),
                    ExprOp::Binary(BinOp::Minus).with_span(TokenId::Operand(1, 0).into()),
                ]),
                Width::Byte,
            ),
        ])
    }

    #[test]
    fn jr_nc_e() {
        let nn = MockSymbolId(7);
        analyze(JR, vec![Condition::Nc.into(), nn.into()]).expect_fragments(vec![
            Fragment::Byte(0x30),
            Fragment::Immediate(
                Expr(vec![
                    ExprOp::Atom(Atom::Name(nn)).with_span(TokenId::Operand(1, 0).into()),
                    ExprOp::Atom(Atom::Location).with_span(TokenId::Operand(1, 0).into()),
                    ExprOp::Binary(BinOp::Minus).with_span(TokenId::Operand(1, 0).into()),
                ]),
                Width::Byte,
            ),
        ])
    }

    #[test]
    fn jr_c_e() {
        let nn = MockSymbolId(7);
        analyze(JR, vec![Condition::C.into(), nn.into()]).expect_fragments(vec![
            Fragment::Byte(0x38),
            Fragment::Immediate(
                Expr(vec![
                    ExprOp::Atom(Atom::Name(nn)).with_span(TokenId::Operand(1, 0).into()),
                    ExprOp::Atom(Atom::Location).with_span(TokenId::Operand(1, 0).into()),
                    ExprOp::Binary(BinOp::Minus).with_span(TokenId::Operand(1, 0).into()),
                ]),
                Width::Byte,
            ),
        ])
    }

    #[test]
    fn jp_deref_hl() {
        analyze(JP, vec![M::DerefHl.into()]).expect_fragments(vec![Fragment::Byte(0xe9)])
    }

    #[test]
    fn call_nn() {
        let nn = MockSymbolId(7);
        analyze(CALL, vec![nn.into()]).expect_fragments(vec![
            Fragment::Byte(0xcd),
            Fragment::Immediate(
                Expr(vec![
                    ExprOp::Atom(Atom::Name(nn)).with_span(TokenId::Operand(0, 0).into())
                ]),
                Width::Word,
            ),
        ])
    }

    #[test]
    fn call_nz_nn() {
        let nn = MockSymbolId(7);
        analyze(CALL, vec![Condition::Nz.into(), nn.into()]).expect_fragments(vec![
            Fragment::Byte(0xc4),
            Fragment::Immediate(
                Expr(vec![
                    ExprOp::Atom(Atom::Name(nn)).with_span(TokenId::Operand(1, 0).into())
                ]),
                Width::Word,
            ),
        ])
    }

    #[test]
    fn call_z_nn() {
        let nn = MockSymbolId(7);
        analyze(CALL, vec![Condition::Z.into(), nn.into()]).expect_fragments(vec![
            Fragment::Byte(0xcc),
            Fragment::Immediate(
                Expr(vec![
                    ExprOp::Atom(Atom::Name(nn)).with_span(TokenId::Operand(1, 0).into())
                ]),
                Width::Word,
            ),
        ])
    }

    #[test]
    fn call_nc_nn() {
        let nn = MockSymbolId(7);
        analyze(CALL, vec![Condition::Nc.into(), nn.into()]).expect_fragments(vec![
            Fragment::Byte(0xd4),
            Fragment::Immediate(
                Expr(vec![
                    ExprOp::Atom(Atom::Name(nn)).with_span(TokenId::Operand(1, 0).into())
                ]),
                Width::Word,
            ),
        ])
    }

    #[test]
    fn call_c_nn() {
        let nn = MockSymbolId(7);
        analyze(CALL, vec![Condition::C.into(), nn.into()]).expect_fragments(vec![
            Fragment::Byte(0xdc),
            Fragment::Immediate(
                Expr(vec![
                    ExprOp::Atom(Atom::Name(nn)).with_span(TokenId::Operand(1, 0).into())
                ]),
                Width::Word,
            ),
        ])
    }

    #[test]
    fn ret() {
        analyze(RET, vec![]).expect_fragments(vec![Fragment::Byte(0xc9)])
    }

    #[test]
    fn reti() {
        analyze(RETI, vec![]).expect_fragments(vec![Fragment::Byte(0xd9)])
    }

    #[test]
    fn ret_nz() {
        analyze(RET, vec![Condition::Nz.into()]).expect_fragments(vec![Fragment::Byte(0xc0)])
    }

    #[test]
    fn ret_z() {
        analyze(RET, vec![Condition::Z.into()]).expect_fragments(vec![Fragment::Byte(0xc8)])
    }

    #[test]
    fn ret_nc() {
        analyze(RET, vec![Condition::Nc.into()]).expect_fragments(vec![Fragment::Byte(0xd0)])
    }

    #[test]
    fn ret_c() {
        analyze(RET, vec![Condition::C.into()]).expect_fragments(vec![Fragment::Byte(0xd8)])
    }

    #[test]
    fn analyze_jp_c_deref_hl() {
        analyze(JP, vec![literal(C), M::DerefHl.into()]).expect_diag(
            ExpectedDiag::new(Message::AlwaysUnconditional).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_jp_z() {
        analyze(JP, vec![literal(Z)]).expect_diag(
            ExpectedDiag::new(Message::MissingTarget).with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_ret_a() {
        analyze(RET, vec![literal(A)]).expect_diag(
            ExpectedDiag::new(Message::CannotBeUsedAsTarget).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_reti_z() {
        analyze(RETI, vec![literal(Z)]).expect_diag(
            ExpectedDiag::new(Message::AlwaysUnconditional).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn reti_ident() {
        analyze(RETI, vec![MockSymbolId(7).into()]).expect_diag(
            ExpectedDiag::new(Message::CannotSpecifyTarget).with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_ret_z_ident() {
        analyze(RET, vec![literal(Z), MockSymbolId(7).into()]).expect_diag(
            ExpectedDiag::new(Message::CannotSpecifyTarget).with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_call_deref_hl() {
        analyze(CALL, vec![deref_symbol(Hl)]).expect_diag(
            ExpectedDiag::new(Message::RequiresConstantTarget {
                mnemonic: TokenId::Mnemonic.into(),
            })
            .with_highlight(TokenSpan::merge(
                TokenId::Operand(0, 0),
                TokenId::Operand(0, 2),
            )),
        )
    }
}
