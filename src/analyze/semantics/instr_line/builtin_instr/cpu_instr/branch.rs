use super::{Analysis, AtomKind, Operand, SimpleOperand};

use crate::diag::{Diagnostics, EmitDiag, Message};
use crate::object::builder::{Branch, Condition, CpuInstr, Nullary};
use crate::span::{Source, SpanSource};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BranchKind {
    Explicit(ExplicitBranch),
    Implicit(ImplicitBranch),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExplicitBranch {
    Call,
    Jp,
    Jr,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImplicitBranch {
    Ret,
    Reti,
}

impl<'a, I, V, D, S> Analysis<'a, I, D, S>
where
    I: Iterator<Item = Result<Operand<V, S>, ()>>,
    V: Source<Span = S>,
    D: Diagnostics<S>,
    S: Clone,
{
    pub fn analyze_branch(&mut self, branch: BranchKind) -> Result<CpuInstr<V>, ()> {
        let (condition, target) = self.collect_branch_operands()?;
        let variant = analyze_branch_variant((branch, &self.mnemonic.1), target, self.diagnostics)?;
        match variant {
            BranchVariant::Unconditional(branch) => match condition {
                None => Ok(branch.into()),
                Some((_, condition_span)) => {
                    self.emit_diag(Message::AlwaysUnconditional.at(condition_span));
                    Err(())
                }
            },
            BranchVariant::PotentiallyConditional(branch) => Ok(CpuInstr::Branch(
                branch,
                condition.map(|(condition, _)| condition),
            )),
        }
    }

    fn collect_branch_operands(&mut self) -> Result<BranchOperands<V>, ()> {
        let first_operand = self.next_operand()?;
        Ok(
            if let Some(Operand::Atom(AtomKind::Condition(condition), range)) = first_operand {
                (
                    Some((condition, range)),
                    analyze_branch_target(self.next_operand()?, self.diagnostics)?,
                )
            } else {
                (
                    None,
                    analyze_branch_target(first_operand, self.diagnostics)?,
                )
            },
        )
    }
}

type BranchOperands<V> = (
    Option<(Condition, <V as SpanSource>::Span)>,
    Option<BranchTarget<V>>,
);

enum BranchTarget<V: SpanSource> {
    DerefHl(V::Span),
    Expr(V),
}

impl<V: SpanSource> SpanSource for BranchTarget<V> {
    type Span = V::Span;
}

impl<V: Source> Source for BranchTarget<V> {
    fn span(&self) -> Self::Span {
        match self {
            BranchTarget::DerefHl(span) => span.clone(),
            BranchTarget::Expr(expr) => expr.span(),
        }
    }
}

fn analyze_branch_target<V, D>(
    target: Option<Operand<V, V::Span>>,
    diagnostics: &mut D,
) -> Result<Option<BranchTarget<V>>, ()>
where
    V: Source,
    D: Diagnostics<V::Span>,
{
    let target = match target {
        Some(target) => target,
        None => return Ok(None),
    };
    match target {
        Operand::Const(expr) => Ok(Some(BranchTarget::Expr(expr))),
        Operand::Atom(AtomKind::Simple(SimpleOperand::DerefHl), span) => {
            Ok(Some(BranchTarget::DerefHl(span)))
        }
        operand => {
            diagnostics.emit_diag(Message::CannotBeUsedAsTarget.at(operand.span()));
            Err(())
        }
    }
}

enum BranchVariant<V> {
    PotentiallyConditional(Branch<V>),
    Unconditional(UnconditionalBranch),
}

enum UnconditionalBranch {
    JpDerefHl,
    Reti,
}

impl<V: Source> From<UnconditionalBranch> for CpuInstr<V> {
    fn from(branch: UnconditionalBranch) -> Self {
        match branch {
            UnconditionalBranch::JpDerefHl => CpuInstr::JpDerefHl,
            UnconditionalBranch::Reti => CpuInstr::Nullary(Nullary::Reti),
        }
    }
}

fn analyze_branch_variant<V, D>(
    kind: (BranchKind, &V::Span),
    target: Option<BranchTarget<V>>,
    diagnostics: &mut D,
) -> Result<BranchVariant<V>, ()>
where
    V: Source,
    D: Diagnostics<V::Span>,
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
            BranchVariant::PotentiallyConditional(mk_explicit_branch(branch, expr)),
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

fn mk_explicit_branch<V>(branch: ExplicitBranch, target: V) -> Branch<V> {
    match branch {
        ExplicitBranch::Call => Branch::Call(target),
        ExplicitBranch::Jp => Branch::Jp(target),
        ExplicitBranch::Jr => Branch::Jr(target),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analyze::semantics::instr_line::builtin_instr::cpu_instr::mnemonic::*;
    use crate::analyze::semantics::instr_line::builtin_instr::cpu_instr::tests::*;
    use crate::diag::Merge;

    #[test]
    fn analyze_legal_branch_instructions() {
        test_instruction_analysis(describe_branch_instuctions())
    }

    #[derive(Clone, Copy, PartialEq)]
    enum PotentiallyConditionalBranch {
        Explicit(ExplicitBranch),
        Ret,
    }

    impl From<PotentiallyConditionalBranch> for Mnemonic {
        fn from(branch: PotentiallyConditionalBranch) -> Self {
            use self::{ExplicitBranch::*, PotentiallyConditionalBranch::*};
            match branch {
                Explicit(Call) => CALL,
                Explicit(Jp) => JP,
                Explicit(Jr) => JR,
                Ret => RET,
            }
        }
    }

    fn describe_branch_instuctions() -> Vec<InstructionDescriptor> {
        use self::{ExplicitBranch::*, PotentiallyConditionalBranch::*};
        let mut descriptors = vec![
            ((JP, vec![deref(literal(Hl))]), CpuInstr::JpDerefHl),
            ((RETI, vec![]), CpuInstr::Nullary(Nullary::Reti)),
        ];
        for &kind in [Explicit(Call), Explicit(Jp), Explicit(Jr), Ret].iter() {
            descriptors.push(describe_branch(kind, None));
            for &condition in &[Condition::C, Condition::Nc, Condition::Nz, Condition::Z] {
                descriptors.push(describe_branch(kind, Some(condition)))
            }
        }
        descriptors
    }

    fn describe_branch(
        branch: PotentiallyConditionalBranch,
        condition: Option<Condition>,
    ) -> InstructionDescriptor {
        use self::PotentiallyConditionalBranch::*;
        let ident = MockSymbolId(7);
        let mut operands = Vec::new();
        let mut has_condition = false;
        if let Some(condition) = condition {
            operands.push(condition.into());
            has_condition = true;
        };
        if branch != Ret {
            operands.push(ident.into());
        };
        (
            (branch.into(), operands),
            CpuInstr::Branch(
                match branch {
                    Ret => Branch::Ret,
                    Explicit(explicit) => mk_explicit_branch(
                        explicit,
                        name(
                            ident,
                            TokenId::Operand(if has_condition { 1 } else { 0 }, 0),
                        ),
                    ),
                },
                condition,
            ),
        )
    }

    #[test]
    fn analyze_jp_c_deref_hl() {
        analyze(JP, vec![literal(C), SimpleOperand::DerefHl.into()]).expect_diag(
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
    fn analyze_ret_z_ident() {
        analyze(RET, vec![literal(Z), MockSymbolId(7).into()]).expect_diag(
            ExpectedDiag::new(Message::CannotSpecifyTarget).with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_call_deref_hl() {
        analyze(CALL, vec![deref(literal(Hl))]).expect_diag(
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
