use super::{Analysis, AtomKind, Operand, SimpleOperand};

use crate::diag::{DownstreamDiagnostics, EmitDiagnostic, Message};
use crate::model::{Branch, Condition, Instruction, Nullary};
use crate::span::Source;

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
    I: Iterator<Item = Result<Operand<V>, ()>>,
    V: Source<Span = S>,
    D: DownstreamDiagnostics<S>,
    S: Clone,
{
    pub fn analyze_branch(&mut self, branch: BranchKind) -> Result<Instruction<V>, ()> {
        let (condition, target) = self.collect_branch_operands()?;
        let variant = analyze_branch_variant((branch, &self.mnemonic.1), target, self.diagnostics)?;
        match variant {
            BranchVariant::Unconditional(branch) => match condition {
                None => Ok(branch.into()),
                Some((_, condition_span)) => {
                    self.emit_diagnostic(Message::AlwaysUnconditional.at(condition_span));
                    Err(())
                }
            },
            BranchVariant::PotentiallyConditional(branch) => Ok(Instruction::Branch(
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
    Option<(Condition, <V as Source>::Span)>,
    Option<BranchTarget<V>>,
);

enum BranchTarget<V: Source> {
    DerefHl(V::Span),
    Expr(V),
}

impl<V: Source> Source for BranchTarget<V> {
    type Span = V::Span;

    fn span(&self) -> Self::Span {
        match self {
            BranchTarget::DerefHl(span) => span.clone(),
            BranchTarget::Expr(expr) => expr.span(),
        }
    }
}

fn analyze_branch_target<V, D>(
    target: Option<Operand<V>>,
    diagnostics: &mut D,
) -> Result<Option<BranchTarget<V>>, ()>
where
    V: Source,
    D: DownstreamDiagnostics<V::Span>,
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
            diagnostics.emit_diagnostic(Message::CannotBeUsedAsTarget.at(operand.span()));
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

impl<V: Source> From<UnconditionalBranch> for Instruction<V> {
    fn from(branch: UnconditionalBranch) -> Self {
        match branch {
            UnconditionalBranch::JpDerefHl => Instruction::JpDerefHl,
            UnconditionalBranch::Reti => Instruction::Nullary(Nullary::Reti),
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
    D: DownstreamDiagnostics<V::Span>,
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
        diagnostics.emit_diagnostic(diagnostic);
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

    use crate::analysis::semantics::instruction::tests::*;
    use crate::analysis::session::SemanticExpr;
    use crate::syntax::keyword::Mnemonic;

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
                Explicit(Call) => Mnemonic::Call,
                Explicit(Jp) => Mnemonic::Jp,
                Explicit(Jr) => Mnemonic::Jr,
                Ret => Mnemonic::Ret,
            }
        }
    }

    fn describe_branch_instuctions() -> Vec<InstructionDescriptor> {
        use self::{ExplicitBranch::*, PotentiallyConditionalBranch::*};
        let mut descriptors = vec![
            (
                (Mnemonic::Jp, vec![deref(literal(Hl))]),
                Instruction::JpDerefHl,
            ),
            (
                (Mnemonic::Reti, vec![]),
                Instruction::Nullary(Nullary::Reti),
            ),
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
        let ident = "ident";
        let mut operands = Vec::new();
        let mut has_condition = false;
        if let Some(condition) = condition {
            operands.push(SemanticExpr::from(condition));
            has_condition = true;
        };
        if branch != Ret {
            operands.push(ident.into());
        };
        (
            (Mnemonic::from(branch), operands),
            Instruction::Branch(
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
        analyze(
            Mnemonic::Jp,
            vec![literal(C), SimpleOperand::DerefHl.into()],
        )
        .expect_diagnostic(
            ExpectedDiagnostic::new(Message::AlwaysUnconditional)
                .with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_jp_z() {
        analyze(Mnemonic::Jp, vec![literal(Z)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::MissingTarget).with_highlight(TokenId::Mnemonic),
        )
    }

    #[test]
    fn analyze_ret_a() {
        analyze(Mnemonic::Ret, vec![literal(A)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::CannotBeUsedAsTarget)
                .with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_reti_z() {
        analyze(Mnemonic::Reti, vec![literal(Z)]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::AlwaysUnconditional)
                .with_highlight(TokenId::Operand(0, 0)),
        )
    }

    #[test]
    fn analyze_ret_z_ident() {
        analyze(Mnemonic::Ret, vec![literal(Z), "target".into()]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::CannotSpecifyTarget)
                .with_highlight(TokenId::Operand(1, 0)),
        )
    }

    #[test]
    fn analyze_call_deref_hl() {
        analyze(Mnemonic::Call, vec![deref(literal(Hl))]).expect_diagnostic(
            ExpectedDiagnostic::new(Message::RequiresConstantTarget {
                mnemonic: TokenId::Mnemonic.into(),
            })
            .with_highlight(TokenSpan::merge(
                &TokenSpan::from(TokenId::Operand(0, 0)),
                &TokenId::Operand(0, 2).into(),
            )),
        )
    }
}
