use super::{
    AnalyzeExpr, ExprAnalysisContext, ExprVariant, SemanticAtom, SemanticExpr, SemanticUnary,
};
use crate::backend::ValueBuilder;
use crate::diagnostics::*;
use crate::frontend::syntax::keyword as kw;
use crate::frontend::syntax::Literal;
use crate::instruction::{Condition, PtrReg, Reg16, RegPair, SimpleOperand};
use crate::span::{Source, Span};

#[derive(Debug, PartialEq)]
pub enum Operand<V: Source> {
    Atom(AtomKind, V::Span),
    Const(V),
    Deref(V),
}

impl<V: Source> Span for Operand<V> {
    type Span = V::Span;
}

impl<V: Source> Source for Operand<V> {
    fn span(&self) -> Self::Span {
        match self {
            Operand::Atom(_, span) => (*span).clone(),
            Operand::Const(expr) | Operand::Deref(expr) => expr.span(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AtomKind {
    Simple(SimpleOperand),
    Condition(Condition),
    Reg16(Reg16),
    RegPair(RegPair),
    DerefPtrReg(PtrReg),
    DerefC,
}

#[derive(Clone, Copy)]
pub enum Context {
    Branch,
    Stack,
    Other,
}

pub fn analyze_operand<I, B, D>(
    expr: SemanticExpr<I, D::Span>,
    context: Context,
    expr_analysis_context: &mut ExprAnalysisContext<B, D>,
) -> Result<Operand<B::Value>, ()>
where
    I: Into<String>,
    B: ValueBuilder<Span = D::Span>,
    D: DownstreamDiagnostics,
{
    match expr.variant {
        ExprVariant::Atom(SemanticAtom::Literal(Literal::Operand(keyword))) => {
            analyze_keyword_operand(
                (keyword, expr.span),
                context,
                expr_analysis_context.diagnostics,
            )
        }
        ExprVariant::Unary(SemanticUnary::Parentheses, inner) => {
            analyze_deref_operand(*inner, expr.span, expr_analysis_context)
        }
        _ => Ok(Operand::Const(expr_analysis_context.analyze_expr(expr)?)),
    }
}

fn analyze_deref_operand<I, B, D>(
    expr: SemanticExpr<I, D::Span>,
    deref_span: D::Span,
    expr_analysis_context: &mut ExprAnalysisContext<B, D>,
) -> Result<Operand<B::Value>, ()>
where
    I: Into<String>,
    B: ValueBuilder<Span = D::Span>,
    D: DownstreamDiagnostics,
{
    match expr.variant {
        ExprVariant::Atom(SemanticAtom::Literal(Literal::Operand(keyword))) => {
            analyze_deref_operand_keyword(
                (keyword, expr.span),
                deref_span,
                expr_analysis_context.diagnostics,
            )
        }
        _ => Ok(Operand::Deref(expr_analysis_context.analyze_expr(expr)?)),
    }
}

fn analyze_deref_operand_keyword<V: Source, D>(
    keyword: (kw::Operand, D::Span),
    deref: D::Span,
    diagnostics: &mut D,
) -> Result<Operand<V>, ()>
where
    V: Source<Span = D::Span>,
    D: DownstreamDiagnostics,
{
    match try_deref_operand_keyword(keyword.0) {
        Ok(atom) => Ok(Operand::Atom(atom, deref)),
        Err(category) => {
            diagnostics.emit_diagnostic(CompactDiagnostic::new(
                Message::CannotDereference {
                    category,
                    operand: keyword.1,
                },
                deref,
            ));
            Err(())
        }
    }
}

fn try_deref_operand_keyword(keyword: kw::Operand) -> Result<AtomKind, KeywordOperandCategory> {
    use self::kw::Operand::*;
    match keyword {
        Bc => Ok(AtomKind::DerefPtrReg(PtrReg::Bc)),
        C => Ok(AtomKind::DerefC),
        De => Ok(AtomKind::DerefPtrReg(PtrReg::De)),
        Hl => Ok(AtomKind::Simple(SimpleOperand::DerefHl)),
        Hld => Ok(AtomKind::DerefPtrReg(PtrReg::Hld)),
        Hli => Ok(AtomKind::DerefPtrReg(PtrReg::Hli)),
        A | B | D | E | H | L | Sp => Err(KeywordOperandCategory::Reg),
        Af => Err(KeywordOperandCategory::RegPair),
        Nc | Nz | Z => Err(KeywordOperandCategory::ConditionCode),
    }
}

fn analyze_keyword_operand<V: Source, D>(
    (keyword, span): (kw::Operand, V::Span),
    context: Context,
    diagnostics: &mut D,
) -> Result<Operand<V>, ()>
where
    V: Source<Span = D::Span>,
    D: DownstreamDiagnostics,
{
    use self::kw::Operand::*;
    use self::Context::*;
    let kind = match keyword {
        A => AtomKind::Simple(SimpleOperand::A),
        Af => AtomKind::RegPair(RegPair::Af),
        B => AtomKind::Simple(SimpleOperand::B),
        Bc => match context {
            Stack => AtomKind::RegPair(RegPair::Bc),
            _ => AtomKind::Reg16(Reg16::Bc),
        },
        C => match context {
            Branch => AtomKind::Condition(Condition::C),
            _ => AtomKind::Simple(SimpleOperand::C),
        },
        D => AtomKind::Simple(SimpleOperand::D),
        De => match context {
            Stack => AtomKind::RegPair(RegPair::De),
            _ => AtomKind::Reg16(Reg16::De),
        },
        E => AtomKind::Simple(SimpleOperand::E),
        H => AtomKind::Simple(SimpleOperand::H),
        Hl => match context {
            Stack => AtomKind::RegPair(RegPair::Hl),
            _ => AtomKind::Reg16(Reg16::Hl),
        },
        Hld | Hli => {
            diagnostics.emit_diagnostic(CompactDiagnostic::new(
                Message::MustBeDeref {
                    operand: span.clone(),
                },
                span.clone(),
            ));
            Err(())?
        }
        L => AtomKind::Simple(SimpleOperand::L),
        Nc => AtomKind::Condition(Condition::Nc),
        Nz => AtomKind::Condition(Condition::Nz),
        Sp => AtomKind::Reg16(Reg16::Sp),
        Z => AtomKind::Condition(Condition::Z),
    };
    Ok(Operand::Atom(kind, span))
}

pub struct OperandCounter<I> {
    operands: I,
    count: usize,
}

impl<I> OperandCounter<I> {
    pub fn new(operands: I) -> OperandCounter<I> {
        OperandCounter { operands, count: 0 }
    }

    pub fn seen(&self) -> usize {
        self.count
    }
}

impl<I: Iterator> Iterator for OperandCounter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.operands.next();
        if item.is_some() {
            self.count += 1;
        }
        item
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{RelocAtom, RelocExpr, RelocExprBuilder};
    use crate::diagnostics::span::MergeSpans;
    use crate::frontend::semantics::DiagnosticsCollector;
    use std::fmt::Debug;

    #[test]
    fn analyze_deref_bc() {
        analyze_deref_ptr_reg(PtrReg::Bc)
    }

    #[test]
    fn analyze_deref_de() {
        analyze_deref_ptr_reg(PtrReg::De)
    }

    #[test]
    fn analyze_deref_hli() {
        analyze_deref_ptr_reg(PtrReg::Hli)
    }

    #[test]
    fn analyze_deref_hld() {
        analyze_deref_ptr_reg(PtrReg::Hld)
    }

    fn analyze_deref_ptr_reg(ptr_reg: PtrReg) {
        let expr = SemanticExpr::<String, _> {
            variant: ExprVariant::Unary(
                SemanticUnary::Parentheses,
                Box::new(SemanticExpr::from_atom(
                    SemanticAtom::Literal(Literal::Operand(ptr_reg.into())),
                    0,
                )),
            ),
            span: 1,
        };
        assert_eq!(
            analyze_operand(expr, Context::Other),
            Ok(Operand::Atom(AtomKind::DerefPtrReg(ptr_reg), 1))
        )
    }

    fn analyze_operand<I: Into<String>, S: Clone + Debug + PartialEq>(
        expr: SemanticExpr<I, S>,
        context: Context,
    ) -> Result<Operand<RelocExpr<S>>, Vec<CompactDiagnostic<S>>>
    where
        DiagnosticsCollector<S>: DownstreamDiagnostics<Span = S>,
    {
        let mut collector = DiagnosticsCollector(Vec::new());
        let result = super::analyze_operand(
            expr,
            context,
            &mut ExprAnalysisContext::new(&mut RelocExprBuilder::new(), &mut collector),
        );
        result.map_err(|_| collector.0)
    }

    impl MergeSpans for DiagnosticsCollector<i32> {
        fn merge_spans(&mut self, _: &Self::Span, _: &Self::Span) -> Self::Span {
            unreachable!()
        }
    }

    #[test]
    fn analyze_deref_af() {
        let parsed_expr = SemanticExpr::<String, _> {
            variant: ExprVariant::Unary(
                SemanticUnary::Parentheses,
                Box::new(SemanticExpr::from_atom(
                    SemanticAtom::Literal(Literal::Operand(kw::Operand::Af)),
                    0,
                )),
            ),
            span: 1,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Err(vec![CompactDiagnostic::new(
                Message::CannotDereference {
                    category: KeywordOperandCategory::RegPair,
                    operand: 0,
                },
                1
            )])
        )
    }

    #[test]
    fn analyze_repeated_parentheses() {
        let n = 0x42;
        let span = 0;
        let parsed_expr = SemanticExpr::<String, _> {
            variant: ExprVariant::Unary(
                SemanticUnary::Parentheses,
                Box::new(SemanticExpr {
                    variant: ExprVariant::Unary(
                        SemanticUnary::Parentheses,
                        Box::new(SemanticExpr::from_atom(
                            SemanticAtom::Literal(Literal::Number(n)),
                            span,
                        )),
                    ),
                    span: 1,
                }),
            ),
            span: 2,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Ok(Operand::Deref(RelocExpr::from_atom(
                RelocAtom::Literal(n),
                span
            )))
        )
    }

    #[test]
    fn analyze_reg_in_expr() {
        let span = 0;
        let parsed_expr = SemanticExpr::<String, _> {
            variant: ExprVariant::Unary(
                SemanticUnary::Parentheses,
                Box::new(SemanticExpr {
                    variant: ExprVariant::Unary(
                        SemanticUnary::Parentheses,
                        Box::new(SemanticExpr::from_atom(
                            SemanticAtom::Literal(Literal::Operand(kw::Operand::Z)),
                            span,
                        )),
                    ),
                    span: 1,
                }),
            ),
            span: 2,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Err(vec![CompactDiagnostic::new(
                Message::KeywordInExpr { keyword: span },
                span
            )])
        )
    }

    #[test]
    fn analyze_string_in_instruction() {
        let span = 0;
        let parsed_expr = SemanticExpr::<String, _>::from_atom(
            SemanticAtom::Literal(Literal::String("some_string".into())),
            span,
        );
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Err(vec![CompactDiagnostic::new(
                Message::StringInInstruction,
                span
            )])
        )
    }

    #[test]
    fn analyze_bare_hld() {
        test_bare_ptr_reg(kw::Operand::Hld)
    }

    #[test]
    fn analyze_bare_hli() {
        test_bare_ptr_reg(kw::Operand::Hli)
    }

    fn test_bare_ptr_reg(keyword: kw::Operand) {
        let span = 0;
        let expr = SemanticExpr::<String, _>::from_atom(
            SemanticAtom::Literal(Literal::Operand(keyword)),
            span,
        );
        assert_eq!(
            analyze_operand(expr, Context::Other),
            Err(vec![CompactDiagnostic::new(
                Message::MustBeDeref { operand: span },
                span
            )])
        )
    }
}
