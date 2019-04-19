use super::{Arg, ArgAtom, ArgEvaluator, ArgUnaryOp, ArgVariant, EvalArg};

use crate::analysis::session::Finish;
use crate::analysis::{Ident, Literal};
use crate::diag::*;
use crate::model::{Condition, PtrReg, Reg16, RegPair, SimpleOperand};
use crate::span::Source;
use crate::syntax::keyword as kw;

#[derive(Debug, PartialEq)]
pub enum Operand<V: Source> {
    Atom(AtomKind, V::Span),
    Const(V),
    Deref(V),
}

impl<V: Source> Source for Operand<V> {
    type Span = V::Span;

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

pub(super) fn analyze_operand<C, R, S>(
    expr: Arg<R, S>,
    context: Context,
    mut value_context: C,
) -> (C::Parent, Result<Operand<C::Value>, ()>)
where
    C: ArgEvaluator<Ident<R>, S> + Finish<S>,
    R: Eq,
    S: Clone,
{
    match expr.variant {
        ArgVariant::Atom(ArgAtom::Literal(Literal::Operand(keyword))) => {
            let result =
                analyze_keyword_operand((keyword, expr.span), context, value_context.diagnostics());
            (value_context.finish().0, result)
        }
        ArgVariant::Unary(ArgUnaryOp::Parentheses, inner) => {
            analyze_deref_operand(*inner, expr.span, value_context)
        }
        _ => match value_context.eval_arg(expr) {
            Ok(()) => {
                let (session, expr) = value_context.finish();
                (session, Ok(Operand::Const(expr)))
            }
            Err(()) => (value_context.finish().0, Err(())),
        },
    }
}

fn analyze_deref_operand<C, R, S>(
    expr: Arg<R, S>,
    deref_span: S,
    mut value_context: C,
) -> (C::Parent, Result<Operand<C::Value>, ()>)
where
    C: ArgEvaluator<Ident<R>, S> + Finish<S>,
    R: Eq,
    S: Clone,
{
    match expr.variant {
        ArgVariant::Atom(ArgAtom::Literal(Literal::Operand(keyword))) => {
            let result = analyze_deref_operand_keyword(
                (keyword, &expr.span),
                deref_span,
                value_context.diagnostics(),
            );
            (value_context.finish().0, result)
        }
        _ => match value_context.eval_arg(expr) {
            Ok(()) => {
                let (session, expr) = value_context.finish();
                (session, Ok(Operand::Deref(expr)))
            }
            Err(()) => (value_context.finish().0, Err(())),
        },
    }
}

fn analyze_deref_operand_keyword<V: Source, D>(
    keyword: (kw::Operand, &V::Span),
    deref: V::Span,
    diagnostics: &mut D,
) -> Result<Operand<V>, ()>
where
    V: Source,
    D: DownstreamDiagnostics<V::Span>,
{
    match try_deref_operand_keyword(keyword.0) {
        Ok(atom) => Ok(Operand::Atom(atom, deref)),
        Err(category) => {
            let stripped = diagnostics.strip_span(keyword.1);
            diagnostics.emit_diagnostic(
                Message::CannotDereference {
                    category,
                    operand: stripped,
                }
                .at(deref),
            );
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
    V: Source,
    D: DownstreamDiagnostics<V::Span>,
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
            let stripped = diagnostics.strip_span(&span);
            diagnostics
                .emit_diagnostic(Message::MustBeDeref { operand: stripped }.at(span.clone()));
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

    use crate::diag::MockSpan;
    use crate::model::{Atom, Expr};

    impl MockSpan for i32 {
        fn default() -> Self {
            unimplemented!()
        }

        fn merge(&self, _: &Self) -> Self {
            unimplemented!()
        }
    }

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
        let expr = Arg::<String, _> {
            variant: ArgVariant::Unary(
                ArgUnaryOp::Parentheses,
                Box::new(Arg::from_atom(
                    ArgAtom::Literal(Literal::Operand(ptr_reg.into())),
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

    type OperandResult<S> = Result<Operand<Expr<Ident<String>, S>>, Vec<DiagnosticsEvent<S>>>;

    fn analyze_operand<S: Clone + MockSpan + PartialEq>(
        expr: Arg<String, S>,
        context: Context,
    ) -> OperandResult<S> {
        use crate::analysis::session::MockBuilder;
        use std::cell::RefCell;

        let log = RefCell::new(Vec::new());
        let builder = MockBuilder::with_log(&log);
        let (_, result) = super::analyze_operand(expr, context, builder);
        result.map_err(|_| log.into_inner())
    }

    #[test]
    fn analyze_deref_af() {
        let parsed_expr = Arg::<String, _> {
            variant: ArgVariant::Unary(
                ArgUnaryOp::Parentheses,
                Box::new(Arg::from_atom(
                    ArgAtom::Literal(Literal::Operand(kw::Operand::Af)),
                    0,
                )),
            ),
            span: 1,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Err(vec![CompactDiagnostic::from(
                Message::CannotDereference {
                    category: KeywordOperandCategory::RegPair,
                    operand: 0,
                }
                .at(1)
            )
            .into()])
        )
    }

    #[test]
    fn analyze_repeated_parentheses() {
        let n = 0x42;
        let span = 0;
        let parsed_expr = Arg::<String, _> {
            variant: ArgVariant::Unary(
                ArgUnaryOp::Parentheses,
                Box::new(Arg {
                    variant: ArgVariant::Unary(
                        ArgUnaryOp::Parentheses,
                        Box::new(Arg::from_atom(ArgAtom::Literal(Literal::Number(n)), span)),
                    ),
                    span: 1,
                }),
            ),
            span: 2,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Ok(Operand::Deref(Expr::from_atom(Atom::Literal(n), span)))
        )
    }

    #[test]
    fn analyze_reg_in_expr() {
        let span = 0;
        let parsed_expr = Arg::<String, _> {
            variant: ArgVariant::Unary(
                ArgUnaryOp::Parentheses,
                Box::new(Arg {
                    variant: ArgVariant::Unary(
                        ArgUnaryOp::Parentheses,
                        Box::new(Arg::from_atom(
                            ArgAtom::Literal(Literal::Operand(kw::Operand::Z)),
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
            Err(vec![CompactDiagnostic::from(
                Message::KeywordInExpr { keyword: span }.at(span)
            )
            .into()])
        )
    }

    #[test]
    fn analyze_string_in_instruction() {
        let span = 0;
        let parsed_expr = Arg::<String, _>::from_atom(
            ArgAtom::Literal(Literal::String("some_string".into())),
            span,
        );
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Err(vec![CompactDiagnostic::from(
                Message::StringInInstruction.at(span)
            )
            .into()])
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
        let expr = Arg::<String, _>::from_atom(ArgAtom::Literal(Literal::Operand(keyword)), span);
        assert_eq!(
            analyze_operand(expr, Context::Other),
            Err(vec![CompactDiagnostic::from(
                Message::MustBeDeref { operand: span }.at(span)
            )
            .into()])
        )
    }
}
