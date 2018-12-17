use super::{analyze_reloc_expr, ExprVariant, SemanticAtom, SemanticExpr, SemanticUnary};
use crate::backend::ValueBuilder;
use crate::diagnostics::{CompactDiagnostic, KeywordOperandCategory, Message};
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

type OperandResult<V> = Result<Operand<V>, CompactDiagnostic<<V as Span>::Span>>;

pub fn analyze_operand<I: Into<String>, V: Source>(
    expr: SemanticExpr<I, V::Span>,
    context: Context,
    builder: &mut impl ValueBuilder<V>,
) -> OperandResult<V> {
    match expr.variant {
        ExprVariant::Atom(SemanticAtom::Literal(Literal::Operand(keyword))) => {
            analyze_keyword_operand((keyword, expr.span), context)
        }
        ExprVariant::Unary(SemanticUnary::Parentheses, inner) => {
            analyze_deref_operand(*inner, expr.span, builder)
        }
        _ => Ok(Operand::Const(analyze_reloc_expr(expr, builder)?)),
    }
}

fn analyze_deref_operand<I: Into<String>, V: Source>(
    expr: SemanticExpr<I, V::Span>,
    deref_span: V::Span,
    builder: &mut impl ValueBuilder<V>,
) -> OperandResult<V> {
    match expr.variant {
        ExprVariant::Atom(SemanticAtom::Literal(Literal::Operand(keyword))) => {
            analyze_deref_operand_keyword((keyword, expr.span), deref_span)
        }
        _ => Ok(Operand::Deref(analyze_reloc_expr(expr, builder)?)),
    }
}

fn analyze_deref_operand_keyword<V: Source>(
    keyword: (kw::Operand, V::Span),
    deref: V::Span,
) -> OperandResult<V> {
    match try_deref_operand_keyword(keyword.0) {
        Ok(atom) => Ok(Operand::Atom(atom, deref)),
        Err(category) => Err(CompactDiagnostic::new(
            Message::CannotDereference {
                category,
                operand: keyword.1,
            },
            deref,
        )),
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

fn analyze_keyword_operand<V: Source>(
    (keyword, span): (kw::Operand, V::Span),
    context: Context,
) -> OperandResult<V> {
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
        Hld | Hli => Err(CompactDiagnostic::new(
            Message::MustBeDeref {
                operand: span.clone(),
            },
            span.clone(),
        ))?,
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

impl<I: Iterator<Item = Result<T, E>>, T, E> OperandCounter<I> {
    pub fn new(operands: I) -> OperandCounter<I> {
        OperandCounter { operands, count: 0 }
    }

    pub fn seen(&self) -> usize {
        self.count
    }

    pub fn next(&mut self) -> Result<Option<T>, E> {
        self.operands.next().map_or(Ok(None), |operand| {
            self.count += 1;
            operand.map(Some)
        })
    }

    pub fn check_for_unexpected_operands<S>(self, span: S) -> Result<(), CompactDiagnostic<S>> {
        let expected = self.count;
        let extra = self.operands.count();
        let actual = expected + extra;
        if actual == expected {
            Ok(())
        } else {
            Err(CompactDiagnostic::new(
                Message::OperandCount { actual, expected },
                span,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{RelocAtom, RelocExpr, RelocExprBuilder};
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
    ) -> OperandResult<RelocExpr<S>> {
        super::analyze_operand(expr, context, &mut RelocExprBuilder::new())
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
            Err(CompactDiagnostic::new(
                Message::CannotDereference {
                    category: KeywordOperandCategory::RegPair,
                    operand: 0,
                },
                1
            ))
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
            Err(CompactDiagnostic::new(
                Message::KeywordInExpr { keyword: span },
                span
            ))
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
            Err(CompactDiagnostic::new(Message::StringInInstruction, span))
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
            Err(CompactDiagnostic::new(
                Message::MustBeDeref { operand: span },
                span
            ))
        )
    }
}
