use diagnostics::{Diagnostic, KeywordOperandCategory, Message, Source, Span};
use frontend::syntax::ast;
use frontend::syntax::ast::ExprVariant;
use frontend::syntax::keyword as kw;
use frontend::syntax::{Literal, TokenSpec};
use instruction::{Condition, PtrReg, Reg16, RegPair, RelocExpr, SimpleOperand};

#[derive(Debug, PartialEq)]
pub enum Operand<R> {
    Atom(AtomKind, R),
    Const(RelocExpr<R>),
    Deref(RelocExpr<R>),
}

impl<S: Span> Source for Operand<S> {
    type Span = S;
    fn span(&self) -> Self::Span {
        match self {
            Operand::Atom(_, span) => (*span).clone(),
            Operand::Const(expr) | Operand::Deref(expr) => expr.span(),
        }
    }
}

#[derive(Debug, PartialEq)]
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

type OperandResult<SI> = Result<Operand<SI>, Diagnostic<SI>>;

type Expr<S> = ast::Expr<<String as TokenSpec>::Ident, <String as TokenSpec>::Literal, S>;

pub fn analyze_operand<SI: Clone>(expr: Expr<SI>, context: Context) -> OperandResult<SI> {
    match expr.variant {
        ExprVariant::Literal(Literal::Operand(keyword)) => {
            Ok(analyze_keyword_operand((keyword, expr.span), context))
        }
        ExprVariant::Parentheses(inner) => analyze_deref_operand(*inner, expr.span),
        _ => Ok(Operand::Const(analyze_reloc_expr(expr)?)),
    }
}

fn analyze_deref_operand<SI: Clone>(expr: Expr<SI>, deref_interval: SI) -> OperandResult<SI> {
    match expr.variant {
        ExprVariant::Literal(Literal::Operand(keyword)) => {
            analyze_deref_operand_keyword((keyword, expr.span), deref_interval)
        }
        _ => Ok(Operand::Deref(analyze_reloc_expr(expr)?)),
    }
}

fn analyze_deref_operand_keyword<SI>(keyword: (kw::Operand, SI), deref: SI) -> OperandResult<SI> {
    match try_deref_operand_keyword(keyword.0) {
        Ok(atom) => Ok(Operand::Atom(atom, deref)),
        Err(category) => Err(Diagnostic::new(
            Message::CannotDereference {
                category,
                keyword: keyword.1,
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

pub fn analyze_reloc_expr<SI: Clone>(expr: Expr<SI>) -> Result<RelocExpr<SI>, Diagnostic<SI>> {
    match expr.variant {
        ExprVariant::Ident(ident) => Ok(RelocExpr::Symbol(ident, expr.span)),
        ExprVariant::Literal(Literal::Number(n)) => Ok(RelocExpr::Literal(n, expr.span)),
        ExprVariant::Literal(Literal::Operand(_)) => Err(Diagnostic::new(
            Message::KeywordInExpr {
                keyword: expr.span.clone(),
            },
            expr.span,
        )),
        ExprVariant::Literal(Literal::String(_)) => {
            Err(Diagnostic::new(Message::StringInInstruction, expr.span))
        }
        ExprVariant::Parentheses(expr) => analyze_reloc_expr(*expr),
    }
}

fn analyze_keyword_operand<R>((keyword, range): (kw::Operand, R), context: Context) -> Operand<R> {
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
        Hld | Hli => panic!(),
        L => AtomKind::Simple(SimpleOperand::L),
        Nc => AtomKind::Condition(Condition::Nc),
        Nz => AtomKind::Condition(Condition::Nz),
        Sp => AtomKind::Reg16(Reg16::Sp),
        Z => AtomKind::Condition(Condition::Z),
    };
    Operand::Atom(kind, range)
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

    pub fn check_for_unexpected_operands<SI>(
        self,
        source_interval: SI,
    ) -> Result<(), Diagnostic<SI>> {
        let expected = self.count;
        let extra = self.operands.count();
        let actual = expected + extra;
        if actual == expected {
            Ok(())
        } else {
            Err(Diagnostic::new(
                Message::OperandCount { actual, expected },
                source_interval,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let expr = Expr {
            variant: ExprVariant::Parentheses(Box::new(Expr {
                variant: ExprVariant::Literal(Literal::Operand(ptr_reg.into())),
                span: 0,
            })),
            span: 1,
        };
        assert_eq!(
            analyze_operand(expr, Context::Other),
            Ok(Operand::Atom(AtomKind::DerefPtrReg(ptr_reg), 1))
        )
    }

    #[test]
    fn analyze_deref_af() {
        let parsed_expr = Expr {
            variant: ExprVariant::Parentheses(Box::new(Expr {
                variant: ExprVariant::Literal(Literal::Operand(kw::Operand::Af)),
                span: 0,
            })),
            span: 1,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Err(Diagnostic::new(
                Message::CannotDereference {
                    category: KeywordOperandCategory::RegPair,
                    keyword: 0
                },
                1
            ))
        )
    }

    #[test]
    fn analyze_repeated_parentheses() {
        let n = 0x42;
        let span = 0;
        let parsed_expr = Expr {
            variant: ExprVariant::Parentheses(Box::new(Expr {
                variant: ExprVariant::Parentheses(Box::new(Expr {
                    variant: ExprVariant::Literal(Literal::Number(n)),
                    span,
                })),
                span: 1,
            })),
            span: 2,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Ok(Operand::Deref(RelocExpr::Literal(n, span)))
        )
    }

    #[test]
    fn analyze_reg_in_expr() {
        let span = 0;
        let parsed_expr = Expr {
            variant: ExprVariant::Parentheses(Box::new(Expr {
                variant: ExprVariant::Parentheses(Box::new(Expr {
                    variant: ExprVariant::Literal(Literal::Operand(kw::Operand::Z)),
                    span,
                })),
                span: 1,
            })),
            span: 2,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Err(Diagnostic::new(
                Message::KeywordInExpr { keyword: span },
                span
            ))
        )
    }

    #[test]
    fn analyze_string_in_instruction() {
        let span = 0;
        let parsed_expr = Expr {
            variant: ExprVariant::Literal(Literal::String("some_string".into())),
            span,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Err(Diagnostic::new(Message::StringInInstruction, span))
        )
    }
}
