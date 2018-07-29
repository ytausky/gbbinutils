use super::{ExprNode, ParsedExpr};
use diagnostics::{Diagnostic, KeywordOperandCategory, Message, Source, SourceRange};
use frontend::syntax::keyword as kw;
use frontend::syntax::Literal;
use instruction::{Condition, PtrReg, Reg16, RegPair, RelocExpr, SimpleOperand};

#[derive(Debug, PartialEq)]
pub enum Operand<R> {
    Atom(AtomKind, R),
    Const(RelocExpr<R>),
    Deref(RelocExpr<R>),
}

impl<SR: SourceRange> Source for Operand<SR> {
    type Range = SR;
    fn source_range(&self) -> Self::Range {
        match self {
            Operand::Atom(_, range) => (*range).clone(),
            Operand::Const(expr) | Operand::Deref(expr) => expr.source_range(),
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

pub fn analyze_operand<SI: Clone>(
    expr: ParsedExpr<String, SI>,
    context: Context,
) -> OperandResult<SI> {
    match expr.node {
        ExprNode::Literal(Literal::Operand(keyword)) => {
            Ok(analyze_keyword_operand((keyword, expr.interval), context))
        }
        ExprNode::Parenthesized(inner) => analyze_deref_operand(*inner, expr.interval),
        _ => Ok(Operand::Const(analyze_reloc_expr(expr)?)),
    }
}

fn analyze_deref_operand<SI: Clone>(
    expr: ParsedExpr<String, SI>,
    deref_interval: SI,
) -> OperandResult<SI> {
    match expr.node {
        ExprNode::Literal(Literal::Operand(keyword)) => {
            analyze_deref_operand_keyword((keyword, expr.interval), deref_interval)
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

pub fn analyze_reloc_expr<SI: Clone>(
    expr: ParsedExpr<String, SI>,
) -> Result<RelocExpr<SI>, Diagnostic<SI>> {
    match expr.node {
        ExprNode::Ident(ident) => Ok(RelocExpr::Symbol(ident, expr.interval)),
        ExprNode::Literal(Literal::Number(n)) => Ok(RelocExpr::Literal(n, expr.interval)),
        ExprNode::Literal(Literal::Operand(_)) => Err(Diagnostic::new(
            Message::KeywordInExpr {
                keyword: expr.interval.clone(),
            },
            expr.interval,
        )),
        ExprNode::Literal(Literal::String(_)) => {
            Err(Diagnostic::new(Message::StringInInstruction, expr.interval))
        }
        ExprNode::Parenthesized(expr) => analyze_reloc_expr(*expr),
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
        let parsed_expr = ParsedExpr {
            node: ExprNode::Parenthesized(Box::new(ParsedExpr {
                node: ExprNode::Literal(Literal::Operand(ptr_reg.into())),
                interval: 0,
            })),
            interval: 1,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Ok(Operand::Atom(AtomKind::DerefPtrReg(ptr_reg), 1))
        )
    }

    #[test]
    fn analyze_deref_af() {
        let parsed_expr = ParsedExpr {
            node: ExprNode::Parenthesized(Box::new(ParsedExpr {
                node: ExprNode::Literal(Literal::Operand(kw::Operand::Af)),
                interval: 0,
            })),
            interval: 1,
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
        let interval = 0;
        let parsed_expr = ParsedExpr {
            node: ExprNode::Parenthesized(Box::new(ParsedExpr {
                node: ExprNode::Parenthesized(Box::new(ParsedExpr {
                    node: ExprNode::Literal(Literal::Number(n)),
                    interval,
                })),
                interval: 1,
            })),
            interval: 2,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Ok(Operand::Deref(RelocExpr::Literal(n, interval)))
        )
    }

    #[test]
    fn analyze_reg_in_expr() {
        let interval = 0;
        let parsed_expr = ParsedExpr {
            node: ExprNode::Parenthesized(Box::new(ParsedExpr {
                node: ExprNode::Parenthesized(Box::new(ParsedExpr {
                    node: ExprNode::Literal(Literal::Operand(kw::Operand::Z)),
                    interval,
                })),
                interval: 1,
            })),
            interval: 2,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Err(Diagnostic::new(
                Message::KeywordInExpr { keyword: interval },
                interval
            ))
        )
    }

    #[test]
    fn analyze_string_in_instruction() {
        let interval = 0;
        let parsed_expr = ParsedExpr {
            node: ExprNode::Literal(Literal::String("some_string".into())),
            interval,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Err(Diagnostic::new(Message::StringInInstruction, interval))
        )
    }
}
