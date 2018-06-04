use diagnostics::{Diagnostic, Message, Source, SourceInterval};
use frontend::syntax::{keyword::OperandKeyword, ExprNode, Literal, ParsedExpr};
use instruction::{Condition, Reg16, RegPair, RelocExpr, SimpleOperand};

#[derive(Debug, PartialEq)]
pub enum Operand<R> {
    Atom(AtomKind, R),
    Const(RelocExpr<R>),
    Deref(RelocExpr<R>),
}

impl<SI: SourceInterval> Source for Operand<SI> {
    type Interval = SI;
    fn source_interval(&self) -> Self::Interval {
        match self {
            Operand::Atom(_, interval) => (*interval).clone(),
            Operand::Const(expr) | Operand::Deref(expr) => expr.source_interval(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum AtomKind {
    Simple(SimpleOperand),
    Condition(Condition),
    Reg16(Reg16),
    RegPair(RegPair),
    DerefC,
}

#[derive(Clone, Copy)]
pub enum Context {
    Branch,
    Stack,
    Other,
}

type OperandResult<SI> = Result<Operand<SI>, Diagnostic<SI>>;

pub fn analyze_operand<SI>(expr: ParsedExpr<String, SI>, context: Context) -> OperandResult<SI> {
    match expr.node {
        ExprNode::Literal(Literal::Operand(keyword)) => {
            Ok(analyze_keyword_operand((keyword, expr.interval), context))
        }
        ExprNode::Deref(inner) => analyze_deref_operand(*inner, expr.interval),
        _ => Ok(Operand::Const(analyze_reloc_expr(expr)?)),
    }
}

fn analyze_deref_operand<SI>(
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

fn analyze_deref_operand_keyword<SI>(
    keyword: (OperandKeyword, SI),
    deref: SI,
) -> OperandResult<SI> {
    use frontend::syntax::OperandKeyword::*;
    match keyword.0 {
        Af => Err(Diagnostic::new(Message::CannotDereference, deref)),
        C => Ok(Operand::Atom(AtomKind::DerefC, keyword.1)),
        Hl => Ok(Operand::Atom(
            AtomKind::Simple(SimpleOperand::DerefHl),
            keyword.1,
        )),
        _ => panic!(),
    }
}

fn analyze_reloc_expr<SI>(expr: ParsedExpr<String, SI>) -> Result<RelocExpr<SI>, Diagnostic<SI>> {
    match expr.node {
        ExprNode::Ident(ident) => Ok(RelocExpr::Symbol(ident, expr.interval)),
        ExprNode::Literal(Literal::Number(n)) => Ok(RelocExpr::Literal(n, expr.interval)),
        _ => panic!(),
    }
}

fn analyze_keyword_operand<R>(
    (keyword, range): (OperandKeyword, R),
    context: Context,
) -> Operand<R> {
    use self::Context::*;
    use frontend::syntax::keyword::OperandKeyword::*;
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
    fn analyze_deref_af() {
        let parsed_expr = ParsedExpr {
            node: ExprNode::Deref(Box::new(ParsedExpr {
                node: ExprNode::Literal(Literal::Operand(OperandKeyword::Af)),
                interval: 0,
            })),
            interval: 1,
        };
        assert_eq!(
            analyze_operand(parsed_expr, Context::Other),
            Err(Diagnostic::new(Message::CannotDereference, 1))
        )
    }
}
