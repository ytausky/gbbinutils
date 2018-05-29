use frontend::ExprFactory;
use frontend::syntax::{keyword, Literal, ParsedExpr};
use instruction::{Condition, Expr, Reg16, RegPair, SimpleOperand};

#[derive(Debug, PartialEq)]
pub enum Operand<R> {
    Atom(AtomKind, R),
    Const(Expr<R>),
    Deref(Expr<R>),
}

#[derive(Debug, PartialEq)]
pub enum AtomKind {
    Simple(SimpleOperand),
    Condition(Condition),
    Reg16(Reg16),
    RegPair(RegPair),
}

pub struct OperandAnalyzer<'a, EF: 'a> {
    expr_factory: &'a mut EF,
}

pub enum Context {
    Branch,
    Stack,
    Other,
}

impl<'a, EF: 'a + ExprFactory> OperandAnalyzer<'a, EF> {
    pub fn new(expr_factory: &'a mut EF) -> OperandAnalyzer<'a, EF> {
        OperandAnalyzer { expr_factory }
    }

    pub fn analyze_operand<R>(
        &mut self,
        expr: ParsedExpr<String, R>,
        context: &Context,
    ) -> Operand<R> {
        match expr {
            ParsedExpr::Deref(expr) => self.analyze_deref_operand(*expr),
            ParsedExpr::Ident(ident) => self.analyze_ident_operand(ident),
            ParsedExpr::Literal(literal) => self.analyze_literal_operand(literal, context),
        }
    }

    fn analyze_ident_operand<R>(&mut self, ident: (String, R)) -> Operand<R> {
        Operand::Const(self.expr_factory.mk_symbol(ident))
    }

    fn analyze_literal_operand<R>(
        &mut self,
        literal: (Literal<String>, R),
        context: &Context,
    ) -> Operand<R> {
        match literal.0 {
            Literal::Operand(operand) => analyze_keyword_operand((operand, literal.1), context),
            Literal::Number(_) => Operand::Const(self.expr_factory.mk_literal(literal)),
            _ => panic!(),
        }
    }

    fn analyze_deref_operand<R>(&mut self, expr: ParsedExpr<String, R>) -> Operand<R> {
        match expr {
            ParsedExpr::Ident(ident) => Operand::Deref(self.expr_factory.mk_symbol(ident)),
            ParsedExpr::Literal((Literal::Operand(keyword::Operand::Hl), range)) => {
                Operand::Atom(AtomKind::Simple(SimpleOperand::DerefHl), range)
            }
            ParsedExpr::Literal(number @ (Literal::Number(_), _)) => {
                Operand::Deref(self.expr_factory.mk_literal(number))
            }
            _ => panic!(),
        }
    }
}

fn analyze_keyword_operand<R>(
    (keyword, range): (keyword::Operand, R),
    context: &Context,
) -> Operand<R> {
    use self::Context::*;
    use frontend::syntax::keyword::Operand::*;
    let kind = match keyword {
        A => AtomKind::Simple(SimpleOperand::A),
        Af => AtomKind::RegPair(RegPair::Af),
        B => AtomKind::Simple(SimpleOperand::B),
        Bc => match *context {
            Stack => AtomKind::RegPair(RegPair::Bc),
            _ => AtomKind::Reg16(Reg16::Bc),
        },
        C => match *context {
            Branch => AtomKind::Condition(Condition::C),
            _ => AtomKind::Simple(SimpleOperand::C),
        },
        D => AtomKind::Simple(SimpleOperand::D),
        De => match *context {
            Stack => AtomKind::RegPair(RegPair::De),
            _ => AtomKind::Reg16(Reg16::De),
        },
        E => AtomKind::Simple(SimpleOperand::E),
        H => AtomKind::Simple(SimpleOperand::H),
        Hl => match *context {
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
        self.operands.next().map(|x| {
            self.count += 1;
            x
        })
    }
}
