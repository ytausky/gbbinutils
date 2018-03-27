use frontend::syntax;

use std::marker::PhantomData;

#[derive(Clone, Debug, PartialEq)]
pub enum AsmItem<'a> {
    Include(&'a str),
}

pub trait ExprFactory {
    type Terminal;
    type Expr;
    fn from_atom(&mut self, atom: Self::Terminal) -> Self::Expr;
    fn apply_deref(&mut self, expr: Self::Expr) -> Self::Expr;
}

pub struct ExprBuilder<T>(PhantomData<T>);

impl<T> ExprBuilder<T> {
    pub fn new() -> ExprBuilder<T> {
        ExprBuilder(PhantomData)
    }
}

impl<T> ExprFactory for ExprBuilder<T> {
    type Terminal = T;
    type Expr = syntax::SynExpr<Self::Terminal>;

    fn from_atom(&mut self, atom: Self::Terminal) -> Self::Expr {
        syntax::SynExpr::Atom(atom)
    }

    fn apply_deref(&mut self, expr: Self::Expr) -> Self::Expr {
        syntax::SynExpr::Deref(Box::new(expr))
    }
}
