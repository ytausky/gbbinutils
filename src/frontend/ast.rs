use std::marker::PhantomData;

#[derive(Clone, Debug, PartialEq)]
pub enum AsmItem<'a> {
    Include(&'a str),
}

#[derive(Clone, Debug, PartialEq)]
pub enum SynExpr<T> {
    Atom(T),
    Deref(Box<SynExpr<T>>),
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
    type Expr = SynExpr<Self::Terminal>;

    fn from_atom(&mut self, atom: Self::Terminal) -> Self::Expr {
        SynExpr::Atom(atom)
    }

    fn apply_deref(&mut self, expr: Self::Expr) -> Self::Expr {
        SynExpr::Deref(Box::new(expr))
    }
}
