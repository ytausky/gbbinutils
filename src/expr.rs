use crate::span::Source;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinaryOperator {
    Minus,
    Plus,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr<A, U, B, S> {
    pub variant: ExprVariant<A, U, B, S>,
    pub span: S,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprVariant<A, U, B, S> {
    Atom(A),
    Unary(U, Box<Expr<A, U, B, S>>),
    Binary(B, Box<Expr<A, U, B, S>>, Box<Expr<A, U, B, S>>),
}

impl<A, U, B, S> Expr<A, U, B, S> {
    pub fn from_atom<T: Into<ExprVariant<A, U, B, S>>>(atom: T, span: S) -> Self {
        Expr {
            variant: atom.into(),
            span,
        }
    }
}

impl<A, U, B, S> From<A> for ExprVariant<A, U, B, S> {
    fn from(atom: A) -> Self {
        ExprVariant::Atom(atom)
    }
}

impl<A, U, B, S: Clone> Source for Expr<A, U, B, S> {
    type Span = S;

    fn span(&self) -> Self::Span {
        self.span.clone()
    }
}
