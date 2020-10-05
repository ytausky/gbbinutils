use crate::span::{Source, SpanSource, Spanned, WithSpan};

#[derive(Clone, Debug, PartialEq)]
pub struct Expr<N, S>(pub Vec<Spanned<ExprOp<N>, S>>);

#[derive(Clone, Debug, PartialEq)]
pub enum ExprOp<N> {
    Atom(Atom<N>),
    Binary(BinOp),
    FnCall(usize),
}

impl<N, S> Default for Expr<N, S> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<N, S: Clone> SpanSource for Expr<N, S> {
    type Span = S;
}

impl<N, S: Clone> Source for Expr<N, S> {
    fn span(&self) -> Self::Span {
        self.0.last().unwrap().span.clone()
    }
}

#[cfg(test)]
impl<N, S: Clone> Expr<N, S> {
    pub fn from_atom(atom: Atom<N>, span: S) -> Self {
        Self(vec![ExprOp::Atom(atom).with_span(span)])
    }
}

#[cfg(test)]
impl<N: Clone, S: Clone> Expr<N, S> {
    pub fn from_items<'a, I>(items: I) -> Self
    where
        N: 'a,
        S: 'a,
        I: IntoIterator<Item = &'a Spanned<ExprOp<N>, S>>,
    {
        Expr(items.into_iter().map(Clone::clone).collect())
    }
}

impl<N> From<i32> for Expr<N, ()> {
    fn from(n: i32) -> Self {
        Atom::Const(n).into()
    }
}

impl<N> From<Atom<N>> for Expr<N, ()> {
    fn from(atom: Atom<N>) -> Self {
        Self(vec![atom.into()])
    }
}

impl<N, T: Into<ExprOp<N>>> From<T> for Spanned<ExprOp<N>, ()> {
    fn from(x: T) -> Self {
        x.into().with_span(())
    }
}

impl<N> From<Atom<N>> for ExprOp<N> {
    fn from(atom: Atom<N>) -> Self {
        ExprOp::Atom(atom)
    }
}

impl<N> From<i32> for ExprOp<N> {
    fn from(n: i32) -> Self {
        ExprOp::Atom(Atom::Const(n))
    }
}

impl<N> From<ParamId> for ExprOp<N> {
    fn from(id: ParamId) -> Self {
        ExprOp::Atom(Atom::Param(id))
    }
}

impl<N> From<BinOp> for ExprOp<N> {
    fn from(op: BinOp) -> Self {
        ExprOp::Binary(op)
    }
}

impl<A> From<FnCall> for ExprOp<A> {
    fn from(FnCall(n): FnCall) -> Self {
        ExprOp::FnCall(n)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinOp {
    BitOr,
    Division,
    Equality,
    Minus,
    Multiplication,
    Plus,
}

pub struct FnCall(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub enum Atom<N> {
    Const(i32),
    Location,
    Name(N),
    Param(ParamId),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocationCounter;

impl<N> From<LocationCounter> for Atom<N> {
    fn from(_: LocationCounter) -> Self {
        Atom::Location
    }
}

impl<N> From<LocationCounter> for ExprOp<N> {
    fn from(_: LocationCounter) -> Self {
        Atom::from(LocationCounter).into()
    }
}

impl<N> From<LocationCounter> for Expr<N, ()> {
    fn from(_: LocationCounter) -> Self {
        Expr(vec![LocationCounter.into()])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamId(pub usize);

impl<N> From<i32> for Atom<N> {
    fn from(n: i32) -> Self {
        Atom::Const(n)
    }
}

impl<N> From<ParamId> for Atom<N> {
    fn from(id: ParamId) -> Self {
        Atom::Param(id)
    }
}
