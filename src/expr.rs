use crate::span::{Source, SpanSource, Spanned, WithSpan};

#[derive(Clone, Debug, PartialEq)]
pub struct Expr<A, S>(pub Vec<Spanned<ExprOp<A>, S>>);

#[derive(Clone, Debug, PartialEq)]
pub enum ExprOp<A> {
    Atom(A),
    Binary(BinOp),
    FnCall(usize),
}

impl<L, N, S> Default for Expr<Atom<L, N>, S> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<L, N, S: Clone> SpanSource for Expr<Atom<L, N>, S> {
    type Span = S;
}

impl<L, N, S: Clone> Source for Expr<Atom<L, N>, S> {
    fn span(&self) -> Self::Span {
        self.0.last().unwrap().span.clone()
    }
}

#[cfg(test)]
impl<L, N, S: Clone> Expr<Atom<L, N>, S> {
    pub fn from_atom(atom: Atom<L, N>, span: S) -> Self {
        Self(vec![ExprOp::Atom(atom).with_span(span)])
    }
}

#[cfg(test)]
impl<L: Clone, N: Clone, S: Clone> Expr<Atom<L, N>, S> {
    pub fn from_items<'a, I>(items: I) -> Self
    where
        L: 'a,
        N: 'a,
        S: 'a,
        I: IntoIterator<Item = &'a Spanned<ExprOp<Atom<L, N>>, S>>,
    {
        Expr(items.into_iter().map(Clone::clone).collect())
    }
}

impl<L, N> From<i32> for Expr<Atom<L, N>, ()> {
    fn from(n: i32) -> Self {
        Atom::Const(n).into()
    }
}

impl<L, N> From<Atom<L, N>> for Expr<Atom<L, N>, ()> {
    fn from(atom: Atom<L, N>) -> Self {
        Self(vec![atom.into()])
    }
}

impl<L, N, T: Into<ExprOp<Atom<L, N>>>> From<T> for Spanned<ExprOp<Atom<L, N>>, ()> {
    fn from(x: T) -> Self {
        x.into().with_span(())
    }
}

impl<L, N> From<Atom<L, N>> for ExprOp<Atom<L, N>> {
    fn from(atom: Atom<L, N>) -> Self {
        ExprOp::Atom(atom)
    }
}

impl<L, N> From<i32> for ExprOp<Atom<L, N>> {
    fn from(n: i32) -> Self {
        ExprOp::Atom(Atom::Const(n))
    }
}

impl<L, N> From<ParamId> for ExprOp<Atom<L, N>> {
    fn from(id: ParamId) -> Self {
        ExprOp::Atom(Atom::Param(id))
    }
}

impl<L, N> From<BinOp> for ExprOp<Atom<L, N>> {
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
    BitwiseOr,
    Division,
    Minus,
    Multiplication,
    Plus,
}

pub struct FnCall(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub enum Atom<L, N> {
    Const(i32),
    Location(L),
    Name(N),
    Param(ParamId),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocationCounter;

impl<N> From<LocationCounter> for Atom<LocationCounter, N> {
    fn from(_: LocationCounter) -> Self {
        Atom::Location(LocationCounter)
    }
}

impl<N> From<LocationCounter> for ExprOp<Atom<LocationCounter, N>> {
    fn from(_: LocationCounter) -> Self {
        Atom::from(LocationCounter).into()
    }
}

impl<N> From<LocationCounter> for Expr<Atom<LocationCounter, N>, ()> {
    fn from(_: LocationCounter) -> Self {
        Expr(vec![LocationCounter.into()])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamId(pub usize);

impl<L, N> From<i32> for Atom<L, N> {
    fn from(n: i32) -> Self {
        Atom::Const(n)
    }
}

impl<L, N> From<ParamId> for Atom<L, N> {
    fn from(id: ParamId) -> Self {
        Atom::Param(id)
    }
}
