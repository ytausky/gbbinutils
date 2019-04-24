use crate::analysis::{Ident, Literal};
use crate::diag::span::Source;
use crate::model::BinOp;

pub(super) type CommandArgs<I, S> = Vec<Arg<I, S>>;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Arg<I, S> {
    pub variant: ArgVariant<I, S>,
    pub span: S,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ArgVariant<I, S> {
    Atom(ArgAtom<I>),
    Unary(ArgUnaryOp, Box<Arg<I, S>>),
    Binary(BinOp, Box<Arg<I, S>>, Box<Arg<I, S>>),
    FnCall((Ident<I>, S), Vec<Arg<I, S>>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ArgAtom<I> {
    Ident(Ident<I>),
    Literal(Literal<I>),
    LocationCounter,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ArgUnaryOp {
    Parentheses,
}

impl<I> From<Literal<I>> for ArgAtom<I> {
    fn from(literal: Literal<I>) -> Self {
        ArgAtom::Literal(literal)
    }
}

#[cfg(test)]
impl<I, S> Arg<I, S> {
    pub fn from_atom<T: Into<ArgVariant<I, S>>>(atom: T, span: S) -> Self {
        Arg {
            variant: atom.into(),
            span,
        }
    }
}

impl<I, S> From<ArgAtom<I>> for ArgVariant<I, S> {
    fn from(atom: ArgAtom<I>) -> Self {
        ArgVariant::Atom(atom)
    }
}

impl<I, S: Clone> Source for Arg<I, S> {
    type Span = S;

    fn span(&self) -> Self::Span {
        self.span.clone()
    }
}
