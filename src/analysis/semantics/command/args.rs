use super::OperandSymbol;

use crate::analysis::Literal;
use crate::diag::span::{Source, SpanSource};
use crate::model::BinOp;

pub(super) type CommandArgs<I, R, S> = Vec<Arg<I, R, S>>;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Arg<I, R, S> {
    pub variant: ArgVariant<I, R, S>,
    pub span: S,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ArgVariant<I, R, S> {
    Atom(ArgAtom<I, R>),
    Unary(ArgUnaryOp, Box<Arg<I, R, S>>),
    Binary(BinOp, Box<Arg<I, R, S>>, Box<Arg<I, R, S>>),
    FnCall((I, S), Vec<Arg<I, R, S>>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ArgAtom<I, R> {
    Ident(I),
    Literal(Literal<R>),
    LocationCounter,
    OperandSymbol(OperandSymbol),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ArgUnaryOp {
    Parentheses,
}

impl<I, R> From<Literal<R>> for ArgAtom<I, R> {
    fn from(literal: Literal<R>) -> Self {
        ArgAtom::Literal(literal)
    }
}

#[cfg(test)]
impl<I, R, S> Arg<I, R, S> {
    pub fn from_atom<T: Into<ArgVariant<I, R, S>>>(atom: T, span: S) -> Self {
        Arg {
            variant: atom.into(),
            span,
        }
    }
}

impl<I, R, S> From<ArgAtom<I, R>> for ArgVariant<I, R, S> {
    fn from(atom: ArgAtom<I, R>) -> Self {
        ArgVariant::Atom(atom)
    }
}

impl<I, R, S: Clone> SpanSource for Arg<I, R, S> {
    type Span = S;
}

impl<I, R, S: Clone> Source for Arg<I, R, S> {
    fn span(&self) -> Self::Span {
        self.span.clone()
    }
}
