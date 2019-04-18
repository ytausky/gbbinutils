use crate::analysis::{Ident, Literal};
use crate::diag::span::Source;
use crate::model::BinOp;

pub(super) type CommandArgs<I, S> = Vec<SemanticExpr<I, S>>;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct SemanticExpr<I, S> {
    pub variant: ExprVariant<I, S>,
    pub span: S,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ExprVariant<I, S> {
    Atom(SemanticAtom<I>),
    Unary(SemanticUnary, Box<SemanticExpr<I, S>>),
    Binary(BinOp, Box<SemanticExpr<I, S>>, Box<SemanticExpr<I, S>>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum SemanticAtom<I> {
    Ident(Ident<I>),
    Literal(Literal<I>),
    LocationCounter,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum SemanticUnary {
    Parentheses,
}

impl<I> From<Literal<I>> for SemanticAtom<I> {
    fn from(literal: Literal<I>) -> Self {
        SemanticAtom::Literal(literal)
    }
}

#[cfg(test)]
impl<I, S> SemanticExpr<I, S> {
    pub fn from_atom<T: Into<ExprVariant<I, S>>>(atom: T, span: S) -> Self {
        Self {
            variant: atom.into(),
            span,
        }
    }
}

impl<I, S> From<SemanticAtom<I>> for ExprVariant<I, S> {
    fn from(atom: SemanticAtom<I>) -> Self {
        ExprVariant::Atom(atom)
    }
}

impl<I, S: Clone> Source for SemanticExpr<I, S> {
    type Span = S;

    fn span(&self) -> Self::Span {
        self.span.clone()
    }
}
