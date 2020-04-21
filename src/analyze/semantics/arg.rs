use crate::diag::span::{Source, SpanSource};

pub(super) enum Arg<V, R, S> {
    Bare(DerefableArg<V, S>),
    Deref(DerefableArg<V, S>, S),
    Error,
    String(R, S),
}

#[derive(Clone)]
pub(super) enum DerefableArg<V, S> {
    Const(V),
    Symbol(OperandSymbol, S),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) struct TreeArg<S> {
    pub variant: TreeArgVariant,
    pub span: S,
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum TreeArgVariant {
    Atom(TreeArgAtom),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum TreeArgAtom {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OperandSymbol {
    A,
    Af,
    B,
    Bc,
    C,
    D,
    De,
    E,
    H,
    Hl,
    Hld,
    Hli,
    L,
    Nc,
    Nz,
    Sp,
    Z,
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum ArgUnaryOp {}

impl From<TreeArgAtom> for TreeArgVariant {
    fn from(atom: TreeArgAtom) -> Self {
        TreeArgVariant::Atom(atom)
    }
}

impl<S: Clone> SpanSource for TreeArg<S> {
    type Span = S;
}

impl<S: Clone> Source for TreeArg<S> {
    fn span(&self) -> Self::Span {
        self.span.clone()
    }
}
