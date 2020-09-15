use crate::expr::Expr;

pub(super) enum ParsedArg<N, R, S> {
    Bare(Expr<N, S>),
    Parenthesized(Expr<N, S>, S),
    String(R, S),
    Error,
}

pub(crate) enum Arg<N, R, S> {
    Bare(BareArg<N, S>),
    Deref(BareArg<N, S>, S),
    String(R, S),
    Error,
}

#[derive(Clone)]
pub(crate) enum BareArg<N, S> {
    Const(Expr<N, S>),
    Symbol(OperandSymbol, S),
}

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
