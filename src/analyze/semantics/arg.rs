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
