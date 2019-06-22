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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Width {
    Byte,
    Word,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item<V: Source> {
    Data(V, Width),
    Instruction(Instruction<V>),
}

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

#[derive(Clone, Debug, PartialEq)]
pub enum Instruction<V: Source> {
    AddHl(Reg16),
    Alu(AluOperation, AluSource<V>),
    Bit(BitOperation, V, SimpleOperand),
    IncDec8(IncDec, SimpleOperand),
    IncDec16(IncDec, Reg16),
    JpDerefHl,
    Branch(Branch<V>, Option<Condition>),
    Ld(Ld<V>),
    Ldhl(V),
    Misc(MiscOperation, SimpleOperand),
    Nullary(Nullary),
    Pop(RegPair),
    Push(RegPair),
    Rst(V),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Nullary {
    Cpl,
    Daa,
    Di,
    Ei,
    Halt,
    Nop,
    Reti,
    Rla,
    Rlca,
    Rra,
    Rrca,
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AluOperation {
    Add,
    Adc,
    Sub,
    Sbc,
    And,
    Xor,
    Or,
    Cp,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AluSource<V> {
    Simple(SimpleOperand),
    Immediate(V),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BitOperation {
    Bit,
    Set,
    Res,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MiscOperation {
    Rlc,
    Rrc,
    Rl,
    Rr,
    Sla,
    Sra,
    Swap,
    Srl,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimpleOperand {
    A,
    B,
    C,
    D,
    E,
    H,
    L,
    DerefHl,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ld<V> {
    Simple(SimpleOperand, SimpleOperand),
    Special(SpecialLd<V>, Direction),
    SpHl,
    Immediate8(SimpleOperand, V),
    Immediate16(Reg16, V),
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpecialLd<V> {
    DerefPtrReg(PtrReg),
    InlineAddr(V),
    RegIndex,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    FromA,
    IntoA,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reg16 {
    Bc,
    De,
    Hl,
    Sp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegPair {
    Bc,
    De,
    Hl,
    Af,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PtrReg {
    Bc,
    De,
    Hli,
    Hld,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Branch<V> {
    Call(V),
    Jp(V),
    Jr(V),
    Ret,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Condition {
    C,
    Nc,
    Nz,
    Z,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IncDec {
    Inc,
    Dec,
}
