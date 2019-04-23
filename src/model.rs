use crate::span::Source;

#[derive(Clone, Debug, PartialEq)]
pub struct Expr<T, S>(pub Vec<ExprItem<T, S>>);

#[derive(Clone, Debug, PartialEq)]
pub struct ExprItem<T, S> {
    pub op: ExprOp<T>,
    pub op_span: S,
    pub expr_span: S,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprOp<T> {
    Atom(T),
    Binary(BinOp),
}

impl<T, S> Default for Expr<T, S> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T, S: Clone> Source for Expr<T, S> {
    type Span = S;

    fn span(&self) -> Self::Span {
        self.0.first().unwrap().expr_span.clone()
    }
}

#[cfg(test)]
impl<L, N, S: Clone> Expr<Atom<L, N>, S> {
    pub fn from_atom(atom: Atom<L, N>, span: S) -> Self {
        Self(vec![ExprItem {
            op: ExprOp::Atom(atom),
            op_span: span.clone(),
            expr_span: span,
        }])
    }
}

#[cfg(test)]
impl<N: Clone, S: Clone> Expr<N, S> {
    pub fn from_items<'a, I>(items: I) -> Self
    where
        N: 'a,
        S: 'a,
        I: IntoIterator<Item = &'a ExprItem<N, S>>,
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

impl<N, T: Into<ExprOp<N>>> From<T> for ExprItem<N, ()> {
    fn from(x: T) -> Self {
        Self {
            op: x.into(),
            op_span: (),
            expr_span: (),
        }
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

impl<T> From<BinOp> for ExprOp<T> {
    fn from(op: BinOp) -> Self {
        ExprOp::Binary(op)
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
