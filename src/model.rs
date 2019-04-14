use crate::span::Source;

#[derive(Clone, Debug, PartialEq)]
pub struct Expr<N, S>(pub Vec<ExprItem<N, S>>);

#[derive(Clone, Debug, PartialEq)]
pub struct ExprItem<N, S> {
    pub op: ExprOperator<N>,
    pub op_span: S,
    pub expr_span: S,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprOperator<N> {
    Atom(Atom<N>),
    Binary(BinOp),
}

impl<N, S> Default for Expr<N, S> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<N, S: Clone> Source for Expr<N, S> {
    type Span = S;

    fn span(&self) -> Self::Span {
        self.0.first().unwrap().expr_span.clone()
    }
}

#[cfg(test)]
impl<N, S: Clone> Expr<N, S> {
    pub fn from_atom(atom: Atom<N>, span: S) -> Self {
        Self(vec![ExprItem::from_atom(atom, span)])
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

#[cfg(test)]
impl<N, S: Clone> ExprItem<N, S> {
    pub fn from_atom(atom: Atom<N>, span: S) -> Self {
        Self {
            op: ExprOperator::Atom(atom),
            op_span: span.clone(),
            expr_span: span,
        }
    }
}

impl<N> From<i32> for Expr<N, ()> {
    fn from(n: i32) -> Self {
        Atom::Literal(n).into()
    }
}

impl<N> From<Atom<N>> for Expr<N, ()> {
    fn from(atom: Atom<N>) -> Self {
        Self(vec![atom.into()])
    }
}

impl<N, T: Into<ExprOperator<N>>> From<T> for ExprItem<N, ()> {
    fn from(x: T) -> Self {
        Self {
            op: x.into(),
            op_span: (),
            expr_span: (),
        }
    }
}

impl<N> From<Atom<N>> for ExprOperator<N> {
    fn from(atom: Atom<N>) -> Self {
        ExprOperator::Atom(atom)
    }
}

impl<N> From<i32> for ExprOperator<N> {
    fn from(n: i32) -> Self {
        ExprOperator::Atom(Atom::Literal(n))
    }
}

impl<N> From<BinOp> for ExprOperator<N> {
    fn from(op: BinOp) -> Self {
        ExprOperator::Binary(op)
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
pub enum Atom<N> {
    Name(N),
    Literal(i32),
    LocationCounter,
}

impl<N> From<i32> for Atom<N> {
    fn from(n: i32) -> Self {
        Atom::Literal(n)
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
