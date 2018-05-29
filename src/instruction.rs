#[derive(Clone, Debug, PartialEq)]
pub enum Instruction<R> {
    AddHl(Reg16),
    Alu(AluOperation, AluSource<R>),
    IncDec8(IncDec, SimpleOperand),
    IncDec16(IncDec, Reg16),
    JpDerefHl,
    Branch(Branch<R>, Option<Condition>),
    Ld(Ld<R>),
    Nullary(Nullary),
    Pop(RegPair),
    Push(RegPair),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Nullary {
    Daa,
    Di,
    Ei,
    Halt,
    Nop,
    Reti,
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AluOperation {
    Add,
    And,
    Cp,
    Xor,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AluSource<R> {
    Simple(SimpleOperand),
    Immediate(Expr<R>),
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
pub enum Ld<R> {
    Simple(SimpleOperand, SimpleOperand),
    Special(SpecialLd<R>, Direction),
    Immediate8(SimpleOperand, Expr<R>),
    Immediate16(Reg16, Expr<R>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpecialLd<R> {
    InlineAddr(Expr<R>),
    InlineIndex(Expr<R>),
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

#[derive(Clone, Debug, PartialEq)]
pub enum Branch<R> {
    Call(Expr<R>),
    Jp(Expr<R>),
    Jr(Expr<R>),
    Ret,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Condition {
    C,
    Nc,
    Nz,
    Z,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr<R> {
    Literal(i32, R),
    LocationCounter,
    Subtract(Box<Expr<R>>, Box<Expr<R>>),
    Symbol(String, R),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IncDec {
    Inc,
    Dec,
}
