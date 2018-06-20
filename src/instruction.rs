use diagnostics::{Source, SourceInterval};

#[derive(Clone, Debug, PartialEq)]
pub enum Instruction<R> {
    AddHl(Reg16),
    Alu(AluOperation, AluSource<R>),
    Bit(BitOperation, RelocExpr<R>, SimpleOperand),
    IncDec8(IncDec, SimpleOperand),
    IncDec16(IncDec, Reg16),
    JpDerefHl,
    Branch(Branch<R>, Option<Condition>),
    Ld(Ld<R>),
    Nullary(Nullary),
    Pop(RegPair),
    Push(RegPair),
    Rst(RelocExpr<R>),
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
    Adc,
    And,
    Cp,
    Xor,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AluSource<R> {
    Simple(SimpleOperand),
    Immediate(RelocExpr<R>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BitOperation {
    Bit,
    Set,
    Res,
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
    Immediate8(SimpleOperand, RelocExpr<R>),
    Immediate16(Reg16, RelocExpr<R>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpecialLd<R> {
    DerefPtrReg(PtrReg),
    InlineAddr(RelocExpr<R>),
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
pub enum Branch<R> {
    Call(RelocExpr<R>),
    Jp(RelocExpr<R>),
    Jr(RelocExpr<R>),
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
pub enum RelocExpr<R> {
    Literal(i32, R),
    LocationCounter,
    Subtract(Box<RelocExpr<R>>, Box<RelocExpr<R>>),
    Symbol(String, R),
}

impl<SI: SourceInterval> Source for RelocExpr<SI> {
    type Interval = SI;
    fn source_interval(&self) -> Self::Interval {
        match self {
            RelocExpr::Literal(_, interval) | RelocExpr::Symbol(_, interval) => (*interval).clone(),
            RelocExpr::LocationCounter | RelocExpr::Subtract(..) => panic!(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IncDec {
    Inc,
    Dec,
}
