pub trait Backend {
    type Object: Object;
    fn mk_object(&mut self) -> Self::Object;
}

pub trait Object {
    fn add_label(&mut self, label: &str);
    fn emit_item(&mut self, item: Item);
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Byte(Expr),
    Instruction(Instruction),
}

mod codegen;

pub struct RomGenerator;

impl RomGenerator {
    pub fn new() -> RomGenerator {
        RomGenerator {}
    }
}

impl Backend for RomGenerator {
    type Object = Rom;
    fn mk_object(&mut self) -> Self::Object {
        Rom::new()
    }
}

pub struct Rom {
    data: Vec<u8>,
}

impl Rom {
    pub fn new() -> Rom {
        Rom {
            data: vec![0x00; 0x8000],
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        self.data.as_slice()
    }
}

impl Object for Rom {
    fn add_label(&mut self, _label: &str) {}
    fn emit_item(&mut self, _item: Item) {}
}

#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    Alu(AluOperation, AluSource),
    Dec(SimpleOperand),
    Halt,
    Branch(Branch, Option<Condition>),
    Ld(LdKind),
    Nop,
    Push(Reg16),
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AluOperation {
    And,
    Cp,
    Xor,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AluSource {
    Simple(SimpleOperand),
    Immediate(Expr),
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
pub enum LdKind {
    Simple(SimpleOperand, SimpleOperand),
    Immediate16(Reg16, Expr),
    ImmediateAddr(Expr, Direction),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Direction {
    FromA,
    IntoA,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reg16 {
    Bc,
    Hl,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Branch {
    Jp(Expr),
    Jr(Expr),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Condition {
    C,
    Nc,
    Nz,
    Z,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Literal(isize),
    Symbol(String),
}
