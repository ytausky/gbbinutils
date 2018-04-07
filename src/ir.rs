pub trait Object {
    fn add_instruction(&mut self, instruction: Instruction);
    fn add_label(&mut self, label: &str);
    fn emit_byte(&mut self, byte: u8);
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

pub struct OutputDumper;

impl OutputDumper {
    pub fn new() -> OutputDumper {
        OutputDumper {}
    }
}

impl Object for OutputDumper {
    fn add_instruction(&mut self, instruction: Instruction) {
        println!("{:?}", instruction)
    }

    fn add_label(&mut self, label: &str) {
        println!("Define symbol: {}", label)
    }

    fn emit_byte(&mut self, byte: u8) {
        println!("Emit byte: {:x}", byte)
    }
}
