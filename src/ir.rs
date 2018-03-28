pub trait Section {
    fn add_instruction(&mut self, instruction: Instruction);
    fn add_label(&mut self, label: &str);
}

#[derive(Debug, PartialEq)]
pub enum Instruction {
    Alu(AluOperation, AluSource),
    Dec(SimpleOperand),
    Halt,
    Jr(Option<Condition>, Expr),
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

#[derive(Debug, PartialEq)]
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

#[derive(Debug, PartialEq)]
pub enum LdKind {
    Simple(SimpleOperand, SimpleOperand),
    ImmediateAddr(Expr, Direction),
}

#[derive(Debug, PartialEq)]
pub enum Direction {
    FromA,
    IntoA,
}

#[derive(Debug, PartialEq)]
pub enum Reg16 {
    Bc,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Condition {
    C,
    Nc,
    Nz,
    Z,
}

#[derive(Debug, PartialEq)]
pub enum Expr {
    Literal(isize),
    Symbol(String),
}

pub struct DumpSection;

impl DumpSection {
    pub fn new() -> DumpSection {
        DumpSection {}
    }
}

impl Section for DumpSection {
    fn add_instruction(&mut self, instruction: Instruction) {
        println!("{:?}", instruction)
    }

    fn add_label(&mut self, label: &str) {
        println!("Define symbol: {}", label)
    }
}
