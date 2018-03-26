pub trait Section {
    fn add_instruction(&mut self, instruction: Instruction);
    fn add_label(&mut self, label: &str);
}

#[derive(Debug, PartialEq)]
pub enum Instruction {
    Alu(AluOperation, AluOperand),
    AluImm8(AluOperation, Expr),
    Halt,
    Jr(Condition, Expr),
    LdAluAlu(AluOperand, AluOperand),
    LdDerefImm16(Expr, Direction),
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AluOperand {
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
    Nz,
    Z,
}

#[derive(Debug, PartialEq)]
pub enum Expr {
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
