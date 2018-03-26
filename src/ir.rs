pub trait Section {
    fn add_instruction(&mut self, instruction: Instruction);
    fn add_label(&mut self, label: &str);
}

#[derive(Debug, PartialEq)]
pub enum Instruction {
    Halt,
    LdAluAlu(AluOperand, AluOperand),
    LdDerefImm16A(Expr),
    Nop,
    Push(Reg16),
    Stop,
    Xor(AluOperand),
}

#[derive(Clone, Debug, PartialEq)]
pub enum AluOperand {
    A,
    B,
    #[cfg(test)] C,
    #[cfg(test)] D,
    #[cfg(test)] E,
    #[cfg(test)] H,
    #[cfg(test)] L,
    DerefHl,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Reg16 {
    Bc,
}

#[derive(Clone, Debug, PartialEq)]
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
