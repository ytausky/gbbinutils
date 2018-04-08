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
    counter: usize,
}

impl Rom {
    pub fn new() -> Rom {
        Rom {
            data: vec![0x00; 0x8000],
            counter: 0,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        self.data.as_slice()
    }
}

impl Object for Rom {
    fn add_label(&mut self, _label: &str) {}

    fn emit_item(&mut self, item: Item) {
        match item {
            Item::Byte(Expr::Literal(n)) => {
                self.data[self.counter] = n as u8;
                self.counter += 1
            }
            _ => panic!(),
        }
    }
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
    Literal(i32),
    Symbol(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::iter;

    #[test]
    fn new_rom_is_0x8000_zero_bytes_long() {
        let rom = Rom::new();
        assert!(
            iter::repeat(0x00)
                .take(0x8000)
                .eq(rom.as_slice().iter().cloned())
        )
    }

    #[test]
    fn emit_literal_byte_item() {
        let mut rom = Rom::new();
        rom.emit_item(Item::Byte(Expr::Literal(0xff)));
        assert!(
            iter::once(0xff)
                .chain(iter::repeat(0x00))
                .take(0x8000)
                .eq(rom.as_slice().iter().cloned())
        )
    }

    #[test]
    fn emit_two_literal_byte_item() {
        let mut rom = Rom::new();
        rom.emit_item(Item::Byte(Expr::Literal(0x12)));
        rom.emit_item(Item::Byte(Expr::Literal(0x34)));
        assert!(
            [0x12, 0x34]
                .iter()
                .cloned()
                .chain(iter::repeat(0x00))
                .take(0x8000)
                .eq(rom.as_slice().iter().cloned())
        )
    }
}
