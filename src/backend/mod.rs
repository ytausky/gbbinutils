use diagnostics::*;

pub trait Backend {
    type CodeRef;
    fn add_label(&mut self, label: (&str, Self::CodeRef));
    fn emit_item(&mut self, item: Item);
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Byte(Expr),
    Instruction(Instruction),
}

mod codegen;

pub struct Rom<'a, T: 'a> {
    data: Vec<u8>,
    counter: usize,
    diagnostics: &'a T,
}

impl<'a, T: 'a> Rom<'a, T> {
    pub fn new(diagnostics: &'a T) -> Rom<'a, T> {
        Rom {
            data: vec![0x00; 0x8000],
            counter: 0,
            diagnostics,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        self.data.as_slice()
    }
}

impl<'a, T: 'a + DiagnosticsListener> Backend for Rom<'a, T> {
    type CodeRef = ();

    fn add_label(&mut self, _label: (&str, Self::CodeRef)) {}

    fn emit_item(&mut self, item: Item) {
        match item {
            Item::Byte(Expr::Literal(n)) => {
                if !is_in_byte_range(n) {
                    self.diagnostics
                        .emit_diagnostic(Diagnostic::ValueOutOfRange {
                            value: n,
                            width: Width::Byte,
                        })
                }
                self.data[self.counter] = n as u8;
                self.counter += 1
            }
            _ => panic!(),
        }
    }
}

fn is_in_byte_range(n: i32) -> bool {
    is_in_i8_range(n) || is_in_u8_range(n)
}

fn is_in_i8_range(n: i32) -> bool {
    n >= i32::from(i8::min_value()) && n <= i32::from(i8::max_value())
}

fn is_in_u8_range(n: i32) -> bool {
    n >= i32::from(u8::min_value()) && n <= i32::from(u8::max_value())
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
        let diagnostics = TestDiagnosticsListener::new();
        let rom = Rom::new(&diagnostics);
        assert!(
            iter::repeat(0x00)
                .take(0x8000)
                .eq(rom.as_slice().iter().cloned())
        )
    }

    #[test]
    fn emit_literal_byte_item() {
        let diagnostics = TestDiagnosticsListener::new();
        let mut rom = Rom::new(&diagnostics);
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
        let diagnostics = TestDiagnosticsListener::new();
        let mut rom = Rom::new(&diagnostics);
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

    #[test]
    fn emit_diagnostic_when_byte_item_out_of_range() {
        test_diagnostic_for_out_of_range_byte(i8::min_value() as i32 - 1);
        test_diagnostic_for_out_of_range_byte(u8::max_value() as i32 + 1)
    }

    fn test_diagnostic_for_out_of_range_byte(value: i32) {
        let listener = TestDiagnosticsListener::new();
        let mut rom = Rom::new(&listener);
        rom.emit_item(Item::Byte(Expr::Literal(value)));
        assert_eq!(
            *listener.diagnostics.borrow(),
            [
                Diagnostic::ValueOutOfRange {
                    value: value,
                    width: Width::Byte,
                }
            ]
        );
    }

    use std::cell::RefCell;

    struct TestDiagnosticsListener {
        diagnostics: RefCell<Vec<Diagnostic<()>>>,
    }

    impl TestDiagnosticsListener {
        fn new() -> TestDiagnosticsListener {
            TestDiagnosticsListener {
                diagnostics: RefCell::new(Vec::new()),
            }
        }
    }

    impl DiagnosticsListener for TestDiagnosticsListener {
        type CodeRef = ();
        fn emit_diagnostic(&self, diagnostic: Diagnostic<Self::CodeRef>) {
            self.diagnostics.borrow_mut().push(diagnostic)
        }
    }
}
