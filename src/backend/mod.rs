use diagnostics::*;

pub trait Backend<R> {
    fn add_label(&mut self, label: (&str, R));
    fn emit_item(&mut self, item: Item<R>);
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item<R> {
    Byte(Expr<R>),
    Instruction(Instruction<R>),
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

    fn emit_byte<R>(&mut self, expr: Expr<R>)
    where
        T: DiagnosticsListener<R>,
    {
        match expr {
            Expr::Literal(n, byte_ref) => {
                if !is_in_byte_range(n) {
                    self.diagnostics.emit_diagnostic(Diagnostic::new(
                        Message::ValueOutOfRange {
                            value: n,
                            width: Width::Byte,
                        },
                        byte_ref,
                    ))
                }
                self.data[self.counter] = n as u8;
                self.counter += 1
            }
            _ => unimplemented!(),
        }
    }

    fn emit_instruction<R>(&mut self, instruction: Instruction<R>) {
        codegen::generate_code(instruction, |byte| {
            self.data[self.counter] = byte;
            self.counter += 1
        })
    }
}

impl<'a, R, T: 'a + DiagnosticsListener<R>> Backend<R> for Rom<'a, T> {
    fn add_label(&mut self, _label: (&str, R)) {}

    fn emit_item(&mut self, item: Item<R>) {
        match item {
            Item::Byte(expr) => self.emit_byte(expr),
            Item::Instruction(instruction) => self.emit_instruction(instruction),
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
pub enum Instruction<R> {
    Alu(AluOperation, AluSource<R>),
    Dec(SimpleOperand),
    Halt,
    Branch(Branch<R>, Option<Condition>),
    Ld(LdKind<R>),
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
pub enum LdKind<R> {
    Simple(SimpleOperand, SimpleOperand),
    Immediate16(Reg16, Expr<R>),
    ImmediateAddr(Expr<R>, Direction),
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
pub enum Branch<R> {
    Jp(Expr<R>),
    Jr(Expr<R>),
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
    Symbol(String, R),
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
        rom.emit_item(Item::Byte(Expr::Literal(0xff, ())));
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
        rom.emit_item(Item::Byte(Expr::Literal(0x12, ())));
        rom.emit_item(Item::Byte(Expr::Literal(0x34, ())));
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
    fn emit_stop() {
        let diagnostics = TestDiagnosticsListener::new();
        let mut rom = Rom::new(&diagnostics);
        rom.emit_item(Item::Instruction(Instruction::Stop));
        assert!(
            [0x10, 0x00]
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
        rom.emit_item(Item::Byte(Expr::Literal(value, ())));
        assert_eq!(
            *listener.diagnostics.borrow(),
            [
                Diagnostic::new(
                    Message::ValueOutOfRange {
                        value: value,
                        width: Width::Byte,
                    },
                    ()
                )
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

    impl DiagnosticsListener<()> for TestDiagnosticsListener {
        fn emit_diagnostic(&self, diagnostic: Diagnostic<()>) {
            self.diagnostics.borrow_mut().push(diagnostic)
        }
    }
}
