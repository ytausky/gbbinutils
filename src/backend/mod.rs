use backend::codegen::ByteEmitter;
use diagnostics::*;

pub trait Backend<R> {
    fn add_label(&mut self, label: (impl Into<String>, R));
    fn emit_item(&mut self, item: Item<R>);
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item<R> {
    Byte(Expr<R>),
    Instruction(Instruction<R>),
}

mod codegen;

pub struct ObjectBuilder<'a, T: 'a> {
    section: Option<Section>,
    diagnostics: &'a T,
}

impl<'a, T: 'a> ObjectBuilder<'a, T> {
    pub fn new(diagnostics: &T) -> ObjectBuilder<T> {
        ObjectBuilder {
            section: Some(Section::new()),
            diagnostics,
        }
    }

    fn emit_byte_expr<R>(&mut self, expr: Expr<R>)
    where
        T: DiagnosticsListener<R>,
    {
        match expr {
            Expr::Literal(value, expr_ref) => {
                self.emit_resolved_byte_expr(value, expr_ref)
            }
            _ => unimplemented!(),
        }
    }

    fn emit_resolved_byte_expr<R>(&mut self, value: i32, expr_ref: R)
    where
        T: DiagnosticsListener<R>
    {
        if !is_in_byte_range(value) {
            self.diagnostics.emit_diagnostic(Diagnostic::new(
                Message::ValueOutOfRange {
                    value,
                    width: Width::Byte,
                },
                expr_ref,
            ))
        }
        self.emit_byte(value as u8)
    }

    fn emit_instruction<R>(&mut self, instruction: &Instruction<R>) {
        codegen::generate_code(&instruction, self)
    }
}

impl<'a, R, T: DiagnosticsListener<R> + 'a> Backend<R> for ObjectBuilder<'a, T> {
    fn add_label(&mut self, _label: (impl Into<String>, R)) {
    }

    fn emit_item(&mut self, item: Item<R>) {
        match item {
            Item::Byte(expr) => self.emit_byte_expr(expr),
            Item::Instruction(instruction) => self.emit_instruction(&instruction),
        }
    }
}

impl<'a, T: 'a> ByteEmitter for ObjectBuilder<'a, T> {
    fn emit_byte(&mut self, value: u8) {
        self.section.as_mut().unwrap().data.push(value)
    }
}

struct Section {
    data: Vec<u8>,
}

impl Section {
    fn new() -> Section {
        Section { data: Vec::new() }
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

    #[test]
    fn emit_literal_byte_item() {
        let diagnostics = TestDiagnosticsListener::new();
        let mut builder = ObjectBuilder::new(&diagnostics);
        builder.emit_item(Item::Byte(Expr::Literal(0xff, ())));
        assert_eq!(builder.section.unwrap().data, [0xff])
    }

    #[test]
    fn emit_two_literal_byte_item() {
        let diagnostics = TestDiagnosticsListener::new();
        let mut builder = ObjectBuilder::new(&diagnostics);
        builder.emit_item(Item::Byte(Expr::Literal(0x12, ())));
        builder.emit_item(Item::Byte(Expr::Literal(0x34, ())));
        assert_eq!(builder.section.unwrap().data, [0x12, 0x34])
    }

    #[test]
    fn emit_stop() {
        let diagnostics = TestDiagnosticsListener::new();
        let mut builder = ObjectBuilder::new(&diagnostics);
        builder.emit_item(Item::Instruction(Instruction::Stop));
        assert_eq!(builder.section.unwrap().data, [0x10, 0x00])
    }

    #[test]
    fn emit_diagnostic_when_byte_item_out_of_range() {
        test_diagnostic_for_out_of_range_byte(i8::min_value() as i32 - 1);
        test_diagnostic_for_out_of_range_byte(u8::max_value() as i32 + 1)
    }

    fn test_diagnostic_for_out_of_range_byte(value: i32) {
        let listener = TestDiagnosticsListener::new();
        let mut builder = ObjectBuilder::new(&listener);
        builder.emit_item(Item::Byte(Expr::Literal(value, ())));
        assert_eq!(
            *listener.diagnostics.borrow(),
            [
                Diagnostic::new(
                    Message::ValueOutOfRange {
                        value,
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
