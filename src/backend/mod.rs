use Width;
use backend::codegen::ByteEmitter;
use diagnostics::*;
use std::collections::HashMap;

pub trait Backend<R> {
    fn add_label(&mut self, label: (impl Into<String>, R));
    fn emit_item(&mut self, item: Item<R>);
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item<R> {
    Data(Expr<R>, Width),
    Instruction(Instruction<R>),
}

mod codegen;

pub struct ObjectBuilder<'a, T: 'a> {
    sections: Vec<Section>,
    symbols: HashMap<String, i32>,
    diagnostics: &'a T,
}

impl<'a, T: 'a> ObjectBuilder<'a, T> {
    pub fn new(diagnostics: &T) -> ObjectBuilder<T> {
        ObjectBuilder {
            sections: vec![Section::new()],
            symbols: HashMap::new(),
            diagnostics,
        }
    }

    #[cfg(test)]
    pub fn resolve_symbols(&mut self) {}

    fn emit_data_expr<R>(&mut self, expr: Expr<R>, width: Width)
    where
        T: DiagnosticsListener<R>,
    {
        match expr {
            Expr::Literal(value, expr_ref) => self.emit_resolved_data_expr(value, width, expr_ref),
            Expr::Symbol(symbol, expr_ref) => {
                let symbol_def = self.symbols.get(&symbol).cloned();
                if let Some(value) = symbol_def {
                    self.emit_resolved_data_expr(value, width, expr_ref)
                } else {
                    self.diagnostics.emit_diagnostic(Diagnostic::new(
                        Message::UnresolvedSymbol { symbol },
                        expr_ref,
                    ))
                }
            }
        }
    }

    fn emit_resolved_data_expr<R>(&mut self, value: i32, width: Width, expr_ref: R)
    where
        T: DiagnosticsListener<R>,
    {
        if !is_in_range(value, width) {
            self.diagnostics.emit_diagnostic(Diagnostic::new(
                Message::ValueOutOfRange { value, width },
                expr_ref,
            ))
        }
        self.emit_data(value, width)
    }

    fn emit_data(&mut self, value: i32, width: Width) {
        let low = value & 0xff;
        self.emit_byte(low as u8);
        if width == Width::Word {
            let high = (value >> 8) & 0xff;
            self.emit_byte(high as u8)
        }
    }

    fn emit_instruction<R>(&mut self, instruction: &Instruction<R>) {
        codegen::generate_code(&instruction, self)
    }
}

impl<'a, R, T: DiagnosticsListener<R> + 'a> Backend<R> for ObjectBuilder<'a, T> {
    fn add_label(&mut self, label: (impl Into<String>, R)) {
        self.symbols.insert(
            label.0.into(),
            self.sections.last().unwrap().data.len() as i32,
        );
    }

    fn emit_item(&mut self, item: Item<R>) {
        match item {
            Item::Data(expr, width) => self.emit_data_expr(expr, width),
            Item::Instruction(instruction) => self.emit_instruction(&instruction),
        }
    }
}

impl<'a, T: 'a> ByteEmitter for ObjectBuilder<'a, T> {
    fn emit_byte(&mut self, value: u8) {
        self.sections.last_mut().unwrap().data.push(value)
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

fn is_in_range(n: i32, width: Width) -> bool {
    match width {
        Width::Byte => is_in_byte_range(n),
        Width::Word => true,
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

    use std::borrow::Borrow;

    #[test]
    fn emit_literal_byte_item() {
        emit_items_and_compare([Item::Data(Expr::Literal(0xff, ()), Width::Byte)], [0xff])
    }

    #[test]
    fn emit_two_literal_byte_item() {
        emit_items_and_compare(
            [
                Item::Data(Expr::Literal(0x12, ()), Width::Byte),
                Item::Data(Expr::Literal(0x34, ()), Width::Byte),
            ],
            [0x12, 0x34],
        )
    }

    #[test]
    fn emit_stop() {
        emit_items_and_compare([Item::Instruction(Instruction::Stop)], [0x10, 0x00])
    }

    fn emit_items_and_compare<I, B>(items: I, bytes: B)
    where
        I: Borrow<[Item<()>]>,
        B: Borrow<[u8]>,
    {
        let diagnostics = TestDiagnosticsListener::new();
        let mut builder = ObjectBuilder::new(&diagnostics);
        for item in items.borrow() {
            builder.emit_item(item.clone())
        }
        assert_eq!(builder.sections.last_mut().unwrap().data, bytes.borrow())
    }

    #[test]
    fn emit_diagnostic_when_byte_item_out_of_range() {
        test_diagnostic_for_out_of_range_byte(i8::min_value() as i32 - 1);
        test_diagnostic_for_out_of_range_byte(u8::max_value() as i32 + 1)
    }

    fn test_diagnostic_for_out_of_range_byte(value: i32) {
        let listener = TestDiagnosticsListener::new();
        let mut builder = ObjectBuilder::new(&listener);
        builder.emit_item(Item::Data(Expr::Literal(value, ()), Width::Byte));
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

    #[test]
    fn diagnose_unresolved_symbol() {
        let ident = "ident";
        let diagnostics = TestDiagnosticsListener::new();
        let mut builder = ObjectBuilder::new(&diagnostics);
        builder.emit_item(Item::Data(Expr::Symbol(ident.to_string(), ()), Width::Word));
        builder.resolve_symbols();
        assert_eq!(
            *diagnostics.diagnostics.borrow(),
            [
                Diagnostic::new(
                    Message::UnresolvedSymbol {
                        symbol: ident.to_string()
                    },
                    ()
                )
            ]
        );
    }

    #[test]
    fn emit_defined_symbol() {
        let label = "label";
        let diagnostics = TestDiagnosticsListener::new();
        let mut builder = ObjectBuilder::new(&diagnostics);
        builder.add_label((label, ()));
        builder.emit_item(Item::Data(Expr::Symbol(label.to_string(), ()), Width::Word));
        builder.resolve_symbols();
        assert_eq!(*diagnostics.diagnostics.borrow(), []);
        assert_eq!(builder.sections.last_mut().unwrap().data, [0x00, 0x00])
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
