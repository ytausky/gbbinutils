use Width;
use backend::codegen::{DataItem, Emit};
use diagnostics::*;
use std::collections::HashMap;

pub trait Backend<R> {
    type Object;
    fn add_label(&mut self, label: (impl Into<String>, R));
    fn emit_item(&mut self, item: Item<R>);
    fn into_object(self) -> Self::Object;
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item<R> {
    Data(Expr<R>, Width),
    Instruction(Instruction<R>),
}

enum Data {
    Byte(u8),
    Word(u16),
}

mod codegen;

pub struct Object {
    resolved_sections: Vec<ResolvedSection>,
}

impl Object {
    fn new() -> Object {
        Object {
            resolved_sections: Vec::new(),
        }
    }
}

struct SymbolTable<'a, D: 'a> {
    symbols: HashMap<String, i32>,
    diagnostics: &'a D,
}

impl<'a, D: 'a> SymbolTable<'a, D> {
    fn new(diagnostics: &'a D) -> SymbolTable<'a, D> {
        SymbolTable {
            symbols: HashMap::new(),
            diagnostics,
        }
    }

    fn resolve_expr_item<R: Clone>(&self, expr: Expr<R>, width: Width) -> Data
    where
        D: DiagnosticsListener<R>,
    {
        let value = self.evaluate_expr(expr);
        self.fit_to_width(value, width)
    }

    fn evaluate_expr<R: Clone>(&self, expr: Expr<R>) -> (i32, R)
    where
        D: DiagnosticsListener<R>,
    {
        match expr {
            Expr::Literal(value, expr_ref) => (value, expr_ref),
            Expr::Symbol(symbol, expr_ref) => {
                let symbol_def = self.symbols.get(&symbol).cloned();
                if let Some(value) = symbol_def {
                    (value, expr_ref)
                } else {
                    self.diagnostics.emit_diagnostic(Diagnostic::new(
                        Message::UnresolvedSymbol {
                            symbol: symbol.clone(),
                        },
                        expr_ref.clone(),
                    ));
                    (0, expr_ref)
                }
            }
        }
    }

    fn fit_to_width<R: Clone>(&self, (value, value_ref): (i32, R), width: Width) -> Data
    where
        D: DiagnosticsListener<R>,
    {
        if !is_in_range(value, width) {
            self.diagnostics.emit_diagnostic(Diagnostic::new(
                Message::ValueOutOfRange { value, width },
                value_ref.clone(),
            ))
        }
        match width {
            Width::Byte => Data::Byte(value as u8),
            Width::Word => Data::Word(value as u16),
        }
    }
}

pub struct ObjectBuilder<'a, R, D: DiagnosticsListener<R> + 'a> {
    pending_sections: Vec<PendingSection<R>>,
    symbol_table: SymbolTable<'a, D>,
}

impl<'a, R: Clone, D: DiagnosticsListener<R> + 'a> ObjectBuilder<'a, R, D> {
    pub fn new(diagnostics: &D) -> ObjectBuilder<R, D> {
        ObjectBuilder {
            pending_sections: vec![PendingSection::new()],
            symbol_table: SymbolTable::new(diagnostics),
        }
    }

    pub fn resolve_symbols(self) -> Object {
        let mut object = Object::new();
        for pending_section in self.pending_sections {
            let mut resolved_section = ResolvedSection::new();
            for item in pending_section.items {
                match item {
                    DataItem::Byte(value) => resolved_section.push(Data::Byte(value)),
                    DataItem::Expr(expr, width) => {
                        resolved_section.push(self.symbol_table.resolve_expr_item(expr, width))
                    }
                }
            }
            object.resolved_sections.push(resolved_section)
        }
        object
    }

    fn emit_data_expr(&mut self, expr: Expr<R>, width: Width) {
        self.pending_sections
            .last_mut()
            .unwrap()
            .push(DataItem::Expr(expr, width))
    }

    fn emit_instruction(&mut self, instruction: &Instruction<R>) {
        codegen::generate_code(&instruction, self)
    }
}

impl<'a, R: Clone, D: DiagnosticsListener<R> + 'a> Backend<R> for ObjectBuilder<'a, R, D> {
    type Object = Object;

    fn add_label(&mut self, label: (impl Into<String>, R)) {
        self.symbol_table.symbols.insert(
            label.0.into(),
            self.pending_sections.last().unwrap().len as i32,
        );
    }

    fn emit_item(&mut self, item: Item<R>) {
        match item {
            Item::Data(expr, width) => self.emit_data_expr(expr, width),
            Item::Instruction(instruction) => self.emit_instruction(&instruction),
        }
    }

    fn into_object(self) -> Self::Object {
        self.resolve_symbols()
    }
}

impl<'a, R, D: DiagnosticsListener<R> + 'a> Emit<R> for ObjectBuilder<'a, R, D> {
    fn emit(&mut self, item: DataItem<R>) {
        self.pending_sections.last_mut().unwrap().push(item)
    }
}

struct PendingSection<R> {
    items: Vec<DataItem<R>>,
    len: usize,
}

impl<R> DataItem<R> {
    fn len(&self) -> usize {
        match self {
            DataItem::Byte(_) => 1,
            DataItem::Expr(_, width) => width.len(),
        }
    }
}

impl<R> PendingSection<R> {
    fn new() -> PendingSection<R> {
        PendingSection {
            items: Vec::new(),
            len: 0,
        }
    }

    fn push(&mut self, item: DataItem<R>) {
        self.len += item.len();
        self.items.push(item)
    }
}

struct ResolvedSection {
    data: Vec<u8>,
}

impl ResolvedSection {
    fn new() -> ResolvedSection {
        ResolvedSection { data: Vec::new() }
    }

    fn push(&mut self, data: Data) {
        match data {
            Data::Byte(value) => self.data.push(value),
            Data::Word(value) => {
                let low = value & 0xff;
                let high = (value >> 8) & 0xff;
                self.data.push(low as u8);
                self.data.push(high as u8);
            }
        }
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
        let object = builder.resolve_symbols();
        assert_eq!(
            object.resolved_sections.last().unwrap().data,
            bytes.borrow()
        )
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
        builder.resolve_symbols();
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
        let object = builder.resolve_symbols();
        assert_eq!(*diagnostics.diagnostics.borrow(), []);
        assert_eq!(object.resolved_sections.last().unwrap().data, [0x00, 0x00])
    }

    #[test]
    fn emit_symbol_defined_after_use() {
        let label = "label";
        let diagnostics = TestDiagnosticsListener::new();
        let mut builder = ObjectBuilder::new(&diagnostics);
        builder.emit_item(Item::Data(Expr::Symbol(label.to_string(), ()), Width::Word));
        builder.add_label((label, ()));
        let object = builder.resolve_symbols();
        assert_eq!(*diagnostics.diagnostics.borrow(), []);
        assert_eq!(object.resolved_sections.last().unwrap().data, [0x02, 0x00])
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
