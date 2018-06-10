use backend::{lowering::Lower,
              section::{Node, PendingSection}};
use diagnostics::*;
use instruction::{Instruction, RelocExpr};
use std::{collections::HashMap, iter::FromIterator, ops::AddAssign};
use Width;

mod section;

pub trait Backend<R> {
    type Object;
    fn add_label(&mut self, label: (impl Into<String>, R));
    fn emit_item(&mut self, item: Item<R>);
    fn into_object(self) -> Self::Object;
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item<R> {
    Data(RelocExpr<R>, Width),
    Instruction(Instruction<R>),
}

#[derive(Clone, Copy)]
enum Data {
    Byte(u8),
    Word(u16),
}

mod lowering;

pub struct Object {
    resolved_sections: Vec<ResolvedSection>,
}

impl Object {
    pub fn into_rom(self) -> Rom {
        let mut data: Vec<u8> = Vec::new();
        self.resolved_sections
            .into_iter()
            .for_each(|section| data.extend(section.data.into_iter()));
        Rom {
            data: data.into_boxed_slice(),
        }
    }
}

pub struct Rom {
    pub data: Box<[u8]>,
}

struct SymbolTable<'a, D: 'a> {
    names: HashMap<String, SymbolId>,
    values: Vec<Value>,
    diagnostics: &'a D,
}

#[derive(Clone, Copy)]
struct SymbolId(usize);

#[derive(Clone)]
pub struct Value {
    min: i32,
    max: i32,
}

impl From<i32> for Value {
    fn from(n: i32) -> Self {
        Value { min: n, max: n }
    }
}

impl AddAssign<Value> for Value {
    fn add_assign(&mut self, rhs: Value) {
        self.min += rhs.min;
        self.max += rhs.max
    }
}

impl<'a, D: 'a> SymbolTable<'a, D> {
    fn new(diagnostics: &'a D) -> SymbolTable<'a, D> {
        SymbolTable {
            names: HashMap::new(),
            values: Vec::new(),
            diagnostics,
        }
    }

    fn resolve_expr_item<SR: SourceInterval>(&self, expr: RelocExpr<SR>, width: Width) -> Data
    where
        D: DiagnosticsListener<SR>,
    {
        let range = expr.source_interval();
        let value = self.evaluate_expr(expr);
        self.fit_to_width((value, range), width)
    }

    fn evaluate_expr<SR: SourceInterval>(&self, expr: RelocExpr<SR>) -> i32
    where
        D: DiagnosticsListener<SR>,
    {
        match expr {
            RelocExpr::Literal(value, _) => value,
            RelocExpr::LocationCounter => panic!(),
            RelocExpr::Subtract(_, _) => panic!(),
            RelocExpr::Symbol(symbol, expr_ref) => {
                let symbol_def = self.symbol_value(&symbol).cloned();
                if let Some(value) = symbol_def {
                    value.min
                } else {
                    self.diagnostics.emit_diagnostic(Diagnostic::new(
                        Message::UnresolvedSymbol {
                            symbol: symbol.clone(),
                        },
                        expr_ref.clone(),
                    ));
                    0
                }
            }
        }
    }

    fn evaluate_expr_value<SR>(&self, expr: &RelocExpr<SR>) -> Option<Value> {
        match expr {
            RelocExpr::Literal(value, _) => Some((*value).into()),
            RelocExpr::LocationCounter => panic!(),
            RelocExpr::Subtract(..) => panic!(),
            RelocExpr::Symbol(symbol, _) => self.symbol_value(symbol).cloned(),
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

    fn symbol_value(&self, name: &str) -> Option<&Value> {
        self.names
            .get(name)
            .and_then(|SymbolId(id)| self.values.get(*id))
    }
}

pub struct ObjectBuilder<'a, R, D: DiagnosticsListener<R> + 'a> {
    pending_sections: Vec<PendingSection<R>>,
    symbol_table: SymbolTable<'a, D>,
}

impl<'a, R: SourceInterval, D: DiagnosticsListener<R> + 'a> ObjectBuilder<'a, R, D> {
    pub fn new(diagnostics: &D) -> ObjectBuilder<R, D> {
        ObjectBuilder {
            pending_sections: vec![PendingSection::new()],
            symbol_table: SymbolTable::new(diagnostics),
        }
    }

    pub fn resolve_symbols(self) -> Object {
        let symbol_table = self.symbol_table;
        Object {
            resolved_sections: self.pending_sections
                .into_iter()
                .map(|pending_section| {
                    pending_section
                        .items
                        .into_iter()
                        .map(|item| match item {
                            Node::Byte(value) => Data::Byte(value),
                            Node::Expr(expr, width) => symbol_table.resolve_expr_item(expr, width),
                            Node::LdInlineAddr(..) => panic!(),
                        })
                        .collect()
                })
                .collect(),
        }
    }
}

impl<'a, R: SourceInterval, D: DiagnosticsListener<R> + 'a> Backend<R> for ObjectBuilder<'a, R, D> {
    type Object = Object;

    fn add_label(&mut self, label: (impl Into<String>, R)) {
        let id = SymbolId(self.symbol_table.values.len());
        let location = self.pending_sections.last().unwrap().location.clone();
        self.symbol_table.values.push(location);
        self.symbol_table.names.insert(label.0.into(), id);
    }

    fn emit_item(&mut self, item: Item<R>) {
        let data_items = self.pending_sections.last_mut().unwrap();
        item.lower()
            .for_each(|data_item| data_items.push(data_item))
    }

    fn into_object(self) -> Self::Object {
        self.resolve_symbols()
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

impl FromIterator<Data> for ResolvedSection {
    fn from_iter<T: IntoIterator<Item = Data>>(iter: T) -> Self {
        let mut section = ResolvedSection::new();
        iter.into_iter().for_each(|x| section.push(x));
        section
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

#[cfg(test)]
mod tests {
    use super::*;
    use instruction::Nullary;

    use std::borrow::Borrow;

    #[test]
    fn emit_literal_byte_item() {
        emit_items_and_compare([byte_literal(0xff)], [0xff])
    }

    #[test]
    fn emit_two_literal_byte_item() {
        emit_items_and_compare([byte_literal(0x12), byte_literal(0x34)], [0x12, 0x34])
    }

    fn byte_literal(value: i32) -> Item<()> {
        Item::Data(RelocExpr::Literal(value, ()), Width::Byte)
    }

    #[test]
    fn emit_stop() {
        emit_items_and_compare(
            [Item::Instruction(Instruction::Nullary(Nullary::Stop))],
            [0x10, 0x00],
        )
    }

    fn emit_items_and_compare<I, B>(items: I, bytes: B)
    where
        I: Borrow<[Item<()>]>,
        B: Borrow<[u8]>,
    {
        let (object, _) = with_object_builder(|builder| {
            for item in items.borrow() {
                builder.emit_item(item.clone())
            }
        });
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
        let (_, diagnostics) =
            with_object_builder(|builder| builder.emit_item(byte_literal(value)));
        assert_eq!(
            *diagnostics,
            [Diagnostic::new(
                Message::ValueOutOfRange {
                    value,
                    width: Width::Byte,
                },
                ()
            )]
        );
    }

    #[test]
    fn diagnose_unresolved_symbol() {
        let ident = "ident";
        let (_, diagnostics) = with_object_builder(|builder| builder.emit_item(symbol_expr(ident)));
        assert_eq!(*diagnostics, [unresolved(ident)]);
    }

    #[test]
    fn emit_defined_symbol() {
        let label = "label";
        let (object, diagnostics) = with_object_builder(|builder| {
            builder.add_label((label, ()));
            builder.emit_item(symbol_expr(label));
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.resolved_sections.last().unwrap().data, [0x00, 0x00])
    }

    #[test]
    fn emit_symbol_defined_after_use() {
        let label = "label";
        let (object, diagnostics) = with_object_builder(|builder| {
            builder.emit_item(symbol_expr(label));
            builder.add_label((label, ()));
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.resolved_sections.last().unwrap().data, [0x02, 0x00])
    }

    type TestObjectBuilder<'a> = ObjectBuilder<'a, (), TestDiagnosticsListener>;

    fn with_object_builder<F: FnOnce(&mut TestObjectBuilder)>(
        f: F,
    ) -> (Object, Box<[Diagnostic<()>]>) {
        let diagnostics = TestDiagnosticsListener::new();
        let object = {
            let mut builder = ObjectBuilder::new(&diagnostics);
            f(&mut builder);
            builder.resolve_symbols()
        };
        let diagnostics = diagnostics.diagnostics.into_inner().into_boxed_slice();
        (object, diagnostics)
    }

    fn symbol_expr(symbol: impl Into<String>) -> Item<()> {
        Item::Data(RelocExpr::Symbol(symbol.into(), ()), Width::Word)
    }

    fn unresolved(symbol: impl Into<String>) -> Diagnostic<()> {
        Diagnostic::new(
            Message::UnresolvedSymbol {
                symbol: symbol.into(),
            },
            (),
        )
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
