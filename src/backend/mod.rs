pub use backend::object::ObjectBuilder;

use backend::{lowering::Lower,
              object::{Node, Object, Section}};
use diagnostics::*;
use instruction::{Instruction, RelocExpr};
use std::{borrow::Borrow, collections::HashMap, iter::FromIterator, ops::AddAssign};
use Width;

mod object;

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

pub struct BinaryObject {
    sections: Vec<BinarySection>,
}

impl BinaryObject {
    pub fn into_rom(self) -> Rom {
        let mut data: Vec<u8> = Vec::new();
        self.sections
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

pub struct SymbolTable {
    symbols: HashMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    min: i32,
    max: i32,
}

impl Value {
    fn exact(&self) -> Option<i32> {
        if self.min == self.max {
            Some(self.min)
        } else {
            None
        }
    }
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

impl SymbolTable {
    fn new() -> SymbolTable {
        SymbolTable {
            symbols: HashMap::new(),
        }
    }

    fn define(&mut self, name: impl Into<String>, value: Value) {
        self.symbols.insert(name.into(), value);
    }

    fn get(&self, name: impl Borrow<str>) -> Option<&Value> {
        self.symbols.get(name.borrow())
    }

    fn resolve_expr_item<SR: SourceInterval>(
        &self,
        expr: RelocExpr<SR>,
        width: Width,
    ) -> Result<Data, Diagnostic<SR>> {
        let range = expr.source_interval();
        let value = expr.evaluate(self)
            .map_err(|undefined| {
                let UndefinedSymbol(symbol, range) = undefined;
                Diagnostic::new(Message::UnresolvedSymbol { symbol }, range)
            })?
            .exact()
            .unwrap();
        self.fit_to_width((value, range), width)
    }

    fn fit_to_width<SR: Clone>(
        &self,
        (value, value_ref): (i32, SR),
        width: Width,
    ) -> Result<Data, Diagnostic<SR>> {
        if !is_in_range(value, width) {
            Err(Diagnostic::new(
                Message::ValueOutOfRange { value, width },
                value_ref.clone(),
            ))
        } else {
            Ok(match width {
                Width::Byte => Data::Byte(value as u8),
                Width::Word => Data::Word(value as u16),
            })
        }
    }

    fn symbol_value(&self, name: &str) -> Option<&Value> {
        self.symbols.get(name)
    }
}

struct UndefinedSymbol<SR>(String, SR);

impl<SR: Clone> RelocExpr<SR> {
    fn evaluate(&self, symbol_table: &SymbolTable) -> Result<Value, UndefinedSymbol<SR>> {
        match self {
            RelocExpr::Literal(value, _) => Ok((*value).into()),
            RelocExpr::LocationCounter => panic!(),
            RelocExpr::Subtract(_, _) => panic!(),
            RelocExpr::Symbol(symbol, expr_ref) => symbol_table
                .symbol_value(&symbol)
                .cloned()
                .ok_or_else(|| UndefinedSymbol((*symbol).clone(), (*expr_ref).clone())),
        }
    }
}

pub fn link<'a, SR, D>(object: Object<SR>, diagnostics: &D) -> BinaryObject
where
    SR: SourceInterval,
    D: DiagnosticsListener<SR> + 'a,
{
    let symbols = resolve_symbols(&object);
    BinaryObject {
        sections: object
            .sections
            .into_iter()
            .map(|section| resolve_section(section, &symbols, diagnostics))
            .collect(),
    }
}

fn resolve_symbols<SR: Clone>(object: &Object<SR>) -> SymbolTable {
    collect_symbols(object)
}

fn collect_symbols<SR: Clone>(object: &Object<SR>) -> SymbolTable {
    let mut symbols = SymbolTable::new();
    for section in &object.sections {
        let mut location = Value::from(0);
        for node in &section.items {
            match node {
                Node::Label(symbol, _) => {
                    symbols.symbols.insert(symbol.to_string(), location.clone());
                }
                node => location += node.size(&symbols),
            }
        }
        symbols.define(format!("{}.size", section.name()), location)
    }
    symbols
}

fn resolve_section<SR: SourceInterval>(
    section: Section<SR>,
    symbols: &SymbolTable,
    diagnostics: &impl DiagnosticsListener<SR>,
) -> BinarySection {
    section
        .items
        .into_iter()
        .flat_map(|item| match item {
            Node::Byte(value) => Some(Data::Byte(value)),
            Node::Expr(expr, width) => Some(symbols.resolve_expr_item(expr, width).unwrap_or_else(
                |diagnostic| {
                    diagnostics.emit_diagnostic(diagnostic);
                    match width {
                        Width::Byte => Data::Byte(0),
                        Width::Word => Data::Word(0),
                    }
                },
            )),
            Node::Label(..) => None,
            Node::LdInlineAddr(..) => panic!(),
        })
        .collect()
}

impl<SR: SourceInterval> Backend<SR> for ObjectBuilder<SR> {
    type Object = Object<SR>;

    fn add_label(&mut self, label: (impl Into<String>, SR)) {
        let section = self.object.sections.last_mut().unwrap();
        section.items.push(Node::Label(label.0.into(), label.1))
    }

    fn emit_item(&mut self, item: Item<SR>) {
        let section = self.object.sections.last_mut().unwrap();
        item.lower()
            .for_each(|data_item| section.items.push(data_item))
    }

    fn into_object(self) -> Self::Object {
        self.object
    }
}

struct BinarySection {
    data: Vec<u8>,
}

impl BinarySection {
    fn new() -> BinarySection {
        BinarySection { data: Vec::new() }
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

impl FromIterator<Data> for BinarySection {
    fn from_iter<T: IntoIterator<Item = Data>>(iter: T) -> Self {
        let mut section = BinarySection::new();
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
    use instruction::{Direction, Nullary};

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
        assert_eq!(object.sections.last().unwrap().data, bytes.borrow())
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
        assert_eq!(object.sections.last().unwrap().data, [0x00, 0x00])
    }

    #[test]
    fn emit_symbol_defined_after_use() {
        let label = "label";
        let (object, diagnostics) = with_object_builder(|builder| {
            builder.emit_item(symbol_expr(label));
            builder.add_label((label, ()));
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x02, 0x00])
    }

    #[test]
    fn empty_section_has_size_zero() {
        assert_section_size(0, |_| ())
    }

    #[test]
    fn section_with_nop_has_size_one() {
        assert_section_size(1, |section| {
            section
                .items
                .extend(Item::Instruction(Instruction::Nullary(Nullary::Nop)).lower())
        });
    }

    #[test]
    fn section_with_const_inline_addr_ld_has_size_two() {
        test_section_size_with_literal_ld_inline_addr(0xff00, 2)
    }

    #[test]
    fn section_with_const_inline_addr_ld_has_size_three() {
        test_section_size_with_literal_ld_inline_addr(0x1234, 3)
    }

    fn test_section_size_with_literal_ld_inline_addr(addr: i32, expected: i32) {
        assert_section_size(expected, |section| {
            section.items.push(Node::LdInlineAddr(
                RelocExpr::Literal(addr, ()),
                Direction::FromA,
            ))
        });
    }

    fn assert_section_size(expected: impl Into<Value>, f: impl FnOnce(&mut Section<()>)) {
        let mut object = Object::<()>::new();
        object.add_section("TestSection");
        f(&mut object.sections[0]);
        let symbols = resolve_symbols(&object);
        assert_eq!(
            symbols.get("TestSection.size").cloned(),
            Some(expected.into())
        )
    }

    type TestObjectBuilder = ObjectBuilder<()>;

    fn with_object_builder<F: FnOnce(&mut TestObjectBuilder)>(
        f: F,
    ) -> (BinaryObject, Box<[Diagnostic<()>]>) {
        let diagnostics = TestDiagnosticsListener::new();
        let object = {
            let mut builder = ObjectBuilder::new();
            f(&mut builder);
            link(builder.object, &diagnostics)
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
