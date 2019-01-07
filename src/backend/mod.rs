pub use crate::backend::object::ObjectBuilder;

use crate::backend::{
    lowering::Lower,
    object::{Node, Object},
};
use crate::expr::{BinaryOperator, Expr, ExprVariant};
use crate::instruction::Instruction;
use crate::span::Source;
#[cfg(test)]
use std::marker::PhantomData;

mod lowering;
mod object;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Width {
    Byte,
    Word,
}

impl Width {
    fn len(self) -> i32 {
        match self {
            Width::Byte => 1,
            Width::Word => 2,
        }
    }
}

pub struct LocationCounter;

pub trait HasValue<S: Clone> {
    type Value: Source<Span = S>;
}

pub trait BuildValue<'a, I, S: Clone>
where
    Self: HasValue<S>,
{
    type Builder: ValueBuilder<I, S, Value = Self::Value>;
    fn build_value(&'a mut self) -> Self::Builder;
}

pub trait ValueBuilder<I, S: Clone>
where
    Self: ToValue<LocationCounter, S>,
    Self: ToValue<i32, S>,
    Self: ToValue<I, S>,
{
    fn apply_binary_operator(
        &mut self,
        operator: (BinaryOperator, S),
        left: Self::Value,
        right: Self::Value,
    ) -> Self::Value;
}

pub trait ToValue<T, S: Clone>
where
    Self: HasValue<S>,
{
    fn to_value(&mut self, atom: (T, S)) -> Self::Value;
}

pub trait Backend<I, S>
where
    S: Clone,
    for<'a> Self: BuildValue<'a, I, S>,
{
    type Object;
    fn define_symbol(&mut self, symbol: (I, S), value: Self::Value);
    fn emit_item(&mut self, item: Item<Self::Value>);
    fn into_object(self) -> Self::Object;
    fn set_origin(&mut self, origin: Self::Value);
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item<V: Source> {
    Data(V, Width),
    Instruction(Instruction<V>),
}

pub type RelocExpr<I, S> = Expr<RelocAtom<I>, Empty, BinaryOperator, S>;

#[derive(Clone, Debug, PartialEq)]
pub enum Empty {}

#[derive(Clone, Debug, PartialEq)]
pub enum RelocAtom<I> {
    Literal(i32),
    LocationCounter,
    Symbol(I),
}

impl<I, S> From<i32> for ExprVariant<RelocAtom<I>, Empty, BinaryOperator, S> {
    fn from(n: i32) -> Self {
        ExprVariant::Atom(RelocAtom::Literal(n))
    }
}

#[cfg(test)]
impl<I, T: Into<ExprVariant<RelocAtom<I>, Empty, BinaryOperator, ()>>> From<T>
    for RelocExpr<I, ()>
{
    fn from(variant: T) -> Self {
        Expr {
            variant: variant.into(),
            span: (),
        }
    }
}

pub struct BinaryObject {
    sections: Vec<BinarySection>,
}

impl BinaryObject {
    pub fn into_rom(self) -> Rom {
        let mut data: Vec<u8> = Vec::new();
        for chunk in self.sections {
            if !chunk.data.is_empty() {
                let end = chunk.origin + chunk.data.len();
                if data.len() < end {
                    data.resize(end, 0x00)
                }
                data[chunk.origin..end].copy_from_slice(&chunk.data)
            }
        }
        if data.len() < MIN_ROM_LEN {
            data.resize(MIN_ROM_LEN, 0x00)
        }
        Rom {
            data: data.into_boxed_slice(),
        }
    }
}

const MIN_ROM_LEN: usize = 0x8000;

pub struct Rom {
    pub data: Box<[u8]>,
}

pub struct RelocExprBuilder<T>(T);

#[cfg(test)]
pub type IndependentValueBuilder<S> = RelocExprBuilder<PhantomData<S>>;

#[cfg(test)]
impl<S> IndependentValueBuilder<S> {
    pub fn new() -> Self {
        RelocExprBuilder(PhantomData)
    }
}

#[cfg(test)]
impl<S: Clone> HasValue<S> for IndependentValueBuilder<S> {
    type Value = RelocExpr<String, S>;
}

impl<I, T, S: Clone> ToValue<LocationCounter, S> for RelocExprBuilder<T>
where
    Self: HasValue<S, Value = RelocExpr<I, S>>,
{
    fn to_value(&mut self, (_, span): (LocationCounter, S)) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::LocationCounter, span)
    }
}

impl<I, T, S: Clone> ToValue<i32, S> for RelocExprBuilder<T>
where
    Self: HasValue<S, Value = RelocExpr<I, S>>,
{
    fn to_value(&mut self, (number, span): (i32, S)) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::Literal(number), span)
    }
}

#[cfg(test)]
impl<S: Clone> ToValue<String, S> for IndependentValueBuilder<S> {
    fn to_value(&mut self, (name, span): (String, S)) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::Symbol(name), span)
    }
}

impl<'a, S: Clone> HasValue<S> for RelocExprBuilder<&'a mut ObjectBuilder<S>> {
    type Value = RelocExpr<String, S>;
}

impl<'a, S: Clone> ToValue<String, S> for RelocExprBuilder<&'a mut ObjectBuilder<S>> {
    fn to_value(&mut self, (name, span): (String, S)) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::Symbol(name), span)
    }
}

impl<I, T, S: Clone> ValueBuilder<I, S> for RelocExprBuilder<T>
where
    Self: HasValue<S, Value = RelocExpr<I, S>>,
    Self: ToValue<LocationCounter, S>,
    Self: ToValue<i32, S>,
    Self: ToValue<I, S>,
{
    fn apply_binary_operator(
        &mut self,
        operator: (BinaryOperator, S),
        left: Self::Value,
        right: Self::Value,
    ) -> Self::Value {
        Expr {
            variant: ExprVariant::Binary(operator.0, Box::new(left), Box::new(right)),
            span: operator.1,
        }
    }
}

impl<S: Clone> HasValue<S> for ObjectBuilder<S> {
    type Value = RelocExpr<String, S>;
}

impl<'a, S: Clone + 'static> BuildValue<'a, String, S> for ObjectBuilder<S> {
    type Builder = RelocExprBuilder<&'a mut Self>;

    fn build_value(&'a mut self) -> Self::Builder {
        RelocExprBuilder(self)
    }
}

impl<S: Clone + 'static> Backend<String, S> for ObjectBuilder<S> {
    type Object = Object<String, S>;

    fn define_symbol(&mut self, symbol: (String, S), value: Self::Value) {
        self.push(Node::Symbol(symbol, value))
    }

    fn emit_item(&mut self, item: Item<Self::Value>) {
        item.lower().for_each(|data_item| self.push(data_item))
    }

    fn into_object(self) -> Self::Object {
        self.build()
    }

    fn set_origin(&mut self, origin: Self::Value) {
        self.constrain_origin(origin)
    }
}

pub struct BinarySection {
    origin: usize,
    data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag::{CompactDiagnostic, Message, TestDiagnosticsListener};
    use crate::instruction::Nullary;
    use std::borrow::Borrow;

    #[test]
    fn empty_object_converted_to_all_zero_rom() {
        let object = BinaryObject {
            sections: Vec::new(),
        };
        let rom = object.into_rom();
        assert_eq!(*rom.data, [0x00u8; MIN_ROM_LEN][..])
    }

    #[test]
    fn chunk_placed_in_rom_starting_at_origin() {
        let byte = 0x42;
        let origin = 0x150;
        let object = BinaryObject {
            sections: vec![BinarySection {
                origin,
                data: vec![byte],
            }],
        };
        let rom = object.into_rom();
        let mut expected = [0x00u8; MIN_ROM_LEN];
        expected[origin] = byte;
        assert_eq!(*rom.data, expected[..])
    }

    #[test]
    fn empty_chunk_does_not_extend_rom() {
        let origin = MIN_ROM_LEN + 1;
        let object = BinaryObject {
            sections: vec![BinarySection {
                origin,
                data: Vec::new(),
            }],
        };
        let rom = object.into_rom();
        assert_eq!(rom.data.len(), MIN_ROM_LEN)
    }

    #[test]
    fn emit_literal_byte_item() {
        emit_items_and_compare([byte_literal(0xff)], [0xff])
    }

    #[test]
    fn emit_two_literal_byte_item() {
        emit_items_and_compare([byte_literal(0x12), byte_literal(0x34)], [0x12, 0x34])
    }

    fn byte_literal(value: i32) -> Item<RelocExpr<String, ()>> {
        Item::Data(value.into(), Width::Byte)
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
        I: Borrow<[Item<RelocExpr<String, ()>>]>,
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
            [CompactDiagnostic::new(
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
        let (_, diagnostics) = with_object_builder(|builder| {
            let value = builder
                .build_value()
                .to_value((ident.to_string(), ident.into()));
            builder.emit_item(word_item(value))
        });
        assert_eq!(*diagnostics, [unresolved(ident)]);
    }

    #[test]
    fn diagnose_two_unresolved_symbols_in_one_expr() {
        let ident1 = "ident1";
        let ident2 = "ident2";
        let (_, diagnostics) = with_object_builder(|builder| {
            let value = {
                let mut builder = builder.build_value();
                let lhs = builder.to_value((ident1.to_string(), ident1.into()));
                let rhs = builder.to_value((ident2.to_string(), ident2.into()));
                builder.apply_binary_operator((BinaryOperator::Minus, "diff".into()), lhs, rhs)
            };
            builder.emit_item(word_item(value))
        });
        assert_eq!(*diagnostics, [unresolved(ident1), unresolved(ident2)]);
    }

    #[test]
    fn emit_defined_symbol() {
        let label = "label";
        let (object, diagnostics) = with_object_builder(|builder| {
            builder.define_symbol((label.into(), ()), RelocAtom::LocationCounter.into());
            let value = builder.build_value().to_value((label.to_string(), ()));
            builder.emit_item(word_item(value));
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x00, 0x00])
    }

    #[test]
    fn emit_symbol_defined_after_use() {
        let label = "label";
        let (object, diagnostics) = with_object_builder(|builder| {
            let value = builder.build_value().to_value((label.to_string(), ()));
            builder.emit_item(word_item(value));
            builder.define_symbol((label.into(), ()), RelocAtom::LocationCounter.into());
        });
        assert_eq!(*diagnostics, []);
        assert_eq!(object.sections.last().unwrap().data, [0x02, 0x00])
    }

    fn with_object_builder<S: Clone, F: FnOnce(&mut ObjectBuilder<S>)>(
        f: F,
    ) -> (BinaryObject, Box<[CompactDiagnostic<S, S>]>) {
        let mut diagnostics = TestDiagnosticsListener::new();
        let object = {
            let mut builder = ObjectBuilder::new();
            f(&mut builder);
            builder.build().link(&mut diagnostics)
        };
        let diagnostics = diagnostics.diagnostics.into_inner().into_boxed_slice();
        (object, diagnostics)
    }

    fn word_item<S: Clone>(value: RelocExpr<String, S>) -> Item<RelocExpr<String, S>> {
        Item::Data(value, Width::Word)
    }

    fn unresolved(symbol: impl Into<String>) -> CompactDiagnostic<String, String> {
        let symbol = symbol.into();
        CompactDiagnostic::new(
            Message::UnresolvedSymbol {
                symbol: symbol.clone(),
            },
            symbol,
        )
    }
}
