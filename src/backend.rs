use crate::expr::{BinaryOperator, Expr, ExprVariant};
use crate::frontend::Ident;
use crate::instruction::Instruction;
use crate::program::NameId;
use crate::span::Source;
#[cfg(test)]
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Width {
    Byte,
    Word,
}

pub trait NameTable<I> {
    type MacroEntry;

    fn get(&self, ident: &I) -> Option<&Name<Self::MacroEntry>>;
    fn insert(&mut self, ident: I, entry: Name<Self::MacroEntry>);
}

pub struct HashMapNameTable<M> {
    table: HashMap<String, Name<M>>,
}

pub enum Name<M> {
    Macro(M),
    Symbol(NameId),
}

impl<M> HashMapNameTable<M> {
    pub fn new() -> Self {
        HashMapNameTable {
            table: HashMap::new(),
        }
    }
}

impl<M> NameTable<Ident<String>> for HashMapNameTable<M> {
    type MacroEntry = M;

    fn get(&self, ident: &Ident<String>) -> Option<&Name<Self::MacroEntry>> {
        self.table.get(&ident.name)
    }

    fn insert(&mut self, ident: Ident<String>, entry: Name<Self::MacroEntry>) {
        self.table.insert(ident.name, entry);
    }
}

pub trait HasValue<S: Clone> {
    type Value: Source<Span = S>;
}

pub trait ValueFromSimple<S: Clone>
where
    Self: HasValue<S>,
{
    fn from_location_counter(&mut self, span: S) -> Self::Value;
    fn from_number(&mut self, n: i32, span: S) -> Self::Value;
}

pub trait ValueFromIdent<N, I, S: Clone>
where
    Self: HasValue<S>,
{
    fn from_ident(&mut self, ident: I, span: S, names: &mut N) -> Self::Value;
}

pub trait ApplyBinaryOperator<S: Clone>
where
    Self: HasValue<S>,
{
    fn apply_binary_operator(
        &mut self,
        operator: (BinaryOperator, S),
        left: Self::Value,
        right: Self::Value,
    ) -> Self::Value;
}

pub trait PartialBackend<S>
where
    S: Clone,
    Self: HasValue<S>,
{
    fn emit_item(&mut self, item: Item<Self::Value>);
    fn set_origin(&mut self, origin: Self::Value);
}

pub trait Backend<I, S, N>
where
    S: Clone,
    Self: PartialBackend<S>,
    Self: ValueFromSimple<S>,
    Self: ValueFromIdent<N, I, S>,
    Self: ApplyBinaryOperator<S>,
{
    fn define_symbol(&mut self, symbol: (I, S), value: Self::Value, names: &mut N);
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
    pub sections: Vec<BinarySection>,
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

pub struct BinarySection {
    pub origin: usize,
    pub data: Vec<u8>,
}

#[cfg(test)]
pub struct MockBackend<'a, T> {
    pub log: &'a RefCell<Vec<T>>,
}

#[cfg(test)]
#[derive(Debug, PartialEq)]
pub enum Event<V: Source> {
    EmitItem(Item<V>),
    SetOrigin(V),
    DefineSymbol((Ident<String>, V::Span), V),
}

#[cfg(test)]
impl<'a, T> MockBackend<'a, T> {
    pub fn new(log: &'a RefCell<Vec<T>>) -> Self {
        MockBackend { log }
    }
}

#[cfg(test)]
impl<'a, T, S, N> Backend<Ident<String>, S, N> for MockBackend<'a, T>
where
    T: From<Event<RelocExpr<Ident<String>, S>>>,
    S: Clone,
    N: 'static,
{
    fn define_symbol(&mut self, symbol: (Ident<String>, S), value: Self::Value, _: &mut N) {
        self.log
            .borrow_mut()
            .push(Event::DefineSymbol(symbol, value).into())
    }
}

#[cfg(test)]
impl<'a, T, S: Clone> ValueFromSimple<S> for MockBackend<'a, T> {
    fn from_location_counter(&mut self, span: S) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::LocationCounter, span)
    }

    fn from_number(&mut self, n: i32, span: S) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::Literal(n), span)
    }
}

#[cfg(test)]
impl<'a, T, N, S: Clone> ValueFromIdent<N, Ident<String>, S> for MockBackend<'a, T> {
    fn from_ident(&mut self, ident: Ident<String>, span: S, _: &mut N) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::Symbol(ident), span)
    }
}

#[cfg(test)]
impl<'a, T, S: Clone> ApplyBinaryOperator<S> for MockBackend<'a, T> {
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

#[cfg(test)]
impl<'a, T, S: Clone> HasValue<S> for MockBackend<'a, T> {
    type Value = RelocExpr<Ident<String>, S>;
}

#[cfg(test)]
impl<'a, T, S> PartialBackend<S> for MockBackend<'a, T>
where
    T: From<Event<RelocExpr<Ident<String>, S>>>,
    S: Clone,
{
    fn emit_item(&mut self, item: Item<Self::Value>) {
        self.log.borrow_mut().push(Event::EmitItem(item).into())
    }

    fn set_origin(&mut self, origin: Self::Value) {
        self.log.borrow_mut().push(Event::SetOrigin(origin).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
