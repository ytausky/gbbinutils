use crate::expr::{BinaryOperator, Expr, ExprVariant};
#[cfg(test)]
use crate::frontend::Ident;
use crate::instruction::Instruction;
use crate::span::Source;
#[cfg(test)]
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Width {
    Byte,
    Word,
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
    Self: ApplyBinaryOperator<S>,
{
}

impl<T, I, S: Clone> ValueBuilder<I, S> for T
where
    T: ToValue<LocationCounter, S>,
    T: ToValue<i32, S>,
    T: ToValue<I, S>,
    T: ApplyBinaryOperator<S>,
{
}

pub trait ToValue<T, S: Clone>
where
    Self: HasValue<S>,
{
    fn to_value(&mut self, atom: (T, S)) -> Self::Value;
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

pub struct RelocExprBuilder<T>(pub T);

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
    type Value = RelocExpr<Ident<String>, S>;
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
impl<S: Clone> ToValue<Ident<String>, S> for IndependentValueBuilder<S> {
    fn to_value(&mut self, (name, span): (Ident<String>, S)) -> Self::Value {
        RelocExpr::from_atom(RelocAtom::Symbol(name), span)
    }
}

impl<I, T, S: Clone> ApplyBinaryOperator<S> for RelocExprBuilder<T>
where
    Self: HasValue<S, Value = RelocExpr<I, S>>,
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

pub struct BinarySection {
    pub origin: usize,
    pub data: Vec<u8>,
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
