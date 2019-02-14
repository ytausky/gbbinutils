use crate::analysis::Ident;
use crate::expr::BinaryOperator;
use crate::model::Item;
use crate::program::NameId;
use crate::span::Source;

use std::collections::HashMap;

#[cfg(test)]
pub use mock::*;

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

#[cfg(test)]
mod mock {
    use super::*;

    use crate::expr::{Expr, ExprVariant};
    use crate::model::{RelocAtom, RelocExpr};

    use std::cell::RefCell;

    pub struct MockBackend<'a, T> {
        pub log: &'a RefCell<Vec<T>>,
    }

    #[derive(Debug, PartialEq)]
    pub enum BackendEvent<V: Source> {
        EmitItem(Item<V>),
        SetOrigin(V),
        DefineSymbol((Ident<String>, V::Span), V),
    }

    impl<'a, T> MockBackend<'a, T> {
        pub fn new(log: &'a RefCell<Vec<T>>) -> Self {
            MockBackend { log }
        }
    }

    impl<'a, T, S, N> Backend<Ident<String>, S, N> for MockBackend<'a, T>
    where
        T: From<BackendEvent<RelocExpr<Ident<String>, S>>>,
        S: Clone,
        N: 'static,
    {
        fn define_symbol(&mut self, symbol: (Ident<String>, S), value: Self::Value, _: &mut N) {
            self.log
                .borrow_mut()
                .push(BackendEvent::DefineSymbol(symbol, value).into())
        }
    }

    impl<'a, T, S: Clone> ValueFromSimple<S> for MockBackend<'a, T> {
        fn from_location_counter(&mut self, span: S) -> Self::Value {
            RelocExpr::from_atom(RelocAtom::LocationCounter, span)
        }

        fn from_number(&mut self, n: i32, span: S) -> Self::Value {
            RelocExpr::from_atom(RelocAtom::Literal(n), span)
        }
    }

    impl<'a, T, N, S: Clone> ValueFromIdent<N, Ident<String>, S> for MockBackend<'a, T> {
        fn from_ident(&mut self, ident: Ident<String>, span: S, _: &mut N) -> Self::Value {
            RelocExpr::from_atom(RelocAtom::Symbol(ident), span)
        }
    }

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

    impl<'a, T, S: Clone> HasValue<S> for MockBackend<'a, T> {
        type Value = RelocExpr<Ident<String>, S>;
    }

    impl<'a, T, S> PartialBackend<S> for MockBackend<'a, T>
    where
        T: From<BackendEvent<RelocExpr<Ident<String>, S>>>,
        S: Clone,
    {
        fn emit_item(&mut self, item: Item<Self::Value>) {
            self.log
                .borrow_mut()
                .push(BackendEvent::EmitItem(item).into())
        }

        fn set_origin(&mut self, origin: Self::Value) {
            self.log
                .borrow_mut()
                .push(BackendEvent::SetOrigin(origin).into())
        }
    }
}
