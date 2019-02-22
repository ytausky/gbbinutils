use crate::expr::BinaryOperator;
use crate::model::Item;
use crate::span::Source;

#[cfg(test)]
pub use mock::*;

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

pub trait StartSection<I, S> {
    fn start_section(&mut self, name: (I, S));
}

pub trait Backend<I, S, N>
where
    S: Clone,
    Self: PartialBackend<S>,
    Self: ValueFromSimple<S>,
    Self: ValueFromIdent<N, I, S>,
    Self: ApplyBinaryOperator<S>,
    Self: StartSection<I, S>,
{
    fn define_symbol(&mut self, symbol: (I, S), value: Self::Value, names: &mut N);
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::expr::{Expr, ExprVariant};
    use crate::model::{RelocAtom, RelocExpr};
    use crate::name::Ident;

    use std::cell::RefCell;

    pub struct MockBackend<'a, T> {
        pub log: &'a RefCell<Vec<T>>,
    }

    #[derive(Debug, PartialEq)]
    pub enum BackendEvent<V: Source> {
        EmitItem(Item<V>),
        SetOrigin(V),
        DefineSymbol((Ident<String>, V::Span), V),
        StartSection((Ident<String>, V::Span)),
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

    impl<'a, T, S> StartSection<Ident<String>, S> for MockBackend<'a, T>
    where
        T: From<BackendEvent<RelocExpr<Ident<String>, S>>>,
        S: Clone,
    {
        fn start_section(&mut self, name: (Ident<String>, S)) {
            self.log
                .borrow_mut()
                .push(BackendEvent::StartSection(name).into())
        }
    }
}
