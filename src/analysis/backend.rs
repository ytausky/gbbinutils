use crate::expr::BinaryOperator;
use crate::model::Item;
use crate::span::Source;

#[cfg(test)]
pub use mock::*;

pub trait HasValue<S: Clone> {
    type Value: Source<Span = S>;
}

pub trait HasName {
    type Name: Clone;
}

pub trait AllocName<S: Clone>
where
    Self: HasName,
{
    fn alloc_name(&mut self, span: S) -> Self::Name;
}

pub trait ValueFromSimple<S: Clone>
where
    Self: HasValue<S>,
{
    fn from_location_counter(&mut self, span: S) -> Self::Value;
    fn from_number(&mut self, n: i32, span: S) -> Self::Value;
}

pub trait ValueFromName<S: Clone>
where
    Self: HasName + HasValue<S>,
{
    fn from_name(&mut self, symbol: Self::Name, span: S) -> Self::Value;
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
    fn reserve(&mut self, bytes: Self::Value);
    fn set_origin(&mut self, origin: Self::Value);
}

pub trait StartSection<N, S> {
    fn start_section(&mut self, name: (N, S));
}

pub trait Backend<I, S>
where
    S: Clone,
    Self: AllocName<S>,
    Self: PartialBackend<S>,
    Self: ValueFromSimple<S>,
    Self: ValueFromName<S>,
    Self: ApplyBinaryOperator<S>,
    Self: StartSection<<Self as HasName>::Name, S>,
{
    fn define_symbol(&mut self, symbol: (Self::Name, S), value: Self::Value);
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::expr::ExprVariant;
    use crate::model::{Atom, Attr, Expr};
    use crate::name::Ident;

    use std::cell::RefCell;

    pub struct MockBackend<'a, T> {
        pub log: &'a RefCell<Vec<T>>,
        next_symbol_id: usize,
    }

    #[derive(Debug, PartialEq)]
    pub enum BackendEvent<V: Source> {
        EmitItem(Item<V>),
        Reserve(V),
        SetOrigin(V),
        DefineSymbol((usize, V::Span), V),
        StartSection((usize, V::Span)),
    }

    impl<'a, T> MockBackend<'a, T> {
        pub fn new(log: &'a RefCell<Vec<T>>) -> Self {
            MockBackend {
                log,
                next_symbol_id: 0,
            }
        }
    }

    impl<'a, T, S> Backend<Ident<String>, S> for MockBackend<'a, T>
    where
        T: From<BackendEvent<Expr<usize, S>>>,
        S: Clone,
    {
        fn define_symbol(&mut self, symbol: (Self::Name, S), value: Self::Value) {
            self.log
                .borrow_mut()
                .push(BackendEvent::DefineSymbol(symbol, value).into())
        }
    }

    impl<'a, T, S: Clone> ValueFromSimple<S> for MockBackend<'a, T> {
        fn from_location_counter(&mut self, span: S) -> Self::Value {
            Expr::from_atom(Atom::LocationCounter, span)
        }

        fn from_number(&mut self, n: i32, span: S) -> Self::Value {
            Expr::from_atom(Atom::Literal(n), span)
        }
    }

    impl<'a, T, S: Clone> ValueFromName<S> for MockBackend<'a, T> {
        fn from_name(&mut self, name: Self::Name, span: S) -> Self::Value {
            Expr::from_atom(Atom::Attr(name, Attr::Addr), span)
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
        type Value = Expr<usize, S>;
    }

    impl<'a, T> HasName for MockBackend<'a, T> {
        type Name = usize;
    }

    impl<'a, T, S: Clone> AllocName<S> for MockBackend<'a, T> {
        fn alloc_name(&mut self, _span: S) -> Self::Name {
            let id = self.next_symbol_id;
            self.next_symbol_id += 1;
            id
        }
    }

    impl<'a, T, S> PartialBackend<S> for MockBackend<'a, T>
    where
        T: From<BackendEvent<Expr<usize, S>>>,
        S: Clone,
    {
        fn emit_item(&mut self, item: Item<Self::Value>) {
            self.log
                .borrow_mut()
                .push(BackendEvent::EmitItem(item).into())
        }

        fn reserve(&mut self, bytes: Self::Value) {
            self.log
                .borrow_mut()
                .push(BackendEvent::Reserve(bytes).into())
        }

        fn set_origin(&mut self, origin: Self::Value) {
            self.log
                .borrow_mut()
                .push(BackendEvent::SetOrigin(origin).into())
        }
    }

    impl<'a, T, S> StartSection<usize, S> for MockBackend<'a, T>
    where
        T: From<BackendEvent<Expr<usize, S>>>,
        S: Clone,
    {
        fn start_section(&mut self, name: (usize, S)) {
            self.log
                .borrow_mut()
                .push(BackendEvent::StartSection(name).into())
        }
    }
}
