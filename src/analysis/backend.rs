pub use crate::model::LocationCounter;

use crate::model::{Atom, BinOp, Expr, ExprOp, FnCall, Item, ParamId};
use crate::span::{Source, WithSpan};

#[cfg(test)]
pub use mock::*;

pub trait AllocName<S: Clone> {
    type Name: Clone;

    fn alloc_name(&mut self, span: S) -> Self::Name;
}

pub trait PushOp<T, S: Clone> {
    fn push_op(&mut self, op: T, span: S);
}

pub trait PartialBackend<S: Clone> {
    type Value: Source<Span = S>;

    fn emit_item(&mut self, item: Item<Self::Value>);
    fn reserve(&mut self, bytes: Self::Value);
    fn set_origin(&mut self, origin: Self::Value);
}

pub trait StartSection<N, S> {
    fn start_section(&mut self, name: (N, S));
}

pub trait ValueBuilder<N, S: Clone>:
    PushOp<LocationCounter, S>
    + PushOp<i32, S>
    + PushOp<Name<N>, S>
    + PushOp<BinOp, S>
    + PushOp<ParamId, S>
    + PushOp<FnCall, S>
{
}

impl<T, N, S: Clone> ValueBuilder<N, S> for T where
    Self: PushOp<LocationCounter, S>
        + PushOp<i32, S>
        + PushOp<Name<N>, S>
        + PushOp<BinOp, S>
        + PushOp<ParamId, S>
        + PushOp<FnCall, S>
{
}

#[derive(Clone)]
pub struct Name<T>(pub T);

impl<L, N> From<Name<N>> for Atom<L, N> {
    fn from(Name(name): Name<N>) -> Self {
        Atom::Name(name)
    }
}

impl<T: Into<ExprOp<A>>, A, S: Clone> PushOp<T, S> for Expr<A, S> {
    fn push_op(&mut self, op: T, span: S) {
        self.0.push(op.into().with_span(span))
    }
}

impl<N, A: From<Name<N>>> From<Name<N>> for ExprOp<A> {
    fn from(name: Name<N>) -> Self {
        ExprOp::Atom(name.into())
    }
}

pub trait Finish {
    type Parent;
    type Value;

    fn finish(self) -> (Self::Parent, Self::Value);
}

pub trait Backend<S>
where
    S: Clone,
    Self: Sized,
    Self: AllocName<S>,
    Self: PartialBackend<S>,
    Self: StartSection<<Self as AllocName<S>>::Name, S>,
{
    type ImmediateBuilder: AllocName<S, Name = Self::Name>
        + ValueBuilder<Self::Name, S>
        + Finish<Parent = Self, Value = Self::Value>;

    type SymbolBuilder: AllocName<S, Name = Self::Name>
        + ValueBuilder<Self::Name, S>
        + Finish<Parent = Self, Value = ()>;

    fn build_immediate(self) -> Self::ImmediateBuilder;
    fn define_symbol(self, name: Self::Name, span: S) -> Self::SymbolBuilder;
}

pub(crate) struct RelocContext<P, B> {
    pub parent: P,
    pub builder: B,
}

impl<P, B: Default> RelocContext<P, B> {
    pub fn new(parent: P) -> Self {
        Self {
            parent,
            builder: Default::default(),
        }
    }
}

macro_rules! impl_push_op_for_reloc_context {
    ($t:ty) => {
        impl<P, B, S> PushOp<$t, S> for RelocContext<P, B>
        where
            B: PushOp<$t, S>,
            S: Clone,
        {
            fn push_op(&mut self, op: $t, span: S) {
                self.builder.push_op(op, span)
            }
        }
    };
}

impl_push_op_for_reloc_context! {LocationCounter}
impl_push_op_for_reloc_context! {i32}
impl_push_op_for_reloc_context! {BinOp}
impl_push_op_for_reloc_context! {ParamId}
impl_push_op_for_reloc_context! {FnCall}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::log::Log;
    use crate::model::Atom;

    use std::marker::PhantomData;

    type Expr<N, S> = crate::model::Expr<Atom<LocationCounter, N>, S>;

    pub(crate) struct MockBackend<A, T> {
        alloc: A,
        pub log: Log<T>,
    }

    #[derive(Debug, PartialEq)]
    pub enum BackendEvent<N, V: Source> {
        EmitItem(Item<V>),
        Reserve(V),
        SetOrigin(V),
        DefineSymbol((N, V::Span), V),
        StartSection((N, V::Span)),
    }

    impl<A, T> MockBackend<A, T> {
        pub fn new(alloc: A, log: Log<T>) -> Self {
            MockBackend { alloc, log }
        }
    }

    impl<A, T, S> Backend<S> for MockBackend<A, T>
    where
        A: AllocName<S>,
        T: From<BackendEvent<A::Name, Expr<A::Name, S>>>,
        S: Clone,
    {
        type ImmediateBuilder = RelocContext<Self, Expr<A::Name, S>>;
        type SymbolBuilder = MockSymbolBuilder<Self, A::Name, S>;

        fn build_immediate(self) -> Self::ImmediateBuilder {
            RelocContext::new(self)
        }

        fn define_symbol(self, name: Self::Name, span: S) -> Self::SymbolBuilder {
            MockSymbolBuilder {
                parent: self,
                name: (name, span),
                expr: Default::default(),
            }
        }
    }

    impl<A, T, S> PushOp<Name<A::Name>, S> for RelocContext<MockBackend<A, T>, Expr<A::Name, S>>
    where
        A: AllocName<S>,
        S: Clone,
    {
        fn push_op(&mut self, op: Name<A::Name>, span: S) {
            self.builder.push_op(op, span)
        }
    }

    impl<A, T, S> AllocName<S> for RelocContext<MockBackend<A, T>, Expr<A::Name, S>>
    where
        A: AllocName<S>,
        S: Clone,
    {
        type Name = A::Name;

        fn alloc_name(&mut self, span: S) -> Self::Name {
            self.parent.alloc_name(span)
        }
    }

    impl<A, T, S> Finish for RelocContext<MockBackend<A, T>, Expr<A::Name, S>>
    where
        A: AllocName<S>,
        S: Clone,
    {
        type Parent = MockBackend<A, T>;
        type Value = Expr<A::Name, S>;

        fn finish(self) -> (Self::Parent, Self::Value) {
            (self.parent, self.builder)
        }
    }

    pub struct MockSymbolBuilder<P, N, S> {
        pub parent: P,
        pub name: (N, S),
        pub expr: crate::model::Expr<Atom<LocationCounter, N>, S>,
    }

    impl<T, P, N, S: Clone> PushOp<T, S> for MockSymbolBuilder<P, N, S>
    where
        crate::model::Expr<Atom<LocationCounter, N>, S>: PushOp<T, S>,
    {
        fn push_op(&mut self, op: T, span: S) {
            self.expr.push_op(op, span)
        }
    }

    impl<A, T, S> Finish for MockSymbolBuilder<MockBackend<A, T>, A::Name, S>
    where
        A: AllocName<S>,
        T: From<BackendEvent<A::Name, Expr<A::Name, S>>>,
        S: Clone,
    {
        type Parent = MockBackend<A, T>;
        type Value = ();

        fn finish(self) -> (Self::Parent, Self::Value) {
            let parent = self.parent;
            parent
                .log
                .push(BackendEvent::DefineSymbol(self.name, self.expr));
            (parent, ())
        }
    }

    impl<A, T, S> AllocName<S> for MockSymbolBuilder<MockBackend<A, T>, A::Name, S>
    where
        A: AllocName<S>,
        S: Clone,
    {
        type Name = A::Name;

        fn alloc_name(&mut self, span: S) -> Self::Name {
            self.parent.alloc_name(span)
        }
    }

    impl<L> From<usize> for Atom<L, usize> {
        fn from(n: usize) -> Self {
            Atom::Name(n)
        }
    }

    impl<A: AllocName<S>, T, S: Clone> AllocName<S> for MockBackend<A, T> {
        type Name = A::Name;

        fn alloc_name(&mut self, span: S) -> Self::Name {
            self.alloc.alloc_name(span)
        }
    }

    impl<A, T, S> PartialBackend<S> for MockBackend<A, T>
    where
        A: AllocName<S>,
        T: From<BackendEvent<A::Name, Expr<A::Name, S>>>,
        S: Clone,
    {
        type Value = Expr<A::Name, S>;

        fn emit_item(&mut self, item: Item<Self::Value>) {
            self.log.push(BackendEvent::EmitItem(item))
        }

        fn reserve(&mut self, bytes: Self::Value) {
            self.log.push(BackendEvent::Reserve(bytes))
        }

        fn set_origin(&mut self, origin: Self::Value) {
            self.log.push(BackendEvent::SetOrigin(origin))
        }
    }

    impl<A, T, S> StartSection<A::Name, S> for MockBackend<A, T>
    where
        A: AllocName<S>,
        T: From<BackendEvent<A::Name, Expr<A::Name, S>>>,
        S: Clone,
    {
        fn start_section(&mut self, name: (A::Name, S)) {
            self.log.push(BackendEvent::StartSection(name))
        }
    }

    pub struct SerialIdAllocator(usize);

    impl SerialIdAllocator {
        pub fn new() -> Self {
            Self(0)
        }
    }

    impl<S: Clone> AllocName<S> for SerialIdAllocator {
        type Name = usize;

        fn alloc_name(&mut self, _: S) -> Self::Name {
            let id = self.0;
            self.0 += 1;
            id
        }
    }

    pub struct PanickingIdAllocator<I>(PhantomData<I>);

    impl<I> PanickingIdAllocator<I> {
        pub fn new() -> Self {
            Self(PhantomData)
        }
    }

    impl<I: Clone, S: Clone> AllocName<S> for PanickingIdAllocator<I> {
        type Name = I;

        fn alloc_name(&mut self, _: S) -> Self::Name {
            panic!("tried to allocate an ID")
        }
    }
}
