use super::{Params, PushOp};

use crate::analyze::resolve::{NameTable, ResolvedName};
use crate::diag::{Diagnostics, Message};
use crate::expr::{BinOp, FnCall, LocationCounter, ParamId};
use crate::object::builder::{AllocSymbol, Finish, Name, SymbolSource};

pub(super) trait RelocLookup<I, S> {
    type RelocId;

    fn reloc_lookup(&mut self, name: I, span: S) -> Self::RelocId;
}

impl<T, I, S> RelocLookup<I, S> for T
where
    T: AllocSymbol<S> + NameTable<I> + Diagnostics<S>,
    S: Clone,
{
    type RelocId = T::SymbolId;

    fn reloc_lookup(&mut self, name: I, span: S) -> Self::RelocId {
        match self.get(&name) {
            Some(ResolvedName::Keyword(_)) => unimplemented!(),
            Some(ResolvedName::Symbol(id)) => id.clone(),
            None => {
                let id = self.alloc_symbol(span.clone());
                self.insert(name, ResolvedName::Symbol(id.clone()));
                id
            }
            Some(ResolvedName::Macro(_)) => {
                self.emit_diag(Message::MacroNameInExpr.at(span.clone()));
                self.alloc_symbol(span)
            }
        }
    }
}

pub(super) struct BuilderAdapter<B, H> {
    builder: B,
    handler: H,
}

pub(super) trait NameHandler<B, T, S> {
    fn handle(&mut self, name: T, span: S, builder: &mut B);
}

impl<B, H, T, S> PushOp<Name<T>, S> for BuilderAdapter<B, H>
where
    H: NameHandler<B, T, S>,
    S: Clone,
{
    fn push_op(&mut self, Name(name): Name<T>, span: S) {
        self.handler.handle(name, span, &mut self.builder)
    }
}

pub(super) trait WithParams<I, S>: Sized {
    fn with_params<'a>(
        self,
        params: &'a Params<I, S>,
    ) -> BuilderAdapter<Self, ConvertParams<'a, I, S>>;
}

impl<B, I, S> WithParams<I, S> for B
where
    B: PushOp<Name<I>, S> + PushOp<ParamId, S>,
    I: PartialEq,
    S: Clone,
{
    fn with_params<'a>(
        self,
        params: &'a Params<I, S>,
    ) -> BuilderAdapter<Self, ConvertParams<'a, I, S>> {
        BuilderAdapter {
            builder: self,
            handler: ConvertParams { params },
        }
    }
}

pub(super) struct ConvertParams<'a, I, S> {
    params: &'a Params<I, S>,
}

impl<'a, B, I, S> NameHandler<B, I, S> for ConvertParams<'a, I, S>
where
    B: PushOp<Name<I>, S> + PushOp<ParamId, S>,
    I: PartialEq,
    S: Clone,
{
    fn handle(&mut self, ident: I, span: S, builder: &mut B) {
        let param = self
            .params
            .0
            .iter()
            .position(|param| *param == ident)
            .map(ParamId);
        if let Some(id) = param {
            builder.push_op(id, span)
        } else {
            builder.push_op(Name(ident), span)
        }
    }
}

pub(super) trait ResolveNames: Sized {
    fn resolve_names(self) -> BuilderAdapter<Self, NameResolver>;
}

impl<B> ResolveNames for B {
    fn resolve_names(self) -> BuilderAdapter<Self, NameResolver> {
        BuilderAdapter {
            builder: self,
            handler: NameResolver,
        }
    }
}

pub(super) struct NameResolver;

impl<B, I, S> NameHandler<B, I, S> for NameResolver
where
    B: AllocSymbol<S>
        + NameTable<I>
        + PushOp<Name<<B as SymbolSource>::SymbolId>, S>
        + Diagnostics<S>,
    S: Clone,
{
    fn handle(&mut self, ident: I, span: S, builder: &mut B) {
        let id = builder.reloc_lookup(ident, span.clone());
        builder.push_op(Name(id), span)
    }
}

macro_rules! impl_push_op_for_builder_adapter {
    ($t:ty) => {
        impl<B, H, S> PushOp<$t, S> for BuilderAdapter<B, H>
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

impl_push_op_for_builder_adapter! {LocationCounter}
impl_push_op_for_builder_adapter! {i32}
impl_push_op_for_builder_adapter! {BinOp}
impl_push_op_for_builder_adapter! {FnCall}
impl_push_op_for_builder_adapter! {ParamId}

impl<B: Finish, H> Finish for BuilderAdapter<B, H> {
    type Parent = B::Parent;
    type Value = B::Value;

    fn finish(self) -> (Self::Parent, Self::Value) {
        self.builder.finish()
    }
}

delegate_diagnostics! {
    {B: Diagnostics<S>, H, S}, BuilderAdapter<B, H>, {builder}, B, S
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analyze::resolve::NameTableEvent;
    use crate::analyze::semantics::Keyword;
    use crate::analyze::session::MockBuilder;
    use crate::diag::{DiagnosticsEvent, MockSpan};
    use crate::expr::{Atom, ParamId};
    use crate::log::Log;
    use crate::object::builder::mock::BackendEvent;
    use crate::object::builder::PushOp;

    type Expr<N, S> = crate::expr::Expr<Atom<LocationCounter, N>, S>;

    #[derive(Debug, PartialEq)]
    enum Event<N, S: Clone> {
        Backend(BackendEvent<N, Expr<N, S>>),
        Diagnostics(DiagnosticsEvent<S>),
        NameTable(NameTableEvent<Keyword, usize, usize>),
    }

    impl<N, S: Clone> From<BackendEvent<N, Expr<N, S>>> for Event<N, S> {
        fn from(event: BackendEvent<N, Expr<N, S>>) -> Self {
            Event::Backend(event)
        }
    }

    impl<N, S: Clone> From<DiagnosticsEvent<S>> for Event<N, S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            Event::Diagnostics(event)
        }
    }

    impl<N, S: Clone> From<NameTableEvent<Keyword, usize, usize>> for Event<N, S> {
        fn from(event: NameTableEvent<Keyword, usize, usize>) -> Self {
            Event::NameTable(event)
        }
    }

    #[test]
    fn translate_param() {
        let name: String = "param".into();
        let builder: Expr<_, _> = Default::default();
        let params = (vec![name.clone()], vec![()]);
        let mut adapter = builder.with_params(&params);
        adapter.push_op(Name(name), ());
        let mut expected: Expr<_, _> = Default::default();
        expected.push_op(ParamId(0), ());
        assert_eq!(adapter.builder, expected)
    }

    #[test]
    fn pass_through_non_param() {
        let param: String = "param".into();
        let builder: Expr<_, _> = Default::default();
        let params = (vec![param.clone()], vec![()]);
        let mut adapter = builder.with_params(&params);
        let unrelated = Name(String::from("ident"));
        adapter.push_op(unrelated.clone(), ());
        let mut expected: Expr<_, _> = Default::default();
        expected.push_op(unrelated, ());
        assert_eq!(adapter.builder, expected)
    }

    #[test]
    fn resolve_known_ident() {
        let ident = String::from("ident");
        let reloc = 42;
        let log = Log::<Event<usize, ()>>::new();
        let mut builder = MockBuilder::with_predefined_names(
            log,
            vec![(ident.clone(), ResolvedName::Symbol(reloc))],
        )
        .resolve_names();
        builder.push_op(Name(ident), ());
        assert_eq!(builder.finish().1, Expr::from_atom(Atom::Name(reloc), ()))
    }

    #[test]
    fn resolve_unknown_ident() {
        let ident = String::from("ident");
        let id = 0;
        let log = Log::<Event<usize, ()>>::new();
        {
            let mut builder = MockBuilder::with_name_resolution(log.clone()).resolve_names();
            builder.push_op(Name(ident.clone()), ());
            assert_eq!(builder.finish().1, Expr::from_atom(Atom::Name(id), ()));
        }
        assert_eq!(
            log.into_inner(),
            [NameTableEvent::Insert(ident, ResolvedName::Symbol(id)).into()]
        )
    }

    #[test]
    fn diagnose_macro_name_in_expr() {
        let ident = String::from("my_macro");
        let span = MockSpan::from("ident");
        let log = Log::<Event<usize, MockSpan<_>>>::new();
        {
            let mut builder = MockBuilder::with_predefined_names(
                log.clone(),
                vec![(ident.clone(), ResolvedName::Macro(0))],
            )
            .resolve_names();
            builder.push_op(Name(ident), span.clone());
            assert_eq!(
                builder.finish().1,
                Expr::from_atom(Atom::Name(0), span.clone())
            );
        }
        assert_eq!(
            log.into_inner(),
            [DiagnosticsEvent::EmitDiag(Message::MacroNameInExpr.at(span).into()).into()]
        )
    }
}
