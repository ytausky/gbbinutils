use super::{Ident, Params, PushOp};

use crate::analysis::backend::{AllocName, Finish, FinishFnDef, LocationCounter, Name};
use crate::analysis::resolve::{NameTable, ResolvedIdent};
use crate::diag::{Diagnostics, Message};
use crate::model::{BinOp, FnCall, ParamId};

pub(super) trait RelocLookup<I, S> {
    type RelocId;

    fn reloc_lookup(&mut self, name: I, span: S) -> Self::RelocId;
}

impl<T, I, S> RelocLookup<I, S> for T
where
    T: AllocName<S> + NameTable<I, BackendEntry = <T as AllocName<S>>::Name> + Diagnostics<S>,
    S: Clone,
{
    type RelocId = T::Name;

    fn reloc_lookup(&mut self, name: I, span: S) -> Self::RelocId {
        match self.get(&name) {
            Some(ResolvedIdent::Backend(id)) => id.clone(),
            None => {
                let id = self.alloc_name(span.clone());
                self.insert(name, ResolvedIdent::Backend(id.clone()));
                id
            }
            Some(ResolvedIdent::Macro(_)) => {
                self.emit_diag(Message::MacroNameInExpr.at(span.clone()));
                self.alloc_name(span)
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

pub(super) trait WithParams<R, S>: Sized {
    fn with_params<'a>(
        self,
        params: &'a Params<R, S>,
    ) -> BuilderAdapter<Self, ConvertParams<'a, R, S>>;
}

impl<B, R, S> WithParams<R, S> for B
where
    B: PushOp<Name<Ident<R>>, S> + PushOp<ParamId, S>,
    R: PartialEq,
    S: Clone,
{
    fn with_params<'a>(
        self,
        params: &'a Params<R, S>,
    ) -> BuilderAdapter<Self, ConvertParams<'a, R, S>> {
        BuilderAdapter {
            builder: self,
            handler: ConvertParams { params },
        }
    }
}

pub(super) struct ConvertParams<'a, R, S> {
    params: &'a Params<R, S>,
}

impl<'a, B, R, S> NameHandler<B, Ident<R>, S> for ConvertParams<'a, R, S>
where
    B: PushOp<Name<Ident<R>>, S> + PushOp<ParamId, S>,
    R: PartialEq,
    S: Clone,
{
    fn handle(&mut self, ident: Ident<R>, span: S, builder: &mut B) {
        let param = self
            .params
            .0
            .iter()
            .position(|param| param.name == ident.name)
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
    B: AllocName<S>
        + NameTable<I, BackendEntry = <B as AllocName<S>>::Name>
        + PushOp<Name<<B as AllocName<S>>::Name>, S>
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

impl<B, H, S> Finish<S> for BuilderAdapter<B, H>
where
    B: Finish<S>,
    S: Clone,
{
    type Parent = B::Parent;
    type Value = B::Value;

    fn finish(self) -> (Self::Parent, Self::Value) {
        self.builder.finish()
    }
}

impl<B, H> FinishFnDef for BuilderAdapter<B, H>
where
    B: FinishFnDef,
{
    type Return = B::Return;

    fn finish_fn_def(self) -> Self::Return {
        self.builder.finish_fn_def()
    }
}

delegate_diagnostics! {
    {B: Diagnostics<S>, H, S}, BuilderAdapter<B, H>, {builder}, B, S
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::backend::{BackendEvent, SerialIdAllocator};
    use crate::analysis::resolve::{BasicNameTable, NameTableEvent};
    use crate::analysis::session::MockBuilder;
    use crate::diag::{DiagnosticsEvent, MockSpan};
    use crate::log::Log;
    use crate::model::{Atom, ParamId};

    type Expr<N, S> = crate::model::Expr<Atom<LocationCounter, N>, S>;

    #[derive(Debug, PartialEq)]
    enum Event<N, S: Clone> {
        Backend(BackendEvent<N, Expr<N, S>>),
        Diagnostics(DiagnosticsEvent<S>),
        NameTable(NameTableEvent<usize, usize>),
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

    impl<N, S: Clone> From<NameTableEvent<usize, usize>> for Event<N, S> {
        fn from(event: NameTableEvent<usize, usize>) -> Self {
            Event::NameTable(event)
        }
    }

    #[test]
    fn translate_param() {
        let name: Ident<_> = "param".into();
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
        let param: Ident<_> = "param".into();
        let builder: Expr<_, _> = Default::default();
        let params = (vec![param.clone()], vec![()]);
        let mut adapter = builder.with_params(&params);
        let unrelated = Name(Ident::from("ident"));
        adapter.push_op(unrelated.clone(), ());
        let mut expected: Expr<_, _> = Default::default();
        expected.push_op(unrelated, ());
        assert_eq!(adapter.builder, expected)
    }

    #[test]
    fn resolve_known_ident() {
        let ident = Ident::from("ident");
        let reloc = 42;
        let log = Log::<Event<usize, ()>>::new();
        let mut idents = BasicNameTable::new();
        idents.insert(ident.clone(), ResolvedIdent::Backend(reloc));
        let mut builder =
            MockBuilder::from_components(SerialIdAllocator::new(), idents, log).resolve_names();
        builder.push_op(Name(ident), ());
        assert_eq!(builder.finish().1, Expr::from_atom(Atom::Name(reloc), ()))
    }

    #[test]
    fn resolve_unknown_ident() {
        let ident = Ident::from("ident");
        let id = 0;
        let log = Log::<Event<usize, ()>>::new();
        {
            let mut builder = MockBuilder::from_components(
                SerialIdAllocator::new(),
                BasicNameTable::new(),
                log.clone(),
            )
            .resolve_names();
            builder.push_op(Name(ident.clone()), ());
            assert_eq!(builder.finish().1, Expr::from_atom(Atom::Name(id), ()));
        }
        assert_eq!(
            log.into_inner(),
            [NameTableEvent::Insert(ident, ResolvedIdent::Backend(id)).into()]
        )
    }

    #[test]
    fn diagnose_macro_name_in_expr() {
        let ident = Ident::from("my_macro");
        let span = MockSpan::from("ident");
        let mut table = BasicNameTable::new();
        table.insert(ident.clone(), ResolvedIdent::Macro(0));
        let log = Log::<Event<usize, MockSpan<_>>>::new();
        {
            let mut builder =
                MockBuilder::from_components(SerialIdAllocator::new(), table, log.clone())
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
