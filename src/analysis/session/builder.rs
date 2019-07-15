use super::{CompositeSession, Downstream, StringSource, Upstream, Wrapper};

use crate::analysis::backend::{AllocName, Finish, FinishFnDef, Name, PushOp, RelocContext};
use crate::analysis::resolve::{Ident, NameTable};
use crate::diag::span::{AddMacroDef, SpanSource};
use crate::diag::{Diagnostics, DiagnosticsSystem};
use crate::model::{BinOp, FnCall, LocationCounter, ParamId};

pub(super) type Builder<'a, 'b, C, A, B, N, D> = RelocContext<
    Upstream<
        'a,
        'b,
        C,
        A,
        <C as StringSource>::StringRef,
        <D as AddMacroDef<<D as SpanSource>::Span>>::MacroDefHandle,
    >,
    Downstream<B, &'a mut N, Wrapper<'a, D>>,
>;

impl<'a, 'b, C, A, B, N, D> PushOp<Name<Ident<C::StringRef>>, D::Span>
    for Builder<'a, 'b, C, A, B, N, D>
where
    C: StringSource,
    B: AllocName<D::Span> + PushOp<Name<<B as AllocName<D::Span>>::Name>, D::Span>,
    N: NameTable<Ident<C::StringRef>, BackendEntry = B::Name>,
    D: DiagnosticsSystem,
{
    fn push_op(&mut self, Name(ident): Name<Ident<C::StringRef>>, span: D::Span) {
        let id = self.builder.look_up_symbol(ident, &span);
        self.builder.backend.push_op(Name(id), span)
    }
}

impl<'a, 'b, C, A, B, N, D> Finish<D::Span> for Builder<'a, 'b, C, A, B, N, D>
where
    C: StringSource,
    B: Finish<D::Span>,
    D: DiagnosticsSystem,
{
    type Parent = CompositeSession<'a, 'b, C, A, B::Parent, N, D>;
    type Value = B::Value;

    fn finish(self) -> (Self::Parent, Self::Value) {
        let (backend, value) = self.builder.backend.finish();
        let parent = CompositeSession {
            upstream: self.parent,
            downstream: Downstream {
                backend,
                names: self.builder.names,
                diagnostics: self.builder.diagnostics,
            },
        };
        (parent, value)
    }
}

impl<'a, 'b, C, A, B, N, D> FinishFnDef for Builder<'a, 'b, C, A, B, N, D>
where
    C: StringSource,
    B: FinishFnDef,
    D: DiagnosticsSystem,
{
    type Return = CompositeSession<'a, 'b, C, A, B::Return, N, D>;

    fn finish_fn_def(self) -> Self::Return {
        CompositeSession {
            upstream: self.parent,
            downstream: self.builder.replace_backend(FinishFnDef::finish_fn_def),
        }
    }
}

macro_rules! impl_push_op_for_downstream {
    ($t:ty) => {
        impl<B, N, D, S> PushOp<$t, S> for Downstream<B, N, D>
        where
            B: PushOp<$t, S>,
            S: Clone,
        {
            fn push_op(&mut self, op: $t, span: S) {
                self.backend.push_op(op, span)
            }
        }
    };
}

impl_push_op_for_downstream! {LocationCounter}
impl_push_op_for_downstream! {i32}
impl_push_op_for_downstream! {BinOp}
impl_push_op_for_downstream! {ParamId}
impl_push_op_for_downstream! {FnCall}

delegate_diagnostics! {
    {P, B, N, D: Diagnostics<S>, S},
    RelocContext<P, Downstream<B, N, D>>,
    {builder.diagnostics},
    D,
    S
}

delegate_diagnostics! {
    {'a, D: Diagnostics<S>, S},
    Wrapper<'a, D>,
    {0},
    D,
    S
}
