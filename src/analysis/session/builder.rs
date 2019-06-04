use super::{CompositeSession, Downstream, Upstream, Wrapper};

use crate::analysis::backend::{AllocName, Finish, FinishFnDef, PushOp, RelocContext};
use crate::analysis::resolve::{Ident, NameTable};
use crate::diag::Diagnostics;
use crate::model::{BinOp, FnCall, LocationCounter, ParamId};

pub(super) type Builder<'a, 'b, C, A, B, N, D> =
    RelocContext<Upstream<'a, 'b, C, A>, Downstream<B, &'a mut N, Wrapper<'a, D>>>;

impl<'a, 'b, C, A, B, N, D, R, S> PushOp<Ident<R>, S> for Builder<'a, 'b, C, A, B, N, D>
where
    B: AllocName<S> + PushOp<<B as AllocName<S>>::Name, S>,
    N: NameTable<Ident<R>, BackendEntry = B::Name>,
    D: Diagnostics<S>,
    S: Clone,
{
    fn push_op(&mut self, ident: Ident<R>, span: S) {
        let id = self.builder.look_up_symbol(ident, &span);
        self.builder.backend.push_op(id, span)
    }
}

impl<'a, 'b, C, A, B, N, D, S> Finish<S> for Builder<'a, 'b, C, A, B, N, D>
where
    B: Finish<S>,
    S: Clone,
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
    B: FinishFnDef,
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
