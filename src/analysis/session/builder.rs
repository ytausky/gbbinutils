use super::{Downstream, SessionComponents};

use crate::analysis::backend::{AllocName, Finish, Name, PushOp, RelocContext};
use crate::diag::Diagnostics;
use crate::model::{BinOp, FnCall, LocationCounter, ParamId};

use std::ops::DerefMut;

impl<U, B, N, D, S> PushOp<Name<B::Name>, S> for SessionComponents<U, B, N, D>
where
    B: AllocName<S> + PushOp<Name<<B as AllocName<S>>::Name>, S>,
    S: Clone,
{
    fn push_op(&mut self, name: Name<B::Name>, span: S) {
        self.downstream.backend.push_op(name, span)
    }
}

impl<U, B: Finish, N, D> Finish for SessionComponents<U, B, N, D> {
    type Parent = SessionComponents<U, B::Parent, N, D>;
    type Value = B::Value;

    fn finish(self) -> (Self::Parent, Self::Value) {
        let (backend, value) = self.downstream.backend.finish();
        let parent = SessionComponents {
            upstream: self.upstream,
            downstream: Downstream {
                backend,
                names: self.downstream.names,
                diagnostics: self.downstream.diagnostics,
            },
        };
        (parent, value)
    }
}

delegate_diagnostics! {
    {'a, P, B, N, D: DerefMut, S},
    {D::Target: Diagnostics<S>},
    RelocContext<P, Downstream<B, N, D>>,
    {builder.diagnostics},
    D::Target,
    S
}

macro_rules! impl_push_op_for_session_components {
    ($t:ty) => {
        impl<U, B, N, D, S> PushOp<$t, S> for SessionComponents<U, B, N, D>
        where
            B: PushOp<$t, S>,
            S: Clone,
        {
            fn push_op(&mut self, op: $t, span: S) {
                self.downstream.push_op(op, span)
            }
        }
    };
}

impl_push_op_for_session_components! {LocationCounter}
impl_push_op_for_session_components! {i32}
impl_push_op_for_session_components! {BinOp}
impl_push_op_for_session_components! {ParamId}
impl_push_op_for_session_components! {FnCall}
