use super::SessionComponents;

use crate::expr::{BinOp, FnCall, LocationCounter, ParamId};
use crate::object::builder::{AllocName, Finish, Name, PushOp};

impl<U, B, N, D, S> PushOp<Name<B::Name>, S> for SessionComponents<U, B, N, D>
where
    B: AllocName<S> + PushOp<Name<<B as AllocName<S>>::Name>, S>,
    S: Clone,
{
    fn push_op(&mut self, name: Name<B::Name>, span: S) {
        self.backend.push_op(name, span)
    }
}

impl<U, B: Finish, N, D> Finish for SessionComponents<U, B, N, D> {
    type Parent = SessionComponents<U, B::Parent, N, D>;
    type Value = B::Value;

    fn finish(self) -> (Self::Parent, Self::Value) {
        let (backend, value) = self.backend.finish();
        let parent = SessionComponents {
            upstream: self.upstream,
            backend,
            names: self.names,
            diagnostics: self.diagnostics,
        };
        (parent, value)
    }
}

macro_rules! impl_push_op_for_session_components {
    ($t:ty) => {
        impl<U, B, N, D, S> PushOp<$t, S> for SessionComponents<U, B, N, D>
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

impl_push_op_for_session_components! {LocationCounter}
impl_push_op_for_session_components! {i32}
impl_push_op_for_session_components! {BinOp}
impl_push_op_for_session_components! {ParamId}
impl_push_op_for_session_components! {FnCall}
