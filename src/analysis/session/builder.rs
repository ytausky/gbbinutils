use super::{Downstream, SessionComponents};

use crate::analysis::backend::{AllocName, Finish, Name, PushOp, RelocContext};
use crate::analysis::resolve::{NameTable, ResolvedIdent};
use crate::diag::Diagnostics;

use std::ops::DerefMut;

pub(super) type Builder<U, B, N, D> = RelocContext<U, Downstream<B, N, D>>;

impl<U, B: AllocName<S>, N, D, S: Clone> AllocName<S> for Builder<U, B, N, D> {
    type Name = B::Name;

    fn alloc_name(&mut self, span: S) -> Self::Name {
        self.builder.backend.alloc_name(span)
    }
}

impl<U, B, N, D, I> NameTable<I> for Builder<U, B, N, D>
where
    N: DerefMut,
    N::Target: NameTable<I>,
{
    type BackendEntry = <N::Target as NameTable<I>>::BackendEntry;
    type MacroEntry = <N::Target as NameTable<I>>::MacroEntry;

    fn get(&self, ident: &I) -> Option<ResolvedIdent<Self::BackendEntry, Self::MacroEntry>> {
        self.builder.names.get(ident)
    }

    fn insert(&mut self, ident: I, entry: ResolvedIdent<Self::BackendEntry, Self::MacroEntry>) {
        self.builder.names.insert(ident, entry)
    }
}

impl<U, B, N, D, S> PushOp<Name<B::Name>, S> for Builder<U, B, N, D>
where
    B: AllocName<S> + PushOp<Name<<B as AllocName<S>>::Name>, S>,
    S: Clone,
{
    fn push_op(&mut self, name: Name<B::Name>, span: S) {
        self.builder.backend.push_op(name, span)
    }
}

impl<U, B: Finish, N, D> Finish for Builder<U, B, N, D> {
    type Parent = SessionComponents<U, B::Parent, N, D>;
    type Value = B::Value;

    fn finish(self) -> (Self::Parent, Self::Value) {
        let (backend, value) = self.builder.backend.finish();
        let parent = SessionComponents {
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

delegate_diagnostics! {
    {'a, P, B, N, D: DerefMut, S},
    {D::Target: Diagnostics<S>},
    RelocContext<P, Downstream<B, N, D>>,
    {builder.diagnostics},
    D::Target,
    S
}
