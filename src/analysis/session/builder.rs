use super::{CompositeSession, Downstream};

use crate::analysis::backend::{AllocName, Finish, FinishFnDef, Name, PushOp, RelocContext};
use crate::analysis::resolve::{Ident, NameTable, ResolvedIdent};
use crate::diag::Diagnostics;

use std::ops::DerefMut;

pub(super) type Builder<U, B, N, D> = RelocContext<U, Downstream<B, N, D>>;

impl<U, B: AllocName<S>, N, D, S: Clone> AllocName<S> for Builder<U, B, N, D> {
    type Name = B::Name;

    fn alloc_name(&mut self, span: S) -> Self::Name {
        self.builder.backend.alloc_name(span)
    }
}

impl<U, B, N, D, R> NameTable<Ident<R>> for Builder<U, B, N, D>
where
    N: DerefMut,
    N::Target: NameTable<Ident<R>>,
{
    type BackendEntry = <N::Target as NameTable<Ident<R>>>::BackendEntry;
    type MacroEntry = <N::Target as NameTable<Ident<R>>>::MacroEntry;

    fn get(&self, ident: &Ident<R>) -> Option<ResolvedIdent<Self::BackendEntry, Self::MacroEntry>> {
        self.builder.names.get(ident)
    }

    fn insert(
        &mut self,
        ident: Ident<R>,
        entry: ResolvedIdent<Self::BackendEntry, Self::MacroEntry>,
    ) {
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

impl<U, B: Finish<S>, N, D, S: Clone> Finish<S> for Builder<U, B, N, D> {
    type Parent = CompositeSession<U, B::Parent, N, D>;
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

impl<U, B: FinishFnDef, N, D> FinishFnDef for Builder<U, B, N, D> {
    type Return = CompositeSession<U, B::Return, N, D>;

    fn finish_fn_def(self) -> Self::Return {
        CompositeSession {
            upstream: self.parent,
            downstream: self.builder.replace_backend(FinishFnDef::finish_fn_def),
        }
    }
}

delegate_diagnostics! {
    {'a, P, B, N, D: Diagnostics<S>, S},
    RelocContext<P, Downstream<B, N, &'a mut D>>,
    {builder.diagnostics},
    D,
    S
}

#[cfg(test)]
mod mock {
    use super::{Diagnostics, Downstream, RelocContext};

    delegate_diagnostics! {
        {P, B, N, D: Diagnostics<S>, S},
        RelocContext<P, Downstream<B, N, Box<D>>>,
        {builder.diagnostics},
        D,
        S
    }
}
