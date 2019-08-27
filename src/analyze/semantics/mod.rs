use self::params::*;
use self::resolve::{NameTable, ResolvedName};

use super::macros::MacroSource;
use super::reentrancy::Params;

use crate::diag::Diagnostics;
use crate::expr::{BinOp, FnCall, LocationCounter, ParamId};
use crate::object::builder::{AllocSymbol, Finish, Name, PushOp, SymbolSource};

use std::ops::{Deref, DerefMut};

macro_rules! set_state {
    ($session:expr, $state:expr) => {
        $crate::analyze::semantics::Session {
            reentrancy: $session.reentrancy,
            names: $session.names,
            builder: $session.builder,
            state: $state,
        }
    };
}

pub(in crate::analyze::semantics) mod cpu_instr;
pub(in crate::analyze::semantics) mod directive;

mod actions;
mod arg;
mod params;
pub(super) mod resolve;

pub(super) struct Session<R, N, B, S> {
    reentrancy: R,
    names: N,
    builder: B,
    state: S,
}

impl<R, N, B, S> Session<R, N, B, S> {
    #[cfg(test)]
    fn map_names<F: FnOnce(N) -> T, T>(self, f: F) -> Session<R, T, B, S> {
        Session {
            reentrancy: self.reentrancy,
            names: f(self.names),
            builder: self.builder,
            state: self.state,
        }
    }

    fn map_builder<F: FnOnce(B) -> T, T>(self, f: F) -> Session<R, N, T, S> {
        Session {
            reentrancy: self.reentrancy,
            names: self.names,
            builder: f(self.builder),
            state: self.state,
        }
    }

    fn map_state<F: FnOnce(S) -> T, T>(self, f: F) -> Session<R, N, B, T> {
        Session {
            reentrancy: self.reentrancy,
            names: self.names,
            builder: self.builder,
            state: f(self.state),
        }
    }
}

delegate_diagnostics! {
    {R: Diagnostics<Span>, N, B, S, Span}, Session<R, N, B, S>, {reentrancy}, R, Span
}

impl<R, N, B, S> MacroSource for Session<R, N, B, S>
where
    N: Deref,
    N::Target: MacroSource,
{
    type MacroId = <N::Target as MacroSource>::MacroId;
}

impl<R, N, B, S> SymbolSource for Session<R, N, B, S>
where
    N: Deref,
    N::Target: SymbolSource,
{
    type SymbolId = <N::Target as SymbolSource>::SymbolId;
}

impl<R, N, B, S, Span> AllocSymbol<Span> for Session<R, N, B, S>
where
    N: Deref,
    N::Target: SymbolSource<SymbolId = B::SymbolId>,
    B: AllocSymbol<Span>,
    Span: Clone,
{
    fn alloc_symbol(&mut self, span: Span) -> Self::SymbolId {
        self.builder.alloc_symbol(span)
    }
}

impl<R, N, B, S, I> NameTable<I> for Session<R, N, B, S>
where
    N: DerefMut,
    N::Target: NameTable<I>,
{
    type Keyword = <N::Target as NameTable<I>>::Keyword;

    fn resolve_name(
        &mut self,
        ident: &I,
    ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>> {
        self.names.resolve_name(ident)
    }

    fn define_name(
        &mut self,
        ident: I,
        entry: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>,
    ) {
        self.names.define_name(ident, entry)
    }
}

impl<R, N, B: Finish, S> Finish for Session<R, N, B, S> {
    type Value = B::Value;
    type Parent = Session<R, N, B::Parent, S>;

    fn finish(self) -> (Self::Parent, Self::Value) {
        let (builder, value) = self.builder.finish();
        (
            Session {
                reentrancy: self.reentrancy,
                names: self.names,
                builder,
                state: self.state,
            },
            value,
        )
    }
}

impl<R, N, B, S, Span, SymbolId> PushOp<Name<SymbolId>, Span> for Session<R, N, B, S>
where
    B: PushOp<Name<SymbolId>, Span>,
    Span: Clone,
{
    fn push_op(&mut self, op: Name<SymbolId>, span: Span) {
        self.builder.push_op(op, span)
    }
}

macro_rules! impl_push_op_for_session {
    ($t:ty) => {
        impl<R, N, B: PushOp<$t, Span>, S, Span: Clone> PushOp<$t, Span> for Session<R, N, B, S> {
            fn push_op(&mut self, op: $t, span: Span) {
                self.builder.push_op(op, span)
            }
        }
    };
}

impl_push_op_for_session! {LocationCounter}
impl_push_op_for_session! {i32}
impl_push_op_for_session! {BinOp}
impl_push_op_for_session! {ParamId}
impl_push_op_for_session! {FnCall}

#[cfg(test)]
mod mock {
    use super::actions::Keyword;
    use super::resolve::{BasicNameTable, MockNameTable};
    use super::*;

    use crate::analyze::macros::mock::MockMacroId;
    use crate::diag::{DiagnosticsEvent, Merge, MockDiagnostics};
    use crate::expr::{Atom, Expr};
    use crate::log::Log;
    use crate::object::builder::mock::{
        BackendEvent, MockBackend, MockSymbolId, SerialIdAllocator,
    };
    use crate::object::builder::{Backend, RelocContext};

    pub(super) type MockExprBuilder<T, S> = Session<
        MockDiagnostics<T, S>,
        Box<MockNameTable<BasicNameTable<&'static Keyword, MockMacroId, MockSymbolId>, T>>,
        RelocContext<
            MockBackend<SerialIdAllocator<MockSymbolId>, T>,
            Expr<Atom<LocationCounter, MockSymbolId>, S>,
        >,
        (),
    >;

    impl<T, S> MockExprBuilder<T, S>
    where
        T: From<BackendEvent<MockSymbolId, Expr<Atom<LocationCounter, MockSymbolId>, S>>>,
        T: From<DiagnosticsEvent<S>>,
        S: Clone + Merge,
    {
        pub fn with_log(log: Log<T>) -> Self {
            Self::with_name_table_entries(log, std::iter::empty())
        }

        pub fn with_name_table_entries<I>(log: Log<T>, entries: I) -> Self
        where
            I: IntoIterator<
                Item = (
                    String,
                    ResolvedName<&'static Keyword, MockMacroId, MockSymbolId>,
                ),
            >,
        {
            let mut names = BasicNameTable::default();
            for (ident, resolution) in entries {
                names.define_name(ident, resolution)
            }
            Session {
                reentrancy: MockDiagnostics::new(log.clone()),
                names: Box::new(MockNameTable::new(names, log.clone())),
                builder: MockBackend::new(SerialIdAllocator::new(MockSymbolId), log).build_const(),
                state: (),
            }
        }
    }
}
