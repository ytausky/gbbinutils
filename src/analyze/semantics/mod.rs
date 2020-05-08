use self::arg::{Arg, OperandSymbol};
use self::builtin_instr::BuiltinInstr;
use self::keywords::{BindingDirective, BuiltinMnemonic, FreeBuiltinMnemonic};
use self::params::*;
use self::resolve::{NameTable, ResolvedName};

use super::macros::MacroSource;
use super::reentrancy::{Params, ReentrancyActions};
use super::syntax::actions::LineRule;
use super::{IdentSource, StringSource, TokenSeq};

use crate::diag::span::SpanSource;
use crate::diag::Diagnostics;
use crate::expr::{BinOp, FnCall, LocationCounter, ParamId};
use crate::object::builder::{AllocSymbol, Finish, Name, PartialBackend, PushOp, SymbolSource};

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

mod actions;
mod arg;
mod builtin_instr;
mod keywords;
mod params;
pub(super) mod resolve;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum Keyword<B, F> {
    BuiltinMnemonic(BuiltinMnemonic<B, F>),
    Operand(OperandSymbol),
}

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

impl<R, N, B, S, Ident> NameTable<Ident> for Session<R, N, B, S>
where
    N: DerefMut,
    N::Target: NameTable<Ident>,
{
    type Keyword = <N::Target as NameTable<Ident>>::Keyword;

    fn resolve_name(
        &mut self,
        ident: &Ident,
    ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>> {
        self.names.resolve_name(ident)
    }

    fn define_name(
        &mut self,
        ident: Ident,
        entry: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>,
    ) {
        self.names.define_name(ident, entry)
    }
}

impl<R, N, B: Finish, S> Finish for Session<R, N, B, S> {
    type Value = B::Value;
    type Parent = Session<R, N, B::Parent, S>;

    fn finish(self) -> (Self::Parent, Option<Self::Value>) {
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
        impl<R, N, B, S, Span> PushOp<$t, Span> for Session<R, N, B, S>
        where
            B: PushOp<$t, Span>,
            Span: Clone,
        {
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

type TokenStreamSemantics<R, N, B> = Session<
    R,
    N,
    B,
    TokenStreamState<
        <R as IdentSource>::Ident,
        <R as StringSource>::StringRef,
        <R as SpanSource>::Span,
    >,
>;

#[derive(Debug, PartialEq)]
pub(super) struct TokenStreamState<I, R, S> {
    mode: LineRule<InstrLineState<I, S>, TokenLineState<I, R, S>>,
}

impl<I, R, S> TokenStreamState<I, R, S> {
    fn new() -> Self {
        Self {
            mode: LineRule::InstrLine(InstrLineState::new()),
        }
    }
}

impl<R, N, B> TokenStreamSemantics<R, N, B>
where
    R: ReentrancyActions,
    R::Ident: for<'r> From<&'r str>,
    N: DerefMut,
    N::Target:
        NameTable<R::Ident, Keyword = &'static Keyword<BindingDirective, FreeBuiltinMnemonic>>,
{
    pub fn from_components(reentrancy: R, mut names: N, builder: B) -> Self {
        for (ident, keyword) in keywords::KEYWORDS {
            names.define_name((*ident).into(), ResolvedName::Keyword(keyword))
        }
        Self {
            reentrancy,
            names,
            builder,
            state: TokenStreamState::new(),
        }
    }
}

type InstrLineSemantics<R, N, B> =
    Session<R, N, B, InstrLineState<<R as IdentSource>::Ident, <R as SpanSource>::Span>>;

#[derive(Debug, PartialEq)]
pub(super) struct InstrLineState<I, S> {
    label: Option<Label<I, S>>,
}

impl<I, S> InstrLineState<I, S> {
    fn new() -> Self {
        Self { label: None }
    }
}

type Label<I, S> = ((I, S), Params<I, S>);

type TokenLineSemantics<R, N, B> = Session<
    R,
    N,
    B,
    TokenLineState<
        <R as IdentSource>::Ident,
        <R as StringSource>::StringRef,
        <R as SpanSource>::Span,
    >,
>;

#[derive(Debug, PartialEq)]
pub(super) struct TokenLineState<I, R, S> {
    context: TokenLineContext<I, R, S>,
}

#[derive(Debug, PartialEq)]
pub(in crate::analyze) enum TokenLineContext<I, R, S> {
    FalseIf,
    MacroDef(MacroDefState<I, R, S>),
}

#[derive(Debug, PartialEq)]
pub(in crate::analyze) struct MacroDefState<I, R, S> {
    label: Option<Label<I, S>>,
    tokens: TokenSeq<I, R, S>,
}

impl<I, R, S> MacroDefState<I, R, S> {
    fn new(label: Option<Label<I, S>>) -> Self {
        Self {
            label,
            tokens: (Vec::new(), Vec::new()),
        }
    }
}

type BuiltinInstrSemantics<R, N, B> =
    Session<R, N, B, BuiltinInstrState<R, <B as PartialBackend<<R as SpanSource>::Span>>::Value>>;

pub(in crate::analyze) struct BuiltinInstrState<R, V>
where
    R: ReentrancyActions,
{
    builtin_instr: BuiltinInstr<&'static BindingDirective, &'static FreeBuiltinMnemonic, R>,
    args: BuiltinInstrArgs<V, R::StringRef, R::Span>,
}

impl<R, V> BuiltinInstrState<R, V>
where
    R: ReentrancyActions,
{
    fn new(
        builtin_instr: BuiltinInstr<&'static BindingDirective, &'static FreeBuiltinMnemonic, R>,
    ) -> Self {
        Self {
            builtin_instr,
            args: Vec::new(),
        }
    }
}

type BuiltinInstrArgs<V, R, S> = Vec<Arg<V, R, S>>;

pub(in crate::analyze::semantics) type ArgSemantics<R, N, B> = Session<
    R,
    N,
    B,
    ExprBuilder<
        <R as StringSource>::StringRef,
        <R as SpanSource>::Span,
        BuiltinInstrState<
            R,
            <<B as Finish>::Parent as PartialBackend<<R as SpanSource>::Span>>::Value,
        >,
    >,
>;

pub(crate) struct ExprBuilder<R, S, P> {
    arg: Option<Arg<(), R, S>>,
    parent: P,
}

impl<R, S, P> ExprBuilder<R, S, P> {
    pub fn new(parent: P) -> Self {
        Self { arg: None, parent }
    }
}

#[cfg(test)]
mod mock {
    use super::resolve::{BasicNameTable, MockNameTable};
    use super::Keyword;
    use super::*;

    use crate::analyze::macros::mock::MockMacroId;
    use crate::diag::{DiagnosticsEvent, Merge, MockDiagnostics};
    use crate::expr::Expr;
    use crate::log::Log;
    use crate::object::builder::mock::{
        BackendEvent, MockBackend, MockSymbolId, SerialIdAllocator,
    };
    use crate::object::builder::{Backend, RelocContext};

    #[derive(Debug, PartialEq)]
    pub(super) struct MockBindingBuiltinInstr;

    #[derive(Debug, PartialEq)]
    pub(super) struct MockNonBindingBuiltinInstr;

    pub(super) type MockExprBuilder<T, S> = Session<
        MockDiagnostics<T, S>,
        Box<
            MockNameTable<
                BasicNameTable<
                    &'static Keyword<MockBindingBuiltinInstr, MockNonBindingBuiltinInstr>,
                    MockMacroId,
                    MockSymbolId,
                >,
                T,
            >,
        >,
        RelocContext<MockBackend<SerialIdAllocator<MockSymbolId>, T>, Expr<MockSymbolId, S>>,
        (),
    >;

    impl<T, S> MockExprBuilder<T, S>
    where
        T: From<BackendEvent<MockSymbolId, Expr<MockSymbolId, S>>> + From<DiagnosticsEvent<S>>,
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
                    ResolvedName<
                        &'static Keyword<MockBindingBuiltinInstr, MockNonBindingBuiltinInstr>,
                        MockMacroId,
                        MockSymbolId,
                    >,
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
