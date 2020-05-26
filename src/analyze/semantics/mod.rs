use self::arg::{Arg, OperandSymbol};
use self::keywords::BuiltinMnemonic;
use self::params::*;
use self::reentrancy::{Meta, Params};
use self::resolve::{NameTable, ResolvedName};

use super::macros::MacroSource;
use super::syntax::actions::{LexerOutput, LineRule};
use super::syntax::LexError;
use super::{IdentSource, Literal, StringSource, TokenSeq};

use crate::diag::span::{SpanSource, Spanned};
use crate::diag::Diagnostics;
use crate::expr::{BinOp, FnCall, LocationCounter, ParamId};
use crate::object::builder::{AllocSymbol, Finish, Name, PartialBackend, PushOp, SymbolSource};

use std::ops::{Deref, DerefMut};

macro_rules! set_state {
    ($session:expr, $state:expr) => {
        $crate::analyze::semantics::Semantics {
            session: $session.session,
            state: $state,
            tokens: $session.tokens,
        }
    };
}

mod actions;
mod arg;
mod keywords;
mod params;
pub(super) mod reentrancy;
pub(super) mod resolve;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analyze) enum Keyword {
    BuiltinMnemonic(BuiltinMnemonic),
    Operand(OperandSymbol),
}

pub(super) struct Semantics<'a, R: Meta, N, B, T> {
    session: CompositeSession<R, N, B>,
    state: T,
    tokens: TokenIterRef<'a, R>,
}

pub(super) struct CompositeSession<R, N, B> {
    reentrancy: R,
    names: N,
    builder: B,
}

type TokenIterRef<'a, R> = &'a mut dyn Iterator<
    Item = LexerOutput<
        <R as IdentSource>::Ident,
        Literal<<R as StringSource>::StringRef>,
        LexError,
        <R as SpanSource>::Span,
    >,
>;

impl<R: Meta, N, B> SpanSource for CompositeSession<R, N, B> {
    type Span = R::Span;
}

impl<R: Meta, N, B> IdentSource for CompositeSession<R, N, B> {
    type Ident = R::Ident;
}

impl<R: Meta, N, B> MacroSource for CompositeSession<R, N, B> {
    type MacroId = R::MacroId;
}

impl<R: Meta, N, B> StringSource for CompositeSession<R, N, B> {
    type StringRef = R::StringRef;
}

impl<'a, R: Meta, N, B, T> Semantics<'a, R, N, B, T> {
    #[cfg(test)]
    fn map_names<F: FnOnce(N) -> M, M>(self, f: F) -> Semantics<'a, R, M, B, T> {
        Semantics {
            session: CompositeSession {
                reentrancy: self.session.reentrancy,
                names: f(self.session.names),
                builder: self.session.builder,
            },
            state: self.state,
            tokens: self.tokens,
        }
    }

    fn map_builder<F: FnOnce(B) -> C, C>(self, f: F) -> Semantics<'a, R, N, C, T> {
        Semantics {
            session: CompositeSession {
                reentrancy: self.session.reentrancy,
                names: self.session.names,
                builder: f(self.session.builder),
            },
            state: self.state,
            tokens: self.tokens,
        }
    }

    fn map_state<F: FnOnce(T) -> U, U>(self, f: F) -> Semantics<'a, R, N, B, U> {
        Semantics {
            session: self.session,
            state: f(self.state),
            tokens: self.tokens,
        }
    }
}

delegate_diagnostics! {
    {'a, R: Meta, N, B, T}, Semantics<'a, R, N, B, T>, {session.reentrancy}, R, R::Span
}

delegate_diagnostics! {
    {R: Meta, N, B}, CompositeSession<R, N, B>, {reentrancy}, R, R::Span
}

impl<'a, R, N, B, S> MacroSource for Semantics<'a, R, N, B, S>
where
    R: Meta,
    N: Deref,
    N::Target: MacroSource,
{
    type MacroId = <N::Target as MacroSource>::MacroId;
}

impl<'a, R, N, B, S> SymbolSource for Semantics<'a, R, N, B, S>
where
    R: Meta,
    N: Deref,
    N::Target: SymbolSource,
{
    type SymbolId = <N::Target as SymbolSource>::SymbolId;
}

impl<'a, R, N, B, S, Span> AllocSymbol<Span> for Semantics<'a, R, N, B, S>
where
    R: Meta,
    N: Deref,
    N::Target: SymbolSource<SymbolId = B::SymbolId>,
    B: AllocSymbol<Span>,
    Span: Clone,
{
    fn alloc_symbol(&mut self, span: Span) -> Self::SymbolId {
        self.session.builder.alloc_symbol(span)
    }
}

impl<'a, R, N, B, T, Ident> NameTable<Ident> for Semantics<'a, R, N, B, T>
where
    R: Meta,
    N: DerefMut,
    N::Target: NameTable<Ident>,
{
    type Keyword = <N::Target as NameTable<Ident>>::Keyword;

    fn resolve_name(
        &mut self,
        ident: &Ident,
    ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>> {
        self.session.names.resolve_name(ident)
    }

    fn define_name(
        &mut self,
        ident: Ident,
        entry: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>,
    ) {
        self.session.names.define_name(ident, entry)
    }
}

impl<'a, R: Meta, N, B: Finish, T> Finish for Semantics<'a, R, N, B, T> {
    type Value = B::Value;
    type Parent = Semantics<'a, R, N, B::Parent, T>;

    fn finish(self) -> (Self::Parent, Option<Self::Value>) {
        let (builder, value) = self.session.builder.finish();
        (
            Semantics {
                session: CompositeSession {
                    reentrancy: self.session.reentrancy,
                    names: self.session.names,
                    builder,
                },
                state: self.state,
                tokens: self.tokens,
            },
            value,
        )
    }
}

impl<'a, R, N, B, T, S, SymbolId> PushOp<Name<SymbolId>, S> for Semantics<'a, R, N, B, T>
where
    R: Meta,
    B: PushOp<Name<SymbolId>, S>,
    S: Clone,
{
    fn push_op(&mut self, op: Name<SymbolId>, span: S) {
        self.session.builder.push_op(op, span)
    }
}

macro_rules! impl_push_op_for_session {
    ($t:ty) => {
        impl<'a, R, N, B, T, S> PushOp<$t, S> for Semantics<'a, R, N, B, T>
        where
            R: Meta,
            B: PushOp<$t, S>,
            S: Clone,
        {
            fn push_op(&mut self, op: $t, span: S) {
                self.session.builder.push_op(op, span)
            }
        }
    };
}

impl_push_op_for_session! {LocationCounter}
impl_push_op_for_session! {i32}
impl_push_op_for_session! {BinOp}
impl_push_op_for_session! {ParamId}
impl_push_op_for_session! {FnCall}

type TokenStreamSemantics<'a, R, N, B> = Semantics<
    'a,
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

impl<'a, R, N, B> TokenStreamSemantics<'a, R, N, B>
where
    R: Meta,
    R::Ident: for<'r> From<&'r str>,
    N: DerefMut,
    N::Target: NameTable<R::Ident, Keyword = &'static Keyword>,
{
    pub fn from_components(
        reentrancy: R,
        mut names: N,
        builder: B,
        tokens: TokenIterRef<'a, R>,
    ) -> Self {
        for (ident, keyword) in keywords::KEYWORDS {
            names.define_name((*ident).into(), ResolvedName::Keyword(keyword))
        }
        Self {
            session: CompositeSession {
                reentrancy,
                names,
                builder,
            },
            state: TokenStreamState::new(),
            tokens,
        }
    }
}

type InstrLineSemantics<'a, R, N, B> =
    Semantics<'a, R, N, B, InstrLineState<<R as IdentSource>::Ident, <R as SpanSource>::Span>>;

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

type TokenLineSemantics<'a, R, N, B> = Semantics<
    'a,
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
    context: TokenContext<I, R, S>,
}

#[derive(Debug, PartialEq)]
pub(in crate::analyze) enum TokenContext<I, R, S> {
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

type BuiltinInstrSemantics<'a, R, N, B> = Semantics<
    'a,
    R,
    N,
    B,
    BuiltinInstrState<R, <B as PartialBackend<<R as SpanSource>::Span>>::Value>,
>;

pub(in crate::analyze) struct BuiltinInstrState<R, V>
where
    R: Meta,
{
    label: Option<Label<R::Ident, R::Span>>,
    mnemonic: Spanned<BuiltinMnemonic, R::Span>,
    args: BuiltinInstrArgs<V, R::StringRef, R::Span>,
}

impl<R, V> BuiltinInstrState<R, V>
where
    R: Meta,
{
    fn new(
        label: Option<Label<R::Ident, R::Span>>,
        mnemonic: Spanned<BuiltinMnemonic, R::Span>,
    ) -> Self {
        Self {
            label,
            mnemonic,
            args: Vec::new(),
        }
    }
}

type BuiltinInstrArgs<V, R, S> = Vec<Arg<V, R, S>>;

pub(in crate::analyze::semantics) type ArgSemantics<'a, R, N, B> = Semantics<
    'a,
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
    use super::reentrancy::{MockSourceComponents, ReentrancyEvent};
    use super::resolve::{BasicNameTable, MockNameTable};
    use super::Keyword;
    use super::*;

    use crate::analyze::macros::mock::MockMacroId;
    use crate::diag::{DiagnosticsEvent, Merge};
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

    pub(super) type MockExprBuilder<'a, T, S> = Semantics<
        'a,
        MockSourceComponents<T, S>,
        Box<MockNameTable<BasicNameTable<&'static Keyword, MockMacroId, MockSymbolId>, T>>,
        RelocContext<MockBackend<SerialIdAllocator<MockSymbolId>, T>, Expr<MockSymbolId, S>>,
        (),
    >;

    impl<'a, T, S> MockExprBuilder<'a, T, S>
    where
        T: From<BackendEvent<MockSymbolId, Expr<MockSymbolId, S>>>
            + From<DiagnosticsEvent<S>>
            + From<ReentrancyEvent>,
        S: Clone + Merge,
    {
        pub fn with_log(log: Log<T>, tokens: TokenIterRef<'a, MockSourceComponents<T, S>>) -> Self {
            Self::with_name_table_entries(log, std::iter::empty(), tokens)
        }

        pub fn with_name_table_entries<I>(
            log: Log<T>,
            entries: I,
            tokens: TokenIterRef<'a, MockSourceComponents<T, S>>,
        ) -> Self
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
            Semantics {
                session: CompositeSession {
                    reentrancy: MockSourceComponents::with_log(log.clone()),
                    names: Box::new(MockNameTable::new(names, log.clone())),
                    builder: MockBackend::new(SerialIdAllocator::new(MockSymbolId), log)
                        .build_const(),
                },
                state: (),
                tokens,
            }
        }
    }
}
