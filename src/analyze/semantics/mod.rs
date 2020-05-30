use self::arg::{Arg, OperandSymbol};
use self::keywords::BuiltinMnemonic;
use self::params::*;

use super::macros::MacroSource;
use super::syntax::actions::{LexerOutput, LineRule};
use super::syntax::LexError;
use super::{IdentSource, Literal, StringSource, TokenSeq};

use crate::diag::span::{SpanSource, Spanned};
use crate::diag::Diagnostics;
use crate::expr::{BinOp, FnCall, LocationCounter, ParamId};
use crate::object::builder::*;
use crate::session::reentrancy::{Meta, Params, ReentrancyActions};
use crate::session::resolve::{NameTable, ResolvedName};
use crate::{CompositeSession, Session};

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

#[derive(Clone, Debug, PartialEq)]
pub enum Keyword {
    BuiltinMnemonic(BuiltinMnemonic),
    Operand(OperandSymbol),
}

pub(crate) struct Semantics<'a, S, T, I, R, Z> {
    pub session: S,
    pub state: T,
    pub tokens: TokenIterRef<'a, I, R, Z>,
}

type TokenIterRef<'a, I, R, S> =
    &'a mut dyn Iterator<Item = LexerOutput<I, Literal<R>, LexError, S>>;

impl<R: SpanSource, N, B> SpanSource for CompositeSession<R, N, B> {
    type Span = R::Span;
}

impl<R: IdentSource, N, B> IdentSource for CompositeSession<R, N, B> {
    type Ident = R::Ident;
}

impl<R: MacroSource, N, B> MacroSource for CompositeSession<R, N, B> {
    type MacroId = R::MacroId;
}

impl<R: StringSource, N, B> StringSource for CompositeSession<R, N, B> {
    type StringRef = R::StringRef;
}

impl<R, N, B: SymbolSource> SymbolSource for CompositeSession<R, N, B> {
    type SymbolId = B::SymbolId;
}

impl<'a, S: Session, T> Semantics<'a, S, T, S::Ident, S::StringRef, S::Span> {
    fn map_state<F: FnOnce(T) -> U, U>(
        self,
        f: F,
    ) -> Semantics<'a, S, U, S::Ident, S::StringRef, S::Span> {
        Semantics {
            session: self.session,
            state: f(self.state),
            tokens: self.tokens,
        }
    }
}

delegate_diagnostics! {
    {'a, S: Meta, T}, Semantics<'a, S, T, S::Ident, S::StringRef, S::Span>, {session}, S, S::Span
}

delegate_diagnostics! {
    {R: Diagnostics<S>, N, B, S}, CompositeSession<R, N, B>, {reentrancy}, R, S
}

impl<'a, R, N, B, S> MacroSource
    for Semantics<'a, CompositeSession<R, N, B>, S, R::Ident, R::StringRef, R::Span>
where
    R: Meta,
    CompositeSession<R, N, B>: MacroSource,
{
    type MacroId = <CompositeSession<R, N, B> as MacroSource>::MacroId;
}

impl<'a, R, N, B, S> SymbolSource
    for Semantics<'a, CompositeSession<R, N, B>, S, R::Ident, R::StringRef, R::Span>
where
    R: Meta,
    CompositeSession<R, N, B>: SymbolSource,
{
    type SymbolId = <CompositeSession<R, N, B> as SymbolSource>::SymbolId;
}

impl<'a, R, N, B, Span> AllocSymbol<Span> for CompositeSession<R, N, B>
where
    Self: SymbolSource<SymbolId = B::SymbolId>,
    B: AllocSymbol<Span>,
    Span: Clone,
{
    fn alloc_symbol(&mut self, span: Span) -> Self::SymbolId {
        self.builder.alloc_symbol(span)
    }
}

impl<'a, R, N, B, S, Span> AllocSymbol<Span>
    for Semantics<'a, CompositeSession<R, N, B>, S, R::Ident, R::StringRef, R::Span>
where
    R: Meta,
    CompositeSession<R, N, B>: SymbolSource<SymbolId = B::SymbolId>,
    B: AllocSymbol<Span>,
    Span: Clone,
{
    fn alloc_symbol(&mut self, span: Span) -> Self::SymbolId {
        self.session.alloc_symbol(span)
    }
}

impl<'a, R, N, B, T> NameTable<R::Ident>
    for Semantics<'a, CompositeSession<R, N, B>, T, R::Ident, R::StringRef, R::Span>
where
    R: Meta,
    CompositeSession<R, N, B>: NameTable<R::Ident>,
{
    type Keyword = <CompositeSession<R, N, B> as NameTable<R::Ident>>::Keyword;

    fn resolve_name(
        &mut self,
        ident: &R::Ident,
    ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>> {
        self.session.resolve_name(ident)
    }

    fn define_name(
        &mut self,
        ident: R::Ident,
        entry: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>,
    ) {
        self.session.define_name(ident, entry)
    }
}

impl<'a, R: Meta, N, B: Finish, T> Finish
    for Semantics<'a, CompositeSession<R, N, B>, T, R::Ident, R::StringRef, R::Span>
{
    type Value = B::Value;
    type Parent =
        Semantics<'a, CompositeSession<R, N, B::Parent>, T, R::Ident, R::StringRef, R::Span>;

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

impl<'a, R, N, B, T, S, SymbolId> PushOp<Name<SymbolId>, S>
    for Semantics<'a, CompositeSession<R, N, B>, T, R::Ident, R::StringRef, R::Span>
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
        impl<'a, R, N, B, T, S> PushOp<$t, S>
            for Semantics<'a, CompositeSession<R, N, B>, T, R::Ident, R::StringRef, R::Span>
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

type TokenStreamSemantics<'a, S> = Semantics<
    'a,
    S,
    TokenStreamState<
        <S as IdentSource>::Ident,
        <S as StringSource>::StringRef,
        <S as SpanSource>::Span,
    >,
    <S as IdentSource>::Ident,
    <S as StringSource>::StringRef,
    <S as SpanSource>::Span,
>;

#[derive(Debug, PartialEq)]
pub struct TokenStreamState<I, R, S> {
    mode: LineRule<InstrLineState<I, S>, TokenLineState<I, R, S>>,
}

impl<I, R, S> TokenStreamState<I, R, S> {
    pub fn new() -> Self {
        Self {
            mode: LineRule::InstrLine(InstrLineState::new()),
        }
    }
}

impl<R, N, B> CompositeSession<R, N, B>
where
    Self: ReentrancyActions,
    <Self as IdentSource>::Ident: for<'r> From<&'r str>,
    Self: NameTable<<Self as IdentSource>::Ident, Keyword = &'static Keyword>,
    Self: Backend<<Self as SpanSource>::Span>,
{
    pub fn from_components(reentrancy: R, names: N, builder: B) -> Self {
        let mut session = Self {
            reentrancy,
            names,
            builder,
        };
        for (ident, keyword) in keywords::KEYWORDS {
            session.define_name((*ident).into(), ResolvedName::Keyword(keyword))
        }
        session
    }
}

type InstrLineSemantics<'a, S> = Semantics<
    'a,
    S,
    InstrLineState<<S as IdentSource>::Ident, <S as SpanSource>::Span>,
    <S as IdentSource>::Ident,
    <S as StringSource>::StringRef,
    <S as SpanSource>::Span,
>;

#[derive(Debug, PartialEq)]
pub struct InstrLineState<I, S> {
    label: Option<Label<I, S>>,
}

impl<I, S> InstrLineState<I, S> {
    fn new() -> Self {
        Self { label: None }
    }
}

type Label<I, S> = ((I, S), Params<I, S>);

type TokenLineSemantics<'a, S> = Semantics<
    'a,
    S,
    TokenLineState<
        <S as IdentSource>::Ident,
        <S as StringSource>::StringRef,
        <S as SpanSource>::Span,
    >,
    <S as IdentSource>::Ident,
    <S as StringSource>::StringRef,
    <S as SpanSource>::Span,
>;

#[derive(Debug, PartialEq)]
pub struct TokenLineState<I, R, S> {
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

type BuiltinInstrSemantics<'a, S> = Semantics<
    'a,
    S,
    BuiltinInstrState<S>,
    <S as IdentSource>::Ident,
    <S as StringSource>::StringRef,
    <S as SpanSource>::Span,
>;

pub(crate) struct BuiltinInstrState<S: Session> {
    label: Option<Label<S::Ident, S::Span>>,
    mnemonic: Spanned<BuiltinMnemonic, S::Span>,
    args: BuiltinInstrArgs<S::Value, S::StringRef, S::Span>,
}

impl<S: Session> BuiltinInstrState<S> {
    fn new(
        label: Option<Label<S::Ident, S::Span>>,
        mnemonic: Spanned<BuiltinMnemonic, S::Span>,
    ) -> Self {
        Self {
            label,
            mnemonic,
            args: Vec::new(),
        }
    }
}

type BuiltinInstrArgs<V, R, S> = Vec<Arg<V, R, S>>;

pub(in crate::analyze::semantics) type ArgSemantics<'a, S> = Semantics<
    'a,
    S,
    ExprBuilder<
        <<S as Finish>::Parent as StringSource>::StringRef,
        <<S as Finish>::Parent as SpanSource>::Span,
        BuiltinInstrState<<S as Finish>::Parent>,
    >,
    <<S as Finish>::Parent as IdentSource>::Ident,
    <<S as Finish>::Parent as StringSource>::StringRef,
    <<S as Finish>::Parent as SpanSource>::Span,
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
    use super::Keyword;
    use super::*;

    use crate::analyze::macros::mock::MockMacroId;
    use crate::diag::{DiagnosticsEvent, Merge};
    use crate::expr::Expr;
    use crate::log::Log;
    use crate::object::builder::mock::*;
    use crate::object::builder::{Backend, RelocContext};
    use crate::session::reentrancy::{MockSourceComponents, ReentrancyEvent};
    use crate::session::resolve::{BasicNameTable, MockNameTable};

    #[derive(Debug, PartialEq)]
    pub(super) struct MockBindingBuiltinInstr;

    #[derive(Debug, PartialEq)]
    pub(super) struct MockNonBindingBuiltinInstr;

    pub(super) type MockExprBuilder<'a, T, S> = Semantics<
        'a,
        CompositeSession<
            MockSourceComponents<T, S>,
            Box<MockNameTable<BasicNameTable<&'static Keyword, MockMacroId, MockSymbolId>, T>>,
            RelocContext<MockBackend<SerialIdAllocator<MockSymbolId>, T>, Expr<MockSymbolId, S>>,
        >,
        (),
        String,
        String,
        S,
    >;

    impl<'a, T, S> MockExprBuilder<'a, T, S>
    where
        T: From<BackendEvent<MockSymbolId, Expr<MockSymbolId, S>>>
            + From<DiagnosticsEvent<S>>
            + From<ReentrancyEvent>,
        S: Clone + Merge,
    {
        pub fn with_log(log: Log<T>, tokens: TokenIterRef<'a, String, String, S>) -> Self {
            Self::with_name_table_entries(log, std::iter::empty(), tokens)
        }

        pub fn with_name_table_entries<I>(
            log: Log<T>,
            entries: I,
            tokens: TokenIterRef<'a, String, String, S>,
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
                    builder: MockBackend::new(SerialIdAllocator::new(MockSymbolId), log),
                }
                .build_const(),
                state: (),
                tokens,
            }
        }
    }
}
