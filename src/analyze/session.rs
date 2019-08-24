use super::macros::{MacroSource, MacroTable};
use super::resolve::{NameTable, ResolvedName, StartScope};
use super::strings::GetString;
use super::syntax::actions::TokenStreamActions;
use super::syntax::parser::ParserFactory;
use super::syntax::{LexError, ParseTokenStream};
use super::{IdentSource, Lex, Literal, SemanticToken, StringSource, TokenSeq};

use crate::codebase::CodebaseError;
use crate::diag::span::SpanSource;
use crate::diag::*;
use crate::expr::{BinOp, FnCall, LocationCounter, ParamId};
use crate::object::builder::*;

use std::ops::{Deref, DerefMut};

#[cfg(test)]
pub(crate) use self::mock::*;

pub(super) trait ReentrancyActions
where
    Self: IdentSource + SpanSource + StringSource,
    Self: PartialSession<<Self as IdentSource>::Ident, <Self as SpanSource>::Span>,
    Self: GetString<<Self as StringSource>::StringRef>,
{
    fn analyze_file<A: IntoSemanticActions<Self>>(
        self,
        path: Self::StringRef,
        actions: A,
    ) -> (Result<(), CodebaseError>, A::SemanticActions)
    where
        A::SemanticActions: TokenStreamActions<Self::Ident, Literal<Self::StringRef>, Self::Span>;

    fn define_macro(
        &mut self,
        name_span: Self::Span,
        params: Params<Self::Ident, Self::Span>,
        body: TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
    ) -> Self::MacroId;

    fn call_macro<A: IntoSemanticActions<Self>>(
        self,
        name: (Self::MacroId, Self::Span),
        args: MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
        actions: A,
    ) -> A::SemanticActions
    where
        A::SemanticActions: TokenStreamActions<Self::Ident, Literal<Self::StringRef>, Self::Span>;
}

pub(super) trait PartialSession<I, S: Clone>
where
    Self: Sized,
    Self: PartialBackend<S>,
    Self: StartScope<I>,
    Self: NameTable<I>,
    Self: Diagnostics<S>,
{
    type ConstBuilder: ValueBuilder<Self::SymbolId, S, Parent = Self, Value = Self::Value>
        + NameTable<I, SymbolId = Self::SymbolId>
        + Diagnostics<S>;
    type SymbolBuilder: ValueBuilder<Self::SymbolId, S, Parent = Self, Value = ()>
        + NameTable<I, SymbolId = Self::SymbolId>
        + Diagnostics<S>;

    fn build_const(self) -> Self::ConstBuilder;
    fn define_symbol(self, name: Self::SymbolId, span: S) -> Self::SymbolBuilder;
}

pub(super) trait IntoSemanticActions<S: ReentrancyActions> {
    type SemanticActions;

    fn into_semantic_actions(self, session: S) -> Self::SemanticActions;
}

pub(super) type MacroArgs<I, R, S> = super::macros::MacroArgs<SemanticToken<I, R>, S>;
pub(super) type Params<I, S> = (Vec<I>, Vec<S>);

pub(super) struct SessionComponents<Source, Synth> {
    source: Source,
    synth: Synth,
}

pub(super) struct SourceComponents<C, P, M, D> {
    codebase: C,
    parser_factory: P,
    macros: M,
    diagnostics: D,
}

pub(super) struct SynthComponents<I, N, B> {
    interner: I,
    names: N,
    builder: B,
}

impl<C, P, M, D> SpanSource for SourceComponents<C, P, M, D>
where
    D: Deref,
    D::Target: SpanSource,
{
    type Span = <D::Target as SpanSource>::Span;
}

delegate_diagnostics! {
    {C, P, M, D: DerefMut, S: Clone},
    {D::Target: Diagnostics<S>},
    SourceComponents<C, P, M, D>,
    {diagnostics},
    D::Target,
    S
}

impl<'a, C, P, M, I, B, N, D>
    SessionComponents<
        SourceComponents<&'a mut C, &'a mut P, &'a mut M, &'a mut D>,
        SynthComponents<&'a mut I, &'a mut N, B>,
    >
where
    C: IdentSource + StringSource,
    D: DiagnosticsSystem,
{
    pub fn new(
        codebase: &'a mut C,
        parser_factory: &'a mut P,
        macros: &'a mut M,
        interner: &'a mut I,
        builder: B,
        names: &'a mut N,
        diagnostics: &'a mut D,
    ) -> Self {
        SessionComponents {
            source: SourceComponents {
                codebase,
                parser_factory,
                macros,
                diagnostics,
            },
            synth: SynthComponents {
                interner,
                names,
                builder,
            },
        }
    }
}

impl<Source, I, N, B> SessionComponents<Source, SynthComponents<I, N, B>> {
    fn replace_backend<T>(
        self,
        f: impl FnOnce(B) -> T,
    ) -> SessionComponents<Source, SynthComponents<I, N, T>> {
        SessionComponents {
            source: self.source,
            synth: SynthComponents {
                interner: self.synth.interner,
                names: self.synth.names,
                builder: f(self.synth.builder),
            },
        }
    }
}

impl<C, P, M, D, Synth> IdentSource for SessionComponents<SourceComponents<C, P, M, D>, Synth>
where
    C: DerefMut,
    C::Target: IdentSource + StringSource,
{
    type Ident = <C::Target as IdentSource>::Ident;
}

impl<Source, I, N, B> MacroSource for SessionComponents<Source, SynthComponents<I, N, B>>
where
    N: Deref,
    N::Target: MacroSource,
{
    type MacroId = <N::Target as MacroSource>::MacroId;
}

impl<Source, I, N, B> SpanSource for SessionComponents<Source, SynthComponents<I, N, B>>
where
    Source: SpanSource,
{
    type Span = Source::Span;
}

impl<C, P, M, D, Synth> StringSource for SessionComponents<SourceComponents<C, P, M, D>, Synth>
where
    C: DerefMut,
    C::Target: IdentSource + StringSource,
{
    type StringRef = <C::Target as StringSource>::StringRef;
}

impl<Source, I, N, B, S> PartialBackend<S> for SessionComponents<Source, SynthComponents<I, N, B>>
where
    B: Backend<S>,
    S: Clone,
{
    type Value = B::Value;

    fn emit_item(&mut self, item: Item<Self::Value>) {
        self.synth.builder.emit_item(item)
    }

    fn is_non_zero(
        &mut self,
        value: Self::Value,
        diagnostics: &mut impl Diagnostics<S>,
    ) -> Option<bool> {
        self.synth.builder.is_non_zero(value, diagnostics)
    }

    fn reserve(&mut self, bytes: Self::Value) {
        self.synth.builder.reserve(bytes)
    }

    fn set_origin(&mut self, origin: Self::Value) {
        self.synth.builder.set_origin(origin)
    }

    fn start_section(&mut self, name: B::SymbolId, span: S) {
        self.synth.builder.start_section(name, span)
    }
}

impl<C, P, M, I, B, N, D> ReentrancyActions
    for SessionComponents<SourceComponents<C, P, M, D>, SynthComponents<I, N, B>>
where
    C: DerefMut,
    C::Target: Lex<D::Target>,
    P: DerefMut,
    P::Target: ParserFactory<
        <Self as IdentSource>::Ident,
        Literal<<Self as StringSource>::StringRef>,
        LexError,
        <Self as SpanSource>::Span,
    >,
    M: DerefMut,
    M::Target: MacroTable<
        D::Target,
        <Self as IdentSource>::Ident,
        Literal<<Self as StringSource>::StringRef>,
        <Self as SpanSource>::Span,
    >,
    I: Deref,
    I::Target: GetString<<Self as StringSource>::StringRef>,
    B: Backend<<D::Target as SpanSource>::Span>,
    N: DerefMut,
    N::Target: NameTable<
            <C::Target as IdentSource>::Ident,
            MacroId = <M::Target as MacroSource>::MacroId,
            SymbolId = B::SymbolId,
        > + StartScope<<C::Target as IdentSource>::Ident>,
    D: DerefMut,
    D::Target: DiagnosticsSystem,
{
    fn analyze_file<A: IntoSemanticActions<Self>>(
        mut self,
        path: Self::StringRef,
        actions: A,
    ) -> (Result<(), CodebaseError>, A::SemanticActions)
    where
        A::SemanticActions: TokenStreamActions<Self::Ident, Literal<Self::StringRef>, Self::Span>,
    {
        let tokens = match self
            .source
            .codebase
            .lex_file(path, &mut *self.source.diagnostics)
        {
            Ok(tokens) => tokens,
            Err(error) => return (Err(error), actions.into_semantic_actions(self)),
        };
        let mut parser = self.source.parser_factory.mk_parser();
        let actions = actions.into_semantic_actions(self);
        (Ok(()), parser.parse_token_stream(tokens, actions))
    }

    fn define_macro(
        &mut self,
        name_span: Self::Span,
        params: Params<Self::Ident, Self::Span>,
        body: TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
    ) -> Self::MacroId {
        self.source
            .macros
            .define_macro(name_span, params, body, &mut *self.source.diagnostics)
    }

    fn call_macro<A: IntoSemanticActions<Self>>(
        mut self,
        id: (Self::MacroId, Self::Span),
        args: MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
        actions: A,
    ) -> A::SemanticActions
    where
        A::SemanticActions: TokenStreamActions<Self::Ident, Literal<Self::StringRef>, Self::Span>,
    {
        let expansion = self
            .source
            .macros
            .expand_macro(id, args, &mut *self.source.diagnostics);
        let mut parser = self.source.parser_factory.mk_parser();
        let actions = actions.into_semantic_actions(self);
        parser.parse_token_stream(expansion.map(|(t, s)| (Ok(t), s)), actions)
    }
}

impl<Source, Interner, B, N, I, S> PartialSession<I, S>
    for SessionComponents<Source, SynthComponents<Interner, N, B>>
where
    Source: Diagnostics<S>,
    B: Backend<S>,
    N: DerefMut,
    N::Target: NameTable<I, SymbolId = B::SymbolId> + StartScope<I>,
    S: Clone,
    Self: Diagnostics<S>,
{
    type ConstBuilder = SessionComponents<Source, SynthComponents<Interner, N, B::ConstBuilder>>;
    type SymbolBuilder = SessionComponents<Source, SynthComponents<Interner, N, B::SymbolBuilder>>;

    fn build_const(self) -> Self::ConstBuilder {
        self.replace_backend(Backend::build_const)
    }

    fn define_symbol(self, name: B::SymbolId, span: S) -> Self::SymbolBuilder {
        self.replace_backend(|backend| backend.define_symbol(name, span))
    }
}

impl<U, I, N, B, R> GetString<R> for SessionComponents<U, SynthComponents<I, N, B>>
where
    I: Deref,
    I::Target: GetString<R>,
{
    fn get_string<'a>(&'a self, id: &'a R) -> &str {
        self.synth.interner.get_string(id)
    }
}

impl<Source, I, N, B> SymbolSource for SessionComponents<Source, SynthComponents<I, N, B>>
where
    B: SymbolSource,
{
    type SymbolId = B::SymbolId;
}

impl<Source, I, N, B, S> AllocSymbol<S> for SessionComponents<Source, SynthComponents<I, N, B>>
where
    B: AllocSymbol<S>,
    S: Clone,
{
    fn alloc_symbol(&mut self, span: S) -> Self::SymbolId {
        self.synth.builder.alloc_symbol(span)
    }
}

impl<Source, Interner, B, N, I> NameTable<I>
    for SessionComponents<Source, SynthComponents<Interner, N, B>>
where
    B: SymbolSource,
    N: DerefMut,
    N::Target: NameTable<I, SymbolId = B::SymbolId> + MacroSource,
{
    type Keyword = <N::Target as NameTable<I>>::Keyword;

    fn get(
        &mut self,
        ident: &I,
    ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>> {
        self.synth.names.get(ident)
    }

    fn insert(
        &mut self,
        ident: I,
        entry: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>,
    ) {
        self.synth.names.insert(ident, entry)
    }
}

delegate_diagnostics! {
    {'a, Source: Diagnostics<S>, I, N, B, S: Clone},
    SessionComponents<Source, SynthComponents<I, N, B>>,
    {source},
    Source,
    S
}

impl<Source, Interner, B, N, I> StartScope<I>
    for SessionComponents<Source, SynthComponents<Interner, N, B>>
where
    N: DerefMut,
    N::Target: StartScope<I>,
{
    fn start_scope(&mut self, ident: &I) {
        self.synth.names.start_scope(ident)
    }
}

impl<Source, I, N, B, S> PushOp<Name<B::SymbolId>, S>
    for SessionComponents<Source, SynthComponents<I, N, B>>
where
    B: AllocSymbol<S> + PushOp<Name<<B as SymbolSource>::SymbolId>, S>,
    S: Clone,
{
    fn push_op(&mut self, name: Name<B::SymbolId>, span: S) {
        self.synth.builder.push_op(name, span)
    }
}

impl<Source, I, N, B: Finish> Finish for SessionComponents<Source, SynthComponents<I, N, B>> {
    type Parent = SessionComponents<Source, SynthComponents<I, N, B::Parent>>;
    type Value = B::Value;

    fn finish(self) -> (Self::Parent, Self::Value) {
        let (builder, value) = self.synth.builder.finish();
        let parent = SessionComponents {
            source: self.source,
            synth: SynthComponents {
                interner: self.synth.interner,
                names: self.synth.names,
                builder,
            },
        };
        (parent, value)
    }
}

macro_rules! impl_push_op_for_session_components {
    ($t:ty) => {
        impl<Source, I, N, B, S> PushOp<$t, S>
            for SessionComponents<Source, SynthComponents<I, N, B>>
        where
            B: PushOp<$t, S>,
            S: Clone,
        {
            fn push_op(&mut self, op: $t, span: S) {
                self.synth.builder.push_op(op, span)
            }
        }
    };
}

impl_push_op_for_session_components! {LocationCounter}
impl_push_op_for_session_components! {i32}
impl_push_op_for_session_components! {BinOp}
impl_push_op_for_session_components! {ParamId}
impl_push_op_for_session_components! {FnCall}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::analyze::macros::mock::MockMacroId;
    use crate::analyze::resolve::{BasicNameTable, FakeNameTable, MockNameTable, NameTableEvent};
    use crate::analyze::semantics::Keyword;
    use crate::analyze::strings::FakeStringInterner;
    use crate::diag::{DiagnosticsEvent, MockDiagnostics};
    use crate::expr::{Atom, LocationCounter};
    use crate::log::Log;
    use crate::object::builder::mock::*;

    use std::marker::PhantomData;

    type Expr<N, S> = crate::expr::Expr<Atom<LocationCounter, N>, S>;

    #[derive(Debug, PartialEq)]
    pub(crate) enum SessionEvent {
        AnalyzeFile(String),
        DefineMacro(Vec<String>, Vec<SemanticToken<String, String>>),
        InvokeMacro(MockMacroId, Vec<Vec<SemanticToken<String, String>>>),
    }

    pub(in crate::analyze) type MockSession<A, N, T, S> =
        SessionComponents<MockSourceComponents<T, S>, MockSynthComponents<A, N, T>>;

    type MockSynthComponents<A, N, T> =
        SynthComponents<Box<FakeStringInterner>, Box<MockNameTable<N, T>>, MockBackend<A, T>>;

    pub(in crate::analyze) struct MockSourceComponents<T, S> {
        diagnostics: Box<MockDiagnostics<T, S>>,
        id_gen: SerialIdAllocator,
        log: Log<T>,
        error: Option<CodebaseError>,
        _span: PhantomData<S>,
    }

    impl<T, S: Clone> SpanSource for MockSourceComponents<T, S> {
        type Span = S;
    }

    delegate_diagnostics! {
        {T: From<DiagnosticsEvent<S>>, S: Clone + Merge},
        MockSourceComponents<T, S>,
        {diagnostics},
        MockDiagnostics<T, S>,
        S
    }

    impl<A, N, T, S> MockSession<A, N, T, S> {
        fn with_name_table(alloc: A, names: N, log: Log<T>) -> Self {
            Self {
                synth: SynthComponents {
                    interner: Box::new(FakeStringInterner),
                    names: Box::new(MockNameTable::new(names, log.clone())),
                    builder: MockBackend::new(alloc, log.clone()),
                },
                source: MockSourceComponents {
                    diagnostics: Box::new(MockDiagnostics::new(log.clone())),
                    id_gen: SerialIdAllocator::new(),
                    log,
                    error: None,
                    _span: PhantomData,
                },
            }
        }

        pub fn fail(&mut self, error: CodebaseError) {
            self.source.error = Some(error)
        }
    }

    impl<T, S>
        MockSession<SerialIdAllocator, BasicNameTable<&'static Keyword, MockMacroId, usize>, T, S>
    {
        pub fn with_predefined_names<I>(log: Log<T>, entries: I) -> Self
        where
            I: IntoIterator<Item = (String, ResolvedName<&'static Keyword, MockMacroId, usize>)>,
        {
            let mut table = BasicNameTable::default();
            for (name, value) in entries {
                table.insert(name, value)
            }
            Self::with_name_table(SerialIdAllocator::new(), table, log)
        }
    }

    impl<T, S> MockSession<PanickingIdAllocator<String>, FakeNameTable<String>, T, S> {
        pub fn without_name_resolution(log: Log<T>) -> Self {
            Self::with_name_table(PanickingIdAllocator::new(), FakeNameTable::new(), log)
        }
    }

    impl<A, N, T, S> IdentSource for MockSession<A, N, T, S> {
        type Ident = String;
    }

    impl<A, N, T, S> StringSource for MockSession<A, N, T, S> {
        type StringRef = String;
    }

    impl<B, N, T, S> ReentrancyActions for MockSession<B, N, T, S>
    where
        B: AllocSymbol<S>,
        N: NameTable<String, SymbolId = B::SymbolId, MacroId = MockMacroId>,
        T: From<SessionEvent>,
        T: From<BackendEvent<B::SymbolId, Expr<B::SymbolId, S>>>,
        T: From<DiagnosticsEvent<S>>,
        T: From<NameTableEvent<N::Keyword, N::MacroId, N::SymbolId>>,
        S: Clone + Merge,
    {
        fn analyze_file<A: IntoSemanticActions<Self>>(
            mut self,
            path: String,
            actions: A,
        ) -> (Result<(), CodebaseError>, A::SemanticActions) {
            self.source.log.push(SessionEvent::AnalyzeFile(path));
            (
                self.source.error.take().map_or(Ok(()), Err),
                actions.into_semantic_actions(self),
            )
        }

        fn define_macro(
            &mut self,
            _: Self::Span,
            (params, _): (Vec<Self::Ident>, Vec<Self::Span>),
            (body, _): TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
        ) -> Self::MacroId {
            self.source
                .log
                .push(SessionEvent::DefineMacro(params, body));
            MockMacroId(self.source.id_gen.gen())
        }

        fn call_macro<A: IntoSemanticActions<Self>>(
            self,
            (id, _): (Self::MacroId, Self::Span),
            (args, _): MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
            actions: A,
        ) -> A::SemanticActions {
            self.source.log.push(SessionEvent::InvokeMacro(id, args));
            actions.into_semantic_actions(self)
        }
    }

    pub(in crate::analyze) type MockBuilder<A, N, T, S> = SessionComponents<
        MockDiagnostics<T, S>,
        SynthComponents<
            FakeStringInterner,
            Box<MockNameTable<N, T>>,
            RelocContext<MockBackend<A, T>, Expr<<A as SymbolSource>::SymbolId, S>>,
        >,
    >;

    impl<A, N, T, S> MockBuilder<A, N, T, S>
    where
        A: AllocSymbol<S>,
        N: NameTable<String>,
        T: From<BackendEvent<A::SymbolId, Expr<A::SymbolId, S>>>,
        S: Clone,
    {
        fn from_components(alloc: A, names: N, log: Log<T>) -> Self {
            Self {
                source: MockDiagnostics::new(log.clone()),
                synth: SynthComponents {
                    interner: FakeStringInterner,
                    names: Box::new(MockNameTable::new(names, log.clone())),
                    builder: MockBackend::new(alloc, log).build_const(),
                },
            }
        }
    }

    impl<T, S> MockBuilder<SerialIdAllocator, BasicNameTable<Keyword, usize, usize>, T, S>
    where
        T: From<BackendEvent<usize, Expr<usize, S>>>,
        S: Clone,
    {
        pub fn with_name_resolution(log: Log<T>) -> Self {
            Self::with_predefined_names(log, std::iter::empty())
        }

        pub fn with_predefined_names<I>(log: Log<T>, entries: I) -> Self
        where
            I: IntoIterator<Item = (String, ResolvedName<Keyword, usize, usize>)>,
        {
            let mut table = BasicNameTable::default();
            for (name, value) in entries {
                table.insert(name, value)
            }
            Self::from_components(SerialIdAllocator::new(), table, log)
        }
    }

    impl<T, S> MockBuilder<PanickingIdAllocator<String>, FakeNameTable<String>, T, S>
    where
        T: From<BackendEvent<String, Expr<String, S>>>,
        S: Clone,
    {
        pub fn without_name_resolution(log: Log<T>) -> Self {
            Self::from_components(PanickingIdAllocator::new(), FakeNameTable::new(), log)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analyze::macros::mock::{MacroTableEvent, MockMacroId};
    use crate::analyze::resolve::{BasicNameTable, NameTableEvent};
    use crate::analyze::semantics::Keyword;
    use crate::analyze::strings::FakeStringInterner;
    use crate::analyze::syntax::actions::mock::{IdentKind, TokenStreamActionCollector};
    use crate::analyze::syntax::parser::mock::*;
    use crate::analyze::syntax::*;
    use crate::analyze::{Literal, MockCodebase};
    use crate::diag::DiagnosticsEvent;
    use crate::expr::{Atom, BinOp, LocationCounter};
    use crate::log::*;
    use crate::object::builder::mock::{BackendEvent, SerialIdAllocator};
    use crate::object::builder::{CpuInstr, Nullary};

    use std::fmt::Debug;
    use std::iter;

    type Expr<S> = crate::expr::Expr<Atom<LocationCounter, usize>, S>;

    impl<S: ReentrancyActions> IntoSemanticActions<S> for () {
        type SemanticActions =
            TokenStreamActionCollector<S, S::Ident, Literal<S::StringRef>, S::Span>;

        fn into_semantic_actions(self, session: S) -> Self::SemanticActions {
            TokenStreamActionCollector::new(session, panic)
        }
    }

    fn panic<I>(_: &I) -> IdentKind {
        panic!("tried annotating an identifier instead of skipping parsing")
    }

    #[test]
    fn emit_instruction_item() {
        let item = Item::CpuInstr(CpuInstr::Nullary(Nullary::Nop));
        let log =
            Fixture::<()>::default().log_session(|mut session| session.emit_item(item.clone()));
        assert_eq!(log, [BackendEvent::EmitItem(item).into()]);
    }

    #[test]
    fn define_label() {
        let label = "label";
        let log = Fixture::default().log_session(|mut session| {
            let id = session.alloc_symbol(());
            session.insert(label.into(), ResolvedName::Symbol(id));
            let mut builder = session.define_symbol(id, ());
            builder.push_op(LocationCounter, ());
            builder.finish();
        });
        let id = 0;
        assert_eq!(
            log,
            [
                NameTableEvent::Insert(label.into(), ResolvedName::Symbol(id)).into(),
                BackendEvent::DefineSymbol((id, ()), LocationCounter.into()).into()
            ]
        );
    }

    #[test]
    fn start_section() {
        let name: String = "my_section".into();
        let log = Fixture::default().log_session(|mut session| {
            let id = session.alloc_symbol(());
            session.insert(name.clone(), ResolvedName::Symbol(id));
            session.start_section(id, ())
        });
        let id = 0;
        assert_eq!(
            log,
            [
                NameTableEvent::Insert(name, ResolvedName::Symbol(id)).into(),
                BackendEvent::StartSection(id, ()).into()
            ]
        )
    }

    #[test]
    fn include_source_file() {
        let path = "my_file.s";
        let tokens = vec![(Ok(Token::Ident("NOP".into())), ())];
        let log = Fixture::new(|fixture| fixture.codebase.set_file(path, tokens.clone()))
            .log_session(|session| session.analyze_file(path.into(), ()).0.unwrap());
        assert_eq!(log, [ParserEvent::ParseTokenStream(tokens).into()]);
    }

    #[test]
    fn define_and_call_macro() {
        let tokens = vec![Token::Ident("NOP".into())];
        let spans: Vec<_> = iter::repeat(()).take(tokens.len()).collect();
        let log = Fixture::default().log_session(|mut session| {
            let id = session.define_macro((), (vec![], vec![]), (tokens.clone(), spans.clone()));
            session.call_macro((id, ()), (vec![], vec![]), ());
        });
        assert_eq!(
            log,
            [
                MacroTableEvent::DefineMacro(vec![], tokens.clone()).into(),
                ParserEvent::ParseTokenStream(
                    tokens.into_iter().map(|token| (Ok(token), ())).collect()
                )
                .into()
            ]
        );
    }

    #[test]
    fn define_and_call_macro_with_param() {
        let db = Token::Ident("DB".into());
        let arg = Token::Literal(Literal::Number(0x42));
        let literal0 = Token::Literal(Literal::Number(0));
        let param = "x";
        let tokens = vec![db.clone(), Token::Ident(param.into()), literal0.clone()];
        let log = Fixture::default().log_session(|mut session| {
            let id = session.define_macro(
                (),
                (vec![param.into()], vec![()]),
                (tokens.clone(), vec![(), (), ()]),
            );
            session.call_macro((id, ()), (vec![vec![arg.clone()]], vec![vec![()]]), ());
        });
        assert_eq!(
            log,
            [
                MacroTableEvent::DefineMacro(vec![param.into()], tokens).into(),
                ParserEvent::ParseTokenStream(
                    vec![db, arg, literal0]
                        .into_iter()
                        .map(|token| (Ok(token), ()))
                        .collect()
                )
                .into()
            ]
        );
    }

    #[test]
    fn define_and_call_macro_with_label() {
        let nop = Token::Ident("NOP".into());
        let label = "label";
        let param = "x";
        let tokens = vec![Token::Label(param.into()), nop.clone()];
        let log = Fixture::default().log_session(|mut session| {
            let id = ReentrancyActions::define_macro(
                &mut session,
                (),
                (vec![param.into()], vec![()]),
                (tokens.clone(), vec![(), ()]),
            );
            session.call_macro(
                (id, ()),
                (vec![vec![Token::Ident(label.into())]], vec![vec![()]]),
                (),
            );
        });
        assert_eq!(
            log,
            [
                MacroTableEvent::DefineMacro(vec![param.into()], tokens).into(),
                ParserEvent::ParseTokenStream(
                    vec![Token::Label(label.into()), nop]
                        .into_iter()
                        .map(|token| (Ok(token), ()))
                        .collect()
                )
                .into()
            ]
        );
    }

    #[test]
    fn reserve_bytes() {
        let bytes = 10;
        let log = Fixture::default().log_session(|mut session| session.reserve(bytes.into()));
        assert_eq!(log, [BackendEvent::Reserve(bytes.into()).into()])
    }

    impl Default for MockSpan<&'static str> {
        fn default() -> Self {
            unreachable!()
        }
    }

    #[test]
    fn build_value_from_number() {
        Fixture::default().log_session(|session| {
            let mut builder = session.build_const();
            builder.push_op(42, ());
            let (_, value) = builder.finish();
            assert_eq!(value, 42.into())
        });
    }

    #[test]
    fn apply_operator_on_two_values() {
        Fixture::default().log_session(|session| {
            let mut builder = session.build_const();
            builder.push_op(42, ());
            builder.push_op(Name(0), ());
            builder.push_op(BinOp::Multiplication, ());
            let (_, value) = builder.finish();
            assert_eq!(
                value,
                Expr::from_items(&[
                    42.into(),
                    Atom::Name(0).into(),
                    BinOp::Multiplication.into()
                ])
            )
        });
    }

    type MockParserFactory<S> = crate::analyze::syntax::parser::mock::MockParserFactory<Event<S>>;
    type MockMacroTable<S> = crate::analyze::macros::mock::MockMacroTable<usize, Event<S>>;
    type MockBackend<S> = crate::object::builder::mock::MockBackend<SerialIdAllocator, Event<S>>;
    type MockDiagnosticsSystem<S> = crate::diag::MockDiagnosticsSystem<Event<S>, S>;
    type MockNameTable<S> = crate::analyze::resolve::MockNameTable<
        BasicNameTable<Keyword, MockMacroId, usize>,
        Event<S>,
    >;
    type TestSession<'a, S> = SessionComponents<
        SourceComponents<
            &'a mut MockCodebase<S>,
            &'a mut MockParserFactory<S>,
            &'a mut MockMacroTable<S>,
            &'a mut MockDiagnosticsSystem<S>,
        >,
        SynthComponents<&'a mut FakeStringInterner, &'a mut MockNameTable<S>, MockBackend<S>>,
    >;

    #[derive(Debug, PartialEq)]
    enum Event<S: Clone> {
        Parser(ParserEvent<String, Literal<String>, LexError, S>),
        MacroTable(MacroTableEvent),
        Backend(BackendEvent<usize, Expr<S>>),
        NameTable(NameTableEvent<Keyword, MockMacroId, usize>),
        Diagnostics(DiagnosticsEvent<S>),
    }

    impl<S: Clone> From<ParserEvent<String, Literal<String>, LexError, S>> for Event<S> {
        fn from(event: ParserEvent<String, Literal<String>, LexError, S>) -> Self {
            Event::Parser(event)
        }
    }

    impl<S: Clone> From<MacroTableEvent> for Event<S> {
        fn from(event: MacroTableEvent) -> Self {
            Self::MacroTable(event)
        }
    }

    impl<S: Clone> From<BackendEvent<usize, Expr<S>>> for Event<S> {
        fn from(event: BackendEvent<usize, Expr<S>>) -> Self {
            Event::Backend(event)
        }
    }

    impl<S: Clone> From<NameTableEvent<Keyword, MockMacroId, usize>> for Event<S> {
        fn from(event: NameTableEvent<Keyword, MockMacroId, usize>) -> Self {
            Event::NameTable(event)
        }
    }

    impl<S: Clone> From<DiagnosticsEvent<S>> for Event<S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            Event::Diagnostics(event)
        }
    }

    struct Fixture<S: Clone + Default + Merge> {
        inner: InnerFixture<S>,
        log: Log<Event<S>>,
    }

    struct InnerFixture<S: Clone + Default + Merge> {
        codebase: MockCodebase<S>,
        analyzer: MockParserFactory<S>,
        macros: MockMacroTable<S>,
        interner: FakeStringInterner,
        backend: Option<MockBackend<S>>,
        names: MockNameTable<S>,
        diagnostics: MockDiagnosticsSystem<S>,
    }

    impl<S: Clone + Default + Merge> Fixture<S> {
        fn new(f: impl FnOnce(&mut InnerFixture<S>)) -> Self {
            let log = Log::new();
            let mut fixture = Self {
                inner: InnerFixture {
                    codebase: MockCodebase::new(),
                    analyzer: MockParserFactory::new(log.clone()),
                    macros: MockMacroTable::new(log.clone()),
                    interner: FakeStringInterner,
                    backend: Some(MockBackend::new(SerialIdAllocator::new(), log.clone())),
                    names: MockNameTable::new(BasicNameTable::default(), log.clone()),
                    diagnostics: MockDiagnosticsSystem::new(log.clone()),
                },
                log,
            };
            f(&mut fixture.inner);
            fixture
        }

        fn log_session(mut self, f: impl FnOnce(TestSession<S>)) -> Vec<Event<S>>
        where
            Event<S>: Debug,
        {
            f(SessionComponents::new(
                &mut self.inner.codebase,
                &mut self.inner.analyzer,
                &mut self.inner.macros,
                &mut self.inner.interner,
                self.inner.backend.take().unwrap(),
                &mut self.inner.names,
                &mut self.inner.diagnostics,
            ));
            drop(self.inner);
            self.log.into_inner()
        }
    }

    impl<S: Clone + Default + Merge> Default for Fixture<S> {
        fn default() -> Self {
            Self::new(|_| {})
        }
    }
}
