use super::macros::{MacroSource, MacroTable};
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
    Self: IdentSource + MacroSource + SpanSource + StringSource,
    Self: SynthActions<<Self as SpanSource>::Span>,
    Self: GetString<<Self as StringSource>::StringRef>,
{
    fn analyze_file<A>(
        self,
        path: Self::StringRef,
        actions: A,
    ) -> (Result<(), CodebaseError>, Self::SemanticActions)
    where
        Self: IntoSemanticActions<A>,
        Self::SemanticActions:
            TokenStreamActions<Self::Ident, Literal<Self::StringRef>, Self::Span>;

    fn define_macro(
        &mut self,
        name_span: Self::Span,
        params: Params<Self::Ident, Self::Span>,
        body: TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
    ) -> Self::MacroId;

    fn call_macro<A>(
        self,
        name: (Self::MacroId, Self::Span),
        args: MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
        actions: A,
    ) -> Self::SemanticActions
    where
        Self: IntoSemanticActions<A>,
        Self::SemanticActions:
            TokenStreamActions<Self::Ident, Literal<Self::StringRef>, Self::Span>;
}

pub(super) trait SynthActions<S: Clone>
where
    Self: Sized,
    Self: PartialBackend<S>,
    Self: Diagnostics<S>,
{
    type ConstBuilder: ValueBuilder<Self::SymbolId, S, Parent = Self, Value = Self::Value>
        + Diagnostics<S>;
    type SymbolBuilder: ValueBuilder<Self::SymbolId, S, Parent = Self, Value = ()> + Diagnostics<S>;

    fn build_const(self) -> Self::ConstBuilder;
    fn define_symbol(self, name: Self::SymbolId, span: S) -> Self::SymbolBuilder;
}

pub(super) trait IntoSemanticActions<S> {
    type SemanticActions;

    fn into_semantic_actions(self, semantics: S) -> Self::SemanticActions;
}

pub(super) type MacroArgs<I, R, S> = super::macros::MacroArgs<SemanticToken<I, R>, S>;
pub(super) type Params<I, S> = (Vec<I>, Vec<S>);

pub(super) struct SessionComponents<Source, Synth> {
    source: Source,
    synth: Synth,
}

pub(super) struct SourceComponents<C, P, M, I, D> {
    codebase: C,
    parser_factory: P,
    macros: M,
    interner: I,
    diagnostics: D,
}

pub(super) struct SynthComponents<B> {
    builder: B,
}

impl<C, P, M, I, D> SpanSource for SourceComponents<C, P, M, I, D>
where
    D: Deref,
    D::Target: SpanSource,
{
    type Span = <D::Target as SpanSource>::Span;
}

delegate_diagnostics! {
    {C, P, M, I, D: DerefMut, S: Clone},
    {D::Target: Diagnostics<S>},
    SourceComponents<C, P, M, I, D>,
    {diagnostics},
    D::Target,
    S
}

impl<'a, C, P, M, I, B, D>
    SessionComponents<
        SourceComponents<&'a mut C, &'a mut P, &'a mut M, &'a mut I, &'a mut D>,
        SynthComponents<B>,
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
        diagnostics: &'a mut D,
    ) -> Self {
        SessionComponents {
            source: SourceComponents {
                codebase,
                parser_factory,
                macros,
                interner,
                diagnostics,
            },
            synth: SynthComponents { builder },
        }
    }
}

impl<Source, B> SessionComponents<Source, SynthComponents<B>> {
    fn replace_backend<T>(
        self,
        f: impl FnOnce(B) -> T,
    ) -> SessionComponents<Source, SynthComponents<T>> {
        SessionComponents {
            source: self.source,
            synth: SynthComponents {
                builder: f(self.synth.builder),
            },
        }
    }
}

impl<C, P, M, I, D> MacroSource for SourceComponents<C, P, M, I, D>
where
    M: Deref,
    M::Target: MacroSource,
{
    type MacroId = <M::Target as MacroSource>::MacroId;
}

impl<C, P, M, I, D, Synth> IdentSource for SessionComponents<SourceComponents<C, P, M, I, D>, Synth>
where
    C: DerefMut,
    C::Target: IdentSource + StringSource,
{
    type Ident = <C::Target as IdentSource>::Ident;
}

impl<Source: MacroSource, B> MacroSource for SessionComponents<Source, SynthComponents<B>> {
    type MacroId = Source::MacroId;
}

impl<Source, B> SpanSource for SessionComponents<Source, SynthComponents<B>>
where
    Source: SpanSource,
{
    type Span = Source::Span;
}

impl<C, P, M, I, D, Synth> StringSource
    for SessionComponents<SourceComponents<C, P, M, I, D>, Synth>
where
    C: DerefMut,
    C::Target: IdentSource + StringSource,
{
    type StringRef = <C::Target as StringSource>::StringRef;
}

impl<Source, B, S> PartialBackend<S> for SessionComponents<Source, SynthComponents<B>>
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

impl<C, P, M, I, B, D> ReentrancyActions
    for SessionComponents<SourceComponents<C, P, M, I, D>, SynthComponents<B>>
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
    D: DerefMut,
    D::Target: DiagnosticsSystem,
{
    fn analyze_file<A>(
        mut self,
        path: Self::StringRef,
        actions: A,
    ) -> (
        Result<(), CodebaseError>,
        <Self as IntoSemanticActions<A>>::SemanticActions,
    )
    where
        Self: IntoSemanticActions<A>,
        <Self as IntoSemanticActions<A>>::SemanticActions:
            TokenStreamActions<Self::Ident, Literal<Self::StringRef>, Self::Span>,
    {
        let tokens = match self
            .source
            .codebase
            .lex_file(path, &mut *self.source.diagnostics)
        {
            Ok(tokens) => tokens,
            Err(error) => return (Err(error), self.into_semantic_actions(actions)),
        };
        let mut parser = self.source.parser_factory.mk_parser();
        let actions = self.into_semantic_actions(actions);
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

    fn call_macro<A>(
        mut self,
        id: (Self::MacroId, Self::Span),
        args: MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
        actions: A,
    ) -> <Self as IntoSemanticActions<A>>::SemanticActions
    where
        Self: IntoSemanticActions<A>,
        <Self as IntoSemanticActions<A>>::SemanticActions:
            TokenStreamActions<Self::Ident, Literal<Self::StringRef>, Self::Span>,
    {
        let expansion = self
            .source
            .macros
            .expand_macro(id, args, &mut *self.source.diagnostics);
        let mut parser = self.source.parser_factory.mk_parser();
        let actions = self.into_semantic_actions(actions);
        parser.parse_token_stream(expansion.map(|(t, s)| (Ok(t), s)), actions)
    }
}

impl<Source, B, S> SynthActions<S> for SessionComponents<Source, SynthComponents<B>>
where
    Source: Diagnostics<S>,
    B: Backend<S>,
    S: Clone,
    Self: Diagnostics<S>,
{
    type ConstBuilder = SessionComponents<Source, SynthComponents<B::ConstBuilder>>;
    type SymbolBuilder = SessionComponents<Source, SynthComponents<B::SymbolBuilder>>;

    fn build_const(self) -> Self::ConstBuilder {
        self.replace_backend(Backend::build_const)
    }

    fn define_symbol(self, name: B::SymbolId, span: S) -> Self::SymbolBuilder {
        self.replace_backend(|backend| backend.define_symbol(name, span))
    }
}

impl<C, P, M, I, D, Synth, R> GetString<R>
    for SessionComponents<SourceComponents<C, P, M, I, D>, Synth>
where
    I: Deref,
    I::Target: GetString<R>,
{
    fn get_string<'a>(&'a self, id: &'a R) -> &str {
        self.source.interner.get_string(id)
    }
}

impl<Source, B> SymbolSource for SessionComponents<Source, SynthComponents<B>>
where
    B: SymbolSource,
{
    type SymbolId = B::SymbolId;
}

impl<Source, B, S> AllocSymbol<S> for SessionComponents<Source, SynthComponents<B>>
where
    B: AllocSymbol<S>,
    S: Clone,
{
    fn alloc_symbol(&mut self, span: S) -> Self::SymbolId {
        self.synth.builder.alloc_symbol(span)
    }
}

delegate_diagnostics! {
    {'a, Source: Diagnostics<S>, Synth, S: Clone},
    SessionComponents<Source, Synth>,
    {source},
    Source,
    S
}

impl<Source, B, S> PushOp<Name<B::SymbolId>, S> for SessionComponents<Source, SynthComponents<B>>
where
    B: AllocSymbol<S> + PushOp<Name<<B as SymbolSource>::SymbolId>, S>,
    S: Clone,
{
    fn push_op(&mut self, name: Name<B::SymbolId>, span: S) {
        self.synth.builder.push_op(name, span)
    }
}

impl<Source, B: Finish> Finish for SessionComponents<Source, SynthComponents<B>> {
    type Parent = SessionComponents<Source, SynthComponents<B::Parent>>;
    type Value = B::Value;

    fn finish(self) -> (Self::Parent, Self::Value) {
        let (builder, value) = self.synth.builder.finish();
        let parent = SessionComponents {
            source: self.source,
            synth: SynthComponents { builder },
        };
        (parent, value)
    }
}

macro_rules! impl_push_op_for_session_components {
    ($t:ty) => {
        impl<Source, B, S> PushOp<$t, S> for SessionComponents<Source, SynthComponents<B>>
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
    use crate::diag::{DiagnosticsEvent, MockDiagnostics};
    use crate::expr::{Atom, Expr, LocationCounter};
    use crate::log::Log;
    use crate::object::builder::mock::*;

    use std::marker::PhantomData;

    #[derive(Debug, PartialEq)]
    pub(crate) enum SessionEvent {
        AnalyzeFile(String),
        DefineMacro(Vec<String>, Vec<SemanticToken<String, String>>),
        InvokeMacro(MockMacroId, Vec<Vec<SemanticToken<String, String>>>),
    }

    pub(in crate::analyze) type MockSession<A, T, S> =
        SessionComponents<MockSourceComponents<T, S>, MockSynthComponents<A, T>>;

    type MockSynthComponents<A, T> = SynthComponents<MockBackend<A, T>>;

    pub(in crate::analyze) struct MockSourceComponents<T, S> {
        diagnostics: Box<MockDiagnostics<T, S>>,
        macro_alloc: SerialIdAllocator<MockMacroId>,
        log: Log<T>,
        error: Option<CodebaseError>,
        _span: PhantomData<S>,
    }

    impl<T, S: Clone> MacroSource for MockSourceComponents<T, S> {
        type MacroId = MockMacroId;
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

    impl<A, T, S> MockSession<A, T, S> {
        fn with_name_table(alloc: A, log: Log<T>) -> Self {
            Self {
                synth: SynthComponents {
                    builder: MockBackend::new(alloc, log.clone()),
                },
                source: MockSourceComponents {
                    diagnostics: Box::new(MockDiagnostics::new(log.clone())),
                    macro_alloc: SerialIdAllocator::new(MockMacroId),
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

    impl<T, S> MockSession<SerialIdAllocator<MockSymbolId>, T, S> {
        pub fn with_log(log: Log<T>) -> Self {
            Self::with_name_table(SerialIdAllocator::new(MockSymbolId), log)
        }
    }

    impl<A, T, S> IdentSource for MockSession<A, T, S> {
        type Ident = String;
    }

    impl<A, T, S> StringSource for MockSession<A, T, S> {
        type StringRef = String;
    }

    impl<B, T, S> GetString<String> for MockSession<B, T, S> {
        fn get_string<'a>(&self, id: &'a String) -> &'a str {
            id.as_ref()
        }
    }

    impl<B, T, S> ReentrancyActions for MockSession<B, T, S>
    where
        B: AllocSymbol<S>,
        T: From<SessionEvent>,
        T: From<BackendEvent<B::SymbolId, Expr<Atom<LocationCounter, B::SymbolId>, S>>>,
        T: From<DiagnosticsEvent<S>>,
        S: Clone + Merge,
    {
        fn analyze_file<A>(
            mut self,
            path: String,
            actions: A,
        ) -> (
            Result<(), CodebaseError>,
            <Self as IntoSemanticActions<A>>::SemanticActions,
        )
        where
            Self: IntoSemanticActions<A>,
        {
            self.source.log.push(SessionEvent::AnalyzeFile(path));
            (
                self.source.error.take().map_or(Ok(()), Err),
                self.into_semantic_actions(actions),
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
            self.source.macro_alloc.gen()
        }

        fn call_macro<A>(
            self,
            (id, _): (Self::MacroId, Self::Span),
            (args, _): MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
            actions: A,
        ) -> <Self as IntoSemanticActions<A>>::SemanticActions
        where
            Self: IntoSemanticActions<A>,
        {
            self.source.log.push(SessionEvent::InvokeMacro(id, args));
            self.into_semantic_actions(actions)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analyze::macros::mock::MacroTableEvent;
    use crate::analyze::strings::FakeStringInterner;
    use crate::analyze::syntax::actions::mock::{IdentKind, TokenStreamActionCollector};
    use crate::analyze::syntax::parser::mock::*;
    use crate::analyze::syntax::*;
    use crate::analyze::{Literal, MockCodebase};
    use crate::diag::DiagnosticsEvent;
    use crate::expr::{Atom, BinOp, LocationCounter};
    use crate::log::*;
    use crate::object::builder::mock::{BackendEvent, MockSymbolId, SerialIdAllocator};
    use crate::object::builder::{CpuInstr, Nullary};

    use std::fmt::Debug;
    use std::iter;

    type Expr<S> = crate::expr::Expr<Atom<LocationCounter, MockSymbolId>, S>;

    impl<S: ReentrancyActions> IntoSemanticActions<()> for S {
        type SemanticActions =
            TokenStreamActionCollector<S, S::Ident, Literal<S::StringRef>, S::Span>;

        fn into_semantic_actions(self, _: ()) -> Self::SemanticActions {
            TokenStreamActionCollector::new(self, panic)
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
        let log = Fixture::default().log_session(|mut session| {
            let id = session.alloc_symbol(());
            let mut builder = session.define_symbol(id, ());
            builder.push_op(LocationCounter, ());
            builder.finish();
        });
        assert_eq!(
            log,
            [BackendEvent::DefineSymbol((MockSymbolId(0), ()), LocationCounter.into()).into()]
        );
    }

    #[test]
    fn start_section() {
        let log = Fixture::default().log_session(|mut session| {
            let id = session.alloc_symbol(());
            session.start_section(id, ())
        });
        assert_eq!(
            log,
            [BackendEvent::StartSection(MockSymbolId(0), ()).into()]
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
            builder.push_op(Name(MockSymbolId(0)), ());
            builder.push_op(BinOp::Multiplication, ());
            let (_, value) = builder.finish();
            assert_eq!(
                value,
                Expr::from_items(&[
                    42.into(),
                    Atom::Name(MockSymbolId(0)).into(),
                    BinOp::Multiplication.into()
                ])
            )
        });
    }

    type MockParserFactory<S> = crate::analyze::syntax::parser::mock::MockParserFactory<Event<S>>;
    type MockMacroTable<S> = crate::analyze::macros::mock::MockMacroTable<usize, Event<S>>;
    type MockBackend<S> =
        crate::object::builder::mock::MockBackend<SerialIdAllocator<MockSymbolId>, Event<S>>;
    type MockDiagnosticsSystem<S> = crate::diag::MockDiagnosticsSystem<Event<S>, S>;
    type TestSession<'a, S> = SessionComponents<
        SourceComponents<
            &'a mut MockCodebase<S>,
            &'a mut MockParserFactory<S>,
            &'a mut MockMacroTable<S>,
            &'a mut FakeStringInterner,
            &'a mut MockDiagnosticsSystem<S>,
        >,
        SynthComponents<MockBackend<S>>,
    >;

    #[derive(Debug, PartialEq)]
    enum Event<S: Clone> {
        Parser(ParserEvent<String, Literal<String>, LexError, S>),
        MacroTable(MacroTableEvent),
        Backend(BackendEvent<MockSymbolId, Expr<S>>),
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

    impl<S: Clone> From<BackendEvent<MockSymbolId, Expr<S>>> for Event<S> {
        fn from(event: BackendEvent<MockSymbolId, Expr<S>>) -> Self {
            Event::Backend(event)
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
                    backend: Some(MockBackend::new(
                        SerialIdAllocator::new(MockSymbolId),
                        log.clone(),
                    )),
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
