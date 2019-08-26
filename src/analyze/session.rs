use super::macros::{MacroSource, MacroTable};
use super::strings::GetString;
use super::syntax::actions::TokenStreamActions;
use super::syntax::parser::ParserFactory;
use super::syntax::{LexError, ParseTokenStream};
use super::{IdentSource, Lex, Literal, SemanticToken, StringSource, TokenSeq};

use crate::codebase::CodebaseError;
use crate::diag::span::SpanSource;
use crate::diag::*;

use std::ops::{Deref, DerefMut};

#[cfg(test)]
pub(crate) use self::mock::*;

pub(super) trait ReentrancyActions
where
    Self: IdentSource + MacroSource + SpanSource + StringSource,
    Self: GetString<<Self as StringSource>::StringRef>,
    Self: Diagnostics<<Self as SpanSource>::Span>,
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

pub(super) trait IntoSemanticActions<S> {
    type SemanticActions;

    fn into_semantic_actions(self, semantics: S) -> Self::SemanticActions;
}

pub(super) type MacroArgs<I, R, S> = super::macros::MacroArgs<SemanticToken<I, R>, S>;
pub(super) type Params<I, S> = (Vec<I>, Vec<S>);

pub(super) struct SourceComponents<C, P, M, I, D> {
    codebase: C,
    parser_factory: P,
    macros: M,
    interner: I,
    diagnostics: D,
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

impl<'a, C, P, M, I, D> SourceComponents<&'a mut C, &'a mut P, &'a mut M, &'a mut I, &'a mut D>
where
    C: IdentSource + StringSource,
    D: DiagnosticsSystem,
{
    pub fn new(
        codebase: &'a mut C,
        parser_factory: &'a mut P,
        macros: &'a mut M,
        interner: &'a mut I,
        diagnostics: &'a mut D,
    ) -> Self {
        SourceComponents {
            codebase,
            parser_factory,
            macros,
            interner,
            diagnostics,
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

impl<C, P, M, I, D> IdentSource for SourceComponents<C, P, M, I, D>
where
    C: DerefMut,
    C::Target: IdentSource + StringSource,
{
    type Ident = <C::Target as IdentSource>::Ident;
}

impl<C, P, M, I, D> StringSource for SourceComponents<C, P, M, I, D>
where
    C: DerefMut,
    C::Target: IdentSource + StringSource,
{
    type StringRef = <C::Target as StringSource>::StringRef;
}

impl<C, P, M, I, D> ReentrancyActions for SourceComponents<C, P, M, I, D>
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
        let tokens = match self.codebase.lex_file(path, &mut *self.diagnostics) {
            Ok(tokens) => tokens,
            Err(error) => return (Err(error), self.into_semantic_actions(actions)),
        };
        let mut parser = self.parser_factory.mk_parser();
        let actions = self.into_semantic_actions(actions);
        (Ok(()), parser.parse_token_stream(tokens, actions))
    }

    fn define_macro(
        &mut self,
        name_span: Self::Span,
        params: Params<Self::Ident, Self::Span>,
        body: TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
    ) -> Self::MacroId {
        self.macros
            .define_macro(name_span, params, body, &mut *self.diagnostics)
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
        let expansion = self.macros.expand_macro(id, args, &mut *self.diagnostics);
        let mut parser = self.parser_factory.mk_parser();
        let actions = self.into_semantic_actions(actions);
        parser.parse_token_stream(expansion.map(|(t, s)| (Ok(t), s)), actions)
    }
}

impl<C, P, M, I, D, R> GetString<R> for SourceComponents<C, P, M, I, D>
where
    I: Deref,
    I::Target: GetString<R>,
{
    fn get_string<'a>(&'a self, id: &'a R) -> &str {
        self.interner.get_string(id)
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::analyze::macros::mock::MockMacroId;
    use crate::diag::{DiagnosticsEvent, MockDiagnostics};
    use crate::log::Log;
    use crate::object::builder::mock::*;

    use std::marker::PhantomData;

    #[derive(Debug, PartialEq)]
    pub(crate) enum SessionEvent {
        AnalyzeFile(String),
        DefineMacro(Vec<String>, Vec<SemanticToken<String, String>>),
        InvokeMacro(MockMacroId, Vec<Vec<SemanticToken<String, String>>>),
    }

    pub(in crate::analyze) type MockSession<T, S> = MockSourceComponents<T, S>;

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

    impl<T, S> MockSession<T, S> {
        fn with_name_table(log: Log<T>) -> Self {
            MockSourceComponents {
                diagnostics: Box::new(MockDiagnostics::new(log.clone())),
                macro_alloc: SerialIdAllocator::new(MockMacroId),
                log,
                error: None,
                _span: PhantomData,
            }
        }

        pub fn fail(&mut self, error: CodebaseError) {
            self.error = Some(error)
        }
    }

    impl<T, S> MockSession<T, S> {
        pub fn with_log(log: Log<T>) -> Self {
            Self::with_name_table(log)
        }
    }

    impl<T, S> IdentSource for MockSession<T, S> {
        type Ident = String;
    }

    impl<T, S> StringSource for MockSession<T, S> {
        type StringRef = String;
    }

    impl<T, S> GetString<String> for MockSession<T, S> {
        fn get_string<'a>(&self, id: &'a String) -> &'a str {
            id.as_ref()
        }
    }

    impl<T, S> ReentrancyActions for MockSession<T, S>
    where
        T: From<SessionEvent>,
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
            self.log.push(SessionEvent::AnalyzeFile(path));
            (
                self.error.take().map_or(Ok(()), Err),
                self.into_semantic_actions(actions),
            )
        }

        fn define_macro(
            &mut self,
            _: Self::Span,
            (params, _): (Vec<Self::Ident>, Vec<Self::Span>),
            (body, _): TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
        ) -> Self::MacroId {
            self.log.push(SessionEvent::DefineMacro(params, body));
            self.macro_alloc.gen()
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
            self.log.push(SessionEvent::InvokeMacro(id, args));
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
    use crate::log::*;

    use std::fmt::Debug;
    use std::iter;

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

    impl Default for MockSpan<&'static str> {
        fn default() -> Self {
            unreachable!()
        }
    }

    type MockParserFactory<S> = crate::analyze::syntax::parser::mock::MockParserFactory<Event<S>>;
    type MockMacroTable<S> = crate::analyze::macros::mock::MockMacroTable<usize, Event<S>>;
    type MockDiagnosticsSystem<S> = crate::diag::MockDiagnosticsSystem<Event<S>, S>;
    type TestSession<'a, S> = SourceComponents<
        &'a mut MockCodebase<S>,
        &'a mut MockParserFactory<S>,
        &'a mut MockMacroTable<S>,
        &'a mut FakeStringInterner,
        &'a mut MockDiagnosticsSystem<S>,
    >;

    #[derive(Debug, PartialEq)]
    enum Event<S: Clone> {
        Parser(ParserEvent<String, Literal<String>, LexError, S>),
        MacroTable(MacroTableEvent),
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
            f(SourceComponents::new(
                &mut self.inner.codebase,
                &mut self.inner.analyzer,
                &mut self.inner.macros,
                &mut self.inner.interner,
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
