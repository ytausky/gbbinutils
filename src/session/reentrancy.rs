use super::resolve::{NameTable, StartScope};
use super::CompositeSession;

use crate::analyze::macros::{MacroSource, MacroTable};
use crate::analyze::semantics::{Keyword, Semantics, TokenStreamState};
use crate::analyze::strings::GetString;
use crate::analyze::{Lex, Literal, SemanticToken, StringSource, TokenSeq};
use crate::codebase::CodebaseError;
use crate::diag::span::SpanSource;
use crate::diag::*;
use crate::object::builder::{Backend, SymbolSource};
use crate::syntax::parser::ParserFactory;
use crate::syntax::{IdentSource, LexError, ParseTokenStream};

use std::ops::{Deref, DerefMut};

#[cfg(test)]
pub(crate) use self::mock::*;

pub(crate) trait Meta:
    IdentSource + MacroSource + SpanSource + StringSource + Diagnostics<<Self as SpanSource>::Span>
{
}

impl<T> Meta for T where
    T: IdentSource
        + MacroSource
        + SpanSource
        + StringSource
        + Diagnostics<<Self as SpanSource>::Span>
{
}

pub(crate) trait ReentrancyActions
where
    Self: Meta + Sized,
{
    fn analyze_file(self, path: Self::StringRef) -> (Result<(), CodebaseError>, Self);

    fn define_macro(
        &mut self,
        name_span: Self::Span,
        params: Params<Self::Ident, Self::Span>,
        body: TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
    ) -> Self::MacroId;

    fn call_macro(
        self,
        name: (Self::MacroId, Self::Span),
        args: MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
    ) -> Self;
}

pub type MacroArgs<I, R, S> = crate::analyze::macros::MacroArgs<SemanticToken<I, R>, S>;
pub type Params<I, S> = (Vec<I>, Vec<S>);

pub(crate) struct SourceComponents<C, P, M, I, D> {
    pub codebase: C,
    pub parser_factory: P,
    pub macros: M,
    pub interner: I,
    pub diagnostics: D,
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

impl<'a, C, P, M, I, D> SourceComponents<&'a mut C, &'a mut P, &'a mut M, &'a mut I, &'a mut D> {
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

impl<C, P, M, I, D, N, B> ReentrancyActions
    for CompositeSession<SourceComponents<C, P, M, I, D>, N, B>
where
    Self: Lex<Span = <D::Target as SpanSource>::Span>,
    SourceComponents<C, P, M, I, D>: IdentSource<Ident = <Self as IdentSource>::Ident>
        + StringSource<StringRef = <Self as StringSource>::StringRef>
        + MacroSource<MacroId = <Self as MacroSource>::MacroId>,
    P: DerefMut,
    P::Target: ParserFactory<
        <Self as IdentSource>::Ident,
        Literal<<Self as StringSource>::StringRef>,
        LexError,
        <Self as SpanSource>::Span,
    >,
    Self: MacroTable<
        <Self as IdentSource>::Ident,
        Literal<<Self as StringSource>::StringRef>,
        <Self as SpanSource>::Span,
    >,
    I: Deref,
    I::Target: GetString<<Self as StringSource>::StringRef>,
    D: DerefMut,
    D::Target: DiagnosticsSystem,
    Self: StartScope<<Self as IdentSource>::Ident>
        + NameTable<<Self as IdentSource>::Ident, Keyword = &'static Keyword>,
    <Self as Backend<<D::Target as SpanSource>::Span>>::ExprBuilder: StartScope<<Self as IdentSource>::Ident>
        + NameTable<
            <Self as IdentSource>::Ident,
            Keyword = &'static Keyword,
            MacroId = <Self as MacroSource>::MacroId,
            SymbolId = <Self as SymbolSource>::SymbolId,
        > + Diagnostics<<D::Target as SpanSource>::Span, Stripped = Self::Stripped>,
    Self: Backend<<D::Target as SpanSource>::Span>,
    <Self as IdentSource>::Ident: 'static,
    <Self as StringSource>::StringRef: 'static,
    <Self as SpanSource>::Span: 'static,
{
    fn analyze_file(mut self, path: Self::StringRef) -> (Result<(), CodebaseError>, Self) {
        let mut tokens = match self.lex_file(path) {
            Ok(tokens) => tokens,
            Err(error) => return (Err(error), self),
        };
        let mut parser = self.reentrancy.parser_factory.mk_parser();
        let semantics = Semantics {
            session: self,
            state: TokenStreamState::new(),
            tokens: &mut tokens,
        };
        (Ok(()), parser.parse_token_stream(semantics).session)
    }

    fn define_macro(
        &mut self,
        name_span: Self::Span,
        params: Params<Self::Ident, Self::Span>,
        body: TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
    ) -> Self::MacroId {
        MacroTable::define_macro(self, name_span, params, body)
    }

    fn call_macro(
        mut self,
        id: (Self::MacroId, Self::Span),
        args: MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
    ) -> Self {
        let expansion = self.expand_macro(id, args);
        let mut parser = self.reentrancy.parser_factory.mk_parser();
        let mut tokens = expansion.map(|(t, s)| (Ok(t), s));
        let semantics = Semantics {
            session: self,
            state: TokenStreamState::new(),
            tokens: &mut tokens,
        };
        parser.parse_token_stream(semantics).session
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
    pub(crate) enum ReentrancyEvent {
        AnalyzeFile(String),
        DefineMacro(Vec<String>, Vec<SemanticToken<String, String>>),
        InvokeMacro(MockMacroId, Vec<Vec<SemanticToken<String, String>>>),
    }

    pub struct MockSourceComponents<T, S> {
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

    impl<T, S> MockSourceComponents<T, S> {
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

    impl<T, S> MockSourceComponents<T, S> {
        pub fn with_log(log: Log<T>) -> Self {
            Self::with_name_table(log)
        }
    }

    impl<T, S> IdentSource for MockSourceComponents<T, S> {
        type Ident = String;
    }

    impl<T, S> StringSource for MockSourceComponents<T, S> {
        type StringRef = String;
    }

    impl<T, S> GetString<String> for MockSourceComponents<T, S> {
        fn get_string<'a>(&self, id: &'a String) -> &'a str {
            id.as_ref()
        }
    }

    impl<T, S, N, B> ReentrancyActions for CompositeSession<MockSourceComponents<T, S>, N, B>
    where
        T: From<ReentrancyEvent> + From<DiagnosticsEvent<S>>,
        S: Clone + Merge,
    {
        fn analyze_file(mut self, path: String) -> (Result<(), CodebaseError>, Self) {
            self.reentrancy.log.push(ReentrancyEvent::AnalyzeFile(path));
            (self.reentrancy.error.take().map_or(Ok(()), Err), self)
        }

        fn define_macro(
            &mut self,
            _: Self::Span,
            (params, _): (Vec<Self::Ident>, Vec<Self::Span>),
            (body, _): TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
        ) -> Self::MacroId {
            self.reentrancy
                .log
                .push(ReentrancyEvent::DefineMacro(params, body));
            self.reentrancy.macro_alloc.gen()
        }

        fn call_macro(
            self,
            (id, _): (Self::MacroId, Self::Span),
            (args, _): MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
        ) -> Self {
            self.reentrancy
                .log
                .push(ReentrancyEvent::InvokeMacro(id, args));
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analyze::macros::mock::{MacroTableEvent, MockMacroId};
    use crate::analyze::strings::FakeStringInterner;
    use crate::analyze::{Literal, MockCodebase};
    use crate::diag::DiagnosticsEvent;
    use crate::expr::Expr;
    use crate::log::*;
    use crate::object::builder::mock::{BackendEvent, MockSymbolId, SerialIdAllocator};
    use crate::session::resolve::{BasicNameTable, NameTableEvent};
    use crate::syntax::parser::mock::*;
    use crate::syntax::*;

    use std::fmt::Debug;
    use std::iter;

    #[test]
    fn include_source_file() {
        let path = "my_file.s";
        let tokens = vec![(Ok(Token::Ident("NOP".into())), ())];
        let log = Fixture::new(|fixture| fixture.codebase.set_file(path, tokens.clone()))
            .log_session(|session| session.analyze_file(path.into()).0.unwrap());
        assert_eq!(log, [ParserEvent::ParseTokenStream(tokens).into()]);
    }

    #[test]
    fn define_and_call_macro() {
        let tokens = vec![Token::Ident("NOP".into())];
        let spans: Vec<_> = iter::repeat(()).take(tokens.len()).collect();
        let log = Fixture::default().log_session(|mut session| {
            let id = ReentrancyActions::define_macro(
                &mut session,
                (),
                (vec![], vec![]),
                (tokens.clone(), spans.clone()),
            );
            session.call_macro((id, ()), (vec![], vec![]));
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
            let id = ReentrancyActions::define_macro(
                &mut session,
                (),
                (vec![param.into()], vec![()]),
                (tokens.clone(), vec![(), (), ()]),
            );
            session.call_macro((id, ()), (vec![vec![arg.clone()]], vec![vec![()]]));
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

    type MockParserFactory<S> = crate::syntax::parser::mock::MockParserFactory<Event<S>>;
    type MockMacroTable<S> = crate::analyze::macros::mock::MockMacroTable<usize, Event<S>>;
    type MockDiagnosticsSystem<S> = crate::diag::MockDiagnosticsSystem<Event<S>, S>;
    type MockNameTable<S> = crate::session::resolve::MockNameTable<
        BasicNameTable<&'static Keyword, MockMacroId, MockSymbolId>,
        Event<S>,
    >;
    type MockBuilder<S> =
        crate::object::builder::mock::MockBackend<SerialIdAllocator<MockSymbolId>, Event<S>>;
    type TestCompositeSession<'a, S> = CompositeSession<
        SourceComponents<
            &'a mut MockCodebase<S>,
            &'a mut MockParserFactory<S>,
            &'a mut MockMacroTable<S>,
            &'a mut FakeStringInterner,
            &'a mut MockDiagnosticsSystem<S>,
        >,
        &'a mut MockNameTable<S>,
        MockBuilder<S>,
    >;

    #[derive(Debug, PartialEq)]
    enum Event<S: Clone> {
        Parser(ParserEvent<String, Literal<String>, LexError, S>),
        MacroTable(MacroTableEvent),
        Diagnostics(DiagnosticsEvent<S>),
        NameTable(NameTableEvent<&'static Keyword, MockMacroId, MockSymbolId>),
        Backend(BackendEvent<MockSymbolId, Expr<MockSymbolId, S>>),
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

    impl<S: Clone> From<NameTableEvent<&'static Keyword, MockMacroId, MockSymbolId>> for Event<S> {
        fn from(event: NameTableEvent<&'static Keyword, MockMacroId, MockSymbolId>) -> Self {
            Event::NameTable(event)
        }
    }

    impl<S: Clone> From<BackendEvent<MockSymbolId, Expr<MockSymbolId, S>>> for Event<S> {
        fn from(event: BackendEvent<MockSymbolId, Expr<MockSymbolId, S>>) -> Self {
            Event::Backend(event)
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

        fn log_session(mut self, f: impl FnOnce(TestCompositeSession<S>)) -> Vec<Event<S>>
        where
            Event<S>: Debug,
        {
            f(CompositeSession {
                reentrancy: SourceComponents::new(
                    &mut self.inner.codebase,
                    &mut self.inner.analyzer,
                    &mut self.inner.macros,
                    &mut self.inner.interner,
                    &mut self.inner.diagnostics,
                ),
                names: &mut MockNameTable::new(BasicNameTable::default(), self.log.clone()),
                builder: MockBuilder::new(SerialIdAllocator::new(MockSymbolId), self.log.clone()),
            });
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
