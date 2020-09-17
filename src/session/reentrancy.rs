use super::lex::{Lex, Literal, SemanticToken, StringSource};
use super::macros::MacroTable;
use super::resolve::{NameTable, StartScope};
use super::strings::GetString;
use super::CompositeSession;

use crate::codebase::CodebaseError;
use crate::diag::*;
use crate::semantics::{Semantics, TokenStreamState};
use crate::session::builder::Backend;
use crate::span::SpanSource;
use crate::syntax::parser::ParserFactory;
use crate::syntax::{IdentSource, LexError, ParseTokenStream};

use std::ops::Deref;

#[cfg(test)]
pub(crate) use self::mock::*;

pub(crate) trait ReentrancyActions: IdentSource + SpanSource + StringSource {
    fn analyze_file(&mut self, path: Self::StringRef) -> Result<(), CodebaseError>;
}

pub type MacroArgs<I, R, S> = super::macros::MacroArgs<SemanticToken<I, R>, S>;
pub type Params<I, S> = (Vec<I>, Vec<S>);

pub(crate) struct SourceComponents<C, P, I> {
    pub codebase: C,
    pub parser_factory: P,
    pub interner: I,
}

impl<C, P, M, I, N, B, D> SpanSource for CompositeSession<SourceComponents<C, P, I>, M, N, B, D>
where
    D: SpanSource,
{
    type Span = D::Span;
}

#[cfg(test)]
impl<C, P, I> SourceComponents<C, P, I> {
    pub fn new(codebase: C, parser_factory: P, interner: I) -> Self {
        SourceComponents {
            codebase,
            parser_factory,
            interner,
        }
    }
}

impl<C, P, M, I, N, B, D> ReentrancyActions
    for CompositeSession<SourceComponents<C, P, I>, M, N, B, D>
where
    Self: Lex<Span = D::Span>,
    P: ParserFactory<
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
    I: GetString<<Self as StringSource>::StringRef>,
    D: DiagnosticsSystem,
    Self: StartScope<<Self as IdentSource>::Ident> + NameTable<<Self as IdentSource>::Ident>,
    Self: Backend<D::Span>,
    <Self as IdentSource>::Ident: 'static,
    <Self as StringSource>::StringRef: 'static,
    <Self as SpanSource>::Span: 'static,
{
    fn analyze_file(&mut self, path: Self::StringRef) -> Result<(), CodebaseError> {
        let mut tokens = self.lex_file(path)?;
        let mut parser = self.reentrancy.parser_factory.mk_parser();
        let semantics = Semantics {
            session: self,
            state: TokenStreamState::new(),
            tokens: &mut tokens,
        };
        parser.parse_token_stream(semantics);
        Ok(())
    }
}

impl<C, P, I, R> GetString<R> for SourceComponents<C, P, I>
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

    use crate::diag::DiagnosticsEvent;
    use crate::log::Log;
    use crate::session::macros::mock::MockMacroId;
    use crate::session::macros::MacroSource;

    use std::marker::PhantomData;

    #[derive(Debug, PartialEq)]
    pub(crate) enum ReentrancyEvent {
        AnalyzeFile(String),
    }

    pub struct MockSourceComponents<T, S> {
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

    impl<T, S> MockSourceComponents<T, S> {
        fn with_name_table(log: Log<T>) -> Self {
            MockSourceComponents {
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

    impl<T, S, M, N, B, D> ReentrancyActions
        for CompositeSession<MockSourceComponents<T, S>, M, N, B, D>
    where
        T: From<ReentrancyEvent> + From<DiagnosticsEvent<S>>,
        S: Clone + Merge,
        D: Diagnostics<S>,
    {
        fn analyze_file(&mut self, path: String) -> Result<(), CodebaseError> {
            self.reentrancy.log.push(ReentrancyEvent::AnalyzeFile(path));
            self.reentrancy.error.take().map_or(Ok(()), Err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::diag::DiagnosticsEvent;
    use crate::expr::Expr;
    use crate::log::*;
    use crate::session::builder::mock::{BackendEvent, MockSymbolId, SerialIdAllocator};
    use crate::session::lex::{Literal, MockCodebase};
    use crate::session::macros::mock::{MacroTableEvent, MockMacroId};
    use crate::session::resolve::{BasicNameTable, NameTableEvent};
    use crate::session::strings::FakeStringInterner;
    use crate::syntax::parser::mock::*;
    use crate::syntax::*;

    use std::fmt::Debug;

    #[test]
    fn include_source_file() {
        let path = "my_file.s";
        let tokens = vec![(Ok(Token::Ident("NOP".into())), ())];
        let log = Fixture::new(|fixture| fixture.codebase.set_file(path, tokens.clone()))
            .log_session(|mut session| session.analyze_file(path.into()).unwrap());
        assert_eq!(log, [ParserEvent::ParseTokenStream(tokens).into()]);
    }

    impl Default for MockSpan<&'static str> {
        fn default() -> Self {
            unreachable!()
        }
    }

    type MockParserFactory<S> = crate::syntax::parser::mock::MockParserFactory<Event<S>>;
    type MockMacroTable<S> = crate::session::macros::mock::MockMacroTable<Event<S>>;
    type MockDiagnosticsSystem<S> = crate::diag::MockDiagnosticsSystem<Event<S>, S>;
    type MockNameTable<S> =
        crate::session::resolve::MockNameTable<BasicNameTable<MockMacroId, MockSymbolId>, Event<S>>;
    type MockBuilder<S> =
        crate::session::builder::mock::MockBackend<SerialIdAllocator<MockSymbolId>, Event<S>>;
    type TestCompositeSession<S> = CompositeSession<
        SourceComponents<MockCodebase<S>, MockParserFactory<S>, FakeStringInterner>,
        MockMacroTable<S>,
        MockNameTable<S>,
        MockBuilder<S>,
        MockDiagnosticsSystem<S>,
    >;

    #[derive(Debug, PartialEq)]
    enum Event<S: Clone> {
        Parser(ParserEvent<String, Literal<String>, LexError, S>),
        MacroTable(MacroTableEvent),
        Diagnostics(DiagnosticsEvent<S>),
        NameTable(NameTableEvent<MockMacroId, MockSymbolId>),
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

    impl<S: Clone> From<NameTableEvent<MockMacroId, MockSymbolId>> for Event<S> {
        fn from(event: NameTableEvent<MockMacroId, MockSymbolId>) -> Self {
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

        fn log_session(self, f: impl FnOnce(TestCompositeSession<S>)) -> Vec<Event<S>>
        where
            Event<S>: Debug,
        {
            f(CompositeSession {
                reentrancy: SourceComponents::new(
                    self.inner.codebase,
                    self.inner.analyzer,
                    self.inner.interner,
                ),
                macros: self.inner.macros,
                names: MockNameTable::new(BasicNameTable::default(), self.log.clone()),
                builder: MockBuilder::new(SerialIdAllocator::new(MockSymbolId), self.log.clone()),
                diagnostics: self.inner.diagnostics,
            });
            self.log.into_inner()
        }
    }

    impl<S: Clone + Default + Merge> Default for Fixture<S> {
        fn default() -> Self {
            Self::new(|_| {})
        }
    }
}
