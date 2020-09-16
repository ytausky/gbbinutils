use super::resolve::{Ident, NameTable, StartScope};
use super::{CompositeSession, SessionImpl};

use crate::analyze::macros::{MacroSource, MacroTable};
use crate::analyze::strings::GetString;
use crate::analyze::{Lex, Literal, SemanticToken, StringSource, TokenSeq};
use crate::codebase::{BufId, BufRange, CodebaseError};
use crate::diag::span::{RcSpan, SpanSource};
use crate::diag::*;
use crate::semantics::{Semantics, TokenStreamState};
use crate::session::builder::Backend;
use crate::syntax::parser::{DefaultParserFactory, ParserFactory};
use crate::syntax::{IdentSource, LexError, ParseTokenStream};

use std::ops::Deref;

#[cfg(test)]
pub(crate) use self::mock::*;

pub(crate) trait ReentrancyActions:
    IdentSource + MacroSource + SpanSource + StringSource
{
    fn analyze_file(&mut self, path: Self::StringRef) -> Result<(), CodebaseError>;

    fn define_macro(
        &mut self,
        name_span: Self::Span,
        params: Params<Self::Ident, Self::Span>,
        body: TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
    ) -> Self::MacroId;

    fn call_macro(
        &mut self,
        name: (Self::MacroId, Self::Span),
        args: MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
    );
}

pub type MacroArgs<I, R, S> = crate::analyze::macros::MacroArgs<SemanticToken<I, R>, S>;
pub type Params<I, S> = (Vec<I>, Vec<S>);

pub(crate) struct SourceComponents<C, P, M, I> {
    pub codebase: C,
    pub parser_factory: P,
    pub macros: M,
    pub interner: I,
}

impl<C, P, M, I, N, B, D> SpanSource for CompositeSession<SourceComponents<C, P, M, I>, N, B, D>
where
    D: SpanSource,
{
    type Span = D::Span;
}

#[cfg(test)]
impl<C, P, M, I> SourceComponents<C, P, M, I> {
    pub fn new(codebase: C, parser_factory: P, macros: M, interner: I) -> Self {
        SourceComponents {
            codebase,
            parser_factory,
            macros,
            interner,
        }
    }
}

impl<'a, 'b> ReentrancyActions for SessionImpl<'a, 'b> {
    fn analyze_file(&mut self, path: Self::StringRef) -> Result<(), CodebaseError> {
        let mut tokens = self.lex_file(path)?;
        let mut parser = <DefaultParserFactory as ParserFactory<
            Ident<String>,
            Literal<String>,
            LexError,
            RcSpan<BufId, BufRange>,
        >>::mk_parser(&mut DefaultParserFactory);
        let semantics = Semantics {
            session: self,
            state: TokenStreamState::new(),
            tokens: &mut tokens,
        };
        parser.parse_token_stream(semantics);
        Ok(())
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
        &mut self,
        id: (Self::MacroId, Self::Span),
        args: MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
    ) {
        let expansion = self.expand_macro(id, args);
        let mut parser = <DefaultParserFactory as ParserFactory<
            Ident<String>,
            Literal<String>,
            LexError,
            RcSpan<BufId, BufRange>,
        >>::mk_parser(&mut DefaultParserFactory);
        let mut tokens = expansion.map(|(t, s)| (Ok(t), s));
        let semantics = Semantics {
            session: self,
            state: TokenStreamState::new(),
            tokens: &mut tokens,
        };
        parser.parse_token_stream(semantics);
    }
}

impl<C, P, M, I, N, B, D> ReentrancyActions
    for CompositeSession<SourceComponents<C, P, M, I>, N, B, D>
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

    fn define_macro(
        &mut self,
        name_span: Self::Span,
        params: Params<Self::Ident, Self::Span>,
        body: TokenSeq<Self::Ident, Self::StringRef, Self::Span>,
    ) -> Self::MacroId {
        MacroTable::define_macro(self, name_span, params, body)
    }

    fn call_macro(
        &mut self,
        id: (Self::MacroId, Self::Span),
        args: MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
    ) {
        let expansion = self.expand_macro(id, args);
        let mut parser = self.reentrancy.parser_factory.mk_parser();
        let mut tokens = expansion.map(|(t, s)| (Ok(t), s));
        let semantics = Semantics {
            session: self,
            state: TokenStreamState::new(),
            tokens: &mut tokens,
        };
        parser.parse_token_stream(semantics);
    }
}

impl<C, P, M, I, R> GetString<R> for SourceComponents<C, P, M, I>
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
    use crate::diag::DiagnosticsEvent;
    use crate::log::Log;
    use crate::session::builder::mock::*;

    use std::marker::PhantomData;

    #[derive(Debug, PartialEq)]
    pub(crate) enum ReentrancyEvent {
        AnalyzeFile(String),
        DefineMacro(Vec<String>, Vec<SemanticToken<String, String>>),
        InvokeMacro(MockMacroId, Vec<Vec<SemanticToken<String, String>>>),
    }

    pub struct MockSourceComponents<T, S> {
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

    impl<T, S> MockSourceComponents<T, S> {
        fn with_name_table(log: Log<T>) -> Self {
            MockSourceComponents {
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

    impl<T, S, N, B, D> ReentrancyActions for CompositeSession<MockSourceComponents<T, S>, N, B, D>
    where
        T: From<ReentrancyEvent> + From<DiagnosticsEvent<S>>,
        S: Clone + Merge,
        D: Diagnostics<S>,
    {
        fn analyze_file(&mut self, path: String) -> Result<(), CodebaseError> {
            self.reentrancy.log.push(ReentrancyEvent::AnalyzeFile(path));
            self.reentrancy.error.take().map_or(Ok(()), Err)
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
            &mut self,
            (id, _): (Self::MacroId, Self::Span),
            (args, _): MacroArgs<Self::Ident, Self::StringRef, Self::Span>,
        ) {
            self.reentrancy
                .log
                .push(ReentrancyEvent::InvokeMacro(id, args))
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
    use crate::session::builder::mock::{BackendEvent, MockSymbolId, SerialIdAllocator};
    use crate::session::resolve::{BasicNameTable, NameTableEvent};
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
    type MockMacroTable<S> = crate::analyze::macros::mock::MockMacroTable<Event<S>>;
    type MockDiagnosticsSystem<S> = crate::diag::MockDiagnosticsSystem<Event<S>, S>;
    type MockNameTable<S> =
        crate::session::resolve::MockNameTable<BasicNameTable<MockMacroId, MockSymbolId>, Event<S>>;
    type MockBuilder<S> =
        crate::session::builder::mock::MockBackend<SerialIdAllocator<MockSymbolId>, Event<S>>;
    type TestCompositeSession<S> = CompositeSession<
        SourceComponents<
            MockCodebase<S>,
            MockParserFactory<S>,
            MockMacroTable<S>,
            FakeStringInterner,
        >,
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
                    self.inner.macros,
                    self.inner.interner,
                ),
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
