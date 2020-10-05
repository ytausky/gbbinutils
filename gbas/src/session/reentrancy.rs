use super::lex::{Lex, Literal, StringSource};
use super::macros::MacroTable;
use super::resolve::{NameTable, StartScope};
use super::{CompositeSession, Interner, NextToken};

use crate::codebase::{Codebase, CodebaseError};
use crate::diagnostics::*;
use crate::semantics::{Semantics, TokenStreamState};
use crate::session::builder::Backend;
use crate::span::{SpanSource, SpanSystem};
use crate::syntax::parser::{DefaultParserFactory, ParserFactory};
use crate::syntax::{LexError, ParseTokenStream, Token};

#[cfg(test)]
pub(crate) use self::mock::*;

pub(crate) trait ReentrancyActions<R, S> {
    fn analyze_file(&mut self, path: R, from: Option<S>) -> Result<(), CodebaseError>;
}

pub type Params<I, S> = Vec<(I, S)>;

impl<C, R, I, M, N, B, D> ReentrancyActions<<Self as StringSource>::StringRef, R::Span>
    for CompositeSession<C, R, I, M, N, B, D>
where
    C: Codebase,
    Self: Lex<R, I, Span = R::Span>,
    Self: Interner,
    Self: NextToken,
    Self: MacroTable<
        <Self as StringSource>::StringRef,
        Literal<<Self as StringSource>::StringRef>,
        <Self as SpanSource>::Span,
    >,
    R: SpanSystem<
        Token<<Self as StringSource>::StringRef, Literal<<Self as StringSource>::StringRef>>,
        <Self as StringSource>::StringRef,
    >,
    I: Interner,
    Self: EmitDiag<R::Span, R::Stripped>,
    Self: StartScope + NameTable<<Self as StringSource>::StringRef>,
    Self: Backend<R::Span>,
    <Self as StringSource>::StringRef: 'static,
    <Self as SpanSource>::Span: 'static,
    <Self as Lex<R, I>>::TokenIter: 'static,
{
    fn analyze_file(
        &mut self,
        path: <Self as StringSource>::StringRef,
        from: Option<R::Span>,
    ) -> Result<(), CodebaseError> {
        let tokens = self.lex_file(path, from)?;
        self.tokens.push(Box::new(tokens));
        let mut parser = <DefaultParserFactory as ParserFactory<
            <Self as StringSource>::StringRef,
            Literal<<Self as StringSource>::StringRef>,
            LexError,
            <Self as SpanSource>::Span,
        >>::mk_parser(&mut DefaultParserFactory);
        let semantics = Semantics {
            session: self,
            state: TokenStreamState::new(),
        };
        parser.parse_token_stream(semantics);
        Ok(())
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::log::Log;
    use crate::session::macros::mock::MockMacroId;
    use crate::session::macros::MacroSource;

    use std::marker::PhantomData;

    #[derive(Debug, PartialEq)]
    pub(crate) enum ReentrancyEvent {
        AnalyzeFile(String),
    }

    pub struct MockCodebase<T, S> {
        log: Log<T>,
        error: Option<CodebaseError>,
        _span: PhantomData<S>,
    }

    impl<T, S: Clone> MacroSource for MockCodebase<T, S> {
        type MacroId = MockMacroId;
    }

    impl<T, S: Clone> SpanSource for MockCodebase<T, S> {
        type Span = S;
    }

    impl<T, S> MockCodebase<T, S> {
        fn with_name_table(log: Log<T>) -> Self {
            MockCodebase {
                log,
                error: None,
                _span: PhantomData,
            }
        }

        pub fn fail(&mut self, error: CodebaseError) {
            self.error = Some(error)
        }
    }

    impl<T, S> MockCodebase<T, S> {
        pub fn with_log(log: Log<T>) -> Self {
            Self::with_name_table(log)
        }
    }

    impl<T, S> StringSource for MockCodebase<T, S> {
        type StringRef = String;
    }

    impl<T, S, R: SpanSource, I, M, N, B, D> ReentrancyActions<String, S>
        for CompositeSession<MockCodebase<T, S>, R, I, M, N, B, D>
    where
        T: From<ReentrancyEvent>,
        I: StringSource,
    {
        fn analyze_file(&mut self, path: String, _from: Option<S>) -> Result<(), CodebaseError> {
            self.codebase.log.push(ReentrancyEvent::AnalyzeFile(path));
            self.codebase.error.take().map_or(Ok(()), Err)
        }
    }
}
