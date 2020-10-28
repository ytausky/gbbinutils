use super::lex::Lex;
use super::macros::MacroTable;
use super::resolve::StartScope;
use super::*;

use crate::assembler::semantics::{Semantics, TokenStreamState};
use crate::assembler::syntax::parser::{DefaultParserFactory, ParserFactory};
use crate::assembler::syntax::{LexError, Literal, ParseTokenStream};
use crate::codebase::{Codebase, CodebaseError};
use crate::span::{SpanSource, SpanSystem};

impl<C, R, D> ReentrancyActions<R::Span> for CompositeSession<C, R, D>
where
    C: Codebase,
    Self: Lex<R, Span = R::Span>,
    Self: NextToken,
    Self: MacroTable<<Self as SpanSource>::Span>,
    R: SpanSystem,
    Self: EmitDiag<R::Span, R::Stripped>,
    Self: StartScope + NameTable<StringRef>,
    Self: Backend<R::Span>,
    <Self as SpanSource>::Span: 'static,
    <Self as Lex<R>>::TokenIter: 'static,
    for<'a> DiagnosticsContext<'a, C, R, D>: EmitDiag<R::Span, R::Stripped>,
    R::Stripped: Clone,
{
    fn analyze_file(
        &mut self,
        path: StringRef,
        from: Option<R::Span>,
    ) -> Result<(), CodebaseError> {
        #[cfg(test)]
        self.log_event(Event::AnalyzeFile {
            path: path.clone(),
            from: from.clone(),
        });

        let tokens = self.lex_file(path, from)?;
        self.tokens.push(Box::new(tokens));
        let mut parser = <DefaultParserFactory as ParserFactory<
            StringRef,
            Literal,
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
