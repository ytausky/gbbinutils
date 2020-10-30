use super::macros::MacroTable;
use super::resolve::StartScope;
use super::*;

use crate::assembler::semantics::SemanticActions;
use crate::assembler::syntax::parser::{DefaultParserFactory, ParserFactory};
use crate::assembler::syntax::{LexError, Literal, ParseTokenStream};
use crate::codebase::CodebaseError;
use crate::span::{SpanSource, SpanSystem};

impl<'a, R> ReentrancyActions<R::Span> for CompositeSession<'a, R>
where
    Self: NextToken,
    Self: MacroTable<<Self as SpanSource>::Span>,
    R: SpanSystem,
    Self: SpanSource<Span = R::Span>,
    Self: EmitDiag<R::Span, R::Stripped>,
    Self: StartScope + NameTable<StringRef>,
    Self: Backend<R::Span>,
    for<'r> DiagnosticsContext<'r, FileCodebase<'a>, R, OutputForwarder<'a>>:
        EmitDiag<R::Span, R::Stripped>,
    R::Span: 'static,
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
        parser.parse_token_stream(self.semantic_actions());
        Ok(())
    }
}
