use super::lex::Lex;
use super::macros::MacroTable;
use super::resolve::StartScope;
use super::*;

use crate::assembler::semantics::{Semantics, TokenStreamState};
use crate::assembler::syntax::parser::{DefaultParserFactory, ParserFactory};
use crate::assembler::syntax::{LexError, Literal, ParseTokenStream};
use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::span::{SpanSource, SpanSystem};

impl<C, R, I, D, L> ReentrancyActions<<Self as StringSource>::StringRef, R::Span>
    for CompositeSession<C, R, I, D, L>
where
    C: Codebase,
    Self: Lex<R, I, Span = R::Span>,
    Self: Interner<StringRef = I::StringRef>,
    Self: NextToken,
    Self: MacroTable<
        <Self as StringSource>::StringRef,
        Literal<<Self as StringSource>::StringRef>,
        <Self as SpanSource>::Span,
    >,
    R: SpanSystem<BufId>,
    I: Interner,
    Self: EmitDiag<R::Span, R::Stripped>,
    Self: StartScope + NameTable<<Self as StringSource>::StringRef>,
    Self: Backend<R::Span>,
    Self: Log<
        <Self as SymbolSource>::SymbolId,
        <Self as MacroSource>::MacroId,
        I::StringRef,
        R::Span,
        R::Stripped,
    >,
    <Self as StringSource>::StringRef: 'static,
    <Self as SpanSource>::Span: 'static,
    <Self as Lex<R, I>>::TokenIter: 'static,
{
    fn analyze_file(
        &mut self,
        path: <Self as StringSource>::StringRef,
        from: Option<R::Span>,
    ) -> Result<(), CodebaseError> {
        self.log(|| Event::AnalyzeFile {
            path: path.clone(),
            from: from.clone(),
        });

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
