use crate::assembler::session::{CompositeSession, TokenStream};
use crate::assembler::string_ref::StringRef;
use crate::assembler::syntax::*;
use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::span::{FileInclusionMetadata, Span, SpanSource, SpanSystem};

use std::rc::Rc;

pub(crate) trait Lex<R>: SpanSource
where
    R: SpanSource,
{
    type TokenIter: TokenStream<R>;

    fn lex_file(
        &mut self,
        path: StringRef,
        from: Option<R::Span>,
    ) -> Result<Self::TokenIter, CodebaseError>;
}

impl<'a, C, R, D> Lex<R> for CompositeSession<C, R, D>
where
    C: Codebase,
    R: SpanSystem<BufId>,
{
    type TokenIter = TokenizedSrc<R::FileInclusionMetadataId>;

    fn lex_file(
        &mut self,
        path: StringRef,
        from: Option<R::Span>,
    ) -> Result<Self::TokenIter, CodebaseError> {
        let buf_id = self.codebase.open(&path)?;
        let rc_src = self.codebase.buf(buf_id);
        Ok(TokenizedSrc::new(
            rc_src,
            self.metadata
                .add_file_inclusion(FileInclusionMetadata { file: buf_id, from }),
        ))
    }
}

pub struct TokenizedSrc<I> {
    tokens: Lexer,
    inclusion: I,
}

impl<I> TokenizedSrc<I> {
    fn new(src: Rc<str>, inclusion: I) -> Self {
        TokenizedSrc {
            tokens: Lexer::new(src),
            inclusion,
        }
    }
}

impl<R> TokenStream<R> for TokenizedSrc<R::FileInclusionMetadataId>
where
    R: SpanSystem<BufId>,
{
    fn next_token(&mut self, registry: &mut R) -> Option<LexItem<R::Span>> {
        self.tokens.next_token(registry).map(|(t, r)| {
            (
                t,
                registry.encode_span(Span::File(self.inclusion.clone(), r)),
            )
        })
    }
}
