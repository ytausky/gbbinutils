use super::StringSource;

use crate::assembler::session::{CompositeSession, Interner, TokenStream};
use crate::assembler::syntax::*;
use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::span::{FileInclusionMetadata, Span, SpanSource, SpanSystem};

use std::rc::Rc;

pub(crate) trait Lex<R, I>: StringSource + SpanSource
where
    R: SpanSource,
    I: Interner,
{
    type TokenIter: TokenStream<R, I>;

    fn lex_file(
        &mut self,
        path: Self::StringRef,
        from: Option<R::Span>,
    ) -> Result<Self::TokenIter, CodebaseError>;
}

impl<'a, C, R, I, M, N, B, D, L> Lex<R, I> for CompositeSession<C, R, I, M, N, B, D, L>
where
    C: Codebase,
    I: Interner,
    R: SpanSystem<BufId>,
{
    type TokenIter = TokenizedSrc<R::FileInclusionMetadataId>;

    fn lex_file(
        &mut self,
        path: Self::StringRef,
        from: Option<R::Span>,
    ) -> Result<Self::TokenIter, CodebaseError> {
        let buf_id = self.codebase.open(&self.interner.get_string(&path))?;
        let rc_src = self.codebase.buf(buf_id);
        Ok(TokenizedSrc::new(
            rc_src,
            self.registry
                .add_file_inclusion(FileInclusionMetadata { file: buf_id, from }),
        ))
    }
}

pub struct TokenizedSrc<I> {
    tokens: Lexer<Rc<str>>,
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

impl<R, I> TokenStream<R, I> for TokenizedSrc<R::FileInclusionMetadataId>
where
    R: SpanSystem<BufId>,
    I: Interner,
{
    fn next_token(
        &mut self,
        registry: &mut R,
        interner: &mut I,
    ) -> Option<LexItem<I::StringRef, R::Span>> {
        self.tokens.next_token(registry, interner).map(|(t, r)| {
            (
                t,
                registry.encode_span(Span::File(self.inclusion.clone(), r)),
            )
        })
    }
}
