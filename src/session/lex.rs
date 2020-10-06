use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::session::{CompositeSession, Interner, TokenStream};
use crate::span::{FileInclusionMetadata, Span, SpanSource, SpanSystem};
use crate::syntax::*;

use std::fmt::Debug;
use std::hash::Hash;
use std::rc::Rc;

pub type LexItem<R, S> = (Result<SemanticToken<R>, LexError>, S);
pub type SemanticToken<R> = crate::syntax::Token<R, Literal<R>>;

#[derive(Clone, Debug, PartialEq)]
pub enum Literal<R> {
    Number(i32),
    String(R),
}

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

pub type TokenSeq<R, S> = (Vec<SemanticToken<R>>, Vec<S>);

impl<'a, C, R, I, M, N, B, D> Lex<R, I> for CompositeSession<C, R, I, M, N, B, D>
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

pub trait StringSource {
    type StringRef: Clone + Debug + Eq + Hash;
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
