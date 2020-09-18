use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::session::resolve::*;
use crate::session::{CompositeSession, Interner, TokenStream};
use crate::span::{FileInclusion, SpanDraft, SpanSource, SpanSystem};
use crate::syntax::*;

use std::fmt::Debug;
use std::rc::Rc;

pub type LexItem<I, R, S> = (Result<SemanticToken<I, R>, LexError>, S);
pub type SemanticToken<I, R> = crate::syntax::Token<I, Literal<R>>;

#[derive(Clone, Debug, PartialEq)]
pub enum Literal<R> {
    Number(i32),
    String(R),
}

pub(crate) trait Lex<R, I>: IdentSource + StringSource + SpanSource
where
    R: SpanSource,
    I: Interner,
{
    type TokenIter: TokenStream<R, I, Ident = Self::Ident>;

    fn lex_file(
        &mut self,
        path: Self::StringRef,
        from: Option<R::Span>,
    ) -> Result<Self::TokenIter, CodebaseError>;
}

pub type TokenSeq<I, R, S> = Vec<(SemanticToken<I, R>, S)>;

impl<'a, C, R, I, M, N, B, D> Lex<R, I> for CompositeSession<C, R, I, M, N, B, D>
where
    C: Codebase,
    I: Interner<StringRef = String>,
    R: SpanSystem<Ident<String>, I::StringRef>,
{
    type TokenIter = TokenizedSrc<DefaultIdentFactory, R::Span>;

    fn lex_file(
        &mut self,
        path: Self::StringRef,
        from: Option<R::Span>,
    ) -> Result<Self::TokenIter, CodebaseError> {
        let buf_id = self.codebase.open(&path)?;
        let rc_src = self.codebase.buf(buf_id);
        Ok(TokenizedSrc::new(
            rc_src,
            Rc::new(FileInclusion { file: buf_id, from }),
        ))
    }
}

pub trait StringSource {
    type StringRef: Clone + Debug + Eq;
}

pub struct TokenizedSrc<F, S> {
    tokens: Lexer<Rc<str>, F>,
    inclusion: Rc<FileInclusion<BufId, S>>,
}

impl<S> TokenizedSrc<DefaultIdentFactory, S> {
    fn new(src: Rc<str>, inclusion: Rc<FileInclusion<BufId, S>>) -> Self {
        TokenizedSrc {
            tokens: Lexer::new(src, DefaultIdentFactory),
            inclusion,
        }
    }
}

impl<F: IdentFactory, S> IdentSource for TokenizedSrc<F, S> {
    type Ident = F::Ident;
}

impl<R, I, F, S> TokenStream<R, I> for TokenizedSrc<F, S>
where
    R: SpanSystem<F::Ident, I::StringRef, Span = S>,
    I: Interner,
    F: IdentFactory,
    S: Clone,
{
    fn next_token(
        &mut self,
        registry: &mut R,
        interner: &mut I,
    ) -> Option<LexItem<Self::Ident, I::StringRef, R::Span>> {
        self.tokens.next_token(registry, interner).map(|(t, r)| {
            (
                t,
                registry.encode_span(SpanDraft::File(Rc::clone(&self.inclusion), r)),
            )
        })
    }
}
