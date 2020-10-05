use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::session::{CompositeSession, Interner, TokenStream};
use crate::span::{FileInclusion, Span, SpanSource, SpanSystem};
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

pub type TokenSeq<R, S> = Vec<(SemanticToken<R>, S)>;

impl<'a, C, R, I, M, N, B, D> Lex<R, I> for CompositeSession<C, R, I, M, N, B, D>
where
    C: Codebase,
    I: Interner,
    R: SpanSystem<Token<I::StringRef, Literal<I::StringRef>>, I::StringRef>,
{
    type TokenIter = TokenizedSrc<R::Span>;

    fn lex_file(
        &mut self,
        path: Self::StringRef,
        from: Option<R::Span>,
    ) -> Result<Self::TokenIter, CodebaseError> {
        let buf_id = self.codebase.open(&self.interner.get_string(&path))?;
        let rc_src = self.codebase.buf(buf_id);
        Ok(TokenizedSrc::new(
            rc_src,
            Rc::new(FileInclusion { file: buf_id, from }),
        ))
    }
}

pub trait StringSource {
    type StringRef: Clone + Debug + Eq + Hash;
}

pub struct TokenizedSrc<S> {
    tokens: Lexer<Rc<str>>,
    inclusion: Rc<FileInclusion<BufId, S>>,
}

impl<S> TokenizedSrc<S> {
    fn new(src: Rc<str>, inclusion: Rc<FileInclusion<BufId, S>>) -> Self {
        TokenizedSrc {
            tokens: Lexer::new(src),
            inclusion,
        }
    }
}

impl<R, I, S> TokenStream<R, I> for TokenizedSrc<S>
where
    R: SpanSystem<Token<I::StringRef, Literal<I::StringRef>>, I::StringRef, Span = S>,
    I: Interner,
    S: Clone,
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
