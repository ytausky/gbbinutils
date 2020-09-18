use crate::codebase::{Codebase, CodebaseError};
use crate::session::resolve::*;
use crate::session::{CompositeSession, TokenStream};
use crate::span::{BufContext, BufContextFactory, SpanSource};
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

pub(crate) trait Lex<R, I>: IdentSource + StringSource + SpanSource {
    type TokenIter: TokenStream<
        R,
        I,
        Ident = Self::Ident,
        StringRef = Self::StringRef,
        Span = Self::Span,
    >;

    fn lex_file(&mut self, path: Self::StringRef) -> Result<Self::TokenIter, CodebaseError>;
}

pub type TokenSeq<I, R, S> = (Vec<SemanticToken<I, R>>, Vec<S>);

impl<'a, C, R, I, M, N, B, D> Lex<R, I> for CompositeSession<C, R, I, M, N, B, D>
where
    C: Codebase,
    I: StringSource<StringRef = String>,
    R: BufContextFactory,
{
    type TokenIter = TokenizedSrc<DefaultIdentFactory, R::BufContext>;

    fn lex_file(&mut self, path: Self::StringRef) -> Result<Self::TokenIter, CodebaseError> {
        let buf_id = self.codebase.open(&path)?;
        let rc_src = self.codebase.buf(buf_id);
        Ok(TokenizedSrc::new(
            rc_src,
            self.registry.mk_buf_context(buf_id, None),
        ))
    }
}

pub trait StringSource {
    type StringRef: Clone + Debug + Eq;
}

pub struct TokenizedSrc<F, C> {
    tokens: Lexer<Rc<str>, F>,
    context: C,
}

impl<C: BufContext> TokenizedSrc<DefaultIdentFactory, C> {
    fn new(src: Rc<str>, context: C) -> Self {
        TokenizedSrc {
            tokens: Lexer::new(src, DefaultIdentFactory),
            context,
        }
    }
}

impl<F: IdentFactory, C> IdentSource for TokenizedSrc<F, C> {
    type Ident = F::Ident;
}

impl<F, C> StringSource for TokenizedSrc<F, C> {
    type StringRef = String;
}

impl<F, C: BufContext> SpanSource for TokenizedSrc<F, C> {
    type Span = C::Span;
}

impl<R, I, F: IdentFactory, C: BufContext> TokenStream<R, I> for TokenizedSrc<F, C> {
    fn next_token(
        &mut self,
        _registry: &mut R,
        _interner: &mut I,
    ) -> Option<LexItem<Self::Ident, Self::StringRef, Self::Span>> {
        self.tokens
            .next()
            .map(|(t, r)| (t, self.context.mk_span(r)))
    }
}
