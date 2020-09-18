use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::session::resolve::*;
use crate::session::CompositeSession;
use crate::span::{BufContext, BufContextFactory, SpanSource};
use crate::syntax::*;

use std::rc::Rc;
use std::fmt::Debug;

pub type LexItem<I, R, S> = (Result<SemanticToken<I, R>, LexError>, S);
pub type SemanticToken<I, R> = crate::syntax::Token<I, Literal<R>>;

#[derive(Clone, Debug, PartialEq)]
pub enum Literal<R> {
    Number(i32),
    String(R),
}

pub trait Lex: IdentSource + StringSource + SpanSource {
    type TokenIter: Iterator<Item = LexItem<Self::Ident, Self::StringRef, Self::Span>>;

    fn lex_file(&mut self, path: Self::StringRef) -> Result<Self::TokenIter, CodebaseError>;
}

pub type TokenSeq<I, R, S> = (Vec<SemanticToken<I, R>>, Vec<S>);

impl<'a, C, R, M, N, B, D> Lex for CompositeSession<C, R, M, N, B, D>
where
    C: Codebase,
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

pub trait Tokenize<C: BufContext>: IdentSource + StringSource {
    type Tokenized: Iterator<Item = LexItem<Self::Ident, Self::StringRef, C::Span>>;

    fn tokenize_file<F: FnOnce(BufId) -> C>(
        &self,
        filename: &str,
        mk_context: F,
    ) -> Result<Self::Tokenized, CodebaseError>;
}

pub struct Tokenizer<T>(pub T);

impl<T> IdentSource for Tokenizer<T> {
    type Ident = <DefaultIdentFactory as IdentSource>::Ident;
}

impl<T> StringSource for Tokenizer<T> {
    type StringRef = String;
}

impl<C: Codebase, B: BufContext> Tokenize<B> for Tokenizer<&C> {
    type Tokenized = TokenizedSrc<DefaultIdentFactory, B>;

    fn tokenize_file<F: FnOnce(BufId) -> B>(
        &self,
        filename: &str,
        mk_context: F,
    ) -> Result<Self::Tokenized, CodebaseError> {
        let buf_id = self.0.open(filename)?;
        let rc_src = self.0.buf(buf_id);
        Ok(TokenizedSrc::new(rc_src, mk_context(buf_id)))
    }
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

impl<'a, F: IdentFactory, C: BufContext> Iterator for TokenizedSrc<F, C> {
    type Item = LexItem<F::Ident, String, C::Span>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tokens
            .next()
            .map(|(t, r)| (t, self.context.mk_span(r)))
    }
}
