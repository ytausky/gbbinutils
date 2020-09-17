use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::session::reentrancy::SourceComponents;
use crate::session::resolve::*;
use crate::session::CompositeSession;
use crate::span::{BufContext, BufContextFactory, SpanSource};
use crate::syntax::*;

use std::rc::Rc;

#[cfg(test)]
pub use self::mock::*;

pub mod macros;
pub mod strings;

type LexItem<I, R, S> = (Result<SemanticToken<I, R>, LexError>, S);
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

pub struct CodebaseAnalyzer<'a, T: 'a> {
    codebase: &'a T,
}

impl<'a, T: StringSource + 'a> CodebaseAnalyzer<'a, T> {
    pub fn new(codebase: &T) -> CodebaseAnalyzer<T> {
        CodebaseAnalyzer { codebase }
    }
}

pub type TokenSeq<I, R, S> = (Vec<SemanticToken<I, R>>, Vec<S>);

impl<'a, T, P, M, I, N, B, D> Lex
    for CompositeSession<SourceComponents<CodebaseAnalyzer<'a, T>, P, I>, M, N, B, D>
where
    T: Tokenize<D::BufContext> + 'a,
    T::StringRef: AsRef<str>,
    D: BufContextFactory,
{
    type TokenIter = T::Tokenized;

    fn lex_file(&mut self, path: Self::StringRef) -> Result<Self::TokenIter, CodebaseError> {
        self.reentrancy
            .codebase
            .codebase
            .tokenize_file(path.as_ref(), |buf_id| {
                self.diagnostics.mk_buf_context(buf_id, None)
            })
    }
}

impl<'a, T: IdentSource> IdentSource for CodebaseAnalyzer<'a, T> {
    type Ident = T::Ident;
}

impl<'a, T: StringSource> StringSource for CodebaseAnalyzer<'a, T> {
    type StringRef = T::StringRef;
}

impl<'a, T: IdentSource, P, I> IdentSource for SourceComponents<CodebaseAnalyzer<'a, T>, P, I> {
    type Ident = T::Ident;
}

impl<'a, T: StringSource, P, I> StringSource for SourceComponents<CodebaseAnalyzer<'a, T>, P, I> {
    type StringRef = T::StringRef;
}

pub trait StringSource {
    type StringRef: Clone + Eq;
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

#[cfg(test)]
mod mock {
    use super::*;

    use std::collections::HashMap;
    use std::vec::IntoIter;

    pub struct MockCodebase<S> {
        files: HashMap<String, Vec<LexItem<String, String, S>>>,
    }

    impl<S> MockCodebase<S> {
        pub fn new() -> Self {
            MockCodebase {
                files: HashMap::new(),
            }
        }

        pub(crate) fn set_file<I>(&mut self, path: &str, tokens: I)
        where
            I: IntoIterator<Item = LexItem<String, String, S>>,
        {
            self.files.insert(path.into(), tokens.into_iter().collect());
        }
    }

    impl<'a, P, M, I, N, B, D> Lex
        for CompositeSession<SourceComponents<MockCodebase<D::Span>, P, I>, M, N, B, D>
    where
        D: SpanSource,
    {
        type TokenIter = IntoIter<LexItem<Self::Ident, Self::StringRef, D::Span>>;

        fn lex_file(&mut self, path: Self::StringRef) -> Result<Self::TokenIter, CodebaseError> {
            Ok(self.reentrancy.codebase.files[&path].clone().into_iter())
        }
    }

    impl<'a, P, I, S> IdentSource for SourceComponents<MockCodebase<S>, P, I> {
        type Ident = String;
    }

    impl<'a, P, I, S> StringSource for SourceComponents<MockCodebase<S>, P, I> {
        type StringRef = String;
    }
}
