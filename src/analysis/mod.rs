use self::backend::*;
use self::resolve::{BiLevelNameTable, DefaultIdentFactory};
use self::session::*;
use self::syntax::*;

use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::diag::*;
use crate::span::{BufContext, BufContextFactory, SpanSource};
use crate::BuiltinNames;

use std::rc::Rc;

#[cfg(test)]
pub use self::mock::*;

pub mod backend;
mod resolve;
mod semantics;
mod session;
mod syntax;

pub(crate) trait Assemble<D>
where
    D: DiagnosticsSystem,
    Self: Backend<D::Span> + BuiltinNames<Name = <Self as AllocName<D::Span>>::Name> + Sized,
{
    fn assemble<C: Codebase>(
        self,
        name: &str,
        codebase: &C,
        diagnostics: &mut D,
    ) -> Result<(), CodebaseError> {
        use self::resolve::{NameTable, ResolvedIdent};

        let tokenizer = Tokenizer(codebase);
        let mut file_parser = CodebaseAnalyzer::new(&tokenizer);
        let mut analyzer = semantics::SemanticAnalyzer;
        let mut names = BiLevelNameTable::new();
        for (string, name) in self.builtin_names() {
            names.insert(
                DefaultIdentFactory.mk_ident(string),
                ResolvedIdent::Backend((*name).clone()),
            )
        }
        let session = CompositeSession::new(
            &mut file_parser,
            &mut analyzer,
            self,
            &mut names,
            diagnostics,
        );
        session.analyze_file(name.into()).0
    }
}

impl<B, D> Assemble<D> for B
where
    D: DiagnosticsSystem,
    B: Backend<D::Span> + BuiltinNames<Name = <Self as AllocName<D::Span>>::Name>,
{
}

type LexItem<I, R, S> = (Result<SemanticToken<I, R>, LexError>, S);
type SemanticToken<I, R> = syntax::Token<I, Literal<R>>;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Literal<R> {
    Number(i32),
    String(R),
}

trait Lex<D: SpanSource>: IdentSource + StringSource {
    type TokenIter: Iterator<Item = LexItem<Self::Ident, Self::StringRef, D::Span>>;

    fn lex_file(
        &mut self,
        path: Self::StringRef,
        diagnostics: &mut D,
    ) -> Result<Self::TokenIter, CodebaseError>;
}

struct CodebaseAnalyzer<'a, T: 'a> {
    codebase: &'a T,
}

impl<'a, T: StringSource + 'a> CodebaseAnalyzer<'a, T> {
    fn new(codebase: &T) -> CodebaseAnalyzer<T> {
        CodebaseAnalyzer { codebase }
    }
}

type TokenSeq<I, R, S> = (Vec<SemanticToken<I, R>>, Vec<S>);

impl<'a, T, D> Lex<D> for CodebaseAnalyzer<'a, T>
where
    T: Tokenize<D::BufContext> + 'a,
    T::StringRef: AsRef<str>,
    D: BufContextFactory,
{
    type TokenIter = T::Tokenized;

    fn lex_file(
        &mut self,
        path: Self::StringRef,
        diagnostics: &mut D,
    ) -> Result<Self::TokenIter, CodebaseError> {
        self.codebase.tokenize_file(path.as_ref(), |buf_id| {
            diagnostics.mk_buf_context(buf_id, None)
        })
    }
}

impl<'a, T: IdentSource> IdentSource for CodebaseAnalyzer<'a, T> {
    type Ident = T::Ident;
}

impl<'a, T: StringSource> StringSource for CodebaseAnalyzer<'a, T> {
    type StringRef = T::StringRef;
}

pub(crate) trait StringSource {
    type StringRef: Clone + Eq;
}

trait Tokenize<C: BufContext>: IdentSource + StringSource {
    type Tokenized: Iterator<Item = LexItem<Self::Ident, Self::StringRef, C::Span>>;

    fn tokenize_file<F: FnOnce(BufId) -> C>(
        &self,
        filename: &str,
        mk_context: F,
    ) -> Result<Self::Tokenized, CodebaseError>;
}

struct Tokenizer<T>(T);

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

struct TokenizedSrc<F, C> {
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

    impl<D: BufContextFactory> Lex<D> for MockCodebase<D::Span> {
        type TokenIter = IntoIter<LexItem<Self::Ident, Self::StringRef, D::Span>>;

        fn lex_file(
            &mut self,
            path: Self::StringRef,
            _diagnostics: &mut D,
        ) -> Result<Self::TokenIter, CodebaseError> {
            Ok(self.files[&path].clone().into_iter())
        }
    }

    impl<S> IdentSource for MockCodebase<S> {
        type Ident = String;
    }

    impl<S> StringSource for MockCodebase<S> {
        type StringRef = String;
    }
}
