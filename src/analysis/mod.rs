use self::backend::*;
use self::resolve::{BiLevelNameTable, Ident};
use self::session::*;
use self::syntax::lexer::{LexError, Lexer};
use self::syntax::*;

use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::diag::*;
use crate::span::BufContext;
use crate::BuiltinNames;

use std::rc::Rc;

#[cfg(test)]
pub use self::mock::*;

pub mod backend;
mod macros;
mod resolve;
mod semantics;
mod session;
mod syntax;

pub(crate) trait Assemble<D>
where
    D: Diagnostics,
    Self: Backend<D::Span> + BuiltinNames<Name = <Self as AllocName<D::Span>>::Name> + Sized,
{
    fn assemble<C: Codebase>(
        self,
        name: &str,
        codebase: &C,
        diagnostics: &mut D,
    ) -> Result<(), CodebaseError> {
        use self::resolve::{Name, NameTable};

        let mut file_parser = CodebaseAnalyzer::new(codebase);
        let mut analyzer = semantics::SemanticAnalyzer;
        let mut names = BiLevelNameTable::new();
        for (string, name) in self.builtin_names() {
            names.insert(resolve::mk_ident(string), Name::Backend((*name).clone()))
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
    D: Diagnostics,
    B: Backend<D::Span> + BuiltinNames<Name = <Self as AllocName<D::Span>>::Name>,
{
}

type LexItem<T, S> = (Result<SemanticToken<T>, LexError>, S);
type SemanticToken<T> = syntax::Token<Ident<T>, Literal<T>, Command>;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Literal<S> {
    Operand(Operand),
    Number(i32),
    String(S),
}

trait Lex<D: Diagnostics> {
    type StringRef: Clone + Eq;
    type TokenIter: Iterator<Item = LexItem<Self::StringRef, D::Span>>;

    fn lex_file(
        &mut self,
        path: Self::StringRef,
        diagnostics: &mut D,
    ) -> Result<Self::TokenIter, CodebaseError>;
}

struct CodebaseAnalyzer<'a, T: 'a> {
    codebase: &'a T,
}

impl<'a, T: 'a> CodebaseAnalyzer<'a, T>
where
    T: StringRef,
{
    fn new(codebase: &T) -> CodebaseAnalyzer<T> {
        CodebaseAnalyzer { codebase }
    }
}

type TokenSeq<I, S> = Vec<(SemanticToken<I>, S)>;

impl<'a, T, D> Lex<D> for CodebaseAnalyzer<'a, T>
where
    T: Tokenize<D::BufContext> + 'a,
    T::StringRef: AsRef<str>,
    D: Diagnostics,
{
    type StringRef = T::StringRef;
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

pub(crate) trait StringRef {
    type StringRef: Clone + Eq;
}

trait Tokenize<C: BufContext>
where
    Self: StringRef,
{
    type Tokenized: Iterator<Item = LexItem<Self::StringRef, C::Span>>;
    fn tokenize_file<F: FnOnce(BufId) -> C>(
        &self,
        filename: &str,
        mk_context: F,
    ) -> Result<Self::Tokenized, CodebaseError>;
}

impl<C: Codebase> StringRef for C {
    type StringRef = String;
}

impl<C: Codebase, B: BufContext> Tokenize<B> for C {
    type Tokenized = TokenizedSrc<B>;

    fn tokenize_file<F: FnOnce(BufId) -> B>(
        &self,
        filename: &str,
        mk_context: F,
    ) -> Result<Self::Tokenized, CodebaseError> {
        let buf_id = self.open(filename)?;
        let rc_src = self.buf(buf_id);
        Ok(TokenizedSrc::new(rc_src, mk_context(buf_id)))
    }
}

struct TokenizedSrc<C> {
    tokens: Lexer<Rc<str>, MkIdent>,
    context: C,
}

type MkIdent = for<'a> fn(&'a str) -> Ident<String>;

impl<C: BufContext> TokenizedSrc<C> {
    fn new(src: Rc<str>, context: C) -> TokenizedSrc<C> {
        TokenizedSrc {
            tokens: syntax::tokenize(src, resolve::mk_ident),
            context,
        }
    }
}

impl<'a, C: BufContext> Iterator for TokenizedSrc<C> {
    type Item = LexItem<String, C::Span>;

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
        files: HashMap<String, Vec<LexItem<String, S>>>,
    }

    impl<S> MockCodebase<S> {
        pub fn new() -> Self {
            MockCodebase {
                files: HashMap::new(),
            }
        }

        pub(crate) fn set_file<I>(&mut self, path: &str, tokens: I)
        where
            I: IntoIterator<Item = LexItem<String, S>>,
        {
            self.files.insert(path.into(), tokens.into_iter().collect());
        }
    }

    impl<'a, D> Lex<D> for MockCodebase<D::Span>
    where
        D: Diagnostics,
    {
        type StringRef = String;
        type TokenIter = IntoIter<LexItem<Self::StringRef, D::Span>>;

        fn lex_file(
            &mut self,
            path: Self::StringRef,
            _diagnostics: &mut D,
        ) -> Result<Self::TokenIter, CodebaseError> {
            Ok(self.files[&path].clone().into_iter())
        }
    }
}
