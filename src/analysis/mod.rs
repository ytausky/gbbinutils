pub use crate::syntax::Token;

use self::backend::*;
use self::macros::{MacroDefData, MacroEntry, MacroTableEntry};

use crate::analysis::session::*;
use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::diag::*;
use crate::span::BufContext;
use crate::syntax::lexer::{LexError, Lexer};
use crate::syntax::*;

use std::rc::Rc;

#[cfg(test)]
pub use self::mock::*;

pub mod backend;
mod macros;
mod semantics;
mod session;

pub(crate) trait Assemble<D>
where
    D: Diagnostics,
    Self: Backend<
        Ident<String>,
        D::Span,
        HashMapNameTable<MacroTableEntry<D::MacroDefId, Rc<MacroDefData<String>>>>,
    >,
{
    fn assemble<C: Codebase>(
        &mut self,
        name: &str,
        codebase: &C,
        diagnostics: &mut D,
    ) -> Result<(), CodebaseError> {
        let mut file_parser = CodebaseAnalyzer::new(codebase);
        let mut analyzer = SemanticAnalyzer;
        let mut names = HashMapNameTable::new();
        let mut session = CompositeSession::new(
            &mut file_parser,
            &mut analyzer,
            self,
            &mut names,
            diagnostics,
        );
        session.analyze_file(name.into())
    }
}

impl<B, D> Assemble<D> for B
where
    D: Diagnostics,
    B: Backend<
        Ident<String>,
        D::Span,
        HashMapNameTable<MacroTableEntry<D::MacroDefId, Rc<MacroDefData<String>>>>,
    >,
{
}

pub struct PartialSession<'a, C: 'a, B: ?Sized + 'a, N: 'a, D: 'a> {
    codebase: &'a mut C,
    backend: &'a mut B,
    names: &'a mut N,
    diagnostics: &'a mut D,
}

type LexItem<T, S> = (Result<SemanticToken<T>, LexError>, S);

pub(crate) type SemanticToken<T> = Token<Ident<T>, Literal<T>, Command>;

#[derive(Clone, Debug, PartialEq)]
pub struct Ident<T> {
    pub name: T,
    visibility: Visibility,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Visibility {
    Global,
}

#[cfg(test)]
impl<T> From<T> for Ident<T> {
    fn from(name: T) -> Ident<T> {
        Ident {
            name,
            visibility: Visibility::Global,
        }
    }
}

#[cfg(test)]
impl From<&str> for Ident<String> {
    fn from(name: &str) -> Ident<String> {
        Ident {
            name: name.into(),
            visibility: Visibility::Global,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Literal<S> {
    Operand(Operand),
    Number(i32),
    String(S),
}

pub(crate) trait Lex<D: Diagnostics> {
    type StringRef: Clone + Eq;
    type TokenIter: Iterator<Item = LexItem<Self::StringRef, D::Span>>;

    fn lex_file(
        &mut self,
        path: Self::StringRef,
        diagnostics: &mut D,
    ) -> Result<Self::TokenIter, CodebaseError>;
}

pub(crate) trait Analyze<R: Clone + Eq, D: Diagnostics> {
    fn analyze_token_seq<I, C, B, N>(
        &mut self,
        tokens: I,
        partial: &mut PartialSession<C, B, N, D>,
    ) where
        I: IntoIterator<Item = LexItem<R, D::Span>>,
        C: Lex<D, StringRef = R>,
        B: Backend<Ident<R>, D::Span, N> + ?Sized,
        N: NameTable<Ident<R>, MacroEntry = MacroEntry<R, D>>;
}

struct SemanticAnalyzer;

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

impl<R: Clone + Eq, D: Diagnostics> Analyze<R, D> for SemanticAnalyzer {
    fn analyze_token_seq<I, C, B, N>(&mut self, tokens: I, partial: &mut PartialSession<C, B, N, D>)
    where
        I: IntoIterator<Item = LexItem<R, D::Span>>,
        C: Lex<D, StringRef = R>,
        B: Backend<Ident<R>, D::Span, N> + ?Sized,
        N: NameTable<Ident<R>, MacroEntry = MacroEntry<R, D>>,
    {
        let session = CompositeSession::new(
            partial.codebase,
            self,
            partial.backend,
            partial.names,
            partial.diagnostics,
        );
        let actions = semantics::SemanticActions::new(session);
        crate::syntax::parse_token_seq(tokens.into_iter(), actions);
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
            tokens: crate::syntax::tokenize(src, |spelling| Ident {
                name: spelling.to_string(),
                visibility: Visibility::Global,
            }),
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

    use std::cell::RefCell;
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
            Ok(self.files.get(&path).unwrap().clone().into_iter())
        }
    }

    pub struct MockAnalyzer<'a, T> {
        log: &'a RefCell<Vec<T>>,
    }

    impl<'a, T> MockAnalyzer<'a, T> {
        pub fn new(log: &'a RefCell<Vec<T>>) -> Self {
            Self { log }
        }
    }

    impl<'a, T, D> Analyze<String, D> for MockAnalyzer<'a, T>
    where
        T: From<FrontendEvent<D::Span>>,
        D: Diagnostics,
    {
        fn analyze_token_seq<I, C, B, N>(
            &mut self,
            tokens: I,
            _downstream: &mut PartialSession<C, B, N, D>,
        ) where
            I: IntoIterator<Item = LexItem<String, D::Span>>,
            C: Lex<D, StringRef = String>,
            B: Backend<Ident<String>, D::Span, N> + ?Sized,
            N: NameTable<Ident<String>, MacroEntry = MacroEntry<String, D>>,
        {
            self.log
                .borrow_mut()
                .push(FrontendEvent::AnalyzeTokenSeq(tokens.into_iter().collect()).into())
        }
    }

    #[derive(Debug, PartialEq)]
    pub(crate) enum FrontendEvent<S> {
        AnalyzeTokenSeq(Vec<LexItem<String, S>>),
    }
}
