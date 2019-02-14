mod macros;
mod semantics;
mod session;

use self::macros::{MacroDefData, MacroEntry, MacroTableEntry};
use crate::analysis::session::*;
use crate::backend::*;
use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::diag::*;
use crate::span::BufContext;
use crate::syntax::lexer::{LexError, Lexer};
use crate::syntax::*;
use std::rc::Rc;

pub use crate::syntax::Token;

#[cfg(test)]
pub use self::mock::*;

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
        let mut file_parser = CodebaseAnalyzer::new(codebase, SemanticAnalysis);
        let mut names = HashMapNameTable::new();
        let mut session = CompositeSession::new(&mut file_parser, self, &mut names, diagnostics);
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

pub struct Downstream<'a, B: ?Sized + 'a, N: 'a, D: 'a> {
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

trait Analysis<Id>
where
    Self: Clone,
    Id: Into<String> + Clone + Eq + AsRef<str>,
{
    fn run<I, F, B, N, D>(&self, tokens: I, session: CompositeSession<F, B, N, D>)
    where
        I: Iterator<Item = LexItem<Id, D::Span>>,
        F: Frontend<D, StringRef = Id>,
        B: Backend<Ident<Id>, D::Span, N> + ?Sized,
        N: NameTable<Ident<F::StringRef>, MacroEntry = MacroEntry<F, D>>,
        D: Diagnostics;
}

#[derive(Clone, Copy)]
struct SemanticAnalysis;

pub(crate) trait Frontend<D: Diagnostics> {
    type StringRef: AsRef<str> + Clone + Eq + Into<String>;
    type TokenIter: Iterator<Item = LexItem<Self::StringRef, D::Span>>;

    fn lex_file(
        &mut self,
        path: Self::StringRef,
        diagnostics: &mut D,
    ) -> Result<Self::TokenIter, CodebaseError>;

    fn analyze_token_seq<I, B, N>(&mut self, tokens: I, downstream: &mut Downstream<B, N, D>)
    where
        I: IntoIterator<Item = LexItem<Self::StringRef, D::Span>>,
        B: Backend<Ident<Self::StringRef>, D::Span, N> + ?Sized,
        N: NameTable<Ident<Self::StringRef>, MacroEntry = MacroEntry<Self, D>>;
}

impl<Id> Analysis<Id> for SemanticAnalysis
where
    Id: Into<String> + Clone + Eq + AsRef<str>,
{
    fn run<'a, I, F, B, N, D>(&self, tokens: I, session: CompositeSession<'a, F, B, N, D>)
    where
        I: Iterator<Item = LexItem<Id, D::Span>>,
        F: Frontend<D, StringRef = Id>,
        B: Backend<Ident<Id>, D::Span, N> + ?Sized,
        N: NameTable<Ident<F::StringRef>, MacroEntry = MacroEntry<F, D>>,
        D: Diagnostics,
    {
        let actions = semantics::SemanticActions::new(session);
        crate::syntax::parse_token_seq(tokens, actions);
    }
}

struct CodebaseAnalyzer<'a, T: 'a, A> {
    codebase: &'a T,
    analysis: A,
}

impl<'a, T: 'a, A> CodebaseAnalyzer<'a, T, A>
where
    T: StringRef,
    A: Analysis<T::StringRef>,
{
    fn new(codebase: &T, analysis: A) -> CodebaseAnalyzer<T, A> {
        CodebaseAnalyzer { codebase, analysis }
    }
}

type TokenSeq<I, S> = Vec<(SemanticToken<I>, S)>;

impl<'a, T, A, D> Frontend<D> for CodebaseAnalyzer<'a, T, A>
where
    T: Tokenize<D::BufContext> + 'a,
    A: Analysis<T::StringRef>,
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

    fn analyze_token_seq<I, B, N>(&mut self, tokens: I, downstream: &mut Downstream<B, N, D>)
    where
        I: IntoIterator<Item = LexItem<Self::StringRef, D::Span>>,
        B: Backend<Ident<Self::StringRef>, D::Span, N> + ?Sized,
        N: NameTable<Ident<Self::StringRef>, MacroEntry = MacroEntry<Self, D>>,
    {
        let analysis = self.analysis.clone();
        let session = CompositeSession::new(
            self,
            downstream.backend,
            downstream.names,
            downstream.diagnostics,
        );
        analysis.run(tokens.into_iter(), session)
    }
}

pub(crate) trait StringRef {
    type StringRef: AsRef<str> + Clone + Eq + Into<String>;
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

    pub struct MockFrontend<'a, T, S> {
        log: &'a RefCell<Vec<T>>,
        files: HashMap<String, Vec<LexItem<String, S>>>,
    }

    impl<'a, T, S> MockFrontend<'a, T, S> {
        pub fn new(log: &'a RefCell<Vec<T>>) -> Self {
            Self {
                log,
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

    impl<'a, T, D> Frontend<D> for MockFrontend<'a, T, D::Span>
    where
        T: From<FrontendEvent<D::Span>>,
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

        fn analyze_token_seq<I, B, N>(&mut self, tokens: I, _downstream: &mut Downstream<B, N, D>)
        where
            I: IntoIterator<Item = LexItem<Self::StringRef, D::Span>>,
            B: Backend<Ident<Self::StringRef>, D::Span, N> + ?Sized,
            N: NameTable<Ident<Self::StringRef>, MacroEntry = MacroEntry<Self, D>>,
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
