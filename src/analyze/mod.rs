use self::macros::VecMacroTable;
use self::semantics::reentrancy::SourceComponents;
use self::semantics::resolve::*;
use self::semantics::Semantics;
use self::strings::FakeStringInterner;
use self::syntax::parser::DefaultParserFactory;
use self::syntax::*;

use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::diag::*;
use crate::object::builder::{Backend, SymbolSource};
use crate::span::{BufContext, BufContextFactory, SpanSource};
use crate::BuiltinSymbols;

use std::rc::Rc;

#[cfg(test)]
pub use self::mock::*;

mod macros;
mod semantics;
mod strings;
mod syntax;

pub(crate) trait Assemble<D>
where
    D: DiagnosticsSystem,
    D::Span: 'static,
    Self: Backend<D::Span> + BuiltinSymbols<Name = <Self as SymbolSource>::SymbolId> + Sized,
{
    fn assemble<C: Codebase>(
        self,
        name: &str,
        codebase: &C,
        diagnostics: &mut D,
    ) -> Result<(), CodebaseError> {
        let tokenizer = Tokenizer(codebase);
        let mut file_parser = CodebaseAnalyzer::new(&tokenizer);
        let mut parser_factory = DefaultParserFactory;
        let mut macros = VecMacroTable::new();
        let mut interner = FakeStringInterner;
        let mut names = BiLevelNameTable::<BasicNameTable<_, _, _>>::new();
        for (string, name) in self.builtin_symbols() {
            names.define_name(
                DefaultIdentFactory.mk_ident(string),
                ResolvedName::Symbol((*name).clone()),
            )
        }
        let session = SourceComponents::new(
            &mut file_parser,
            &mut parser_factory,
            &mut macros,
            &mut interner,
            diagnostics,
        );
        Semantics::from_components(session, &mut names, self, &mut std::iter::empty())
            .analyze_file(name.into())
    }
}

impl<B, D> Assemble<D> for B
where
    D: DiagnosticsSystem,
    D::Span: 'static,
    B: Backend<D::Span> + BuiltinSymbols<Name = <Self as SymbolSource>::SymbolId>,
{
}

type LexItem<I, R, S> = (Result<SemanticToken<I, R>, LexError>, S);
type SemanticToken<I, R> = syntax::Token<I, Literal<R>>;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Literal<R> {
    Number(i32),
    String(R),
}

trait Lex<D: SpanSource + ?Sized>: IdentSource + StringSource {
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
