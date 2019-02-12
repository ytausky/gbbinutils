mod macros;
mod semantics;
mod session;

use self::macros::{MacroDefData, MacroEntry, MacroTableEntry};
use crate::backend::*;
use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::diag::*;
use crate::frontend::session::*;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend;
    use crate::backend::HashMapNameTable;
    use crate::diag;
    use crate::diag::MockSpan;
    use crate::frontend::macros::MacroEntry;
    use crate::instruction::{Instruction, Nullary};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::vec;

    #[test]
    fn emit_instruction_item() {
        let item = Item::Instruction(Instruction::Nullary(Nullary::Nop));
        let log = TestLog::<()>::default();
        TestFixture::new(&log).when(|mut fixture| fixture.session().emit_item(item.clone()));
        assert_eq!(*log.borrow(), [backend::Event::EmitItem(item).into()]);
    }

    #[test]
    fn define_label() {
        let label = "label";
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut fixture| {
            fixture
                .session()
                .define_symbol((label.into(), ()), RelocAtom::LocationCounter.into())
        });
        assert_eq!(
            *log.borrow(),
            [
                backend::Event::DefineSymbol((label.into(), ()), RelocAtom::LocationCounter.into())
                    .into()
            ]
        );
    }

    struct MockTokenSource<S> {
        files: HashMap<String, Vec<LexItem<String, S>>>,
    }

    impl<S> MockTokenSource<S> {
        fn new() -> MockTokenSource<S> {
            MockTokenSource {
                files: HashMap::new(),
            }
        }
    }

    impl<'a, S> StringRef for MockTokenSource<S> {
        type StringRef = String;
    }

    impl<'a, S: Clone + MockSpan> Tokenize<MockDiagnostics<'a, S>> for MockTokenSource<S> {
        type Tokenized = MockTokenized<S>;

        fn tokenize_file<F: FnOnce(BufId) -> MockDiagnostics<'a, S>>(
            &self,
            filename: &str,
            _: F,
        ) -> Result<Self::Tokenized, CodebaseError> {
            Ok(MockTokenized(
                self.files.get(filename).unwrap().clone().into_iter(),
            ))
        }
    }

    struct MockTokenized<S>(vec::IntoIter<LexItem<String, S>>);

    impl<'b, S: Clone> Iterator for MockTokenized<S> {
        type Item = LexItem<String, S>;

        fn next(&mut self) -> Option<Self::Item> {
            self.0.next()
        }
    }

    #[derive(Clone)]
    struct Mock<'a, S: Clone> {
        log: &'a TestLog<S>,
    }

    impl<'a, S: Clone> Mock<'a, S> {
        fn new(log: &'a TestLog<S>) -> Mock<'a, S> {
            Mock { log }
        }
    }

    impl<'a, S: Clone> Analysis<String> for Mock<'a, S> {
        fn run<I, F, B, N, D>(&self, tokens: I, _frontend: CompositeSession<F, B, N, D>)
        where
            I: Iterator<Item = LexItem<String, D::Span>>,
            F: Frontend<D, StringRef = String>,
            B: Backend<Ident<F::StringRef>, D::Span, N> + ?Sized,
            D: Diagnostics,
        {
            self.log.borrow_mut().push(TestEvent::AnalyzeTokens(
                tokens.map(|(t, _)| t.into()).collect(),
            ))
        }
    }

    type TestLog<S> = RefCell<Vec<TestEvent<S>>>;

    #[derive(Debug, PartialEq)]
    enum TestEvent<S: Clone> {
        Backend(backend::Event<RelocExpr<S>>),
        AnalyzeTokens(Vec<Result<SemanticToken<String>, LexError>>),
        Diagnostics(diag::Event<S>),
    }

    type RelocExpr<S> = backend::RelocExpr<Ident<String>, S>;

    impl<S: Clone> From<backend::Event<RelocExpr<S>>> for TestEvent<S> {
        fn from(event: backend::Event<RelocExpr<S>>) -> Self {
            TestEvent::Backend(event)
        }
    }

    impl<S: Clone> From<diag::Event<S>> for TestEvent<S> {
        fn from(event: diag::Event<S>) -> Self {
            TestEvent::Diagnostics(event)
        }
    }

    type MockBackend<'a, S> = backend::MockBackend<'a, TestEvent<S>>;
    type MockDiagnostics<'a, S> = diag::MockDiagnostics<'a, TestEvent<S>, S>;

    struct TestFixture<'a, S: Clone> {
        mock_token_source: MockTokenSource<S>,
        analysis: Mock<'a, S>,
        object: MockBackend<'a, S>,
        diagnostics: MockDiagnostics<'a, S>,
    }

    struct PreparedFixture<'a, 'b, S: Clone + MockSpan> {
        code_analyzer: TestCodebaseAnalyzer<'a, 'b, S>,
        object: MockBackend<'a, S>,
        names: TestNameTable<'a, 'b, S>,
        diagnostics: MockDiagnostics<'a, S>,
    }

    impl<'a, 'b, S: Clone + MockSpan> PreparedFixture<'a, 'b, S> {
        fn session<'r>(&'r mut self) -> TestSession<'a, 'b, 'r, S> {
            CompositeSession::new(
                &mut self.code_analyzer,
                &mut self.object,
                &mut self.names,
                &mut self.diagnostics,
            )
        }
    }

    impl<'a, S: Clone + MockSpan> TestFixture<'a, S> {
        fn new(log: &'a TestLog<S>) -> TestFixture<'a, S> {
            TestFixture {
                mock_token_source: MockTokenSource::new(),
                analysis: Mock::new(log),
                object: MockBackend::new(log),
                diagnostics: MockDiagnostics::new(log),
            }
        }

        fn when<F: for<'b> FnOnce(PreparedFixture<'a, 'b, S>)>(self, f: F) {
            let prepared = PreparedFixture {
                code_analyzer: CodebaseAnalyzer::new(&self.mock_token_source, self.analysis),
                object: self.object,
                names: HashMapNameTable::new(),
                diagnostics: self.diagnostics,
            };
            f(prepared);
        }
    }

    type TestCodebaseAnalyzer<'a, 'r, S> = CodebaseAnalyzer<'r, MockTokenSource<S>, Mock<'a, S>>;

    type TestNameTable<'a, 'b, S> =
        HashMapNameTable<MacroEntry<TestCodebaseAnalyzer<'a, 'b, S>, MockDiagnostics<'a, S>>>;

    type TestSession<'a, 'b, 'r, S> = CompositeSession<
        'r,
        TestCodebaseAnalyzer<'a, 'b, S>,
        MockBackend<'a, S>,
        TestNameTable<'a, 'b, S>,
        MockDiagnostics<'a, S>,
    >;
}
