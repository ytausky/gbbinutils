mod macros;
mod semantics;
mod session;
mod syntax;

use self::macros::{Expand, MacroExpander, MacroTable};
use crate::backend::*;
use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::diag::*;
use crate::frontend::session::*;
use crate::frontend::syntax::lexer::LexError;
use crate::frontend::syntax::*;
use crate::span::BufContext;
use std::rc::Rc;

pub use crate::frontend::syntax::Token;

pub(crate) fn analyze_file<C, B, D>(
    name: String,
    codebase: &C,
    mut backend: B,
    diagnostics: &mut D,
) -> Result<B::Object, CodebaseError>
where
    C: Codebase,
    B: Backend<String, D::Span>,
    D: Diagnostics,
{
    let mut file_parser =
        CodebaseAnalyzer::new(MacroExpander::new(), codebase, SemanticAnalysis {});
    let mut session = Session::new(&mut file_parser, &mut backend, diagnostics);
    session.analyze_file(name)?;
    Ok(backend.into_object())
}

pub struct Downstream<'a, B: 'a, D: 'a> {
    backend: &'a mut B,
    diagnostics: &'a mut D,
}

type LexItem<T, S> = (Result<SemanticToken<T>, LexError>, S);

pub type SemanticToken<T> = Token<T, Literal<T>>;

pub type Ident<T> = T;

#[derive(Clone, Debug, PartialEq)]
pub enum Literal<S> {
    Operand(Operand),
    Number(i32),
    String(S),
}

trait Analysis<Id>
where
    Self: Copy,
    Id: Into<String> + Clone + AsRef<str> + PartialEq,
{
    fn run<I, F, B, D>(&self, tokens: I, session: Session<F, B, D>)
    where
        I: Iterator<Item = LexItem<Id, D::Span>>,
        F: Frontend<D, StringRef = Id>,
        B: Backend<Id, D::Span>,
        D: Diagnostics;
}

#[derive(Clone, Copy)]
struct SemanticAnalysis;

pub(crate) trait Frontend<D: Diagnostics> {
    type StringRef: AsRef<str> + Clone + Into<String> + PartialEq;
    type MacroDefId: Clone;

    fn analyze_file<B>(
        &mut self,
        path: Self::StringRef,
        downstream: Downstream<B, D>,
    ) -> Result<(), CodebaseError>
    where
        B: Backend<Ident<Self::StringRef>, D::Span>;

    fn invoke_macro<B>(
        &mut self,
        name: (Ident<Self::StringRef>, D::Span),
        args: MacroArgs<Ident<Self::StringRef>, D::Span>,
        downstream: Downstream<B, D>,
    ) where
        B: Backend<Ident<Self::StringRef>, D::Span>;

    fn define_macro(
        &mut self,
        name: (impl Into<Ident<Self::StringRef>>, D::Span),
        params: Vec<(Ident<Self::StringRef>, D::Span)>,
        tokens: Vec<(SemanticToken<Self::StringRef>, D::Span)>,
        diagnostics: &mut D,
    );
}

impl<Id> Analysis<Id> for SemanticAnalysis
where
    Id: Into<String> + Clone + AsRef<str> + PartialEq,
{
    fn run<'a, I, F, B, D>(&self, tokens: I, session: Session<'a, F, B, D>)
    where
        I: Iterator<Item = LexItem<Id, D::Span>>,
        F: Frontend<D, StringRef = Id>,
        B: Backend<Id, D::Span>,
        D: Diagnostics,
    {
        let actions = semantics::SemanticActions::new(session);
        syntax::parse_token_seq(tokens, actions);
    }
}

struct CodebaseAnalyzer<'a, M, T: 'a, A> {
    macro_table: M,
    codebase: &'a T,
    analysis: A,
}

impl<'a, M, T: 'a, A> CodebaseAnalyzer<'a, M, T, A>
where
    M: MacroTable<T::StringRef>,
    T: StringRef,
    A: Analysis<T::StringRef>,
{
    fn new(macro_table: M, codebase: &T, analysis: A) -> CodebaseAnalyzer<M, T, A> {
        CodebaseAnalyzer {
            macro_table,
            codebase,
            analysis,
        }
    }

    fn analyze_token_seq<I, F>(
        &mut self,
        tokens: I,
        downstream: &mut Downstream<impl Backend<Ident<T::StringRef>, F::Span>, F>,
    ) where
        I: IntoIterator<Item = LexItem<T::StringRef, F::Span>>,
        F: Diagnostics<MacroDefId = M::MacroDefId>,
        T: Tokenize<F::BufContext>,
        M::Entry: Expand<T::StringRef, F, F::Span>,
        for<'b> &'b T::Tokenized: IntoIterator<Item = LexItem<T::StringRef, F::Span>>,
    {
        let analysis = self.analysis;
        let session = Session::new(self, downstream.backend, downstream.diagnostics);
        analysis.run(tokens.into_iter(), session)
    }
}

type TokenSeq<I, S> = Vec<(SemanticToken<I>, S)>;

impl<'a, M, T, A, D> Frontend<D> for CodebaseAnalyzer<'a, M, T, A>
where
    M: MacroTable<T::StringRef, MacroDefId = D::MacroDefId>,
    M::Entry: Expand<T::StringRef, D, D::Span>,
    T: Tokenize<D::BufContext> + 'a,
    A: Analysis<T::StringRef>,
    D: Diagnostics,
    for<'b> &'b T::Tokenized: IntoIterator<Item = LexItem<T::StringRef, D::Span>>,
{
    type StringRef = T::StringRef;
    type MacroDefId = D::MacroDefId;

    fn analyze_file<B>(
        &mut self,
        path: Self::StringRef,
        mut downstream: Downstream<B, D>,
    ) -> Result<(), CodebaseError>
    where
        B: Backend<Ident<Self::StringRef>, D::Span>,
    {
        let tokenized_src = {
            self.codebase.tokenize_file(path.as_ref(), |buf_id| {
                downstream.diagnostics.mk_buf_context(buf_id, None)
            })?
        };
        self.analyze_token_seq(&tokenized_src, &mut downstream);
        Ok(())
    }

    fn invoke_macro<B>(
        &mut self,
        (name, span): (Ident<Self::StringRef>, D::Span),
        args: MacroArgs<Self::StringRef, D::Span>,
        mut downstream: Downstream<B, D>,
    ) where
        B: Backend<Ident<Self::StringRef>, D::Span>,
    {
        let expansion = match self.macro_table.get(&name) {
            Some(entry) => Some(entry.expand(span, args, downstream.diagnostics)),
            None => {
                downstream
                    .diagnostics
                    .emit_diagnostic(CompactDiagnostic::new(
                        Message::UndefinedMacro { name: name.into() },
                        span,
                    ));
                None
            }
        };
        if let Some(expansion) = expansion {
            self.analyze_token_seq(expansion.map(|(t, s)| (Ok(t), s)), &mut downstream)
        }
    }

    fn define_macro(
        &mut self,
        name: (impl Into<Ident<Self::StringRef>>, D::Span),
        params: Vec<(Ident<Self::StringRef>, D::Span)>,
        body: Vec<(SemanticToken<Self::StringRef>, D::Span)>,
        diagnostics: &mut D,
    ) {
        self.macro_table.define(name, params, body, diagnostics);
    }
}

trait StringRef {
    type StringRef: AsRef<str> + Clone + Into<String> + PartialEq;
}

trait Tokenize<C: BufContext>
where
    Self: StringRef,
    for<'c> &'c Self::Tokenized: IntoIterator<Item = LexItem<Self::StringRef, C::Span>>,
{
    type Tokenized;
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
    src: Rc<str>,
    context: C,
}

impl<C: BufContext> TokenizedSrc<C> {
    fn new(src: Rc<str>, context: C) -> TokenizedSrc<C> {
        TokenizedSrc { src, context }
    }
}

impl<'a, C: BufContext> IntoIterator for &'a TokenizedSrc<C> {
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = TokenizedSrcIter<'a, C>;
    fn into_iter(self) -> Self::IntoIter {
        TokenizedSrcIter {
            tokens: syntax::tokenize(&self.src),
            context: &self.context,
        }
    }
}

struct TokenizedSrcIter<'a, C: BufContext + 'a> {
    tokens: syntax::lexer::Lexer<'a>,
    context: &'a C,
}

impl<'a, C: BufContext> Iterator for TokenizedSrcIter<'a, C> {
    type Item = LexItem<String, C::Span>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tokens
            .next()
            .map(|(t, r)| (t, self.context.mk_span(r)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebase;
    use crate::frontend::syntax::keyword::Mnemonic;
    use crate::instruction::{Instruction, Nullary};
    use crate::span::*;
    use std::collections::HashMap;
    use std::{self, cell::RefCell};

    #[test]
    fn include_source_file() {
        let filename = "my_file.asm";
        let contents = vec![Ok(Token::Command(Command::Mnemonic(Mnemonic::Nop)))];
        let log = TestLog::default();
        TestFixture::new(&log)
            .given(|f| {
                f.mock_token_source
                    .add_file(filename, add_code_refs(&contents))
            })
            .when(|mut fixture| {
                fixture
                    .session()
                    .analyze_file(filename.to_string())
                    .unwrap()
            });
        assert_eq!(*log.borrow(), [TestEvent::AnalyzeTokens(contents)]);
    }

    fn add_code_refs<'a, T, I>(tokens: I) -> Vec<(T, ())>
    where
        T: Clone + 'a,
        I: IntoIterator<Item = &'a T>,
    {
        tokens.into_iter().map(|t| (t.clone(), ())).collect()
    }

    #[test]
    fn emit_instruction_item() {
        let item = Item::Instruction(Instruction::Nullary(Nullary::Nop));
        let log = TestLog::default();
        TestFixture::new(&log)
            .when(|mut fixture| fixture.session().backend.emit_item(item.clone()));
        assert_eq!(*log.borrow(), [TestEvent::EmitItem(item)]);
    }

    #[test]
    fn define_label() {
        let label = "label";
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut fixture| {
            fixture
                .session()
                .backend
                .define_symbol((label.to_string(), ()), RelocAtom::LocationCounter.into())
        });
        assert_eq!(
            *log.borrow(),
            [TestEvent::DefineSymbol(
                String::from(label),
                RelocAtom::LocationCounter.into()
            )]
        );
    }

    use crate::frontend::syntax::keyword::*;

    #[test]
    fn define_and_invoke_macro() {
        let name = "my_macro";
        let tokens = vec![Token::Command(Command::Mnemonic(Mnemonic::Nop))];
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut fixture| {
            let mut session = fixture.session();
            session.define_macro((name.to_string(), ()), Vec::new(), add_code_refs(&tokens));
            session.invoke_macro((name.to_string(), ()), vec![])
        });
        assert_eq!(
            *log.borrow(),
            [TestEvent::AnalyzeTokens(
                tokens.into_iter().map(Ok).collect()
            )]
        );
    }

    #[test]
    fn define_and_invoke_macro_with_param() {
        let db = Token::Command(Command::Directive(Directive::Db));
        let arg = Token::Literal(Literal::Number(0x42));
        let literal0 = Token::Literal(Literal::Number(0));
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut fixture| {
            let name = "my_db";
            let param = "x";
            let mut session = fixture.session();
            session.define_macro(
                (name.to_string(), ()),
                vec![(param.to_string(), ())],
                vec![
                    (db.clone(), ()),
                    (Token::Ident(param.to_string()), ()),
                    (literal0.clone(), ()),
                ],
            );
            session.invoke_macro((name.to_string(), ()), vec![vec![(arg.clone(), ())]])
        });
        assert_eq!(
            *log.borrow(),
            [TestEvent::AnalyzeTokens(vec![
                Ok(db),
                Ok(arg),
                Ok(literal0)
            ])]
        );
    }

    #[test]
    fn define_and_invoke_macro_with_label() {
        let nop = Token::Command(Command::Mnemonic(Mnemonic::Nop));
        let label = String::from("label");
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut fixture| {
            let name = String::from("my_macro");
            let param = String::from("x");
            let mut session = fixture.session();
            session.define_macro(
                (name.clone(), ()),
                vec![(param.clone(), ())],
                vec![(Token::Label(param), ()), (nop.clone(), ())],
            );
            session.invoke_macro((name, ()), vec![vec![(Token::Ident(label.clone()), ())]])
        });
        assert_eq!(
            *log.borrow(),
            [TestEvent::AnalyzeTokens(vec![
                Ok(Token::Label(label)),
                Ok(nop)
            ])]
        );
    }

    use crate::diag::CompactDiagnostic;

    #[test]
    fn diagnose_undefined_macro() {
        let name = "my_macro".to_string();
        let log = TestLog::default();
        TestFixture::new(&log)
            .when(|mut fixture| fixture.session().invoke_macro((name.clone(), ()), vec![]));
        assert_eq!(
            *log.borrow(),
            [TestEvent::Diagnostic(CompactDiagnostic::new(
                Message::UndefinedMacro { name },
                ()
            ))]
        );
    }

    struct MockTokenSource {
        files: HashMap<String, Vec<LexItem<String, ()>>>,
    }

    impl MockTokenSource {
        fn new() -> MockTokenSource {
            MockTokenSource {
                files: HashMap::new(),
            }
        }

        fn add_file(&mut self, name: &str, tokens: Vec<LexItem<String, ()>>) {
            self.files.insert(name.to_string(), tokens);
        }
    }

    impl<'a> StringRef for MockTokenSource {
        type StringRef = String;
    }

    impl<'a> Tokenize<Mock<'a>> for MockTokenSource {
        type Tokenized = MockTokenized;

        fn tokenize_file<F: FnOnce(BufId) -> Mock<'a>>(
            &self,
            filename: &str,
            _: F,
        ) -> Result<Self::Tokenized, CodebaseError> {
            Ok(MockTokenized(self.files.get(filename).unwrap().clone()))
        }
    }

    struct MockTokenized(Vec<LexItem<String, ()>>);

    impl<'b> IntoIterator for &'b MockTokenized {
        type Item = LexItem<String, ()>;
        type IntoIter = std::iter::Cloned<std::slice::Iter<'b, LexItem<String, ()>>>;

        fn into_iter(self) -> Self::IntoIter {
            (&self.0).into_iter().cloned()
        }
    }

    #[derive(Clone, Copy)]
    struct Mock<'a> {
        log: &'a TestLog,
    }

    impl<'a> Mock<'a> {
        fn new(log: &'a TestLog) -> Mock<'a> {
            Mock { log }
        }
    }

    impl<'a> Span for Mock<'a> {
        type Span = ();
    }

    impl<'a> MacroContextFactory<()> for Mock<'a> {
        type MacroDefId = ();
        type MacroExpansionContext = Mock<'a>;

        fn add_macro_def<P, B>(&mut self, _: (), _: P, _: B) -> Self::MacroDefId
        where
            P: IntoIterator<Item = ()>,
            B: IntoIterator<Item = ()>,
        {
        }

        fn mk_macro_expansion_context<I, J>(
            &mut self,
            _: (),
            _: I,
            _: &Self::MacroDefId,
        ) -> Self::MacroExpansionContext
        where
            I: IntoIterator<Item = J>,
            J: IntoIterator<Item = ()>,
        {
            self.clone()
        }
    }

    impl<'a> StripSpan<()> for Mock<'a> {
        type Stripped = ();

        fn strip_span(&mut self, _: &()) {}
    }

    impl<'a> ContextFactory for Mock<'a> {
        type BufContext = Mock<'a>;

        fn mk_buf_context(&mut self, _: BufId, _: Option<Self::Span>) -> Self::BufContext {
            self.clone()
        }
    }

    impl<'a> BufContext for Mock<'a> {
        type Span = ();
        fn mk_span(&self, _: codebase::BufRange) -> Self::Span {}
    }

    impl<'a> MacroExpansionContext for Mock<'a> {
        type Span = ();
        fn mk_span(&self, _: usize, _: Option<TokenExpansion>) -> Self::Span {}
    }

    impl<'a> Analysis<String> for Mock<'a> {
        fn run<I, F, B, D>(&self, tokens: I, _frontend: Session<F, B, D>)
        where
            I: Iterator<Item = LexItem<String, D::Span>>,
            F: Frontend<D, StringRef = String>,
            B: Backend<Ident<F::StringRef>, D::Span>,
            D: Diagnostics,
        {
            self.log.borrow_mut().push(TestEvent::AnalyzeTokens(
                tokens.map(|(t, _)| t.into()).collect(),
            ))
        }
    }

    impl<'a, 'b> BuildValue<'b, String, ()> for Mock<'a> {
        type Builder = IndependentValueBuilder<()>;

        fn build_value(&'b mut self) -> Self::Builder {
            IndependentValueBuilder::new()
        }
    }

    impl<'a> HasValue<()> for Mock<'a> {
        type Value = RelocExpr<String, ()>;
    }

    impl<'a> Backend<String, ()> for Mock<'a> {
        type Object = ();

        fn define_symbol(&mut self, symbol: (String, ()), value: RelocExpr<String, ()>) {
            self.log
                .borrow_mut()
                .push(TestEvent::DefineSymbol(symbol.0, value))
        }

        fn emit_item(&mut self, item: Item<RelocExpr<String, ()>>) {
            self.log.borrow_mut().push(TestEvent::EmitItem(item))
        }

        fn into_object(self) {}

        fn set_origin(&mut self, _origin: RelocExpr<String, ()>) {
            unimplemented!()
        }
    }

    impl<'a> EmitDiagnostic<(), ()> for Mock<'a> {
        fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<(), ()>) {
            self.log
                .borrow_mut()
                .push(TestEvent::Diagnostic(diagnostic))
        }
    }

    impl<'a> MergeSpans<()> for Mock<'a> {
        fn merge_spans(&mut self, _: &(), _: &()) {}
    }

    impl<'a> Diagnostics for Mock<'a> {}

    type TestLog = RefCell<Vec<TestEvent>>;

    #[derive(Debug, PartialEq)]
    enum TestEvent {
        AnalyzeTokens(Vec<Result<SemanticToken<String>, syntax::lexer::LexError>>),
        DefineSymbol(String, RelocExpr<String, ()>),
        Diagnostic(CompactDiagnostic<(), ()>),
        EmitItem(Item<RelocExpr<String, ()>>),
    }

    struct TestFixture<'a> {
        macro_table: MacroExpander<String, ()>,
        mock_token_source: MockTokenSource,
        analysis: Mock<'a>,
        object: Mock<'a>,
        diagnostics: Mock<'a>,
    }

    struct PreparedFixture<'a, 'b> {
        code_analyzer: CodebaseAnalyzer<'b, MacroExpander<String, ()>, MockTokenSource, Mock<'a>>,
        object: Mock<'a>,
        diagnostics: Mock<'a>,
    }

    impl<'a, 'b> PreparedFixture<'a, 'b> {
        fn session<'r>(&'r mut self) -> TestSession<'a, 'b, 'r> {
            Session::new(
                &mut self.code_analyzer,
                &mut self.object,
                &mut self.diagnostics,
            )
        }
    }

    impl<'a> TestFixture<'a> {
        fn new(log: &'a TestLog) -> TestFixture<'a> {
            TestFixture {
                macro_table: MacroExpander::new(),
                mock_token_source: MockTokenSource::new(),
                analysis: Mock::new(log),
                object: Mock::new(log),
                diagnostics: Mock::new(log),
            }
        }

        fn given<F: FnOnce(&mut Self)>(mut self, f: F) -> Self {
            f(&mut self);
            self
        }

        fn when<F: for<'b> FnOnce(PreparedFixture<'a, 'b>)>(self, f: F) {
            let prepared = PreparedFixture {
                code_analyzer: CodebaseAnalyzer::new(
                    self.macro_table,
                    &self.mock_token_source,
                    self.analysis,
                ),
                object: self.object,
                diagnostics: self.diagnostics,
            };
            f(prepared);
        }
    }

    type TestCodebaseAnalyzer<'a, 'r> =
        CodebaseAnalyzer<'r, MacroExpander<String, ()>, MockTokenSource, Mock<'a>>;
    type TestSession<'a, 'b, 'r> = Session<'r, TestCodebaseAnalyzer<'a, 'b>, Mock<'a>, Mock<'a>>;
}
