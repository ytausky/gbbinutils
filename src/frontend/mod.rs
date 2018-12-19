mod macros;
mod semantics;
mod session;
mod syntax;

use self::macros::{Expand, MacroExpander, MacroTable};
use crate::backend::*;
use crate::codebase::{BufId, Codebase, CodebaseError};
use crate::diagnostics::*;
use crate::frontend::session::*;
use crate::frontend::syntax::*;
use crate::span::BufContext;
use std::rc::Rc;

pub use crate::frontend::syntax::Token;

pub fn analyze_file<C, B, D>(
    name: String,
    codebase: &C,
    mut backend: B,
    diagnostics: &mut D,
) -> Result<B::Object, CodebaseError>
where
    C: Codebase,
    B: Backend<D::Span>,
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

trait Analysis<Id>
where
    Self: Copy,
    Id: Into<String> + Clone + AsRef<str> + PartialEq,
{
    fn run<I, F, B, D>(&self, tokens: I, session: Session<F, B, D>)
    where
        I: Iterator<Item = (Token<Id>, D::Span)>,
        F: Frontend<D, Ident = Id>,
        B: Backend<D::Span>,
        D: Diagnostics;
}

#[derive(Clone, Copy)]
struct SemanticAnalysis;

pub trait Frontend<D: Diagnostics> {
    type Ident: AsRef<str> + Clone + Into<String> + PartialEq;
    type MacroDefId: Clone;

    fn analyze_file<B>(
        &mut self,
        path: Self::Ident,
        downstream: Downstream<B, D>,
    ) -> Result<(), CodebaseError>
    where
        B: Backend<D::Span>;

    fn invoke_macro<B>(
        &mut self,
        name: (Self::Ident, D::Span),
        args: MacroArgs<Self::Ident, D::Span>,
        downstream: Downstream<B, D>,
    ) where
        B: Backend<D::Span>;

    fn define_macro(
        &mut self,
        name: (impl Into<Self::Ident>, D::Span),
        params: Vec<(Self::Ident, D::Span)>,
        tokens: Vec<(Token<Self::Ident>, D::Span)>,
        diagnostics: &mut D,
    );
}

impl<Id> Analysis<Id> for SemanticAnalysis
where
    Id: Into<String> + Clone + AsRef<str> + PartialEq,
{
    fn run<'a, I, F, B, D>(&self, tokens: I, session: Session<'a, F, B, D>)
    where
        I: Iterator<Item = (Token<Id>, D::Span)>,
        F: Frontend<D, Ident = Id>,
        B: Backend<D::Span>,
        D: Diagnostics,
    {
        let actions = semantics::SemanticActions::new(session);
        syntax::parse_token_seq(tokens, actions)
    }
}

struct CodebaseAnalyzer<'a, M, T: 'a, A> {
    macro_table: M,
    codebase: &'a T,
    analysis: A,
}

impl<'a, M, T: 'a, A> CodebaseAnalyzer<'a, M, T, A>
where
    M: MacroTable<T::Ident>,
    T: Ident,
    A: Analysis<T::Ident>,
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
        downstream: &mut Downstream<impl Backend<F::Span>, F>,
    ) where
        I: IntoIterator<Item = (Token<T::Ident>, F::Span)>,
        F: Diagnostics<MacroDefId = M::MacroDefId>,
        T: Tokenize<F::BufContext>,
        M::Entry: Expand<T::Ident, F>,
        for<'b> &'b T::Tokenized: IntoIterator<Item = (Token<T::Ident>, F::Span)>,
    {
        let analysis = self.analysis;
        let session = Session::new(self, downstream.backend, downstream.diagnostics);
        analysis.run(tokens.into_iter(), session)
    }
}

type TokenSeq<I, S> = Vec<(Token<I>, S)>;

impl<'a, M, T, A, D> Frontend<D> for CodebaseAnalyzer<'a, M, T, A>
where
    M: MacroTable<T::Ident, MacroDefId = D::MacroDefId>,
    M::Entry: Expand<T::Ident, D>,
    T: Tokenize<D::BufContext> + 'a,
    A: Analysis<T::Ident>,
    D: Diagnostics,
    for<'b> &'b T::Tokenized: IntoIterator<Item = (Token<T::Ident>, D::Span)>,
{
    type Ident = T::Ident;
    type MacroDefId = D::MacroDefId;

    fn analyze_file<B>(
        &mut self,
        path: Self::Ident,
        mut downstream: Downstream<B, D>,
    ) -> Result<(), CodebaseError>
    where
        B: Backend<D::Span>,
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
        (name, span): (Self::Ident, D::Span),
        args: MacroArgs<Self::Ident, D::Span>,
        mut downstream: Downstream<B, D>,
    ) where
        B: Backend<D::Span>,
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
            self.analyze_token_seq(expansion, &mut downstream)
        }
    }

    fn define_macro(
        &mut self,
        name: (impl Into<Self::Ident>, D::Span),
        params: Vec<(Self::Ident, D::Span)>,
        body: Vec<(Token<Self::Ident>, D::Span)>,
        diagnostics: &mut D,
    ) {
        self.macro_table.define(name, params, body, diagnostics);
    }
}

trait Ident {
    type Ident: AsRef<str> + Clone + Into<String> + PartialEq;
}

trait Tokenize<C: BufContext>
where
    Self: Ident,
    for<'c> &'c Self::Tokenized: IntoIterator<Item = (Token<Self::Ident>, C::Span)>,
{
    type Tokenized;
    fn tokenize_file<F: FnOnce(BufId) -> C>(
        &self,
        filename: &str,
        mk_context: F,
    ) -> Result<Self::Tokenized, CodebaseError>;
}

impl<C: Codebase> Ident for C {
    type Ident = String;
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
    type Item = (Token<String>, C::Span);
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
        let contents = vec![Token::Command(Command::Mnemonic(Mnemonic::Nop))];
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

    fn add_code_refs<'a, I: IntoIterator<Item = &'a Token<String>>>(
        tokens: I,
    ) -> Vec<(Token<String>, ())> {
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
        assert_eq!(*log.borrow(), [TestEvent::AnalyzeTokens(tokens)]);
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
            [TestEvent::AnalyzeTokens(vec![db, arg, literal0])]
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
            [TestEvent::AnalyzeTokens(vec![Token::Label(label), nop])]
        );
    }

    use crate::diagnostics::CompactDiagnostic;

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
        files: HashMap<String, Vec<(Token<String>, ())>>,
    }

    impl MockTokenSource {
        fn new() -> MockTokenSource {
            MockTokenSource {
                files: HashMap::new(),
            }
        }

        fn add_file(&mut self, name: &str, tokens: Vec<(Token<String>, ())>) {
            self.files.insert(name.to_string(), tokens);
        }
    }

    impl<'a> Ident for MockTokenSource {
        type Ident = String;
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

    struct MockTokenized(Vec<(Token<String>, ())>);

    impl<'b> IntoIterator for &'b MockTokenized {
        type Item = (Token<String>, ());
        type IntoIter = std::iter::Cloned<std::slice::Iter<'b, (Token<String>, ())>>;
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

    impl<'a> StrippedSpan for Mock<'a> {
        type StrippedSpan = ();
    }

    impl<'a> MacroContextFactory for Mock<'a> {
        type MacroDefId = ();
        type MacroExpansionContext = Mock<'a>;

        fn add_macro_def<P, B>(&mut self, _: Self::Span, _: P, _: B) -> Self::MacroDefId
        where
            P: IntoIterator<Item = Self::Span>,
            B: IntoIterator<Item = Self::Span>,
        {
        }

        fn mk_macro_expansion_context<I, J>(
            &mut self,
            _: Self::Span,
            _: I,
            _: &Self::MacroDefId,
        ) -> Self::MacroExpansionContext
        where
            I: IntoIterator<Item = J>,
            J: IntoIterator<Item = Self::Span>,
        {
            self.clone()
        }
    }

    impl<'a> StripSpan for Mock<'a> {
        fn strip_span(&mut self, _: &Self::Span) {}
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
            I: Iterator<Item = (Token<String>, D::Span)>,
            F: Frontend<D, Ident = String>,
            B: Backend<D::Span>,
            D: Diagnostics,
        {
            self.log.borrow_mut().push(TestEvent::AnalyzeTokens(
                tokens.map(|(t, _)| t.into()).collect(),
            ))
        }
    }

    impl<'a, 'b> BuildValue<'b, RelocExpr<()>> for Mock<'a> {
        type Builder = RelocExprBuilder<()>;

        fn build_value(&'b mut self) -> Self::Builder {
            RelocExprBuilder::new()
        }
    }

    impl<'a> HasValue for Mock<'a> {
        type Value = RelocExpr<()>;
    }

    impl<'a> Backend<()> for Mock<'a> {
        type Object = ();

        fn define_symbol(&mut self, symbol: (impl Into<String>, ()), value: RelocExpr<()>) {
            self.log
                .borrow_mut()
                .push(TestEvent::DefineSymbol(symbol.0.into(), value))
        }

        fn emit_item(&mut self, item: Item<RelocExpr<()>>) {
            self.log.borrow_mut().push(TestEvent::EmitItem(item))
        }

        fn into_object(self) {}

        fn set_origin(&mut self, _origin: RelocExpr<()>) {
            unimplemented!()
        }
    }

    impl<'a> EmitDiagnostic for Mock<'a> {
        fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<(), ()>) {
            self.log
                .borrow_mut()
                .push(TestEvent::Diagnostic(diagnostic))
        }
    }

    impl<'a> MergeSpans for Mock<'a> {
        fn merge_spans(&mut self, _: &(), _: &()) {}
    }

    impl<'a> Diagnostics for Mock<'a> {}

    type TestLog = RefCell<Vec<TestEvent>>;

    #[derive(Debug, PartialEq)]
    enum TestEvent {
        AnalyzeTokens(Vec<Token<String>>),
        DefineSymbol(String, RelocExpr<()>),
        Diagnostic(CompactDiagnostic<(), ()>),
        EmitItem(Item<RelocExpr<()>>),
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
