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
use crate::span::{BufContext, ContextFactory, Span};
use std::fmt::Debug;
use std::rc::Rc;

pub use crate::frontend::syntax::Token;

pub fn analyze_file<
    C: Codebase,
    F: ContextFactory,
    B: Backend<F::Span>,
    D: DiagnosticsListener<F::Span>,
>(
    name: String,
    codebase: &C,
    context_factory: F,
    backend: B,
    diagnostics: &mut D,
) -> Result<B::Object, CodebaseError> {
    let file_parser = CodebaseAnalyzer::new(
        MacroExpander::new(),
        codebase,
        context_factory,
        SemanticAnalysis {},
    );
    let mut session: Components<_, _, D, _, _, _> =
        Components::new(file_parser, backend, diagnostics);
    session.analyze_file(name)?;
    Ok(session.build_object())
}

pub struct Downstream<'a, B: 'a, D: 'a> {
    backend: &'a mut B,
    diagnostics: &'a mut D,
}

trait Analysis<Id: Into<String> + Debug + PartialEq>
where
    Self: Copy,
{
    fn run<I, S>(&self, tokens: I, session: &mut S)
    where
        I: Iterator<Item = (Token<Id>, S::Span)>,
        S: Session<Ident = Id>;
}

#[derive(Clone, Copy)]
struct SemanticAnalysis;

pub trait ExprFactory {
    fn mk_literal<S>(&mut self, literal: (i32, S)) -> RelocExpr<S>;
    fn mk_symbol<I: Into<String>, S>(&mut self, symbol: (I, S)) -> RelocExpr<S>;
}

pub struct StrExprFactory;

impl StrExprFactory {
    fn new() -> StrExprFactory {
        StrExprFactory {}
    }
}

impl ExprFactory for StrExprFactory {
    fn mk_literal<S>(&mut self, (literal, span): (i32, S)) -> RelocExpr<S> {
        RelocExpr {
            variant: literal.into(),
            span,
        }
    }

    fn mk_symbol<I: Into<String>, S>(&mut self, (symbol, span): (I, S)) -> RelocExpr<S> {
        RelocExpr {
            variant: RelocAtom::Symbol(symbol.into()).into(),
            span,
        }
    }
}

pub trait Frontend {
    type Ident: AsRef<str> + Clone + Into<String> + Debug + PartialEq;
    type Span: Span;

    fn analyze_file<B, D>(
        &mut self,
        path: Self::Ident,
        downstream: Downstream<B, D>,
    ) -> Result<(), CodebaseError>
    where
        B: Backend<Self::Span>,
        D: DiagnosticsListener<Self::Span>;

    fn invoke_macro<B, D>(
        &mut self,
        name: (Self::Ident, Self::Span),
        args: MacroArgs<Self::Ident, Self::Span>,
        downstream: Downstream<B, D>,
    ) where
        B: Backend<Self::Span>,
        D: DiagnosticsListener<Self::Span>;

    fn define_macro(
        &mut self,
        name: (impl Into<Self::Ident>, Self::Span),
        params: Vec<(Self::Ident, Self::Span)>,
        tokens: Vec<(Token<Self::Ident>, Self::Span)>,
    );
}

impl<Id: Into<String> + Debug + PartialEq> Analysis<Id> for SemanticAnalysis {
    fn run<I, S>(&self, tokens: I, session: &mut S)
    where
        I: Iterator<Item = (Token<Id>, S::Span)>,
        S: Session<Ident = Id>,
    {
        let actions = semantics::SemanticActions::new(session);
        syntax::parse_token_seq(tokens, actions)
    }
}

struct CodebaseAnalyzer<'a, M, T: 'a, F, A> {
    macro_table: M,
    codebase: &'a T,
    context_factory: F,
    analysis: A,
}

impl<'a, M, T: 'a, F, A> CodebaseAnalyzer<'a, M, T, F, A>
where
    M: MacroTable<T::Ident, F::MacroContextFactory>,
    M::Entry: Expand<T::Ident, F::MacroContextFactory>,
    T: Tokenize<F::BufContext>,
    F: ContextFactory,
    A: Analysis<T::Ident>,
    for<'b> &'b T::Tokenized: IntoIterator<Item = (Token<T::Ident>, F::Span)>,
{
    fn new(
        macro_table: M,
        codebase: &T,
        context_factory: F,
        analysis: A,
    ) -> CodebaseAnalyzer<M, T, F, A> {
        CodebaseAnalyzer {
            macro_table,
            codebase,
            context_factory,
            analysis,
        }
    }

    fn analyze_token_seq<I: IntoIterator<Item = (Token<T::Ident>, F::Span)>>(
        &mut self,
        tokens: I,
        downstream: &mut Downstream<impl Backend<F::Span>, impl DiagnosticsListener<F::Span>>,
    ) {
        let analysis = self.analysis;
        let mut session = BorrowedComponents::new(self, downstream.backend, downstream.diagnostics);
        analysis.run(tokens.into_iter(), &mut session)
    }
}

type TokenSeq<I, S> = Vec<(Token<I>, S)>;

impl<'a, M, T, F, A> Frontend for CodebaseAnalyzer<'a, M, T, F, A>
where
    M: MacroTable<T::Ident, F::MacroContextFactory>,
    M::Entry: Expand<T::Ident, F::MacroContextFactory>,
    T: Tokenize<F::BufContext> + 'a,
    F: ContextFactory,
    A: Analysis<T::Ident>,
    for<'b> &'b T::Tokenized: IntoIterator<Item = (Token<T::Ident>, F::Span)>,
{
    type Ident = T::Ident;
    type Span = F::Span;

    fn analyze_file<B, D>(
        &mut self,
        path: Self::Ident,
        mut downstream: Downstream<B, D>,
    ) -> Result<(), CodebaseError>
    where
        B: Backend<Self::Span>,
        D: DiagnosticsListener<Self::Span>,
    {
        let tokenized_src = {
            let context_factory = &mut self.context_factory;
            self.codebase.tokenize_file(path.as_ref(), |buf_id| {
                context_factory.mk_buf_context(buf_id, None)
            })?
        };
        self.analyze_token_seq(&tokenized_src, &mut downstream);
        Ok(())
    }

    fn invoke_macro<B, D>(
        &mut self,
        (name, span): (Self::Ident, Self::Span),
        args: MacroArgs<Self::Ident, Self::Span>,
        mut downstream: Downstream<B, D>,
    ) where
        B: Backend<Self::Span>,
        D: DiagnosticsListener<Self::Span>,
    {
        let expansion = match self.macro_table.get(&name) {
            Some(entry) => {
                Some(entry.expand(span, args, self.context_factory.macro_context_factory()))
            }
            None => {
                downstream
                    .diagnostics
                    .emit_diagnostic(InternalDiagnostic::new(
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
        name: (impl Into<Self::Ident>, Self::Span),
        params: Vec<(Self::Ident, Self::Span)>,
        body: Vec<(Token<Self::Ident>, Self::Span)>,
    ) {
        self.macro_table.define(
            name,
            params,
            body,
            self.context_factory.macro_context_factory(),
        );
    }
}

trait Tokenize<C: BufContext>
where
    for<'c> &'c Self::Tokenized: IntoIterator<Item = (Token<Self::Ident>, C::Span)>,
{
    type Ident: AsRef<str> + Clone + Into<String> + Debug + PartialEq;
    type Tokenized;
    fn tokenize_file<F: FnOnce(BufId) -> C>(
        &self,
        filename: &str,
        mk_context: F,
    ) -> Result<Self::Tokenized, CodebaseError>;
}

impl<C: Codebase, B: BufContext> Tokenize<B> for C {
    type Ident = String;
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
            }).when(|mut session| session.analyze_file(filename.to_string()).unwrap());
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
        TestFixture::new(&log).when(|mut session| session.emit_item(item.clone()));
        assert_eq!(*log.borrow(), [TestEvent::EmitItem(item)]);
    }

    #[test]
    fn define_label() {
        let label = "label";
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut session| session.define_label((label.to_string(), ())));
        assert_eq!(*log.borrow(), [TestEvent::AddLabel(String::from(label))]);
    }

    use crate::frontend::syntax::keyword::*;

    #[test]
    fn define_and_invoke_macro() {
        let name = "my_macro";
        let tokens = vec![Token::Command(Command::Mnemonic(Mnemonic::Nop))];
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut session| {
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
        TestFixture::new(&log).when(|mut session| {
            let name = "my_db";
            let param = "x";
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
        TestFixture::new(&log).when(|mut session| {
            let name = String::from("my_macro");
            let param = String::from("x");
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

    use crate::diagnostics::InternalDiagnostic;

    #[test]
    fn diagnose_undefined_macro() {
        let name = "my_macro".to_string();
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut session| session.invoke_macro((name.clone(), ()), vec![]));
        assert_eq!(
            *log.borrow(),
            [TestEvent::Diagnostic(InternalDiagnostic::new(
                Message::UndefinedMacro { name },
                ()
            ))]
        );
    }

    impl Span for SpanData<(), i32> {
        fn extend(&self, _: &Self) -> Self {
            unimplemented!()
        }
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

    impl<'a> Tokenize<Mock<'a>> for MockTokenSource {
        type Ident = String;
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

    impl<'a> MacroContextFactory for Mock<'a> {
        type Span = ();
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

    impl<'a> ContextFactory for Mock<'a> {
        type Span = ();
        type BufContext = Mock<'a>;
        type MacroContextFactory = Self;

        fn macro_context_factory(&mut self) -> &mut Self::MacroContextFactory {
            self
        }

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
        fn run<I, F>(&self, tokens: I, _frontend: &mut F)
        where
            I: Iterator<Item = (Token<String>, F::Span)>,
            F: Session<Ident = String>,
        {
            self.log.borrow_mut().push(TestEvent::AnalyzeTokens(
                tokens.map(|(t, _)| t.into()).collect(),
            ))
        }
    }

    impl<'a> Backend<()> for Mock<'a> {
        type Object = ();

        fn add_label(&mut self, (label, _): (impl Into<String>, ())) {
            self.log
                .borrow_mut()
                .push(TestEvent::AddLabel(label.into()))
        }

        fn emit_item(&mut self, item: Item<()>) {
            self.log.borrow_mut().push(TestEvent::EmitItem(item))
        }

        fn into_object(self) {}

        fn set_origin(&mut self, _origin: RelocExpr<()>) {
            unimplemented!()
        }
    }

    impl<'a> DiagnosticsListener<()> for Mock<'a> {
        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<()>) {
            self.log
                .borrow_mut()
                .push(TestEvent::Diagnostic(diagnostic))
        }
    }

    type TestLog = RefCell<Vec<TestEvent>>;

    #[derive(Debug, PartialEq)]
    enum TestEvent {
        AnalyzeTokens(Vec<Token<String>>),
        AddLabel(String),
        Diagnostic(InternalDiagnostic<()>),
        EmitItem(Item<()>),
    }

    struct TestFixture<'a> {
        macro_table: MacroExpander<String, ()>,
        mock_token_source: MockTokenSource,
        mock_context_factory: Mock<'a>,
        analysis: Mock<'a>,
        object: Mock<'a>,
        diagnostics: Mock<'a>,
    }

    impl<'a> TestFixture<'a> {
        fn new(log: &'a TestLog) -> TestFixture<'a> {
            TestFixture {
                macro_table: MacroExpander::new(),
                mock_token_source: MockTokenSource::new(),
                mock_context_factory: Mock::new(log),
                analysis: Mock::new(log),
                object: Mock::new(log),
                diagnostics: Mock::new(log),
            }
        }

        fn given<F: FnOnce(&mut Self)>(mut self, f: F) -> Self {
            f(&mut self);
            self
        }

        fn when<F: for<'b> FnOnce(TestSession<'b>)>(self, f: F) {
            let file_parser = CodebaseAnalyzer::new(
                self.macro_table,
                &self.mock_token_source,
                self.mock_context_factory,
                self.analysis,
            );
            let session = Components::new(file_parser, self.object, self.diagnostics);
            f(session);
        }
    }

    type TestCodebaseAnalyzer<'a> =
        CodebaseAnalyzer<'a, MacroExpander<String, ()>, MockTokenSource, Mock<'a>, Mock<'a>>;
    type TestSession<'a> = Components<
        TestCodebaseAnalyzer<'a>,
        Mock<'a>,
        Mock<'a>,
        TestCodebaseAnalyzer<'a>,
        Mock<'a>,
        Mock<'a>,
    >;
}
