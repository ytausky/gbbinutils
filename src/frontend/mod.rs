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
    B: Backend<Ident<String>, D::Span>,
    D: Diagnostics,
{
    let mut file_parser =
        CodebaseAnalyzer::new(MacroExpander::new(), codebase, SemanticAnalysis {});
    let mut names = NameTable::new();
    let mut session = Session::new(&mut file_parser, &mut backend, &mut names, diagnostics);
    session.analyze_file(name)?;
    Ok(backend.into_object())
}

pub struct Downstream<'a, B: 'a, N: 'a, D: 'a> {
    backend: &'a mut B,
    names: &'a mut N,
    diagnostics: &'a mut D,
}

type LexItem<T, S> = (Result<SemanticToken<T>, LexError>, S);

type SemanticToken<T> = Token<Ident<T>, Literal<T>, syntax::Command>;

#[derive(Clone, Debug, PartialEq)]
pub struct Ident<T> {
    pub name: T,
}

#[cfg(test)]
impl<T> From<T> for Ident<T> {
    fn from(name: T) -> Ident<T> {
        Ident { name }
    }
}

#[cfg(test)]
impl From<&str> for Ident<String> {
    fn from(name: &str) -> Ident<String> {
        Ident { name: name.into() }
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
    Id: Into<String> + Clone + AsRef<str> + PartialEq,
{
    fn run<I, F, B, D>(&self, tokens: I, session: Session<F, B, NameTable, D>)
    where
        I: Iterator<Item = LexItem<Id, D::Span>>,
        F: Frontend<D, StringRef = Id>,
        B: Backend<Ident<Id>, D::Span>,
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
        downstream: Downstream<B, NameTable, D>,
    ) -> Result<(), CodebaseError>
    where
        B: Backend<Ident<Self::StringRef>, D::Span>;

    fn invoke_macro<B>(
        &mut self,
        name: (Ident<Self::StringRef>, D::Span),
        args: MacroArgs<Self::StringRef, D::Span>,
        downstream: Downstream<B, NameTable, D>,
    ) where
        B: Backend<Ident<Self::StringRef>, D::Span>;

    fn define_macro(
        &mut self,
        name: (Ident<Self::StringRef>, D::Span),
        params: Vec<(Ident<Self::StringRef>, D::Span)>,
        tokens: Vec<(SemanticToken<Self::StringRef>, D::Span)>,
        diagnostics: &mut D,
    );
}

impl<Id> Analysis<Id> for SemanticAnalysis
where
    Id: Into<String> + Clone + AsRef<str> + PartialEq,
{
    fn run<'a, I, F, B, D>(&self, tokens: I, session: Session<'a, F, B, NameTable, D>)
    where
        I: Iterator<Item = LexItem<Id, D::Span>>,
        F: Frontend<D, StringRef = Id>,
        B: Backend<Ident<Id>, D::Span>,
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

    fn analyze_token_seq<I, B, D>(
        &mut self,
        tokens: I,
        downstream: &mut Downstream<B, NameTable, D>,
    ) where
        I: IntoIterator<Item = LexItem<T::StringRef, D::Span>>,
        B: Backend<Ident<T::StringRef>, D::Span>,
        D: Diagnostics<MacroDefId = M::MacroDefId>,
        T: Tokenize<D::BufContext>,
        M::Entry: Expand<T::StringRef, D, D::Span>,
        for<'b> &'b T::Tokenized: IntoIterator<Item = LexItem<T::StringRef, D::Span>>,
    {
        let analysis = self.analysis.clone();
        let session = Session::new(
            self,
            downstream.backend,
            downstream.names,
            downstream.diagnostics,
        );
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
        mut downstream: Downstream<B, NameTable, D>,
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
        mut downstream: Downstream<B, NameTable, D>,
    ) where
        B: Backend<Ident<Self::StringRef>, D::Span>,
    {
        let expansion = match self.macro_table.get(&name) {
            Some(entry) => Some(entry.expand(span, args, downstream.diagnostics)),
            None => {
                let stripped = downstream.diagnostics.strip_span(&span);
                downstream
                    .diagnostics
                    .emit_diagnostic(CompactDiagnostic::new(
                        Message::UndefinedMacro { name: stripped },
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
        name: (Ident<Self::StringRef>, D::Span),
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
    use crate::backend::NameTable;
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
        let log = TestLog::<()>::default();
        TestFixture::new(&log)
            .when(|mut fixture| fixture.session().backend.emit_item(item.clone()));
        assert_eq!(*log.borrow(), [TestEvent::EmitItem(item)]);
    }

    #[test]
    fn define_label() {
        let label = "label";
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut fixture| {
            fixture.session().backend.define_symbol(
                (label.into(), ()),
                RelocAtom::LocationCounter.into(),
                &mut NameTable::new(),
            )
        });
        assert_eq!(
            *log.borrow(),
            [TestEvent::DefineSymbol(
                label.into(),
                RelocAtom::LocationCounter.into()
            )]
        );
    }

    use crate::frontend::syntax::keyword::*;

    #[test]
    fn define_and_invoke_macro() {
        let name = "my_macro";
        let tokens = vec![Token::Command(Command::Mnemonic(Mnemonic::Nop))];
        let log = TestLog::<()>::default();
        TestFixture::new(&log).when(|mut fixture| {
            let mut session = fixture.session();
            session.define_macro((name.into(), ()), Vec::new(), add_code_refs(&tokens));
            session.invoke_macro((name.into(), ()), vec![])
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
                (name.into(), ()),
                vec![(param.into(), ())],
                vec![
                    (db.clone(), ()),
                    (Token::Ident(param.into()), ()),
                    (literal0.clone(), ()),
                ],
            );
            session.invoke_macro((name.into(), ()), vec![vec![(arg.clone(), ())]])
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
        let label = "label";
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut fixture| {
            let name = "my_macro";
            let param = "x";
            let mut session = fixture.session();
            session.define_macro(
                (name.into(), ()),
                vec![(param.into(), ())],
                vec![(Token::Label(param.into()), ()), (nop.clone(), ())],
            );
            session.invoke_macro(
                (name.into(), ()),
                vec![vec![(Token::Ident(label.into()), ())]],
            )
        });
        assert_eq!(
            *log.borrow(),
            [TestEvent::AnalyzeTokens(vec![
                Ok(Token::Label(label.into())),
                Ok(nop)
            ])]
        );
    }

    use crate::diag::CompactDiagnostic;

    #[test]
    fn diagnose_undefined_macro() {
        let name = "my_macro";
        let log = TestLog::<&'static str>::default();
        TestFixture::new(&log)
            .when(|mut fixture| fixture.session().invoke_macro((name.into(), name), vec![]));
        assert_eq!(
            *log.borrow(),
            [TestEvent::Diagnostic(CompactDiagnostic::new(
                Message::UndefinedMacro { name },
                name
            ))]
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

        fn add_file(&mut self, name: &str, tokens: Vec<LexItem<String, S>>) {
            self.files.insert(name.to_string(), tokens);
        }
    }

    impl<'a, S> StringRef for MockTokenSource<S> {
        type StringRef = String;
    }

    impl<'a, S: Clone> Tokenize<Mock<'a, S>> for MockTokenSource<S> {
        type Tokenized = MockTokenized<S>;

        fn tokenize_file<F: FnOnce(BufId) -> Mock<'a, S>>(
            &self,
            filename: &str,
            _: F,
        ) -> Result<Self::Tokenized, CodebaseError> {
            Ok(MockTokenized(self.files.get(filename).unwrap().clone()))
        }
    }

    struct MockTokenized<S>(Vec<LexItem<String, S>>);

    impl<'b, S: Clone> IntoIterator for &'b MockTokenized<S> {
        type Item = LexItem<String, S>;
        type IntoIter = std::iter::Cloned<std::slice::Iter<'b, LexItem<String, S>>>;

        fn into_iter(self) -> Self::IntoIter {
            (&self.0).into_iter().cloned()
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

    impl<'a, S: Clone> Span for Mock<'a, S> {
        type Span = S;
    }

    impl<'a, S: Clone + Default> MacroContextFactory<S> for Mock<'a, S> {
        type MacroDefId = ();
        type MacroExpansionContext = Mock<'a, S>;

        fn add_macro_def<P, B>(&mut self, _: S, _: P, _: B) -> Self::MacroDefId
        where
            P: IntoIterator<Item = S>,
            B: IntoIterator<Item = S>,
        {
        }

        fn mk_macro_expansion_context<I, J>(
            &mut self,
            _: S,
            _: I,
            _: &Self::MacroDefId,
        ) -> Self::MacroExpansionContext
        where
            I: IntoIterator<Item = J>,
            J: IntoIterator<Item = S>,
        {
            self.clone()
        }
    }

    impl<'a, S: Clone> StripSpan<S> for Mock<'a, S> {
        type Stripped = S;

        fn strip_span(&mut self, span: &S) -> S {
            span.clone()
        }
    }

    impl<'a, S: Clone + Default> ContextFactory for Mock<'a, S> {
        type BufContext = Mock<'a, S>;

        fn mk_buf_context(&mut self, _: BufId, _: Option<Self::Span>) -> Self::BufContext {
            self.clone()
        }
    }

    impl<'a, S: Clone> BufContext for Mock<'a, S> {
        type Span = S;

        fn mk_span(&self, _: codebase::BufRange) -> Self::Span {
            unreachable!()
        }
    }

    impl<'a, S: Clone + Default> MacroExpansionContext for Mock<'a, S> {
        type Span = S;

        fn mk_span(&self, _: usize, _: Option<TokenExpansion>) -> Self::Span {
            S::default()
        }
    }

    impl<'a, S: Clone> Analysis<String> for Mock<'a, S> {
        fn run<I, F, B, D>(&self, tokens: I, _frontend: Session<F, B, NameTable, D>)
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

    impl<'a, 'b, S: Clone> BuildValue<'b, Ident<String>, S> for Mock<'a, S> {
        type Builder = IndependentValueBuilder<'b, S>;

        fn build_value(&'b mut self, names: &'b mut NameTable) -> Self::Builder {
            IndependentValueBuilder::new(names)
        }
    }

    impl<'a, S: Clone> HasValue<S> for Mock<'a, S> {
        type Value = RelocExpr<Ident<String>, S>;
    }

    impl<'a, S: Clone> Backend<Ident<String>, S> for Mock<'a, S> {
        type Object = ();

        fn define_symbol(
            &mut self,
            symbol: (Ident<String>, S),
            value: Self::Value,
            _: &mut NameTable,
        ) {
            self.log
                .borrow_mut()
                .push(TestEvent::DefineSymbol(symbol.0, value))
        }

        fn emit_item(&mut self, item: Item<Self::Value>) {
            self.log.borrow_mut().push(TestEvent::EmitItem(item))
        }

        fn into_object(self) {}

        fn set_origin(&mut self, _origin: Self::Value) {
            unimplemented!()
        }
    }

    impl<'a, S: Clone> EmitDiagnostic<S, S> for Mock<'a, S> {
        fn emit_diagnostic(&mut self, diagnostic: CompactDiagnostic<S, S>) {
            self.log
                .borrow_mut()
                .push(TestEvent::Diagnostic(diagnostic))
        }
    }

    impl<'a, S: Clone> MergeSpans<S> for Mock<'a, S> {
        fn merge_spans(&mut self, _: &S, _: &S) -> S {
            unreachable!()
        }
    }

    impl<'a, S: Clone + Default> Diagnostics for Mock<'a, S> {}

    type TestLog<S> = RefCell<Vec<TestEvent<S>>>;

    #[derive(Debug, PartialEq)]
    enum TestEvent<S: Clone> {
        AnalyzeTokens(Vec<Result<SemanticToken<String>, syntax::lexer::LexError>>),
        DefineSymbol(Ident<String>, RelocExpr<Ident<String>, S>),
        Diagnostic(CompactDiagnostic<S, S>),
        EmitItem(Item<RelocExpr<Ident<String>, S>>),
    }

    struct TestFixture<'a, S: Clone> {
        macro_table: MacroExpander<String, ()>,
        mock_token_source: MockTokenSource<S>,
        analysis: Mock<'a, S>,
        object: Mock<'a, S>,
        diagnostics: Mock<'a, S>,
    }

    struct PreparedFixture<'a, 'b, S: Clone> {
        code_analyzer:
            CodebaseAnalyzer<'b, MacroExpander<String, ()>, MockTokenSource<S>, Mock<'a, S>>,
        object: Mock<'a, S>,
        names: NameTable,
        diagnostics: Mock<'a, S>,
    }

    impl<'a, 'b, S: Clone> PreparedFixture<'a, 'b, S> {
        fn session<'r>(&'r mut self) -> TestSession<'a, 'b, 'r, S> {
            Session::new(
                &mut self.code_analyzer,
                &mut self.object,
                &mut self.names,
                &mut self.diagnostics,
            )
        }
    }

    impl<'a, S: Clone> TestFixture<'a, S> {
        fn new(log: &'a TestLog<S>) -> TestFixture<'a, S> {
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

        fn when<F: for<'b> FnOnce(PreparedFixture<'a, 'b, S>)>(self, f: F) {
            let prepared = PreparedFixture {
                code_analyzer: CodebaseAnalyzer::new(
                    self.macro_table,
                    &self.mock_token_source,
                    self.analysis,
                ),
                object: self.object,
                names: NameTable::new(),
                diagnostics: self.diagnostics,
            };
            f(prepared);
        }
    }

    type TestCodebaseAnalyzer<'a, 'r, S> =
        CodebaseAnalyzer<'r, MacroExpander<String, ()>, MockTokenSource<S>, Mock<'a, S>>;
    type TestSession<'a, 'b, 'r, S> =
        Session<'r, TestCodebaseAnalyzer<'a, 'b, S>, Mock<'a, S>, NameTable, Mock<'a, S>>;
}
