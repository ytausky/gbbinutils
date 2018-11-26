mod semantics;
mod session;
mod syntax;

use crate::backend::*;
use crate::codebase::{Codebase, CodebaseError};
use crate::diagnostics::*;
use crate::frontend::session::*;
use crate::frontend::syntax::*;
use crate::span::{BufContext, ContextFactory, Span};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
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
    let factory = SemanticAnalysisFactory::new();
    let token_provider = TokenStreamSource::new(codebase, context_factory);
    let file_parser = FileParser::new(factory, MacroExpander::new(), token_provider);
    let mut session: Components<_, _, D, _, _, _> =
        Components::new(file_parser, backend, diagnostics);
    session.analyze_file(name)?;
    Ok(session.build_object())
}

pub struct Downstream<'a, B: 'a, D: 'a> {
    backend: &'a mut B,
    diagnostics: &'a mut D,
}

trait Analysis<Id: Into<String> + Debug + PartialEq> {
    fn run<I, S>(&mut self, tokens: I, session: &mut S)
    where
        I: Iterator<Item = (Token<Id>, S::Span)>,
        S: Session<Ident = Id>;
}

struct SemanticAnalysis;

impl SemanticAnalysis {
    fn new() -> SemanticAnalysis {
        SemanticAnalysis {}
    }
}

impl<Id: Into<String> + Debug + PartialEq> Analysis<Id> for SemanticAnalysis {
    fn run<I, S>(&mut self, tokens: I, session: &mut S)
    where
        I: Iterator<Item = (Token<Id>, S::Span)>,
        S: Session<Ident = Id>,
    {
        let actions = semantics::SemanticActions::new(session);
        syntax::parse_token_seq(tokens, actions)
    }
}

trait AnalysisFactory<Id: Into<String> + Debug + PartialEq> {
    type Analysis: Analysis<Id>;
    fn mk_analysis(&mut self) -> Self::Analysis;
}

struct SemanticAnalysisFactory;

impl SemanticAnalysisFactory {
    fn new() -> SemanticAnalysisFactory {
        SemanticAnalysisFactory {}
    }
}

impl<Id: Into<String> + Debug + PartialEq> AnalysisFactory<Id> for SemanticAnalysisFactory {
    type Analysis = SemanticAnalysis;

    fn mk_analysis(&mut self) -> Self::Analysis {
        SemanticAnalysis::new()
    }
}

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

struct FileParser<AF, M, TCS> {
    analysis_factory: AF,
    macros: M,
    tokenized_code_source: TCS,
}

impl<AF, M, TCS> FileParser<AF, M, TCS>
where
    AF: AnalysisFactory<TCS::Ident>,
    M: MacroTable<Ident = TCS::Ident, Span = TCS::Span>,
    TCS: TokenizedCodeSource,
    for<'a> &'a TCS::Tokenized: IntoIterator<Item = (Token<TCS::Ident>, TCS::Span)>,
{
    fn new(analysis_factory: AF, macros: M, tokenized_code_source: TCS) -> FileParser<AF, M, TCS> {
        FileParser {
            analysis_factory,
            macros,
            tokenized_code_source,
        }
    }

    fn analyze_token_seq<I: IntoIterator<Item = (Token<TCS::Ident>, TCS::Span)>>(
        &mut self,
        tokens: I,
        downstream: &mut Downstream<impl Backend<TCS::Span>, impl DiagnosticsListener<TCS::Span>>,
    ) {
        let mut analysis = self.analysis_factory.mk_analysis();
        let mut session = BorrowedComponents::new(self, downstream.backend, downstream.diagnostics);
        analysis.run(tokens.into_iter(), &mut session)
    }
}

type TokenSeq<I, S> = Vec<(Token<I>, S)>;

impl<AF, M, TCS> Frontend for FileParser<AF, M, TCS>
where
    AF: AnalysisFactory<TCS::Ident>,
    M: MacroTable<Ident = TCS::Ident, Span = TCS::Span>,
    TCS: TokenizedCodeSource,
    for<'a> &'a TCS::Tokenized: IntoIterator<Item = (Token<TCS::Ident>, TCS::Span)>,
{
    type Ident = TCS::Ident;
    type Span = TCS::Span;

    fn analyze_file<B, D>(
        &mut self,
        path: Self::Ident,
        mut downstream: Downstream<B, D>,
    ) -> Result<(), CodebaseError>
    where
        B: Backend<Self::Span>,
        D: DiagnosticsListener<Self::Span>,
    {
        let tokenized_src = self.tokenized_code_source.tokenize_file(path.as_ref())?;
        self.analyze_token_seq(&tokenized_src, &mut downstream);
        Ok(())
    }

    fn invoke_macro<B, D>(
        &mut self,
        name: (Self::Ident, Self::Span),
        args: MacroArgs<Self::Ident, Self::Span>,
        mut downstream: Downstream<B, D>,
    ) where
        B: Backend<Self::Span>,
        D: DiagnosticsListener<Self::Span>,
    {
        match self.macros.get(&name) {
            Some(def) => self.analyze_token_seq(def.expand(args, ()), &mut downstream),
            None => {
                let (name, name_ref) = name;
                downstream
                    .diagnostics
                    .emit_diagnostic(InternalDiagnostic::new(
                        Message::UndefinedMacro { name: name.into() },
                        name_ref,
                    ))
            }
        }
    }

    fn define_macro(
        &mut self,
        name: (impl Into<Self::Ident>, Self::Span),
        params: Vec<(Self::Ident, Self::Span)>,
        tokens: Vec<(Token<Self::Ident>, Self::Span)>,
    ) {
        self.macros.define(name, params, tokens)
    }
}

trait MacroTable {
    type Ident;
    type Span: Span;
    type Def: MacroDef<Ident = Self::Ident, Span = Self::Span> + Clone;

    fn define(
        &mut self,
        name: (impl Into<Self::Ident>, Self::Span),
        params: Vec<(Self::Ident, Self::Span)>,
        body: Vec<(Token<Self::Ident>, Self::Span)>,
    );

    fn get(&self, name: &(Self::Ident, Self::Span)) -> Option<Self::Def>;
}

trait MacroDef {
    type Ident;
    type Span: Span;
    type ExpandedMacro: Iterator<Item = (Token<Self::Ident>, Self::Span)>;

    fn expand(
        &self,
        args: Vec<TokenSeq<Self::Ident, Self::Span>>,
        context: (),
    ) -> Self::ExpandedMacro;
}

struct MacroExpander<I, S: Span> {
    macro_defs: HashMap<I, Rc<MacroDefData<I, S>>>,
}

struct MacroDefData<I, S> {
    params: Vec<(I, S)>,
    body: Vec<(Token<I>, S)>,
}

impl<I: Eq + Hash, S: Span> MacroExpander<I, S> {
    fn new() -> MacroExpander<I, S> {
        MacroExpander {
            macro_defs: HashMap::new(),
        }
    }
}

impl<I: AsRef<str> + Clone + Eq + Hash, S: Span> MacroTable for MacroExpander<I, S> {
    type Ident = I;
    type Span = S;
    type Def = Rc<MacroDefData<I, S>>;

    fn define(
        &mut self,
        name: (impl Into<Self::Ident>, Self::Span),
        params: Vec<(Self::Ident, Self::Span)>,
        body: Vec<(Token<Self::Ident>, Self::Span)>,
    ) {
        self.macro_defs
            .insert(name.0.into(), Rc::new(MacroDefData { params, body }));
    }

    fn get(&self, name: &(Self::Ident, Self::Span)) -> Option<Self::Def> {
        self.macro_defs.get(&name.0).cloned()
    }
}

impl<I: AsRef<str> + Clone + Eq, S: Span> MacroDef for Rc<MacroDefData<I, S>> {
    type Ident = I;
    type Span = S;
    type ExpandedMacro = ExpandedMacro<I, S>;

    fn expand(&self, args: Vec<TokenSeq<I, S>>, _context: ()) -> Self::ExpandedMacro {
        ExpandedMacro::new(self.clone(), args)
    }
}

struct ExpandedMacro<I, S> {
    def: Rc<MacroDefData<I, S>>,
    args: Vec<Vec<(Token<I>, S)>>,
    body_index: usize,
    expansion_state: Option<ExpansionState>,
}

#[derive(Debug, PartialEq)]
enum ExpansionState {
    Ident(usize, usize),
    Label(usize),
}

impl<I: AsRef<str> + PartialEq, S> ExpandedMacro<I, S> {
    fn new(def: Rc<MacroDefData<I, S>>, args: Vec<Vec<(Token<I>, S)>>) -> ExpandedMacro<I, S> {
        let mut expanded_macro = ExpandedMacro {
            def,
            args,
            body_index: 0,
            expansion_state: None,
        };
        expanded_macro.try_expand();
        expanded_macro
    }

    fn param_position(&self, name: &str) -> Option<usize> {
        for i in 0..self.def.params.len() {
            if self.def.params[i].0.as_ref() == name {
                return Some(i);
            }
        }
        None
    }

    fn advance(&mut self) {
        self.expansion_state = match self.expansion_state {
            Some(ExpansionState::Ident(position, index))
                if index + 1 < self.args[position].len() =>
            {
                Some(ExpansionState::Ident(position, index + 1))
            }
            _ => None,
        };
        if self.expansion_state.is_none() {
            self.body_index += 1;
            self.try_expand()
        }
    }

    fn try_expand(&mut self) {
        assert_eq!(self.expansion_state, None);
        if self.body_index < self.def.body.len() {
            self.expansion_state = self.expand_token(&self.def.body[self.body_index].0);
        }
    }

    fn expand_token(&self, token: &Token<I>) -> Option<ExpansionState> {
        match token {
            Token::Ident(ident) => self
                .param_position(ident.as_ref())
                .map(|position| ExpansionState::Ident(position, 0)),
            Token::Label(label) => self
                .param_position(label.as_ref())
                .map(ExpansionState::Label),
            _ => None,
        }
    }
}

impl<I: AsRef<str> + Clone + Eq, S: Clone> Iterator for ExpandedMacro<I, S> {
    type Item = (Token<I>, S);
    fn next(&mut self) -> Option<Self::Item> {
        if self.body_index < self.def.body.len() {
            let item = match self.expansion_state {
                Some(ExpansionState::Ident(position, index)) => self.args[position][index].clone(),
                Some(ExpansionState::Label(position)) => match self.args[position][0] {
                    (Token::Ident(ref ident), ref span) => {
                        (Token::Label(ident.clone()), span.clone())
                    }
                    _ => unimplemented!(),
                },
                None => self.def.body[self.body_index].clone(),
            };
            self.advance();
            Some(item)
        } else {
            None
        }
    }
}

trait TokenizedCodeSource
where
    for<'c> &'c Self::Tokenized: IntoIterator<Item = (Token<Self::Ident>, Self::Span)>,
{
    type Ident: AsRef<str> + Clone + Into<String> + Debug + PartialEq;
    type Span: Span;
    type Tokenized;
    fn tokenize_file(&mut self, filename: &str) -> Result<Self::Tokenized, CodebaseError>;
}

struct TokenStreamSource<'a, C: Codebase + 'a, F: ContextFactory> {
    codebase: &'a C,
    context_factory: F,
}

impl<'a, C: Codebase + 'a, F: ContextFactory> TokenStreamSource<'a, C, F> {
    fn new(codebase: &'a C, context_factory: F) -> TokenStreamSource<C, F> {
        TokenStreamSource {
            codebase,
            context_factory,
        }
    }
}

impl<'a, C: Codebase + 'a, F: ContextFactory> TokenizedCodeSource for TokenStreamSource<'a, C, F> {
    type Ident = String;
    type Span = F::Span;
    type Tokenized = TokenizedSrc<F::BufContext>;

    fn tokenize_file(&mut self, filename: &str) -> Result<Self::Tokenized, CodebaseError> {
        let buf_id = self.codebase.open(filename)?;
        let rc_src = self.codebase.buf(buf_id);
        Ok(TokenizedSrc::new(
            rc_src,
            self.context_factory.mk_buf_context(buf_id, None),
        ))
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
    use crate::frontend::syntax::keyword::Mnemonic;
    use crate::instruction::{Instruction, Nullary};
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

    impl TokenizedCodeSource for MockTokenSource {
        type Ident = String;
        type Span = ();
        type Tokenized = MockTokenized;

        fn tokenize_file(&mut self, filename: &str) -> Result<Self::Tokenized, CodebaseError> {
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

    #[derive(Clone)]
    struct Mock<'a> {
        log: &'a TestLog,
    }

    impl<'a> Mock<'a> {
        fn new(log: &'a TestLog) -> Mock<'a> {
            Mock { log }
        }
    }

    impl<'a> AnalysisFactory<String> for Mock<'a> {
        type Analysis = Self;
        fn mk_analysis(&mut self) -> Self::Analysis {
            self.clone()
        }
    }

    impl<'a> Analysis<String> for Mock<'a> {
        fn run<I, F>(&mut self, tokens: I, _frontend: &mut F)
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
        macros: MacroExpander<String, ()>,
        mock_token_source: MockTokenSource,
        analysis_factory: Mock<'a>,
        object: Mock<'a>,
        diagnostics: Mock<'a>,
    }

    impl<'a> TestFixture<'a> {
        fn new(log: &'a TestLog) -> TestFixture<'a> {
            TestFixture {
                macros: MacroExpander::new(),
                mock_token_source: MockTokenSource::new(),
                analysis_factory: Mock::new(log),
                object: Mock::new(log),
                diagnostics: Mock::new(log),
            }
        }

        fn given<F: FnOnce(&mut Self)>(mut self, f: F) -> Self {
            f(&mut self);
            self
        }

        fn when<F: for<'b> FnOnce(TestSession<'b>)>(self, f: F) {
            let file_parser =
                FileParser::new(self.analysis_factory, self.macros, self.mock_token_source);
            let session = Components::new(file_parser, self.object, self.diagnostics);
            f(session);
        }
    }

    type TestFileParser<'a> = FileParser<Mock<'a>, MacroExpander<String, ()>, MockTokenSource>;
    type TestSession<'a> =
        Components<TestFileParser<'a>, Mock<'a>, Mock<'a>, TestFileParser<'a>, Mock<'a>, Mock<'a>>;
}
