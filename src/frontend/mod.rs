mod semantics;
mod session;
mod syntax;

use backend::*;
use codebase::Codebase;
use diagnostics::*;
use frontend::session::{BorrowedComponents, ChunkId, Components, Session};
use frontend::syntax::*;
use span::{LexemeRefFactory, Span, TokenTracker};
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

pub use frontend::syntax::Token;

pub fn analyze_file<
    C: Codebase,
    TT: TokenTracker,
    B: Backend<TT::Span>,
    D: DiagnosticsListener<TT::Span>,
>(
    name: String,
    codebase: &C,
    token_tracker: TT,
    backend: B,
    diagnostics: &mut D,
) -> B::Object {
    let factory = SemanticAnalysisFactory::new();
    let token_provider = TokenStreamSource::new(codebase, token_tracker);
    let file_parser = FileParser::new(factory, MacroExpander::new(), token_provider);
    let mut session: Components<_, _, D, _, _, _> =
        Components::new(file_parser, backend, diagnostics);
    session.analyze_chunk(ChunkId::File((name, None)));
    session.build_object()
}

pub struct Downstream<'a, B: 'a, D: 'a> {
    backend: &'a mut B,
    diagnostics: &'a mut D,
}

trait Analysis<Id: Into<String>> {
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

impl<Id: Into<String>> Analysis<Id> for SemanticAnalysis {
    fn run<I, S>(&mut self, tokens: I, session: &mut S)
    where
        I: Iterator<Item = (Token<Id>, S::Span)>,
        S: Session<Ident = Id>,
    {
        let actions = semantics::SemanticActions::new(session);
        syntax::parse_token_seq(tokens, actions)
    }
}

trait AnalysisFactory<Id: Into<String>> {
    type Analysis: Analysis<Id>;
    fn mk_analysis(&mut self) -> Self::Analysis;
}

struct SemanticAnalysisFactory;

impl SemanticAnalysisFactory {
    fn new() -> SemanticAnalysisFactory {
        SemanticAnalysisFactory {}
    }
}

impl<Id: Into<String>> AnalysisFactory<Id> for SemanticAnalysisFactory {
    type Analysis = SemanticAnalysis;

    fn mk_analysis(&mut self) -> Self::Analysis {
        SemanticAnalysis::new()
    }
}

pub trait ExprFactory {
    fn mk_literal<I: Into<String>, S>(&mut self, literal: (Literal<I>, S)) -> RelocExpr<S>;
    fn mk_symbol<I: Into<String>, S>(&mut self, symbol: (I, S)) -> RelocExpr<S>;
}

pub struct StrExprFactory;

impl StrExprFactory {
    fn new() -> StrExprFactory {
        StrExprFactory {}
    }
}

impl ExprFactory for StrExprFactory {
    fn mk_literal<I: Into<String>, S>(
        &mut self,
        (literal, token_ref): (Literal<I>, S),
    ) -> RelocExpr<S> {
        match literal {
            Literal::Number(number) => RelocExpr::Literal(number, token_ref),
            _ => panic!(),
        }
    }

    fn mk_symbol<I: Into<String>, S>(&mut self, symbol: (I, S)) -> RelocExpr<S> {
        RelocExpr::Symbol(symbol.0.into(), symbol.1)
    }
}

pub trait Frontend {
    type Ident: AsRef<str> + Clone + Into<String>;
    type Span: Span;
    fn analyze_chunk(
        &mut self,
        chunk_id: ChunkId<Self::Ident, Self::Span>,
        downstream: Downstream<impl Backend<Self::Span>, impl DiagnosticsListener<Self::Span>>,
    );
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
    M: Macros<Ident = TCS::Ident, Span = TCS::Span>,
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

    fn include_source_file(
        &mut self,
        filename: &str,
        mut downstream: Downstream<impl Backend<TCS::Span>, impl DiagnosticsListener<TCS::Span>>,
    ) {
        let tokenized_src = self.tokenized_code_source.tokenize_file(filename);
        self.analyze_token_seq(&tokenized_src, &mut downstream)
    }

    fn invoke_macro(
        &mut self,
        name: (<Self as Frontend>::Ident, <Self as Frontend>::Span),
        args: Vec<TokenSeq<<Self as Frontend>::Ident, <Self as Frontend>::Span>>,
        mut downstream: Downstream<impl Backend<TCS::Span>, impl DiagnosticsListener<TCS::Span>>,
    ) {
        match self.macros.expand(name.clone(), args) {
            Some(tokens) => self.analyze_token_seq(tokens, &mut downstream),
            None => {
                let (name, name_ref) = name;
                downstream
                    .diagnostics
                    .emit_diagnostic(InternalDiagnostic::new(
                        Message::UndefinedMacro { name: name.into() },
                        vec![],
                        name_ref,
                    ))
            }
        }
    }
}

type TokenSeq<I, S> = Vec<(Token<I>, S)>;

impl<AF, M, TCS> Frontend for FileParser<AF, M, TCS>
where
    AF: AnalysisFactory<TCS::Ident>,
    M: Macros<Ident = TCS::Ident, Span = TCS::Span>,
    TCS: TokenizedCodeSource,
    for<'a> &'a TCS::Tokenized: IntoIterator<Item = (Token<TCS::Ident>, TCS::Span)>,
{
    type Ident = TCS::Ident;
    type Span = TCS::Span;

    fn analyze_chunk(
        &mut self,
        chunk_id: ChunkId<Self::Ident, Self::Span>,
        downstream: Downstream<impl Backend<Self::Span>, impl DiagnosticsListener<Self::Span>>,
    ) {
        match chunk_id {
            ChunkId::File((name, _)) => self.include_source_file(name.as_ref(), downstream),
            ChunkId::Macro { name, args } => self.invoke_macro(name, args, downstream),
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

trait Macros {
    type Ident;
    type Span: Span;
    type ExpandedMacro: Iterator<Item = (Token<Self::Ident>, Self::Span)>;

    fn define(
        &mut self,
        name: (impl Into<Self::Ident>, Self::Span),
        params: Vec<(Self::Ident, Self::Span)>,
        body: Vec<(Token<Self::Ident>, Self::Span)>,
    );

    fn expand(
        &mut self,
        name: (Self::Ident, Self::Span),
        args: Vec<TokenSeq<Self::Ident, Self::Span>>,
    ) -> Option<Self::ExpandedMacro>;
}

struct MacroExpander<I, S: Span> {
    macro_defs: HashMap<I, Rc<MacroDef<I, S>>>,
}

struct MacroDef<I, S> {
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

impl<I: AsRef<str> + Clone + Eq + Hash, S: Span> Macros for MacroExpander<I, S> {
    type Ident = I;
    type Span = S;
    type ExpandedMacro = ExpandedMacro<Self::Ident, Self::Span>;

    fn define(
        &mut self,
        name: (impl Into<Self::Ident>, Self::Span),
        params: Vec<(Self::Ident, Self::Span)>,
        body: Vec<(Token<Self::Ident>, Self::Span)>,
    ) {
        self.macro_defs
            .insert(name.0.into(), Rc::new(MacroDef { params, body }));
    }

    fn expand(
        &mut self,
        name: (Self::Ident, Self::Span),
        args: Vec<TokenSeq<Self::Ident, Self::Span>>,
    ) -> Option<Self::ExpandedMacro> {
        self.macro_defs
            .get(&name.0)
            .cloned()
            .map(|def| ExpandedMacro::new(def, args))
    }
}

struct ExpandedMacro<I, S> {
    def: Rc<MacroDef<I, S>>,
    args: Vec<Vec<(Token<I>, S)>>,
    body_index: usize,
    arg_index: Option<(usize, usize)>,
}

impl<I: AsRef<str> + PartialEq, S> ExpandedMacro<I, S> {
    fn new(def: Rc<MacroDef<I, S>>, args: Vec<Vec<(Token<I>, S)>>) -> ExpandedMacro<I, S> {
        let mut expanded_macro = ExpandedMacro {
            def,
            args,
            body_index: 0,
            arg_index: None,
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
        if let Some((position, index)) = self.arg_index {
            if index + 1 < self.args[position].len() {
                self.arg_index = Some((position, index + 1))
            } else {
                self.arg_index = None
            }
        }
        if self.arg_index.is_none() {
            self.body_index += 1;
            self.try_expand()
        }
    }

    fn try_expand(&mut self) {
        assert_eq!(self.arg_index, None);
        if self.body_index < self.def.body.len() {
            if let Token::Ident(ref name) = self.def.body[self.body_index].0 {
                if let Some(position) = self.param_position(name.as_ref()) {
                    self.arg_index = Some((position, 0))
                }
            }
        }
    }
}

impl<I: AsRef<str> + Clone + Eq, S: Clone> Iterator for ExpandedMacro<I, S> {
    type Item = (Token<I>, S);
    fn next(&mut self) -> Option<Self::Item> {
        if self.body_index < self.def.body.len() {
            let item = match self.arg_index {
                Some((position, index)) => self.args[position][index].clone(),
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
    type Ident: AsRef<str> + Clone + Into<String>;
    type Span: Span;
    type Tokenized;
    fn tokenize_file(&mut self, filename: &str) -> Self::Tokenized;
}

struct TokenStreamSource<'a, C: Codebase + 'a, TT: TokenTracker> {
    codebase: &'a C,
    token_tracker: TT,
}

impl<'a, C: Codebase + 'a, TT: TokenTracker> TokenStreamSource<'a, C, TT> {
    fn new(codebase: &'a C, token_tracker: TT) -> TokenStreamSource<C, TT> {
        TokenStreamSource {
            codebase,
            token_tracker,
        }
    }
}

impl<'a, C: Codebase + 'a, TT: TokenTracker> TokenizedCodeSource for TokenStreamSource<'a, C, TT> {
    type Ident = String;
    type Span = TT::Span;
    type Tokenized = TokenizedSrc<TT::BufContext>;

    fn tokenize_file(&mut self, filename: &str) -> Self::Tokenized {
        let buf_id = self.codebase.open(filename);
        let rc_src = self.codebase.buf(buf_id);
        TokenizedSrc::new(rc_src, self.token_tracker.mk_buf_context(buf_id, None))
    }
}

struct TokenizedSrc<LRF> {
    src: Rc<str>,
    lexeme_ref_factory: LRF,
}

impl<LRF: LexemeRefFactory> TokenizedSrc<LRF> {
    fn new(src: Rc<str>, lexeme_ref_factory: LRF) -> TokenizedSrc<LRF> {
        TokenizedSrc {
            src,
            lexeme_ref_factory,
        }
    }
}

impl<'a, LRF: LexemeRefFactory> IntoIterator for &'a TokenizedSrc<LRF> {
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = TokenizedSrcIter<'a, LRF>;
    fn into_iter(self) -> Self::IntoIter {
        TokenizedSrcIter {
            tokens: syntax::tokenize(&self.src),
            lexeme_ref_factory: &self.lexeme_ref_factory,
        }
    }
}

struct TokenizedSrcIter<'a, LRF: LexemeRefFactory + 'a> {
    tokens: syntax::lexer::Lexer<'a>,
    lexeme_ref_factory: &'a LRF,
}

impl<'a, LRF: LexemeRefFactory> Iterator for TokenizedSrcIter<'a, LRF> {
    type Item = (Token<String>, LRF::Span);
    fn next(&mut self) -> Option<Self::Item> {
        self.tokens
            .next()
            .map(|(t, r)| (t, self.lexeme_ref_factory.mk_lexeme_ref(r)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frontend::syntax::keyword::Mnemonic;
    use instruction::{Instruction, Nullary};
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
            .when(|mut session| session.analyze_chunk(ChunkId::File((filename.to_string(), None))));
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

    use frontend::syntax::keyword::*;

    #[test]
    fn define_and_invoke_macro() {
        let name = "my_macro";
        let tokens = vec![Token::Command(Command::Mnemonic(Mnemonic::Nop))];
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut session| {
            session.define_macro((name.to_string(), ()), Vec::new(), add_code_refs(&tokens));
            session.analyze_chunk(ChunkId::Macro {
                name: (name.to_string(), ()),
                args: vec![],
            })
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
            session.analyze_chunk(ChunkId::Macro {
                name: (name.to_string(), ()),
                args: vec![vec![(arg.clone(), ())]],
            })
        });
        assert_eq!(
            *log.borrow(),
            [TestEvent::AnalyzeTokens(vec![db, arg, literal0])]
        );
    }

    use diagnostics::InternalDiagnostic;

    #[test]
    fn diagnose_undefined_macro() {
        let name = "my_macro".to_string();
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut session| {
            session.analyze_chunk(ChunkId::Macro {
                name: (name.clone(), ()),
                args: vec![],
            })
        });
        assert_eq!(
            *log.borrow(),
            [TestEvent::Diagnostic(InternalDiagnostic::new(
                Message::UndefinedMacro { name },
                vec![],
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

        fn tokenize_file(&mut self, filename: &str) -> Self::Tokenized {
            MockTokenized(self.files.get(filename).unwrap().clone())
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
