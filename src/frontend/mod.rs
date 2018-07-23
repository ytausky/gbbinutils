mod semantics;
mod syntax;

use backend::*;
use codebase::Codebase;
use diagnostics::*;
use frontend::syntax::*;
use session::{BorrowedComponents, ChunkId, Components, Session};
use std::collections::HashMap;
use std::rc::Rc;

pub use frontend::syntax::Token;

pub fn analyze_file<
    C: Codebase,
    TT: TokenTracker,
    B: Backend<TT::TokenRef>,
    D: DiagnosticsListener<TT::TokenRef>,
>(
    name: String,
    codebase: &C,
    token_tracker: TT,
    backend: B,
    diagnostics: &D,
) -> B::Object {
    let factory = SemanticAnalysisFactory::new();
    let token_provider = TokenStreamSource::new(codebase, token_tracker);
    let file_parser = FileParser::new(factory, MacroExpander::new(), token_provider, diagnostics);
    let mut session: Components<_, _, D, _, _, _> =
        Components::new(file_parser, backend, diagnostics);
    session.analyze_chunk(ChunkId::File((name, None)));
    session.build_object()
}

trait Analysis {
    fn run<I, S>(&mut self, tokens: I, session: &mut S)
    where
        I: Iterator<Item = (Token, S::TokenRef)>,
        S: Session;
}

struct SemanticAnalysis;

impl SemanticAnalysis {
    fn new() -> SemanticAnalysis {
        SemanticAnalysis {}
    }
}

impl Analysis for SemanticAnalysis {
    fn run<I, S>(&mut self, tokens: I, session: &mut S)
    where
        I: Iterator<Item = (Token, S::TokenRef)>,
        S: Session,
    {
        let actions = semantics::SemanticActions::new(session);
        syntax::parse_token_seq(tokens, actions)
    }
}

trait AnalysisFactory {
    type Analysis: Analysis;
    fn mk_analysis(&mut self) -> Self::Analysis;
}

struct SemanticAnalysisFactory;

impl SemanticAnalysisFactory {
    fn new() -> SemanticAnalysisFactory {
        SemanticAnalysisFactory {}
    }
}

impl AnalysisFactory for SemanticAnalysisFactory {
    type Analysis = SemanticAnalysis;

    fn mk_analysis(&mut self) -> Self::Analysis {
        SemanticAnalysis::new()
    }
}

pub trait ExprFactory {
    fn mk_literal<R>(&mut self, literal: (Literal<String>, R)) -> RelocExpr<R>;
    fn mk_symbol<R>(&mut self, symbol: (String, R)) -> RelocExpr<R>;
}

pub struct StrExprFactory;

impl StrExprFactory {
    fn new() -> StrExprFactory {
        StrExprFactory {}
    }
}

impl ExprFactory for StrExprFactory {
    fn mk_literal<R>(&mut self, (literal, token_ref): (Literal<String>, R)) -> RelocExpr<R> {
        match literal {
            Literal::Number(number) => RelocExpr::Literal(number, token_ref),
            _ => panic!(),
        }
    }

    fn mk_symbol<R>(&mut self, symbol: (String, R)) -> RelocExpr<R> {
        RelocExpr::Symbol(symbol.0, symbol.1)
    }
}

pub trait Frontend {
    type TokenRef: SourceRange;
    fn analyze_chunk(
        &mut self,
        chunk_id: ChunkId<Self::TokenRef>,
        backend: &mut impl Backend<Self::TokenRef>,
    );
    fn define_macro(
        &mut self,
        name: (impl Into<String>, Self::TokenRef),
        params: Vec<(String, Self::TokenRef)>,
        tokens: Vec<(Token, Self::TokenRef)>,
    );
}

struct FileParser<'a, AF, M, TCS, DL: 'a> {
    analysis_factory: AF,
    macros: M,
    tokenized_code_source: TCS,
    diagnostics: &'a DL,
}

impl<'a, AF, M, TCS, DL> FileParser<'a, AF, M, TCS, DL>
where
    AF: AnalysisFactory,
    M: Macros<SourceRange = TCS::TokenRef>,
    TCS: TokenizedCodeSource,
    for<'b> &'b TCS::Tokenized: IntoIterator<Item = (Token, TCS::TokenRef)>,
    DL: DiagnosticsListener<TCS::TokenRef>,
{
    fn new(
        analysis_factory: AF,
        macros: M,
        tokenized_code_source: TCS,
        diagnostics: &DL,
    ) -> FileParser<AF, M, TCS, DL> {
        FileParser {
            analysis_factory,
            macros,
            tokenized_code_source,
            diagnostics,
        }
    }

    fn analyze_token_seq<
        I: IntoIterator<Item = (Token, TCS::TokenRef)>,
        B: Backend<TCS::TokenRef>,
    >(
        &mut self,
        tokens: I,
        backend: &mut B,
    ) {
        let mut analysis = self.analysis_factory.mk_analysis();
        let diagnostics = self.diagnostics;
        let mut session = BorrowedComponents::new(self, backend, diagnostics);
        analysis.run(tokens.into_iter(), &mut session)
    }

    fn include_source_file(&mut self, filename: &str, backend: &mut impl Backend<TCS::TokenRef>) {
        let tokenized_src = self.tokenized_code_source.tokenize_file(filename);
        self.analyze_token_seq(&tokenized_src, backend)
    }

    fn invoke_macro(
        &mut self,
        name: (String, <Self as Frontend>::TokenRef),
        args: Vec<Vec<(Token, <Self as Frontend>::TokenRef)>>,
        backend: &mut impl Backend<TCS::TokenRef>,
    ) {
        match self.macros.expand(name.clone(), args) {
            Some(tokens) => self.analyze_token_seq(tokens, backend),
            None => {
                let (name, name_ref) = name;
                self.diagnostics
                    .emit_diagnostic(Diagnostic::new(Message::UndefinedMacro { name }, name_ref))
            }
        }
    }
}

impl<'a, AF, M, TCS, DL> Frontend for FileParser<'a, AF, M, TCS, DL>
where
    AF: AnalysisFactory,
    M: Macros<SourceRange = TCS::TokenRef>,
    TCS: TokenizedCodeSource,
    for<'b> &'b TCS::Tokenized: IntoIterator<Item = (Token, TCS::TokenRef)>,
    DL: DiagnosticsListener<TCS::TokenRef>,
{
    type TokenRef = TCS::TokenRef;

    fn analyze_chunk(
        &mut self,
        chunk_id: ChunkId<Self::TokenRef>,
        backend: &mut impl Backend<Self::TokenRef>,
    ) {
        match chunk_id {
            ChunkId::File((name, _)) => self.include_source_file(&name, backend),
            ChunkId::Macro { name, args } => self.invoke_macro(name, args, backend),
        }
    }

    fn define_macro(
        &mut self,
        name: (impl Into<String>, Self::TokenRef),
        params: Vec<(String, Self::TokenRef)>,
        tokens: Vec<(Token, Self::TokenRef)>,
    ) {
        self.macros.define(name, params, tokens)
    }
}

trait Macros {
    type SourceRange: SourceRange;
    type ExpandedMacro: Iterator<Item = (Token, Self::SourceRange)>;

    fn define(
        &mut self,
        name: (impl Into<String>, Self::SourceRange),
        params: Vec<(String, Self::SourceRange)>,
        body: Vec<(Token, Self::SourceRange)>,
    );

    fn expand(
        &mut self,
        name: (String, Self::SourceRange),
        args: Vec<Vec<(Token, Self::SourceRange)>>,
    ) -> Option<Self::ExpandedMacro>;
}

struct MacroExpander<SR: SourceRange> {
    macro_defs: HashMap<String, Rc<MacroDef<SR>>>,
}

struct MacroDef<SR> {
    params: Vec<(String, SR)>,
    body: Vec<(Token, SR)>,
}

impl<SR: SourceRange> MacroExpander<SR> {
    fn new() -> MacroExpander<SR> {
        MacroExpander {
            macro_defs: HashMap::new(),
        }
    }
}

impl<SR: SourceRange> Macros for MacroExpander<SR> {
    type SourceRange = SR;
    type ExpandedMacro = ExpandedMacro<Self::SourceRange>;

    fn define(
        &mut self,
        name: (impl Into<String>, Self::SourceRange),
        params: Vec<(String, Self::SourceRange)>,
        body: Vec<(Token, Self::SourceRange)>,
    ) {
        self.macro_defs
            .insert(name.0.into(), Rc::new(MacroDef { params, body }));
    }

    fn expand(
        &mut self,
        name: (String, Self::SourceRange),
        args: Vec<Vec<(Token, Self::SourceRange)>>,
    ) -> Option<Self::ExpandedMacro> {
        self.macro_defs
            .get(&name.0)
            .cloned()
            .map(|def| ExpandedMacro::new(def, args))
    }
}

struct ExpandedMacro<SR> {
    def: Rc<MacroDef<SR>>,
    args: Vec<Vec<(Token, SR)>>,
    body_index: usize,
    arg_index: Option<(usize, usize)>,
}

impl<SR> ExpandedMacro<SR> {
    fn new(def: Rc<MacroDef<SR>>, args: Vec<Vec<(Token, SR)>>) -> ExpandedMacro<SR> {
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
            if self.def.params[i].0 == name {
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
        if let None = self.arg_index {
            self.body_index += 1;
            self.try_expand()
        }
    }

    fn try_expand(&mut self) {
        assert_eq!(self.arg_index, None);
        if self.body_index < self.def.body.len() {
            if let token::Ident(ref name) = self.def.body[self.body_index].0 {
                if let Some(position) = self.param_position(&name) {
                    self.arg_index = Some((position, 0))
                }
            }
        }
    }
}

impl<SR: Clone> Iterator for ExpandedMacro<SR> {
    type Item = (Token, SR);
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
    for<'c> &'c Self::Tokenized: IntoIterator<Item = (Token, Self::TokenRef)>,
{
    type TokenRef: SourceRange;
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
    type TokenRef = TT::TokenRef;
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
    type Item = (Token, LRF::TokenRef);
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
        let contents = vec![token::Command(Command::Mnemonic(Mnemonic::Nop))];
        let log = TestLog::default();
        TestFixture::new(&log)
            .given(|f| {
                f.mock_token_source
                    .add_file(filename, add_code_refs(&contents))
            })
            .when(|mut session| session.analyze_chunk(ChunkId::File((filename.to_string(), None))));
        assert_eq!(*log.borrow(), [TestEvent::AnalyzeTokens(contents)]);
    }

    fn add_code_refs<'a, I: IntoIterator<Item = &'a Token>>(tokens: I) -> Vec<(Token, ())> {
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
        let tokens = vec![token::Command(Command::Mnemonic(Mnemonic::Nop))];
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
        let db = token::Command(Command::Directive(Directive::Db));
        let arg = token::Literal(Literal::Number(0x42));
        let literal0 = token::Literal(Literal::Number(0));
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut session| {
            let name = "my_db";
            let param = "x";
            session.define_macro(
                (name.to_string(), ()),
                vec![(param.to_string(), ())],
                vec![
                    (db.clone(), ()),
                    (token::Ident(param.to_string()), ()),
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

    use diagnostics::Diagnostic;

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
            [TestEvent::Diagnostic(Diagnostic::new(
                Message::UndefinedMacro { name },
                ()
            ))]
        );
    }

    struct MockTokenSource {
        files: HashMap<String, Vec<(Token, ())>>,
    }

    impl MockTokenSource {
        fn new() -> MockTokenSource {
            MockTokenSource {
                files: HashMap::new(),
            }
        }

        fn add_file(&mut self, name: &str, tokens: Vec<(Token, ())>) {
            self.files.insert(name.to_string(), tokens);
        }
    }

    impl TokenizedCodeSource for MockTokenSource {
        type TokenRef = ();
        type Tokenized = MockTokenized;

        fn tokenize_file(&mut self, filename: &str) -> Self::Tokenized {
            MockTokenized(self.files.get(filename).unwrap().clone())
        }
    }

    struct MockTokenized(Vec<(Token, ())>);

    impl<'b> IntoIterator for &'b MockTokenized {
        type Item = (Token, ());
        type IntoIter = std::iter::Cloned<std::slice::Iter<'b, (Token, ())>>;
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

    impl<'a> AnalysisFactory for Mock<'a> {
        type Analysis = Self;
        fn mk_analysis(&mut self) -> Self::Analysis {
            self.clone()
        }
    }

    impl<'a> Analysis for Mock<'a> {
        fn run<I, F>(&mut self, tokens: I, _frontend: &mut F)
        where
            I: Iterator<Item = (Token, F::TokenRef)>,
            F: Session,
        {
            self.log
                .borrow_mut()
                .push(TestEvent::AnalyzeTokens(tokens.map(|(t, _)| t).collect()))
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
        fn emit_diagnostic(&self, diagnostic: Diagnostic<()>) {
            self.log
                .borrow_mut()
                .push(TestEvent::Diagnostic(diagnostic))
        }
    }

    type TestLog = RefCell<Vec<TestEvent>>;

    #[derive(Debug, PartialEq)]
    enum TestEvent {
        AnalyzeTokens(Vec<Token>),
        AddLabel(String),
        Diagnostic(Diagnostic<()>),
        EmitItem(Item<()>),
    }

    struct TestFixture<'a> {
        macros: MacroExpander<()>,
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
            let file_parser = FileParser::new(
                self.analysis_factory,
                self.macros,
                self.mock_token_source,
                &self.diagnostics,
            );
            let session = Components::new(file_parser, self.object, &self.diagnostics);
            f(session);
        }
    }

    type TestFileParser<'a> =
        FileParser<'a, Mock<'a>, MacroExpander<()>, MockTokenSource, Mock<'a>>;
    type TestSession<'a> = Components<
        TestFileParser<'a>,
        Mock<'a>,
        Mock<'a>,
        TestFileParser<'a>,
        Mock<'a>,
        &'a Mock<'a>,
    >;
}
