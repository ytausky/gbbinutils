use std::fmt::Debug;

mod semantics;
mod syntax;

use AssemblySession;
use backend::*;
use diagnostics::*;
use frontend::syntax::*;

use codebase::Codebase;

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
) -> B {
    let factory = SemanticTokenSeqAnalyzerFactory::new();
    let token_provider = TokenStreamSource::new(codebase, token_tracker);
    let mut session = Session::new(token_provider, factory, backend, diagnostics);
    session.analyze_chunk(::ChunkId::File((name, None)));
    session.into_object()
}

trait TokenSeqAnalyzer {
    fn analyze<I, F>(&mut self, tokens: I, frontend: &mut F)
    where
        I: Iterator<Item = (Token, F::TokenRef)>,
        F: AssemblySession;
}

struct SemanticTokenSeqAnalyzer;

impl SemanticTokenSeqAnalyzer {
    fn new() -> SemanticTokenSeqAnalyzer {
        SemanticTokenSeqAnalyzer {}
    }
}

impl TokenSeqAnalyzer for SemanticTokenSeqAnalyzer {
    fn analyze<I, F>(&mut self, tokens: I, frontend: &mut F)
    where
        I: Iterator<Item = (Token, F::TokenRef)>,
        F: AssemblySession,
    {
        let actions = semantics::SemanticActions::new(frontend);
        syntax::parse_token_seq(tokens, actions)
    }
}

trait TokenSeqAnalyzerFactory {
    type TokenSeqAnalyzer: TokenSeqAnalyzer;
    fn mk_token_seq_analyzer(&mut self) -> Self::TokenSeqAnalyzer;
}

struct SemanticTokenSeqAnalyzerFactory;

impl SemanticTokenSeqAnalyzerFactory {
    fn new() -> SemanticTokenSeqAnalyzerFactory {
        SemanticTokenSeqAnalyzerFactory {}
    }
}

impl TokenSeqAnalyzerFactory for SemanticTokenSeqAnalyzerFactory {
    type TokenSeqAnalyzer = SemanticTokenSeqAnalyzer;

    fn mk_token_seq_analyzer(&mut self) -> Self::TokenSeqAnalyzer {
        SemanticTokenSeqAnalyzer::new()
    }
}

pub trait ExprFactory {
    fn mk_atom<R>(&mut self, token: (Token, R)) -> Expr<R>;
}

pub struct StrExprFactory;

impl StrExprFactory {
    fn new() -> StrExprFactory {
        StrExprFactory {}
    }
}

impl ExprFactory for StrExprFactory {
    fn mk_atom<R>(&mut self, (token, token_ref): (Token, R)) -> Expr<R> {
        match token {
            token::Atom(Atom::Ident(ident)) => Expr::Symbol(ident, token_ref),
            token::Atom(Atom::Number(number)) => Expr::Literal(number, token_ref),
            _ => panic!(),
        }
    }
}

pub trait Frontend {
    type TokenRef: Debug + PartialEq;
    fn analyze_chunk(
        &mut self,
        chunk_id: ::ChunkId<Self::TokenRef>,
        backend: &mut impl Backend<Self::TokenRef>,
    );
    fn define_macro(
        &mut self,
        name: (impl Into<String>, Self::TokenRef),
        tokens: Vec<(Token, Self::TokenRef)>,
    );
}

struct FileParser<'a, SAF, TCS, DL: 'a> {
    analyzer_factory: SAF,
    tokenized_code_source: TCS,
    diagnostics: &'a DL,
}

impl<'a, SAF, TCS, DL> FileParser<'a, SAF, TCS, DL>
where
    SAF: TokenSeqAnalyzerFactory,
    TCS: TokenizedCodeSource,
    for<'b> &'b TCS::Tokenized: IntoIterator<Item = (Token, TCS::TokenRef)>,
    DL: DiagnosticsListener<TCS::TokenRef>,
{
    fn analyze_token_seq<
        I: IntoIterator<Item = (Token, TCS::TokenRef)>,
        B: Backend<TCS::TokenRef>,
    >(
        &mut self,
        tokens: I,
        backend: &mut B,
    ) {
        let mut analyzer = self.analyzer_factory.mk_token_seq_analyzer();
        let diagnostics = self.diagnostics;
        let mut session: ::Session<TCS::TokenRef, Self, B, DL, _, _, _> =
            ::Session::new(self, backend, diagnostics);
        analyzer.analyze(tokens.into_iter(), &mut session)
    }

    fn include_source_file(&mut self, filename: &str, backend: &mut impl Backend<TCS::TokenRef>) {
        let tokenized_src = self.tokenized_code_source.tokenize_file(filename);
        self.analyze_token_seq(&tokenized_src, backend)
    }

    fn invoke_macro(
        &mut self,
        name: (String, <Self as Frontend>::TokenRef),
        args: Vec<Vec<Token>>,
        backend: &mut impl Backend<TCS::TokenRef>,
    ) {
        match self.tokenized_code_source
            .macro_invocation(name.clone(), args)
        {
            Some(tokens) => self.analyze_token_seq(tokens, backend),
            None => {
                let (name, name_ref) = name;
                self.diagnostics
                    .emit_diagnostic(Diagnostic::new(Message::UndefinedMacro { name }, name_ref))
            }
        }
    }
}

impl<'a, SAF, TCS, DL> Frontend for FileParser<'a, SAF, TCS, DL>
where
    SAF: TokenSeqAnalyzerFactory,
    TCS: TokenizedCodeSource,
    for<'b> &'b TCS::Tokenized: IntoIterator<Item = (Token, TCS::TokenRef)>,
    DL: DiagnosticsListener<TCS::TokenRef>,
{
    type TokenRef = TCS::TokenRef;

    fn analyze_chunk(
        &mut self,
        chunk_id: ::ChunkId<Self::TokenRef>,
        backend: &mut impl Backend<Self::TokenRef>,
    ) {
        match chunk_id {
            ::ChunkId::File((name, _)) => self.include_source_file(&name, backend),
            ::ChunkId::Macro { name, args } => self.invoke_macro(name, args, backend),
        }
    }

    fn define_macro(
        &mut self,
        name: (impl Into<String>, Self::TokenRef),
        tokens: Vec<(Token, Self::TokenRef)>,
    ) {
        self.tokenized_code_source.define_macro(name, tokens)
    }
}

use std::{collections::HashMap, rc::Rc};

struct Session<'a, TCS: TokenizedCodeSource, SAF, B: Backend<TCS::TokenRef>, DL: 'a>
where
    for<'b> &'b TCS::Tokenized: IntoIterator<Item = (Token, TCS::TokenRef)>,
{
    analyzer_factory: SAF,
    backend: B,
    section: Option<B::Section>,
    tokenized_code_source: TCS,
    diagnostics: &'a DL,
}

impl<'a, TCS, SAF, B, DL> Session<'a, TCS, SAF, B, DL>
where
    TCS: TokenizedCodeSource,
    for<'b> &'b TCS::Tokenized: IntoIterator<Item = (Token, TCS::TokenRef)>,
    SAF: TokenSeqAnalyzerFactory,
    B: Backend<TCS::TokenRef>,
    DL: DiagnosticsListener<TCS::TokenRef> + 'a,
{
    fn new(
        tokenized_code_source: TCS,
        analyzer_factory: SAF,
        backend: B,
        diagnostics: &DL,
    ) -> Session<TCS, SAF, B, DL> {
        let mut session = Session {
            analyzer_factory,
            backend,
            section: None,
            tokenized_code_source,
            diagnostics,
        };
        session.section = Some(session.backend.mk_section());
        session
    }

    fn analyze_token_seq<I: IntoIterator<Item = (Token, TCS::TokenRef)>>(&mut self, tokens: I) {
        let mut analyzer = self.analyzer_factory.mk_token_seq_analyzer();
        analyzer.analyze(tokens.into_iter(), self)
    }

    fn into_object(self) -> B {
        self.backend
    }

    fn include_source_file(&mut self, filename: &str) {
        let tokenized_src = self.tokenized_code_source.tokenize_file(filename);
        self.analyze_token_seq(&tokenized_src)
    }

    fn invoke_macro(
        &mut self,
        name: (String, <Self as ::AssemblySession>::TokenRef),
        args: Vec<Vec<Token>>,
    ) {
        match self.tokenized_code_source
            .macro_invocation(name.clone(), args)
        {
            Some(tokens) => self.analyze_token_seq(tokens),
            None => {
                let (name, name_ref) = name;
                self.diagnostics
                    .emit_diagnostic(Diagnostic::new(Message::UndefinedMacro { name }, name_ref))
            }
        }
    }
}

impl<'a, TCS, SAF, B, DL> ::AssemblySession for Session<'a, TCS, SAF, B, DL>
where
    TCS: TokenizedCodeSource,
    for<'b> &'b TCS::Tokenized: IntoIterator<Item = (Token, TCS::TokenRef)>,
    SAF: TokenSeqAnalyzerFactory,
    B: Backend<TCS::TokenRef>,
    DL: DiagnosticsListener<TCS::TokenRef> + 'a,
{
    type TokenRef = TCS::TokenRef;

    fn analyze_chunk(&mut self, chunk_id: ::ChunkId<Self::TokenRef>) {
        match chunk_id {
            ::ChunkId::File((name, _)) => self.include_source_file(&name),
            ::ChunkId::Macro { name, args } => self.invoke_macro(name, args),
        }
    }

    fn emit_diagnostic(&mut self, diagnostic: Diagnostic<Self::TokenRef>) {
        self.diagnostics.emit_diagnostic(diagnostic)
    }

    fn emit_item(&mut self, item: Item<Self::TokenRef>) {
        self.section.as_mut().unwrap().emit_item(item)
    }

    fn define_label(&mut self, label: (String, Self::TokenRef)) {
        self.section.as_mut().unwrap().add_label((label.0, label.1))
    }

    fn define_macro(
        &mut self,
        name: (impl Into<String>, Self::TokenRef),
        tokens: Vec<(Token, Self::TokenRef)>,
    ) {
        self.tokenized_code_source.define_macro(name, tokens)
    }
}

trait TokenizedCodeSource
where
    for<'c> &'c Self::Tokenized: IntoIterator<Item = (Token, Self::TokenRef)>,
{
    type TokenRef: Clone + Debug + PartialEq;
    fn define_macro(
        &mut self,
        name: (impl Into<String>, Self::TokenRef),
        tokens: Vec<(Token, Self::TokenRef)>,
    );
    type MacroInvocationIter: Iterator<Item = (Token, Self::TokenRef)>;
    fn macro_invocation(
        &mut self,
        name: (String, Self::TokenRef),
        args: Vec<Vec<Token>>,
    ) -> Option<Self::MacroInvocationIter>;
    type Tokenized;
    fn tokenize_file(&mut self, filename: &str) -> Self::Tokenized;
}

struct TokenStreamSource<'a, C: Codebase + 'a, TT: TokenTracker> {
    codebase: &'a C,
    token_tracker: TT,
    macro_defs: HashMap<String, Rc<MacroDef<TT::TokenRef>>>,
}

type MacroDef<TR> = Vec<(Token, TR)>;

impl<'a, C: Codebase + 'a, TT: TokenTracker> TokenStreamSource<'a, C, TT> {
    fn new(codebase: &'a C, token_tracker: TT) -> TokenStreamSource<C, TT> {
        TokenStreamSource {
            codebase,
            token_tracker,
            macro_defs: HashMap::new(),
        }
    }
}

impl<'a, C: Codebase + 'a, TT: TokenTracker> TokenizedCodeSource for TokenStreamSource<'a, C, TT> {
    type TokenRef = TT::TokenRef;

    fn define_macro(
        &mut self,
        name: (impl Into<String>, TT::TokenRef),
        tokens: Vec<(Token, TT::TokenRef)>,
    ) {
        self.macro_defs.insert(name.0.into(), Rc::new(tokens));
    }

    type MacroInvocationIter = MacroDefIter<TT::TokenRef>;

    fn macro_invocation(
        &mut self,
        name: (String, TT::TokenRef),
        _args: Vec<Vec<Token>>,
    ) -> Option<Self::MacroInvocationIter> {
        self.macro_defs.get(&name.0).cloned().map(MacroDefIter::new)
    }

    type Tokenized = TokenizedSrc<TT::BufContext>;

    fn tokenize_file(&mut self, filename: &str) -> Self::Tokenized {
        let buf_id = self.codebase.open(filename);
        let rc_src = self.codebase.buf(buf_id);
        TokenizedSrc::new(rc_src, self.token_tracker.mk_buf_context(buf_id, None))
    }
}

struct MacroDefIter<CR> {
    tokens: Rc<Vec<(Token, CR)>>,
    index: usize,
}

impl<CR> MacroDefIter<CR> {
    fn new(tokens: Rc<Vec<(Token, CR)>>) -> MacroDefIter<CR> {
        MacroDefIter { tokens, index: 0 }
    }
}

impl<CR: Clone> Iterator for MacroDefIter<CR> {
    type Item = (Token, CR);
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.tokens.len() {
            let item = self.tokens[self.index].clone();
            self.index += 1;
            Some(item)
        } else {
            None
        }
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

    use std::{self, cell::RefCell};

    #[test]
    fn include_source_file() {
        let filename = "my_file.asm";
        let contents = vec![token::Command(Command::Nop)];
        let log = TestLog::default();
        TestFixture::new(&log)
            .given(|f| {
                f.mock_token_source
                    .add_file(filename, add_code_refs(&contents))
            })
            .when(|mut session| session.include_source_file(filename));
        assert_eq!(*log.borrow(), [TestEvent::AnalyzeTokens(contents)]);
    }

    fn add_code_refs<'a, I: IntoIterator<Item = &'a Token>>(tokens: I) -> Vec<(Token, ())> {
        tokens.into_iter().map(|t| (t.clone(), ())).collect()
    }

    #[test]
    fn emit_instruction_item() {
        let item = Item::Instruction(Instruction::Nop);
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

    use frontend::syntax::keyword::Command;

    #[test]
    fn define_and_invoke_macro() {
        let name = "my_macro";
        let tokens = vec![token::Command(Command::Nop)];
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut session| {
            session.define_macro((name.to_string(), ()), add_code_refs(&tokens));
            session.invoke_macro((name.to_string(), ()), vec![])
        });
        assert_eq!(*log.borrow(), [TestEvent::AnalyzeTokens(tokens)]);
    }

    use diagnostics::Diagnostic;

    #[test]
    fn diagnose_undefined_macro() {
        let name = "my_macro".to_string();
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut session| session.invoke_macro((name.clone(), ()), vec![]));
        assert_eq!(
            *log.borrow(),
            [
                TestEvent::Diagnostic(Diagnostic::new(Message::UndefinedMacro { name }, ()))
            ]
        );
    }

    struct MockTokenSource {
        files: HashMap<String, Vec<(Token, ())>>,
        macros: HashMap<String, Vec<(Token, ())>>,
    }

    impl MockTokenSource {
        fn new() -> MockTokenSource {
            MockTokenSource {
                files: HashMap::new(),
                macros: HashMap::new(),
            }
        }

        fn add_file(&mut self, name: &str, tokens: Vec<(Token, ())>) {
            self.files.insert(name.to_string(), tokens);
        }
    }

    impl TokenizedCodeSource for MockTokenSource {
        type TokenRef = ();

        fn define_macro(
            &mut self,
            name: (impl Into<String>, Self::TokenRef),
            tokens: Vec<(Token, Self::TokenRef)>,
        ) {
            self.macros.insert(name.0.into(), tokens);
        }

        type MacroInvocationIter = std::vec::IntoIter<(Token, Self::TokenRef)>;

        fn macro_invocation(
            &mut self,
            name: (String, Self::TokenRef),
            _args: Vec<Vec<Token>>,
        ) -> Option<Self::MacroInvocationIter> {
            self.macros.get(&name.0).cloned().map(|v| v.into_iter())
        }

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

    impl<'a> TokenSeqAnalyzerFactory for Mock<'a> {
        type TokenSeqAnalyzer = Self;
        fn mk_token_seq_analyzer(&mut self) -> Self::TokenSeqAnalyzer {
            self.clone()
        }
    }

    impl<'a> TokenSeqAnalyzer for Mock<'a> {
        fn analyze<I, F>(&mut self, tokens: I, _frontend: &mut F)
        where
            I: Iterator<Item = (Token, F::TokenRef)>,
            F: ::AssemblySession,
        {
            self.log
                .borrow_mut()
                .push(TestEvent::AnalyzeTokens(tokens.map(|(t, _)| t).collect()))
        }
    }

    impl<'a> Backend<()> for Mock<'a> {
        type Section = Self;
        fn mk_section(&mut self) -> Self::Section {
            self.clone()
        }

        fn add_label(&mut self, (label, _): (impl Into<String>, ())) {
            self.log
                .borrow_mut()
                .push(TestEvent::AddLabel(label.into()))
        }

        fn emit_item(&mut self, item: Item<()>) {
            self.log.borrow_mut().push(TestEvent::EmitItem(item))
        }
    }

    impl<'a> Section<()> for Mock<'a> {
        fn add_label(&mut self, (label, _): (impl Into<String>, ())) {
            self.log
                .borrow_mut()
                .push(TestEvent::AddLabel(label.into()))
        }

        fn emit_item(&mut self, item: Item<()>) {
            self.log.borrow_mut().push(TestEvent::EmitItem(item))
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
        mock_token_source: MockTokenSource,
        analyzer_factory: Mock<'a>,
        object: Mock<'a>,
        diagnostics: Mock<'a>,
    }

    impl<'a> TestFixture<'a> {
        fn new(log: &'a TestLog) -> TestFixture<'a> {
            TestFixture {
                mock_token_source: MockTokenSource::new(),
                analyzer_factory: Mock::new(log),
                object: Mock::new(log),
                diagnostics: Mock::new(log),
            }
        }

        fn given<F: FnOnce(&mut Self)>(mut self, f: F) -> Self {
            f(&mut self);
            self
        }

        fn when<F: FnOnce(Session<MockTokenSource, Mock<'a>, Mock<'a>, Mock<'a>>)>(self, f: F) {
            f(Session::new(
                self.mock_token_source,
                self.analyzer_factory,
                self.object,
                &self.diagnostics,
            ))
        }
    }
}
