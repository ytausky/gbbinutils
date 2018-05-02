use std;

mod semantics;
mod syntax;

use codebase::StringCodebase;
use diagnostics::*;
use backend::*;
use self::syntax::*;

pub fn analyze_file<B: Backend<(BufId, BufRange)>>(name: String, backend: B) -> B {
    let fs = StdFileSystem::new();
    let factory = SemanticTokenSeqAnalyzerFactory::new();
    let token_provider = TokenStreamSource::new(fs, TrivialCodeRefFactory {});
    let mut session = Session::new(token_provider, factory, backend, DiagnosticsDumper::new());
    session.analyze_chunk(ChunkId::File((name, None)));
    session.into_object()
}

use codebase::{BufId, BufRange};

trait CodeRefFactory
where
    Self: Clone,
{
    type CodeRef: Clone;
    fn mk_code_ref(&self, buf_id: BufId, buf_range: BufRange) -> Self::CodeRef;
}

#[derive(Clone)]
struct TrivialCodeRefFactory;

impl CodeRefFactory for TrivialCodeRefFactory {
    type CodeRef = (BufId, BufRange);
    fn mk_code_ref(&self, buf_id: BufId, buf_range: BufRange) -> Self::CodeRef {
        (buf_id, buf_range)
    }
}

trait FileSystem {
    fn read_file(&mut self, filename: &str) -> String;
}

struct StdFileSystem;

impl StdFileSystem {
    fn new() -> StdFileSystem {
        StdFileSystem {}
    }
}

impl FileSystem for StdFileSystem {
    fn read_file(&mut self, filename: &str) -> String {
        use std::io::prelude::*;
        let mut file = std::fs::File::open(filename).unwrap();
        let mut src = String::new();
        file.read_to_string(&mut src).unwrap();
        src
    }
}

trait TokenSeqAnalyzer {
    fn analyze<I, F>(&mut self, tokens: I, frontend: &mut F)
    where
        I: Iterator<Item = (Token, F::CodeRef)>,
        F: Frontend;
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
        I: Iterator<Item = (Token, F::CodeRef)>,
        F: Frontend,
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
    fn mk_atom(&mut self, token: Token) -> Expr;
}

pub struct StrExprFactory;

impl StrExprFactory {
    fn new() -> StrExprFactory {
        StrExprFactory {}
    }
}

impl ExprFactory for StrExprFactory {
    fn mk_atom(&mut self, token: Token) -> Expr {
        match token {
            token::Atom(Atom::Ident(ident)) => Expr::Symbol(ident),
            token::Atom(Atom::Number(number)) => Expr::Literal(number),
            _ => panic!(),
        }
    }
}

pub trait Frontend {
    type CodeRef;
    fn analyze_chunk(&mut self, chunk_id: ChunkId<Self::CodeRef>);
    fn emit_item(&mut self, item: Item);
    fn define_label(&mut self, label: (String, Self::CodeRef));
    fn define_macro(&mut self, name: (String, Self::CodeRef), tokens: Vec<(Token, Self::CodeRef)>);
}

#[derive(Debug, PartialEq)]
pub enum ChunkId<T> {
    File((String, Option<T>)),
    Macro {
        name: (String, T),
        args: Vec<Vec<Token>>,
    },
}

use std::{collections::HashMap, rc::Rc};

struct Session<TCS, SAF, B, DL> {
    analyzer_factory: SAF,
    backend: B,
    tokenized_code_source: TCS,
    diagnostics: DL,
}

impl<TCS, SAF, B, DL> Session<TCS, SAF, B, DL>
where
    TCS: TokenizedCodeSource,
    for<'a> &'a TCS::Tokenized: IntoIterator<Item = (Token, TCS::CodeRef)>,
    SAF: TokenSeqAnalyzerFactory,
    B: Backend<TCS::CodeRef>,
    DL: DiagnosticsListener<TCS::CodeRef>,
{
    fn new(
        tokenized_code_source: TCS,
        analyzer_factory: SAF,
        backend: B,
        diagnostics: DL,
    ) -> Session<TCS, SAF, B, DL> {
        Session {
            analyzer_factory,
            backend,
            tokenized_code_source,
            diagnostics,
        }
    }

    fn analyze_token_seq<I: IntoIterator<Item = (Token, TCS::CodeRef)>>(&mut self, tokens: I) {
        let mut analyzer = self.analyzer_factory.mk_token_seq_analyzer();
        analyzer.analyze(tokens.into_iter(), self)
    }

    fn into_object(self) -> B {
        self.backend
    }

    fn include_source_file(&mut self, filename: String) {
        let tokenized_src = self.tokenized_code_source.tokenize_file(&filename);
        self.analyze_token_seq(&tokenized_src)
    }

    fn invoke_macro(&mut self, name: (String, <Self as Frontend>::CodeRef), args: Vec<Vec<Token>>) {
        match self.tokenized_code_source
            .macro_invocation(name.clone(), args)
        {
            Some(tokens) => self.analyze_token_seq(tokens),
            None => self.diagnostics
                .emit_diagnostic(Diagnostic::UndefinedMacro { name }),
        }
    }
}

impl<TCS, SAF, B, DL> Frontend for Session<TCS, SAF, B, DL>
where
    TCS: TokenizedCodeSource,
    for<'a> &'a TCS::Tokenized: IntoIterator<Item = (Token, TCS::CodeRef)>,
    SAF: TokenSeqAnalyzerFactory,
    B: Backend<TCS::CodeRef>,
    DL: DiagnosticsListener<TCS::CodeRef>,
{
    type CodeRef = TCS::CodeRef;

    fn analyze_chunk(&mut self, chunk_id: ChunkId<Self::CodeRef>) {
        match chunk_id {
            ChunkId::File((name, _)) => self.include_source_file(name),
            ChunkId::Macro { name, args } => self.invoke_macro(name, args),
        }
    }

    fn emit_item(&mut self, item: Item) {
        self.backend.emit_item(item)
    }

    fn define_label(&mut self, label: (String, Self::CodeRef)) {
        self.backend.add_label((&label.0, label.1))
    }

    fn define_macro(&mut self, name: (String, Self::CodeRef), tokens: Vec<(Token, Self::CodeRef)>) {
        self.tokenized_code_source.define_macro(name, tokens)
    }
}

trait TokenizedCodeSource
where
    for<'c> &'c Self::Tokenized: IntoIterator<Item = (Token, Self::CodeRef)>,
{
    type CodeRef: Clone;
    fn define_macro(&mut self, name: (String, Self::CodeRef), tokens: Vec<(Token, Self::CodeRef)>);
    type MacroInvocationIter: Iterator<Item = (Token, Self::CodeRef)>;
    fn macro_invocation(
        &mut self,
        name: (String, Self::CodeRef),
        args: Vec<Vec<Token>>,
    ) -> Option<Self::MacroInvocationIter>;
    type Tokenized;
    fn tokenize_file(&mut self, filename: &str) -> Self::Tokenized;
}

struct TokenStreamSource<CRF: CodeRefFactory, FS> {
    fs: FS,
    codebase: StringCodebase,
    code_ref_factory: CRF,
    macro_defs: HashMap<String, Rc<Vec<(Token, CRF::CodeRef)>>>,
}

impl<CRF: CodeRefFactory, FS: FileSystem> TokenStreamSource<CRF, FS> {
    fn new(fs: FS, code_ref_factory: CRF) -> TokenStreamSource<CRF, FS> {
        TokenStreamSource {
            fs,
            codebase: StringCodebase::new(),
            code_ref_factory,
            macro_defs: HashMap::new(),
        }
    }
}

impl<CRF: CodeRefFactory, FS: FileSystem> TokenizedCodeSource for TokenStreamSource<CRF, FS> {
    type CodeRef = CRF::CodeRef;

    fn define_macro(&mut self, name: (String, CRF::CodeRef), tokens: Vec<(Token, CRF::CodeRef)>) {
        self.macro_defs.insert(name.0, Rc::new(tokens));
    }

    type MacroInvocationIter = MacroDefIter<CRF::CodeRef>;

    fn macro_invocation(
        &mut self,
        name: (String, CRF::CodeRef),
        _args: Vec<Vec<Token>>,
    ) -> Option<Self::MacroInvocationIter> {
        self.macro_defs.get(&name.0).cloned().map(MacroDefIter::new)
    }

    type Tokenized = TokenizedSrc<CRF>;

    fn tokenize_file(&mut self, filename: &str) -> Self::Tokenized {
        let src = self.fs.read_file(&filename);
        let buf_id = self.codebase.add_src_buf(src);
        let rc_src = self.codebase.buf(buf_id).text();
        TokenizedSrc::new(buf_id, rc_src, self.code_ref_factory.clone())
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

struct TokenizedSrc<CRF> {
    buf_id: BufId,
    src: Rc<str>,
    code_ref_factory: CRF,
}

impl<CRF: CodeRefFactory> TokenizedSrc<CRF> {
    fn new(buf_id: BufId, src: Rc<str>, code_ref_factory: CRF) -> TokenizedSrc<CRF> {
        TokenizedSrc {
            buf_id,
            src,
            code_ref_factory,
        }
    }
}

impl<'a, CRF: CodeRefFactory> IntoIterator for &'a TokenizedSrc<CRF> {
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = TokenizedSrcIter<'a, CRF>;
    fn into_iter(self) -> Self::IntoIter {
        TokenizedSrcIter {
            buf_id: self.buf_id,
            tokens: syntax::tokenize(&self.src),
            code_ref_factory: &self.code_ref_factory,
        }
    }
}

struct TokenizedSrcIter<'a, CRF: CodeRefFactory + 'a> {
    buf_id: BufId,
    tokens: syntax::lexer::Lexer<'a>,
    code_ref_factory: &'a CRF,
}

impl<'a, CRF: CodeRefFactory> Iterator for TokenizedSrcIter<'a, CRF> {
    type Item = (Token, CRF::CodeRef);
    fn next(&mut self) -> Option<Self::Item> {
        self.tokens
            .next()
            .map(|(t, r)| (t, self.code_ref_factory.mk_code_ref(self.buf_id, r)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;

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
            .when(|mut session| session.include_source_file(filename.to_string()));
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
        let name = "my_macro";
        let log = TestLog::default();
        TestFixture::new(&log)
            .when(|mut session| session.invoke_macro((name.to_string(), ()), vec![]));
        assert_eq!(
            *log.borrow(),
            [
                TestEvent::Diagnostic(Diagnostic::UndefinedMacro {
                    name: (name.to_string(), ()),
                })
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
        type CodeRef = ();

        fn define_macro(
            &mut self,
            name: (String, Self::CodeRef),
            tokens: Vec<(Token, Self::CodeRef)>,
        ) {
            self.macros.insert(name.0, tokens);
        }

        type MacroInvocationIter = std::vec::IntoIter<(Token, Self::CodeRef)>;

        fn macro_invocation(
            &mut self,
            name: (String, Self::CodeRef),
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
            I: Iterator<Item = (Token, F::CodeRef)>,
            F: Frontend,
        {
            self.log
                .borrow_mut()
                .push(TestEvent::AnalyzeTokens(tokens.map(|(t, _)| t).collect()))
        }
    }

    impl<'a> Backend<()> for Mock<'a> {
        fn add_label(&mut self, (label, _): (&str, ())) {
            self.log
                .borrow_mut()
                .push(TestEvent::AddLabel(String::from(label)))
        }

        fn emit_item(&mut self, item: Item) {
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
        EmitItem(Item),
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
            f(Session::from(self))
        }
    }

    impl<'a> From<TestFixture<'a>> for Session<MockTokenSource, Mock<'a>, Mock<'a>, Mock<'a>> {
        fn from(
            fixture: TestFixture<'a>,
        ) -> Session<MockTokenSource, Mock<'a>, Mock<'a>, Mock<'a>> {
            Session::new(
                fixture.mock_token_source,
                fixture.analyzer_factory,
                fixture.object,
                fixture.diagnostics,
            )
        }
    }
}
