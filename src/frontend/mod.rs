use std::{self, fmt};

mod semantics;
mod syntax;

use codebase::StringCodebase;
use diagnostics::*;
use backend::*;
use self::syntax::*;

pub fn analyze_file<B: Backend<()>>(name: String, backend: B) -> B {
    let fs = StdFileSystem::new();
    let factory = SemanticTokenSeqAnalyzerFactory::new();
    let mut session = Session::new(
        NullCodeRefFactory {},
        fs,
        factory,
        backend,
        DebugDiagnosticsListener {},
    );
    session.include_source_file(name);
    session.into_object()
}

trait CodeRefFactory
where
    Self: Clone,
{
    type CodeRef: Clone;
    fn mk_code_ref(&self, byte_range: std::ops::Range<usize>) -> Self::CodeRef;
}

#[derive(Clone)]
struct NullCodeRefFactory;

impl CodeRefFactory for NullCodeRefFactory {
    type CodeRef = ();
    fn mk_code_ref(&self, _byte_range: std::ops::Range<usize>) -> Self::CodeRef {
        ()
    }
}

struct DebugDiagnosticsListener;

impl<R: fmt::Debug> DiagnosticsListener<R> for DebugDiagnosticsListener {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<R>) {
        println!("Diagnostic: {:?}", diagnostic)
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
    fn include_source_file(&mut self, filename: String);
    fn emit_item(&mut self, item: Item);
    fn define_label(&mut self, label: (String, Self::CodeRef));
    fn define_macro(&mut self, name: (String, Self::CodeRef), tokens: Vec<(Token, Self::CodeRef)>);
    fn invoke_macro(&mut self, name: (String, Self::CodeRef), args: Vec<Vec<Token>>);
}

use std::{collections::HashMap, rc::Rc};

struct Session<CRF: CodeRefFactory, FS, SAF, B, DL> {
    analyzer_factory: SAF,
    backend: B,
    token_stream_source: TokenStreamSource<CRF, FS>,
    diagnostics: DL,
}

impl<CRF, FS, SAF, B, DL> Session<CRF, FS, SAF, B, DL>
where
    CRF: CodeRefFactory,
    FS: FileSystem,
    SAF: TokenSeqAnalyzerFactory,
    B: Backend<CRF::CodeRef>,
    DL: DiagnosticsListener<CRF::CodeRef>,
{
    fn new(
        code_ref_factory: CRF,
        fs: FS,
        analyzer_factory: SAF,
        backend: B,
        diagnostics: DL,
    ) -> Session<CRF, FS, SAF, B, DL> {
        Session {
            analyzer_factory,
            backend,
            token_stream_source: TokenStreamSource::new(fs, code_ref_factory),
            diagnostics,
        }
    }

    fn analyze_token_seq<I: Iterator<Item = (Token, CRF::CodeRef)>>(&mut self, tokens: I) {
        let mut analyzer = self.analyzer_factory.mk_token_seq_analyzer();
        analyzer.analyze(tokens, self)
    }

    fn into_object(self) -> B {
        self.backend
    }
}

impl<CRF, FS, SAF, B, DL> Frontend for Session<CRF, FS, SAF, B, DL>
where
    CRF: CodeRefFactory,
    FS: FileSystem,
    SAF: TokenSeqAnalyzerFactory,
    B: Backend<CRF::CodeRef>,
    DL: DiagnosticsListener<CRF::CodeRef>,
{
    type CodeRef = CRF::CodeRef;

    fn include_source_file(&mut self, filename: String) {
        let tokenized_src = self.token_stream_source.tokenize_file(&filename);
        self.analyze_token_seq(tokenized_src.iter())
    }

    fn emit_item(&mut self, item: Item) {
        self.backend.emit_item(item)
    }

    fn define_label(&mut self, label: (String, Self::CodeRef)) {
        self.backend.add_label((&label.0, label.1))
    }

    fn define_macro(&mut self, name: (String, Self::CodeRef), tokens: Vec<(Token, Self::CodeRef)>) {
        self.token_stream_source.define_macro(name, tokens)
    }

    fn invoke_macro(&mut self, name: (String, Self::CodeRef), args: Vec<Vec<Token>>) {
        match self.token_stream_source
            .macro_invocation(name.clone(), args)
        {
            Some(tokens) => self.analyze_token_seq(tokens),
            None => self.diagnostics
                .emit_diagnostic(Diagnostic::UndefinedMacro { name }),
        }
    }
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

    fn define_macro(&mut self, name: (String, CRF::CodeRef), tokens: Vec<(Token, CRF::CodeRef)>) {
        self.macro_defs.insert(name.0, Rc::new(tokens));
    }

    fn macro_invocation(
        &mut self,
        name: (String, CRF::CodeRef),
        _args: Vec<Vec<Token>>,
    ) -> Option<MacroDefIter<CRF::CodeRef>> {
        self.macro_defs.get(&name.0).cloned().map(MacroDefIter::new)
    }

    fn tokenize_file(&mut self, filename: &str) -> TokenizedSrc<CRF> {
        let src = self.fs.read_file(&filename);
        let buf_id = self.codebase.add_src_buf(src);
        let rc_src = self.codebase.buf(buf_id);
        TokenizedSrc::new(rc_src, self.code_ref_factory.clone())
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
    src: Rc<str>,
    code_ref_factory: CRF,
}

impl<CRF: CodeRefFactory> TokenizedSrc<CRF> {
    fn new(src: Rc<str>, code_ref_factory: CRF) -> TokenizedSrc<CRF> {
        TokenizedSrc {
            src,
            code_ref_factory,
        }
    }

    fn iter(&self) -> TokenizedSrcIter<CRF> {
        TokenizedSrcIter {
            tokens: syntax::tokenize(&self.src),
            code_ref_factory: &self.code_ref_factory,
        }
    }
}

struct TokenizedSrcIter<'a, CRF: 'a> {
    tokens: syntax::lexer::Lexer<'a>,
    code_ref_factory: &'a CRF,
}

impl<'a, CRF: CodeRefFactory + 'a> Iterator for TokenizedSrcIter<'a, CRF> {
    type Item = (Token, CRF::CodeRef);
    fn next(&mut self) -> Option<Self::Item> {
        self.tokens
            .next()
            .map(|(t, r)| (t, self.code_ref_factory.mk_code_ref(r)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;

    #[test]
    fn include_source_file() {
        let filename = "my_file.asm";
        let contents = "nop";
        let log = TestLog::default();
        TestFixture::new(&log)
            .given(|f| f.fs.add_file(filename, contents))
            .when(|mut session| session.include_source_file(filename.to_string()));
        assert_eq!(
            *log.borrow(),
            [
                TestEvent::AnalyzeTokens([token::Command(Command::Nop)].to_vec())
            ]
        );
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
            session.define_macro(
                (name.to_string(), ()),
                tokens.iter().cloned().map(|t| (t, ())).collect(),
            );
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

    struct MockFileSystem<'a> {
        files: Vec<(&'a str, &'a str)>,
    }

    impl<'a> MockFileSystem<'a> {
        fn new() -> MockFileSystem<'a> {
            MockFileSystem { files: Vec::new() }
        }

        fn add_file(&mut self, filename: &'a str, contents: &'a str) {
            self.files.push((filename, contents))
        }
    }

    impl<'a> FileSystem for MockFileSystem<'a> {
        fn read_file(&mut self, filename: &str) -> String {
            self.files
                .iter()
                .find(|&&(f, _)| f == filename)
                .map(|&(_, c)| String::from(c))
                .unwrap()
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
        fs: MockFileSystem<'a>,
        analyzer_factory: Mock<'a>,
        object: Mock<'a>,
        diagnostics: Mock<'a>,
    }

    impl<'a> TestFixture<'a> {
        fn new(log: &'a TestLog) -> TestFixture<'a> {
            TestFixture {
                fs: MockFileSystem::new(),
                analyzer_factory: Mock::new(log),
                object: Mock::new(log),
                diagnostics: Mock::new(log),
            }
        }

        fn given<F: FnOnce(&mut Self)>(mut self, f: F) -> Self {
            f(&mut self);
            self
        }

        fn when<
            F: FnOnce(Session<NullCodeRefFactory, MockFileSystem<'a>, Mock<'a>, Mock<'a>, Mock<'a>>),
        >(
            self,
            f: F,
        ) {
            f(Session::from(self))
        }
    }

    impl<'a> From<TestFixture<'a>>
        for Session<NullCodeRefFactory, MockFileSystem<'a>, Mock<'a>, Mock<'a>, Mock<'a>>
    {
        fn from(
            fixture: TestFixture<'a>,
        ) -> Session<NullCodeRefFactory, MockFileSystem<'a>, Mock<'a>, Mock<'a>, Mock<'a>> {
            Session::new(
                NullCodeRefFactory {},
                fixture.fs,
                fixture.analyzer_factory,
                fixture.object,
                fixture.diagnostics,
            )
        }
    }
}
