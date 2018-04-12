use std;

mod semantics;
mod syntax;

use diagnostics::*;
use backend::*;
use self::syntax::*;

pub fn analyze_file<B: Backend>(name: String, backend: B) -> B {
    let fs = StdFileSystem::new();
    let factory = SemanticTokenSeqAnalyzerFactory::new();
    let mut session = Session::new(fs, factory, backend, DebugDiagnosticsListener {});
    session.include_source_file(name);
    session.into_object()
}

struct DebugDiagnosticsListener;

impl DiagnosticsListener for DebugDiagnosticsListener {
    fn emit_diagnostic(&self, diagnostic: Diagnostic) {
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
        I: Iterator<Item = Token>,
        F: Frontend<TrackingData = ()>;
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
        I: Iterator<Item = Token>,
        F: Frontend<TrackingData = ()>,
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
    type TrackingData;
    fn include_source_file(&mut self, filename: String);
    fn emit_item(&mut self, item: Item);
    fn define_label(&mut self, label: String);
    fn define_macro(&mut self, name: String, tokens: Vec<Token>);
    fn invoke_macro(&mut self, name: String, args: Vec<Vec<Token>>);
}

use std::{collections::HashMap, rc::Rc};

struct Session<FS, SAF, B, DL> {
    fs: FS,
    analyzer_factory: SAF,
    backend: B,
    macro_defs: HashMap<String, Rc<Vec<Token>>>,
    diagnostics: DL,
}

impl<FS, SAF, B, DL> Session<FS, SAF, B, DL>
where
    FS: FileSystem,
    SAF: TokenSeqAnalyzerFactory,
    B: Backend,
    DL: DiagnosticsListener,
{
    fn new(fs: FS, analyzer_factory: SAF, backend: B, diagnostics: DL) -> Session<FS, SAF, B, DL> {
        Session {
            fs,
            analyzer_factory,
            backend,
            macro_defs: HashMap::new(),
            diagnostics,
        }
    }

    fn analyze_token_seq<I: Iterator<Item = Token>>(&mut self, tokens: I) {
        let mut analyzer = self.analyzer_factory.mk_token_seq_analyzer();
        analyzer.analyze(tokens, self)
    }

    fn into_object(self) -> B {
        self.backend
    }
}

impl<FS, SAF, B, DL> Frontend for Session<FS, SAF, B, DL>
where
    FS: FileSystem,
    SAF: TokenSeqAnalyzerFactory,
    B: Backend,
    DL: DiagnosticsListener,
{
    type TrackingData = ();

    fn include_source_file(&mut self, filename: String) {
        let src = self.fs.read_file(&filename);
        let tokens = syntax::tokenize(&src);
        self.analyze_token_seq(tokens)
    }

    fn emit_item(&mut self, item: Item) {
        self.backend.emit_item(item)
    }

    fn define_label(&mut self, label: String) {
        self.backend.add_label(&label)
    }

    fn define_macro(&mut self, name: String, tokens: Vec<Token>) {
        self.macro_defs.insert(name, Rc::new(tokens));
    }

    fn invoke_macro(&mut self, name: String, _args: Vec<Vec<Token>>) {
        let macro_def = self.macro_defs.get(&name).cloned();
        match macro_def {
            Some(rc) => self.analyze_token_seq(rc.iter().cloned()),
            None => self.diagnostics
                .emit_diagnostic(Diagnostic::UndefinedMacro { name }),
        }
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
        TestFixture::new(&log).when(|mut session| session.define_label(label.to_string()));
        assert_eq!(*log.borrow(), [TestEvent::AddLabel(String::from(label))]);
    }

    use frontend::syntax::keyword::Command;

    #[test]
    fn define_and_invoke_macro() {
        let name = "my_macro";
        let tokens = vec![token::Command(Command::Nop)];
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut session| {
            session.define_macro(name.to_string(), tokens.clone());
            session.invoke_macro(name.to_string(), vec![])
        });
        assert_eq!(*log.borrow(), [TestEvent::AnalyzeTokens(tokens)]);
    }

    use diagnostics::Diagnostic;

    #[test]
    fn diagnose_undefined_macro() {
        let name = "my_macro";
        let log = TestLog::default();
        TestFixture::new(&log).when(|mut session| session.invoke_macro(name.to_string(), vec![]));
        assert_eq!(
            *log.borrow(),
            [
                TestEvent::Diagnostic(Diagnostic::UndefinedMacro {
                    name: name.to_string(),
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
            I: Iterator<Item = Token>,
            F: Frontend,
        {
            self.log
                .borrow_mut()
                .push(TestEvent::AnalyzeTokens(tokens.collect()))
        }
    }

    impl<'a> Backend for Mock<'a> {
        fn add_label(&mut self, label: &str) {
            self.log
                .borrow_mut()
                .push(TestEvent::AddLabel(String::from(label)))
        }

        fn emit_item(&mut self, item: Item) {
            self.log.borrow_mut().push(TestEvent::EmitItem(item))
        }
    }

    impl<'a> DiagnosticsListener for Mock<'a> {
        fn emit_diagnostic(&self, diagnostic: Diagnostic) {
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
        Diagnostic(Diagnostic),
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

        fn when<F: FnOnce(Session<MockFileSystem<'a>, Mock<'a>, Mock<'a>, Mock<'a>>)>(self, f: F) {
            f(Session::from(self))
        }
    }

    impl<'a> From<TestFixture<'a>> for Session<MockFileSystem<'a>, Mock<'a>, Mock<'a>, Mock<'a>> {
        fn from(
            fixture: TestFixture<'a>,
        ) -> Session<MockFileSystem<'a>, Mock<'a>, Mock<'a>, Mock<'a>> {
            Session::new(
                fixture.fs,
                fixture.analyzer_factory,
                fixture.object,
                fixture.diagnostics,
            )
        }
    }
}
