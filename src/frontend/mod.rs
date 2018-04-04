use ir;

use std;

#[cfg(test)]
mod codebase;
mod semantics;
mod syntax;

use ir::*;
use self::syntax::*;

pub fn analyze_file<S: ir::Section>(name: String, mut section: S) {
    let mut fs = StdFileSystem::new();
    let mut factory = SemanticTokenSeqAnalyzerFactory::new();
    let mut session = Session::new(&mut fs, &mut factory, &mut section);
    session.include_source_file(name);
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
    fn analyze<OR: OperationReceiver>(&mut self, src: String, receiver: &mut OR);
}

struct SemanticTokenSeqAnalyzer;

impl SemanticTokenSeqAnalyzer {
    fn new() -> SemanticTokenSeqAnalyzer {
        SemanticTokenSeqAnalyzer {}
    }
}

impl TokenSeqAnalyzer for SemanticTokenSeqAnalyzer {
    fn analyze<OR: OperationReceiver>(&mut self, src: String, receiver: &mut OR) {
        let actions = semantics::SemanticActions::new(receiver);
        let tokens = syntax::tokenize(&src);
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
    fn mk_atom(&mut self, token: Token<String>) -> Expr;
}

pub struct StrExprFactory;

impl StrExprFactory {
    fn new() -> StrExprFactory {
        StrExprFactory {}
    }
}

impl ExprFactory for StrExprFactory {
    fn mk_atom(&mut self, token: Token<String>) -> Expr {
        match token {
            Token::Atom(Atom::Ident(ident)) => Expr::Symbol(ident),
            Token::Atom(Atom::Number(number)) => Expr::Literal(number),
            _ => panic!(),
        }
    }
}

pub trait OperationReceiver {
    fn include_source_file(&mut self, filename: String);
    fn emit_instruction(&mut self, instruction: ir::Instruction);
    fn define_label(&mut self, label: String);
}

struct Session<'session, FS: 'session, SAF: 'session, S: 'session> {
    fs: &'session mut FS,
    analyzer_factory: &'session mut SAF,
    section: &'session mut S,
}

impl<'session, FS: FileSystem, SAF: TokenSeqAnalyzerFactory, S: ir::Section>
    Session<'session, FS, SAF, S>
{
    fn new(
        fs: &'session mut FS,
        analyzer_factory: &'session mut SAF,
        section: &'session mut S,
    ) -> Session<'session, FS, SAF, S> {
        Session {
            fs,
            analyzer_factory,
            section,
        }
    }
}

impl<'session, FS, SAF, S> OperationReceiver for Session<'session, FS, SAF, S>
where
    FS: FileSystem,
    SAF: TokenSeqAnalyzerFactory,
    S: ir::Section,
{
    fn include_source_file(&mut self, filename: String) {
        let src = self.fs.read_file(&filename);
        let mut analyzer = self.analyzer_factory.mk_token_seq_analyzer();
        analyzer.analyze(src, self)
    }

    fn emit_instruction(&mut self, instruction: ir::Instruction) {
        self.section.add_instruction(instruction)
    }

    fn define_label(&mut self, label: String) {
        self.section.add_label(&label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn include_source_file() {
        let log = TestLog::default();
        let filename = "my_file.asm";
        let contents = "nop";
        let mut fs = MockFileSystem::new();
        fs.add_file(filename, contents);
        let mut analyzer_factory = MockSrcAnalyzerFactory::new();
        let mut section = MockSection::new(&log);
        {
            let mut session = Session::new(&mut fs, &mut analyzer_factory, &mut section);
            session.include_source_file(filename.to_string())
        }
        assert_eq!(
            Rc::try_unwrap(analyzer_factory.src).unwrap().into_inner(),
            [contents]
        )
    }

    #[test]
    fn emit_instruction() {
        let log = TestLog::default();
        let mut fs = MockFileSystem::new();
        let mut analyzer_factory = MockSrcAnalyzerFactory::new();
        let mut section = MockSection::new(&log);
        let instruction = Instruction::Nop;
        {
            let mut session = Session::new(&mut fs, &mut analyzer_factory, &mut section);
            session.emit_instruction(instruction.clone())
        }
        assert_eq!(
            *log.borrow(),
            [MockSectionOperation::AddInstruction(instruction)]
        );
    }

    #[test]
    fn define_label() {
        let log = TestLog::default();
        let mut fs = MockFileSystem::new();
        let mut analyzer_factory = MockSrcAnalyzerFactory::new();
        let mut section = MockSection::new(&log);
        let label = "label";
        {
            let mut session = Session::new(&mut fs, &mut analyzer_factory, &mut section);
            session.define_label(label.to_string())
        }
        assert_eq!(
            *log.borrow(),
            [MockSectionOperation::AddLabel(String::from(label))]
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

    use std::{cell::RefCell, rc::Rc};

    struct MockSrcAnalyzerFactory {
        src: Rc<RefCell<Vec<String>>>,
    }

    impl MockSrcAnalyzerFactory {
        fn new() -> MockSrcAnalyzerFactory {
            MockSrcAnalyzerFactory {
                src: Rc::new(RefCell::new(Vec::new())),
            }
        }
    }

    impl TokenSeqAnalyzerFactory for MockSrcAnalyzerFactory {
        type TokenSeqAnalyzer = MockSrcAnalyzer;

        fn mk_token_seq_analyzer(&mut self) -> Self::TokenSeqAnalyzer {
            MockSrcAnalyzer::new(self.src.clone())
        }
    }

    struct MockSrcAnalyzer {
        src: Rc<RefCell<Vec<String>>>,
    }

    impl MockSrcAnalyzer {
        fn new(src: Rc<RefCell<Vec<String>>>) -> MockSrcAnalyzer {
            MockSrcAnalyzer { src }
        }
    }

    impl TokenSeqAnalyzer for MockSrcAnalyzer {
        fn analyze<OR: OperationReceiver>(&mut self, src: String, _receiver: &mut OR) {
            self.src.borrow_mut().push(src)
        }
    }

    #[derive(Debug, PartialEq)]
    enum MockSectionOperation {
        AddInstruction(Instruction),
        AddLabel(String),
    }

    struct MockSection<'a, E: 'a> {
        log: &'a TestLog<E>,
    }

    impl<'a, E: 'a> MockSection<'a, E> {
        fn new(log: &'a TestLog<E>) -> MockSection<'a, E> {
            MockSection {
                log,
            }
        }
    }

    impl<'a> ir::Section for MockSection<'a, MockSectionOperation> {
        fn add_instruction(&mut self, instruction: Instruction) {
            self.log.borrow_mut()
                .push(MockSectionOperation::AddInstruction(instruction).into())
        }

        fn add_label(&mut self, label: &str) {
            self.log.borrow_mut()
                .push(MockSectionOperation::AddLabel(String::from(label)).into())
        }
    }

    type TestLog<E> = RefCell<Vec<E>>;
}
