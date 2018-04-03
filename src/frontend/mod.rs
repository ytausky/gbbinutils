use ir;

use std::{self, marker::PhantomData};

#[cfg(test)]
mod codebase;
mod semantics;
mod syntax;

use ir::*;
use self::syntax::*;

pub fn analyze_file<S: ir::Section>(name: &str, mut section: S) {
    let mut fs = StdFileSystem::new();
    let mut factory = SemanticSrcAnalyzerFactory::new();
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

trait SrcAnalyzer {
    fn analyze<'src, OR: OperationReceiver<'src>>(&mut self, src: &'src str, receiver: &mut OR);
}

struct SemanticSrcAnalyzer;

impl SemanticSrcAnalyzer {
    fn new() -> SemanticSrcAnalyzer {
        SemanticSrcAnalyzer {}
    }
}

impl SrcAnalyzer for SemanticSrcAnalyzer {
    fn analyze<'src, OR: OperationReceiver<'src>>(&mut self, src: &'src str, receiver: &mut OR) {
        let actions = semantics::SemanticActions::new(receiver);
        syntax::parse(src, actions)
    }
}

trait SrcAnalyzerFactory {
    type SrcAnalyzer: SrcAnalyzer;
    fn mk_src_analyzer(&mut self) -> Self::SrcAnalyzer;
}

struct SemanticSrcAnalyzerFactory;

impl SemanticSrcAnalyzerFactory {
    fn new() -> SemanticSrcAnalyzerFactory {
        SemanticSrcAnalyzerFactory {}
    }
}

impl SrcAnalyzerFactory for SemanticSrcAnalyzerFactory {
    type SrcAnalyzer = SemanticSrcAnalyzer;

    fn mk_src_analyzer(&mut self) -> Self::SrcAnalyzer {
        SemanticSrcAnalyzer::new()
    }
}

pub trait ExprFactory {
    type String;
    fn mk_atom(&mut self, token: StrToken<Self::String>) -> Expr;
}

pub struct StrExprFactory<'a>(PhantomData<&'a ()>);

impl<'a> StrExprFactory<'a> {
    fn new() -> StrExprFactory<'a> {
        StrExprFactory(PhantomData)
    }
}

impl<'a> ExprFactory for StrExprFactory<'a> {
    type String = &'a str;
    fn mk_atom(&mut self, token: StrToken<Self::String>) -> Expr {
        match token {
            StrToken::Atom(Atom::Ident(ident)) => Expr::Symbol(ident.to_string()),
            StrToken::Atom(Atom::Number(number)) => Expr::Literal(number),
            _ => panic!(),
        }
    }
}

pub trait OperationReceiver<'src> {
    fn include_source_file(&mut self, filename: &'src str);
    fn emit_instruction(&mut self, instruction: ir::Instruction);
    fn define_label(&mut self, label: &'src str);
}

struct Session<'session, FS: 'session, SAF: 'session, S: 'session> {
    fs: &'session mut FS,
    analyzer_factory: &'session mut SAF,
    section: &'session mut S,
}

impl<'session, FS: FileSystem, SAF: SrcAnalyzerFactory, S: ir::Section>
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

impl<'src, 'session, FS, SAF, S> OperationReceiver<'src> for Session<'session, FS, SAF, S>
where
    'session: 'src,
    FS: FileSystem,
    SAF: SrcAnalyzerFactory,
    S: ir::Section,
{
    fn include_source_file(&mut self, filename: &'src str) {
        let src = self.fs.read_file(filename);
        let mut analyzer = self.analyzer_factory.mk_src_analyzer();
        analyzer.analyze(&src, self)
    }

    fn emit_instruction(&mut self, instruction: ir::Instruction) {
        self.section.add_instruction(instruction)
    }

    fn define_label(&mut self, label: &'src str) {
        self.section.add_label(label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn include_source_file() {
        let filename = "my_file.asm";
        let contents = "nop";
        let mut fs = MockFileSystem::new();
        fs.add_file(filename, contents);
        let mut analyzer_factory = MockSrcAnalyzerFactory::new();
        let mut section = MockSection::new();
        {
            let mut session = Session::new(&mut fs, &mut analyzer_factory, &mut section);
            session.include_source_file(filename);
        }
        assert_eq!(
            Rc::try_unwrap(analyzer_factory.src).unwrap().into_inner(),
            [contents]
        )
    }

    #[test]
    fn emit_instruction() {
        let mut fs = MockFileSystem::new();
        let mut analyzer_factory = MockSrcAnalyzerFactory::new();
        let mut section = MockSection::new();
        let instruction = Instruction::Nop;
        {
            let mut session = Session::new(&mut fs, &mut analyzer_factory, &mut section);
            session.emit_instruction(instruction.clone())
        }
        assert_eq!(
            section.operations,
            [MockSectionOperation::AddInstruction(instruction)]
        )
    }

    #[test]
    fn define_label() {
        let mut fs = MockFileSystem::new();
        let mut analyzer_factory = MockSrcAnalyzerFactory::new();
        let mut section = MockSection::new();
        let label = "label";
        {
            let mut session = Session::new(&mut fs, &mut analyzer_factory, &mut section);
            session.define_label(label)
        }
        assert_eq!(
            section.operations,
            [MockSectionOperation::AddLabel(String::from(label))]
        )
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

    impl SrcAnalyzerFactory for MockSrcAnalyzerFactory {
        type SrcAnalyzer = MockSrcAnalyzer;

        fn mk_src_analyzer(&mut self) -> Self::SrcAnalyzer {
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

    impl SrcAnalyzer for MockSrcAnalyzer {
        fn analyze<'src, OR: OperationReceiver<'src>>(
            &mut self,
            src: &'src str,
            _receiver: &mut OR,
        ) {
            self.src.borrow_mut().push(String::from(src))
        }
    }

    #[derive(Debug, PartialEq)]
    enum MockSectionOperation {
        AddInstruction(Instruction),
        AddLabel(String),
    }

    struct MockSection {
        operations: Vec<MockSectionOperation>,
    }

    impl MockSection {
        fn new() -> MockSection {
            MockSection {
                operations: Vec::new(),
            }
        }
    }

    impl ir::Section for MockSection {
        fn add_instruction(&mut self, instruction: Instruction) {
            self.operations
                .push(MockSectionOperation::AddInstruction(instruction))
        }

        fn add_label(&mut self, label: &str) {
            self.operations
                .push(MockSectionOperation::AddLabel(String::from(label)))
        }
    }
}
