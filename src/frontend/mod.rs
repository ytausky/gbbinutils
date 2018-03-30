use ir;

use std::{self, marker::PhantomData};

mod semantics;
mod syntax;

use ir::*;
use self::syntax::*;

pub fn analyze_file<S: ir::Section>(name: &str, section: S) {
    let mut fs = StdFileSystem::new();
    let mut factory = SemanticSrcAnalyzerFactory::new();
    let mut session = Session::new(&mut fs, &mut factory, section);
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
    type Token: Token;
    fn mk_atom(&mut self, token: Self::Token) -> Expr;
}

pub struct StrExprFactory<'a>(PhantomData<&'a ()>);

impl<'a> StrExprFactory<'a> {
    fn new() -> StrExprFactory<'a> {
        StrExprFactory(PhantomData)
    }
}

impl<'a> ExprFactory for StrExprFactory<'a> {
    type Token = StrToken<'a>;
    fn mk_atom(&mut self, token: Self::Token) -> Expr {
        match token {
            StrToken::Identifier(ident) => Expr::Symbol(ident.to_string()),
            StrToken::Number(number) => Expr::Literal(number),
            _ => panic!(),
        }
    }
}

pub trait OperationReceiver<'src> {
    fn include_source_file(&mut self, filename: &'src str);
    fn emit_instruction(&mut self, instruction: ir::Instruction);
    fn define_label(&mut self, label: &'src str);
}

struct Session<'session, FS: 'session, SAF: 'session, S> {
    fs: &'session mut FS,
    analyzer_factory: &'session mut SAF,
    section: S,
    _phantom: std::marker::PhantomData<&'session ()>,
}

impl<'session, FS: FileSystem, SAF: SrcAnalyzerFactory, S: ir::Section>
    Session<'session, FS, SAF, S>
{
    fn new(
        fs: &'session mut FS,
        analyzer_factory: &'session mut SAF,
        section: S,
    ) -> Session<'session, FS, SAF, S> {
        Session {
            fs,
            analyzer_factory,
            section,
            _phantom: std::marker::PhantomData,
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
        let mut analyzer_factory = CollectingSrcAnalyzerFactory::new();
        {
            let mut session = Session::new(&mut fs, &mut analyzer_factory, NullSection::new());
            session.include_source_file(filename);
        }
        assert_eq!(
            Rc::try_unwrap(analyzer_factory.src).unwrap().into_inner(),
            [contents]
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

    struct CollectingSrcAnalyzerFactory {
        src: Rc<RefCell<Vec<String>>>,
    }

    impl CollectingSrcAnalyzerFactory {
        fn new() -> CollectingSrcAnalyzerFactory {
            CollectingSrcAnalyzerFactory {
                src: Rc::new(RefCell::new(Vec::new())),
            }
        }
    }

    impl SrcAnalyzerFactory for CollectingSrcAnalyzerFactory {
        type SrcAnalyzer = CollectingSrcAnalyzer;

        fn mk_src_analyzer(&mut self) -> Self::SrcAnalyzer {
            CollectingSrcAnalyzer::new(self.src.clone())
        }
    }

    struct CollectingSrcAnalyzer {
        src: Rc<RefCell<Vec<String>>>,
    }

    impl CollectingSrcAnalyzer {
        fn new(src: Rc<RefCell<Vec<String>>>) -> CollectingSrcAnalyzer {
            CollectingSrcAnalyzer { src }
        }
    }

    impl SrcAnalyzer for CollectingSrcAnalyzer {
        fn analyze<'src, OR: OperationReceiver<'src>>(
            &mut self,
            src: &'src str,
            _receiver: &mut OR,
        ) {
            self.src.borrow_mut().push(String::from(src))
        }
    }

    struct NullSection;

    impl NullSection {
        fn new() -> NullSection {
            NullSection {}
        }
    }

    impl ir::Section for NullSection {
        fn add_instruction(&mut self, _instruction: Instruction) {}
        fn add_label(&mut self, _label: &str) {}
    }
}
