use ir;

use std;

mod ast;
mod semantics;
mod syntax;

use ir::*;
use self::ast::Expression;
use self::syntax::*;

pub fn analyze_file<S: ir::Section>(name: &str, section: S) {
    use std::io::prelude::*;
    let mut file = std::fs::File::open(name).unwrap();
    let mut src = String::new();
    file.read_to_string(&mut src).unwrap();
    let ast_builder = AstBuilder::new(section);
    syntax::parse(&src, ast_builder)
}

pub struct AstBuilder<'a, S: ir::Section> {
    ast: Vec<ast::AsmItem<'a>>,
    contexts: Vec<Context<'a>>,
    section: S,
}

enum Context<'a> {
    Block,
    Instruction(syntax::Token<'a>, Vec<ast::Expression<syntax::Token<'a>>>),
}

impl<'a, S: ir::Section> AstBuilder<'a, S> {
    pub fn new(section: S) -> AstBuilder<'a, S> {
        AstBuilder {
            ast: Vec::new(),
            contexts: vec![Context::Block],
            section: section,
        }
    }

    #[cfg(test)]
    fn ast(&self) -> &Vec<ast::AsmItem<'a>> {
        &self.ast
    }
}

impl<'a, S: Section> syntax::BlockContext for AstBuilder<'a, S> {
    type Terminal = Token<'a>;
    type Expr = Expression<Self::Terminal>;
    type CommandContext = Self;
    type TerminalSequenceContext = Self;

    fn add_label(&mut self, label: Self::Terminal) {
        match label {
            Token::Label(spelling) => self.section.add_label(spelling),
            _ => panic!(),
        }
    }

    fn enter_command(&mut self, name: Self::Terminal) -> &mut Self::CommandContext {
        self.contexts.push(Context::Instruction(name, vec![]));
        self
    }

    fn enter_macro_definition(
        &mut self,
        _label: Self::Terminal,
    ) -> &mut Self::TerminalSequenceContext {
        unimplemented!()
    }
}

impl<'a, S: Section> syntax::CommandContext for AstBuilder<'a, S> {
    type Terminal = Token<'a>;
    type Expr = Expression<Self::Terminal>;

    fn add_argument(&mut self, expr: Self::Expr) {
        match self.contexts.last_mut() {
            Some(&mut Context::Instruction(_, ref mut args)) => args.push(expr),
            _ => panic!(),
        }
    }

    fn exit_command(&mut self) {
        if let Some(Context::Instruction(name, args)) = self.contexts.pop() {
            match name {
                Token::Keyword(Keyword::Include) => self.ast.push(semantics::reduce_include(args)),
                Token::Keyword(keyword) => self.section.add_instruction(
                    semantics::interpret_instruction(keyword, args.into_iter()).unwrap(),
                ),
                _ => panic!(),
            }
        } else {
            panic!()
        }
    }
}

impl<'a, S: Section> syntax::TerminalSequenceContext for AstBuilder<'a, S> {
    type Terminal = Token<'a>;

    fn push_terminal(&mut self, _terminal: Self::Terminal) {
        unimplemented!()
    }

    fn exit_terminal_sequence(&mut self) {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use self::ast::ExprFactory;
    use frontend::semantics::*;

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let mut actions = Vec::new();
        let mut builder = AstBuilder::new(TestSection::new(&mut actions));
        builder.enter_command(Token::Keyword(Keyword::Include));
        let mut expr_builder = ast::ExprBuilder::new();
        let expr = expr_builder.from_atom(Token::QuotedString(filename));
        builder.add_argument(expr);
        builder.exit_command();
        let ast = builder.ast().to_vec();
        assert_eq!(*ast.last().unwrap(), include(filename))
    }

    type TestActions = Vec<Action>;

    #[derive(Debug, PartialEq)]
    enum Action {
        AddLabel(String),
        AddInstruction(Instruction),
    }

    struct TestSection<'a> {
        actions: &'a mut TestActions,
    }

    impl<'a> TestSection<'a> {
        fn new(actions: &'a mut TestActions) -> TestSection<'a> {
            TestSection { actions: actions }
        }
    }

    impl<'a> Section for TestSection<'a> {
        fn add_instruction(&mut self, instruction: Instruction) {
            self.actions.push(Action::AddInstruction(instruction))
        }

        fn add_label(&mut self, label: &str) {
            self.actions.push(Action::AddLabel(label.to_string()))
        }
    }

    #[test]
    fn analyze_label() {
        let mut actions = Vec::new();
        {
            let mut builder = AstBuilder::new(TestSection::new(&mut actions));
            builder.add_label(Token::Label("label"));
        }
        assert_eq!(actions, vec![Action::AddLabel("label".to_string())])
    }
}
