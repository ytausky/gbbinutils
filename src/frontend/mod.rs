use ir;

use std;

mod semantics;
mod syntax;

use ir::*;
use self::syntax::*;

pub fn analyze_file<S: ir::Section>(name: &str, section: S) {
    use std::io::prelude::*;
    let mut file = std::fs::File::open(name).unwrap();
    let mut src = String::new();
    file.read_to_string(&mut src).unwrap();
    let mut session = Session::new(section);
    let ast_builder = AstBuilder::new(&mut session);
    syntax::parse(&src, ast_builder)
}

pub trait ExprFactory {
    type Token: Token;
    fn mk_atom(&mut self, token: Self::Token) -> Expr;
}

pub struct StrExprFactory<'a>(std::marker::PhantomData<&'a ()>);

impl<'a> StrExprFactory<'a> {
    fn new() -> StrExprFactory<'a> {
        StrExprFactory(std::marker::PhantomData)
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

trait OperationReceiver<'a> {
    fn include_source_file(&mut self, filename: &'a str);
    fn emit_instruction(&mut self, instruction: ir::Instruction);
    fn emit_label(&mut self, label: &'a str);
}

struct Session<'a, S> {
    ast: Vec<AsmItem<'a>>,
    section: S,
}

impl<'a, S: ir::Section> Session<'a, S> {
    fn new(section: S) -> Session<'a, S> {
        Session {
            ast: Vec::new(),
            section,
        }
    }

    #[cfg(test)]
    fn ast(&self) -> &Vec<AsmItem<'a>> {
        &self.ast
    }
}

impl<'a, S: ir::Section> OperationReceiver<'a> for Session<'a, S> {
    fn include_source_file(&mut self, filename: &'a str) {
        self.ast.push(AsmItem::Include(filename))
    }

    fn emit_instruction(&mut self, instruction: ir::Instruction) {
        self.section.add_instruction(instruction)
    }

    fn emit_label(&mut self, label: &'a str) {
        self.section.add_label(label)
    }
}

struct AstBuilder<'s, 'a, OR: 's + OperationReceiver<'a>> {
    session: &'s mut OR,
    contexts: Vec<Context<'a>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AsmItem<'a> {
    Include(&'a str),
}

enum Context<'a> {
    Block,
    Instruction(syntax::StrToken<'a>, Vec<SynExpr<syntax::StrToken<'a>>>),
}

impl<'s, 'a, OR: 's + OperationReceiver<'a>> AstBuilder<'s, 'a, OR> {
    fn new(session: &'s mut OR) -> AstBuilder<'s, 'a, OR> {
        AstBuilder {
            session,
            contexts: vec![Context::Block],
        }
    }
}

impl<'s, 'a, OR: 's + OperationReceiver<'a>> syntax::BlockContext for AstBuilder<'s, 'a, OR> {
    type Terminal = StrToken<'a>;
    type CommandContext = Self;
    type TerminalSequenceContext = Self;

    fn add_label(&mut self, label: Self::Terminal) {
        match label {
            StrToken::Label(spelling) => self.session.emit_label(spelling),
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

impl<'s, 'a, OR: 's + OperationReceiver<'a>> syntax::CommandContext for AstBuilder<'s, 'a, OR> {
    type Terminal = StrToken<'a>;

    fn add_argument(&mut self, expr: SynExpr<Self::Terminal>) {
        match self.contexts.last_mut() {
            Some(&mut Context::Instruction(_, ref mut args)) => args.push(expr),
            _ => panic!(),
        }
    }

    fn exit_command(&mut self) {
        if let Some(Context::Instruction(name, args)) = self.contexts.pop() {
            match name {
                StrToken::Keyword(Keyword::Include) => {
                    self.session.include_source_file(reduce_include(args))
                }
                StrToken::Keyword(keyword) => {
                    let mut analyzer = semantics::CommandAnalyzer::new(StrExprFactory::new());
                    self.session.emit_instruction(
                        analyzer
                            .analyze_instruction(keyword, args.into_iter())
                            .unwrap(),
                    )
                }
                StrToken::Identifier(ident) => {
                    println!("Probably macro invocation: {:?} {:?}", ident, args)
                }
                _ => panic!(),
            }
        } else {
            panic!()
        }
    }
}

impl<'s, 'a, OR: 's + OperationReceiver<'a>> syntax::TerminalSequenceContext
    for AstBuilder<'s, 'a, OR> {
    type Terminal = StrToken<'a>;

    fn push_terminal(&mut self, _terminal: Self::Terminal) {
        unimplemented!()
    }

    fn exit_terminal_sequence(&mut self) {
        unimplemented!()
    }
}

fn reduce_include(mut arguments: Vec<SynExpr<StrToken>>) -> &str {
    assert_eq!(arguments.len(), 1);
    let path = arguments.pop().unwrap();
    match path {
        SynExpr::Atom(StrToken::QuotedString(path_str)) => path_str,
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let mut actions = Vec::new();
        let mut session = Session::new(TestSection::new(&mut actions));
        {
            let mut builder = AstBuilder::new(&mut session);
            builder.enter_command(StrToken::Keyword(Keyword::Include));
            let expr = SynExpr::from(StrToken::QuotedString(filename));
            builder.add_argument(expr);
            builder.exit_command();
        }
        let ast = session.ast().to_vec();
        assert_eq!(*ast.last().unwrap(), AsmItem::Include(filename))
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
            let mut session = Session::new(TestSection::new(&mut actions));
            let mut builder = AstBuilder::new(&mut session);
            builder.add_label(StrToken::Label("label"));
        }
        assert_eq!(actions, vec![Action::AddLabel("label".to_string())])
    }
}
