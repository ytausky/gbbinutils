use frontend::{Atom, Command, OperationReceiver, StrExprFactory};
use frontend::syntax::{self, SynExpr, Token};

use std::marker::PhantomData;

mod instruction;

pub struct SemanticActions<'actions, 'session, 'src, OR>
where
    'session: 'actions,
    'src: 'actions,
    OR: 'session + OperationReceiver<'src>,
{
    session: &'session mut OR,
    contexts: Vec<Context<&'src str>>,
    _phantom: PhantomData<&'actions ()>,
}

enum Context<S> {
    Block,
    Instruction(syntax::Token<S>, Vec<SynExpr<syntax::Token<S>>>),
}

impl<'actions, 'session, 'src, OR> SemanticActions<'actions, 'session, 'src, OR>
where
    'session: 'actions,
    'src: 'actions,
    OR: 'session + OperationReceiver<'src>,
{
    pub fn new(session: &'session mut OR) -> SemanticActions<'actions, 'session, 'src, OR> {
        SemanticActions {
            session,
            contexts: vec![Context::Block],
            _phantom: PhantomData,
        }
    }
}

impl<'actions, 'session, 'src, OR> syntax::BlockContext
    for SemanticActions<'actions, 'session, 'src, OR>
where
    'session: 'actions,
    'src: 'actions,
    OR: 'session + OperationReceiver<'src>,
{
    type Terminal = Token<&'src str>;
    type CommandContext = Self;
    type MacroInvocationContext = Self;
    type TerminalSequenceContext = Self;

    fn add_label(&mut self, label: Self::Terminal) {
        match label {
            Token::Label(spelling) => self.session.define_label(spelling),
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

    fn enter_macro_invocation(
        &mut self,
        _name: Self::Terminal,
    ) -> &mut Self::MacroInvocationContext {
        unimplemented!()
    }
}

impl<'actions, 'session, 'src, OR> syntax::CommandContext
    for SemanticActions<'actions, 'session, 'src, OR>
where
    'session: 'actions,
    'src: 'actions,
    OR: 'session + OperationReceiver<'src>,
{
    type Terminal = Token<&'src str>;

    fn add_argument(&mut self, expr: SynExpr<Self::Terminal>) {
        match self.contexts.last_mut() {
            Some(&mut Context::Instruction(_, ref mut args)) => args.push(expr),
            _ => panic!(),
        }
    }

    fn exit_command(&mut self) {
        if let Some(Context::Instruction(name, args)) = self.contexts.pop() {
            match name {
                Token::Command(Command::Include) => {
                    self.session.include_source_file(reduce_include(args))
                }
                Token::Command(command) => {
                    let mut analyzer =
                        self::instruction::CommandAnalyzer::new(StrExprFactory::new());
                    self.session.emit_instruction(
                        analyzer
                            .analyze_instruction(command, args.into_iter())
                            .unwrap(),
                    )
                }
                Token::Atom(Atom::Ident(ident)) => {
                    println!("Probably macro invocation: {:?} {:?}", ident, args)
                }
                _ => panic!(),
            }
        } else {
            panic!()
        }
    }
}

impl<'actions, 'session, 'src, OR> syntax::MacroInvocationContext
    for SemanticActions<'actions, 'session, 'src, OR>
where
    'session: 'actions,
    'src: 'actions,
    OR: 'session + OperationReceiver<'src>,
{
    type Terminal = Token<&'src str>;
    type TerminalSequenceContext = Self;

    fn enter_macro_argument(&mut self) -> &mut Self::TerminalSequenceContext {
        self
    }

    fn exit_macro_invocation(&mut self) {}
}

impl<'actions, 'session, 'src, OR> syntax::TerminalSequenceContext
    for SemanticActions<'actions, 'session, 'src, OR>
where
    'session: 'actions,
    'src: 'actions,
    OR: 'session + OperationReceiver<'src>,
{
    type Terminal = Token<&'src str>;

    fn push_terminal(&mut self, _terminal: Self::Terminal) {
        unimplemented!()
    }

    fn exit_terminal_sequence(&mut self) {
        unimplemented!()
    }
}

fn reduce_include<S>(mut arguments: Vec<SynExpr<Token<S>>>) -> S {
    assert_eq!(arguments.len(), 1);
    let path = arguments.pop().unwrap();
    match path {
        SynExpr::Atom(Token::Atom(Atom::String(path_str))) => path_str,
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use frontend::syntax::{BlockContext, CommandContext};
    use ir;

    struct TestOperationReceiver(Vec<TestOperation>);

    impl TestOperationReceiver {
        fn new() -> TestOperationReceiver {
            TestOperationReceiver(Vec::new())
        }
    }

    #[derive(Debug, PartialEq)]
    enum TestOperation {
        Include(&'static str),
        Instruction(ir::Instruction),
        Label(&'static str),
    }

    impl OperationReceiver<'static> for TestOperationReceiver {
        fn include_source_file(&mut self, filename: &'static str) {
            self.0.push(TestOperation::Include(filename))
        }

        fn emit_instruction(&mut self, instruction: ir::Instruction) {
            self.0.push(TestOperation::Instruction(instruction))
        }

        fn define_label(&mut self, label: &'static str) {
            self.0.push(TestOperation::Label(label))
        }
    }

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let actions = collect_semantic_actions(|mut actions| {
            actions.enter_command(Token::Command(Command::Include));
            let expr = SynExpr::from(Token::Atom(Atom::String(filename)));
            actions.add_argument(expr);
            actions.exit_command();
        });
        assert_eq!(actions, [TestOperation::Include(filename)])
    }

    #[test]
    fn analyze_label() {
        let actions =
            collect_semantic_actions(|mut actions| actions.add_label(Token::Label("label")));
        assert_eq!(actions, [TestOperation::Label("label")])
    }

    fn collect_semantic_actions<F>(f: F) -> Vec<TestOperation>
    where
        F: for<'actions, 'session: 'actions> FnOnce(
            SemanticActions<
                'actions,
                'session,
                'static,
                TestOperationReceiver,
            >,
        ),
    {
        let mut operations = TestOperationReceiver::new();
        f(SemanticActions::new(&mut operations));
        operations.0
    }
}
