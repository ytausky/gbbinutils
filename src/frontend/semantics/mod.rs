use frontend::{Atom, OperationReceiver, StrExprFactory};
use frontend::syntax::{self, SynExpr, Token, keyword::Command};

use std::marker::PhantomData;

mod instruction;

pub struct SemanticActions<'actions, 'session, OR>
where
    'session: 'actions,
    OR: 'session + OperationReceiver,
{
    session: &'session mut OR,
    contexts: Vec<Context<String>>,
    expr_factory: StrExprFactory,
    _phantom: PhantomData<&'actions ()>,
}

enum Context<S> {
    Block,
    MacroArg(Vec<Token<S>>),
    MacroDef(syntax::Token<S>, Vec<syntax::Token<S>>),
    MacroInvocation(syntax::Token<S>, Vec<Vec<syntax::Token<S>>>),
}

impl<'actions, 'session, OR> SemanticActions<'actions, 'session, OR>
where
    'session: 'actions,
    OR: 'session + OperationReceiver,
{
    pub fn new(session: &'session mut OR) -> SemanticActions<'actions, 'session, OR> {
        SemanticActions {
            session,
            contexts: vec![Context::Block],
            expr_factory: StrExprFactory::new(),
            _phantom: PhantomData,
        }
    }
}

pub struct CommandActions<'actions, 'session, OR>
where
    'session: 'actions,
    OR: 'session + OperationReceiver,
{
    name: Token<String>,
    args: Vec<SynExpr<syntax::Token<String>>>,
    enclosing_context: SemanticActions<'actions, 'session, OR>,
}

impl<'actions, 'session, OR> CommandActions<'actions, 'session, OR>
where
    'session: 'actions,
    OR: 'session + OperationReceiver,
{
    fn new(
        name: Token<String>,
        enclosing_context: SemanticActions<'actions, 'session, OR>,
    ) -> CommandActions<'actions, 'session, OR> {
        CommandActions {
            name,
            args: Vec::new(),
            enclosing_context,
        }
    }
}

impl<'actions, 'session, OR> syntax::CommandContext for CommandActions<'actions, 'session, OR>
where
    'session: 'actions,
    OR: 'session + OperationReceiver,
{
    type Terminal = Token<String>;
    type EnclosingContext = SemanticActions<'actions, 'session, OR>;

    fn add_argument(&mut self, expr: SynExpr<Self::Terminal>) {
        self.args.push(expr)
    }

    fn exit_command(mut self) -> Self::EnclosingContext {
        match self.name {
            Token::Command(Command::Include) => self.enclosing_context
                .session
                .include_source_file(reduce_include(self.args)),
            Token::Command(command) => {
                let mut analyzer = self::instruction::CommandAnalyzer::new(
                    &mut self.enclosing_context.expr_factory,
                );
                self.enclosing_context.session.emit_instruction(
                    analyzer
                        .analyze_instruction(command, self.args.into_iter())
                        .unwrap(),
                )
            }
            _ => panic!(),
        }
        self.enclosing_context
    }
}

impl<'actions, 'session, OR> syntax::BlockContext for SemanticActions<'actions, 'session, OR>
where
    'session: 'actions,
    OR: 'session + OperationReceiver,
{
    type Terminal = Token<String>;
    type CommandContext = CommandActions<'actions, 'session, OR>;
    type MacroInvocationContext = Self;
    type TerminalSequenceContext = Self;

    fn add_label(&mut self, label: Self::Terminal) {
        match label {
            Token::Label(spelling) => self.session.define_label(spelling),
            _ => panic!(),
        }
    }

    fn enter_command(self, name: Self::Terminal) -> Self::CommandContext {
        CommandActions::new(name, self)
    }

    fn enter_macro_definition(mut self, name: Self::Terminal) -> Self::TerminalSequenceContext {
        self.contexts.push(Context::MacroDef(name, vec![]));
        self
    }

    fn enter_macro_invocation(mut self, name: Self::Terminal) -> Self::MacroInvocationContext {
        self.contexts.push(Context::MacroInvocation(name, vec![]));
        self
    }
}

impl<'actions, 'session, OR> syntax::MacroInvocationContext
    for SemanticActions<'actions, 'session, OR>
where
    'session: 'actions,
    OR: 'session + OperationReceiver,
{
    type Terminal = Token<String>;
    type EnclosingContext = Self;
    type TerminalSequenceContext = Self;

    fn enter_macro_argument(mut self) -> Self::TerminalSequenceContext {
        self.contexts.push(Context::MacroArg(vec![]));
        self
    }

    fn exit_macro_invocation(mut self) -> Self::EnclosingContext {
        match self.contexts.pop() {
            Some(Context::MacroInvocation(Token::Atom(Atom::Ident(name)), args)) => {
                self.session.invoke_macro(name, args)
            }
            _ => panic!(),
        }
        self
    }
}

impl<'actions, 'session, OR> syntax::TerminalSequenceContext
    for SemanticActions<'actions, 'session, OR>
where
    'session: 'actions,
    OR: 'session + OperationReceiver,
{
    type Terminal = Token<String>;
    type EnclosingContext = Self;

    fn push_terminal(&mut self, terminal: Self::Terminal) {
        match self.contexts.last_mut() {
            Some(&mut Context::MacroDef(_, ref mut tokens))
            | Some(&mut Context::MacroArg(ref mut tokens)) => tokens.push(terminal),
            _ => panic!(),
        }
    }

    fn exit_terminal_sequence(mut self) -> Self::EnclosingContext {
        match self.contexts.pop() {
            Some(Context::MacroArg(tokens)) => match self.contexts.last_mut() {
                Some(&mut Context::MacroInvocation(_, ref mut args)) => args.push(tokens),
                _ => panic!(),
            },
            Some(Context::MacroDef(Token::Label(name), tokens)) => {
                self.session.define_macro(name, tokens)
            }
            _ => panic!(),
        }
        self
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

    use frontend::syntax::{BlockContext, CommandContext, MacroInvocationContext,
                           TerminalSequenceContext, keyword::Operand};
    use ir;

    struct TestOperationReceiver(Vec<TestOperation>);

    impl TestOperationReceiver {
        fn new() -> TestOperationReceiver {
            TestOperationReceiver(Vec::new())
        }
    }

    #[derive(Debug, PartialEq)]
    enum TestOperation {
        DefineMacro(String, Vec<Token<String>>),
        Include(String),
        Instruction(ir::Instruction),
        InvokeMacro(String, Vec<Vec<Token<String>>>),
        Label(String),
    }

    impl OperationReceiver for TestOperationReceiver {
        fn include_source_file(&mut self, filename: String) {
            self.0.push(TestOperation::Include(filename))
        }

        fn emit_instruction(&mut self, instruction: ir::Instruction) {
            self.0.push(TestOperation::Instruction(instruction))
        }

        fn define_label(&mut self, label: String) {
            self.0.push(TestOperation::Label(label))
        }

        fn define_macro(&mut self, name: String, tokens: Vec<Token<String>>) {
            self.0.push(TestOperation::DefineMacro(name, tokens))
        }

        fn invoke_macro(&mut self, name: String, args: Vec<Vec<Token<String>>>) {
            self.0.push(TestOperation::InvokeMacro(name, args))
        }
    }

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions.enter_command(Token::Command(Command::Include));
            let expr = SynExpr::from(Token::Atom(Atom::String(filename.to_string())));
            command.add_argument(expr);
            command.exit_command();
        });
        assert_eq!(actions, [TestOperation::Include(filename.to_string())])
    }

    #[test]
    fn analyze_label() {
        let label = "label";
        let actions = collect_semantic_actions(|mut actions| {
            actions.add_label(Token::Label(label.to_string()))
        });
        assert_eq!(actions, [TestOperation::Label(label.to_string())])
    }

    #[test]
    fn define_macro() {
        let name = "my_macro";
        let tokens = vec![
            Token::Command(Command::Xor),
            Token::Atom(Atom::Operand(Operand::A)),
        ];
        let actions = collect_semantic_actions(|actions| {
            let mut token_seq_context =
                actions.enter_macro_definition(Token::Label(name.to_string()));
            for token in tokens.clone() {
                token_seq_context.push_terminal(token)
            }
            token_seq_context.exit_terminal_sequence();
        });
        assert_eq!(
            actions,
            [TestOperation::DefineMacro(name.to_string(), tokens)]
        )
    }

    #[test]
    fn invoke_nullary_macro() {
        let name = "my_macro";
        let actions = collect_semantic_actions(|actions| {
            let invocation =
                actions.enter_macro_invocation(Token::Atom(Atom::Ident(name.to_string())));
            invocation.exit_macro_invocation();
        });
        assert_eq!(
            actions,
            [TestOperation::InvokeMacro(name.to_string(), Vec::new())]
        )
    }

    #[test]
    fn invoke_unary_macro() {
        let name = "my_macro";
        let arg_token = Token::Atom(Atom::Operand(Operand::A));
        let actions = collect_semantic_actions(|actions| {
            let mut invocation =
                actions.enter_macro_invocation(Token::Atom(Atom::Ident(name.to_string())));
            invocation = {
                let mut arg = invocation.enter_macro_argument();
                arg.push_terminal(arg_token.clone());
                arg.exit_terminal_sequence()
            };
            invocation.exit_macro_invocation();
        });
        assert_eq!(
            actions,
            [
                TestOperation::InvokeMacro(name.to_string(), vec![vec![arg_token]])
            ]
        )
    }

    fn collect_semantic_actions<F>(f: F) -> Vec<TestOperation>
    where
        F: for<'actions, 'session: 'actions> FnOnce(
            SemanticActions<
                'actions,
                'session,
                TestOperationReceiver,
            >,
        ),
    {
        let mut operations = TestOperationReceiver::new();
        f(SemanticActions::new(&mut operations));
        operations.0
    }
}
