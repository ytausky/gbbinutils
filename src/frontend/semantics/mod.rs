use frontend::{Atom, OperationReceiver, StrExprFactory};
use frontend::syntax::{self, SynExpr, Token, keyword::Command};

mod instruction;

pub struct SemanticActions<'session, OR: 'session> {
    session: &'session mut OR,
    expr_factory: StrExprFactory,
}

impl<'session, OR: 'session> SemanticActions<'session, OR> {
    pub fn new(session: &'session mut OR) -> SemanticActions<'session, OR> {
        SemanticActions {
            session,
            expr_factory: StrExprFactory::new(),
        }
    }
}

impl<'session, OR> syntax::BlockContext for SemanticActions<'session, OR>
where
    OR: 'session + OperationReceiver,
{
    type Terminal = Token<String>;
    type CommandContext = CommandActions<'session, OR>;
    type MacroDefContext = MacroDefActions<'session, OR>;
    type MacroInvocationContext = MacroInvocationActions<'session, OR>;

    fn add_label(&mut self, label: Self::Terminal) {
        match label {
            Token::Label(spelling) => self.session.define_label(spelling),
            _ => panic!(),
        }
    }

    fn enter_command(self, name: Self::Terminal) -> Self::CommandContext {
        CommandActions::new(name, self)
    }

    fn enter_macro_def(self, name: Self::Terminal) -> Self::MacroDefContext {
        MacroDefActions::new(name, self)
    }

    fn enter_macro_invocation(self, name: Self::Terminal) -> Self::MacroInvocationContext {
        MacroInvocationActions::new(name, self)
    }
}

pub struct CommandActions<'session, OR: 'session> {
    name: Token<String>,
    args: Vec<SynExpr<syntax::Token<String>>>,
    enclosing_context: SemanticActions<'session, OR>,
}

impl<'session, OR: 'session> CommandActions<'session, OR> {
    fn new(
        name: Token<String>,
        enclosing_context: SemanticActions<'session, OR>,
    ) -> CommandActions<'session, OR> {
        CommandActions {
            name,
            args: Vec::new(),
            enclosing_context,
        }
    }
}

impl<'session, OR> syntax::CommandContext for CommandActions<'session, OR>
where
    OR: 'session + OperationReceiver,
{
    type Terminal = Token<String>;
    type EnclosingContext = SemanticActions<'session, OR>;

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

pub struct MacroDefActions<'session, OR: 'session> {
    name: Token<String>,
    tokens: Vec<Token<String>>,
    enclosing_context: SemanticActions<'session, OR>,
}

impl<'session, OR> MacroDefActions<'session, OR>
where
    OR: 'session + OperationReceiver,
{
    fn new(
        name: Token<String>,
        enclosing_context: SemanticActions<'session, OR>,
    ) -> MacroDefActions<'session, OR> {
        MacroDefActions {
            name,
            tokens: Vec::new(),
            enclosing_context,
        }
    }
}

impl<'session, OR> syntax::TerminalSequenceContext for MacroDefActions<'session, OR>
where
    OR: 'session + OperationReceiver,
{
    type Terminal = Token<String>;
    type EnclosingContext = SemanticActions<'session, OR>;

    fn push_terminal(&mut self, terminal: Self::Terminal) {
        self.tokens.push(terminal)
    }

    fn exit_terminal_sequence(self) -> Self::EnclosingContext {
        match self.name {
            Token::Label(label) => self.enclosing_context
                .session
                .define_macro(label, self.tokens),
            _ => panic!(),
        }
        self.enclosing_context
    }
}

pub struct MacroInvocationActions<'session, OR: 'session> {
    name: Token<String>,
    args: Vec<Vec<Token<String>>>,
    enclosing_context: SemanticActions<'session, OR>,
}

impl<'session, OR: 'session> MacroInvocationActions<'session, OR> {
    fn new(
        name: Token<String>,
        enclosing_context: SemanticActions<'session, OR>,
    ) -> MacroInvocationActions<'session, OR> {
        MacroInvocationActions {
            name,
            args: Vec::new(),
            enclosing_context,
        }
    }

    fn push_arg(&mut self, arg: Vec<Token<String>>) {
        self.args.push(arg)
    }
}

impl<'session, OR> syntax::MacroInvocationContext for MacroInvocationActions<'session, OR>
where
    OR: 'session + OperationReceiver,
{
    type Terminal = Token<String>;
    type EnclosingContext = SemanticActions<'session, OR>;
    type MacroArgContext = MacroArgActions<'session, OR>;

    fn enter_macro_arg(self) -> Self::MacroArgContext {
        MacroArgActions::new(self)
    }

    fn exit_macro_invocation(self) -> Self::EnclosingContext {
        match self.name {
            Token::Atom(Atom::Ident(name)) => {
                self.enclosing_context.session.invoke_macro(name, self.args)
            }
            _ => panic!(),
        }
        self.enclosing_context
    }
}

pub struct MacroArgActions<'session, OR: 'session> {
    tokens: Vec<Token<String>>,
    enclosing_context: MacroInvocationActions<'session, OR>,
}

impl<'session, OR: 'session> MacroArgActions<'session, OR> {
    fn new(
        enclosing_context: MacroInvocationActions<'session, OR>,
    ) -> MacroArgActions<'session, OR> {
        MacroArgActions {
            tokens: Vec::new(),
            enclosing_context,
        }
    }
}

impl<'session, OR> syntax::TerminalSequenceContext for MacroArgActions<'session, OR> {
    type Terminal = Token<String>;
    type EnclosingContext = MacroInvocationActions<'session, OR>;

    fn push_terminal(&mut self, terminal: Self::Terminal) {
        self.tokens.push(terminal)
    }

    fn exit_terminal_sequence(mut self) -> Self::EnclosingContext {
        self.enclosing_context.push_arg(self.tokens);
        self.enclosing_context
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
            let mut token_seq_context = actions.enter_macro_def(Token::Label(name.to_string()));
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
                let mut arg = invocation.enter_macro_arg();
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
        F: for<'session> FnOnce(SemanticActions<'session, TestOperationReceiver>),
    {
        let mut operations = TestOperationReceiver::new();
        f(SemanticActions::new(&mut operations));
        operations.0
    }
}
