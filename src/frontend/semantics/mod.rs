use frontend::{Atom, Frontend, StrExprFactory};
use frontend::syntax::{self, SynExpr, Token, keyword::Command};

mod instruction;

pub struct SemanticActions<'session, F: 'session> {
    session: &'session mut F,
    expr_factory: StrExprFactory,
}

impl<'session, F: 'session> SemanticActions<'session, F> {
    pub fn new(session: &'session mut F) -> SemanticActions<'session, F> {
        SemanticActions {
            session,
            expr_factory: StrExprFactory::new(),
        }
    }
}

impl<'session, F> syntax::BlockContext for SemanticActions<'session, F>
where
    F: 'session + Frontend,
{
    type Terminal = Token<String>;
    type CommandContext = CommandActions<'session, F>;
    type MacroDefContext = MacroDefActions<'session, F>;
    type MacroInvocationContext = MacroInvocationActions<'session, F>;

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

pub struct CommandActions<'session, F: 'session> {
    name: Token<String>,
    args: Vec<SynExpr<syntax::Token<String>>>,
    enclosing_context: SemanticActions<'session, F>,
}

impl<'session, F: 'session> CommandActions<'session, F> {
    fn new(
        name: Token<String>,
        enclosing_context: SemanticActions<'session, F>,
    ) -> CommandActions<'session, F> {
        CommandActions {
            name,
            args: Vec::new(),
            enclosing_context,
        }
    }
}

impl<'session, F> syntax::CommandContext for CommandActions<'session, F>
where
    F: 'session + Frontend,
{
    type Terminal = Token<String>;
    type EnclosingContext = SemanticActions<'session, F>;

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

pub struct MacroDefActions<'session, F: 'session> {
    name: Token<String>,
    tokens: Vec<Token<String>>,
    enclosing_context: SemanticActions<'session, F>,
}

impl<'session, F> MacroDefActions<'session, F>
where
    F: 'session + Frontend,
{
    fn new(
        name: Token<String>,
        enclosing_context: SemanticActions<'session, F>,
    ) -> MacroDefActions<'session, F> {
        MacroDefActions {
            name,
            tokens: Vec::new(),
            enclosing_context,
        }
    }
}

impl<'session, F> syntax::TerminalSequenceContext for MacroDefActions<'session, F>
where
    F: 'session + Frontend,
{
    type Terminal = Token<String>;
    type EnclosingContext = SemanticActions<'session, F>;

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

pub struct MacroInvocationActions<'session, F: 'session> {
    name: Token<String>,
    args: Vec<Vec<Token<String>>>,
    enclosing_context: SemanticActions<'session, F>,
}

impl<'session, F: 'session> MacroInvocationActions<'session, F> {
    fn new(
        name: Token<String>,
        enclosing_context: SemanticActions<'session, F>,
    ) -> MacroInvocationActions<'session, F> {
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

impl<'session, F> syntax::MacroInvocationContext for MacroInvocationActions<'session, F>
where
    F: 'session + Frontend,
{
    type Terminal = Token<String>;
    type EnclosingContext = SemanticActions<'session, F>;
    type MacroArgContext = MacroArgActions<'session, F>;

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

pub struct MacroArgActions<'session, F: 'session> {
    tokens: Vec<Token<String>>,
    enclosing_context: MacroInvocationActions<'session, F>,
}

impl<'session, F: 'session> MacroArgActions<'session, F> {
    fn new(enclosing_context: MacroInvocationActions<'session, F>) -> MacroArgActions<'session, F> {
        MacroArgActions {
            tokens: Vec::new(),
            enclosing_context,
        }
    }
}

impl<'session, F> syntax::TerminalSequenceContext for MacroArgActions<'session, F> {
    type Terminal = Token<String>;
    type EnclosingContext = MacroInvocationActions<'session, F>;

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

    struct TestFrontend(Vec<TestOperation>);

    impl TestFrontend {
        fn new() -> TestFrontend {
            TestFrontend(Vec::new())
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

    impl Frontend for TestFrontend {
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
        F: for<'session> FnOnce(SemanticActions<'session, TestFrontend>),
    {
        let mut operations = TestFrontend::new();
        f(SemanticActions::new(&mut operations));
        operations.0
    }
}
