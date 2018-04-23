use backend;
use frontend::{Atom, Frontend, StrExprFactory};
use frontend::syntax::{self, token, SynExpr, Token, TokenSpec, keyword::Command};

mod instruction;

pub struct SemanticActions<'a, F: Frontend + 'a> {
    session: &'a mut F,
    expr_factory: StrExprFactory,
}

impl<'a, F: Frontend + 'a> SemanticActions<'a, F> {
    pub fn new(session: &'a mut F) -> SemanticActions<'a, F> {
        SemanticActions {
            session,
            expr_factory: StrExprFactory::new(),
        }
    }
}

impl<'a, F: Frontend + 'a> syntax::FileContext<String, F::CodeRef> for SemanticActions<'a, F> {
    type CommandContext = CommandActions<'a, F>;
    type MacroDefContext = MacroDefActions<'a, F>;
    type MacroInvocationContext = MacroInvocationActions<'a, F>;

    fn add_label(&mut self, label: (<String as TokenSpec>::Label, F::CodeRef)) {
        self.session.define_label(label);
    }

    fn enter_command(
        self,
        name: (<String as TokenSpec>::Command, F::CodeRef),
    ) -> Self::CommandContext {
        CommandActions::new(name, self)
    }

    fn enter_macro_def(
        self,
        name: (<String as TokenSpec>::Label, F::CodeRef),
    ) -> Self::MacroDefContext {
        MacroDefActions::new(name, self)
    }

    fn enter_macro_invocation(
        self,
        name: (<String as TokenSpec>::Atom, F::CodeRef),
    ) -> Self::MacroInvocationContext {
        MacroInvocationActions::new(name, self)
    }
}

pub struct CommandActions<'a, F: Frontend + 'a> {
    name: (Command, F::CodeRef),
    args: Vec<SynExpr<syntax::Token>>,
    parent: SemanticActions<'a, F>,
}

impl<'a, F: Frontend + 'a> CommandActions<'a, F> {
    fn new(name: (Command, F::CodeRef), parent: SemanticActions<'a, F>) -> CommandActions<'a, F> {
        CommandActions {
            name,
            args: Vec::new(),
            parent,
        }
    }
}

impl<'a, F: Frontend + 'a> syntax::CommandContext for CommandActions<'a, F> {
    type Token = Token;
    type Parent = SemanticActions<'a, F>;

    fn add_argument(&mut self, expr: SynExpr<Self::Token>) {
        self.args.push(expr)
    }

    fn exit_command(mut self) -> Self::Parent {
        match self.name {
            (Command::Db, _) => for arg in self.args {
                match arg {
                    SynExpr::Atom(atom) => {
                        use frontend::ExprFactory;
                        let expr = self.parent.expr_factory.mk_atom(atom);
                        self.parent.session.emit_item(backend::Item::Byte(expr))
                    }
                    _ => panic!(),
                }
            },
            (Command::Include, _) => self.parent
                .session
                .include_source_file(reduce_include(self.args)),
            (command, _) => {
                let mut analyzer =
                    self::instruction::CommandAnalyzer::new(&mut self.parent.expr_factory);
                self.parent.session.emit_item(
                    analyzer
                        .analyze_instruction(command, self.args.into_iter())
                        .map(backend::Item::Instruction)
                        .unwrap(),
                )
            }
        }
        self.parent
    }
}

pub struct MacroDefActions<'a, F: Frontend + 'a> {
    name: (String, F::CodeRef),
    tokens: Vec<Token>,
    parent: SemanticActions<'a, F>,
}

impl<'a, F: Frontend + 'a> MacroDefActions<'a, F> {
    fn new(name: (String, F::CodeRef), parent: SemanticActions<'a, F>) -> MacroDefActions<'a, F> {
        MacroDefActions {
            name,
            tokens: Vec::new(),
            parent,
        }
    }
}

impl<'a, F: Frontend + 'a> syntax::TokenSeqContext for MacroDefActions<'a, F> {
    type Token = Token;
    type Parent = SemanticActions<'a, F>;

    fn push_token(&mut self, token: Self::Token) {
        self.tokens.push(token)
    }

    fn exit_token_seq(self) -> Self::Parent {
        self.parent.session.define_macro(self.name, self.tokens);
        self.parent
    }
}

pub struct MacroInvocationActions<'a, F: Frontend + 'a> {
    name: (Atom<String>, F::CodeRef),
    args: Vec<Vec<Token>>,
    parent: SemanticActions<'a, F>,
}

impl<'a, F: Frontend + 'a> MacroInvocationActions<'a, F> {
    fn new(
        name: (Atom<String>, F::CodeRef),
        parent: SemanticActions<'a, F>,
    ) -> MacroInvocationActions<'a, F> {
        MacroInvocationActions {
            name,
            args: Vec::new(),
            parent,
        }
    }

    fn push_arg(&mut self, arg: Vec<Token>) {
        self.args.push(arg)
    }
}

impl<'a, F: Frontend + 'a> syntax::MacroInvocationContext for MacroInvocationActions<'a, F> {
    type Token = Token;
    type Parent = SemanticActions<'a, F>;
    type MacroArgContext = MacroArgActions<'a, F>;

    fn enter_macro_arg(self) -> Self::MacroArgContext {
        MacroArgActions::new(self)
    }

    fn exit_macro_invocation(self) -> Self::Parent {
        match self.name {
            (Atom::Ident(name), code_ref) => self.parent
                .session
                .invoke_macro((name, code_ref), self.args),
            _ => panic!(),
        }
        self.parent
    }
}

pub struct MacroArgActions<'a, F: Frontend + 'a> {
    tokens: Vec<Token>,
    parent: MacroInvocationActions<'a, F>,
}

impl<'a, F: Frontend + 'a> MacroArgActions<'a, F> {
    fn new(parent: MacroInvocationActions<'a, F>) -> MacroArgActions<'a, F> {
        MacroArgActions {
            tokens: Vec::new(),
            parent,
        }
    }
}

impl<'a, F: Frontend + 'a> syntax::TokenSeqContext for MacroArgActions<'a, F> {
    type Token = Token;
    type Parent = MacroInvocationActions<'a, F>;

    fn push_token(&mut self, token: Self::Token) {
        self.tokens.push(token)
    }

    fn exit_token_seq(mut self) -> Self::Parent {
        self.parent.push_arg(self.tokens);
        self.parent
    }
}

fn reduce_include(mut arguments: Vec<SynExpr<Token>>) -> String {
    assert_eq!(arguments.len(), 1);
    let path = arguments.pop().unwrap();
    match path {
        SynExpr::Atom(token::Atom(Atom::String(path_str))) => path_str,
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use frontend::syntax::{CommandContext, FileContext, MacroInvocationContext, TokenSeqContext,
                           keyword::Operand};
    use backend;

    struct TestFrontend(Vec<TestOperation>);

    impl TestFrontend {
        fn new() -> TestFrontend {
            TestFrontend(Vec::new())
        }
    }

    #[derive(Debug, PartialEq)]
    enum TestOperation {
        DefineMacro(String, Vec<Token>),
        EmitItem(backend::Item),
        Include(String),
        InvokeMacro(String, Vec<Vec<Token>>),
        Label(String),
    }

    impl Frontend for TestFrontend {
        type CodeRef = ();

        fn include_source_file(&mut self, filename: String) {
            self.0.push(TestOperation::Include(filename))
        }

        fn emit_item(&mut self, item: backend::Item) {
            self.0.push(TestOperation::EmitItem(item))
        }

        fn define_label(&mut self, (label, _): (String, Self::CodeRef)) {
            self.0.push(TestOperation::Label(label))
        }

        fn define_macro(&mut self, (name, _): (String, Self::CodeRef), tokens: Vec<Token>) {
            self.0.push(TestOperation::DefineMacro(name, tokens))
        }

        fn invoke_macro(&mut self, name: (String, Self::CodeRef), args: Vec<Vec<Token>>) {
            self.0.push(TestOperation::InvokeMacro(name.0, args))
        }
    }

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions.enter_command((Command::Include, ()));
            let expr = SynExpr::from(token::Atom(Atom::String(filename.to_string())));
            command.add_argument(expr);
            command.exit_command();
        });
        assert_eq!(actions, [TestOperation::Include(filename.to_string())])
    }

    #[test]
    fn emit_byte_item() {
        let bytes = [0x42, 0x78];
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions.enter_command((Command::Db, ()));
            for &byte in bytes.iter() {
                command.add_argument(mk_literal(byte))
            }
            command.exit_command();
        });
        assert_eq!(
            actions,
            bytes
                .iter()
                .map(mk_byte)
                .map(TestOperation::EmitItem)
                .collect::<Vec<_>>()
        )
    }

    fn mk_literal(n: i32) -> SynExpr<Token> {
        SynExpr::from(token::Atom(Atom::Number(n)))
    }

    fn mk_byte(byte: &i32) -> backend::Item {
        backend::Item::Byte(backend::Expr::Literal(*byte))
    }

    #[test]
    fn analyze_label() {
        let label = "label";
        let actions =
            collect_semantic_actions(|mut actions| actions.add_label((label.to_string(), ())));
        assert_eq!(actions, [TestOperation::Label(label.to_string())])
    }

    #[test]
    fn define_macro() {
        let name = "my_macro";
        let tokens = vec![
            token::Command(Command::Xor),
            token::Atom(Atom::Operand(Operand::A)),
        ];
        let actions = collect_semantic_actions(|actions| {
            let mut token_seq_context = actions.enter_macro_def((name.to_string(), ()));
            for token in tokens.clone() {
                token_seq_context.push_token(token)
            }
            token_seq_context.exit_token_seq();
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
            let invocation = actions.enter_macro_invocation((Atom::Ident(name.to_string()), ()));
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
        let arg_token = token::Atom(Atom::Operand(Operand::A));
        let actions = collect_semantic_actions(|actions| {
            let mut invocation =
                actions.enter_macro_invocation((Atom::Ident(name.to_string()), ()));
            invocation = {
                let mut arg = invocation.enter_macro_arg();
                arg.push_token(arg_token.clone());
                arg.exit_token_seq()
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
        F: for<'a> FnOnce(SemanticActions<'a, TestFrontend>),
    {
        let mut operations = TestFrontend::new();
        f(SemanticActions::new(&mut operations));
        operations.0
    }
}
