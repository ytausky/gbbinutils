use backend;
use frontend::syntax::{self, token, SynExpr, Token, TokenSpec, keyword::Command};
use frontend::{Atom, ChunkId, Frontend, StrExprFactory};

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

impl<'a, F: Frontend + 'a> syntax::FileContext<String, F::TokenRef> for SemanticActions<'a, F> {
    type CommandContext = CommandActions<'a, F>;
    type MacroDefContext = MacroDefActions<'a, F>;
    type MacroInvocationContext = MacroInvocationActions<'a, F>;

    fn add_label(&mut self, label: (<String as TokenSpec>::Label, F::TokenRef)) {
        self.session.define_label(label);
    }

    fn enter_command(
        self,
        name: (<String as TokenSpec>::Command, F::TokenRef),
    ) -> Self::CommandContext {
        CommandActions::new(name, self)
    }

    fn enter_macro_def(
        self,
        name: (<String as TokenSpec>::Label, F::TokenRef),
    ) -> Self::MacroDefContext {
        MacroDefActions::new(name, self)
    }

    fn enter_macro_invocation(
        self,
        name: (<String as TokenSpec>::Atom, F::TokenRef),
    ) -> Self::MacroInvocationContext {
        MacroInvocationActions::new(name, self)
    }
}

pub struct CommandActions<'a, F: Frontend + 'a> {
    name: (Command, F::TokenRef),
    args: CommandArgs<F>,
    parent: SemanticActions<'a, F>,
}

type CommandArgs<F> = Vec<SynExpr<(syntax::Token, <F as Frontend>::TokenRef)>>;

impl<'a, F: Frontend + 'a> CommandActions<'a, F> {
    fn new(name: (Command, F::TokenRef), parent: SemanticActions<'a, F>) -> CommandActions<'a, F> {
        CommandActions {
            name,
            args: Vec::new(),
            parent,
        }
    }
}

impl<'a, F: Frontend + 'a> syntax::CommandContext<F::TokenRef> for CommandActions<'a, F> {
    type Token = Token;
    type Parent = SemanticActions<'a, F>;

    fn add_argument(&mut self, expr: SynExpr<(Self::Token, F::TokenRef)>) {
        self.args.push(expr)
    }

    fn exit(mut self) -> Self::Parent {
        match self.name {
            (Command::Db, _) => analyze_db(self.args, &mut self.parent),
            (Command::Include, _) => analyze_include(self.args, &mut self.parent),
            name => analyze_command(name, self.args, &mut self.parent),
        }
        self.parent
    }
}

fn analyze_db<'a, F: Frontend + 'a>(args: CommandArgs<F>, actions: &mut SemanticActions<'a, F>) {
    for arg in args {
        match arg {
            SynExpr::Atom(atom) => {
                use frontend::ExprFactory;
                let expr = actions.expr_factory.mk_atom(atom);
                actions.session.emit_item(backend::Item::Byte(expr))
            }
            _ => panic!(),
        }
    }
}

fn analyze_include<'a, F: Frontend + 'a>(
    args: CommandArgs<F>,
    actions: &mut SemanticActions<'a, F>,
) {
    actions.session.analyze_chunk(reduce_include(args));
}

fn analyze_command<'a, F: Frontend + 'a>(
    name: (Command, F::TokenRef),
    args: CommandArgs<F>,
    actions: &mut SemanticActions<'a, F>,
) {
    let mut analyzer = instruction::CommandAnalyzer::new(&mut actions.expr_factory);
    let analysis_result = analyzer.analyze_instruction(name, args.into_iter());
    match analysis_result {
        Ok(instruction) => actions
            .session
            .emit_item(backend::Item::Instruction(instruction)),
        Err(diagnostic) => actions.session.emit_diagnostic(diagnostic),
    }
}

pub struct MacroDefActions<'a, F: Frontend + 'a> {
    name: (String, F::TokenRef),
    tokens: Vec<(Token, F::TokenRef)>,
    parent: SemanticActions<'a, F>,
}

impl<'a, F: Frontend + 'a> MacroDefActions<'a, F> {
    fn new(name: (String, F::TokenRef), parent: SemanticActions<'a, F>) -> MacroDefActions<'a, F> {
        MacroDefActions {
            name,
            tokens: Vec::new(),
            parent,
        }
    }
}

impl<'a, F: Frontend + 'a> syntax::TokenSeqContext<F::TokenRef> for MacroDefActions<'a, F> {
    type Token = Token;
    type Parent = SemanticActions<'a, F>;

    fn push_token(&mut self, token: (Self::Token, F::TokenRef)) {
        self.tokens.push(token)
    }

    fn exit(self) -> Self::Parent {
        self.parent.session.define_macro(self.name, self.tokens);
        self.parent
    }
}

pub struct MacroInvocationActions<'a, F: Frontend + 'a> {
    name: (Atom<String>, F::TokenRef),
    args: Vec<Vec<Token>>,
    parent: SemanticActions<'a, F>,
}

impl<'a, F: Frontend + 'a> MacroInvocationActions<'a, F> {
    fn new(
        name: (Atom<String>, F::TokenRef),
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

impl<'a, F: Frontend + 'a> syntax::MacroInvocationContext<F::TokenRef>
    for MacroInvocationActions<'a, F>
{
    type Token = Token;
    type Parent = SemanticActions<'a, F>;
    type MacroArgContext = MacroArgActions<'a, F>;

    fn enter_macro_arg(self) -> Self::MacroArgContext {
        MacroArgActions::new(self)
    }

    fn exit(self) -> Self::Parent {
        match self.name {
            (Atom::Ident(name), code_ref) => self.parent.session.analyze_chunk(ChunkId::Macro {
                name: (name, code_ref),
                args: self.args,
            }),
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

impl<'a, F: Frontend + 'a> syntax::TokenSeqContext<F::TokenRef> for MacroArgActions<'a, F> {
    type Token = Token;
    type Parent = MacroInvocationActions<'a, F>;

    fn push_token(&mut self, token: (Self::Token, F::TokenRef)) {
        self.tokens.push(token.0)
    }

    fn exit(mut self) -> Self::Parent {
        self.parent.push_arg(self.tokens);
        self.parent
    }
}

fn reduce_include<R>(mut arguments: Vec<SynExpr<(Token, R)>>) -> ChunkId<R> {
    assert_eq!(arguments.len(), 1);
    let path = arguments.pop().unwrap();
    match path {
        SynExpr::Atom((token::Atom(Atom::String(path_str)), token_ref)) => {
            ChunkId::File((path_str, Some(token_ref)))
        }
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use backend;
    use diagnostics::Diagnostic;
    use frontend::ChunkId;
    use frontend::syntax::{CommandContext, FileContext, MacroInvocationContext, TokenSeqContext,
                           keyword::Operand};

    struct TestFrontend(Vec<TestOperation>);

    impl TestFrontend {
        fn new() -> TestFrontend {
            TestFrontend(Vec::new())
        }
    }

    #[derive(Debug, PartialEq)]
    enum TestOperation {
        AnalyzeChunk(ChunkId<()>),
        DefineMacro(String, Vec<Token>),
        EmitDiagnostic(Diagnostic<()>),
        EmitItem(backend::Item<()>),
        Label(String),
    }

    impl Frontend for TestFrontend {
        type TokenRef = ();

        fn analyze_chunk(&mut self, chunk_id: ChunkId<Self::TokenRef>) {
            self.0.push(TestOperation::AnalyzeChunk(chunk_id))
        }

        fn emit_diagnostic(&mut self, diagnostic: Diagnostic<Self::TokenRef>) {
            self.0.push(TestOperation::EmitDiagnostic(diagnostic))
        }

        fn emit_item(&mut self, item: backend::Item<()>) {
            self.0.push(TestOperation::EmitItem(item))
        }

        fn define_label(&mut self, (label, _): (String, Self::TokenRef)) {
            self.0.push(TestOperation::Label(label))
        }

        fn define_macro(
            &mut self,
            (name, _): (impl Into<String>, Self::TokenRef),
            tokens: Vec<(Token, ())>,
        ) {
            self.0.push(TestOperation::DefineMacro(
                name.into(),
                tokens.into_iter().map(|(t, _)| t).collect(),
            ))
        }
    }

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions.enter_command((Command::Include, ()));
            let expr = SynExpr::from((token::Atom(Atom::String(filename.to_string())), ()));
            command.add_argument(expr);
            command.exit();
        });
        assert_eq!(
            actions,
            [
                TestOperation::AnalyzeChunk(ChunkId::File((filename.to_string(), Some(()))))
            ]
        )
    }

    #[test]
    fn emit_byte_item() {
        let bytes = [0x42, 0x78];
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions.enter_command((Command::Db, ()));
            for &byte in bytes.iter() {
                command.add_argument(mk_literal(byte))
            }
            command.exit();
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

    fn mk_literal(n: i32) -> SynExpr<(Token, ())> {
        SynExpr::from((token::Atom(Atom::Number(n)), ()))
    }

    fn mk_byte(byte: &i32) -> backend::Item<()> {
        backend::Item::Byte(backend::Expr::Literal(*byte, ()))
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
            for token in tokens.iter().cloned().map(|t| (t, ())) {
                token_seq_context.push_token(token)
            }
            token_seq_context.exit();
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
            invocation.exit();
        });
        assert_eq!(
            actions,
            [
                TestOperation::AnalyzeChunk(ChunkId::Macro {
                    name: (name.to_string(), ()),
                    args: Vec::new(),
                })
            ]
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
                arg.push_token((arg_token.clone(), ()));
                arg.exit()
            };
            invocation.exit();
        });
        assert_eq!(
            actions,
            [
                TestOperation::AnalyzeChunk(ChunkId::Macro {
                    name: (name.to_string(), ()),
                    args: vec![vec![arg_token]],
                }),
            ]
        )
    }

    #[test]
    fn diagnoze_wrong_operand_count() {
        use diagnostics::{Diagnostic, Message};
        let actions = collect_semantic_actions(|actions| {
            let mut command_context = actions.enter_command((Command::Nop, ()));
            let token_a = token::Atom(Atom::Operand(Operand::A));
            command_context.add_argument((token_a, ()).into());
            command_context.exit();
        });
        assert_eq!(
            actions,
            [
                TestOperation::EmitDiagnostic(Diagnostic::new(
                    Message::OperandCount {
                        actual: 1,
                        expected: 0
                    },
                    ()
                )),
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
