use backend;
use frontend::syntax::{self, keyword::*, ExprNode, ParsedExpr, Token, TokenSpec};
use frontend::{Literal, StrExprFactory};
use session::{ChunkId, Session};
use Width;

mod instruction;
mod operand;

pub struct SemanticActions<'a, F: Session + 'a> {
    session: &'a mut F,
    expr_factory: StrExprFactory,
}

impl<'a, F: Session + 'a> SemanticActions<'a, F> {
    pub fn new(session: &'a mut F) -> SemanticActions<'a, F> {
        SemanticActions {
            session,
            expr_factory: StrExprFactory::new(),
        }
    }
}

impl<'a, F: Session + 'a> syntax::FileContext<String, F::TokenRef> for SemanticActions<'a, F> {
    type CommandContext = CommandActions<'a, F>;
    type MacroDefContext = MacroDefActions<'a, F>;
    type MacroInvocationContext = MacroInvocationActions<'a, F>;

    fn add_label(&mut self, label: (<String as TokenSpec>::Ident, F::TokenRef)) {
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
        name: (<String as TokenSpec>::Ident, F::TokenRef),
    ) -> Self::MacroDefContext {
        MacroDefActions::new(name, self)
    }

    fn enter_macro_invocation(
        self,
        name: (<String as TokenSpec>::Ident, F::TokenRef),
    ) -> Self::MacroInvocationContext {
        MacroInvocationActions::new(name, self)
    }
}

pub struct CommandActions<'a, F: Session + 'a> {
    name: (Command, F::TokenRef),
    args: CommandArgs<F>,
    parent: SemanticActions<'a, F>,
}

type CommandArgs<F> = Vec<ParsedExpr<String, <F as Session>::TokenRef>>;

impl<'a, F: Session + 'a> CommandActions<'a, F> {
    fn new(name: (Command, F::TokenRef), parent: SemanticActions<'a, F>) -> CommandActions<'a, F> {
        CommandActions {
            name,
            args: Vec::new(),
            parent,
        }
    }
}

impl<'a, F: Session + 'a> syntax::CommandContext<F::TokenRef> for CommandActions<'a, F> {
    type TokenSpec = String;
    type Parent = SemanticActions<'a, F>;

    fn add_argument(&mut self, expr: ParsedExpr<String, F::TokenRef>) {
        self.args.push(expr)
    }

    fn exit(mut self) -> Self::Parent {
        match self.name {
            (Command::Directive(directive), _) => {
                analyze_directive(directive, self.args, &mut self.parent)
            }
            (Command::Mnemonic(mnemonic), range) => {
                analyze_mnemonic((mnemonic, range), self.args, &mut self.parent)
            }
        }
        self.parent
    }
}

fn analyze_directive<'a, S: Session + 'a>(
    directive: Directive,
    args: CommandArgs<S>,
    actions: &mut SemanticActions<'a, S>,
) {
    match directive {
        Directive::Db => analyze_data(Width::Byte, args, actions),
        Directive::Dw => analyze_data(Width::Word, args, actions),
        Directive::Include => analyze_include(args, actions),
        Directive::Org => analyze_org(args, actions),
    }
}

fn analyze_data<'a, S: Session + 'a>(
    width: Width,
    args: CommandArgs<S>,
    actions: &mut SemanticActions<'a, S>,
) {
    for arg in args {
        use frontend::ExprFactory;
        let expr = match arg.node {
            ExprNode::Literal(literal) => actions.expr_factory.mk_literal((literal, arg.interval)),
            ExprNode::Ident(ident) => actions.expr_factory.mk_symbol((ident, arg.interval)),
            _ => panic!(),
        };
        actions.session.emit_item(backend::Item::Data(expr, width))
    }
}

fn analyze_include<'a, F: Session + 'a>(
    args: CommandArgs<F>,
    actions: &mut SemanticActions<'a, F>,
) {
    actions.session.analyze_chunk(reduce_include(args));
}

fn analyze_org<'a, S: Session + 'a>(args: CommandArgs<S>, actions: &mut SemanticActions<'a, S>) {
    let mut args = args.into_iter();
    let expr = operand::analyze_reloc_expr(args.next().unwrap()).unwrap();
    assert!(args.next().is_none());
    actions.session.set_origin(expr)
}

fn analyze_mnemonic<'a, F: Session + 'a>(
    name: (Mnemonic, F::TokenRef),
    args: CommandArgs<F>,
    actions: &mut SemanticActions<'a, F>,
) {
    let analysis_result = instruction::analyze_instruction(name, args.into_iter());
    match analysis_result {
        Ok(instruction) => actions
            .session
            .emit_item(backend::Item::Instruction(instruction)),
        Err(diagnostic) => actions.session.emit_diagnostic(diagnostic),
    }
}

pub struct MacroDefActions<'a, F: Session + 'a> {
    name: (String, F::TokenRef),
    tokens: Vec<(Token, F::TokenRef)>,
    parent: SemanticActions<'a, F>,
}

impl<'a, F: Session + 'a> MacroDefActions<'a, F> {
    fn new(name: (String, F::TokenRef), parent: SemanticActions<'a, F>) -> MacroDefActions<'a, F> {
        MacroDefActions {
            name,
            tokens: Vec::new(),
            parent,
        }
    }
}

impl<'a, F: Session + 'a> syntax::TokenSeqContext<F::TokenRef> for MacroDefActions<'a, F> {
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

pub struct MacroInvocationActions<'a, F: Session + 'a> {
    name: (String, F::TokenRef),
    args: Vec<Vec<Token>>,
    parent: SemanticActions<'a, F>,
}

impl<'a, F: Session + 'a> MacroInvocationActions<'a, F> {
    fn new(
        name: (String, F::TokenRef),
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

impl<'a, F: Session + 'a> syntax::MacroInvocationContext<F::TokenRef>
    for MacroInvocationActions<'a, F>
{
    type Token = Token;
    type Parent = SemanticActions<'a, F>;
    type MacroArgContext = MacroArgActions<'a, F>;

    fn enter_macro_arg(self) -> Self::MacroArgContext {
        MacroArgActions::new(self)
    }

    fn exit(self) -> Self::Parent {
        self.parent.session.analyze_chunk(ChunkId::Macro {
            name: self.name,
            args: self.args,
        });
        self.parent
    }
}

pub struct MacroArgActions<'a, F: Session + 'a> {
    tokens: Vec<Token>,
    parent: MacroInvocationActions<'a, F>,
}

impl<'a, F: Session + 'a> MacroArgActions<'a, F> {
    fn new(parent: MacroInvocationActions<'a, F>) -> MacroArgActions<'a, F> {
        MacroArgActions {
            tokens: Vec::new(),
            parent,
        }
    }
}

impl<'a, F: Session + 'a> syntax::TokenSeqContext<F::TokenRef> for MacroArgActions<'a, F> {
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

fn reduce_include<R>(mut arguments: Vec<ParsedExpr<String, R>>) -> ChunkId<R> {
    assert_eq!(arguments.len(), 1);
    let path = arguments.pop().unwrap();
    match path.node {
        ExprNode::Literal(Literal::String(path_str)) => {
            ChunkId::File((path_str, Some(path.interval)))
        }
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use backend;
    use diagnostics::Diagnostic;
    use frontend::syntax::{
        keyword::Operand, token, CommandContext, FileContext, MacroInvocationContext,
        TokenSeqContext,
    };
    use instruction::RelocExpr;

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
        SetOrigin(RelocExpr<()>),
    }

    impl Session for TestFrontend {
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

        fn set_origin(&mut self, origin: RelocExpr<()>) {
            self.0.push(TestOperation::SetOrigin(origin))
        }
    }

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions.enter_command((Command::Directive(Directive::Include), ()));
            let expr = ParsedExpr {
                node: ExprNode::Literal(Literal::String(filename.to_string())),
                interval: (),
            };
            command.add_argument(expr);
            command.exit();
        });
        assert_eq!(
            actions,
            [TestOperation::AnalyzeChunk(ChunkId::File((
                filename.to_string(),
                Some(())
            )))]
        )
    }

    #[test]
    fn set_origin() {
        let origin = 0x3000;
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions.enter_command((Command::Directive(Directive::Org), ()));
            let expr = ParsedExpr {
                node: ExprNode::Literal(Literal::Number(origin)),
                interval: (),
            };
            command.add_argument(expr);
            command.exit();
        });
        assert_eq!(
            actions,
            [TestOperation::SetOrigin(RelocExpr::Literal(origin, ()))]
        )
    }

    #[test]
    fn emit_byte_items() {
        let bytes = [0x42, 0x78];
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions.enter_command((Command::Directive(Directive::Db), ()));
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

    #[test]
    fn emit_word_items() {
        let words = [0x4332, 0x780f];
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions.enter_command((Command::Directive(Directive::Dw), ()));
            for &word in words.iter() {
                command.add_argument(mk_literal(word))
            }
            command.exit();
        });
        assert_eq!(
            actions,
            words
                .iter()
                .map(mk_word)
                .map(TestOperation::EmitItem)
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn emit_label_word() {
        let label = "my_label";
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions.enter_command((Command::Directive(Directive::Dw), ()));
            command.add_argument(ParsedExpr {
                node: ExprNode::Ident(label.to_string()),
                interval: (),
            });
            command.exit();
        });
        assert_eq!(
            actions,
            [TestOperation::EmitItem(backend::Item::Data(
                RelocExpr::Symbol(label.to_string(), ()),
                Width::Word
            ))]
        );
    }

    fn mk_literal(n: i32) -> ParsedExpr<String, ()> {
        ParsedExpr {
            node: ExprNode::Literal(Literal::Number(n)),
            interval: (),
        }
    }

    fn mk_byte(byte: &i32) -> backend::Item<()> {
        backend::Item::Data(RelocExpr::Literal(*byte, ()), Width::Byte)
    }

    fn mk_word(word: &i32) -> backend::Item<()> {
        backend::Item::Data(RelocExpr::Literal(*word, ()), Width::Word)
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
            token::Command(Command::Mnemonic(Mnemonic::Xor)),
            token::Literal(Literal::Operand(Operand::A)),
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
            let invocation = actions.enter_macro_invocation((name.to_string(), ()));
            invocation.exit();
        });
        assert_eq!(
            actions,
            [TestOperation::AnalyzeChunk(ChunkId::Macro {
                name: (name.to_string(), ()),
                args: Vec::new(),
            })]
        )
    }

    #[test]
    fn invoke_unary_macro() {
        let name = "my_macro";
        let arg_token = token::Literal(Literal::Operand(Operand::A));
        let actions = collect_semantic_actions(|actions| {
            let mut invocation = actions.enter_macro_invocation((name.to_string(), ()));
            invocation = {
                let mut arg = invocation.enter_macro_arg();
                arg.push_token((arg_token.clone(), ()));
                arg.exit()
            };
            invocation.exit();
        });
        assert_eq!(
            actions,
            [TestOperation::AnalyzeChunk(ChunkId::Macro {
                name: (name.to_string(), ()),
                args: vec![vec![arg_token]],
            })]
        )
    }

    #[test]
    fn diagnoze_wrong_operand_count() {
        use diagnostics::{Diagnostic, Message};
        let actions = collect_semantic_actions(|actions| {
            let mut command_context = actions.enter_command((Command::Mnemonic(Mnemonic::Nop), ()));
            let literal_a = Literal::Operand(Operand::A);
            command_context.add_argument(ParsedExpr {
                node: ExprNode::Literal(literal_a),
                interval: (),
            });
            command_context.exit();
        });
        assert_eq!(
            actions,
            [TestOperation::EmitDiagnostic(Diagnostic::new(
                Message::OperandCount {
                    actual: 1,
                    expected: 0
                },
                ()
            ))]
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
