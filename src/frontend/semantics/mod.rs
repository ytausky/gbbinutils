use backend::{self, BinaryOperator};
use diagnostics::{Diagnostic, DiagnosticsListener, Span};
use frontend::session::{ChunkId, Session};
use frontend::syntax::{self, keyword::*, ExprAtom, ExprOperator, Token};
use frontend::{Literal, StrExprFactory};
use Width;

use self::expr::*;

mod instruction;
mod operand;

mod expr {
    use frontend::Literal;

    #[derive(Clone)]
    pub struct Expr<I, S: Clone> {
        pub variant: ExprVariant<I, S>,
        pub span: S,
    }

    #[derive(Clone)]
    pub enum ExprVariant<I, S: Clone> {
        Ident(I),
        Literal(Literal<I>),
        Parentheses(Box<Expr<I, S>>),
    }
}

pub struct SemanticActions<'a, F: Session + 'a> {
    session: &'a mut F,
    expr_factory: StrExprFactory,
    label: Option<(F::Ident, F::Span)>,
}

impl<'a, F: Session + 'a> SemanticActions<'a, F> {
    pub fn new(session: &'a mut F) -> SemanticActions<'a, F> {
        SemanticActions {
            session,
            expr_factory: StrExprFactory::new(),
            label: None,
        }
    }

    fn define_label_if_present(&mut self) {
        if let Some((label, span)) = self.label.take() {
            self.session.define_label((label.into(), span))
        }
    }
}

impl<'a, F: Session + 'a> DiagnosticsListener<F::Span> for SemanticActions<'a, F> {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<F::Span>) {
        self.session.emit_diagnostic(diagnostic)
    }
}

impl<'a, F: Session + 'a> syntax::FileContext<F::Ident, Command, Literal<F::Ident>, F::Span>
    for SemanticActions<'a, F>
{
    type LineActions = Self;

    fn enter_line(mut self, label: Option<(F::Ident, F::Span)>) -> Self::LineActions {
        self.label = label;
        self
    }
}

impl<'a, F: Session + 'a> syntax::LineActions<F::Ident, Command, Literal<F::Ident>, F::Span>
    for SemanticActions<'a, F>
{
    type CommandContext = CommandActions<'a, F>;
    type MacroParamsActions = MacroDefActions<'a, F>;
    type MacroInvocationContext = MacroInvocationActions<'a, F>;
    type Parent = Self;

    fn enter_command(mut self, name: (Command, F::Span)) -> Self::CommandContext {
        self.define_label_if_present();
        CommandActions::new(name, self)
    }

    fn enter_macro_def(mut self) -> Self::MacroParamsActions {
        MacroDefActions::new(self.label.take().unwrap(), self)
    }

    fn enter_macro_invocation(mut self, name: (F::Ident, F::Span)) -> Self::MacroInvocationContext {
        self.define_label_if_present();
        MacroInvocationActions::new(name, self)
    }

    fn exit(mut self) -> Self::Parent {
        self.define_label_if_present();
        self
    }
}

pub struct CommandActions<'a, F: Session + 'a> {
    name: (Command, F::Span),
    args: CommandArgs<F>,
    parent: SemanticActions<'a, F>,
}

type CommandArgs<F> = Vec<Expr<<F as Session>::Ident, <F as Session>::Span>>;

impl<'a, F: Session + 'a> CommandActions<'a, F> {
    fn new(name: (Command, F::Span), parent: SemanticActions<'a, F>) -> CommandActions<'a, F> {
        CommandActions {
            name,
            args: Vec::new(),
            parent,
        }
    }
}

impl<'a, F: Session + 'a> DiagnosticsListener<F::Span> for CommandActions<'a, F> {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<F::Span>) {
        self.parent.emit_diagnostic(diagnostic)
    }
}

impl<'a, F: Session + 'a> syntax::CommandContext<F::Span> for CommandActions<'a, F> {
    type Ident = F::Ident;
    type Command = Command;
    type Literal = Literal<F::Ident>;
    type ArgActions = ExprActions<'a, F>;
    type Parent = SemanticActions<'a, F>;

    fn add_argument(self) -> Self::ArgActions {
        ExprActions {
            stack: Vec::new(),
            parent: self,
        }
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

pub struct ExprActions<'a, F: Session + 'a> {
    stack: Vec<Expr<F::Ident, F::Span>>,
    parent: CommandActions<'a, F>,
}

impl<'a, F: Session + 'a> syntax::ExprActions<F::Span> for ExprActions<'a, F> {
    type Ident = F::Ident;
    type Literal = Literal<F::Ident>;
    type Parent = CommandActions<'a, F>;

    fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, F::Span)) {
        self.stack.push(Expr {
            variant: match atom.0 {
                ExprAtom::Ident(ident) => ExprVariant::Ident(ident),
                ExprAtom::Literal(literal) => ExprVariant::Literal(literal),
            },
            span: atom.1,
        })
    }

    fn apply_operator(&mut self, operator: (ExprOperator, F::Span)) {
        match operator.0 {
            ExprOperator::Parentheses => {
                let inner = self.stack.pop().unwrap();
                self.stack.push(Expr {
                    variant: ExprVariant::Parentheses(Box::new(inner)),
                    span: operator.1,
                })
            }
        }
    }

    fn exit(mut self) -> Self::Parent {
        assert_eq!(self.stack.len(), 1);
        self.parent.args.push(self.stack.pop().unwrap());
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
        Directive::Ds => analyze_ds(args, actions),
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
        let expr = match arg.variant {
            ExprVariant::Literal(literal) => actions.expr_factory.mk_literal((literal, arg.span)),
            ExprVariant::Ident(ident) => actions.expr_factory.mk_symbol((ident, arg.span)),
            _ => panic!(),
        };
        actions.session.emit_item(backend::Item::Data(expr, width))
    }
}

fn analyze_ds<'a, S: Session + 'a>(args: CommandArgs<S>, actions: &mut SemanticActions<'a, S>) {
    use backend::RelocExpr;
    use frontend::ExprFactory;
    let arg = args.into_iter().next().unwrap();
    let count = match arg.variant {
        ExprVariant::Literal(literal) => {
            actions.expr_factory.mk_literal((literal, arg.span.clone()))
        }
        _ => panic!(),
    };
    let expr = RelocExpr::BinaryOperation(
        Box::new(RelocExpr::LocationCounter(arg.span.clone())),
        Box::new(count),
        BinaryOperator::Plus,
        arg.span,
    );
    actions.session.set_origin(expr)
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
    name: (Mnemonic, F::Span),
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
    name: (F::Ident, F::Span),
    params: Vec<(F::Ident, F::Span)>,
    tokens: Vec<(Token<F::Ident>, F::Span)>,
    parent: SemanticActions<'a, F>,
}

impl<'a, F: Session + 'a> MacroDefActions<'a, F> {
    fn new(name: (F::Ident, F::Span), parent: SemanticActions<'a, F>) -> MacroDefActions<'a, F> {
        MacroDefActions {
            name,
            params: Vec::new(),
            tokens: Vec::new(),
            parent,
        }
    }
}

impl<'a, F: Session + 'a> DiagnosticsListener<F::Span> for MacroDefActions<'a, F> {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<F::Span>) {
        self.parent.emit_diagnostic(diagnostic)
    }
}

impl<'a, F: Session + 'a> syntax::MacroParamsActions<F::Span> for MacroDefActions<'a, F> {
    type Ident = F::Ident;
    type Command = Command;
    type Literal = Literal<F::Ident>;
    type MacroBodyActions = Self;
    type Parent = SemanticActions<'a, F>;

    fn add_parameter(&mut self, param: (Self::Ident, F::Span)) {
        self.params.push(param)
    }

    fn exit(self) -> Self::MacroBodyActions {
        self
    }
}

impl<'a, F: Session + 'a> syntax::TokenSeqContext<F::Span> for MacroDefActions<'a, F> {
    type Token = Token<F::Ident>;
    type Parent = SemanticActions<'a, F>;

    fn push_token(&mut self, token: (Self::Token, F::Span)) {
        self.tokens.push(token)
    }

    fn exit(self) -> Self::Parent {
        self.parent
            .session
            .define_macro(self.name, self.params, self.tokens);
        self.parent
    }
}

pub struct MacroInvocationActions<'a, F: Session + 'a> {
    name: (F::Ident, F::Span),
    args: Vec<super::TokenSeq<F::Ident, F::Span>>,
    parent: SemanticActions<'a, F>,
}

impl<'a, F: Session + 'a> MacroInvocationActions<'a, F> {
    fn new(
        name: (F::Ident, F::Span),
        parent: SemanticActions<'a, F>,
    ) -> MacroInvocationActions<'a, F> {
        MacroInvocationActions {
            name,
            args: Vec::new(),
            parent,
        }
    }

    fn push_arg(&mut self, arg: Vec<(Token<F::Ident>, F::Span)>) {
        self.args.push(arg)
    }
}

impl<'a, F: Session + 'a> DiagnosticsListener<F::Span> for MacroInvocationActions<'a, F> {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<F::Span>) {
        self.parent.emit_diagnostic(diagnostic)
    }
}

impl<'a, F: Session + 'a> syntax::MacroInvocationContext<F::Span>
    for MacroInvocationActions<'a, F>
{
    type Token = Token<F::Ident>;
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
    tokens: Vec<(Token<F::Ident>, F::Span)>,
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

impl<'a, F: Session + 'a> DiagnosticsListener<F::Span> for MacroArgActions<'a, F> {
    fn emit_diagnostic(&self, diagnostic: Diagnostic<F::Span>) {
        self.parent.parent.session.emit_diagnostic(diagnostic)
    }
}

impl<'a, F: Session + 'a> syntax::TokenSeqContext<F::Span> for MacroArgActions<'a, F> {
    type Token = Token<F::Ident>;
    type Parent = MacroInvocationActions<'a, F>;

    fn push_token(&mut self, token: (Self::Token, F::Span)) {
        self.tokens.push(token)
    }

    fn exit(mut self) -> Self::Parent {
        self.parent.push_arg(self.tokens);
        self.parent
    }
}

fn reduce_include<I, S: Span>(mut arguments: Vec<Expr<I, S>>) -> ChunkId<I, S> {
    assert_eq!(arguments.len(), 1);
    let path = arguments.pop().unwrap();
    match path.variant {
        ExprVariant::Literal(Literal::String(path_str)) => {
            ChunkId::File((path_str, Some(path.span)))
        }
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use backend;
    use diagnostics::{Diagnostic, Message};
    use frontend::syntax::{
        keyword::Operand, CommandContext, ExprActions, FileContext, LineActions,
        MacroInvocationContext, MacroParamsActions, TokenSeqContext,
    };
    use instruction::RelocExpr;
    use std::borrow::Borrow;
    use std::cell::RefCell;

    struct TestFrontend(RefCell<Vec<TestOperation>>);

    impl TestFrontend {
        fn new() -> TestFrontend {
            TestFrontend(RefCell::new(Vec::new()))
        }
    }

    #[derive(Debug, PartialEq)]
    enum TestOperation {
        AnalyzeChunk(ChunkId<String, ()>),
        DefineMacro(String, Vec<String>, Vec<Token<String>>),
        EmitDiagnostic(Diagnostic<()>),
        EmitItem(backend::Item<()>),
        Label(String),
        SetOrigin(RelocExpr<()>),
    }

    impl Session for TestFrontend {
        type Ident = String;
        type Span = ();

        fn analyze_chunk(&mut self, chunk_id: ChunkId<String, Self::Span>) {
            self.0
                .borrow_mut()
                .push(TestOperation::AnalyzeChunk(chunk_id))
        }

        fn emit_diagnostic(&self, diagnostic: Diagnostic<Self::Span>) {
            self.0
                .borrow_mut()
                .push(TestOperation::EmitDiagnostic(diagnostic))
        }

        fn emit_item(&mut self, item: backend::Item<()>) {
            self.0.borrow_mut().push(TestOperation::EmitItem(item))
        }

        fn define_label(&mut self, (label, _): (String, Self::Span)) {
            self.0.borrow_mut().push(TestOperation::Label(label))
        }

        fn define_macro(
            &mut self,
            (name, _): (impl Into<String>, Self::Span),
            params: Vec<(String, Self::Span)>,
            tokens: Vec<(Token<String>, ())>,
        ) {
            self.0.borrow_mut().push(TestOperation::DefineMacro(
                name.into(),
                params.into_iter().map(|(s, _)| s).collect(),
                tokens.into_iter().map(|(t, _)| t).collect(),
            ))
        }

        fn set_origin(&mut self, origin: RelocExpr<()>) {
            self.0.borrow_mut().push(TestOperation::SetOrigin(origin))
        }
    }

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let actions = collect_semantic_actions(|actions| {
            let mut arg = actions
                .enter_line(None)
                .enter_command((Command::Directive(Directive::Include), ()))
                .add_argument();
            arg.push_atom((ExprAtom::Literal(Literal::String(filename.to_string())), ()));
            arg.exit().exit().exit()
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
            let mut arg = actions
                .enter_line(None)
                .enter_command((Command::Directive(Directive::Org), ()))
                .add_argument();
            arg.push_atom((ExprAtom::Literal(Literal::Number(origin)), ()));
            arg.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [TestOperation::SetOrigin(RelocExpr::Literal(origin, ()))]
        )
    }

    #[test]
    fn emit_byte_items() {
        test_data_items_emission(Directive::Db, mk_byte, [0x42, 0x78])
    }

    #[test]
    fn emit_word_items() {
        test_data_items_emission(Directive::Dw, mk_word, [0x4332, 0x780f])
    }

    fn test_data_items_emission(
        directive: Directive,
        mk_item: impl Fn(&i32) -> backend::Item<()>,
        data: impl Borrow<[i32]>,
    ) {
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions
                .enter_line(None)
                .enter_command((Command::Directive(directive), ()));
            for datum in data.borrow().iter() {
                let mut arg = command.add_argument();
                arg.push_atom(mk_literal(*datum));
                command = arg.exit();
            }
            command.exit().exit()
        });
        assert_eq!(
            actions,
            data.borrow()
                .iter()
                .map(mk_item)
                .map(TestOperation::EmitItem)
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn emit_ld_b_deref_hl() {
        use instruction::*;
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions
                .enter_line(None)
                .enter_command((Command::Mnemonic(Mnemonic::Ld), ()));
            let mut arg1 = command.add_argument();
            arg1.push_atom((ExprAtom::Literal(Literal::Operand(Operand::B)), ()));
            command = arg1.exit();
            let mut arg2 = command.add_argument();
            arg2.push_atom((ExprAtom::Literal(Literal::Operand(Operand::Hl)), ()));
            arg2.apply_operator((ExprOperator::Parentheses, ()));
            arg2.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [TestOperation::EmitItem(backend::Item::Instruction(
                Instruction::Ld(Ld::Simple(SimpleOperand::B, SimpleOperand::DerefHl))
            ))]
        )
    }

    #[test]
    fn emit_label_word() {
        let label = "my_label";
        let actions = collect_semantic_actions(|actions| {
            let mut arg = actions
                .enter_line(None)
                .enter_command((Command::Directive(Directive::Dw), ()))
                .add_argument();
            arg.push_atom((ExprAtom::Ident(label.to_string()), ()));
            arg.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [TestOperation::EmitItem(backend::Item::Data(
                RelocExpr::Symbol(label.to_string(), ()),
                Width::Word
            ))]
        );
    }

    fn mk_literal(n: i32) -> (ExprAtom<String, Literal<String>>, ()) {
        (ExprAtom::Literal(Literal::Number(n)), ())
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
        let actions = collect_semantic_actions(|actions| {
            actions.enter_line(Some((label.to_string(), ()))).exit()
        });
        assert_eq!(actions, [TestOperation::Label(label.to_string())])
    }

    #[test]
    fn define_nullary_macro() {
        test_macro_definition(
            "my_macro",
            [],
            [
                Token::Command(Command::Mnemonic(Mnemonic::Xor)),
                Token::Literal(Literal::Operand(Operand::A)),
            ],
        )
    }

    #[test]
    fn define_unary_macro() {
        let param = "reg";
        test_macro_definition(
            "my_xor",
            [param],
            [
                Token::Command(Command::Mnemonic(Mnemonic::Xor)),
                Token::Ident(param.to_string()),
            ],
        )
    }

    fn test_macro_definition(
        name: &str,
        params: impl Borrow<[&'static str]>,
        body: impl Borrow<[Token<String>]>,
    ) {
        let actions = collect_semantic_actions(|actions| {
            let mut params_actions = actions
                .enter_line(Some((name.to_string(), ())))
                .enter_macro_def();
            for param in params.borrow().iter().map(|t| (t.to_string(), ())) {
                params_actions.add_parameter(param)
            }
            let mut token_seq_actions = MacroParamsActions::exit(params_actions);
            for token in body.borrow().iter().cloned().map(|t| (t, ())) {
                token_seq_actions.push_token(token)
            }
            TokenSeqContext::exit(token_seq_actions)
        });
        assert_eq!(
            actions,
            [TestOperation::DefineMacro(
                name.to_string(),
                params.borrow().iter().cloned().map(String::from).collect(),
                body.borrow().iter().cloned().collect()
            )]
        )
    }

    #[test]
    fn invoke_nullary_macro() {
        let name = "my_macro";
        let actions = collect_semantic_actions(|actions| {
            let invocation = actions
                .enter_line(None)
                .enter_macro_invocation((name.to_string(), ()));
            invocation.exit().exit()
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
        let arg_token = Token::Literal(Literal::Operand(Operand::A));
        let actions = collect_semantic_actions(|actions| {
            let mut invocation = actions
                .enter_line(None)
                .enter_macro_invocation((name.to_string(), ()));
            invocation = {
                let mut arg = invocation.enter_macro_arg();
                arg.push_token((arg_token.clone(), ()));
                arg.exit()
            };
            invocation.exit().exit()
        });
        assert_eq!(
            actions,
            [TestOperation::AnalyzeChunk(ChunkId::Macro {
                name: (name.to_string(), ()),
                args: vec![vec![(arg_token, ())]],
            })]
        )
    }

    #[test]
    fn diagnose_wrong_operand_count() {
        let actions = collect_semantic_actions(|actions| {
            let mut arg = actions
                .enter_line(None)
                .enter_command((Command::Mnemonic(Mnemonic::Nop), ()))
                .add_argument();
            let literal_a = Literal::Operand(Operand::A);
            arg.push_atom((ExprAtom::Literal(literal_a), ()));
            arg.exit().exit().exit()
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

    #[test]
    fn diagnose_parsing_error() {
        let diagnostic = Diagnostic::new(Message::UnexpectedToken { token: () }, ());
        let actions = collect_semantic_actions(|actions| {
            let stmt = actions.enter_line(None);
            stmt.emit_diagnostic(diagnostic.clone());
            stmt.exit()
        });
        assert_eq!(actions, [TestOperation::EmitDiagnostic(diagnostic)])
    }

    #[test]
    fn reserve_3_bytes() {
        let actions = collect_semantic_actions(|actions| {
            let mut arg = actions
                .enter_line(None)
                .enter_command((Command::Directive(Directive::Ds), ()))
                .add_argument();
            arg.push_atom(mk_literal(3));
            arg.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [TestOperation::SetOrigin(RelocExpr::BinaryOperation(
                Box::new(RelocExpr::LocationCounter(())),
                Box::new(RelocExpr::Literal(3, ())),
                BinaryOperator::Plus,
                ()
            ))]
        )
    }

    fn collect_semantic_actions<F>(f: F) -> Vec<TestOperation>
    where
        F: for<'a> FnOnce(SemanticActions<'a, TestFrontend>) -> SemanticActions<'a, TestFrontend>,
    {
        let mut operations = TestFrontend::new();
        f(SemanticActions::new(&mut operations));
        operations.0.into_inner()
    }
}
