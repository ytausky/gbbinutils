use crate::backend::{self, BinaryOperator, RelocExpr};
use crate::diagnostics::{DiagnosticsListener, InternalDiagnostic, Message};
use crate::expr::ExprVariant;
use crate::frontend::session::Session;
use crate::frontend::syntax::{self, keyword::*, ExprAtom, ExprOperator, Token};
use crate::frontend::{ExprFactory, Literal, StrExprFactory};

mod directive;
mod instruction;
mod operand;

mod expr {
    use crate::expr::Expr;
    #[cfg(test)]
    use crate::expr::ExprVariant;
    use crate::frontend::Literal;

    #[derive(Debug, PartialEq)]
    pub enum SemanticAtom<I> {
        Ident(I),
        Literal(Literal<I>),
    }

    impl<I> From<Literal<I>> for SemanticAtom<I> {
        fn from(literal: Literal<I>) -> Self {
            SemanticAtom::Literal(literal)
        }
    }

    #[derive(Debug, PartialEq)]
    pub enum SemanticUnary {
        Parentheses,
    }

    #[derive(Debug, PartialEq)]
    pub enum SemanticBinary {
        Plus,
    }

    pub type SemanticExpr<I, S> = Expr<SemanticAtom<I>, SemanticUnary, SemanticBinary, S>;

    #[cfg(test)]
    pub type SemanticExprVariant<I, S> =
        ExprVariant<SemanticAtom<I>, SemanticUnary, SemanticBinary, S>;
}

use self::expr::*;

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
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<F::Span>) {
        self.session.emit_diagnostic(diagnostic)
    }
}

impl<'a, F: Session + 'a> syntax::FileContext<F::Ident, Command, Literal<F::Ident>, F::Span>
    for SemanticActions<'a, F>
{
    type StmtContext = Self;

    fn enter_stmt(mut self, label: Option<(F::Ident, F::Span)>) -> Self::StmtContext {
        self.label = label;
        self
    }
}

impl<'a, F: Session + 'a> syntax::StmtContext<F::Ident, Command, Literal<F::Ident>, F::Span>
    for SemanticActions<'a, F>
{
    type CommandContext = CommandActions<'a, F>;
    type MacroParamsContext = MacroDefActions<'a, F>;
    type MacroInvocationContext = MacroInvocationActions<'a, F>;
    type Parent = Self;

    fn enter_command(mut self, name: (Command, F::Span)) -> Self::CommandContext {
        self.define_label_if_present();
        CommandActions::new(name, self)
    }

    fn enter_macro_def(mut self, keyword: F::Span) -> Self::MacroParamsContext {
        if self.label.is_none() {
            self.emit_diagnostic(InternalDiagnostic::new(Message::MacroRequiresName, keyword))
        }
        MacroDefActions::new(self.label.take(), self)
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
    has_errors: bool,
}

type CommandArgs<F> = Vec<SemanticExpr<<F as Session>::Ident, <F as Session>::Span>>;

impl<'a, F: Session + 'a> CommandActions<'a, F> {
    fn new(name: (Command, F::Span), parent: SemanticActions<'a, F>) -> CommandActions<'a, F> {
        CommandActions {
            name,
            args: Vec::new(),
            parent,
            has_errors: false,
        }
    }
}

impl<'a, F: Session + 'a> DiagnosticsListener<F::Span> for CommandActions<'a, F> {
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<F::Span>) {
        self.has_errors = true;
        self.parent.emit_diagnostic(diagnostic)
    }
}

impl<'a, F: Session + 'a> syntax::CommandContext<F::Span> for CommandActions<'a, F> {
    type Ident = F::Ident;
    type Command = Command;
    type Literal = Literal<F::Ident>;
    type ArgContext = ExprContext<'a, F>;
    type Parent = SemanticActions<'a, F>;

    fn add_argument(self) -> Self::ArgContext {
        ExprContext {
            stack: Vec::new(),
            parent: self,
        }
    }

    fn exit(mut self) -> Self::Parent {
        if !self.has_errors {
            let result = match self.name {
                (Command::Directive(directive), span) => {
                    directive::analyze_directive((directive, span), self.args, &mut self.parent)
                }
                (Command::Mnemonic(mnemonic), range) => {
                    analyze_mnemonic((mnemonic, range), self.args, &mut self.parent)
                }
            };
            if let Err(diagnostic) = result {
                self.parent.emit_diagnostic(diagnostic);
            }
        }
        self.parent
    }
}

pub struct ExprContext<'a, F: Session + 'a> {
    stack: Vec<SemanticExpr<F::Ident, F::Span>>,
    parent: CommandActions<'a, F>,
}

impl<'a, F: Session + 'a> DiagnosticsListener<F::Span> for ExprContext<'a, F> {
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<F::Span>) {
        self.parent.emit_diagnostic(diagnostic)
    }
}

impl<'a, F: Session + 'a> syntax::ExprContext<F::Span> for ExprContext<'a, F> {
    type Ident = F::Ident;
    type Literal = Literal<F::Ident>;
    type Parent = CommandActions<'a, F>;

    fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, F::Span)) {
        self.stack.push(SemanticExpr {
            variant: ExprVariant::Atom(match atom.0 {
                ExprAtom::Ident(ident) => SemanticAtom::Ident(ident),
                ExprAtom::Literal(literal) => SemanticAtom::Literal(literal),
            }),
            span: atom.1,
        })
    }

    fn apply_operator(&mut self, operator: (ExprOperator, F::Span)) {
        match operator.0 {
            ExprOperator::Parentheses => {
                let inner = self.stack.pop().unwrap_or_else(|| unreachable!());
                self.stack.push(SemanticExpr {
                    variant: ExprVariant::Unary(SemanticUnary::Parentheses, Box::new(inner)),
                    span: operator.1,
                })
            }
            ExprOperator::Plus => {
                let rhs = self.stack.pop().unwrap_or_else(|| unreachable!());
                let lhs = self.stack.pop().unwrap_or_else(|| unreachable!());
                self.stack.push(SemanticExpr {
                    variant: ExprVariant::Binary(
                        SemanticBinary::Plus,
                        Box::new(lhs),
                        Box::new(rhs),
                    ),
                    span: operator.1,
                })
            }
        }
    }

    fn exit(mut self) -> Self::Parent {
        if !self.parent.has_errors {
            assert_eq!(self.stack.len(), 1);
            self.parent.args.push(self.stack.pop().unwrap());
        }
        self.parent
    }
}

fn analyze_mnemonic<'a, F: Session + 'a>(
    name: (Mnemonic, F::Span),
    args: CommandArgs<F>,
    actions: &mut SemanticActions<'a, F>,
) -> Result<(), InternalDiagnostic<F::Span>> {
    let instruction =
        instruction::analyze_instruction(name, args.into_iter(), &mut actions.expr_factory)?;
    actions
        .session
        .emit_item(backend::Item::Instruction(instruction));
    Ok(())
}

pub struct MacroDefActions<'a, F: Session + 'a> {
    name: Option<(F::Ident, F::Span)>,
    params: Vec<(F::Ident, F::Span)>,
    tokens: Vec<(Token<F::Ident>, F::Span)>,
    parent: SemanticActions<'a, F>,
}

impl<'a, F: Session + 'a> MacroDefActions<'a, F> {
    fn new(
        name: Option<(F::Ident, F::Span)>,
        parent: SemanticActions<'a, F>,
    ) -> MacroDefActions<'a, F> {
        MacroDefActions {
            name,
            params: Vec::new(),
            tokens: Vec::new(),
            parent,
        }
    }
}

impl<'a, F: Session + 'a> DiagnosticsListener<F::Span> for MacroDefActions<'a, F> {
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<F::Span>) {
        self.parent.emit_diagnostic(diagnostic)
    }
}

impl<'a, F: Session + 'a> syntax::MacroParamsContext<F::Span> for MacroDefActions<'a, F> {
    type Ident = F::Ident;
    type Command = Command;
    type Literal = Literal<F::Ident>;
    type MacroBodyContext = Self;
    type Parent = SemanticActions<'a, F>;

    fn add_parameter(&mut self, param: (Self::Ident, F::Span)) {
        self.params.push(param)
    }

    fn exit(self) -> Self::MacroBodyContext {
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
        if let Some(name) = self.name {
            self.parent
                .session
                .define_macro(name, self.params, self.tokens)
        }
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
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<F::Span>) {
        self.parent.emit_diagnostic(diagnostic)
    }
}

impl<'a, F: Session + 'a> syntax::MacroInvocationContext<F::Span>
    for MacroInvocationActions<'a, F>
{
    type Token = Token<F::Ident>;
    type Parent = SemanticActions<'a, F>;
    type MacroArgContext = MacroArgContext<'a, F>;

    fn enter_macro_arg(self) -> Self::MacroArgContext {
        MacroArgContext::new(self)
    }

    fn exit(self) -> Self::Parent {
        self.parent.session.invoke_macro(self.name, self.args);
        self.parent
    }
}

pub struct MacroArgContext<'a, F: Session + 'a> {
    tokens: Vec<(Token<F::Ident>, F::Span)>,
    parent: MacroInvocationActions<'a, F>,
}

impl<'a, F: Session + 'a> MacroArgContext<'a, F> {
    fn new(parent: MacroInvocationActions<'a, F>) -> MacroArgContext<'a, F> {
        MacroArgContext {
            tokens: Vec::new(),
            parent,
        }
    }
}

impl<'a, F: Session + 'a> DiagnosticsListener<F::Span> for MacroArgContext<'a, F> {
    fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<F::Span>) {
        self.parent.parent.session.emit_diagnostic(diagnostic)
    }
}

impl<'a, F: Session + 'a> syntax::TokenSeqContext<F::Span> for MacroArgContext<'a, F> {
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

fn analyze_reloc_expr<I: Into<String>, S: Clone>(
    expr: SemanticExpr<I, S>,
    factory: &mut impl ExprFactory,
) -> Result<RelocExpr<S>, InternalDiagnostic<S>> {
    match expr.variant {
        ExprVariant::Atom(SemanticAtom::Ident(ident)) => Ok(factory.mk_symbol((ident, expr.span))),
        ExprVariant::Atom(SemanticAtom::Literal(Literal::Number(n))) => {
            Ok(factory.mk_literal((n, expr.span)))
        }
        ExprVariant::Atom(SemanticAtom::Literal(Literal::Operand(_))) => {
            Err(InternalDiagnostic::new(
                Message::KeywordInExpr {
                    keyword: expr.span.clone(),
                },
                expr.span,
            ))
        }
        ExprVariant::Atom(SemanticAtom::Literal(Literal::String(_))) => Err(
            InternalDiagnostic::new(Message::StringInInstruction, expr.span),
        ),
        ExprVariant::Unary(SemanticUnary::Parentheses, expr) => analyze_reloc_expr(*expr, factory),
        ExprVariant::Binary(SemanticBinary::Plus, left, right) => {
            let left = analyze_reloc_expr(*left, factory)?;
            let right = analyze_reloc_expr(*right, factory)?;
            Ok(RelocExpr {
                variant: ExprVariant::Binary(BinaryOperator::Plus, Box::new(left), Box::new(right)),
                span: expr.span,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::{RelocAtom, Width};
    use crate::codebase::CodebaseError;
    use crate::diagnostics::{InternalDiagnostic, Message};
    use crate::frontend::syntax::{
        keyword::Operand, CommandContext, ExprContext, FileContext, MacroInvocationContext,
        MacroParamsContext, StmtContext, TokenSeqContext,
    };
    use crate::instruction::RelocExpr;
    use std::borrow::Borrow;
    use std::cell::RefCell;

    pub struct TestFrontend {
        operations: RefCell<Vec<TestOperation>>,
        error: Option<CodebaseError>,
    }

    impl TestFrontend {
        pub fn new() -> TestFrontend {
            TestFrontend {
                operations: RefCell::new(Vec::new()),
                error: None,
            }
        }

        pub fn fail(&mut self, error: CodebaseError) {
            self.error = Some(error)
        }

        pub fn into_inner(self) -> Vec<TestOperation> {
            self.operations.into_inner()
        }
    }

    #[derive(Debug, PartialEq)]
    pub enum TestOperation {
        AnalyzeFile(String),
        InvokeMacro(String, Vec<Vec<Token<String>>>),
        DefineMacro(String, Vec<String>, Vec<Token<String>>),
        EmitDiagnostic(InternalDiagnostic<()>),
        EmitItem(backend::Item<()>),
        Label(String),
        SetOrigin(RelocExpr<()>),
    }

    impl Session for TestFrontend {
        type Ident = String;
        type Span = ();

        fn analyze_file(&mut self, path: Self::Ident) -> Result<(), CodebaseError> {
            self.operations
                .borrow_mut()
                .push(TestOperation::AnalyzeFile(path));
            match self.error.take() {
                Some(error) => Err(error),
                None => Ok(()),
            }
        }

        fn invoke_macro(
            &mut self,
            name: (Self::Ident, Self::Span),
            args: Vec<Vec<(Token<Self::Ident>, Self::Span)>>,
        ) {
            self.operations
                .borrow_mut()
                .push(TestOperation::InvokeMacro(
                    name.0,
                    args.into_iter()
                        .map(|arg| arg.into_iter().map(|(token, _)| token).collect())
                        .collect(),
                ))
        }

        fn emit_diagnostic(&mut self, diagnostic: InternalDiagnostic<Self::Span>) {
            self.operations
                .borrow_mut()
                .push(TestOperation::EmitDiagnostic(diagnostic))
        }

        fn emit_item(&mut self, item: backend::Item<()>) {
            self.operations
                .borrow_mut()
                .push(TestOperation::EmitItem(item))
        }

        fn define_label(&mut self, (label, _): (String, Self::Span)) {
            self.operations
                .borrow_mut()
                .push(TestOperation::Label(label))
        }

        fn define_macro(
            &mut self,
            (name, _): (impl Into<String>, Self::Span),
            params: Vec<(String, Self::Span)>,
            tokens: Vec<(Token<String>, ())>,
        ) {
            self.operations
                .borrow_mut()
                .push(TestOperation::DefineMacro(
                    name.into(),
                    params.into_iter().map(|(s, _)| s).collect(),
                    tokens.into_iter().map(|(t, _)| t).collect(),
                ))
        }

        fn set_origin(&mut self, origin: RelocExpr<()>) {
            self.operations
                .borrow_mut()
                .push(TestOperation::SetOrigin(origin))
        }
    }

    #[test]
    fn emit_ld_b_deref_hl() {
        use crate::instruction::*;
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions
                .enter_stmt(None)
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
    fn emit_rst_1_plus_1() {
        use crate::instruction::*;
        let actions = collect_semantic_actions(|actions| {
            let command = actions
                .enter_stmt(None)
                .enter_command((Command::Mnemonic(Mnemonic::Rst), ()));
            let mut expr = command.add_argument();
            expr.push_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.push_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.apply_operator((ExprOperator::Plus, ()));
            expr.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [TestOperation::EmitItem(backend::Item::Instruction(
                Instruction::Rst(
                    ExprVariant::Binary(
                        BinaryOperator::Plus,
                        Box::new(1.into()),
                        Box::new(1.into()),
                    ).into()
                )
            ))]
        )
    }

    #[test]
    fn emit_label_word() {
        let label = "my_label";
        let actions = collect_semantic_actions(|actions| {
            let mut arg = actions
                .enter_stmt(None)
                .enter_command((Command::Directive(Directive::Dw), ()))
                .add_argument();
            arg.push_atom((ExprAtom::Ident(label.to_string()), ()));
            arg.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [TestOperation::EmitItem(backend::Item::Data(
                RelocAtom::Symbol(label.to_string()).into(),
                Width::Word
            ))]
        );
    }

    #[test]
    fn analyze_label() {
        let label = "label";
        let actions = collect_semantic_actions(|actions| {
            actions.enter_stmt(Some((label.to_string(), ()))).exit()
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

    #[test]
    fn define_nameless_macro() {
        let actions = collect_semantic_actions(|actions| {
            let params = actions.enter_stmt(None).enter_macro_def(());
            TokenSeqContext::exit(MacroParamsContext::exit(params))
        });
        assert_eq!(
            actions,
            [TestOperation::EmitDiagnostic(InternalDiagnostic::new(
                Message::MacroRequiresName,
                ()
            ))]
        )
    }

    fn test_macro_definition(
        name: &str,
        params: impl Borrow<[&'static str]>,
        body: impl Borrow<[Token<String>]>,
    ) {
        let actions = collect_semantic_actions(|actions| {
            let mut params_actions = actions
                .enter_stmt(Some((name.to_string(), ())))
                .enter_macro_def(());
            for param in params.borrow().iter().map(|t| (t.to_string(), ())) {
                params_actions.add_parameter(param)
            }
            let mut token_seq_actions = MacroParamsContext::exit(params_actions);
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
                .enter_stmt(None)
                .enter_macro_invocation((name.to_string(), ()));
            invocation.exit().exit()
        });
        assert_eq!(
            actions,
            [TestOperation::InvokeMacro(name.to_string(), Vec::new())]
        )
    }

    #[test]
    fn invoke_unary_macro() {
        let name = "my_macro";
        let arg_token = Token::Literal(Literal::Operand(Operand::A));
        let actions = collect_semantic_actions(|actions| {
            let mut invocation = actions
                .enter_stmt(None)
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
            [TestOperation::InvokeMacro(
                name.to_string(),
                vec![vec![arg_token]]
            )]
        )
    }

    #[test]
    fn diagnose_wrong_operand_count() {
        let actions = collect_semantic_actions(|actions| {
            let mut arg = actions
                .enter_stmt(None)
                .enter_command((Command::Mnemonic(Mnemonic::Nop), ()))
                .add_argument();
            let literal_a = Literal::Operand(Operand::A);
            arg.push_atom((ExprAtom::Literal(literal_a), ()));
            arg.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [TestOperation::EmitDiagnostic(InternalDiagnostic::new(
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
        let diagnostic = InternalDiagnostic::new(Message::UnexpectedToken { token: () }, ());
        let actions = collect_semantic_actions(|actions| {
            let mut stmt = actions.enter_stmt(None);
            stmt.emit_diagnostic(diagnostic.clone());
            stmt.exit()
        });
        assert_eq!(actions, [TestOperation::EmitDiagnostic(diagnostic)])
    }

    #[test]
    fn recover_from_malformed_expr() {
        let diagnostic = InternalDiagnostic::new(Message::UnexpectedToken { token: () }, ());
        let actions = collect_semantic_actions(|file| {
            let mut expr = file
                .enter_stmt(None)
                .enter_command((Command::Mnemonic(Mnemonic::Add), ()))
                .add_argument();
            expr.emit_diagnostic(diagnostic.clone());
            expr.exit().exit().exit()
        });
        assert_eq!(actions, [TestOperation::EmitDiagnostic(diagnostic)])
    }

    pub fn collect_semantic_actions<F>(f: F) -> Vec<TestOperation>
    where
        F: for<'a> FnOnce(SemanticActions<'a, TestFrontend>) -> SemanticActions<'a, TestFrontend>,
    {
        let mut operations = TestFrontend::new();
        f(SemanticActions::new(&mut operations));
        operations.operations.into_inner()
    }
}
