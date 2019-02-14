use self::invoke::MacroInvocationActions;

use crate::analysis::session::{Session, ValueBuilder};
use crate::analysis::{Ident, Literal, SemanticToken};
use crate::diag::span::{MergeSpans, Source, StripSpan};
use crate::diag::*;
use crate::expr::{BinaryOperator, Expr, ExprVariant};
use crate::model::Item;
use crate::syntax::{self, keyword::*, ExprAtom, Operator, UnaryOperator};

mod directive;
mod instruction;
mod invoke;
mod operand;

#[derive(Debug, PartialEq)]
pub(crate) enum SemanticAtom<I> {
    Ident(Ident<I>),
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

pub(crate) type SemanticExpr<I, S> = Expr<SemanticAtom<I>, SemanticUnary, BinaryOperator, S>;

#[cfg(test)]
type SemanticExprVariant<I, S> = ExprVariant<SemanticAtom<I>, SemanticUnary, BinaryOperator, S>;

pub(crate) struct SemanticActions<S: Session> {
    session: S,
    label: Option<(Ident<S::StringRef>, S::Span)>,
}

impl<S: Session> SemanticActions<S> {
    pub fn new(session: S) -> SemanticActions<S> {
        SemanticActions {
            session,
            label: None,
        }
    }

    fn define_label_if_present(&mut self) {
        if let Some((label, span)) = self.label.take() {
            let value = self.session.from_location_counter(span.clone());
            self.session.define_symbol((label, span), value)
        }
    }
}

impl<S: Session> DelegateDiagnostics<S::Span> for SemanticActions<S> {
    type Delegate = S::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.session.diagnostics()
    }
}

impl<S: Session> syntax::FileContext<Ident<S::StringRef>, Literal<S::StringRef>, Command, S::Span>
    for SemanticActions<S>
{
    type StmtContext = Self;

    fn enter_stmt(mut self, label: Option<(Ident<S::StringRef>, S::Span)>) -> Self::StmtContext {
        self.label = label;
        self
    }
}

impl<S: Session> syntax::StmtContext<Ident<S::StringRef>, Literal<S::StringRef>, Command, S::Span>
    for SemanticActions<S>
{
    type CommandContext = CommandActions<S>;
    type MacroParamsContext = MacroDefActions<S>;
    type MacroInvocationContext = MacroInvocationActions<S>;
    type Parent = Self;

    fn enter_command(self, name: (Command, S::Span)) -> Self::CommandContext {
        CommandActions::new(name, self)
    }

    fn enter_macro_def(mut self, keyword: S::Span) -> Self::MacroParamsContext {
        if self.label.is_none() {
            self.diagnostics()
                .emit_diagnostic(CompactDiagnostic::new(Message::MacroRequiresName, keyword))
        }
        MacroDefActions::new(self.label.take(), self)
    }

    fn enter_macro_invocation(
        mut self,
        name: (Ident<S::StringRef>, S::Span),
    ) -> Self::MacroInvocationContext {
        self.define_label_if_present();
        MacroInvocationActions::new(name, self)
    }

    fn exit(mut self) -> Self::Parent {
        self.define_label_if_present();
        self
    }
}

pub(crate) struct CommandActions<S: Session> {
    name: (Command, S::Span),
    args: CommandArgs<S::StringRef, S::Span>,
    parent: SemanticActions<S>,
    has_errors: bool,
}

type CommandArgs<I, S> = Vec<SemanticExpr<I, S>>;

impl<S: Session> CommandActions<S> {
    fn new(name: (Command, S::Span), parent: SemanticActions<S>) -> CommandActions<S> {
        CommandActions {
            name,
            args: Vec::new(),
            parent,
            has_errors: false,
        }
    }
}

impl<S: Session> MergeSpans<S::Span> for CommandActions<S> {
    fn merge_spans(&mut self, left: &S::Span, right: &S::Span) -> S::Span {
        self.parent.diagnostics().merge_spans(left, right)
    }
}

impl<S: Session> StripSpan<S::Span> for CommandActions<S> {
    type Stripped = <S::Delegate as StripSpan<S::Span>>::Stripped;

    fn strip_span(&mut self, span: &S::Span) -> Self::Stripped {
        self.parent.diagnostics().strip_span(span)
    }
}

impl<S: Session> EmitDiagnostic<S::Span, <S::Delegate as StripSpan<S::Span>>::Stripped>
    for CommandActions<S>
{
    fn emit_diagnostic(
        &mut self,
        diagnostic: CompactDiagnostic<S::Span, <S::Delegate as StripSpan<S::Span>>::Stripped>,
    ) {
        self.has_errors = true;
        self.parent.diagnostics().emit_diagnostic(diagnostic)
    }
}

impl<S: Session> DelegateDiagnostics<S::Span> for CommandActions<S> {
    type Delegate = Self;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self
    }
}

impl<S: Session> syntax::CommandContext<S::Span> for CommandActions<S> {
    type Ident = Ident<S::StringRef>;
    type Command = Command;
    type Literal = Literal<S::StringRef>;
    type ArgContext = ExprContext<S::StringRef, S::Span, Self>;
    type Parent = SemanticActions<S>;

    fn add_argument(self) -> Self::ArgContext {
        ExprContext {
            stack: Vec::new(),
            parent: self,
        }
    }

    fn exit(mut self) -> Self::Parent {
        if !self.has_errors {
            match self.name {
                (Command::Directive(directive), span) => {
                    if !directive.requires_symbol() {
                        self.parent.define_label_if_present()
                    }
                    directive::analyze_directive((directive, span), self.args, &mut self.parent)
                }
                (Command::Mnemonic(mnemonic), range) => {
                    self.parent.define_label_if_present();
                    analyze_mnemonic((mnemonic, range), self.args, &mut self.parent)
                }
            };
        }
        self.parent
    }
}

impl Directive {
    fn requires_symbol(self) -> bool {
        match self {
            Directive::Equ => true,
            _ => false,
        }
    }
}

pub(crate) struct ExprContext<R, S, P> {
    stack: Vec<SemanticExpr<R, S>>,
    parent: P,
}

pub(crate) trait Args<R, S> {
    fn push_arg(&mut self, arg: SemanticExpr<R, S>);
    fn has_errors(&self) -> bool;
}

impl<S: Session> Args<S::StringRef, S::Span> for CommandActions<S> {
    fn push_arg(&mut self, arg: SemanticExpr<S::StringRef, S::Span>) {
        self.args.push(arg)
    }

    fn has_errors(&self) -> bool {
        self.has_errors
    }
}

impl<R, S, P> DelegateDiagnostics<S> for ExprContext<R, S, P>
where
    P: DownstreamDiagnostics<S>,
{
    type Delegate = P;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        &mut self.parent
    }
}

impl<R, S, P> syntax::ExprContext<S> for ExprContext<R, S, P>
where
    S: Clone,
    P: Args<R, S>,
    P: DelegateDiagnostics<S> + MergeSpans<S> + StripSpan<S>,
    P: EmitDiagnostic<S, <P as StripSpan<S>>::Stripped>,
{
    type Ident = Ident<R>;
    type Literal = Literal<R>;
    type Parent = P;

    fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, S)) {
        self.stack.push(SemanticExpr {
            variant: ExprVariant::Atom(match atom.0 {
                ExprAtom::Ident(ident) => SemanticAtom::Ident(ident),
                ExprAtom::Literal(literal) => SemanticAtom::Literal(literal),
            }),
            span: atom.1,
        })
    }

    fn apply_operator(&mut self, operator: (Operator, S)) {
        match operator.0 {
            Operator::Unary(UnaryOperator::Parentheses) => {
                let inner = self.stack.pop().unwrap_or_else(|| unreachable!());
                self.stack.push(SemanticExpr {
                    variant: ExprVariant::Unary(SemanticUnary::Parentheses, Box::new(inner)),
                    span: operator.1,
                })
            }
            Operator::Binary(binary) => {
                let rhs = self.stack.pop().unwrap_or_else(|| unreachable!());
                let lhs = self.stack.pop().unwrap_or_else(|| unreachable!());
                self.stack.push(SemanticExpr {
                    variant: ExprVariant::Binary(binary, Box::new(lhs), Box::new(rhs)),
                    span: operator.1,
                })
            }
        }
    }

    fn exit(mut self) -> Self::Parent {
        if !self.parent.has_errors() {
            assert_eq!(self.stack.len(), 1);
            self.parent.push_arg(self.stack.pop().unwrap());
        }
        self.parent
    }
}

fn analyze_mnemonic<S: Session>(
    name: (Mnemonic, S::Span),
    args: CommandArgs<S::StringRef, S::Span>,
    actions: &mut SemanticActions<S>,
) {
    let operands: Vec<_> = args
        .into_iter()
        .map(|arg| operand::analyze_operand(arg, name.0.context(), &mut actions.session))
        .collect();
    let result = instruction::analyze_instruction(name, operands, actions.session.diagnostics());
    if let Ok(instruction) = result {
        actions.session.emit_item(Item::Instruction(instruction))
    }
}

pub(crate) struct MacroDefActions<S: Session> {
    name: Option<(Ident<S::StringRef>, S::Span)>,
    params: (Vec<Ident<S::StringRef>>, Vec<S::Span>),
    tokens: (Vec<SemanticToken<S::StringRef>>, Vec<S::Span>),
    parent: SemanticActions<S>,
}

impl<S: Session> MacroDefActions<S> {
    fn new(
        name: Option<(Ident<S::StringRef>, S::Span)>,
        parent: SemanticActions<S>,
    ) -> MacroDefActions<S> {
        MacroDefActions {
            name,
            params: (Vec::new(), Vec::new()),
            tokens: (Vec::new(), Vec::new()),
            parent,
        }
    }
}

impl<S: Session> DelegateDiagnostics<S::Span> for MacroDefActions<S> {
    type Delegate = S::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics()
    }
}

impl<S: Session> syntax::MacroParamsContext<S::Span> for MacroDefActions<S> {
    type Ident = Ident<S::StringRef>;
    type Command = Command;
    type Literal = Literal<S::StringRef>;
    type MacroBodyContext = Self;
    type Parent = SemanticActions<S>;

    fn add_parameter(&mut self, (param, span): (Self::Ident, S::Span)) {
        self.params.0.push(param);
        self.params.1.push(span)
    }

    fn exit(self) -> Self::MacroBodyContext {
        self
    }
}

impl<S: Session> syntax::TokenSeqContext<S::Span> for MacroDefActions<S> {
    type Token = SemanticToken<S::StringRef>;
    type Parent = SemanticActions<S>;

    fn push_token(&mut self, (token, span): (Self::Token, S::Span)) {
        self.tokens.0.push(token);
        self.tokens.1.push(span)
    }

    fn exit(mut self) -> Self::Parent {
        if let Some(name) = self.name {
            self.parent
                .session
                .define_macro(name, self.params, self.tokens)
        }
        self.parent
    }
}

trait AnalyzeExpr<I, S: Clone> {
    type Value: Source<Span = S>;

    fn analyze_expr(&mut self, expr: SemanticExpr<I, S>) -> Result<Self::Value, ()>;
}

impl<'a, T, I, S> AnalyzeExpr<I, S> for T
where
    T: ValueBuilder<Ident<I>, S> + DelegateDiagnostics<S>,
    S: Clone,
{
    type Value = T::Value;

    fn analyze_expr(&mut self, expr: SemanticExpr<I, S>) -> Result<Self::Value, ()> {
        match expr.variant {
            ExprVariant::Atom(SemanticAtom::Ident(ident)) => Ok(self.from_ident(ident, expr.span)),
            ExprVariant::Atom(SemanticAtom::Literal(Literal::Number(n))) => {
                Ok(self.from_number(n, expr.span))
            }
            ExprVariant::Atom(SemanticAtom::Literal(Literal::Operand(_))) => {
                Err(CompactDiagnostic::new(
                    Message::KeywordInExpr {
                        keyword: self.diagnostics().strip_span(&expr.span),
                    },
                    expr.span,
                ))
            }
            ExprVariant::Atom(SemanticAtom::Literal(Literal::String(_))) => Err(
                CompactDiagnostic::new(Message::StringInInstruction, expr.span),
            ),
            ExprVariant::Unary(SemanticUnary::Parentheses, expr) => Ok(self.analyze_expr(*expr)?),
            ExprVariant::Binary(binary, left, right) => {
                let left = self.analyze_expr(*left)?;
                let right = self.analyze_expr(*right)?;
                Ok(self.apply_binary_operator((binary, expr.span), left, right))
            }
        }
        .map_err(|diagnostic| {
            self.diagnostics().emit_diagnostic(diagnostic);
        })
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TokenId {
    Mnemonic,
    Operand(usize, usize),
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct TokenSpan {
    first: TokenId,
    last: TokenId,
}

#[cfg(test)]
impl MockSpan for TokenSpan {
    fn default() -> Self {
        unimplemented!()
    }

    fn merge(&self, other: &Self) -> Self {
        TokenSpan::merge(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::backend::BackendEvent;
    use crate::analysis::session::SessionEvent;
    use crate::diag::{CompactDiagnostic, Message};
    use crate::expr::BinaryOperator;
    use crate::model::{RelocAtom, Width};
    use crate::syntax::{
        CommandContext, ExprContext, FileContext, MacroInvocationContext, MacroParamsContext,
        StmtContext, TokenSeqContext,
    };
    use crate::syntax::{Operand, Token};

    use std::borrow::Borrow;
    use std::cell::RefCell;

    #[derive(Debug, PartialEq)]
    pub(crate) enum TestOperation {
        Backend(BackendEvent<RelocExpr>),
        Diagnostics(DiagnosticsEvent<()>),
        Session(SessionEvent),
    }

    type RelocExpr = crate::model::RelocExpr<Ident<String>, ()>;

    impl<'a> From<BackendEvent<RelocExpr>> for TestOperation {
        fn from(event: BackendEvent<RelocExpr>) -> Self {
            TestOperation::Backend(event)
        }
    }

    impl<'a> From<DiagnosticsEvent<()>> for TestOperation {
        fn from(event: DiagnosticsEvent<()>) -> Self {
            TestOperation::Diagnostics(event)
        }
    }

    impl<'a> From<SessionEvent> for TestOperation {
        fn from(event: SessionEvent) -> Self {
            TestOperation::Session(event)
        }
    }

    #[test]
    fn emit_ld_b_deref_hl() {
        use crate::model::*;
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions
                .enter_stmt(None)
                .enter_command((Command::Mnemonic(Mnemonic::Ld), ()));
            let mut arg1 = command.add_argument();
            arg1.push_atom((ExprAtom::Literal(Literal::Operand(Operand::B)), ()));
            command = arg1.exit();
            let mut arg2 = command.add_argument();
            arg2.push_atom((ExprAtom::Literal(Literal::Operand(Operand::Hl)), ()));
            arg2.apply_operator((Operator::Unary(UnaryOperator::Parentheses), ()));
            arg2.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [
                BackendEvent::EmitItem(Item::Instruction(Instruction::Ld(Ld::Simple(
                    SimpleOperand::B,
                    SimpleOperand::DerefHl
                ))))
                .into()
            ]
        )
    }

    #[test]
    fn emit_rst_1_minus_1() {
        test_rst_1_op_1(BinaryOperator::Minus)
    }

    #[test]
    fn emit_rst_1_plus_1() {
        test_rst_1_op_1(BinaryOperator::Plus)
    }

    fn test_rst_1_op_1(op: BinaryOperator) {
        use crate::model::*;
        let actions = collect_semantic_actions(|actions| {
            let command = actions
                .enter_stmt(None)
                .enter_command((Command::Mnemonic(Mnemonic::Rst), ()));
            let mut expr = command.add_argument();
            expr.push_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.push_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.apply_operator((Operator::Binary(op), ()));
            expr.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [BackendEvent::EmitItem(Item::Instruction(Instruction::Rst(
                ExprVariant::Binary(op, Box::new(1.into()), Box::new(1.into()),).into()
            )))
            .into()]
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
            arg.push_atom((ExprAtom::Ident(label.into()), ()));
            arg.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [BackendEvent::EmitItem(Item::Data(
                RelocAtom::Symbol(label.into()).into(),
                Width::Word
            ))
            .into()]
        );
    }

    #[test]
    fn analyze_label() {
        let label = "label";
        let actions =
            collect_semantic_actions(|actions| actions.enter_stmt(Some((label.into(), ()))).exit());
        assert_eq!(
            actions,
            [
                BackendEvent::DefineSymbol((label.into(), ()), RelocAtom::LocationCounter.into())
                    .into()
            ]
        )
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
                Token::Ident(param.into()),
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
            [DiagnosticsEvent::EmitDiagnostic(CompactDiagnostic::new(
                Message::MacroRequiresName,
                ()
            ))
            .into()]
        )
    }

    fn test_macro_definition(
        name: &str,
        params: impl Borrow<[&'static str]>,
        body: impl Borrow<[SemanticToken<String>]>,
    ) {
        let actions = collect_semantic_actions(|actions| {
            let mut params_actions = actions
                .enter_stmt(Some((name.into(), ())))
                .enter_macro_def(());
            for param in params.borrow().iter().map(|&t| (t.into(), ())) {
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
            [SessionEvent::DefineMacro(
                name.into(),
                params.borrow().iter().cloned().map(Into::into).collect(),
                body.borrow().iter().cloned().collect()
            )
            .into()]
        )
    }

    #[test]
    fn invoke_nullary_macro() {
        let name = "my_macro";
        let actions = collect_semantic_actions(|actions| {
            let invocation = actions
                .enter_stmt(None)
                .enter_macro_invocation((name.into(), ()));
            invocation.exit().exit()
        });
        assert_eq!(
            actions,
            [SessionEvent::InvokeMacro(name.into(), Vec::new()).into()]
        )
    }

    #[test]
    fn invoke_unary_macro() {
        let name = "my_macro";
        let arg_token = Token::Literal(Literal::Operand(Operand::A));
        let actions = collect_semantic_actions(|actions| {
            let mut invocation = actions
                .enter_stmt(None)
                .enter_macro_invocation((name.into(), ()));
            invocation = {
                let mut arg = invocation.enter_macro_arg();
                arg.push_token((arg_token.clone(), ()));
                arg.exit()
            };
            invocation.exit().exit()
        });
        assert_eq!(
            actions,
            [SessionEvent::InvokeMacro(name.into(), vec![vec![arg_token]]).into()]
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
            [DiagnosticsEvent::EmitDiagnostic(CompactDiagnostic::new(
                Message::OperandCount {
                    actual: 1,
                    expected: 0
                },
                ()
            ))
            .into()]
        )
    }

    #[test]
    fn diagnose_parsing_error() {
        let diagnostic = CompactDiagnostic::new(Message::UnexpectedToken { token: () }, ());
        let actions = collect_semantic_actions(|actions| {
            let mut stmt = actions.enter_stmt(None);
            stmt.diagnostics().emit_diagnostic(diagnostic.clone());
            stmt.exit()
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiagnostic(diagnostic).into()]
        )
    }

    #[test]
    fn recover_from_malformed_expr() {
        let diagnostic = CompactDiagnostic::new(Message::UnexpectedToken { token: () }, ());
        let actions = collect_semantic_actions(|file| {
            let mut expr = file
                .enter_stmt(None)
                .enter_command((Command::Mnemonic(Mnemonic::Add), ()))
                .add_argument();
            expr.diagnostics().emit_diagnostic(diagnostic.clone());
            expr.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiagnostic(diagnostic).into()]
        )
    }

    pub(super) type MockSession<'a> = crate::analysis::session::MockSession<'a, TestOperation, ()>;

    pub(crate) fn collect_semantic_actions<F>(f: F) -> Vec<TestOperation>
    where
        F: for<'a> FnOnce(TestSemanticActions<'a>) -> TestSemanticActions<'a>,
    {
        let operations = RefCell::new(Vec::new());
        let session = MockSession::new(&operations);
        f(SemanticActions::new(session));
        operations.into_inner()
    }

    type TestSemanticActions<'a> = SemanticActions<MockSession<'a>>;
}
