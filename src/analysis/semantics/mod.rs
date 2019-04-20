use self::command::CommandActions;
use self::invoke::MacroCallActions;
use self::params::ParamsAdapter;

use super::backend::{Backend, LocationCounter, PushOp};
use super::macros::MacroEntry;
use super::session::*;
use super::{Ident, Lex, LexItem, Literal, SemanticToken};

use crate::diag::*;
use crate::name::{NameTable, StartScope};
use crate::syntax::keyword::*;
use crate::syntax::*;

#[cfg(test)]
pub use self::mock::*;

mod command;
mod invoke;
mod params;

pub(crate) trait Analyze<R: Clone + Eq, D: Diagnostics> {
    fn analyze_token_seq<'a, I, C, B, N>(
        &mut self,
        tokens: I,
        partial: PartialSession<'a, C, B, N, D>,
    ) -> PartialSession<'a, C, B, N, D>
    where
        I: IntoIterator<Item = LexItem<R, D::Span>>,
        C: Lex<D, StringRef = R>,
        B: Backend<D::Span>,
        N: NameTable<Ident<R>, BackendEntry = B::Name, MacroEntry = MacroEntry<R, D>>
            + StartScope<Ident<R>>,
        B::Value: Default + ValueBuilder<B::Name, D::Span>;
}

pub struct SemanticAnalyzer;

impl<R: Clone + Eq, D: Diagnostics> Analyze<R, D> for SemanticAnalyzer {
    fn analyze_token_seq<'a, I, C, B, N>(
        &mut self,
        tokens: I,
        partial: PartialSession<'a, C, B, N, D>,
    ) -> PartialSession<'a, C, B, N, D>
    where
        I: IntoIterator<Item = LexItem<R, D::Span>>,
        C: Lex<D, StringRef = R>,
        B: Backend<D::Span>,
        N: NameTable<Ident<R>, BackendEntry = B::Name, MacroEntry = MacroEntry<R, D>>
            + StartScope<Ident<R>>,
        B::Value: Default + ValueBuilder<B::Name, D::Span>,
    {
        let session = CompositeSession::new(
            partial.codebase,
            self,
            partial.backend,
            partial.names,
            partial.diagnostics,
        );
        let mut actions =
            crate::syntax::parse_token_seq(tokens.into_iter(), SemanticActions::new(session));
        actions.session.take().unwrap().into()
    }
}

pub(crate) struct SemanticActions<S: Session> {
    session: Option<S>,
}

impl<S: Session> SemanticActions<S> {
    pub fn new(session: S) -> SemanticActions<S> {
        SemanticActions {
            session: Some(session),
        }
    }

    fn session(&mut self) -> &mut S {
        self.session.as_mut().unwrap()
    }

    fn build_value<F, T>(&mut self, params: &Params<S::StringRef, S::Span>, f: F) -> T
    where
        F: FnOnce(ParamsAdapter<S::GeneralBuilder, S::StringRef, S::Span>) -> (S, T),
    {
        let builder = self.session.take().unwrap().build_value();
        let adapter = ParamsAdapter::new(builder, params);
        let result = f(adapter);
        self.session = Some(result.0);
        result.1
    }
}

impl<S: Session> DelegateDiagnostics<S::Span> for SemanticActions<S> {
    type Delegate = S::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.session().diagnostics()
    }
}

impl<S: Session> FileContext<Ident<S::StringRef>, Literal<S::StringRef>, Command, S::Span>
    for SemanticActions<S>
{
    type LabelContext = LabelActions<S>;
    type StmtContext = StmtActions<S>;

    fn enter_labeled_stmt(self, label: (Ident<S::StringRef>, S::Span)) -> Self::LabelContext {
        LabelActions::new(self, label)
    }

    fn enter_unlabeled_stmt(self) -> Self::StmtContext {
        StmtActions::new(self, None)
    }
}

pub(crate) struct LabelActions<S: Session> {
    parent: SemanticActions<S>,
    label: (Ident<S::StringRef>, S::Span),
    params: Params<S::StringRef, S::Span>,
}

impl<S: Session> LabelActions<S> {
    fn new(parent: SemanticActions<S>, label: (Ident<S::StringRef>, S::Span)) -> Self {
        Self {
            parent,
            label,
            params: (Vec::new(), Vec::new()),
        }
    }
}

impl<S: Session> DelegateDiagnostics<S::Span> for LabelActions<S> {
    type Delegate = S::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics()
    }
}

impl<S: Session> ParamsContext<Ident<S::StringRef>, S::Span> for LabelActions<S> {
    type Next = StmtActions<S>;

    fn add_parameter(&mut self, (ident, span): (Ident<S::StringRef>, S::Span)) {
        self.params.0.push(ident);
        self.params.1.push(span)
    }

    fn next(self) -> Self::Next {
        StmtActions::new(self.parent, Some((self.label, self.params)))
    }
}

pub(crate) struct StmtActions<S: Session> {
    parent: SemanticActions<S>,
    label: Option<Label<S::StringRef, S::Span>>,
}

type Label<R, S> = ((Ident<R>, S), Params<R, S>);

impl<S: Session> StmtActions<S> {
    fn new(parent: SemanticActions<S>, label: Option<Label<S::StringRef, S::Span>>) -> Self {
        Self { parent, label }
    }

    fn define_label_if_present(&mut self) {
        if let Some(((label, span), params)) = self.label.take() {
            let value = self.parent.build_value(&params, |mut builder| {
                builder.push_op(LocationCounter, span.clone());
                builder.finish()
            });
            self.session().define_symbol((label, span), value)
        }
    }

    fn session(&mut self) -> &mut S {
        self.parent.session()
    }
}

impl<S: Session> StmtContext<Ident<S::StringRef>, Literal<S::StringRef>, Command, S::Span>
    for StmtActions<S>
{
    type CommandContext = CommandActions<S>;
    type MacroDefContext = MacroDefActions<S>;
    type MacroCallContext = MacroCallActions<S>;
    type Parent = SemanticActions<S>;

    fn enter_command(self, command: (Command, S::Span)) -> Self::CommandContext {
        CommandActions::new(self, command)
    }

    fn enter_macro_def(mut self, keyword: S::Span) -> Self::MacroDefContext {
        if self.label.is_none() {
            self.diagnostics()
                .emit_diagnostic(Message::MacroRequiresName.at(keyword))
        }
        MacroDefActions::new(self)
    }

    fn enter_macro_call(mut self, name: (Ident<S::StringRef>, S::Span)) -> Self::MacroCallContext {
        self.define_label_if_present();
        MacroCallActions::new(self, name)
    }

    fn exit(mut self) -> Self::Parent {
        self.define_label_if_present();
        self.parent
    }
}

impl<S: Session> DelegateDiagnostics<S::Span> for StmtActions<S> {
    type Delegate = S::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics()
    }
}

pub(crate) struct MacroDefActions<S: Session> {
    parent: StmtActions<S>,
    tokens: (Vec<SemanticToken<S::StringRef>>, Vec<S::Span>),
}

impl<S: Session> MacroDefActions<S> {
    fn new(parent: StmtActions<S>) -> Self {
        Self {
            parent,
            tokens: (Vec::new(), Vec::new()),
        }
    }
}

impl<S: Session> DelegateDiagnostics<S::Span> for MacroDefActions<S> {
    type Delegate = S::Delegate;

    fn diagnostics(&mut self) -> &mut Self::Delegate {
        self.parent.diagnostics()
    }
}

impl<S: Session> TokenSeqContext<S::Span> for MacroDefActions<S> {
    type Token = SemanticToken<S::StringRef>;
    type Parent = StmtActions<S>;

    fn push_token(&mut self, (token, span): (Self::Token, S::Span)) {
        self.tokens.0.push(token);
        self.tokens.1.push(span)
    }

    fn exit(self) -> Self::Parent {
        let mut parent = self.parent;
        if let Some((name, params)) = parent.label.take() {
            parent.session().define_macro(name, params, self.tokens)
        }
        parent
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
mod mock {
    use super::*;

    use std::cell::RefCell;

    pub struct MockAnalyzer<'a, T> {
        log: &'a RefCell<Vec<T>>,
    }

    impl<'a, T> MockAnalyzer<'a, T> {
        pub fn new(log: &'a RefCell<Vec<T>>) -> Self {
            Self { log }
        }
    }

    impl<'a, T, D> Analyze<String, D> for MockAnalyzer<'a, T>
    where
        T: From<AnalyzerEvent<D::Span>>,
        D: Diagnostics,
    {
        fn analyze_token_seq<'b, I, C, B, N>(
            &mut self,
            tokens: I,
            downstream: PartialSession<'b, C, B, N, D>,
        ) -> PartialSession<'b, C, B, N, D>
        where
            I: IntoIterator<Item = LexItem<String, D::Span>>,
            C: Lex<D, StringRef = String>,
            B: Backend<D::Span>,
            N: NameTable<Ident<String>, MacroEntry = MacroEntry<String, D>>,
            B::Value: Default + ValueBuilder<B::Name, D::Span>,
        {
            self.log
                .borrow_mut()
                .push(AnalyzerEvent::AnalyzeTokenSeq(tokens.into_iter().collect()).into());
            downstream
        }
    }

    #[derive(Debug, PartialEq)]
    pub(crate) enum AnalyzerEvent<S> {
        AnalyzeTokenSeq(Vec<LexItem<String, S>>),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::backend::BackendEvent;
    use crate::analysis::session::SessionEvent;
    use crate::diag::Message;
    use crate::model::{Atom, BinOp, Item, Width};

    use std::borrow::Borrow;
    use std::cell::RefCell;

    #[derive(Debug, PartialEq)]
    pub(crate) enum TestOperation {
        Backend(BackendEvent<Expr>),
        Diagnostics(DiagnosticsEvent<()>),
        Session(SessionEvent<()>),
    }

    type Expr = crate::model::Expr<Atom<Ident<String>>, ()>;

    impl<'a> From<BackendEvent<Expr>> for TestOperation {
        fn from(event: BackendEvent<Expr>) -> Self {
            TestOperation::Backend(event)
        }
    }

    impl<'a> From<DiagnosticsEvent<()>> for TestOperation {
        fn from(event: DiagnosticsEvent<()>) -> Self {
            TestOperation::Diagnostics(event)
        }
    }

    impl<'a> From<SessionEvent<()>> for TestOperation {
        fn from(event: SessionEvent<()>) -> Self {
            TestOperation::Session(event)
        }
    }

    #[test]
    fn emit_ld_b_deref_hl() {
        use crate::model::*;
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions
                .enter_unlabeled_stmt()
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
        test_rst_1_op_1(BinOp::Minus)
    }

    #[test]
    fn emit_rst_1_plus_1() {
        test_rst_1_op_1(BinOp::Plus)
    }

    fn test_rst_1_op_1(op: BinOp) {
        use crate::model::*;
        let actions = collect_semantic_actions(|actions| {
            let command = actions
                .enter_unlabeled_stmt()
                .enter_command((Command::Mnemonic(Mnemonic::Rst), ()));
            let mut expr = command.add_argument();
            expr.push_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.push_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.apply_operator((Operator::Binary(op), ()));
            expr.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [
                BackendEvent::EmitItem(Item::Instruction(Instruction::Rst(Expr::from_items(&[
                    1.into(),
                    1.into(),
                    op.into()
                ]))))
                .into()
            ]
        )
    }

    #[test]
    fn emit_label_word() {
        let label = "my_label";
        let actions = collect_semantic_actions(|actions| {
            let mut arg = actions
                .enter_unlabeled_stmt()
                .enter_command((Command::Directive(Directive::Dw), ()))
                .add_argument();
            arg.push_atom((ExprAtom::Ident(label.into()), ()));
            arg.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [
                BackendEvent::EmitItem(Item::Data(Atom::Name(label.into()).into(), Width::Word))
                    .into()
            ]
        );
    }

    #[test]
    fn analyze_label() {
        let label = "label";
        let actions = collect_semantic_actions(|actions| {
            actions.enter_labeled_stmt((label.into(), ())).next().exit()
        });
        assert_eq!(
            actions,
            [SessionEvent::DefineSymbol((label.into(), ()), Atom::LocationCounter.into()).into()]
        )
    }

    #[test]
    fn analyze_org_dot() {
        let actions = collect_semantic_actions(|actions| {
            let mut actions = actions
                .enter_unlabeled_stmt()
                .enter_command((Directive::Org.into(), ()))
                .add_argument();
            actions.push_atom((ExprAtom::LocationCounter, ()));
            actions.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [BackendEvent::SetOrigin(Atom::LocationCounter.into()).into()]
        );
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
            actions
                .enter_unlabeled_stmt()
                .enter_macro_def(())
                .exit()
                .exit()
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiagnostic(Message::MacroRequiresName.at(()).into()).into()]
        )
    }

    fn test_macro_definition(
        name: &str,
        params: impl Borrow<[&'static str]>,
        body: impl Borrow<[SemanticToken<String>]>,
    ) {
        let actions = collect_semantic_actions(|actions| {
            let mut params_actions = actions.enter_labeled_stmt((name.into(), ()));
            for param in params.borrow().iter().map(|&t| (t.into(), ())) {
                params_actions.add_parameter(param)
            }
            let mut token_seq_actions = params_actions.next().enter_macro_def(());
            for token in body.borrow().iter().cloned().map(|t| (t, ())) {
                token_seq_actions.push_token(token)
            }
            token_seq_actions.exit().exit()
        });
        assert_eq!(
            actions,
            [SessionEvent::DefineMacro(
                name.into(),
                params.borrow().iter().cloned().map(Into::into).collect(),
                body.borrow().to_vec()
            )
            .into()]
        )
    }

    #[test]
    fn call_nullary_macro() {
        let name = "my_macro";
        let actions = collect_semantic_actions(|actions| {
            let call = actions
                .enter_unlabeled_stmt()
                .enter_macro_call((name.into(), ()));
            call.exit().exit()
        });
        assert_eq!(
            actions,
            [SessionEvent::InvokeMacro(name.into(), Vec::new()).into()]
        )
    }

    #[test]
    fn call_unary_macro() {
        let name = "my_macro";
        let arg_token = Token::Literal(Literal::Operand(Operand::A));
        let actions = collect_semantic_actions(|actions| {
            let mut call = actions
                .enter_unlabeled_stmt()
                .enter_macro_call((name.into(), ()));
            call = {
                let mut arg = call.enter_macro_arg();
                arg.push_token((arg_token.clone(), ()));
                arg.exit()
            };
            call.exit().exit()
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
                .enter_unlabeled_stmt()
                .enter_command((Command::Mnemonic(Mnemonic::Nop), ()))
                .add_argument();
            let literal_a = Literal::Operand(Operand::A);
            arg.push_atom((ExprAtom::Literal(literal_a), ()));
            arg.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiagnostic(
                Message::OperandCount {
                    actual: 1,
                    expected: 0
                }
                .at(())
                .into()
            )
            .into()]
        )
    }

    #[test]
    fn diagnose_parsing_error() {
        let diagnostic = Message::UnexpectedToken { token: () }.at(());
        let actions = collect_semantic_actions(|actions| {
            let mut stmt = actions.enter_unlabeled_stmt();
            stmt.diagnostics().emit_diagnostic(diagnostic.clone());
            stmt.exit()
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiagnostic(diagnostic.into()).into()]
        )
    }

    #[test]
    fn recover_from_malformed_expr() {
        let diagnostic = Message::UnexpectedToken { token: () }.at(());
        let actions = collect_semantic_actions(|file| {
            let mut expr = file
                .enter_unlabeled_stmt()
                .enter_command((Command::Mnemonic(Mnemonic::Add), ()))
                .add_argument();
            expr.diagnostics().emit_diagnostic(diagnostic.clone());
            expr.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiagnostic(diagnostic.into()).into()]
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
