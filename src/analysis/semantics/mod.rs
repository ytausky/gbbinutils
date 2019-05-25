use self::command::CommandActions;
use self::invoke::MacroCallActions;
use self::params::ParamsAdapter;

use super::backend::{Backend, FinishFnDef, LocationCounter, PushOp};
use super::macros::MacroEntry;
use super::resolve::{NameTable, StartScope};
use super::session::*;
use super::syntax::keyword::*;
use super::syntax::*;
use super::{Ident, Lex, LexItem, Literal, SemanticToken};

use crate::diag::*;

#[cfg(test)]
pub use self::mock::*;

mod command;
mod invoke;
mod params;

pub(super) trait Analyze<R: Clone + Eq, D: DiagnosticsSystem> {
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
            + StartScope<Ident<R>>;
}

pub struct SemanticAnalyzer;

impl<R: Clone + Eq, D: DiagnosticsSystem> Analyze<R, D> for SemanticAnalyzer {
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
    {
        let session = CompositeSession::new(
            partial.codebase,
            self,
            partial.backend,
            partial.names,
            partial.diagnostics,
        );
        let mut actions =
            super::syntax::parse_token_seq(tokens.into_iter(), SemanticActions::new(session));
        actions.session.take().unwrap().into()
    }
}

pub(super) struct SemanticActions<S: Session> {
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

    fn with_session<F>(&mut self, f: F)
    where
        F: FnOnce(S) -> S,
    {
        self.session = Some(f(self.session.take().unwrap()))
    }
}

delegate_diagnostics! {
    {S: Session}, SemanticActions<S>, {session()}, S, S::Span
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

pub(super) struct LabelActions<S: Session> {
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

delegate_diagnostics! {
    {S: Session}, LabelActions<S>, {parent}, S, S::Span
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

pub(super) struct StmtActions<S: Session> {
    parent: SemanticActions<S>,
    label: Option<Label<S::StringRef, S::Span>>,
}

type Label<R, S> = ((Ident<R>, S), Params<R, S>);

impl<S: Session> StmtActions<S> {
    fn new(parent: SemanticActions<S>, label: Option<Label<S::StringRef, S::Span>>) -> Self {
        Self { parent, label }
    }

    fn define_label_if_present(&mut self) {
        if let Some(((label, span), _params)) = self.label.take() {
            self.parent.with_session(|session| {
                let mut builder = session.define_symbol(label, span.clone());
                PushOp::<LocationCounter, _>::push_op(&mut builder, LocationCounter, span);
                builder.finish_fn_def()
            })
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
            self.emit_diag(Message::MacroRequiresName.at(keyword))
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

delegate_diagnostics! {
    {S: Session}, StmtActions<S>, {parent}, S, S::Span
}

pub(super) struct MacroDefActions<S: Session> {
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

delegate_diagnostics! {
    {S: Session}, MacroDefActions<S>, {parent}, S, S::Span
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
pub(crate) type TokenSpan = MockSpan<TokenId>;

#[cfg(test)]
impl Default for TokenSpan {
    fn default() -> Self {
        unreachable!()
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::log::Log;

    pub struct MockAnalyzer<T> {
        log: Log<T>,
    }

    impl<T> MockAnalyzer<T> {
        pub fn new(log: Log<T>) -> Self {
            Self { log }
        }
    }

    impl<T, D> Analyze<String, D> for MockAnalyzer<T>
    where
        T: From<AnalyzerEvent<D::Span>>,
        D: DiagnosticsSystem,
    {
        fn analyze_token_seq<'a, I, C, B, N>(
            &mut self,
            tokens: I,
            downstream: PartialSession<'a, C, B, N, D>,
        ) -> PartialSession<'a, C, B, N, D>
        where
            I: IntoIterator<Item = LexItem<String, D::Span>>,
            C: Lex<D, StringRef = String>,
            B: Backend<D::Span>,
            N: NameTable<Ident<String>, MacroEntry = MacroEntry<String, D>>,
        {
            self.log
                .push(AnalyzerEvent::AnalyzeTokenSeq(tokens.into_iter().collect()));
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
    use crate::model::{Atom, BinOp, ExprOp, Instruction, Item, Width};

    use std::borrow::Borrow;
    use std::fmt::Debug;

    #[derive(Debug, PartialEq)]
    pub(crate) enum TestOperation<S: Clone> {
        Backend(BackendEvent<Expr<S>>),
        Diagnostics(DiagnosticsEvent<S>),
        Session(SessionEvent<S>),
    }

    pub type Expr<S> = crate::model::Expr<LocationCounter, Ident<String>, S>;

    impl<'a, S: Clone> From<BackendEvent<Expr<S>>> for TestOperation<S> {
        fn from(event: BackendEvent<Expr<S>>) -> Self {
            TestOperation::Backend(event)
        }
    }

    impl<'a, S: Clone> From<DiagnosticsEvent<S>> for TestOperation<S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            TestOperation::Diagnostics(event)
        }
    }

    impl<'a, S: Clone> From<SessionEvent<S>> for TestOperation<S> {
        fn from(event: SessionEvent<S>) -> Self {
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
    fn emit_rst_f_of_1() {
        let ident = Ident::from("f");
        let actions = collect_semantic_actions(|actions| {
            let command = actions
                .enter_unlabeled_stmt()
                .enter_command((Command::Mnemonic(Mnemonic::Rst), ()));
            let mut expr = command.add_argument();
            expr.push_atom((ExprAtom::Ident(ident.clone()), ()));
            expr.push_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.apply_operator((Operator::FnCall(1), ()));
            expr.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [
                BackendEvent::EmitItem(Item::Instruction(Instruction::Rst(Expr::from_items(&[
                    ident.into(),
                    1.into(),
                    ExprOp::FnCall(1).into()
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
            [SessionEvent::DefineSymbol((label.into(), ()), LocationCounter.into()).into()]
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
            [BackendEvent::SetOrigin(LocationCounter.into()).into()]
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
            [DiagnosticsEvent::EmitDiag(Message::MacroRequiresName.at(()).into()).into()]
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
            [DiagnosticsEvent::EmitDiag(
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
            stmt.emit_diag(diagnostic.clone());
            stmt.exit()
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiag(diagnostic.into()).into()]
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
            expr.emit_diag(diagnostic.clone());
            expr.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiag(diagnostic.into()).into()]
        )
    }

    pub(super) type MockSession<S> = crate::analysis::session::MockSession<TestOperation<S>, S>;

    pub(super) fn collect_semantic_actions<F, S>(f: F) -> Vec<TestOperation<S>>
    where
        F: FnOnce(TestSemanticActions<S>) -> TestSemanticActions<S>,
        S: Clone + Debug + Merge,
    {
        crate::log::with_log(|log| {
            f(SemanticActions::new(MockSession::new(log)));
        })
    }

    type TestSemanticActions<S> = SemanticActions<MockSession<S>>;
}
