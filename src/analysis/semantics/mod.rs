use self::command::{Command, Directive, *};
use self::invoke::MacroCallActions;
use self::params::*;

use super::backend::{Finish, LocationCounter, PushOp};
use super::resolve::ResolvedIdent;
use super::session::{Analyze, IntoSession, Params, Session};
use super::syntax::*;
use super::{LexItem, Literal, SemanticToken, StringSource, TokenSeq};

use crate::diag::span::{SpanSource, StripSpan};
use crate::diag::{EmitDiag, Message};

#[cfg(test)]
pub use self::mock::*;

mod command;
mod invoke;
mod params;

pub struct SemanticAnalyzer;

impl<I: Clone + PartialEq, R: Clone + Eq, S: Clone> Analyze<I, R, S> for SemanticAnalyzer
where
    I: AsRef<str>,
{
    fn analyze_token_seq<'b, T, P>(&'b mut self, tokens: T, partial: P) -> P
    where
        T: IntoIterator<Item = LexItem<I, R, S>>,
        P: IntoSession<'b, Self>,
        P::Session: IdentSource<Ident = I> + StringSource<StringRef = R> + SpanSource<Span = S>,
    {
        let session = partial.into_session(self);
        let mut actions =
            super::syntax::parse_token_seq(tokens.into_iter(), SemanticActions::new(session));
        actions.session.take().unwrap().into()
    }
}

pub(super) struct SemanticActions<S: Session> {
    session: Option<S>,
    label: Option<Label<S::Ident, S::Span>>,
}

impl<S: Session> SemanticActions<S> {
    pub fn new(session: S) -> SemanticActions<S> {
        SemanticActions {
            session: Some(session),
            label: None,
        }
    }

    fn session(&mut self) -> &mut S {
        self.session.as_mut().unwrap()
    }

    fn build_value<F, T>(&mut self, params: &Params<S::Ident, S::Span>, f: F) -> T
    where
        F: FnOnce(
            BuilderAdapter<
                BuilderAdapter<S::GeneralBuilder, NameResolver>,
                ConvertParams<S::Ident, S::Span>,
            >,
        ) -> (S, T),
    {
        self.with_session(|session| f(session.build_value().resolve_names().with_params(params)))
    }

    fn with_session<F, T>(&mut self, f: F) -> T
    where
        F: FnOnce(S) -> (S, T),
    {
        let (session, output) = f(self.session.take().unwrap());
        self.session = Some(session);
        output
    }

    fn define_label_if_present(&mut self) {
        if let Some(((label, span), _params)) = self.label.take() {
            self.with_session(|mut session| {
                session.start_scope(&label);
                let id = session.reloc_lookup(label, span.clone());
                let mut builder = session.define_symbol(id, span.clone());
                PushOp::<LocationCounter, _>::push_op(&mut builder, LocationCounter, span);
                builder.finish()
            })
        }
    }
}

delegate_diagnostics! {
    {S: Session}, SemanticActions<S>, {session()}, S, S::Span
}

impl<S: Session> FileContext<S::Ident, Literal<S::StringRef>, S::Span> for SemanticActions<S>
where
    S::Ident: AsRef<str>,
{
    type LabelContext = LabelActions<S>;
    type StmtContext = Self;

    fn enter_labeled_stmt(self, label: (S::Ident, S::Span)) -> Self::LabelContext {
        LabelActions::new(self, label)
    }

    fn enter_unlabeled_stmt(self) -> Self::StmtContext {
        self
    }
}

pub(super) struct LabelActions<S: Session> {
    parent: SemanticActions<S>,
    label: (S::Ident, S::Span),
    params: Params<S::Ident, S::Span>,
}

impl<S: Session> LabelActions<S> {
    fn new(parent: SemanticActions<S>, label: (S::Ident, S::Span)) -> Self {
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

impl<S: Session> ParamsContext<S::Ident, S::Span> for LabelActions<S> {
    type Next = SemanticActions<S>;

    fn add_parameter(&mut self, (ident, span): (S::Ident, S::Span)) {
        self.params.0.push(ident);
        self.params.1.push(span)
    }

    fn next(mut self) -> Self::Next {
        self.parent.label = Some((self.label, self.params));
        self.parent
    }
}

type Label<I, S> = ((I, S), Params<I, S>);

impl<S: Session> StmtContext<S::Ident, Literal<S::StringRef>, S::Span> for SemanticActions<S>
where
    S::Ident: AsRef<str>,
{
    type CommandContext = CommandActions<S>;
    type MacroDefContext = MacroDefActions<S>;
    type MacroCallContext = MacroCallActions<S>;
    type Parent = SemanticActions<S>;

    fn key_lookup(
        mut self,
        ident: S::Ident,
        span: S::Span,
    ) -> Production<Self::CommandContext, Self::MacroCallContext, Self::MacroDefContext, Self> {
        match KEYS
            .iter()
            .find(|(spelling, _)| spelling.eq_ignore_ascii_case(ident.as_ref()))
            .map(|(_, entry)| entry)
        {
            Some(KeyEntry::Command(command)) => {
                Production::Command(CommandActions::new(self, (command.clone(), span)))
            }
            Some(KeyEntry::Keyword(Keyword::Macro)) => {
                if self.label.is_none() {
                    self.emit_diag(Message::MacroRequiresName.at(span))
                }
                Production::MacroDef(MacroDefActions::new(self))
            }
            None => {
                let result = self.with_session(|session| {
                    let result = session.get(&ident);
                    (session, result)
                });
                match result {
                    Some(ResolvedIdent::Macro(id)) => {
                        self.define_label_if_present();
                        Production::MacroCall(MacroCallActions::new(self, (id, span)))
                    }
                    Some(ResolvedIdent::Backend(_)) => {
                        let name = self.strip_span(&span);
                        self.emit_diag(Message::CannotUseSymbolNameAsMacroName { name }.at(span));
                        Production::Error(self)
                    }
                    None => {
                        let name = self.strip_span(&span);
                        self.emit_diag(Message::UndefinedMacro { name }.at(span));
                        Production::Error(self)
                    }
                }
            }
        }
    }

    fn exit(mut self) -> Self::Parent {
        self.define_label_if_present();
        self
    }
}

#[derive(Clone)]
enum KeyEntry {
    Command(Command),
    Keyword(Keyword),
}

const KEYS: &[(&str, KeyEntry)] = &[
    ("adc", KeyEntry::Command(Command::Mnemonic(ADC))),
    ("add", KeyEntry::Command(Command::Mnemonic(ADD))),
    ("and", KeyEntry::Command(Command::Mnemonic(AND))),
    ("bit", KeyEntry::Command(Command::Mnemonic(BIT))),
    ("call", KeyEntry::Command(Command::Mnemonic(CALL))),
    ("cp", KeyEntry::Command(Command::Mnemonic(CP))),
    ("cpl", KeyEntry::Command(Command::Mnemonic(CPL))),
    ("daa", KeyEntry::Command(Command::Mnemonic(DAA))),
    ("db", KeyEntry::Command(Command::Directive(Directive::Db))),
    ("dec", KeyEntry::Command(Command::Mnemonic(DEC))),
    ("di", KeyEntry::Command(Command::Mnemonic(DI))),
    ("ds", KeyEntry::Command(Command::Directive(Directive::Ds))),
    ("dw", KeyEntry::Command(Command::Directive(Directive::Dw))),
    ("ei", KeyEntry::Command(Command::Mnemonic(EI))),
    ("equ", KeyEntry::Command(Command::Directive(Directive::Equ))),
    ("halt", KeyEntry::Command(Command::Mnemonic(HALT))),
    ("inc", KeyEntry::Command(Command::Mnemonic(INC))),
    (
        "include",
        KeyEntry::Command(Command::Directive(Directive::Include)),
    ),
    ("jp", KeyEntry::Command(Command::Mnemonic(JP))),
    ("jr", KeyEntry::Command(Command::Mnemonic(JR))),
    ("ld", KeyEntry::Command(Command::Mnemonic(LD))),
    ("ldhl", KeyEntry::Command(Command::Mnemonic(LDHL))),
    ("macro", KeyEntry::Keyword(Keyword::Macro)),
    ("nop", KeyEntry::Command(Command::Mnemonic(NOP))),
    ("or", KeyEntry::Command(Command::Mnemonic(OR))),
    ("org", KeyEntry::Command(Command::Directive(Directive::Org))),
    ("pop", KeyEntry::Command(Command::Mnemonic(POP))),
    ("push", KeyEntry::Command(Command::Mnemonic(PUSH))),
    ("res", KeyEntry::Command(Command::Mnemonic(RES))),
    ("ret", KeyEntry::Command(Command::Mnemonic(RET))),
    ("reti", KeyEntry::Command(Command::Mnemonic(RETI))),
    ("rl", KeyEntry::Command(Command::Mnemonic(RL))),
    ("rla", KeyEntry::Command(Command::Mnemonic(RLA))),
    ("rlc", KeyEntry::Command(Command::Mnemonic(RLC))),
    ("rlca", KeyEntry::Command(Command::Mnemonic(RLCA))),
    ("rr", KeyEntry::Command(Command::Mnemonic(RR))),
    ("rra", KeyEntry::Command(Command::Mnemonic(RRA))),
    ("rrc", KeyEntry::Command(Command::Mnemonic(RRC))),
    ("rrca", KeyEntry::Command(Command::Mnemonic(RRCA))),
    ("rst", KeyEntry::Command(Command::Mnemonic(RST))),
    ("sbc", KeyEntry::Command(Command::Mnemonic(SBC))),
    (
        "section",
        KeyEntry::Command(Command::Directive(Directive::Section)),
    ),
    ("set", KeyEntry::Command(Command::Mnemonic(SET))),
    ("sla", KeyEntry::Command(Command::Mnemonic(SLA))),
    ("sra", KeyEntry::Command(Command::Mnemonic(SRA))),
    ("srl", KeyEntry::Command(Command::Mnemonic(SRL))),
    ("stop", KeyEntry::Command(Command::Mnemonic(STOP))),
    ("sub", KeyEntry::Command(Command::Mnemonic(SUB))),
    ("swap", KeyEntry::Command(Command::Mnemonic(SWAP))),
    ("xor", KeyEntry::Command(Command::Mnemonic(XOR))),
];

pub(super) struct MacroDefActions<S: Session> {
    parent: SemanticActions<S>,
    tokens: TokenSeq<S::Ident, S::StringRef, S::Span>,
}

impl<S: Session> MacroDefActions<S> {
    fn new(parent: SemanticActions<S>) -> Self {
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
    type Token = SemanticToken<S::Ident, S::StringRef>;
    type Parent = SemanticActions<S>;

    fn push_token(&mut self, (token, span): (Self::Token, S::Span)) {
        self.tokens.0.push(token);
        self.tokens.1.push(span)
    }

    fn exit(self) -> Self::Parent {
        let Self { mut parent, tokens } = self;
        if let Some(((name, span), params)) = parent.label.take() {
            parent.with_session(|mut session| {
                let id = session.define_macro(span, params, tokens);
                session.insert(name, ResolvedIdent::Macro(id));
                (session, ())
            })
        }
        parent
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

    impl<T, S> Analyze<String, String, S> for MockAnalyzer<T>
    where
        T: From<AnalyzerEvent<S>>,
        S: Clone,
    {
        fn analyze_token_seq<'b, I, P>(&'b mut self, tokens: I, partial: P) -> P
        where
            I: IntoIterator<Item = LexItem<String, String, S>>,
            P: IntoSession<'b, Self>,
        {
            self.log
                .push(AnalyzerEvent::AnalyzeTokenSeq(tokens.into_iter().collect()));
            partial
        }
    }

    #[derive(Debug, PartialEq)]
    pub(crate) enum AnalyzerEvent<S> {
        AnalyzeTokenSeq(Vec<LexItem<String, String, S>>),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub use crate::analysis::resolve::BasicNameTable;

    use crate::analysis::backend::{BackendEvent, Name, SerialIdAllocator};
    use crate::analysis::resolve::{NameTableEvent, ResolvedIdent};
    use crate::analysis::session::{MockMacroId, SessionEvent};
    use crate::diag::{DiagnosticsEvent, Merge, Message, MockSpan};
    use crate::log::with_log;
    use crate::model::{Atom, BinOp, ExprOp, Instruction, Item, Width};

    use std::borrow::Borrow;
    use std::fmt::Debug;

    #[derive(Debug, PartialEq)]
    pub(crate) enum TestOperation<S: Clone> {
        Backend(BackendEvent<usize, Expr<S>>),
        Diagnostics(DiagnosticsEvent<S>),
        NameTable(NameTableEvent<usize, MockMacroId>),
        Session(SessionEvent),
    }

    pub type Expr<S> = crate::model::Expr<Atom<LocationCounter, usize>, S>;

    impl<S: Clone> From<BackendEvent<usize, Expr<S>>> for TestOperation<S> {
        fn from(event: BackendEvent<usize, Expr<S>>) -> Self {
            TestOperation::Backend(event)
        }
    }

    impl<S: Clone> From<DiagnosticsEvent<S>> for TestOperation<S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            TestOperation::Diagnostics(event)
        }
    }

    impl<S: Clone> From<NameTableEvent<usize, MockMacroId>> for TestOperation<S> {
        fn from(event: NameTableEvent<usize, MockMacroId>) -> Self {
            TestOperation::NameTable(event)
        }
    }

    impl<S: Clone> From<SessionEvent> for TestOperation<S> {
        fn from(event: SessionEvent) -> Self {
            TestOperation::Session(event)
        }
    }

    #[test]
    fn emit_ld_b_deref_hl() {
        use crate::model::*;
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions.key_lookup("LD".into(), ()).command().unwrap();
            let mut arg1 = command.add_argument();
            arg1.push_atom((ExprAtom::Ident("B".into()), ()));
            command = arg1.exit();
            let mut arg2 = command.add_argument();
            arg2.push_atom((ExprAtom::Ident("HL".into()), ()));
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
            let command = actions.key_lookup("RST".into(), ()).command().unwrap();
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
        let ident = String::from("f");
        let actions = collect_semantic_actions(|actions| {
            let command = actions.key_lookup("RST".into(), ()).command().unwrap();
            let mut expr = command.add_argument();
            expr.push_atom((ExprAtom::Ident(ident.clone()), ()));
            expr.push_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.apply_operator((Operator::FnCall(1), ()));
            expr.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [
                NameTableEvent::Insert(ident, ResolvedIdent::Backend(0)).into(),
                BackendEvent::EmitItem(Item::Instruction(Instruction::Rst(Expr::from_items(&[
                    1.into(),
                    Name(0).into(),
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
                .key_lookup("DW".into(), ())
                .command()
                .unwrap()
                .add_argument();
            arg.push_atom((ExprAtom::Ident(label.into()), ()));
            arg.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [
                NameTableEvent::Insert(label.into(), ResolvedIdent::Backend(0)).into(),
                BackendEvent::EmitItem(Item::Data(Atom::Name(0).into(), Width::Word)).into()
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
            [
                NameTableEvent::StartScope(label.into()).into(),
                NameTableEvent::Insert(label.into(), ResolvedIdent::Backend(0)).into(),
                BackendEvent::DefineSymbol((0, ()), LocationCounter.into()).into()
            ]
        )
    }

    #[test]
    fn analyze_org_dot() {
        let actions = collect_semantic_actions(|actions| {
            let mut actions = actions
                .key_lookup("ORG".into(), ())
                .command()
                .unwrap()
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
            [Token::Ident("XOR".into()), Token::Ident("A".into())],
        )
    }

    #[test]
    fn define_unary_macro() {
        let param = "reg";
        test_macro_definition(
            "my_xor",
            [param],
            [Token::Ident("XOR".into()), Token::Ident(param.into())],
        )
    }

    #[test]
    fn define_nameless_macro() {
        let actions = collect_semantic_actions(|actions| {
            actions
                .key_lookup("MACRO".into(), ())
                .macro_def()
                .unwrap()
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
        body: impl Borrow<[SemanticToken<String, String>]>,
    ) {
        let actions = collect_semantic_actions(|actions| {
            let mut params_actions = actions.enter_labeled_stmt((name.into(), ()));
            for param in params.borrow().iter().map(|&t| (t.into(), ())) {
                params_actions.add_parameter(param)
            }
            let mut token_seq_actions = params_actions
                .next()
                .key_lookup("MACRO".into(), ())
                .macro_def()
                .unwrap();
            for token in body.borrow().iter().cloned().map(|t| (t, ())) {
                token_seq_actions.push_token(token)
            }
            token_seq_actions.exit().exit()
        });
        assert_eq!(
            actions,
            [
                SessionEvent::DefineMacro(
                    params.borrow().iter().cloned().map(Into::into).collect(),
                    body.borrow().to_vec()
                )
                .into(),
                NameTableEvent::Insert(name.into(), ResolvedIdent::Macro(MockMacroId(0))).into(),
            ]
        )
    }

    #[test]
    fn diagnose_wrong_operand_count() {
        let actions = collect_semantic_actions(|actions| {
            let mut arg = actions
                .key_lookup("NOP".into(), ())
                .command()
                .unwrap()
                .add_argument();
            arg.push_atom((ExprAtom::Ident("A".into()), ()));
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
        let actions = collect_semantic_actions(|mut actions| {
            actions.emit_diag(diagnostic.clone());
            actions.exit()
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
                .key_lookup("ADD".into(), ())
                .command()
                .unwrap()
                .add_argument();
            expr.emit_diag(diagnostic.clone());
            expr.exit().exit().exit()
        });
        assert_eq!(
            actions,
            [DiagnosticsEvent::EmitDiag(diagnostic.into()).into()]
        )
    }

    #[test]
    fn diagnose_unknown_key() {
        let name = "unknown";
        let log = collect_semantic_actions::<_, MockSpan<_>>(|session| {
            session
                .key_lookup(name.into(), name.into())
                .error()
                .unwrap()
                .exit()
        });
        assert_eq!(
            log,
            [DiagnosticsEvent::EmitDiag(
                Message::UndefinedMacro { name: name.into() }
                    .at(name.into())
                    .into()
            )
            .into()]
        )
    }

    #[test]
    fn diagnose_reloc_name_as_key() {
        let name = "symbol";
        let log = log_with_predefined_names::<_, _, MockSpan<_>>(
            vec![(name.into(), ResolvedIdent::Backend(42))],
            |session| {
                session
                    .key_lookup(name.into(), name.into())
                    .error()
                    .unwrap()
                    .exit()
            },
        );
        assert_eq!(
            log,
            [DiagnosticsEvent::EmitDiag(
                Message::CannotUseSymbolNameAsMacroName { name: name.into() }
                    .at(name.into())
                    .into()
            )
            .into()]
        )
    }

    pub(super) type MockSession<S> = crate::analysis::session::MockSession<
        SerialIdAllocator,
        BasicNameTable<usize, MockMacroId>,
        TestOperation<S>,
        S,
    >;

    pub(super) fn collect_semantic_actions<F, S>(f: F) -> Vec<TestOperation<S>>
    where
        F: FnOnce(TestSemanticActions<S>) -> TestSemanticActions<S>,
        S: Clone + Debug + Merge,
    {
        with_log(|log| {
            f(SemanticActions::new(MockSession::with_log(log)));
        })
    }

    pub(super) fn log_with_predefined_names<I, F, S>(entries: I, f: F) -> Vec<TestOperation<S>>
    where
        I: IntoIterator<Item = (String, ResolvedIdent<usize, MockMacroId>)>,
        F: FnOnce(TestSemanticActions<S>) -> TestSemanticActions<S>,
        S: Clone + Debug + Merge,
    {
        with_log(|log| {
            f(SemanticActions::new(MockSession::with_predefined_names(
                log, entries,
            )));
        })
    }

    pub(super) type TestSemanticActions<S> = SemanticActions<MockSession<S>>;
}
