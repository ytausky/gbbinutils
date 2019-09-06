use self::token_line::TokenContextFinalizationSemantics;

use super::builtin_instr::Dispatch;
use super::Session;
use super::*;

use crate::analyze::reentrancy::{IntoSemanticActions, ReentrancyActions};
use crate::analyze::semantics::resolve::{NameTable, StartScope};
use crate::analyze::syntax::actions::*;
use crate::analyze::Literal;
use crate::codebase::CodebaseError;
use crate::diag::Message;
use crate::object::builder::Backend;

use std::ops::DerefMut;

mod instr_line;
mod token_line;

type WithoutReentrancy<I, R, N, B> = Session<
    I,
    (),
    N,
    B,
    TokenStreamState<
        <R as IdentSource>::Ident,
        <R as StringSource>::StringRef,
        <R as SpanSource>::Span,
    >,
>;

impl<I, R, N, B> Session<I, R, N, B, TokenStreamState<R::Ident, R::StringRef, R::Span>>
where
    I: BuiltinInstrSet<R>,
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword<I::Binding, I::Free>,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
    BuiltinInstr<&'static I::Binding, &'static I::Free, R>: Dispatch<I, R>,
{
    pub fn analyze_file(self, path: R::StringRef) -> Result<(), CodebaseError> {
        let (reentrancy, session) = self.split_reentrancy();
        reentrancy.analyze_file(path, session).0
    }

    fn split_reentrancy(self) -> (R, WithoutReentrancy<I, R, N, B>) {
        (
            self.reentrancy,
            Session {
                instr_set: self.instr_set,
                reentrancy: (),
                names: self.names,
                builder: self.builder,
                state: self.state,
                stack: self.stack,
            },
        )
    }
}

impl<I, R: ReentrancyActions, N, B>
    IntoSemanticActions<Session<I, (), N, B, TokenStreamState<R::Ident, R::StringRef, R::Span>>>
    for R
{
    type SemanticActions = TokenStreamSemantics<I, R, N, B>;

    fn into_semantic_actions(
        self,
        session: WithoutReentrancy<I, R, N, B>,
    ) -> Self::SemanticActions {
        Session {
            instr_set: session.instr_set,
            reentrancy: self,
            names: session.names,
            builder: session.builder,
            state: session.state,
            stack: session.stack,
        }
    }
}

impl<I, R, S> From<InstrLineState<I, S>> for TokenStreamState<I, R, S> {
    fn from(actions: InstrLineState<I, S>) -> Self {
        Self {
            mode: LineRule::InstrLine(actions),
        }
    }
}

impl<I, R, S> From<TokenLineState<I, R, S>> for TokenStreamState<I, R, S> {
    fn from(actions: TokenLineState<I, R, S>) -> Self {
        Self {
            mode: LineRule::TokenLine(actions),
        }
    }
}

impl<I, R, N, B> TokenStreamActions<R::Ident, Literal<R::StringRef>, R::Span>
    for TokenStreamSemantics<I, R, N, B>
where
    I: BuiltinInstrSet<R>,
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword<I::Binding, I::Free>,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
    BuiltinInstr<&'static I::Binding, &'static I::Free, R>: Dispatch<I, R>,
{
    type InstrLineActions = InstrLineSemantics<I, R, N, B>;
    type TokenLineActions = TokenLineSemantics<I, R, N, B>;
    type TokenLineFinalizer = TokenContextFinalizationSemantics<I, R, N, B>;

    fn will_parse_line(self) -> LineRule<Self::InstrLineActions, Self::TokenLineActions> {
        match self.state.mode {
            LineRule::InstrLine(state) => LineRule::InstrLine(set_state!(self, state)),
            LineRule::TokenLine(state) => LineRule::TokenLine(set_state!(self, state)),
        }
    }

    fn act_on_eos(mut self, span: R::Span) -> Self {
        match self.state.mode {
            LineRule::InstrLine(state) => {
                let semantics = set_state!(self, state).flush_label();
                set_state!(semantics, semantics.state.into())
            }
            LineRule::TokenLine(ref state) => {
                match state.context {
                    TokenContext::FalseIf => unimplemented!(),
                    TokenContext::MacroDef(_) => {
                        self.reentrancy.emit_diag(Message::UnexpectedEof.at(span))
                    }
                }
                self
            }
        }
    }
}

impl<I, R: ReentrancyActions, N, B> InstrFinalizer<R::Span> for InstrLineSemantics<I, R, N, B> {
    type Next = TokenStreamSemantics<I, R, N, B>;

    fn did_parse_instr(self) -> Self::Next {
        set_state!(self, self.state.into())
    }
}

impl<I, R: ReentrancyActions, N, B> LineFinalizer<R::Span> for InstrLineSemantics<I, R, N, B> {
    type Next = TokenStreamSemantics<I, R, N, B>;

    fn did_parse_line(self, _: R::Span) -> Self::Next {
        set_state!(self, self.state.into())
    }
}

impl<I, R: ReentrancyActions, N, B> LineFinalizer<R::Span> for TokenStreamSemantics<I, R, N, B> {
    type Next = Self;

    fn did_parse_line(self, _: R::Span) -> Self::Next {
        self
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub use crate::analyze::semantics::resolve::BasicNameTable;

    use crate::analyze::macros::mock::MockMacroId;
    use crate::analyze::reentrancy::ReentrancyEvent;
    use crate::analyze::semantics::builtin_instr::directive::BindingDirective;
    use crate::analyze::semantics::builtin_instr::*;
    use crate::analyze::semantics::resolve::{MockNameTable, NameTableEvent, ResolvedName};
    use crate::analyze::syntax::{Sigil, Token};
    use crate::analyze::SemanticToken;
    use crate::diag::{DiagnosticsEvent, EmitDiag, Merge, Message, MockSpan};
    use crate::expr::{Atom, BinOp, ExprOp, LocationCounter};
    use crate::log::with_log;
    use crate::object::builder::mock::*;
    use crate::object::builder::{CpuInstr, Item, Ld, Name, SimpleOperand, Width};

    use std::borrow::Borrow;
    use std::fmt::Debug;

    #[derive(Debug, PartialEq)]
    pub(in crate::analyze) enum TestOperation<S: Clone> {
        Backend(BackendEvent<MockSymbolId, Expr<S>>),
        Diagnostics(DiagnosticsEvent<S>),
        NameTable(
            NameTableEvent<
                &'static Keyword<BindingDirective, FreeBuiltinMnemonic>,
                MockMacroId,
                MockSymbolId,
            >,
        ),
        Reentrancy(ReentrancyEvent),
    }

    pub(in crate::analyze::semantics) type Expr<S> =
        crate::expr::Expr<Atom<LocationCounter, MockSymbolId>, S>;

    impl<S: Clone> From<BackendEvent<MockSymbolId, Expr<S>>> for TestOperation<S> {
        fn from(event: BackendEvent<MockSymbolId, Expr<S>>) -> Self {
            TestOperation::Backend(event)
        }
    }

    impl<S: Clone> From<DiagnosticsEvent<S>> for TestOperation<S> {
        fn from(event: DiagnosticsEvent<S>) -> Self {
            TestOperation::Diagnostics(event)
        }
    }

    impl<S: Clone>
        From<
            NameTableEvent<
                &'static Keyword<BindingDirective, FreeBuiltinMnemonic>,
                MockMacroId,
                MockSymbolId,
            >,
        > for TestOperation<S>
    {
        fn from(
            event: NameTableEvent<
                &'static Keyword<BindingDirective, FreeBuiltinMnemonic>,
                MockMacroId,
                MockSymbolId,
            >,
        ) -> Self {
            TestOperation::NameTable(event)
        }
    }

    impl<S: Clone> From<ReentrancyEvent> for TestOperation<S> {
        fn from(event: ReentrancyEvent) -> Self {
            TestOperation::Reentrancy(event)
        }
    }

    #[test]
    fn emit_ld_b_deref_hl() {
        let actions = collect_semantic_actions(|actions| {
            let mut command = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("LD".into(), ())
                .into_builtin_instr();
            let mut arg1 = command.will_parse_arg();
            arg1.act_on_atom(ExprAtom::Ident("B".into()), ());
            command = arg1.did_parse_arg();
            let mut arg2 = command.will_parse_arg();
            arg2.act_on_atom(ExprAtom::Ident("HL".into()), ());
            arg2.act_on_operator(Operator::Unary(UnaryOperator::Parentheses), ());
            arg2.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [
                BackendEvent::EmitItem(Item::CpuInstr(CpuInstr::Ld(Ld::Simple(
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
        use crate::expr::*;

        let actions = collect_semantic_actions(|actions| {
            let command = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("RST".into(), ())
                .into_builtin_instr();
            let mut expr = command.will_parse_arg();
            expr.act_on_atom(ExprAtom::Literal(Literal::Number(1)), ());
            expr.act_on_atom(ExprAtom::Literal(Literal::Number(1)), ());
            expr.act_on_operator(Operator::Binary(op), ());
            expr.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [
                BackendEvent::EmitItem(Item::CpuInstr(CpuInstr::Rst(Expr::from_items(&[
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
            let command = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("RST".into(), ())
                .into_builtin_instr();
            let mut expr = command.will_parse_arg();
            expr.act_on_atom(ExprAtom::Ident(ident.clone()), ());
            expr.act_on_atom(ExprAtom::Literal(Literal::Number(1)), ());
            expr.act_on_operator(Operator::FnCall(1), ());
            expr.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [
                NameTableEvent::Insert(ident, ResolvedName::Symbol(MockSymbolId(0))).into(),
                BackendEvent::EmitItem(Item::CpuInstr(CpuInstr::Rst(Expr::from_items(&[
                    1.into(),
                    Name(MockSymbolId(0)).into(),
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
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("DW".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            arg.act_on_atom(ExprAtom::Ident(label.into()), ());
            arg.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [
                NameTableEvent::Insert(label.into(), ResolvedName::Symbol(MockSymbolId(0))).into(),
                BackendEvent::EmitItem(Item::Data(Atom::Name(MockSymbolId(0)).into(), Width::Word))
                    .into()
            ]
        );
    }

    #[test]
    fn analyze_label() {
        let label = "label";
        let actions = collect_semantic_actions(|actions| {
            actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_label((label.into(), ()))
                .did_parse_label()
                .did_parse_line(())
                .act_on_eos(())
        });
        assert_eq!(
            actions,
            [
                NameTableEvent::StartScope(label.into()).into(),
                NameTableEvent::Insert(label.into(), ResolvedName::Symbol(MockSymbolId(0))).into(),
                BackendEvent::DefineSymbol((MockSymbolId(0), ()), LocationCounter.into()).into()
            ]
        )
    }

    #[test]
    fn analyze_org_dot() {
        let actions = collect_semantic_actions(|actions| {
            let mut actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("ORG".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            actions.act_on_atom(ExprAtom::LocationCounter, ());
            actions
                .did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
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
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("MACRO".into(), ())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line(())
                .will_parse_line()
                .into_token_line()
                .act_on_mnemonic("ENDM".into(), ())
                .into_line_end()
                .did_parse_line(())
                .act_on_eos(())
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
            let mut params_actions = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_label((name.into(), ()));
            for param in params.borrow().iter().cloned().map(Into::into) {
                params_actions.act_on_param(param, ())
            }
            let mut token_seq_actions = params_actions
                .did_parse_label()
                .will_parse_instr("MACRO".into(), ())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line(())
                .will_parse_line()
                .into_token_line();
            for token in body.borrow().iter().cloned() {
                token_seq_actions.act_on_token(token, ())
            }
            token_seq_actions
                .act_on_mnemonic("ENDM".into(), ())
                .into_line_end()
                .did_parse_line(())
                .act_on_eos(())
        });
        let mut body = body.borrow().to_vec();
        body.push(Sigil::Eos.into());
        assert_eq!(
            actions,
            [
                ReentrancyEvent::DefineMacro(
                    params.borrow().iter().cloned().map(Into::into).collect(),
                    body
                )
                .into(),
                NameTableEvent::Insert(name.into(), ResolvedName::Macro(MockMacroId(0))).into(),
            ]
        )
    }

    #[test]
    fn diagnose_wrong_operand_count() {
        let actions = collect_semantic_actions(|actions| {
            let mut arg = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("NOP".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            arg.act_on_atom(ExprAtom::Ident("A".into()), ());
            arg.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
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
            actions.did_parse_line(()).act_on_eos(())
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
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("ADD".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            expr.act_on_atom(ExprAtom::Error, ());
            expr.emit_diag(diagnostic.clone());
            expr.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
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
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr(name.into(), name.into())
                .error()
                .unwrap()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .act_on_eos("eos".into())
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
            vec![(name.into(), ResolvedName::Symbol(MockSymbolId(42)))],
            |session| {
                session
                    .will_parse_line()
                    .into_instr_line()
                    .will_parse_instr(name.into(), name.into())
                    .error()
                    .unwrap()
                    .did_parse_line("eol".into())
                    .act_on_eos("eos".into())
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

    #[test]
    fn diagnose_eos_in_macro_body() {
        let log = collect_semantic_actions::<_, MockSpan<_>>(|actions| {
            actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_label(("my_macro".into(), "label".into()))
                .did_parse_label()
                .will_parse_instr("MACRO".into(), "key".into())
                .into_builtin_instr()
                .did_parse_instr()
                .did_parse_line("eol".into())
                .will_parse_line()
                .into_token_line()
                .did_parse_line("eos".into())
                .act_on_eos("eos".into())
        });
        assert_eq!(
            log,
            [DiagnosticsEvent::EmitDiag(Message::UnexpectedEof.at("eos".into()).into()).into()]
        )
    }

    pub(in crate::analyze::semantics) type MockSourceComponents<S> =
        crate::analyze::reentrancy::MockSourceComponents<TestOperation<S>, S>;

    pub(in crate::analyze::semantics) fn collect_semantic_actions<F, S>(
        f: F,
    ) -> Vec<TestOperation<S>>
    where
        F: FnOnce(TestTokenStreamSemantics<S>) -> TestTokenStreamSemantics<S>,
        S: Clone + Debug + Merge,
    {
        log_with_predefined_names(std::iter::empty(), f)
    }

    pub(super) fn log_with_predefined_names<I, F, S>(entries: I, f: F) -> Vec<TestOperation<S>>
    where
        I: IntoIterator<
            Item = (
                String,
                ResolvedName<
                    &'static Keyword<BindingDirective, FreeBuiltinMnemonic>,
                    MockMacroId,
                    MockSymbolId,
                >,
            ),
        >,
        F: FnOnce(TestTokenStreamSemantics<S>) -> TestTokenStreamSemantics<S>,
        S: Clone + Debug + Merge,
    {
        with_log(|log| {
            let mut session = Session::from_components(
                MockSourceComponents::with_log(log.clone()),
                Box::new(BasicNameTable::default()),
                MockBackend::new(SerialIdAllocator::new(MockSymbolId), log.clone()),
            );
            for (ident, resolution) in entries {
                session.names.define_name(ident, resolution)
            }
            f(session.map_names(|names| Box::new(MockNameTable::new(*names, log))));
        })
    }

    pub(super) type TestTokenStreamSemantics<S> = TokenStreamSemantics<
        DefaultBuiltinInstrSet,
        MockSourceComponents<S>,
        Box<
            MockNameTable<
                BasicNameTable<
                    &'static Keyword<BindingDirective, FreeBuiltinMnemonic>,
                    MockMacroId,
                    MockSymbolId,
                >,
                TestOperation<S>,
            >,
        >,
        MockBackend<SerialIdAllocator<MockSymbolId>, TestOperation<S>>,
    >;
}
