use self::token_line::TokenContextFinalizationSemantics;

use super::Semantics;
use super::*;

use crate::analyze::Literal;
use crate::diag::{CompactDiag, Message};
use crate::session::Session;
use crate::span::StripSpan;
use crate::syntax::actions::*;
use crate::syntax::LexError;

mod instr_line;
mod token_line;

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

impl<'a, 'b, S, T, I, R, Z> ParsingContext for Semantics<'a, 'b, S, T, I, R, Z>
where
    S: Diagnostics<Z>,
    Z: Clone,
{
    type Ident = I;
    type Literal = Literal<R>;
    type Error = LexError;
    type Span = Z;
    type Stripped = <S as StripSpan<Z>>::Stripped;

    fn next_token(
        &mut self,
    ) -> Option<LexerOutput<Self::Ident, Self::Literal, Self::Error, Self::Span>> {
        self.tokens.next()
    }

    fn merge_spans(&mut self, left: &Self::Span, right: &Self::Span) -> Self::Span {
        self.session.merge_spans(left, right)
    }

    fn strip_span(&mut self, span: &Self::Span) -> Self::Stripped {
        self.session.strip_span(span)
    }

    fn emit_diag(&mut self, diag: impl Into<CompactDiag<Self::Span, Self::Stripped>>) {
        self.session.emit_diag(diag)
    }
}

impl<'a, 'b, S: Session> TokenStreamContext for TokenStreamSemantics<'a, 'b, S>
where
    S::Ident: 'static,
    S::StringRef: 'static,
    S::Span: 'static,
{
    type InstrLineContext = InstrLineSemantics<'a, 'b, S>;
    type TokenLineContext = TokenLineSemantics<'a, 'b, S>;
    type TokenLineFinalizer = TokenContextFinalizationSemantics<'a, 'b, S>;

    fn will_parse_line(self) -> LineRule<Self::InstrLineContext, Self::TokenLineContext> {
        match self.state.mode {
            LineRule::InstrLine(state) => LineRule::InstrLine(set_state!(self, state)),
            LineRule::TokenLine(state) => LineRule::TokenLine(set_state!(self, state)),
        }
    }

    fn act_on_eos(self, span: S::Span) -> Self {
        match self.state.mode {
            LineRule::InstrLine(state) => {
                let semantics = set_state!(self, state).flush_label();
                set_state!(semantics, semantics.state.into())
            }
            LineRule::TokenLine(ref state) => {
                match state.context {
                    TokenContext::FalseIf => unimplemented!(),
                    TokenContext::MacroDef(_) => {
                        self.session.emit_diag(Message::UnexpectedEof.at(span))
                    }
                }
                self
            }
        }
    }
}

impl<'a, 'b, S: Session> InstrFinalizer for InstrLineSemantics<'a, 'b, S> {
    type Next = TokenStreamSemantics<'a, 'b, S>;

    fn did_parse_instr(self) -> Self::Next {
        set_state!(self, self.state.into())
    }
}

impl<'a, 'b, S: Session> LineFinalizer for InstrLineSemantics<'a, 'b, S> {
    type Next = TokenStreamSemantics<'a, 'b, S>;

    fn did_parse_line(self, _: S::Span) -> Self::Next {
        set_state!(self, self.state.into())
    }
}

impl<'a, 'b, S: Session> LineFinalizer for TokenStreamSemantics<'a, 'b, S> {
    type Next = Self;

    fn did_parse_line(self, _: S::Span) -> Self::Next {
        self
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub use crate::session::resolve::BasicNameTable;

    use crate::analyze::macros::mock::MockMacroId;
    use crate::analyze::SemanticToken;
    use crate::diag::{DiagnosticsEvent, Merge, Message, MockDiagnostics, MockSpan};
    use crate::expr::{Atom, BinOp, ExprOp, LocationCounter};
    use crate::log::with_log;
    use crate::object::Fragment;
    use crate::session::builder::mock::*;
    use crate::session::builder::Width;
    use crate::session::reentrancy::ReentrancyEvent;
    use crate::session::resolve::*;
    use crate::session::CompositeSession;
    use crate::syntax::{Sigil, Token};

    use std::borrow::Borrow;
    use std::fmt::Debug;

    #[derive(Debug, PartialEq)]
    pub(crate) enum TestOperation<S: Clone> {
        Backend(BackendEvent<MockSymbolId, Expr<S>>),
        Diagnostics(DiagnosticsEvent<S>),
        NameTable(NameTableEvent<MockMacroId, MockSymbolId>),
        Reentrancy(ReentrancyEvent),
    }

    type Expr<S> = crate::expr::Expr<MockSymbolId, S>;

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

    impl<S: Clone> From<NameTableEvent<MockMacroId, MockSymbolId>> for TestOperation<S> {
        fn from(event: NameTableEvent<MockMacroId, MockSymbolId>) -> Self {
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
            [BackendEvent::EmitFragment(Fragment::Byte(0x46)).into()]
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
            [BackendEvent::EmitFragment(Fragment::Embedded(
                0b11_000_111,
                Expr::from_items(&[1.into(), 1.into(), op.into()])
            ))
            .into()]
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
                BackendEvent::EmitFragment(Fragment::Embedded(
                    0b11_000_111,
                    Expr::from_items(&[
                        Atom::Name(MockSymbolId(0)).into(),
                        1.into(),
                        ExprOp::FnCall(1).into()
                    ])
                ))
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
                BackendEvent::EmitFragment(Fragment::Immediate(
                    Atom::Name(MockSymbolId(0)).into(),
                    Width::Word
                ))
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

    pub(crate) type MockSourceComponents<S> =
        crate::session::reentrancy::MockSourceComponents<TestOperation<S>, S>;

    pub(crate) fn collect_semantic_actions<F, S>(f: F) -> Vec<TestOperation<S>>
    where
        F: for<'a, 'b> FnOnce(
            TestTokenStreamSemantics<'a, 'b, S>,
        ) -> TestTokenStreamSemantics<'a, 'b, S>,
        S: Clone + Debug + Merge,
    {
        log_with_predefined_names(std::iter::empty(), f)
    }

    pub(super) fn log_with_predefined_names<I, F, S>(entries: I, f: F) -> Vec<TestOperation<S>>
    where
        I: IntoIterator<Item = (String, ResolvedName<MockMacroId, MockSymbolId>)>,
        F: for<'a, 'b> FnOnce(
            TestTokenStreamSemantics<'a, 'b, S>,
        ) -> TestTokenStreamSemantics<'a, 'b, S>,
        S: Clone + Debug + Merge,
    {
        with_log(|log| {
            let mut session = CompositeSession::from_components(
                MockSourceComponents::with_log(log.clone()),
                BasicNameTable::default(),
                MockBackend::new(SerialIdAllocator::new(MockSymbolId), log.clone()),
                MockDiagnostics::new(log.clone()),
            );
            for (ident, resolution) in entries {
                session.define_name(ident, resolution)
            }
            let mut session = CompositeSession {
                reentrancy: session.reentrancy,
                names: MockNameTable::new(session.names, log),
                builder: session.builder,
                diagnostics: session.diagnostics,
            };
            let mut tokens = std::iter::empty();
            f(Semantics {
                session: &mut session,
                state: TokenStreamState::new(),
                tokens: &mut tokens,
            });
        })
    }

    pub(super) type TestTokenStreamSemantics<'a, 'b, S> = TokenStreamSemantics<
        'a,
        'b,
        CompositeSession<
            MockSourceComponents<S>,
            MockNameTable<BasicNameTable<MockMacroId, MockSymbolId>, TestOperation<S>>,
            MockBackend<SerialIdAllocator<MockSymbolId>, TestOperation<S>>,
            MockDiagnostics<TestOperation<S>, S>,
        >,
    >;
}
