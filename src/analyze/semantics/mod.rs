use self::instr_line::{BuiltinInstr, InstrLineSemantics, InstrLineState, OperandSymbol};
use self::keywords::KEYWORDS;
use self::params::*;
use self::token_line::{TokenContext, TokenContextFinalizationSemantics, TokenLineSemantics};

use super::macros::MacroSource;
use super::reentrancy::{IntoSemanticActions, Params, ReentrancyActions};
use super::resolve::{NameTable, ResolvedName, StartScope};
use super::syntax;
use super::syntax::actions::*;
use super::Literal;

use crate::codebase::CodebaseError;
use crate::diag;
use crate::diag::{Diagnostics, Message};
use crate::expr::{BinOp, FnCall, LocationCounter, ParamId};
use crate::object::builder::{AllocSymbol, Backend, Finish, Name, PushOp, SymbolSource};

use std::ops::{Deref, DerefMut};

macro_rules! set_state {
    ($session:expr, $state:expr) => {
        $crate::analyze::semantics::Session {
            reentrancy: $session.reentrancy,
            names: $session.names,
            builder: $session.builder,
            state: $state,
        }
    };
}

mod instr_line;
mod keywords;
mod params;
mod token_line;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Keyword {
    BuiltinInstr(BuiltinInstr),
    Operand(OperandSymbol),
}

pub(super) type TokenStreamSemantics<R, N, B> = Session<R, N, B, TokenStreamState<R>>;

pub(super) struct TokenStreamState<S: ReentrancyActions>(
    LineRule<InstrLineState<S>, TokenContext<S>>,
);

impl<S: ReentrancyActions> TokenStreamState<S> {
    pub fn new() -> Self {
        Self(LineRule::InstrLine(InstrLineState::new()))
    }
}

impl<R: ReentrancyActions, N, B> IntoSemanticActions<Session<(), N, B, TokenStreamState<R>>> for R {
    type SemanticActions = TokenStreamSemantics<R, N, B>;

    fn into_semantic_actions(
        self,
        session: Session<(), N, B, TokenStreamState<R>>,
    ) -> Self::SemanticActions {
        Session {
            reentrancy: self,
            names: session.names,
            builder: session.builder,
            state: session.state,
        }
    }
}

pub(super) struct Session<R, N, B, S> {
    reentrancy: R,
    names: N,
    builder: B,
    state: S,
}

impl<R, N, B, S> Session<R, N, B, S> {
    #[cfg(test)]
    fn map_names<F: FnOnce(N) -> T, T>(self, f: F) -> Session<R, T, B, S> {
        Session {
            reentrancy: self.reentrancy,
            names: f(self.names),
            builder: self.builder,
            state: self.state,
        }
    }

    fn map_builder<F: FnOnce(B) -> T, T>(self, f: F) -> Session<R, N, T, S> {
        Session {
            reentrancy: self.reentrancy,
            names: self.names,
            builder: f(self.builder),
            state: self.state,
        }
    }

    fn map_state<F: FnOnce(S) -> T, T>(self, f: F) -> Session<R, N, B, T> {
        Session {
            reentrancy: self.reentrancy,
            names: self.names,
            builder: self.builder,
            state: f(self.state),
        }
    }
}

impl<R, N, B> Session<R, N, B, TokenStreamState<R>>
where
    R: ReentrancyActions,
    R::Ident: for<'r> From<&'r str>,
    N: DerefMut,
    N::Target: NameTable<R::Ident, Keyword = &'static Keyword>,
{
    pub fn from_components(reentrancy: R, mut names: N, builder: B) -> Self {
        for (ident, keyword) in KEYWORDS {
            names.define_name((*ident).into(), ResolvedName::Keyword(keyword))
        }
        Self {
            reentrancy,
            names,
            builder,
            state: TokenStreamState::new(),
        }
    }
}

impl<R, N, B> Session<R, N, B, TokenStreamState<R>>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
{
    pub fn analyze_file(self, path: R::StringRef) -> Result<(), CodebaseError> {
        let (reentrancy, session) = self.split_reentrancy();
        reentrancy.analyze_file(path, session).0
    }

    fn split_reentrancy(self) -> (R, Session<(), N, B, TokenStreamState<R>>) {
        (
            self.reentrancy,
            Session {
                reentrancy: (),
                names: self.names,
                builder: self.builder,
                state: self.state,
            },
        )
    }
}

delegate_diagnostics! {
    {R: Diagnostics<Span>, N, B, S, Span}, Session<R, N, B, S>, {reentrancy}, R, Span
}

impl<R, N, B, S> MacroSource for Session<R, N, B, S>
where
    N: Deref,
    N::Target: MacroSource,
{
    type MacroId = <N::Target as MacroSource>::MacroId;
}

impl<R, N, B, S> SymbolSource for Session<R, N, B, S>
where
    N: Deref,
    N::Target: SymbolSource,
{
    type SymbolId = <N::Target as SymbolSource>::SymbolId;
}

impl<R, N, B, S, Span> AllocSymbol<Span> for Session<R, N, B, S>
where
    N: Deref,
    N::Target: SymbolSource<SymbolId = B::SymbolId>,
    B: AllocSymbol<Span>,
    Span: Clone,
{
    fn alloc_symbol(&mut self, span: Span) -> Self::SymbolId {
        self.builder.alloc_symbol(span)
    }
}

impl<R, N, B, S, I> NameTable<I> for Session<R, N, B, S>
where
    N: DerefMut,
    N::Target: NameTable<I>,
{
    type Keyword = <N::Target as NameTable<I>>::Keyword;

    fn resolve_name(
        &mut self,
        ident: &I,
    ) -> Option<ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>> {
        self.names.resolve_name(ident)
    }

    fn define_name(
        &mut self,
        ident: I,
        entry: ResolvedName<Self::Keyword, Self::MacroId, Self::SymbolId>,
    ) {
        self.names.define_name(ident, entry)
    }
}

impl<R, N, B: Finish, S> Finish for Session<R, N, B, S> {
    type Value = B::Value;
    type Parent = Session<R, N, B::Parent, S>;

    fn finish(self) -> (Self::Parent, Self::Value) {
        let (builder, value) = self.builder.finish();
        (
            Session {
                reentrancy: self.reentrancy,
                names: self.names,
                builder,
                state: self.state,
            },
            value,
        )
    }
}

impl<R, N, B, S, Span, SymbolId> PushOp<Name<SymbolId>, Span> for Session<R, N, B, S>
where
    B: PushOp<Name<SymbolId>, Span>,
    Span: Clone,
{
    fn push_op(&mut self, op: Name<SymbolId>, span: Span) {
        self.builder.push_op(op, span)
    }
}

macro_rules! impl_push_op_for_session {
    ($t:ty) => {
        impl<R, N, B: PushOp<$t, Span>, S, Span: Clone> PushOp<$t, Span> for Session<R, N, B, S> {
            fn push_op(&mut self, op: $t, span: Span) {
                self.builder.push_op(op, span)
            }
        }
    };
}

impl_push_op_for_session! {LocationCounter}
impl_push_op_for_session! {i32}
impl_push_op_for_session! {BinOp}
impl_push_op_for_session! {ParamId}
impl_push_op_for_session! {FnCall}

impl<S: ReentrancyActions> From<InstrLineState<S>> for TokenStreamState<S> {
    fn from(actions: InstrLineState<S>) -> Self {
        Self(LineRule::InstrLine(actions))
    }
}

impl<S: ReentrancyActions> From<TokenContext<S>> for TokenStreamState<S> {
    fn from(actions: TokenContext<S>) -> Self {
        Self(LineRule::TokenLine(actions))
    }
}

impl<R, N, B> TokenStreamActions<R::Ident, Literal<R::StringRef>, R::Span>
    for TokenStreamSemantics<R, N, B>
where
    R: ReentrancyActions,
    N: DerefMut,
    N::Target: StartScope<R::Ident>
        + NameTable<
            R::Ident,
            Keyword = &'static Keyword,
            MacroId = R::MacroId,
            SymbolId = B::SymbolId,
        >,
    B: Backend<R::Span>,
{
    type InstrLineActions = InstrLineSemantics<R, N, B>;
    type TokenLineActions = TokenLineSemantics<R, N, B>;
    type TokenLineFinalizer = TokenContextFinalizationSemantics<R, N, B>;

    fn will_parse_line(self) -> LineRule<Self::InstrLineActions, Self::TokenLineActions> {
        match self.state.0 {
            LineRule::InstrLine(state) => LineRule::InstrLine(set_state!(self, state)),
            LineRule::TokenLine(state) => LineRule::TokenLine(set_state!(self, state)),
        }
    }

    fn act_on_eos(mut self, span: R::Span) -> Self {
        match self.state.0 {
            LineRule::InstrLine(state) => {
                let semantics = set_state!(self, state).flush_label();
                set_state!(semantics, semantics.state.into())
            }
            LineRule::TokenLine(ref state) => {
                match state {
                    TokenContext::MacroDef(_) => {
                        self.reentrancy.emit_diag(Message::UnexpectedEof.at(span))
                    }
                }
                self
            }
        }
    }
}

type Label<I, S> = ((I, S), Params<I, S>);

impl<R: ReentrancyActions, N, B> InstrFinalizer<R::Span> for InstrLineSemantics<R, N, B> {
    type Next = TokenStreamSemantics<R, N, B>;

    fn did_parse_instr(self) -> Self::Next {
        set_state!(self, self.state.into())
    }
}

impl<R: ReentrancyActions, N, B> LineFinalizer<R::Span> for InstrLineSemantics<R, N, B> {
    type Next = TokenStreamSemantics<R, N, B>;

    fn did_parse_line(self, _: R::Span) -> Self::Next {
        set_state!(self, self.state.into())
    }
}

impl<R: ReentrancyActions, N, B> LineFinalizer<R::Span> for TokenStreamSemantics<R, N, B> {
    type Next = Self;

    fn did_parse_line(self, _: R::Span) -> Self::Next {
        self
    }
}

#[cfg(test)]
mod mock {
    use super::*;

    use crate::analyze::macros::mock::MockMacroId;
    use crate::analyze::resolve::{BasicNameTable, MockNameTable};
    use crate::diag::{DiagnosticsEvent, Merge, MockDiagnostics};
    use crate::expr::{Atom, Expr};
    use crate::log::Log;
    use crate::object::builder::mock::{
        BackendEvent, MockBackend, MockSymbolId, SerialIdAllocator,
    };
    use crate::object::builder::{Backend, RelocContext};

    pub(super) type MockExprBuilder<T, S> = Session<
        MockDiagnostics<T, S>,
        Box<MockNameTable<BasicNameTable<&'static Keyword, MockMacroId, MockSymbolId>, T>>,
        RelocContext<
            MockBackend<SerialIdAllocator<MockSymbolId>, T>,
            Expr<Atom<LocationCounter, MockSymbolId>, S>,
        >,
        (),
    >;

    impl<T, S> MockExprBuilder<T, S>
    where
        T: From<BackendEvent<MockSymbolId, Expr<Atom<LocationCounter, MockSymbolId>, S>>>,
        T: From<DiagnosticsEvent<S>>,
        S: Clone + Merge,
    {
        pub fn with_log(log: Log<T>) -> Self {
            Self::with_name_table_entries(log, std::iter::empty())
        }

        pub fn with_name_table_entries<I>(log: Log<T>, entries: I) -> Self
        where
            I: IntoIterator<
                Item = (
                    String,
                    ResolvedName<&'static Keyword, MockMacroId, MockSymbolId>,
                ),
            >,
        {
            let mut names = BasicNameTable::default();
            for (ident, resolution) in entries {
                names.define_name(ident, resolution)
            }
            Session {
                reentrancy: MockDiagnostics::new(log.clone()),
                names: Box::new(MockNameTable::new(names, log.clone())),
                builder: MockBackend::new(SerialIdAllocator::new(MockSymbolId), log).build_const(),
                state: (),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub use crate::analyze::resolve::BasicNameTable;

    use super::syntax::{Sigil, Token};

    use crate::analyze::macros::mock::MockMacroId;
    use crate::analyze::reentrancy::ReentrancyEvent;
    use crate::analyze::resolve::{MockNameTable, NameTableEvent, ResolvedName};
    use crate::analyze::SemanticToken;
    use crate::diag::{DiagnosticsEvent, EmitDiag, Merge, Message, MockSpan};
    use crate::expr::{Atom, BinOp, ExprOp, LocationCounter};
    use crate::log::with_log;
    use crate::object::builder::mock::{
        BackendEvent, MockBackend, MockSymbolId, SerialIdAllocator,
    };
    use crate::object::builder::{CpuInstr, Item, Ld, Name, SimpleOperand, Width};

    use std::borrow::Borrow;
    use std::fmt::Debug;

    #[derive(Debug, PartialEq)]
    pub(in crate::analyze) enum TestOperation<S: Clone> {
        Backend(BackendEvent<MockSymbolId, Expr<S>>),
        Diagnostics(DiagnosticsEvent<S>),
        NameTable(NameTableEvent<&'static Keyword, MockMacroId, MockSymbolId>),
        Reentrancy(ReentrancyEvent),
    }

    pub(super) type Expr<S> = crate::expr::Expr<Atom<LocationCounter, MockSymbolId>, S>;

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

    impl<S: Clone> From<NameTableEvent<&'static Keyword, MockMacroId, MockSymbolId>>
        for TestOperation<S>
    {
        fn from(event: NameTableEvent<&'static Keyword, MockMacroId, MockSymbolId>) -> Self {
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
                .act_on_ident("ENDM".into(), ())
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
                .act_on_ident("ENDM".into(), ())
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

    pub(super) type MockSourceComponents<S> =
        crate::analyze::reentrancy::MockSourceComponents<TestOperation<S>, S>;

    pub(super) fn collect_semantic_actions<F, S>(f: F) -> Vec<TestOperation<S>>
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
                ResolvedName<&'static Keyword, MockMacroId, MockSymbolId>,
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
        MockSourceComponents<S>,
        Box<
            MockNameTable<
                BasicNameTable<&'static Keyword, MockMacroId, MockSymbolId>,
                TestOperation<S>,
            >,
        >,
        MockBackend<SerialIdAllocator<MockSymbolId>, TestOperation<S>>,
    >;
}
