use self::builtin_instr::*;
use self::builtin_instr::{BuiltinInstr::*, Directive::*};
use self::macro_instr::MacroInstrSemantics;
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

mod builtin_instr;
mod macro_instr;
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
        let Done(session) =
            super::syntax::parse_token_seq(tokens.into_iter(), TokenStreamSemantics::new(session));
        session.into()
    }
}

pub(super) struct TokenStreamSemantics<S: Session> {
    mode: LineRule<InstrLineSemantics<S>, TokenLineSemantics<S>>,
}

type InstrLineSemantics<S> = SemanticState<InstrLineState<S>, S>;
type TokenLineSemantics<S> = SemanticState<TokenContext<S>, S>;

pub(super) struct SemanticState<L, S: Session> {
    line: L,
    session: S,
}

impl<L, S: Session> SemanticState<L, S> {
    fn build_value<F, T>(mut self, params: &Params<S::Ident, S::Span>, f: F) -> (T, Self)
    where
        F: FnOnce(
            BuilderAdapter<
                BuilderAdapter<S::GeneralBuilder, NameResolver>,
                ConvertParams<S::Ident, S::Span>,
            >,
        ) -> (T, S),
    {
        let (value, session) = f(self
            .session
            .build_value()
            .resolve_names()
            .with_params(params));
        self.session = session;
        (value, self)
    }
}

delegate_diagnostics! {
    {L, S: Session}, SemanticState<L, S>, {session}, S, S::Span
}

pub(super) struct InstrLineState<S: Session> {
    label: Option<Label<S::Ident, S::Span>>,
}

impl<S: Session> InstrLineSemantics<S> {
    fn new(session: S) -> Self {
        Self {
            line: InstrLineState { label: None },
            session,
        }
    }

    fn define_label_if_present(mut self) -> Self {
        if let Some(((label, span), _params)) = self.line.label.take() {
            self.session.start_scope(&label);
            let id = self.session.reloc_lookup(label, span.clone());
            let mut builder = self.session.define_symbol(id, span.clone());
            PushOp::<LocationCounter, _>::push_op(&mut builder, LocationCounter, span);
            let (session, ()) = builder.finish();
            self.session = session;
        }
        self
    }
}

pub(super) enum TokenContext<S: Session> {
    MacroDef(MacroDefState<S>),
}

pub(super) struct MacroDefState<S: Session> {
    label: Option<Label<S::Ident, S::Span>>,
    tokens: TokenSeq<S::Ident, S::StringRef, S::Span>,
}

impl<S: Session> MacroDefState<S> {
    fn new(label: Option<Label<S::Ident, S::Span>>) -> Self {
        Self {
            label,
            tokens: (Vec::new(), Vec::new()),
        }
    }
}

impl<S: Session> TokenStreamSemantics<S> {
    pub fn new(session: S) -> TokenStreamSemantics<S> {
        TokenStreamSemantics {
            mode: LineRule::InstrLine(InstrLineSemantics::new(session)),
        }
    }

    fn session(&mut self) -> &mut S {
        match &mut self.mode {
            LineRule::InstrLine(actions) => &mut actions.session,
            LineRule::TokenLine(actions) => &mut actions.session,
        }
    }
}

delegate_diagnostics! {
    {S: Session}, TokenStreamSemantics<S>, {session()}, S, S::Span
}

impl<S: Session> From<InstrLineSemantics<S>> for TokenStreamSemantics<S> {
    fn from(actions: InstrLineSemantics<S>) -> Self {
        Self {
            mode: LineRule::InstrLine(actions),
        }
    }
}

impl<S: Session> From<TokenLineSemantics<S>> for TokenStreamSemantics<S> {
    fn from(actions: TokenLineSemantics<S>) -> Self {
        Self {
            mode: LineRule::TokenLine(actions),
        }
    }
}

impl<S: Session> TokenStreamActions<S::Ident, Literal<S::StringRef>, S::Span>
    for TokenStreamSemantics<S>
where
    S::Ident: AsRef<str>,
{
    type InstrLineActions = InstrLineSemantics<S>;
    type TokenLineActions = TokenLineSemantics<S>;
    type TokenLineFinalizer = TokenContextFinalizationSemantics<S>;
    type Next = Done<S>;

    fn will_parse_line(self) -> LineRule<Self::InstrLineActions, Self::TokenLineActions> {
        self.mode
    }

    fn act_on_eos(self, span: S::Span) -> Self::Next {
        match self.mode {
            LineRule::InstrLine(actions) => Done(actions.define_label_if_present().session),
            LineRule::TokenLine(mut actions) => {
                match actions.line {
                    TokenContext::MacroDef(_) => actions.emit_diag(Message::UnexpectedEof.at(span)),
                }
                Done(actions.session)
            }
        }
    }
}

pub(super) struct Done<S>(S);

impl<S: Session> InstrLineActions<S::Ident, Literal<S::StringRef>, S::Span>
    for InstrLineSemantics<S>
where
    S::Ident: AsRef<str>,
{
    type LabelActions = LabelSemantics<S>;
    type InstrActions = Self;

    fn will_parse_label(mut self, label: (S::Ident, S::Span)) -> Self::LabelActions {
        self = self.define_label_if_present();
        LabelSemantics::new(self, label)
    }
}

pub(super) struct LabelSemantics<S: Session> {
    parent: InstrLineSemantics<S>,
    label: (S::Ident, S::Span),
    params: Params<S::Ident, S::Span>,
}

impl<S: Session> LabelSemantics<S> {
    fn new(parent: InstrLineSemantics<S>, label: (S::Ident, S::Span)) -> Self {
        Self {
            parent,
            label,
            params: (Vec::new(), Vec::new()),
        }
    }
}

delegate_diagnostics! {
    {S: Session}, LabelSemantics<S>, {parent}, S, S::Span
}

impl<S: Session> LabelActions<S::Ident, S::Span> for LabelSemantics<S> {
    type Next = InstrLineSemantics<S>;

    fn act_on_param(&mut self, (ident, span): (S::Ident, S::Span)) {
        self.params.0.push(ident);
        self.params.1.push(span)
    }

    fn did_parse_label(mut self) -> Self::Next {
        self.parent.line.label = Some((self.label, self.params));
        self.parent
    }
}

type Label<I, S> = ((I, S), Params<I, S>);

impl<S: Session> InstrActions<S::Ident, Literal<S::StringRef>, S::Span> for InstrLineSemantics<S>
where
    S::Ident: AsRef<str>,
{
    type BuiltinInstrActions = BuiltinInstrSemantics<S>;
    type MacroInstrActions = MacroInstrSemantics<S>;
    type ErrorActions = Self;
    type LineFinalizer = TokenStreamSemantics<S>;

    fn will_parse_instr(
        mut self,
        ident: S::Ident,
        span: S::Span,
    ) -> InstrRule<Self::BuiltinInstrActions, Self::MacroInstrActions, Self> {
        match KEYS
            .iter()
            .find(|(spelling, _)| spelling.eq_ignore_ascii_case(ident.as_ref()))
            .map(|(_, entry)| entry)
        {
            Some(KeyEntry::BuiltinInstr(command)) => {
                InstrRule::BuiltinInstr(BuiltinInstrSemantics::new(self, (command.clone(), span)))
            }
            None => match self.session.get(&ident) {
                Some(ResolvedIdent::Macro(id)) => {
                    self = self.define_label_if_present();
                    InstrRule::MacroInstr(MacroInstrSemantics::new(self, (id, span)))
                }
                Some(ResolvedIdent::Backend(_)) => {
                    let name = self.strip_span(&span);
                    self.emit_diag(Message::CannotUseSymbolNameAsMacroName { name }.at(span));
                    InstrRule::Error(self)
                }
                None => {
                    let name = self.strip_span(&span);
                    self.emit_diag(Message::UndefinedMacro { name }.at(span));
                    InstrRule::Error(self)
                }
            },
        }
    }
}

impl<S: Session> InstrFinalizer<S::Span> for InstrLineSemantics<S> {
    type Next = TokenStreamSemantics<S>;

    fn did_parse_instr(self) -> Self::Next {
        self.into()
    }
}

impl<S: Session> LineFinalizer<S::Span> for InstrLineSemantics<S> {
    type Next = TokenStreamSemantics<S>;

    fn did_parse_line(self, _: S::Span) -> Self::Next {
        self.into()
    }
}

impl<S: Session> LineFinalizer<S::Span> for TokenStreamSemantics<S> {
    type Next = Self;

    fn did_parse_line(self, _: S::Span) -> Self::Next {
        self
    }
}

#[derive(Clone)]
enum KeyEntry {
    BuiltinInstr(BuiltinInstr),
}

const KEYS: &[(&str, KeyEntry)] = &[
    ("adc", KeyEntry::BuiltinInstr(Mnemonic(ADC))),
    ("add", KeyEntry::BuiltinInstr(Mnemonic(ADD))),
    ("and", KeyEntry::BuiltinInstr(Mnemonic(AND))),
    ("bit", KeyEntry::BuiltinInstr(Mnemonic(BIT))),
    ("call", KeyEntry::BuiltinInstr(Mnemonic(CALL))),
    ("cp", KeyEntry::BuiltinInstr(Mnemonic(CP))),
    ("cpl", KeyEntry::BuiltinInstr(Mnemonic(CPL))),
    ("daa", KeyEntry::BuiltinInstr(Mnemonic(DAA))),
    ("db", KeyEntry::BuiltinInstr(Directive(Db))),
    ("dec", KeyEntry::BuiltinInstr(Mnemonic(DEC))),
    ("di", KeyEntry::BuiltinInstr(Mnemonic(DI))),
    ("ds", KeyEntry::BuiltinInstr(Directive(Ds))),
    ("dw", KeyEntry::BuiltinInstr(Directive(Dw))),
    ("ei", KeyEntry::BuiltinInstr(Mnemonic(EI))),
    ("equ", KeyEntry::BuiltinInstr(Directive(Equ))),
    ("halt", KeyEntry::BuiltinInstr(Mnemonic(HALT))),
    ("inc", KeyEntry::BuiltinInstr(Mnemonic(INC))),
    ("include", KeyEntry::BuiltinInstr(Directive(Include))),
    ("jp", KeyEntry::BuiltinInstr(Mnemonic(JP))),
    ("jr", KeyEntry::BuiltinInstr(Mnemonic(JR))),
    ("ld", KeyEntry::BuiltinInstr(Mnemonic(LD))),
    ("ldhl", KeyEntry::BuiltinInstr(Mnemonic(LDHL))),
    ("macro", KeyEntry::BuiltinInstr(Directive(Macro))),
    ("nop", KeyEntry::BuiltinInstr(Mnemonic(NOP))),
    ("or", KeyEntry::BuiltinInstr(Mnemonic(OR))),
    ("org", KeyEntry::BuiltinInstr(Directive(Org))),
    ("pop", KeyEntry::BuiltinInstr(Mnemonic(POP))),
    ("push", KeyEntry::BuiltinInstr(Mnemonic(PUSH))),
    ("res", KeyEntry::BuiltinInstr(Mnemonic(RES))),
    ("ret", KeyEntry::BuiltinInstr(Mnemonic(RET))),
    ("reti", KeyEntry::BuiltinInstr(Mnemonic(RETI))),
    ("rl", KeyEntry::BuiltinInstr(Mnemonic(RL))),
    ("rla", KeyEntry::BuiltinInstr(Mnemonic(RLA))),
    ("rlc", KeyEntry::BuiltinInstr(Mnemonic(RLC))),
    ("rlca", KeyEntry::BuiltinInstr(Mnemonic(RLCA))),
    ("rr", KeyEntry::BuiltinInstr(Mnemonic(RR))),
    ("rra", KeyEntry::BuiltinInstr(Mnemonic(RRA))),
    ("rrc", KeyEntry::BuiltinInstr(Mnemonic(RRC))),
    ("rrca", KeyEntry::BuiltinInstr(Mnemonic(RRCA))),
    ("rst", KeyEntry::BuiltinInstr(Mnemonic(RST))),
    ("sbc", KeyEntry::BuiltinInstr(Mnemonic(SBC))),
    ("section", KeyEntry::BuiltinInstr(Directive(Section))),
    ("set", KeyEntry::BuiltinInstr(Mnemonic(SET))),
    ("sla", KeyEntry::BuiltinInstr(Mnemonic(SLA))),
    ("sra", KeyEntry::BuiltinInstr(Mnemonic(SRA))),
    ("srl", KeyEntry::BuiltinInstr(Mnemonic(SRL))),
    ("stop", KeyEntry::BuiltinInstr(Mnemonic(STOP))),
    ("sub", KeyEntry::BuiltinInstr(Mnemonic(SUB))),
    ("swap", KeyEntry::BuiltinInstr(Mnemonic(SWAP))),
    ("xor", KeyEntry::BuiltinInstr(Mnemonic(XOR))),
];

impl<S: Session> TokenLineActions<S::Ident, Literal<S::StringRef>, S::Span>
    for TokenLineSemantics<S>
where
    S::Ident: AsRef<str>,
{
    type ContextFinalizer = TokenContextFinalizationSemantics<S>;

    fn act_on_token(&mut self, token: SemanticToken<S::Ident, S::StringRef>, span: S::Span) {
        match &mut self.line {
            TokenContext::MacroDef(state) => {
                state.tokens.0.push(token);
                state.tokens.1.push(span);
            }
        }
    }

    fn act_on_ident(
        mut self,
        ident: S::Ident,
        span: S::Span,
    ) -> TokenLineRule<Self, Self::ContextFinalizer> {
        match &mut self.line {
            TokenContext::MacroDef(state) => {
                if ident.as_ref().eq_ignore_ascii_case("ENDM") {
                    state.tokens.0.push(Sigil::Eos.into());
                    state.tokens.1.push(span);
                    TokenLineRule::LineEnd(TokenContextFinalizationSemantics { parent: self })
                } else {
                    state.tokens.0.push(Token::Ident(ident));
                    state.tokens.1.push(span);
                    TokenLineRule::TokenSeq(self)
                }
            }
        }
    }
}

impl<S: Session> LineFinalizer<S::Span> for TokenLineSemantics<S> {
    type Next = TokenStreamSemantics<S>;

    fn did_parse_line(mut self, span: S::Span) -> Self::Next {
        match &mut self.line {
            TokenContext::MacroDef(state) => {
                state.tokens.0.push(Sigil::Eol.into());
                state.tokens.1.push(span);
            }
        }
        self.into()
    }
}

pub(super) struct TokenContextFinalizationSemantics<S: Session> {
    parent: TokenLineSemantics<S>,
}

delegate_diagnostics! {
    {S: Session}, TokenContextFinalizationSemantics<S>, {parent}, S, S::Span
}

impl<S: Session> LineFinalizer<S::Span> for TokenContextFinalizationSemantics<S> {
    type Next = TokenStreamSemantics<S>;

    fn did_parse_line(mut self, _: S::Span) -> Self::Next {
        match self.parent.line {
            TokenContext::MacroDef(state) => {
                if let Some((name, params)) = state.label {
                    let tokens = state.tokens;
                    let id = self.parent.session.define_macro(name.1, params, tokens);
                    self.parent.session.insert(name.0, ResolvedIdent::Macro(id));
                }
            }
        }
        TokenStreamSemantics::new(self.parent.session)
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
            let mut command = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("LD".into(), ())
                .into_builtin_instr();
            let mut arg1 = command.will_parse_arg();
            arg1.act_on_atom((ExprAtom::Ident("B".into()), ()));
            command = arg1.did_parse_arg();
            let mut arg2 = command.will_parse_arg();
            arg2.act_on_atom((ExprAtom::Ident("HL".into()), ()));
            arg2.act_on_operator((Operator::Unary(UnaryOperator::Parentheses), ()));
            arg2.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
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
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("RST".into(), ())
                .into_builtin_instr();
            let mut expr = command.will_parse_arg();
            expr.act_on_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.act_on_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.act_on_operator((Operator::Binary(op), ()));
            expr.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
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
            let command = actions
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("RST".into(), ())
                .into_builtin_instr();
            let mut expr = command.will_parse_arg();
            expr.act_on_atom((ExprAtom::Ident(ident.clone()), ()));
            expr.act_on_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.act_on_operator((Operator::FnCall(1), ()));
            expr.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
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
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("DW".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            arg.act_on_atom((ExprAtom::Ident(label.into()), ()));
            arg.did_parse_arg()
                .did_parse_instr()
                .did_parse_line(())
                .act_on_eos(())
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
                NameTableEvent::Insert(label.into(), ResolvedIdent::Backend(0)).into(),
                BackendEvent::DefineSymbol((0, ()), LocationCounter.into()).into()
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
            actions.act_on_atom((ExprAtom::LocationCounter, ()));
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
            for param in params.borrow().iter().map(|&t| (t.into(), ())) {
                params_actions.act_on_param(param)
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
                SessionEvent::DefineMacro(
                    params.borrow().iter().cloned().map(Into::into).collect(),
                    body
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
                .will_parse_line()
                .into_instr_line()
                .will_parse_instr("NOP".into(), ())
                .into_builtin_instr()
                .will_parse_arg();
            arg.act_on_atom((ExprAtom::Ident("A".into()), ()));
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
            vec![(name.into(), ResolvedIdent::Backend(42))],
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

    pub(super) type MockSession<S> = crate::analysis::session::MockSession<
        SerialIdAllocator,
        BasicNameTable<usize, MockMacroId>,
        TestOperation<S>,
        S,
    >;

    pub(super) fn collect_semantic_actions<F, S>(f: F) -> Vec<TestOperation<S>>
    where
        F: FnOnce(TestTokenStreamSemantics<S>) -> Done<MockSession<S>>,
        S: Clone + Debug + Merge,
    {
        with_log(|log| {
            f(TokenStreamSemantics::new(MockSession::with_log(log)));
        })
    }

    pub(super) fn log_with_predefined_names<I, F, S>(entries: I, f: F) -> Vec<TestOperation<S>>
    where
        I: IntoIterator<Item = (String, ResolvedIdent<usize, MockMacroId>)>,
        F: FnOnce(TestTokenStreamSemantics<S>) -> Done<MockSession<S>>,
        S: Clone + Debug + Merge,
    {
        with_log(|log| {
            f(TokenStreamSemantics::new(
                MockSession::with_predefined_names(log, entries),
            ));
        })
    }

    pub(super) type TestTokenStreamSemantics<S> = TokenStreamSemantics<MockSession<S>>;
}
