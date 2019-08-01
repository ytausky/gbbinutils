use self::builtin_instr::{BuiltinInstr, Directive, *};
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

mod builtin_instr;
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
        let actions =
            super::syntax::parse_token_seq(tokens.into_iter(), SemanticActions::new(session));
        actions.session.into()
    }
}

pub(super) struct SemanticActions<S: Session> {
    session: S,
    mode: Option<LineRule<(), TokenFrame<S>>>,
    label: Option<Label<S::Ident, S::Span>>,
}

enum TokenFrame<S: Session> {
    MacroDef(MacroDefState<S>),
}

struct MacroDefState<S: Session> {
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

impl<S: Session> SemanticActions<S> {
    pub fn new(session: S) -> SemanticActions<S> {
        SemanticActions {
            session,
            mode: Some(LineRule::InstrLine(())),
            label: None,
        }
    }

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

    fn define_label_if_present(mut self) -> Self {
        if let Some(((label, span), _params)) = self.label.take() {
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

delegate_diagnostics! {
    {S: Session}, SemanticActions<S>, {session}, S, S::Span
}

impl<S: Session> TokenStreamContext<S::Ident, Literal<S::StringRef>, S::Span> for SemanticActions<S>
where
    S::Ident: AsRef<str>,
{
    type InstrLineContext = Self;
    type TokenLineContext = TokenLineActions<S>;
    type TokenLineEndContext = TokenFrameEndActions<S>;
    type FinalContext = Done;

    fn will_parse_line(mut self) -> LineRule<Self::InstrLineContext, Self::TokenLineContext> {
        match self.mode.take().unwrap() {
            LineRule::InstrLine(()) => LineRule::InstrLine(self),
            LineRule::TokenLine(frame) => LineRule::TokenLine(TokenLineActions {
                parent: self,
                frame,
            }),
        }
    }

    fn act_on_eos(mut self, span: S::Span) -> Self::FinalContext {
        self = self.define_label_if_present();
        if let Some(LineRule::TokenLine(TokenFrame::MacroDef(_))) = &self.mode {
            self.emit_diag(Message::UnexpectedEof.at(span))
        }
        Done
    }
}

pub(super) struct Done;

impl<S: Session> InstrLineContext<S::Ident, Literal<S::StringRef>, S::Span> for SemanticActions<S>
where
    S::Ident: AsRef<str>,
{
    type LabelContext = LabelActions<S>;
    type InstrContext = Self;

    fn will_parse_label(mut self, label: (S::Ident, S::Span)) -> Self::LabelContext {
        self = self.define_label_if_present();
        LabelActions::new(self, label)
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

impl<S: Session> LabelContext<S::Ident, S::Span> for LabelActions<S> {
    type ParentContext = SemanticActions<S>;

    fn act_on_param(&mut self, (ident, span): (S::Ident, S::Span)) {
        self.params.0.push(ident);
        self.params.1.push(span)
    }

    fn did_parse_label(mut self) -> Self::ParentContext {
        self.parent.label = Some((self.label, self.params));
        self.parent
    }
}

type Label<I, S> = ((I, S), Params<I, S>);

impl<S: Session> InstrContext<S::Ident, Literal<S::StringRef>, S::Span> for SemanticActions<S>
where
    S::Ident: AsRef<str>,
{
    type BuiltinInstrContext = BuiltinInstrActions<S>;
    type MacroCallContext = MacroCallActions<S>;
    type ErrorContext = Self;
    type LineEndContext = Self;

    fn will_parse_instr(
        mut self,
        ident: S::Ident,
        span: S::Span,
    ) -> InstrRule<Self::BuiltinInstrContext, Self::MacroCallContext, Self> {
        match KEYS
            .iter()
            .find(|(spelling, _)| spelling.eq_ignore_ascii_case(ident.as_ref()))
            .map(|(_, entry)| entry)
        {
            Some(KeyEntry::BuiltinInstr(command)) => {
                InstrRule::BuiltinInstr(BuiltinInstrActions::new(self, (command.clone(), span)))
            }
            None => match self.session.get(&ident) {
                Some(ResolvedIdent::Macro(id)) => {
                    self = self.define_label_if_present();
                    InstrRule::MacroInstr(MacroCallActions::new(self, (id, span)))
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

impl<S: Session> InstrEndContext<S::Span> for SemanticActions<S>
where
    S::Ident: AsRef<str>,
{
    type ParentContext = Self;

    fn did_parse_instr(self) -> Self::ParentContext {
        self
    }
}

impl<S: Session> LineEndContext<S::Span> for SemanticActions<S>
where
    S::Ident: AsRef<str>,
{
    type ParentContext = Self;

    fn did_parse_line(mut self, _: S::Span) -> Self::ParentContext {
        if self.mode.is_none() {
            self.mode = Some(LineRule::InstrLine(()))
        }
        self
    }
}

#[derive(Clone)]
enum KeyEntry {
    BuiltinInstr(BuiltinInstr),
}

const KEYS: &[(&str, KeyEntry)] = &[
    ("adc", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(ADC))),
    ("add", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(ADD))),
    ("and", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(AND))),
    ("bit", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(BIT))),
    ("call", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(CALL))),
    ("cp", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(CP))),
    ("cpl", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(CPL))),
    ("daa", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(DAA))),
    (
        "db",
        KeyEntry::BuiltinInstr(BuiltinInstr::Directive(Directive::Db)),
    ),
    ("dec", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(DEC))),
    ("di", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(DI))),
    (
        "ds",
        KeyEntry::BuiltinInstr(BuiltinInstr::Directive(Directive::Ds)),
    ),
    (
        "dw",
        KeyEntry::BuiltinInstr(BuiltinInstr::Directive(Directive::Dw)),
    ),
    ("ei", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(EI))),
    (
        "equ",
        KeyEntry::BuiltinInstr(BuiltinInstr::Directive(Directive::Equ)),
    ),
    ("halt", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(HALT))),
    ("inc", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(INC))),
    (
        "include",
        KeyEntry::BuiltinInstr(BuiltinInstr::Directive(Directive::Include)),
    ),
    ("jp", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(JP))),
    ("jr", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(JR))),
    ("ld", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(LD))),
    ("ldhl", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(LDHL))),
    (
        "macro",
        KeyEntry::BuiltinInstr(BuiltinInstr::Directive(Directive::Macro)),
    ),
    ("nop", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(NOP))),
    ("or", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(OR))),
    (
        "org",
        KeyEntry::BuiltinInstr(BuiltinInstr::Directive(Directive::Org)),
    ),
    ("pop", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(POP))),
    ("push", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(PUSH))),
    ("res", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(RES))),
    ("ret", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(RET))),
    ("reti", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(RETI))),
    ("rl", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(RL))),
    ("rla", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(RLA))),
    ("rlc", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(RLC))),
    ("rlca", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(RLCA))),
    ("rr", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(RR))),
    ("rra", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(RRA))),
    ("rrc", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(RRC))),
    ("rrca", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(RRCA))),
    ("rst", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(RST))),
    ("sbc", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(SBC))),
    (
        "section",
        KeyEntry::BuiltinInstr(BuiltinInstr::Directive(Directive::Section)),
    ),
    ("set", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(SET))),
    ("sla", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(SLA))),
    ("sra", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(SRA))),
    ("srl", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(SRL))),
    ("stop", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(STOP))),
    ("sub", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(SUB))),
    ("swap", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(SWAP))),
    ("xor", KeyEntry::BuiltinInstr(BuiltinInstr::Mnemonic(XOR))),
];

pub(super) struct TokenLineActions<S: Session> {
    parent: SemanticActions<S>,
    frame: TokenFrame<S>,
}

delegate_diagnostics! {
    {S: Session}, TokenLineActions<S>, {parent}, S, S::Span
}

impl<S: Session> TokenLineContext<S::Ident, Literal<S::StringRef>, S::Span> for TokenLineActions<S>
where
    S::Ident: AsRef<str>,
{
    type SemanticContextTerminationContext = TokenFrameEndActions<S>;

    fn act_on_token(&mut self, token: SemanticToken<S::Ident, S::StringRef>, span: S::Span) {
        match &mut self.frame {
            TokenFrame::MacroDef(state) => {
                state.tokens.0.push(token);
                state.tokens.1.push(span);
            }
        }
    }

    fn act_on_ident(
        mut self,
        ident: S::Ident,
        span: S::Span,
    ) -> TokenLineRule<Self, Self::SemanticContextTerminationContext> {
        match &mut self.frame {
            TokenFrame::MacroDef(state) => {
                if ident.as_ref().eq_ignore_ascii_case("ENDM") {
                    state.tokens.0.push(SimpleToken::Eof.into());
                    state.tokens.1.push(span);
                    TokenLineRule::LineEnd(TokenFrameEndActions {
                        parent: self.parent,
                        frame: self.frame,
                    })
                } else {
                    state.tokens.0.push(Token::Ident(ident));
                    state.tokens.1.push(span);
                    TokenLineRule::TokenSeq(self)
                }
            }
        }
    }
}

impl<S: Session> LineEndContext<S::Span> for TokenLineActions<S> {
    type ParentContext = SemanticActions<S>;

    fn did_parse_line(mut self, span: S::Span) -> Self::ParentContext {
        match &mut self.frame {
            TokenFrame::MacroDef(state) => {
                state.tokens.0.push(SimpleToken::Eol.into());
                state.tokens.1.push(span);
                self.parent.mode = Some(LineRule::TokenLine(self.frame));
            }
        }
        self.parent
    }
}

pub(super) struct TokenFrameEndActions<S: Session> {
    parent: SemanticActions<S>,
    frame: TokenFrame<S>,
}

delegate_diagnostics! {
    {S: Session}, TokenFrameEndActions<S>, {parent}, S, S::Span
}

impl<S: Session> LineEndContext<S::Span> for TokenFrameEndActions<S> {
    type ParentContext = SemanticActions<S>;

    fn did_parse_line(mut self, _: S::Span) -> Self::ParentContext {
        match self.frame {
            TokenFrame::MacroDef(state) => {
                if let Some((name, params)) = state.label {
                    let tokens = state.tokens;
                    let id = self.parent.session.define_macro(name.1, params, tokens);
                    self.parent.session.insert(name.0, ResolvedIdent::Macro(id));
                }
                self.parent.mode = Some(LineRule::InstrLine(()))
            }
        }
        self.parent
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
            let mut arg1 = command.add_argument();
            arg1.push_atom((ExprAtom::Ident("B".into()), ()));
            command = arg1.exit();
            let mut arg2 = command.add_argument();
            arg2.push_atom((ExprAtom::Ident("HL".into()), ()));
            arg2.apply_operator((Operator::Unary(UnaryOperator::Parentheses), ()));
            arg2.exit()
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
            let mut expr = command.add_argument();
            expr.push_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.push_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.apply_operator((Operator::Binary(op), ()));
            expr.exit()
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
            let mut expr = command.add_argument();
            expr.push_atom((ExprAtom::Ident(ident.clone()), ()));
            expr.push_atom((ExprAtom::Literal(Literal::Number(1)), ()));
            expr.apply_operator((Operator::FnCall(1), ()));
            expr.exit()
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
                .add_argument();
            arg.push_atom((ExprAtom::Ident(label.into()), ()));
            arg.exit()
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
                .add_argument();
            actions.push_atom((ExprAtom::LocationCounter, ()));
            actions
                .exit()
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
        body.push(SimpleToken::Eof.into());
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
                .add_argument();
            arg.push_atom((ExprAtom::Ident("A".into()), ()));
            arg.exit()
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
                .add_argument();
            expr.emit_diag(diagnostic.clone());
            expr.exit()
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
    fn diagnose_eof_in_macro_body() {
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
        F: FnOnce(TestSemanticActions<S>) -> Done,
        S: Clone + Debug + Merge,
    {
        with_log(|log| {
            f(SemanticActions::new(MockSession::with_log(log)));
        })
    }

    pub(super) fn log_with_predefined_names<I, F, S>(entries: I, f: F) -> Vec<TestOperation<S>>
    where
        I: IntoIterator<Item = (String, ResolvedIdent<usize, MockMacroId>)>,
        F: FnOnce(TestSemanticActions<S>) -> Done,
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
