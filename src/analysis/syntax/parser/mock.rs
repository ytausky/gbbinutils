use super::Sigil::*;
use super::Token::*;
use super::{ExprAtom, Operator, Token, UnaryOperator};

use crate::analysis::syntax::*;
use crate::diag::span::{MergeSpans, StripSpan};
use crate::diag::{CompactDiag, EmitDiag, Merge, Message};
use crate::model::BinOp;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::iter;

pub(super) type MockSpan = crate::diag::MockSpan<TokenRef>;

pub(super) fn with_spans<'a>(
    tokens: impl IntoIterator<Item = &'a (MockToken, TokenRef)>,
) -> impl Iterator<Item = (Result<MockToken, ()>, MockSpan)> {
    tokens.into_iter().cloned().map(|(t, r)| (Ok(t), r.into()))
}

macro_rules! impl_diag_traits {
    ($($t:ty),* $(,)?) => {
        $(
            impl MergeSpans<MockSpan> for $t {
                fn merge_spans(&mut self, left: &MockSpan, right: &MockSpan) -> MockSpan {
                    MockSpan::merge(left.clone(), right.clone())
                }
            }

            impl StripSpan<MockSpan> for $t {
                type Stripped = MockSpan;

                fn strip_span(&mut self, span: &MockSpan) -> Self::Stripped {
                    span.clone()
                }
            }
        )*
    };
}

impl_diag_traits! {
    InstrLineActionCollector,
    LabelActionCollector,
    InstrActionCollector,
    BuiltinInstrActionCollector,
    ExprActionCollector<BuiltinInstrActionCollector>,
    ExprActionCollector<()>,
    TokenLineActionCollector,
    MacroInstrActionCollector,
    MacroArgActionCollector,
    ErrorActionCollector,
}

pub(super) struct TokenStreamActionCollector {
    pub actions: Vec<TokenStreamAction<MockSpan>>,
    mode: LineRule<(), ()>,
}

impl TokenStreamActionCollector {
    pub fn new() -> TokenStreamActionCollector {
        TokenStreamActionCollector {
            actions: Vec::new(),
            mode: LineRule::InstrLine(()),
        }
    }
}

impl TokenStreamActions<MockIdent, MockLiteral, MockSpan> for TokenStreamActionCollector {
    type InstrLineActions = InstrLineActionCollector;
    type TokenLineActions = TokenLineActionCollector;
    type TokenLineFinalizer = TokenLineActionCollector;
    type Next = Self;

    fn will_parse_line(self) -> LineRule<Self::InstrLineActions, Self::TokenLineActions> {
        match self.mode {
            LineRule::InstrLine(()) => LineRule::InstrLine(InstrLineActionCollector::new(self)),
            LineRule::TokenLine(()) => LineRule::TokenLine(TokenLineActionCollector::new(self)),
        }
    }

    fn act_on_eos(mut self, span: MockSpan) -> Self::Next {
        self.actions.push(TokenStreamAction::Eos(span));
        self
    }
}

pub(super) struct InstrLineActionCollector {
    parent: TokenStreamActionCollector,
    actions: Vec<InstrLineAction<MockSpan>>,
}

impl InstrLineActionCollector {
    fn new(parent: TokenStreamActionCollector) -> Self {
        Self {
            parent,
            actions: Vec::new(),
        }
    }
}

impl EmitDiag<MockSpan, MockSpan> for InstrLineActionCollector {
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<MockSpan>>) {
        self.actions.push(InstrLineAction::EmitDiag(diag.into()))
    }
}

impl InstrLineActions<MockIdent, MockLiteral, MockSpan> for InstrLineActionCollector {
    type InstrActions = InstrActionCollector;
    type LabelActions = LabelActionCollector;

    fn will_parse_label(self, label: (MockIdent, MockSpan)) -> Self::LabelActions {
        LabelActionCollector {
            label,
            actions: Vec::new(),
            parent: self,
        }
    }
}

impl InstrActions<MockIdent, MockLiteral, MockSpan> for InstrLineActionCollector {
    type BuiltinInstrActions = BuiltinInstrActionCollector;
    type MacroInstrActions = MacroInstrActionCollector;
    type ErrorActions = ErrorActionCollector;
    type LineFinalizer = InstrActionCollector;

    fn will_parse_instr(
        self,
        ident: MockIdent,
        span: MockSpan,
    ) -> InstrRule<Self::BuiltinInstrActions, Self::MacroInstrActions, Self::ErrorActions> {
        InstrActionCollector {
            actions: Vec::new(),
            parent: self,
        }
        .will_parse_instr(ident, span)
    }
}

impl LineFinalizer<MockSpan> for InstrLineActionCollector {
    type Next = TokenStreamActionCollector;

    fn did_parse_line(mut self, span: MockSpan) -> Self::Next {
        self.parent
            .actions
            .push(TokenStreamAction::InstrLine(self.actions, span));
        self.parent
    }
}

pub(super) struct LabelActionCollector {
    label: (MockIdent, MockSpan),
    actions: Vec<ParamsAction<MockSpan>>,
    parent: InstrLineActionCollector,
}

impl EmitDiag<MockSpan, MockSpan> for LabelActionCollector {
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<MockSpan>>) {
        self.actions.push(ParamsAction::EmitDiag(diag.into()))
    }
}

impl LabelActions<MockIdent, MockSpan> for LabelActionCollector {
    type Next = InstrActionCollector;

    fn act_on_param(&mut self, param: MockIdent, span: MockSpan) {
        self.actions.push(ParamsAction::AddParameter(param, span))
    }

    fn did_parse_label(mut self) -> Self::Next {
        self.parent
            .actions
            .push(InstrLineAction::Label((self.label, self.actions)));
        InstrActionCollector {
            parent: self.parent,
            actions: Vec::new(),
        }
    }
}

pub(super) struct InstrActionCollector {
    actions: Vec<InstrAction<MockSpan>>,
    parent: InstrLineActionCollector,
}

impl EmitDiag<MockSpan, MockSpan> for InstrActionCollector {
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<MockSpan>>) {
        self.actions.push(InstrAction::EmitDiag(diag.into()))
    }
}

#[derive(Debug, PartialEq)]
pub struct MacroId(pub TokenRef);

impl InstrActions<MockIdent, MockLiteral, MockSpan> for InstrActionCollector {
    type BuiltinInstrActions = BuiltinInstrActionCollector;
    type MacroInstrActions = MacroInstrActionCollector;
    type ErrorActions = ErrorActionCollector;
    type LineFinalizer = Self;

    fn will_parse_instr(
        self,
        ident: MockIdent,
        span: MockSpan,
    ) -> InstrRule<Self::BuiltinInstrActions, Self::MacroInstrActions, Self::ErrorActions> {
        match ident.0 {
            IdentKind::BuiltinInstr | IdentKind::MacroKeyword | IdentKind::Endm => {
                InstrRule::BuiltinInstr(BuiltinInstrActionCollector {
                    builtin_instr: (ident, span),
                    actions: Vec::new(),
                    parent: self,
                })
            }
            IdentKind::MacroName => InstrRule::MacroInstr(MacroInstrActionCollector {
                name: (ident, span),
                actions: Vec::new(),
                parent: self,
            }),
            IdentKind::Other => InstrRule::Error(ErrorActionCollector {
                parent: self,
                actions: Vec::new(),
            }),
        }
    }
}

impl LineFinalizer<MockSpan> for InstrActionCollector {
    type Next = TokenStreamActionCollector;

    fn did_parse_line(mut self, span: MockSpan) -> Self::Next {
        if !self.actions.is_empty() {
            self.parent
                .actions
                .push(InstrLineAction::Instr(self.actions));
        }
        self.parent.did_parse_line(span)
    }
}

pub(super) struct ErrorActionCollector {
    parent: InstrActionCollector,
    actions: Vec<ErrorAction<MockSpan>>,
}

impl EmitDiag<MockSpan, MockSpan> for ErrorActionCollector {
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<MockSpan>>) {
        self.actions.push(ErrorAction::EmitDiag(diag.into()))
    }
}

impl InstrFinalizer<MockSpan> for ErrorActionCollector {
    type Next = InstrActionCollector;

    fn did_parse_instr(mut self) -> Self::Next {
        self.parent.actions.push(InstrAction::Error(self.actions));
        self.parent
    }
}

pub(super) struct BuiltinInstrActionCollector {
    builtin_instr: (MockIdent, MockSpan),
    actions: Vec<BuiltinInstrAction<MockSpan>>,
    parent: InstrActionCollector,
}

impl EmitDiag<MockSpan, MockSpan> for BuiltinInstrActionCollector {
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<MockSpan>>) {
        self.actions.push(BuiltinInstrAction::EmitDiag(diag.into()))
    }
}

impl BuiltinInstrActions<MockIdent, MockLiteral, MockSpan> for BuiltinInstrActionCollector {
    type ArgActions = ExprActionCollector<Self>;

    fn will_parse_arg(self) -> Self::ArgActions {
        ExprActionCollector::new(self)
    }
}

impl InstrFinalizer<MockSpan> for BuiltinInstrActionCollector {
    type Next = InstrActionCollector;

    fn did_parse_instr(mut self) -> Self::Next {
        if (self.builtin_instr.0).0 == IdentKind::MacroKeyword {
            self.parent.parent.parent.mode = LineRule::TokenLine(())
        }
        self.parent.actions.push(InstrAction::BuiltinInstr {
            builtin_instr: self.builtin_instr,
            actions: self.actions,
        });
        self.parent
    }
}

pub(super) struct ExprActionCollector<P> {
    actions: Vec<ExprAction<MockSpan>>,
    parent: P,
}

impl<P> ExprActionCollector<P> {
    pub fn new(parent: P) -> Self {
        Self {
            actions: Vec::new(),
            parent,
        }
    }
}

impl<P> EmitDiag<MockSpan, MockSpan> for ExprActionCollector<P> {
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<MockSpan>>) {
        self.actions.push(ExprAction::EmitDiag(diag.into()))
    }
}

impl ArgFinalizer for ExprActionCollector<BuiltinInstrActionCollector> {
    type Next = BuiltinInstrActionCollector;

    fn did_parse_arg(mut self) -> Self::Next {
        self.parent.actions.push(BuiltinInstrAction::AddArgument {
            actions: self.actions,
        });
        self.parent
    }
}

impl ArgFinalizer for ExprActionCollector<()> {
    type Next = Vec<ExprAction<MockSpan>>;

    fn did_parse_arg(self) -> Self::Next {
        self.actions
    }
}

impl<P> ArgActions<MockIdent, MockLiteral, MockSpan> for ExprActionCollector<P>
where
    Self: Diagnostics<MockSpan>,
{
    fn act_on_atom(&mut self, atom: ExprAtom<MockIdent, MockLiteral>, span: MockSpan) {
        self.actions.push(ExprAction::PushAtom(atom, span))
    }

    fn act_on_operator(&mut self, operator: Operator, span: MockSpan) {
        self.actions.push(ExprAction::ApplyOperator(operator, span))
    }
}

pub(super) struct TokenLineActionCollector {
    actions: Vec<TokenLineAction<MockSpan>>,
    parent: TokenStreamActionCollector,
}

impl TokenLineActionCollector {
    fn new(parent: TokenStreamActionCollector) -> Self {
        Self {
            actions: Vec::new(),
            parent,
        }
    }
}

impl EmitDiag<MockSpan, MockSpan> for TokenLineActionCollector {
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<MockSpan>>) {
        self.actions.push(TokenLineAction::EmitDiag(diag.into()))
    }
}

impl TokenLineActions<MockIdent, MockLiteral, MockSpan> for TokenLineActionCollector {
    type ContextFinalizer = Self;

    fn act_on_token(&mut self, token: MockToken, span: MockSpan) {
        self.actions.push(TokenLineAction::Token((token, span)))
    }

    fn act_on_ident(
        mut self,
        ident: MockIdent,
        span: MockSpan,
    ) -> TokenLineRule<Self, Self::ContextFinalizer> {
        let kind = ident.0;
        self.actions.push(TokenLineAction::Ident((ident, span)));
        match kind {
            IdentKind::Endm => {
                self.parent.mode = LineRule::InstrLine(());
                TokenLineRule::LineEnd(self)
            }
            _ => TokenLineRule::TokenSeq(self),
        }
    }
}

impl LineFinalizer<MockSpan> for TokenLineActionCollector {
    type Next = TokenStreamActionCollector;

    fn did_parse_line(mut self, span: MockSpan) -> Self::Next {
        self.parent
            .actions
            .push(TokenStreamAction::TokenLine(self.actions, span));
        self.parent
    }
}

pub(super) struct MacroInstrActionCollector {
    name: (MockIdent, MockSpan),
    actions: Vec<MacroInstrAction<MockSpan>>,
    parent: InstrActionCollector,
}

impl EmitDiag<MockSpan, MockSpan> for MacroInstrActionCollector {
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<MockSpan>>) {
        self.actions.push(MacroInstrAction::EmitDiag(diag.into()))
    }
}

impl MacroInstrActions<MockSpan> for MacroInstrActionCollector {
    type Token = MockToken;
    type MacroArgActions = MacroArgActionCollector;

    fn will_parse_macro_arg(self) -> MacroArgActionCollector {
        MacroArgActionCollector {
            actions: Vec::new(),
            parent: self,
        }
    }
}

impl InstrFinalizer<MockSpan> for MacroInstrActionCollector {
    type Next = InstrActionCollector;

    fn did_parse_instr(mut self) -> Self::Next {
        self.parent.actions.push(InstrAction::MacroInstr {
            name: self.name,
            actions: self.actions,
        });
        self.parent
    }
}

pub(super) struct MacroArgActionCollector {
    actions: Vec<TokenSeqAction<MockSpan>>,
    parent: MacroInstrActionCollector,
}

impl EmitDiag<MockSpan, MockSpan> for MacroArgActionCollector {
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<MockSpan>>) {
        self.actions.push(TokenSeqAction::EmitDiag(diag.into()))
    }
}

impl MacroArgActions<MockSpan> for MacroArgActionCollector {
    type Token = MockToken;
    type Next = MacroInstrActionCollector;

    fn act_on_token(&mut self, token: (Self::Token, MockSpan)) {
        self.actions.push(TokenSeqAction::PushToken(token))
    }

    fn did_parse_macro_arg(mut self) -> MacroInstrActionCollector {
        self.parent
            .actions
            .push(MacroInstrAction::MacroArg(self.actions));
        self.parent
    }
}

pub(super) fn expr() -> SymExpr {
    SymExpr(Vec::new())
}

impl SymExpr {
    pub fn ident(self, token: impl Into<TokenRef>) -> Self {
        self.push(token, |t| ExprAtom::Ident(MockIdent(IdentKind::Other, t)))
    }

    pub fn literal(self, token: impl Into<TokenRef>) -> Self {
        self.push(token, |t| ExprAtom::Literal(MockLiteral(t)))
    }

    pub fn location_counter(self, token: impl Into<TokenRef>) -> Self {
        self.push(token, |_| ExprAtom::LocationCounter)
    }

    fn push(
        mut self,
        token: impl Into<TokenRef>,
        atom_ctor: impl Fn(TokenRef) -> ExprAtom<MockIdent, MockLiteral>,
    ) -> Self {
        let token_ref = token.into();
        self.0.push(ExprAction::PushAtom(
            atom_ctor(token_ref.clone()),
            token_ref.into(),
        ));
        self
    }

    pub fn divide(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator(
            Operator::Binary(BinOp::Division),
            token.into().into(),
        ));
        self
    }

    pub fn multiply(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator(
            Operator::Binary(BinOp::Multiplication),
            token.into().into(),
        ));
        self
    }

    pub fn parentheses(mut self, left: impl Into<TokenRef>, right: impl Into<TokenRef>) -> Self {
        let span = MockSpan::merge(left.into(), right.into());
        self.0.push(ExprAction::ApplyOperator(
            Operator::Unary(UnaryOperator::Parentheses),
            span,
        ));
        self
    }

    pub fn plus(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator(
            Operator::Binary(BinOp::Plus),
            token.into().into(),
        ));
        self
    }

    pub fn minus(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator(
            Operator::Binary(BinOp::Minus),
            token.into().into(),
        ));
        self
    }

    pub fn bitwise_or(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator(
            Operator::Binary(BinOp::BitwiseOr),
            token.into().into(),
        ));
        self
    }

    pub fn fn_call(mut self, args: usize, span: impl Into<MockSpan>) -> Self {
        self.0.push(ExprAction::ApplyOperator(
            Operator::FnCall(args),
            span.into(),
        ));
        self
    }

    pub fn error(mut self, span: impl Into<TokenRef>) -> Self {
        self.0
            .push(ExprAction::PushAtom(ExprAtom::Error, span.into().into()));
        self
    }

    pub fn diag(mut self, message: Message<MockSpan>, highlight: impl Into<MockSpan>) -> Self {
        self.0
            .push(ExprAction::EmitDiag(message.at(highlight.into()).into()));
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymCommand(pub TokenRef);

#[derive(Clone, Debug, PartialEq)]
pub struct MockIdent(pub IdentKind, pub TokenRef);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IdentKind {
    BuiltinInstr,
    Endm,
    MacroKeyword,
    MacroName,
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MockLiteral(pub TokenRef);

pub type MockToken = Token<MockIdent, MockLiteral>;

fn mk_mock_token(id: impl Into<TokenRef>, token: Token<IdentKind, ()>) -> (MockToken, TokenRef) {
    let token_ref = id.into();
    (
        match token {
            Ident(kind) => Ident(MockIdent(kind, token_ref.clone())),
            Label(kind) => Label(MockIdent(kind, token_ref.clone())),
            Literal(()) => Literal(MockLiteral(token_ref.clone())),
            Sigil(sigil) => Sigil(sigil),
        },
        token_ref,
    )
}

pub(super) struct InputTokens {
    pub tokens: Vec<(MockToken, TokenRef)>,
    pub names: HashMap<String, usize>,
}

impl InputTokens {
    pub fn insert_token(&mut self, id: impl Into<TokenRef>, token: Token<IdentKind, ()>) {
        self.tokens.push(mk_mock_token(id, token))
    }

    pub fn token_seq<T>(&self, tokens: impl Borrow<[T]>) -> Vec<TokenSeqAction<MockSpan>>
    where
        T: Clone + Into<TokenRef>,
    {
        tokens
            .borrow()
            .iter()
            .cloned()
            .map(Into::into)
            .map(|t| TokenSeqAction::PushToken(self.token(t.clone())))
            .collect()
    }

    pub fn token(&self, token_ref: impl Into<TokenRef>) -> (MockToken, MockSpan) {
        let token_ref = token_ref.into();
        let id = match &token_ref {
            TokenRef::Id(n) => *n,
            TokenRef::Name(name) => self.names[name],
        };
        (self.tokens[id].0.clone(), token_ref.into())
    }

    pub fn ident(&self, token_ref: impl Into<TokenRef>) -> TokenLineAction<MockSpan> {
        match self.token(token_ref.into()) {
            (Token::Ident(ident), span) => TokenLineAction::Ident((ident, span)),
            _ => panic!("expected identifier"),
        }
    }

    pub fn eos(&self) -> TokenStreamAction<MockSpan> {
        TokenStreamAction::Eos(self.tokens.last().unwrap().1.clone().into())
    }
}

macro_rules! add_token {
    ($input:expr, $token:expr) => {
        let id = $input.tokens.len();
        $input.insert_token(id, $token.into())
    };
    ($input:expr, $name:ident @ $token:expr) => {
        let id = stringify!($name);
        $input.names.insert(id.into(), $input.tokens.len());
        $input.insert_token(id, $token.into())
    };
}

macro_rules! input_tokens_impl {
    ($input:expr, ) => {};
    ($input:expr, $token:expr) => {
        add_token!($input, $token)
    };
    ($input:expr, $token:expr, $($tail:tt)*) => {
        add_token!($input, $token);
        input_tokens_impl![$input, $($tail)*]
    };
    ($input:expr, $name:ident @ $token:expr) => {
        add_token!($input, $name @ $token)
    };
    ($input:expr, $name:ident @ $token:expr, $($tail:tt)*) => {
        add_token!($input, $name @ $token);
        input_tokens_impl![$input, $($tail)*]
    }
}

macro_rules! input_tokens {
    ($($tokens:tt)*) => {{
        let mut input = InputTokens {
            tokens: Vec::new(),
            names: std::collections::HashMap::new(),
        };
        input_tokens_impl!(input, $($tokens)*);
        if input
            .tokens
            .last()
            .map(|(token, _)| *token != Eos.into())
            .unwrap_or(true)
        {
            let eos_id = input.tokens.len().into();
            input.tokens.push((Eos.into(), eos_id))
        }
        input
    }};
}

#[derive(Clone, Debug, PartialEq)]
pub enum TokenRef {
    Id(usize),
    Name(String),
}

impl From<usize> for TokenRef {
    fn from(id: usize) -> Self {
        TokenRef::Id(id)
    }
}

impl From<&'static str> for TokenRef {
    fn from(name: &'static str) -> Self {
        TokenRef::Name(name.to_string())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum TokenStreamAction<S> {
    InstrLine(Vec<InstrLineAction<S>>, S),
    TokenLine(Vec<TokenLineAction<S>>, S),
    Eos(S),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum InstrLineAction<S> {
    Label(Label<S>),
    Instr(Vec<InstrAction<S>>),
    EmitDiag(CompactDiag<S>),
}

pub(super) type Label<S> = ((MockIdent, S), Vec<ParamsAction<S>>);

#[derive(Clone, Debug, PartialEq)]
pub(super) enum InstrAction<S> {
    BuiltinInstr {
        builtin_instr: (MockIdent, S),
        actions: Vec<BuiltinInstrAction<S>>,
    },
    MacroInstr {
        name: (MockIdent, S),
        actions: Vec<MacroInstrAction<S>>,
    },
    Error(Vec<ErrorAction<S>>),
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum BuiltinInstrAction<S> {
    AddArgument { actions: Vec<ExprAction<S>> },
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ExprAction<S> {
    PushAtom(ExprAtom<MockIdent, MockLiteral>, S),
    ApplyOperator(Operator, S),
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ParamsAction<S> {
    AddParameter(MockIdent, S),
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum TokenLineAction<S> {
    Token((MockToken, S)),
    Ident((MockIdent, S)),
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum TokenSeqAction<S> {
    PushToken((MockToken, S)),
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ErrorAction<S> {
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum MacroInstrAction<S> {
    MacroArg(Vec<TokenSeqAction<S>>),
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone)]
pub(super) struct SymExpr(pub Vec<ExprAction<MockSpan>>);

pub(super) fn instr_line(
    actions: Vec<InstrLineAction<MockSpan>>,
    terminator: impl Into<TokenRef>,
) -> TokenStreamAction<MockSpan> {
    TokenStreamAction::InstrLine(actions, terminator.into().into())
}

pub(super) fn token_line(
    actions: Vec<TokenLineAction<MockSpan>>,
    terminator: impl Into<TokenRef>,
) -> TokenStreamAction<MockSpan> {
    TokenStreamAction::TokenLine(actions, terminator.into().into())
}

pub(super) fn labeled(
    label: impl Into<TokenRef>,
    params: impl Borrow<[TokenRef]>,
    actions: Option<Vec<InstrAction<MockSpan>>>,
    terminator: impl Into<TokenRef>,
) -> TokenStreamAction<MockSpan> {
    let label = label.into();
    let mut instr_line_actions = vec![InstrLineAction::Label((
        (MockIdent(IdentKind::Other, label.clone()), label.into()),
        convert_params(params),
    ))];
    if let Some(actions) = actions {
        instr_line_actions.push(InstrLineAction::Instr(actions))
    }
    TokenStreamAction::InstrLine(instr_line_actions, terminator.into().into())
}

pub(super) fn unlabeled(
    actions: Vec<InstrAction<MockSpan>>,
    terminator: impl Into<TokenRef>,
) -> TokenStreamAction<MockSpan> {
    TokenStreamAction::InstrLine(
        vec![InstrLineAction::Instr(actions)],
        terminator.into().into(),
    )
}

pub(super) fn builtin_instr(
    kind: IdentKind,
    id: impl Into<TokenRef>,
    args: impl Borrow<[SymExpr]>,
) -> Vec<InstrAction<MockSpan>> {
    let id = id.into();
    vec![InstrAction::BuiltinInstr {
        builtin_instr: (MockIdent(kind, id.clone()), id.into()),
        actions: args
            .borrow()
            .iter()
            .cloned()
            .map(|SymExpr(expr)| BuiltinInstrAction::AddArgument { actions: expr })
            .collect(),
    }]
}

pub(super) fn malformed_builtin_instr(
    id: impl Into<TokenRef>,
    args: impl Borrow<[SymExpr]>,
    diag: CompactDiag<MockSpan>,
) -> Vec<InstrAction<MockSpan>> {
    let id = id.into();
    vec![InstrAction::BuiltinInstr {
        builtin_instr: (MockIdent(IdentKind::BuiltinInstr, id.clone()), id.into()),
        actions: args
            .borrow()
            .iter()
            .cloned()
            .map(|SymExpr(expr)| BuiltinInstrAction::AddArgument { actions: expr })
            .chain(iter::once(BuiltinInstrAction::EmitDiag(diag)))
            .collect(),
    }]
}

pub(super) fn macro_instr(
    id: impl Into<TokenRef>,
    args: impl Borrow<[Vec<TokenSeqAction<MockSpan>>]>,
) -> Vec<InstrAction<MockSpan>> {
    let id = id.into();
    vec![InstrAction::MacroInstr {
        name: (MockIdent(IdentKind::MacroName, id.clone()), id.into()),
        actions: args
            .borrow()
            .iter()
            .cloned()
            .map(MacroInstrAction::MacroArg)
            .collect(),
    }]
}

fn convert_params(params: impl Borrow<[TokenRef]>) -> Vec<ParamsAction<MockSpan>> {
    params
        .borrow()
        .iter()
        .cloned()
        .map(|t| ParamsAction::AddParameter(MockIdent(IdentKind::Other, t.clone()), t.into()))
        .collect()
}

pub(super) fn arg_error(
    message: Message<MockSpan>,
    highlight: impl Into<TokenRef>,
) -> CompactDiag<MockSpan> {
    message.at(highlight.into().into()).into()
}

mod tests {
    use super::*;

    #[test]
    fn test_token_macro() {
        let tokens = input_tokens![
            my_tok @ Plus,
            Literal(()),
            next_one @ Star,
        ];
        assert_eq!(
            tokens.tokens,
            [
                (Plus.into(), "my_tok".into()),
                (Literal(MockLiteral(1.into())), 1.into()),
                (Star.into(), "next_one".into()),
                (Eos.into(), 3.into()),
            ]
        );
        assert_eq!(tokens.names.get("my_tok"), Some(&0));
        assert_eq!(tokens.names.get("next_one"), Some(&2))
    }
}
