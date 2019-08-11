use super::Sigil::*;
use super::Token::*;
use super::{ExprAtom, Operator, Token, UnaryOperator};

use crate::analysis::syntax::actions::*;
use crate::diag::span::{MergeSpans, StripSpan};
use crate::diag::{CompactDiag, Diagnostics, DiagnosticsEvent, EmitDiag, Merge, Message};
use crate::model::BinOp;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::iter;

pub(in crate::analysis) fn annotate(&MockIdent(kind, _): &MockIdent) -> IdentKind {
    kind
}

pub(in crate::analysis) struct ActionCollector<D, I> {
    data: D,
    annotate: fn(&I) -> IdentKind,
}

pub(in crate::analysis) struct CollectedData<A, T, P> {
    actions: Vec<A>,
    state: T,
    parent: P,
}

impl<A1, T1, P, I> ActionCollector<CollectedData<A1, T1, P>, I> {
    fn push_layer<A2, T2>(
        self,
        state: T2,
    ) -> ActionCollector<NestedCollectedData<A1, A2, T1, T2, P>, I> {
        ActionCollector {
            data: CollectedData {
                actions: Vec::new(),
                state,
                parent: self.data,
            },
            annotate: self.annotate,
        }
    }
}

type NestedCollectedData<A1, A2, T1, T2, P> = CollectedData<A2, T2, CollectedData<A1, T1, P>>;

macro_rules! pop_layer {
    ($collector:expr) => {
        ActionCollector {
            data: $collector.data.parent,
            annotate: $collector.annotate,
        }
    };
}

impl<D, I, S: Clone + Merge> MergeSpans<S> for ActionCollector<D, I> {
    fn merge_spans(&mut self, left: &S, right: &S) -> S {
        S::merge(left.clone(), right.clone())
    }
}

impl<D, I, S: Clone> StripSpan<S> for ActionCollector<D, I> {
    type Stripped = S;

    fn strip_span(&mut self, span: &S) -> Self::Stripped {
        span.clone()
    }
}

impl<A, T, P, I, S> EmitDiag<S, S> for ActionCollector<CollectedData<A, T, P>, I>
where
    A: From<DiagnosticsEvent<S>>,
    S: Clone,
{
    fn emit_diag(&mut self, diag: impl Into<CompactDiag<S, S>>) {
        self.data
            .actions
            .push(DiagnosticsEvent::EmitDiag(diag.into()).into())
    }
}

pub(super) type MockSpan = crate::diag::MockSpan<TokenRef>;

pub(super) fn with_spans<'a>(
    tokens: impl IntoIterator<Item = &'a (MockToken, TokenRef)>,
) -> impl Iterator<Item = (Result<MockToken, ()>, MockSpan)> {
    tokens.into_iter().cloned().map(|(t, r)| (Ok(t), r.into()))
}

pub(in crate::analysis) type TokenStreamActionCollector<P, I, L, S> =
    ActionCollector<CollectedTokenStreamData<P, I, L, S>, I>;

pub(in crate::analysis) type CollectedTokenStreamData<P, I, L, S> =
    CollectedData<TokenStreamAction<I, L, S>, LineRule<(), ()>, P>;

impl<P, I, L, S> TokenStreamActionCollector<P, I, L, S> {
    pub fn new(parent: P, annotate: fn(&I) -> IdentKind) -> Self {
        ActionCollector {
            data: CollectedData {
                actions: Vec::new(),
                state: LineRule::InstrLine(()),
                parent,
            },
            annotate,
        }
    }

    pub fn into_actions(self) -> Vec<TokenStreamAction<I, L, S>> {
        self.data.actions
    }
}

impl<P, I, L, S: Clone + Merge> TokenStreamActions<I, L, S>
    for TokenStreamActionCollector<P, I, L, S>
{
    type InstrLineActions = InstrLineActionCollector<P, I, L, S>;
    type TokenLineActions = TokenLineActionCollector<P, I, L, S>;
    type TokenLineFinalizer = TokenLineActionCollector<P, I, L, S>;

    fn will_parse_line(self) -> LineRule<Self::InstrLineActions, Self::TokenLineActions> {
        match self.data.state {
            LineRule::InstrLine(()) => LineRule::InstrLine(self.push_layer(())),
            LineRule::TokenLine(()) => LineRule::TokenLine(self.push_layer(())),
        }
    }

    fn act_on_eos(mut self, span: S) -> Self {
        self.data.actions.push(TokenStreamAction::Eos(span));
        self
    }
}

type InstrLineActionCollector<P, I, L, S> =
    ActionCollector<InstrLineActionCollectorData<P, I, L, S>, I>;

type InstrLineActionCollectorData<P, I, L, S> =
    CollectedData<InstrLineAction<I, L, S>, (), CollectedTokenStreamData<P, I, L, S>>;

impl<I, L, S> From<DiagnosticsEvent<S>> for InstrLineAction<I, L, S> {
    fn from(event: DiagnosticsEvent<S>) -> Self {
        match event {
            DiagnosticsEvent::EmitDiag(diag) => InstrLineAction::EmitDiag(diag),
        }
    }
}

impl<P, I, L, S: Clone + Merge> InstrLineActions<I, L, S> for InstrLineActionCollector<P, I, L, S> {
    type InstrActions = InstrActionCollector<P, I, L, S>;
    type LabelActions = LabelActionCollector<P, I, L, S>;

    fn will_parse_label(self, label: (I, S)) -> Self::LabelActions {
        self.push_layer(label)
    }
}

impl<P, I, L, S: Clone + Merge> InstrActions<I, L, S> for InstrLineActionCollector<P, I, L, S> {
    type BuiltinInstrActions = BuiltinInstrActionCollector<P, I, L, S>;
    type MacroInstrActions = MacroInstrActionCollector<P, I, L, S>;
    type ErrorActions = ErrorActionCollector<P, I, L, S>;
    type LineFinalizer = InstrActionCollector<P, I, L, S>;

    fn will_parse_instr(
        self,
        ident: I,
        span: S,
    ) -> InstrRule<Self::BuiltinInstrActions, Self::MacroInstrActions, Self::ErrorActions> {
        self.push_layer(()).will_parse_instr(ident, span)
    }
}

impl<P, I, L, S: Clone + Merge> LineFinalizer<S> for InstrLineActionCollector<P, I, L, S> {
    type Next = TokenStreamActionCollector<P, I, L, S>;

    fn did_parse_line(mut self, span: S) -> Self::Next {
        self.data.parent.actions.push(TokenStreamAction::InstrLine(
            self.data.actions.split_off(0),
            span,
        ));
        pop_layer!(self)
    }
}

type LabelActionCollector<P, I, L, S> = ActionCollector<CollectedLabelActionData<P, I, L, S>, I>;

type CollectedLabelActionData<P, I, L, S> =
    CollectedData<ParamsAction<I, S>, (I, S), InstrLineActionCollectorData<P, I, L, S>>;

impl<I, S> From<DiagnosticsEvent<S>> for ParamsAction<I, S> {
    fn from(event: DiagnosticsEvent<S>) -> Self {
        match event {
            DiagnosticsEvent::EmitDiag(diag) => ParamsAction::EmitDiag(diag),
        }
    }
}

impl<P, I, L, S: Clone + Merge> LabelActions<I, S> for LabelActionCollector<P, I, L, S> {
    type Next = InstrActionCollector<P, I, L, S>;

    fn act_on_param(&mut self, param: I, span: S) {
        self.data
            .actions
            .push(ParamsAction::AddParameter(param, span))
    }

    fn did_parse_label(mut self) -> Self::Next {
        self.data
            .parent
            .actions
            .push(InstrLineAction::Label((self.data.state, self.data.actions)));
        pop_layer!(self).push_layer(())
    }
}

type InstrActionCollector<P, I, L, S> = ActionCollector<CollectedInstrActionData<P, I, L, S>, I>;

type CollectedInstrActionData<P, I, L, S> =
    CollectedData<InstrAction<I, L, S>, (), InstrLineActionCollectorData<P, I, L, S>>;

impl<I, L, S> From<DiagnosticsEvent<S>> for InstrAction<I, L, S> {
    fn from(event: DiagnosticsEvent<S>) -> Self {
        match event {
            DiagnosticsEvent::EmitDiag(diag) => InstrAction::EmitDiag(diag),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct MacroId(pub TokenRef);

impl<P, I, L, S: Clone + Merge> InstrActions<I, L, S> for InstrActionCollector<P, I, L, S> {
    type BuiltinInstrActions = BuiltinInstrActionCollector<P, I, L, S>;
    type MacroInstrActions = MacroInstrActionCollector<P, I, L, S>;
    type ErrorActions = ErrorActionCollector<P, I, L, S>;
    type LineFinalizer = Self;

    fn will_parse_instr(
        self,
        ident: I,
        span: S,
    ) -> InstrRule<Self::BuiltinInstrActions, Self::MacroInstrActions, Self::ErrorActions> {
        match (self.annotate)(&ident) {
            IdentKind::BuiltinInstr | IdentKind::MacroKeyword | IdentKind::Endm => {
                InstrRule::BuiltinInstr(self.push_layer((ident, span)))
            }
            IdentKind::MacroName => InstrRule::MacroInstr(self.push_layer((ident, span))),
            IdentKind::Other => InstrRule::Error(self.push_layer(())),
        }
    }
}

impl<P, I, L, S: Clone + Merge> LineFinalizer<S> for InstrActionCollector<P, I, L, S> {
    type Next = TokenStreamActionCollector<P, I, L, S>;

    fn did_parse_line(mut self, span: S) -> Self::Next {
        if !self.data.actions.is_empty() {
            self.data
                .parent
                .actions
                .push(InstrLineAction::Instr(self.data.actions));
        }
        pop_layer!(self).did_parse_line(span)
    }
}

type ErrorActionCollector<P, I, L, S> = ActionCollector<CollectedErrorData<P, I, L, S>, I>;

type CollectedErrorData<P, I, L, S> =
    CollectedData<ErrorAction<S>, (), CollectedInstrActionData<P, I, L, S>>;

impl<S> From<DiagnosticsEvent<S>> for ErrorAction<S> {
    fn from(event: DiagnosticsEvent<S>) -> Self {
        match event {
            DiagnosticsEvent::EmitDiag(diag) => ErrorAction::EmitDiag(diag),
        }
    }
}

impl<P, I, L, S: Clone + Merge> InstrFinalizer<S> for ErrorActionCollector<P, I, L, S> {
    type Next = InstrActionCollector<P, I, L, S>;

    fn did_parse_instr(mut self) -> Self::Next {
        self.data
            .parent
            .actions
            .push(InstrAction::Error(self.data.actions));
        pop_layer!(self)
    }
}

type BuiltinInstrActionCollector<P, I, L, S> =
    ActionCollector<CollectedBuiltinInstrActionData<P, I, L, S>, I>;

type CollectedBuiltinInstrActionData<P, I, L, S> =
    CollectedData<BuiltinInstrAction<I, L, S>, (I, S), CollectedInstrActionData<P, I, L, S>>;

impl<I, L, S> From<DiagnosticsEvent<S>> for BuiltinInstrAction<I, L, S> {
    fn from(event: DiagnosticsEvent<S>) -> Self {
        match event {
            DiagnosticsEvent::EmitDiag(diag) => BuiltinInstrAction::EmitDiag(diag),
        }
    }
}

impl<P, I, L, S: Clone + Merge> BuiltinInstrActions<I, L, S>
    for BuiltinInstrActionCollector<P, I, L, S>
{
    type ArgActions = ExprActionCollector<CollectedBuiltinInstrActionData<P, I, L, S>, I, L, S>;

    fn will_parse_arg(self) -> Self::ArgActions {
        self.push_layer(())
    }
}

impl<P, I, L, S: Clone + Merge> InstrFinalizer<S> for BuiltinInstrActionCollector<P, I, L, S> {
    type Next = InstrActionCollector<P, I, L, S>;

    fn did_parse_instr(mut self) -> Self::Next {
        if (self.annotate)(&self.data.state.0) == IdentKind::MacroKeyword {
            self.data.parent.parent.parent.state = LineRule::TokenLine(())
        }
        self.data.parent.actions.push(InstrAction::BuiltinInstr {
            builtin_instr: self.data.state,
            actions: self.data.actions,
        });
        pop_layer!(self)
    }
}

pub(in crate::analysis) type ExprActionCollector<P, I, L, S> =
    ActionCollector<CollectedExprData<P, I, L, S>, I>;

type CollectedExprData<P, I, L, S> = CollectedData<ExprAction<I, L, S>, (), P>;

impl<S> ExprActionCollector<(), MockIdent, MockLiteral, S> {
    pub fn new() -> Self {
        Self {
            data: CollectedData {
                actions: Vec::new(),
                state: (),
                parent: (),
            },
            annotate,
        }
    }
}

impl<I, L, S> From<DiagnosticsEvent<S>> for ExprAction<I, L, S> {
    fn from(event: DiagnosticsEvent<S>) -> Self {
        match event {
            DiagnosticsEvent::EmitDiag(diag) => ExprAction::EmitDiag(diag),
        }
    }
}

impl<P, I, L, S> ArgFinalizer
    for ExprActionCollector<CollectedBuiltinInstrActionData<P, I, L, S>, I, L, S>
{
    type Next = BuiltinInstrActionCollector<P, I, L, S>;

    fn did_parse_arg(mut self) -> Self::Next {
        self.data
            .parent
            .actions
            .push(BuiltinInstrAction::AddArgument {
                actions: self.data.actions,
            });
        pop_layer!(self)
    }
}

impl<I, L, S> ArgFinalizer for ExprActionCollector<(), I, L, S> {
    type Next = Vec<ExprAction<I, L, S>>;

    fn did_parse_arg(self) -> Self::Next {
        self.data.actions
    }
}

impl<P, I, L, S: Clone> ArgActions<I, L, S> for ExprActionCollector<P, I, L, S>
where
    Self: Diagnostics<S>,
{
    fn act_on_atom(&mut self, atom: ExprAtom<I, L>, span: S) {
        self.data.actions.push(ExprAction::PushAtom(atom, span))
    }

    fn act_on_operator(&mut self, operator: Operator, span: S) {
        self.data
            .actions
            .push(ExprAction::ApplyOperator(operator, span))
    }
}

type TokenLineActionCollector<P, I, L, S> =
    ActionCollector<CollectedTokenLineActionData<P, I, L, S>, I>;

type CollectedTokenLineActionData<P, I, L, S> =
    CollectedData<TokenLineAction<I, L, S>, (), CollectedTokenStreamData<P, I, L, S>>;

impl<I, L, S> From<DiagnosticsEvent<S>> for TokenLineAction<I, L, S> {
    fn from(event: DiagnosticsEvent<S>) -> Self {
        match event {
            DiagnosticsEvent::EmitDiag(diag) => TokenLineAction::EmitDiag(diag),
        }
    }
}

impl<P, I, L, S: Clone + Merge> TokenLineActions<I, L, S> for TokenLineActionCollector<P, I, L, S> {
    type ContextFinalizer = Self;

    fn act_on_token(&mut self, token: Token<I, L>, span: S) {
        self.data
            .actions
            .push(TokenLineAction::Token((token, span)))
    }

    fn act_on_ident(mut self, ident: I, span: S) -> TokenLineRule<Self, Self::ContextFinalizer> {
        let kind = (self.annotate)(&ident);
        self.data
            .actions
            .push(TokenLineAction::Ident((ident, span)));
        match kind {
            IdentKind::Endm => {
                self.data.parent.state = LineRule::InstrLine(());
                TokenLineRule::LineEnd(self)
            }
            _ => TokenLineRule::TokenSeq(self),
        }
    }
}

impl<P, I, L, S: Clone + Merge> LineFinalizer<S> for TokenLineActionCollector<P, I, L, S> {
    type Next = TokenStreamActionCollector<P, I, L, S>;

    fn did_parse_line(mut self, span: S) -> Self::Next {
        self.data
            .parent
            .actions
            .push(TokenStreamAction::TokenLine(self.data.actions, span));
        pop_layer!(self)
    }
}

type MacroInstrActionCollector<P, I, L, S> =
    ActionCollector<CollectedMacroInstrData<P, I, L, S>, I>;

type CollectedMacroInstrData<P, I, L, S> =
    CollectedData<MacroInstrAction<I, L, S>, (I, S), CollectedInstrActionData<P, I, L, S>>;

impl<I, L, S> From<DiagnosticsEvent<S>> for MacroInstrAction<I, L, S> {
    fn from(event: DiagnosticsEvent<S>) -> Self {
        match event {
            DiagnosticsEvent::EmitDiag(diag) => MacroInstrAction::EmitDiag(diag),
        }
    }
}

impl<P, I, L, S: Clone + Merge> MacroInstrActions<S> for MacroInstrActionCollector<P, I, L, S> {
    type Token = Token<I, L>;
    type MacroArgActions = MacroArgActionCollector<P, I, L, S>;

    fn will_parse_macro_arg(self) -> Self::MacroArgActions {
        self.push_layer(())
    }
}

impl<P, I, L, S: Clone + Merge> InstrFinalizer<S> for MacroInstrActionCollector<P, I, L, S> {
    type Next = InstrActionCollector<P, I, L, S>;

    fn did_parse_instr(mut self) -> Self::Next {
        self.data.parent.actions.push(InstrAction::MacroInstr {
            name: self.data.state,
            actions: self.data.actions,
        });
        pop_layer!(self)
    }
}

type MacroArgActionCollector<P, I, L, S> = ActionCollector<CollectedMacroArgData<P, I, L, S>, I>;

type CollectedMacroArgData<P, I, L, S> =
    CollectedData<TokenSeqAction<I, L, S>, (), CollectedMacroInstrData<P, I, L, S>>;

impl<I, L, S> From<DiagnosticsEvent<S>> for TokenSeqAction<I, L, S> {
    fn from(event: DiagnosticsEvent<S>) -> Self {
        match event {
            DiagnosticsEvent::EmitDiag(diag) => TokenSeqAction::EmitDiag(diag),
        }
    }
}

impl<P, I, L, S: Clone + Merge> MacroArgActions<S> for MacroArgActionCollector<P, I, L, S> {
    type Token = Token<I, L>;
    type Next = MacroInstrActionCollector<P, I, L, S>;

    fn act_on_token(&mut self, token: (Self::Token, S)) {
        self.data.actions.push(TokenSeqAction::PushToken(token))
    }

    fn did_parse_macro_arg(mut self) -> Self::Next {
        self.data
            .parent
            .actions
            .push(MacroInstrAction::MacroArg(self.data.actions));
        pop_layer!(self)
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

    pub fn token_seq<T>(
        &self,
        tokens: impl Borrow<[T]>,
    ) -> Vec<TokenSeqAction<MockIdent, MockLiteral, MockSpan>>
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

    pub fn ident(
        &self,
        token_ref: impl Into<TokenRef>,
    ) -> TokenLineAction<MockIdent, MockLiteral, MockSpan> {
        match self.token(token_ref.into()) {
            (Token::Ident(ident), span) => TokenLineAction::Ident((ident, span)),
            _ => panic!("expected identifier"),
        }
    }

    pub fn eos(&self) -> TokenStreamAction<MockIdent, MockLiteral, MockSpan> {
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
pub(in crate::analysis) enum TokenStreamAction<I, L, S> {
    InstrLine(Vec<InstrLineAction<I, L, S>>, S),
    TokenLine(Vec<TokenLineAction<I, L, S>>, S),
    Eos(S),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analysis) enum InstrLineAction<I, L, S> {
    Label(Label<I, S>),
    Instr(Vec<InstrAction<I, L, S>>),
    EmitDiag(CompactDiag<S>),
}

pub(in crate::analysis) type Label<I, S> = ((I, S), Vec<ParamsAction<I, S>>);

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analysis) enum InstrAction<I, L, S> {
    BuiltinInstr {
        builtin_instr: (I, S),
        actions: Vec<BuiltinInstrAction<I, L, S>>,
    },
    MacroInstr {
        name: (I, S),
        actions: Vec<MacroInstrAction<I, L, S>>,
    },
    Error(Vec<ErrorAction<S>>),
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analysis) enum BuiltinInstrAction<I, L, S> {
    AddArgument { actions: Vec<ExprAction<I, L, S>> },
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analysis) enum ExprAction<I, L, S> {
    PushAtom(ExprAtom<I, L>, S),
    ApplyOperator(Operator, S),
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analysis) enum ParamsAction<I, S> {
    AddParameter(I, S),
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analysis) enum TokenLineAction<I, L, S> {
    Token((Token<I, L>, S)),
    Ident((I, S)),
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analysis) enum TokenSeqAction<I, L, S> {
    PushToken((Token<I, L>, S)),
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analysis) enum ErrorAction<S> {
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::analysis) enum MacroInstrAction<I, L, S> {
    MacroArg(Vec<TokenSeqAction<I, L, S>>),
    EmitDiag(CompactDiag<S>),
}

#[derive(Clone)]
pub(super) struct SymExpr(pub Vec<ExprAction<MockIdent, MockLiteral, MockSpan>>);

pub(super) fn instr_line(
    actions: Vec<InstrLineAction<MockIdent, MockLiteral, MockSpan>>,
    terminator: impl Into<TokenRef>,
) -> TokenStreamAction<MockIdent, MockLiteral, MockSpan> {
    TokenStreamAction::InstrLine(actions, terminator.into().into())
}

pub(super) fn token_line(
    actions: Vec<TokenLineAction<MockIdent, MockLiteral, MockSpan>>,
    terminator: impl Into<TokenRef>,
) -> TokenStreamAction<MockIdent, MockLiteral, MockSpan> {
    TokenStreamAction::TokenLine(actions, terminator.into().into())
}

pub(super) fn labeled(
    label: impl Into<TokenRef>,
    params: impl Borrow<[TokenRef]>,
    actions: Option<Vec<InstrAction<MockIdent, MockLiteral, MockSpan>>>,
    terminator: impl Into<TokenRef>,
) -> TokenStreamAction<MockIdent, MockLiteral, MockSpan> {
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
    actions: Vec<InstrAction<MockIdent, MockLiteral, MockSpan>>,
    terminator: impl Into<TokenRef>,
) -> TokenStreamAction<MockIdent, MockLiteral, MockSpan> {
    TokenStreamAction::InstrLine(
        vec![InstrLineAction::Instr(actions)],
        terminator.into().into(),
    )
}

pub(super) fn builtin_instr(
    kind: IdentKind,
    id: impl Into<TokenRef>,
    args: impl Borrow<[SymExpr]>,
) -> Vec<InstrAction<MockIdent, MockLiteral, MockSpan>> {
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
) -> Vec<InstrAction<MockIdent, MockLiteral, MockSpan>> {
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
    args: impl Borrow<[Vec<TokenSeqAction<MockIdent, MockLiteral, MockSpan>>]>,
) -> Vec<InstrAction<MockIdent, MockLiteral, MockSpan>> {
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

fn convert_params(params: impl Borrow<[TokenRef]>) -> Vec<ParamsAction<MockIdent, MockSpan>> {
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
