use super::Sigil::*;
use super::Token::*;
use super::{ExprAtom, Operator, Token, UnaryOperator};

use crate::analysis::syntax::actions::mock::*;
use crate::diag::{CompactDiag, Merge, Message};
use crate::model::BinOp;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::iter;

pub(in crate::analysis) fn annotate(&MockIdent(kind, _): &MockIdent) -> IdentKind {
    kind
}

pub(super) type MockSpan = crate::diag::MockSpan<TokenRef>;

pub(super) fn with_spans<'a>(
    tokens: impl IntoIterator<Item = &'a (MockToken, TokenRef)>,
) -> impl Iterator<Item = (Result<MockToken, ()>, MockSpan)> {
    tokens.into_iter().cloned().map(|(t, r)| (Ok(t), r.into()))
}

#[derive(Debug, PartialEq)]
pub struct MacroId(pub TokenRef);

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
