use super::SimpleToken::*;
use super::Token::*;
use super::{ExprAtom, Operator, Token, UnaryOperator};
use crate::diag::{CompactDiagnostic, Message};
use crate::expr::BinaryOperator;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::iter;

pub(crate) fn expr() -> SymExpr {
    SymExpr(Vec::new())
}

impl SymExpr {
    pub fn ident(self, token: impl Into<TokenRef>) -> Self {
        self.push(token, |t| ExprAtom::Ident(SymIdent(t)))
    }

    pub fn literal(self, token: impl Into<TokenRef>) -> Self {
        self.push(token, |t| ExprAtom::Literal(SymLiteral(t)))
    }

    fn push(
        mut self,
        token: impl Into<TokenRef>,
        atom_ctor: impl Fn(TokenRef) -> ExprAtom<SymIdent, SymLiteral>,
    ) -> Self {
        let token_ref = token.into();
        self.0.push(ExprAction::PushAtom((
            atom_ctor(token_ref.clone()),
            token_ref.into(),
        )));
        self
    }

    pub fn divide(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator((
            Operator::Binary(BinaryOperator::Division),
            token.into().into(),
        )));
        self
    }

    pub fn multiply(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator((
            Operator::Binary(BinaryOperator::Multiplication),
            token.into().into(),
        )));
        self
    }

    pub fn parentheses(mut self, left: impl Into<TokenRef>, right: impl Into<TokenRef>) -> Self {
        let span = SymSpan::merge(left.into(), right.into());
        self.0.push(ExprAction::ApplyOperator((
            Operator::Unary(UnaryOperator::Parentheses),
            span,
        )));
        self
    }

    pub fn plus(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator((
            Operator::Binary(BinaryOperator::Plus),
            token.into().into(),
        )));
        self
    }

    pub fn minus(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator((
            Operator::Binary(BinaryOperator::Minus),
            token.into().into(),
        )));
        self
    }

    pub fn bitwise_or(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator((
            Operator::Binary(BinaryOperator::BitwiseOr),
            token.into().into(),
        )));
        self
    }

    pub fn fn_call(mut self, args: usize, span: impl Into<SymSpan>) -> Self {
        self.0.push(ExprAction::ApplyOperator((
            Operator::FnCall(args),
            span.into(),
        )));
        self
    }

    pub fn error(mut self, message: Message<SymSpan>, highlight: impl Into<SymSpan>) -> Self {
        self.0.push(ExprAction::EmitDiagnostic(
            message.at(highlight.into()).into(),
        ));
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymCommand(pub TokenRef);

#[derive(Clone, Debug, PartialEq)]
pub struct SymIdent(pub TokenRef);

#[derive(Clone, Debug, PartialEq)]
pub struct SymLiteral(pub TokenRef);

pub type SymToken = Token<SymIdent, SymLiteral, SymCommand>;

pub fn mk_sym_token(id: impl Into<TokenRef>, token: Token<(), (), ()>) -> (SymToken, TokenRef) {
    let token_ref = id.into();
    (
        match token {
            Command(()) => Command(SymCommand(token_ref.clone())),
            Ident(()) => Ident(SymIdent(token_ref.clone())),
            Label(()) => Label(SymIdent(token_ref.clone())),
            Literal(()) => Literal(SymLiteral(token_ref.clone())),
            Simple(simple) => Simple(simple),
        },
        token_ref,
    )
}

pub(crate) struct InputTokens {
    pub tokens: Vec<(SymToken, TokenRef)>,
    pub names: HashMap<String, usize>,
}

impl InputTokens {
    pub fn insert_token(&mut self, id: impl Into<TokenRef>, token: Token<(), (), ()>) {
        self.tokens.push(mk_sym_token(id, token))
    }

    pub fn token_seq<T>(&self, tokens: impl Borrow<[T]>) -> Vec<TokenSeqAction<SymSpan>>
    where
        T: Clone + Into<TokenRef>,
    {
        tokens
            .borrow()
            .iter()
            .cloned()
            .map(Into::into)
            .map(|t| TokenSeqAction::PushToken((self.token(t.clone()), t.into())))
            .collect()
    }

    fn token(&self, token_ref: impl Into<TokenRef>) -> SymToken {
        let id = match token_ref.into() {
            TokenRef::Id(n) => n,
            TokenRef::Name(name) => *self.names.get(&name).unwrap(),
        };
        self.tokens[id].0.clone()
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
            names: HashMap::new(),
        };
        input_tokens_impl!(input, $($tokens)*);
        if input
            .tokens
            .last()
            .map(|(token, _)| *token != Eof.into())
            .unwrap_or(true)
        {
            let eof_id = input.tokens.len().into();
            input.tokens.push((Eof.into(), eof_id))
        }
        input
    }};
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymSpan {
    pub start: TokenRef,
    pub end: TokenRef,
}

impl From<TokenRef> for SymSpan {
    fn from(token_ref: TokenRef) -> SymSpan {
        SymSpan {
            start: token_ref.clone(),
            end: token_ref,
        }
    }
}

impl SymSpan {
    pub fn merge(left: impl Into<SymSpan>, right: impl Into<SymSpan>) -> SymSpan {
        SymSpan {
            start: left.into().start.clone(),
            end: right.into().end.clone(),
        }
    }
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

#[derive(Debug, PartialEq)]
pub(crate) enum FileAction<S> {
    Stmt {
        label: Option<(SymIdent, S)>,
        actions: Vec<StmtAction<S>>,
    },
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

#[derive(Debug, PartialEq)]
pub(crate) enum StmtAction<S> {
    Command {
        command: (SymCommand, S),
        actions: Vec<CommandAction<S>>,
    },
    MacroDef {
        keyword: S,
        params: Vec<MacroParamsAction<S>>,
        body: Vec<TokenSeqAction<S>>,
    },
    MacroInvocation {
        name: (SymIdent, S),
        actions: Vec<MacroInvocationAction<S>>,
    },
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

#[derive(Debug, PartialEq)]
pub(crate) enum CommandAction<S> {
    AddArgument { actions: Vec<ExprAction<S>> },
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum ExprAction<S> {
    PushAtom((ExprAtom<SymIdent, SymLiteral>, S)),
    ApplyOperator((Operator, S)),
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

#[derive(Debug, PartialEq)]
pub(crate) enum MacroParamsAction<S> {
    AddParameter((SymIdent, S)),
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TokenSeqAction<S> {
    PushToken((Token<SymIdent, SymLiteral, SymCommand>, S)),
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

#[derive(Debug, PartialEq)]
pub(crate) enum MacroInvocationAction<S> {
    MacroArg(Vec<TokenSeqAction<S>>),
    EmitDiagnostic(CompactDiagnostic<S, S>),
}

#[derive(Clone)]
pub(crate) struct SymExpr(pub Vec<ExprAction<SymSpan>>);

pub(crate) fn labeled(
    label: impl Into<TokenRef>,
    actions: Vec<StmtAction<SymSpan>>,
) -> FileAction<SymSpan> {
    let label = label.into();
    FileAction::Stmt {
        label: Some((SymIdent(label.clone()), label.into())),
        actions,
    }
}

pub(crate) fn unlabeled(actions: Vec<StmtAction<SymSpan>>) -> FileAction<SymSpan> {
    FileAction::Stmt {
        label: None,
        actions,
    }
}

pub(crate) fn empty() -> Vec<StmtAction<SymSpan>> {
    Vec::new()
}

pub(crate) fn command(
    id: impl Into<TokenRef>,
    args: impl Borrow<[SymExpr]>,
) -> Vec<StmtAction<SymSpan>> {
    let id = id.into();
    vec![StmtAction::Command {
        command: (SymCommand(id.clone()), id.into()),
        actions: args
            .borrow()
            .iter()
            .cloned()
            .map(|SymExpr(expr)| CommandAction::AddArgument { actions: expr })
            .collect(),
    }]
}

pub(crate) fn malformed_command(
    id: impl Into<TokenRef>,
    args: impl Borrow<[SymExpr]>,
    diagnostic: CompactDiagnostic<SymSpan, SymSpan>,
) -> Vec<StmtAction<SymSpan>> {
    let id = id.into();
    vec![StmtAction::Command {
        command: (SymCommand(id.clone()), id.into()),
        actions: args
            .borrow()
            .iter()
            .cloned()
            .map(|SymExpr(expr)| CommandAction::AddArgument { actions: expr })
            .chain(iter::once(CommandAction::EmitDiagnostic(diagnostic)))
            .collect(),
    }]
}

pub(crate) fn invoke(
    id: impl Into<TokenRef>,
    args: impl Borrow<[Vec<TokenSeqAction<SymSpan>>]>,
) -> Vec<StmtAction<SymSpan>> {
    let id = id.into();
    vec![StmtAction::MacroInvocation {
        name: (SymIdent(id.clone()), id.into()),
        actions: args
            .borrow()
            .iter()
            .cloned()
            .map(MacroInvocationAction::MacroArg)
            .collect(),
    }]
}

pub(crate) fn macro_def(
    keyword: impl Into<TokenRef>,
    params: impl Borrow<[TokenRef]>,
    mut body: Vec<TokenSeqAction<SymSpan>>,
    endm: impl Into<TokenRef>,
) -> Vec<StmtAction<SymSpan>> {
    body.push(TokenSeqAction::PushToken((Eof.into(), endm.into().into())));
    vec![StmtAction::MacroDef {
        keyword: keyword.into().into(),
        params: params
            .borrow()
            .iter()
            .cloned()
            .map(|t| MacroParamsAction::AddParameter((SymIdent(t.clone()), t.into())))
            .collect(),
        body,
    }]
}

pub(crate) fn malformed_macro_def_head(
    keyword: impl Into<TokenRef>,
    params: impl Borrow<[TokenRef]>,
    diagnostic: CompactDiagnostic<SymSpan, SymSpan>,
) -> Vec<StmtAction<SymSpan>> {
    vec![StmtAction::MacroDef {
        keyword: keyword.into().into(),
        params: params
            .borrow()
            .iter()
            .cloned()
            .map(|t| MacroParamsAction::AddParameter((SymIdent(t.clone()), t.into())))
            .chain(iter::once(MacroParamsAction::EmitDiagnostic(diagnostic)))
            .collect(),
        body: Vec::new(),
    }]
}

pub(crate) fn malformed_macro_def(
    keyword: impl Into<TokenRef>,
    params: impl Borrow<[TokenRef]>,
    mut body: Vec<TokenSeqAction<SymSpan>>,
    diagnostic: CompactDiagnostic<SymSpan, SymSpan>,
) -> Vec<StmtAction<SymSpan>> {
    body.push(TokenSeqAction::EmitDiagnostic(diagnostic));
    vec![StmtAction::MacroDef {
        keyword: keyword.into().into(),
        params: params
            .borrow()
            .iter()
            .cloned()
            .map(|t| MacroParamsAction::AddParameter((SymIdent(t.clone()), t.into())))
            .collect(),
        body,
    }]
}

pub(crate) fn stmt_error(
    message: Message<SymSpan>,
    highlight: impl Into<TokenRef>,
) -> Vec<StmtAction<SymSpan>> {
    vec![StmtAction::EmitDiagnostic(arg_error(message, highlight))]
}

pub(crate) fn arg_error(
    message: Message<SymSpan>,
    highlight: impl Into<TokenRef>,
) -> CompactDiagnostic<SymSpan, SymSpan> {
    message.at(highlight.into().into()).into()
}

mod tests {
    use super::*;

    #[test]
    fn test_token_macro() {
        let tokens = input_tokens![
            my_tok @ Command(()),
            Literal(()),
            next_one @ Macro,
        ];
        assert_eq!(
            tokens.tokens,
            [
                (Command(SymCommand("my_tok".into())), "my_tok".into()),
                (Literal(SymLiteral(1.into())), 1.into()),
                (Macro.into(), "next_one".into()),
                (Eof.into(), 3.into()),
            ]
        );
        assert_eq!(tokens.names.get("my_tok"), Some(&0));
        assert_eq!(tokens.names.get("next_one"), Some(&2))
    }
}
