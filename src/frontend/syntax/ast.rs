use super::Token::*;
use super::{ExprAtom, ExprOperator, Token};
use diagnostics::{InternalDiagnostic, Message};
use span::Span;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter;

pub fn expr() -> SymExpr {
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

    pub fn parentheses(mut self, left: impl Into<TokenRef>, right: impl Into<TokenRef>) -> Self {
        let span = SymRange::from(left.into()).extend(&right.into().into());
        self.0
            .push(ExprAction::ApplyOperator((ExprOperator::Parentheses, span)));
        self
    }

    pub fn plus(mut self, token: impl Into<TokenRef>) -> Self {
        self.0.push(ExprAction::ApplyOperator((
            ExprOperator::Plus,
            token.into().into(),
        )));
        self
    }

    pub fn error(mut self, message: Message, highlight: impl Into<SymRange<TokenRef>>) -> Self {
        self.0
            .push(ExprAction::EmitDiagnostic(InternalDiagnostic::new(
                message,
                iter::empty(),
                highlight.into(),
            )));
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymCommand(pub TokenRef);

#[derive(Clone, Debug, PartialEq)]
pub struct SymIdent(pub TokenRef);

#[derive(Clone, Debug, PartialEq)]
pub struct SymLiteral(pub TokenRef);

pub type SymToken = Token<SymIdent, SymCommand, SymLiteral>;

pub fn mk_sym_token(id: impl Into<TokenRef>, token: Token<(), (), ()>) -> (SymToken, TokenRef) {
    let token_ref = id.into();
    (
        match token {
            Command(()) => Command(SymCommand(token_ref.clone())),
            Ident(()) => Ident(SymIdent(token_ref.clone())),
            Literal(()) => Literal(SymLiteral(token_ref.clone())),
            ClosingParenthesis => ClosingParenthesis,
            Colon => Colon,
            Comma => Comma,
            Endm => Endm,
            Eof => Eof,
            Eol => Eol,
            Error(error) => Error(error),
            Macro => Macro,
            OpeningParenthesis => OpeningParenthesis,
            Plus => Plus,
        },
        token_ref,
    )
}

pub struct InputTokens {
    pub tokens: Vec<(SymToken, TokenRef)>,
    pub names: HashMap<String, usize>,
}

impl InputTokens {
    fn token(&self, token_ref: impl Into<TokenRef>) -> SymToken {
        let id = match token_ref.into() {
            TokenRef::Id(n) => n,
            TokenRef::Name(name) => *self.names.get(&name).unwrap(),
        };
        self.tokens[id].0.clone()
    }

    pub fn token_seq<T>(&self, tokens: impl Borrow<[T]>) -> Vec<TokenSeqAction<SymRange<TokenRef>>>
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
}

macro_rules! add_token {
    ($input:expr, $token:expr) => {
        let id = $input.tokens.len();
        add_token!($input, $token, id)
    };
    ($input:expr, $name:ident @ $token:expr) => {
        let id = stringify!($name);
        $input.names.insert(id.into(), $input.tokens.len());
        add_token!($input, $token, id)
    };
    ($input:expr, $token:expr, $id:expr) => {
        $input.tokens.push(mk_sym_token($id, $token))
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
    () => {
        InputTokens {
            tokens: Vec::new(),
            names: HashMap::new(),
        }
    };
    ($($tokens:tt)*) => {{
        let mut input = input_tokens![];
        input_tokens_impl!(input, $($tokens)*);
        input
    }};
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymRange<T> {
    pub start: T,
    pub end: T,
}

impl<T: Clone> From<T> for SymRange<T> {
    fn from(x: T) -> SymRange<T> {
        SymRange {
            start: x.clone(),
            end: x,
        }
    }
}

impl<T: Clone + Debug + PartialEq> Span for SymRange<T> {
    fn extend(&self, other: &Self) -> Self {
        SymRange {
            start: self.start.clone(),
            end: other.end.clone(),
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
pub enum FileAction<S> {
    Stmt {
        label: Option<(SymIdent, S)>,
        actions: Vec<StmtAction<S>>,
    },
    EmitDiagnostic(InternalDiagnostic<S>),
}

#[derive(Debug, PartialEq)]
pub enum StmtAction<S> {
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
    EmitDiagnostic(InternalDiagnostic<S>),
}

#[derive(Debug, PartialEq)]
pub enum CommandAction<S> {
    AddArgument { actions: Vec<ExprAction<S>> },
    EmitDiagnostic(InternalDiagnostic<S>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum ExprAction<S> {
    PushAtom((ExprAtom<SymIdent, SymLiteral>, S)),
    ApplyOperator((ExprOperator, S)),
    EmitDiagnostic(InternalDiagnostic<S>),
}

#[derive(Debug, PartialEq)]
pub enum MacroParamsAction<S> {
    AddParameter((SymIdent, S)),
    EmitDiagnostic(InternalDiagnostic<S>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum TokenSeqAction<S> {
    PushToken((Token<SymIdent, SymCommand, SymLiteral>, S)),
    EmitDiagnostic(InternalDiagnostic<S>),
}

#[derive(Debug, PartialEq)]
pub enum MacroInvocationAction<S> {
    MacroArg(Vec<TokenSeqAction<S>>),
    EmitDiagnostic(InternalDiagnostic<S>),
}

#[derive(Clone)]
pub struct SymExpr(pub Vec<ExprAction<SymRange<TokenRef>>>);

pub fn labeled(
    label: impl Into<TokenRef>,
    actions: Vec<StmtAction<SymRange<TokenRef>>>,
) -> FileAction<SymRange<TokenRef>> {
    let label = label.into();
    FileAction::Stmt {
        label: Some((SymIdent(label.clone()), label.into())),
        actions,
    }
}

pub fn unlabeled(actions: Vec<StmtAction<SymRange<TokenRef>>>) -> FileAction<SymRange<TokenRef>> {
    FileAction::Stmt {
        label: None,
        actions,
    }
}

pub fn empty() -> Vec<StmtAction<SymRange<TokenRef>>> {
    Vec::new()
}

pub fn command(
    id: impl Into<TokenRef>,
    args: impl Borrow<[SymExpr]>,
) -> Vec<StmtAction<SymRange<TokenRef>>> {
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

pub fn malformed_command(
    id: impl Into<TokenRef>,
    args: impl Borrow<[SymExpr]>,
    diagnostic: InternalDiagnostic<SymRange<TokenRef>>,
) -> Vec<StmtAction<SymRange<TokenRef>>> {
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

pub fn invoke(
    id: impl Into<TokenRef>,
    args: impl Borrow<[Vec<TokenSeqAction<SymRange<TokenRef>>>]>,
) -> Vec<StmtAction<SymRange<TokenRef>>> {
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

pub fn macro_def(
    keyword: impl Into<TokenRef>,
    params: impl Borrow<[TokenRef]>,
    mut body: Vec<TokenSeqAction<SymRange<TokenRef>>>,
    endm: impl Into<TokenRef>,
) -> Vec<StmtAction<SymRange<TokenRef>>> {
    body.push(TokenSeqAction::PushToken((Token::Eof, endm.into().into())));
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

pub fn malformed_macro_def_head(
    keyword: impl Into<TokenRef>,
    params: impl Borrow<[TokenRef]>,
    diagnostic: InternalDiagnostic<SymRange<TokenRef>>,
) -> Vec<StmtAction<SymRange<TokenRef>>> {
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

pub fn malformed_macro_def(
    keyword: impl Into<TokenRef>,
    params: impl Borrow<[TokenRef]>,
    mut body: Vec<TokenSeqAction<SymRange<TokenRef>>>,
    diagnostic: InternalDiagnostic<SymRange<TokenRef>>,
) -> Vec<StmtAction<SymRange<TokenRef>>> {
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

pub fn stmt_error(
    message: Message,
    ranges: impl Borrow<[&'static str]>,
    highlight: impl Into<TokenRef>,
) -> Vec<StmtAction<SymRange<TokenRef>>> {
    vec![StmtAction::EmitDiagnostic(arg_error(
        message, ranges, highlight,
    ))]
}

pub fn arg_error(
    message: Message,
    ranges: impl Borrow<[&'static str]>,
    highlight: impl Into<TokenRef>,
) -> InternalDiagnostic<SymRange<TokenRef>> {
    InternalDiagnostic::new(
        message,
        ranges.borrow().iter().map(|s| TokenRef::from(*s).into()),
        highlight.into().into(),
    )
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
                (Macro, "next_one".into())
            ]
        );
        assert_eq!(tokens.names.get("my_tok"), Some(&0));
        assert_eq!(tokens.names.get("next_one"), Some(&2))
    }
}
