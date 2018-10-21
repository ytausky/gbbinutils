use super::Token::*;
use super::{ExprAtom, ExprOperator, Token};
use diagnostics::{InternalDiagnostic, Message};
use span::Span;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq)]
pub enum RpnAction<I, L> {
    Push(ExprAtom<I, L>),
    Apply(ExprOperator),
}

pub type RpnExpr<I, L, S> = Vec<(RpnAction<I, L>, S)>;

pub fn expr() -> SymExpr {
    SymExpr(Vec::new())
}

impl SymExpr {
    pub fn ident(self, token: impl Into<TokenRef>) -> Self {
        self.push(token, ExprAtom::Ident)
    }

    pub fn literal(self, token: impl Into<TokenRef>) -> Self {
        self.push(token, ExprAtom::Literal)
    }

    fn push(
        mut self,
        token: impl Into<TokenRef>,
        atom_ctor: impl Fn(TokenRef) -> ExprAtom<TokenRef, TokenRef>,
    ) -> Self {
        let token_ref = token.into();
        self.0.push((
            RpnAction::Push(atom_ctor(token_ref.clone())),
            token_ref.into(),
        ));
        self
    }

    pub fn parentheses(mut self, left: impl Into<TokenRef>, right: impl Into<TokenRef>) -> Self {
        let span = SymRange::from(left.into()).extend(&right.into().into());
        self.0
            .push((RpnAction::Apply(ExprOperator::Parentheses), span));
        self
    }

    pub fn plus(mut self, token: impl Into<TokenRef>) -> Self {
        self.0
            .push((RpnAction::Apply(ExprOperator::Plus), token.into().into()));
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymCommand(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub struct SymIdent(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub struct SymLiteral(pub usize);

pub type SymToken = Token<SymIdent, SymCommand, SymLiteral>;

pub fn mk_sym_token(id: usize, token: Token<(), (), ()>) -> SymToken {
    match token {
        Command(()) => Command(SymCommand(id)),
        Ident(()) => Ident(SymIdent(id)),
        Literal(()) => Literal(SymLiteral(id)),
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
    }
}

pub struct InputTokens {
    pub tokens: Vec<SymToken>,
    pub names: HashMap<String, usize>,
}

macro_rules! add_token {
    ($input:expr, $token:expr) => {
        let id = $input.tokens.len();
        $input.tokens.push(mk_sym_token(id, $token))
    };
    ($input:expr, $name:ident @ $token:expr) => {
        $input
            .names
            .insert(stringify!($name).into(), $input.tokens.len());
        add_token!($input, $token)
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

#[derive(Debug, PartialEq)]
pub enum Action {
    AcceptExpr(RpnExpr<SymIdent, SymLiteral, SymRange<usize>>),
    AddParameter(SymIdent),
    EnterArgument,
    EnterInstruction(SymCommand),
    EnterLine(Option<SymIdent>),
    EnterMacroArg,
    EnterMacroBody,
    EnterMacroDef,
    EnterMacroInvocation(SymIdent),
    Error(InternalDiagnostic<SymRange<usize>>),
    ExitArgument,
    ExitInstruction,
    ExitLine,
    ExitMacroArg,
    ExitMacroDef,
    ExitMacroInvocation,
    PushTerminal(usize),
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

impl TokenRef {
    fn resolve(&self, input: &InputTokens) -> usize {
        match self {
            TokenRef::Id(id) => *id,
            TokenRef::Name(name) => *input.names.get(name).unwrap(),
        }
    }
}

impl SymRange<TokenRef> {
    fn resolve(&self, input: &InputTokens) -> SymRange<usize> {
        SymRange {
            start: self.start.resolve(input),
            end: self.end.resolve(input),
        }
    }
}

pub struct File(Vec<Line>);

pub fn file(lines: impl Borrow<[Line]>) -> File {
    File(lines.borrow().iter().cloned().collect())
}

impl File {
    pub fn into_actions(self, input: &InputTokens) -> Vec<Action> {
        self.0
            .into_iter()
            .flat_map(|line| line.into_actions(input))
            .collect()
    }
}

#[derive(Clone)]
pub struct Line(Option<usize>, Option<LineBody>);

pub fn labeled(label: usize, body: Option<LineBody>) -> Line {
    Line(Some(label), body)
}

pub fn unlabeled(body: Option<LineBody>) -> Line {
    Line(None, body)
}

impl Line {
    fn into_actions(self, input: &InputTokens) -> Vec<Action> {
        let mut actions = vec![Action::EnterLine(self.0.map(|id| SymIdent(id)))];
        if let Some(body) = self.1 {
            actions.append(&mut body.into_actions(input))
        }
        actions.push(Action::ExitLine);
        actions
    }
}

pub fn empty() -> Option<LineBody> {
    None
}

#[derive(Clone)]
pub enum LineBody {
    Command(TokenRef, Vec<SymExpr>, Option<SymDiagnostic>),
    Error(SymDiagnostic),
    Invoke(usize, Vec<TokenSeq>),
    MacroDef(Vec<usize>, MacroDefTail),
}

#[derive(Clone)]
pub struct SymExpr(RpnExpr<TokenRef, TokenRef, SymRange<TokenRef>>);

impl SymExpr {
    pub fn resolve(self, input: &InputTokens) -> RpnExpr<SymIdent, SymLiteral, SymRange<usize>> {
        self.0
            .into_iter()
            .map(|(rpn, span)| {
                (
                    match rpn {
                        RpnAction::Push(ExprAtom::Ident(ident)) => {
                            RpnAction::Push(ExprAtom::Ident(SymIdent(ident.resolve(input))))
                        }
                        RpnAction::Push(ExprAtom::Literal(literal)) => {
                            RpnAction::Push(ExprAtom::Literal(SymLiteral(literal.resolve(input))))
                        }
                        RpnAction::Apply(operator) => RpnAction::Apply(operator),
                    },
                    span.resolve(input),
                )
            }).collect()
    }

    fn into_action(self, input: &InputTokens) -> Action {
        Action::AcceptExpr(self.resolve(input))
    }
}

#[derive(Clone)]
pub enum MacroDefTail {
    Body(Vec<usize>, Option<SymDiagnostic>),
    Error(SymDiagnostic),
}

#[derive(Clone)]
pub struct SymDiagnostic(pub InternalDiagnostic<SymRange<TokenRef>>);

impl From<SymDiagnostic> for LineBody {
    fn from(diagnostic: SymDiagnostic) -> Self {
        LineBody::Error(diagnostic)
    }
}

impl SymDiagnostic {
    pub fn resolve(self, input: &InputTokens) -> InternalDiagnostic<SymRange<usize>> {
        InternalDiagnostic::new(
            self.0.message,
            self.0.spans.into_iter().map(|span| span.resolve(input)),
            self.0.highlight.resolve(input),
        )
    }

    fn into_action(self, input: &InputTokens) -> Action {
        Action::Error(self.resolve(input))
    }
}

pub fn command(id: impl Into<TokenRef>, args: impl Borrow<[SymExpr]>) -> Option<LineBody> {
    Some(LineBody::Command(
        id.into(),
        args.borrow().iter().cloned().collect(),
        None,
    ))
}

pub fn malformed_command(
    id: impl Into<TokenRef>,
    args: impl Borrow<[SymExpr]>,
    diagnostic: SymDiagnostic,
) -> Option<LineBody> {
    Some(LineBody::Command(
        id.into(),
        args.borrow().iter().cloned().collect(),
        Some(diagnostic),
    ))
}

pub fn invoke(id: usize, args: impl Borrow<[TokenSeq]>) -> Option<LineBody> {
    Some(LineBody::Invoke(
        id,
        args.borrow().iter().cloned().collect(),
    ))
}

pub fn macro_def(
    params: impl Borrow<[usize]>,
    body: impl Borrow<[usize]>,
    endm: usize,
) -> Option<LineBody> {
    use std::iter::once;
    Some(LineBody::MacroDef(
        params.borrow().iter().cloned().collect(),
        MacroDefTail::Body(
            body.borrow().iter().cloned().chain(once(endm)).collect(),
            None,
        ),
    ))
}

pub fn malformed_macro_def_head(
    params: impl Borrow<[usize]>,
    diagnostic: SymDiagnostic,
) -> Option<LineBody> {
    Some(LineBody::MacroDef(
        params.borrow().iter().cloned().collect(),
        MacroDefTail::Error(diagnostic),
    ))
}

pub fn malformed_macro_def(
    params: impl Borrow<[usize]>,
    body: impl Borrow<[usize]>,
    diagnostic: SymDiagnostic,
) -> Option<LineBody> {
    Some(LineBody::MacroDef(
        params.borrow().iter().cloned().collect(),
        MacroDefTail::Body(body.borrow().iter().cloned().collect(), Some(diagnostic)),
    ))
}

impl LineBody {
    fn into_actions(self, input: &InputTokens) -> Vec<Action> {
        let mut actions = Vec::new();
        match self {
            LineBody::Command(id, args, error) => {
                actions.push(Action::EnterInstruction(SymCommand(id.resolve(input))));
                for mut arg in args {
                    actions.push(Action::EnterArgument);
                    actions.push(arg.into_action(input));
                    actions.push(Action::ExitArgument)
                }
                if let Some(diagnostic) = error {
                    actions.push(diagnostic.into_action(input))
                }
                actions.push(Action::ExitInstruction)
            }
            LineBody::Error(diagnostic) => actions.push(diagnostic.into_action(input)),
            LineBody::Invoke(id, args) => {
                actions.push(Action::EnterMacroInvocation(SymIdent(id)));
                for arg in args {
                    actions.push(Action::EnterMacroArg);
                    actions.append(&mut arg.into_actions());
                    actions.push(Action::ExitMacroArg)
                }
                actions.push(Action::ExitMacroInvocation)
            }
            LineBody::MacroDef(params, tail) => {
                actions.push(Action::EnterMacroDef);
                actions.extend(
                    params
                        .into_iter()
                        .map(|id| Action::AddParameter(SymIdent(id))),
                );
                match tail {
                    MacroDefTail::Body(body, error) => {
                        actions.push(Action::EnterMacroBody);
                        actions.extend(body.into_iter().map(|t| Action::PushTerminal(t)));
                        if let Some(diagnostic) = error {
                            actions.push(diagnostic.into_action(input))
                        }
                    }
                    MacroDefTail::Error(diagnostic) => {
                        actions.push(diagnostic.into_action(input));
                        actions.push(Action::EnterMacroBody)
                    }
                }
                actions.push(Action::ExitMacroDef)
            }
        }
        actions
    }
}

#[derive(Clone)]
pub struct TokenSeq(Vec<usize>);

pub fn token_seq(ids: impl Borrow<[usize]>) -> TokenSeq {
    TokenSeq(ids.borrow().iter().cloned().collect())
}

impl TokenSeq {
    fn into_actions(self) -> Vec<Action> {
        self.0
            .into_iter()
            .map(|id| Action::PushTerminal(id))
            .collect()
    }
}

pub fn line_error(
    message: Message,
    ranges: impl Borrow<[&'static str]>,
    highlight: impl Into<TokenRef>,
) -> Option<LineBody> {
    Some(arg_error(message, ranges, highlight).into())
}

pub fn arg_error(
    message: Message,
    ranges: impl Borrow<[&'static str]>,
    highlight: impl Into<TokenRef>,
) -> SymDiagnostic {
    SymDiagnostic(InternalDiagnostic::new(
        message,
        ranges.borrow().iter().map(|s| TokenRef::from(*s).into()),
        highlight.into().into(),
    ))
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
            [Command(SymCommand(0)), Literal(SymLiteral(1)), Macro]
        );
        assert_eq!(tokens.names.get("my_tok"), Some(&0));
        assert_eq!(tokens.names.get("next_one"), Some(&2))
    }
}
