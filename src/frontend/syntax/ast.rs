use super::parser::Token;
use super::parser::Token::*;
use super::{ExprAtom, ExprOperator, TokenSpec};
use diagnostics::{Diagnostic, Message, SourceRange};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq)]
pub struct Symbolic;

impl TokenSpec for Symbolic {
    type Command = SymCommand;
    type Ident = SymIdent;
    type Literal = SymLiteral;
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymCommand(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub struct SymIdent(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub struct SymLiteral(pub usize);

pub type SymToken = Token<Symbolic>;

pub fn mk_sym_token(id: usize, token: Token<()>) -> SymToken {
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
        Macro => Macro,
        OpeningParenthesis => OpeningParenthesis,
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

impl<T: Clone + Debug> SourceRange for SymRange<T> {
    fn extend(&self, other: &Self) -> Self {
        SymRange {
            start: self.start.clone(),
            end: other.end.clone(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Action {
    AddParameter(SymIdent),
    ApplyExprOperator(ExprOperator),
    EnterArgument,
    EnterInstruction(SymCommand),
    EnterLine(Option<SymIdent>),
    EnterMacroArg,
    EnterMacroBody,
    EnterMacroDef,
    EnterMacroInvocation(SymIdent),
    Error(Diagnostic<SymRange<usize>>),
    ExitArgument,
    ExitInstruction,
    ExitLine,
    ExitMacroArg,
    ExitMacroDef,
    ExitMacroInvocation,
    PushExprAtom(ExprAtom<Symbolic>),
    PushTerminal(usize),
}

#[derive(Clone, Debug)]
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

pub type SymExpr = Expr<TokenRef, TokenRef, SymRange<TokenRef>>;

#[derive(Clone)]
pub enum MacroDefTail {
    Body(Vec<usize>, Option<SymDiagnostic>),
    Error(SymDiagnostic),
}

#[derive(Clone)]
pub struct SymDiagnostic {
    message_ctor: MessageCtor,
    ranges: Vec<SymRange<TokenRef>>,
    highlight: SymRange<TokenRef>,
}

pub type MessageCtor = fn(Vec<SymRange<usize>>) -> Message<SymRange<usize>>;

impl From<SymDiagnostic> for LineBody {
    fn from(diagnostic: SymDiagnostic) -> Self {
        LineBody::Error(diagnostic)
    }
}

impl SymDiagnostic {
    fn into_action(self, input: &InputTokens) -> Action {
        let message = (self.message_ctor)(
            self.ranges
                .into_iter()
                .map(|range| range.resolve(input))
                .collect(),
        );
        Action::Error(Diagnostic::new(message, self.highlight.resolve(input)))
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
                    actions.append(&mut arg.into_actions(input));
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

pub fn ident(id: impl Into<TokenRef>) -> SymExpr {
    let token_ref = id.into();
    Expr {
        variant: ExprVariant::Ident(token_ref.clone()),
        span: token_ref.into(),
    }
}

pub fn literal(id: impl Into<TokenRef>) -> SymExpr {
    let token_ref = id.into();
    Expr {
        variant: ExprVariant::Literal(token_ref.clone()),
        span: token_ref.into(),
    }
}

pub fn parentheses(
    open_id: impl Into<TokenRef>,
    expr: SymExpr,
    close_id: impl Into<TokenRef>,
) -> SymExpr {
    Expr {
        variant: ExprVariant::Parentheses(Box::new(expr)),
        span: SymRange::from(open_id.into()).extend(&close_id.into().into()),
    }
}

#[derive(Clone)]
pub struct Expr<I, L, S> {
    variant: ExprVariant<I, L, S>,
    span: S,
}

#[derive(Clone)]
pub enum ExprVariant<I, L, S> {
    Ident(I),
    Literal(L),
    Parentheses(Box<Expr<I, L, S>>),
}

impl Expr<TokenRef, TokenRef, SymRange<TokenRef>> {
    fn into_actions(self, input: &InputTokens) -> Vec<Action> {
        match self.variant {
            ExprVariant::Ident(ident) => vec![Action::PushExprAtom(ExprAtom::Ident(SymIdent(
                ident.resolve(input),
            )))],
            ExprVariant::Literal(literal) => vec![Action::PushExprAtom(ExprAtom::Literal(
                SymLiteral(literal.resolve(input)),
            ))],
            ExprVariant::Parentheses(inner) => {
                let mut actions = inner.into_actions(input);
                actions.push(Action::ApplyExprOperator(ExprOperator::Parentheses));
                actions
            }
        }
    }
}

pub fn line_error(
    message_ctor: MessageCtor,
    ranges: impl Borrow<[&'static str]>,
    highlight: impl Into<TokenRef>,
) -> Option<LineBody> {
    Some(arg_error(message_ctor, ranges, highlight).into())
}

pub fn arg_error(
    message_ctor: MessageCtor,
    ranges: impl Borrow<[&'static str]>,
    highlight: impl Into<TokenRef>,
) -> SymDiagnostic {
    SymDiagnostic {
        message_ctor,
        ranges: ranges
            .borrow()
            .iter()
            .map(|s| TokenRef::from(*s).into())
            .collect(),
        highlight: highlight.into().into(),
    }
}

pub fn unexpected_token(ranges: Vec<SymRange<usize>>) -> Message<SymRange<usize>> {
    Message::UnexpectedToken {
        token: ranges.into_iter().next().unwrap(),
    }
}

pub fn unexpected_eof(_ranges: Vec<SymRange<usize>>) -> Message<SymRange<usize>> {
    Message::UnexpectedEof
}

#[cfg(test)]
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
