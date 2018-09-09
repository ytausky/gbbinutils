use diagnostics::{Message, Span};
use std::fmt::Debug;

#[cfg(test)]
use super::ExprAtom;
#[cfg(test)]
use super::ExprOperator;
#[cfg(test)]
use super::Token;
#[cfg(test)]
use super::Token::*;
#[cfg(test)]
use diagnostics::Diagnostic;
#[cfg(test)]
use std::borrow::Borrow;
#[cfg(test)]
use std::collections::HashMap;

#[derive(Clone)]
pub struct Expr<I, L, S: Clone> {
    pub variant: ExprVariant<I, L, S>,
    pub span: S,
}

#[derive(Clone)]
pub enum ExprVariant<I, L, S: Clone> {
    Ident(I),
    Literal(L),
    Parentheses(Box<Expr<I, L, S>>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymCommand(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub struct SymIdent(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub struct SymLiteral(pub usize);

#[cfg(test)]
pub type SymToken = Token<SymIdent, SymCommand, SymLiteral>;

#[cfg(test)]
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
        Macro => Macro,
        OpeningParenthesis => OpeningParenthesis,
    }
}

#[cfg(test)]
pub struct InputTokens {
    pub tokens: Vec<SymToken>,
    pub names: HashMap<String, usize>,
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

impl<T: Clone + Debug> Span for SymRange<T> {
    fn extend(&self, other: &Self) -> Self {
        SymRange {
            start: self.start.clone(),
            end: other.end.clone(),
        }
    }
}

#[cfg(test)]
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
    PushExprAtom(ExprAtom<SymIdent, SymLiteral>),
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

#[cfg(test)]
impl TokenRef {
    fn resolve(&self, input: &InputTokens) -> usize {
        match self {
            TokenRef::Id(id) => *id,
            TokenRef::Name(name) => *input.names.get(name).unwrap(),
        }
    }
}

#[cfg(test)]
impl SymRange<TokenRef> {
    fn resolve(&self, input: &InputTokens) -> SymRange<usize> {
        SymRange {
            start: self.start.resolve(input),
            end: self.end.resolve(input),
        }
    }
}

#[cfg(test)]
pub struct File(Vec<Line>);

#[cfg(test)]
pub fn file(lines: impl Borrow<[Line]>) -> File {
    File(lines.borrow().iter().cloned().collect())
}

#[cfg(test)]
impl File {
    pub fn into_actions(self, input: &InputTokens) -> Vec<Action> {
        self.0
            .into_iter()
            .flat_map(|line| line.into_actions(input))
            .collect()
    }
}

#[cfg(test)]
#[derive(Clone)]
pub struct Line(Option<usize>, Option<LineBody>);

#[cfg(test)]
pub fn labeled(label: usize, body: Option<LineBody>) -> Line {
    Line(Some(label), body)
}

#[cfg(test)]
pub fn unlabeled(body: Option<LineBody>) -> Line {
    Line(None, body)
}

#[cfg(test)]
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

#[cfg(test)]
pub fn empty() -> Option<LineBody> {
    None
}

#[cfg(test)]
#[derive(Clone)]
pub enum LineBody {
    Command(TokenRef, Vec<SymExpr>, Option<SymDiagnostic>),
    Error(SymDiagnostic),
    Invoke(usize, Vec<TokenSeq>),
    MacroDef(Vec<usize>, MacroDefTail),
}

#[cfg(test)]
pub type SymExpr = Expr<TokenRef, TokenRef, SymRange<TokenRef>>;

#[cfg(test)]
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

#[cfg(test)]
impl From<SymDiagnostic> for LineBody {
    fn from(diagnostic: SymDiagnostic) -> Self {
        LineBody::Error(diagnostic)
    }
}

#[cfg(test)]
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

#[cfg(test)]
pub fn command(id: impl Into<TokenRef>, args: impl Borrow<[SymExpr]>) -> Option<LineBody> {
    Some(LineBody::Command(
        id.into(),
        args.borrow().iter().cloned().collect(),
        None,
    ))
}

#[cfg(test)]
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

#[cfg(test)]
pub fn invoke(id: usize, args: impl Borrow<[TokenSeq]>) -> Option<LineBody> {
    Some(LineBody::Invoke(
        id,
        args.borrow().iter().cloned().collect(),
    ))
}

#[cfg(test)]
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

#[cfg(test)]
pub fn malformed_macro_def_head(
    params: impl Borrow<[usize]>,
    diagnostic: SymDiagnostic,
) -> Option<LineBody> {
    Some(LineBody::MacroDef(
        params.borrow().iter().cloned().collect(),
        MacroDefTail::Error(diagnostic),
    ))
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
pub fn token_seq(ids: impl Borrow<[usize]>) -> TokenSeq {
    TokenSeq(ids.borrow().iter().cloned().collect())
}

#[cfg(test)]
impl TokenSeq {
    fn into_actions(self) -> Vec<Action> {
        self.0
            .into_iter()
            .map(|id| Action::PushTerminal(id))
            .collect()
    }
}

#[cfg(test)]
pub fn ident(id: impl Into<TokenRef>) -> SymExpr {
    let token_ref = id.into();
    Expr {
        variant: ExprVariant::Ident(token_ref.clone()),
        span: token_ref.into(),
    }
}

#[cfg(test)]
pub fn literal(id: impl Into<TokenRef>) -> SymExpr {
    let token_ref = id.into();
    Expr {
        variant: ExprVariant::Literal(token_ref.clone()),
        span: token_ref.into(),
    }
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
pub fn line_error(
    message_ctor: MessageCtor,
    ranges: impl Borrow<[&'static str]>,
    highlight: impl Into<TokenRef>,
) -> Option<LineBody> {
    Some(arg_error(message_ctor, ranges, highlight).into())
}

#[cfg(test)]
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

#[cfg(test)]
pub fn unexpected_token(ranges: Vec<SymRange<usize>>) -> Message<SymRange<usize>> {
    Message::UnexpectedToken {
        token: ranges.into_iter().next().unwrap(),
    }
}

#[cfg(test)]
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
