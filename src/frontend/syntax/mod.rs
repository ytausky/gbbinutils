use diagnostics::{DiagnosticsListener, SourceRange};
use std::{cmp::PartialEq, fmt::Debug};

#[cfg(test)]
#[macro_use]
mod ast;
pub mod keyword;
pub mod lexer;
mod parser;

pub use frontend::syntax::keyword::Operand;

pub fn tokenize(src: &str) -> self::lexer::Lexer {
    self::lexer::Lexer::new(src)
}

pub fn parse_token_seq<R: SourceRange, I, F>(tokens: I, actions: F)
where
    I: Iterator<Item = (Token, R)>,
    F: FileContext<String, R>,
{
    self::parser::parse_src(tokens, actions)
}

pub type Token = self::parser::Token<String>;

pub mod token {
    pub use super::parser::Token::*;
}

pub trait TokenSpec {
    type Command: Debug + PartialEq;
    type Ident: Debug + PartialEq;
    type Literal: Debug + PartialEq;
}

pub trait StringRef: Debug + PartialEq {}

impl StringRef for String {}
impl<'a> StringRef for &'a str {}

impl<T: StringRef> TokenSpec for T {
    type Command = keyword::Command;
    type Ident = T;
    type Literal = Literal<T>;
}

#[derive(Clone, Debug, PartialEq)]
pub enum Literal<S> {
    Operand(Operand),
    Number(i32),
    String(S),
}

impl TokenSpec for () {
    type Command = ();
    type Ident = ();
    type Literal = ();
}

pub trait FileContext<S: TokenSpec, SR>
where
    Self: DiagnosticsListener<SR> + Sized,
{
    type LineActions: LineActions<S, SR, Parent = Self>;
    fn enter_line(self, label: Option<(S::Ident, SR)>) -> Self::LineActions;
}

pub trait LineActions<TS: TokenSpec, SR>
where
    Self: DiagnosticsListener<SR> + Sized,
{
    type CommandContext: CommandContext<SR, TokenSpec = TS, Parent = Self>;
    type MacroParamsActions: MacroParamsActions<SR, TokenSpec = TS, Parent = Self>;
    type MacroInvocationContext: MacroInvocationContext<
        SR,
        Token = parser::Token<TS>,
        Parent = Self,
    >;
    type Parent;
    fn enter_command(self, name: (TS::Command, SR)) -> Self::CommandContext;
    fn enter_macro_def(self) -> Self::MacroParamsActions;
    fn enter_macro_invocation(self, name: (TS::Ident, SR)) -> Self::MacroInvocationContext;
    fn exit(self) -> Self::Parent;
}

pub trait CommandContext<SR>
where
    Self: DiagnosticsListener<SR> + Sized,
{
    type TokenSpec: TokenSpec;
    type ArgActions: ExprActions<SR, TokenSpec = Self::TokenSpec, Parent = Self>;
    type Parent;
    fn add_argument(self) -> Self::ArgActions;
    fn exit(self) -> Self::Parent;
}

pub trait ExprActions<SR> {
    type TokenSpec: TokenSpec;
    type Parent;
    fn push_atom(&mut self, atom: (ExprAtom<Self::TokenSpec>, SR));
    fn apply_operator(&mut self, operator: (ExprOperator, SR));
    fn exit(self) -> Self::Parent;
}

#[derive(Debug, PartialEq)]
pub enum ExprAtom<S: TokenSpec> {
    Ident(S::Ident),
    Literal(S::Literal),
}

#[derive(Debug, PartialEq)]
pub enum ExprOperator {
    Parentheses,
}

pub trait MacroParamsActions<SR>: DiagnosticsListener<SR> {
    type TokenSpec: TokenSpec;
    type MacroBodyActions: TokenSeqContext<
        SR,
        Token = parser::Token<Self::TokenSpec>,
        Parent = Self::Parent,
    >;
    type Parent;
    fn add_parameter(&mut self, param: (<Self::TokenSpec as TokenSpec>::Ident, SR));
    fn exit(self) -> Self::MacroBodyActions;
}

pub trait MacroInvocationContext<SR>
where
    Self: DiagnosticsListener<SR> + Sized,
{
    type Token;
    type Parent;
    type MacroArgContext: TokenSeqContext<SR, Token = Self::Token, Parent = Self>;
    fn enter_macro_arg(self) -> Self::MacroArgContext;
    fn exit(self) -> Self::Parent;
}

pub trait TokenSeqContext<R> {
    type Token;
    type Parent;
    fn push_token(&mut self, token: (Self::Token, R));
    fn exit(self) -> Self::Parent;
}
