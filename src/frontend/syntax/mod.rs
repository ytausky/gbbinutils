use diagnostics::SourceRange;
use std::{cmp::PartialEq, fmt::Debug};

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

pub trait FileContext<S: TokenSpec, R>
where
    Self: Sized,
{
    type LineActions: LineActions<S, R, Parent = Self>;
    fn enter_line(self, label: Option<(S::Ident, R)>) -> Self::LineActions;
}

pub trait LineActions<TS: TokenSpec, SR>
where
    Self: Sized,
{
    type CommandContext: CommandContext<SR, TokenSpec = TS, Parent = Self>;
    type MacroDefContext: TokenSeqContext<SR, Token = parser::Token<TS>, Parent = Self>;
    type MacroInvocationContext: MacroInvocationContext<
        SR,
        Token = parser::Token<TS>,
        Parent = Self,
    >;
    type Parent;
    fn enter_command(self, name: (TS::Command, SR)) -> Self::CommandContext;
    fn enter_macro_def(self, name: (TS::Ident, SR)) -> Self::MacroDefContext;
    fn enter_macro_invocation(self, name: (TS::Ident, SR)) -> Self::MacroInvocationContext;
    fn exit(self) -> Self::Parent;
}

pub trait CommandContext<R> {
    type TokenSpec: TokenSpec;
    type Parent;
    fn add_argument(&mut self, expr: ParsedExpr<Self::TokenSpec, R>);
    fn exit(self) -> Self::Parent;
}

pub trait MacroInvocationContext<R>
where
    Self: Sized,
{
    type Token;
    type Parent;
    type MacroArgContext: TokenSeqContext<R, Token = Self::Token, Parent = Self>;
    fn enter_macro_arg(self) -> Self::MacroArgContext;
    fn exit(self) -> Self::Parent;
}

pub trait TokenSeqContext<R> {
    type Token;
    type Parent;
    fn push_token(&mut self, token: (Self::Token, R));
    fn exit(self) -> Self::Parent;
}

pub trait ExprSpec {
    type Ident: Debug + PartialEq;
    type Literal: Debug + PartialEq;
}

impl<T: TokenSpec> ExprSpec for T {
    type Ident = T::Ident;
    type Literal = T::Literal;
}

#[derive(Debug, PartialEq)]
pub enum ExprNode<S: ExprSpec, I> {
    Ident(S::Ident),
    Parenthesized(Box<ParsedExpr<S, I>>),
    Literal(S::Literal),
}

#[derive(Debug, PartialEq)]
pub struct ParsedExpr<S: ExprSpec, I> {
    pub node: ExprNode<S, I>,
    pub interval: I,
}
