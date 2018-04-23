pub mod keyword;
pub mod lexer;
mod parser;

pub fn tokenize(src: &str) -> self::lexer::Lexer {
    self::lexer::Lexer::new(src)
}

pub fn parse_token_seq<I, F>(tokens: I, actions: F)
where
    I: Iterator<Item = Token>,
    F: FileContext<String, ()>,
{
    use std::iter;
    self::parser::parse_src(tokens.zip(iter::repeat(())), actions)
}

pub type Token = self::parser::Token<String>;

pub mod token {
    pub use super::parser::Token::*;
}

pub trait TokenSpec {
    type Atom;
    type Command;
    type Label;
}

pub trait StringRef {}

impl StringRef for String {}
impl<'a> StringRef for &'a str {}

impl<T: StringRef> TokenSpec for T {
    type Atom = Atom<T>;
    type Command = keyword::Command;
    type Label = T;
}

#[derive(Clone, Debug, PartialEq)]
pub enum Atom<S> {
    Ident(S),
    Operand(keyword::Operand),
    Number(i32),
    String(S),
}

impl TokenSpec for () {
    type Atom = ();
    type Command = ();
    type Label = ();
}

pub trait FileContext<S: TokenSpec, T>
where
    Self: Sized,
{
    type CommandContext: CommandContext<Token = parser::Token<S>, Parent = Self>;
    type MacroDefContext: TokenSeqContext<Token = parser::Token<S>, Parent = Self>;
    type MacroInvocationContext: MacroInvocationContext<Token = parser::Token<S>, Parent = Self>;
    fn add_label(&mut self, label: (S::Label, T));
    fn enter_command(self, name: (S::Command, T)) -> Self::CommandContext;
    fn enter_macro_def(self, name: (S::Label, T)) -> Self::MacroDefContext;
    fn enter_macro_invocation(self, name: (S::Atom, T)) -> Self::MacroInvocationContext;
}

pub trait CommandContext {
    type Token;
    type Parent;
    fn add_argument(&mut self, expr: SynExpr<Self::Token>);
    fn exit_command(self) -> Self::Parent;
}

pub trait MacroInvocationContext
where
    Self: Sized,
{
    type Token;
    type Parent;
    type MacroArgContext: TokenSeqContext<Token = Self::Token, Parent = Self>;
    fn enter_macro_arg(self) -> Self::MacroArgContext;
    fn exit_macro_invocation(self) -> Self::Parent;
}

pub trait TokenSeqContext {
    type Token;
    type Parent;
    fn push_token(&mut self, token: Self::Token);
    fn exit_token_seq(self) -> Self::Parent;
}

#[derive(Clone, Debug, PartialEq)]
pub enum SynExpr<T> {
    Atom(T),
    Deref(Box<SynExpr<T>>),
}

impl<T> From<T> for SynExpr<T> {
    fn from(atom: T) -> Self {
        SynExpr::Atom(atom)
    }
}

impl<T> SynExpr<T> {
    pub fn deref(self) -> Self {
        SynExpr::Deref(Box::new(self))
    }
}
