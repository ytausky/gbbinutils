pub mod keyword;
pub mod lexer;
mod parser;

pub fn tokenize(src: &str) -> self::lexer::Lexer {
    self::lexer::Lexer::new(src)
}

pub fn parse_token_seq<R, I, F>(tokens: I, actions: F)
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

pub trait FileContext<S: TokenSpec, R>
where
    Self: Sized,
{
    type CommandContext: CommandContext<R, TokenSpec = S, Parent = Self>;
    type MacroDefContext: TokenSeqContext<R, Token = parser::Token<S>, Parent = Self>;
    type MacroInvocationContext: MacroInvocationContext<R, Token = parser::Token<S>, Parent = Self>;
    fn add_label(&mut self, label: (S::Label, R));
    fn enter_command(self, name: (S::Command, R)) -> Self::CommandContext;
    fn enter_macro_def(self, name: (S::Label, R)) -> Self::MacroDefContext;
    fn enter_macro_invocation(self, name: (S::Atom, R)) -> Self::MacroInvocationContext;
}

pub trait CommandContext<R> {
    type TokenSpec: TokenSpec;
    type Parent;
    fn add_argument(&mut self, expr: SynExpr<Self::TokenSpec, R>);
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
    type Atom;
}

impl<T: TokenSpec> ExprSpec for T {
    type Atom = T::Atom;
}

#[derive(Clone, Debug, PartialEq)]
pub enum SynExpr<S: ExprSpec, R> {
    Atom((S::Atom, R)),
    Deref(Box<SynExpr<S, R>>),
}

impl<S: ExprSpec, R> SynExpr<S, R> {
    pub fn deref(self) -> Self {
        SynExpr::Deref(Box::new(self))
    }
}
