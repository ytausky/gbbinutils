use diagnostics::{DiagnosticsListener, Span};

#[cfg(test)]
#[macro_use]
mod ast;
pub mod keyword;
pub mod lexer;
mod parser;

pub use frontend::syntax::keyword::Operand;

#[derive(Clone, Debug, PartialEq)]
pub enum Token<I, C = keyword::Command, L = Literal<I>> {
    ClosingParenthesis,
    Colon,
    Comma,
    Command(C),
    Endm,
    Eof,
    Eol,
    Ident(I),
    Literal(L),
    Macro,
    OpeningParenthesis,
    Plus,
}

pub fn tokenize(src: &str) -> self::lexer::Lexer {
    self::lexer::Lexer::new(src)
}

pub fn parse_token_seq<Id, S: Span, I, F>(tokens: I, actions: F)
where
    I: Iterator<Item = (Token<Id>, S)>,
    F: FileContext<Id, keyword::Command, Literal<Id>, S>,
{
    self::parser::parse_src(tokens, actions)
}

#[derive(Clone, Debug, PartialEq)]
pub enum Literal<S> {
    Operand(Operand),
    Number(i32),
    String(S),
}

pub trait FileContext<I, C, L, S>
where
    Self: DiagnosticsListener<S> + Sized,
{
    type LineActions: LineActions<I, C, L, S, Parent = Self>;
    fn enter_line(self, label: Option<(I, S)>) -> Self::LineActions;
}

pub trait LineActions<I, C, L, S>
where
    Self: DiagnosticsListener<S> + Sized,
{
    type CommandContext: CommandContext<S, Ident = I, Command = C, Literal = L, Parent = Self>;
    type MacroParamsActions: MacroParamsActions<
        S,
        Ident = I,
        Command = C,
        Literal = L,
        Parent = Self,
    >;
    type MacroInvocationContext: MacroInvocationContext<S, Token = Token<I, C, L>, Parent = Self>;
    type Parent;
    fn enter_command(self, name: (C, S)) -> Self::CommandContext;
    fn enter_macro_def(self) -> Self::MacroParamsActions;
    fn enter_macro_invocation(self, name: (I, S)) -> Self::MacroInvocationContext;
    fn exit(self) -> Self::Parent;
}

pub trait CommandContext<S>
where
    Self: DiagnosticsListener<S> + Sized,
{
    type Ident;
    type Command;
    type Literal;
    type ArgActions: ExprActions<S, Ident = Self::Ident, Literal = Self::Literal, Parent = Self>;
    type Parent;
    fn add_argument(self) -> Self::ArgActions;
    fn exit(self) -> Self::Parent;
}

pub trait ExprActions<S> {
    type Ident;
    type Literal;
    type Parent;
    fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, S));
    fn apply_operator(&mut self, operator: (ExprOperator, S));
    fn exit(self) -> Self::Parent;
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprAtom<I, L> {
    Ident(I),
    Literal(L),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprOperator {
    Parentheses,
    Plus,
}

pub trait MacroParamsActions<S>: DiagnosticsListener<S> {
    type Ident;
    type Command;
    type Literal;
    type MacroBodyActions: TokenSeqContext<
        S,
        Token = Token<Self::Ident, Self::Command, Self::Literal>,
        Parent = Self::Parent,
    >;
    type Parent;
    fn add_parameter(&mut self, param: (Self::Ident, S));
    fn exit(self) -> Self::MacroBodyActions;
}

pub trait MacroInvocationContext<S>
where
    Self: DiagnosticsListener<S> + Sized,
{
    type Token;
    type Parent;
    type MacroArgContext: TokenSeqContext<S, Token = Self::Token, Parent = Self>;
    fn enter_macro_arg(self) -> Self::MacroArgContext;
    fn exit(self) -> Self::Parent;
}

pub trait TokenSeqContext<S>
where
    Self: DiagnosticsListener<S>,
{
    type Token;
    type Parent;
    fn push_token(&mut self, token: (Self::Token, S));
    fn exit(self) -> Self::Parent;
}
