use crate::diagnostics::DownstreamDiagnostics;

#[cfg(test)]
#[macro_use]
mod ast;
pub mod keyword;
pub mod lexer;
mod parser;

pub use crate::frontend::syntax::keyword::Operand;

#[derive(Clone, Debug, PartialEq)]
pub enum Token<I, C = keyword::Command, L = Literal<I>, E = lexer::LexError> {
    ClosingParenthesis,
    Comma,
    Command(C),
    Endm,
    Eof,
    Eol,
    Error(E),
    Ident(I),
    Label(I),
    Literal(L),
    Macro,
    OpeningParenthesis,
    Plus,
}

pub fn tokenize(src: &str) -> self::lexer::Lexer {
    self::lexer::Lexer::new(src)
}

pub fn parse_token_seq<Id, I, F>(tokens: I, actions: F)
where
    I: Iterator<Item = (Token<Id>, F::Span)>,
    F: FileContext<Id, keyword::Command, Literal<Id>>,
{
    self::parser::parse_src(tokens, actions);
}

#[derive(Clone, Debug, PartialEq)]
pub enum Literal<S> {
    Operand(Operand),
    Number(i32),
    String(S),
}

pub trait FileContext<I, C, L>
where
    Self: DownstreamDiagnostics + Sized,
{
    type StmtContext: StmtContext<I, C, L, Span = Self::Span, Parent = Self>;
    fn enter_stmt(self, label: Option<(I, Self::Span)>) -> Self::StmtContext;
}

pub trait StmtContext<I, C, L>
where
    Self: DownstreamDiagnostics + Sized,
{
    type CommandContext: CommandContext<
        Ident = I,
        Command = C,
        Literal = L,
        Span = Self::Span,
        Parent = Self,
    >;
    type MacroParamsContext: MacroParamsContext<
        Ident = I,
        Command = C,
        Literal = L,
        Span = Self::Span,
        Parent = Self,
    >;
    type MacroInvocationContext: MacroInvocationContext<
        Token = Token<I, C, L>,
        Span = Self::Span,
        Parent = Self,
    >;
    type Parent;
    fn enter_command(self, name: (C, Self::Span)) -> Self::CommandContext;
    fn enter_macro_def(self, keyword: Self::Span) -> Self::MacroParamsContext;
    fn enter_macro_invocation(self, name: (I, Self::Span)) -> Self::MacroInvocationContext;
    fn exit(self) -> Self::Parent;
}

pub trait CommandContext
where
    Self: DownstreamDiagnostics + Sized,
{
    type Ident;
    type Command;
    type Literal;
    type ArgContext: ExprContext<
        Ident = Self::Ident,
        Literal = Self::Literal,
        Span = Self::Span,
        Parent = Self,
    >;
    type Parent;
    fn add_argument(self) -> Self::ArgContext;
    fn exit(self) -> Self::Parent;
}

pub trait ExprContext
where
    Self: DownstreamDiagnostics,
{
    type Ident;
    type Literal;
    type Parent;
    fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, Self::Span));
    fn apply_operator(&mut self, operator: (ExprOperator, Self::Span));
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

pub trait MacroParamsContext
where
    Self: DownstreamDiagnostics,
{
    type Ident;
    type Command;
    type Literal;
    type MacroBodyContext: TokenSeqContext<
        Token = Token<Self::Ident, Self::Command, Self::Literal>,
        Span = Self::Span,
        Parent = Self::Parent,
    >;
    type Parent;
    fn add_parameter(&mut self, param: (Self::Ident, Self::Span));
    fn exit(self) -> Self::MacroBodyContext;
}

pub trait MacroInvocationContext
where
    Self: DownstreamDiagnostics + Sized,
{
    type Token;
    type Parent;
    type MacroArgContext: TokenSeqContext<Token = Self::Token, Span = Self::Span, Parent = Self>;
    fn enter_macro_arg(self) -> Self::MacroArgContext;
    fn exit(self) -> Self::Parent;
}

pub trait TokenSeqContext
where
    Self: DownstreamDiagnostics,
{
    type Token;
    type Parent;
    fn push_token(&mut self, token: (Self::Token, Self::Span));
    fn exit(self) -> Self::Parent;
}
