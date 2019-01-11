use crate::diag::DelegateDiagnostics;
use crate::expr::BinaryOperator;

#[cfg(test)]
#[macro_use]
mod ast;
pub mod keyword;
pub mod lexer;
mod parser;

pub use crate::frontend::syntax::keyword::Operand;

#[derive(Clone, Debug, PartialEq)]
pub enum Token<I, L = Literal<I>, C = keyword::Command, E = lexer::LexError> {
    Command(C),
    Error(E),
    Ident(I),
    Label(I),
    Literal(L),
    Simple(SimpleToken),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimpleToken {
    ClosingParenthesis,
    Comma,
    Endm,
    Eof,
    Eol,
    Macro,
    Minus,
    OpeningParenthesis,
    Plus,
    Slash,
    Star,
}

impl<I, L, C, E> From<SimpleToken> for Token<I, L, C, E> {
    fn from(simple: SimpleToken) -> Self {
        Token::Simple(simple)
    }
}

pub fn tokenize(src: &str) -> self::lexer::Lexer {
    self::lexer::Lexer::new(src)
}

pub(crate) fn parse_token_seq<Id, I, F, S>(tokens: I, actions: F)
where
    I: Iterator<Item = (Token<Id>, S)>,
    F: FileContext<Id, Literal<Id>, keyword::Command, S>,
    S: Clone,
{
    self::parser::parse_src(tokens, actions);
}

#[derive(Clone, Debug, PartialEq)]
pub enum Literal<S> {
    Operand(Operand),
    Number(i32),
    String(S),
}

pub(crate) trait FileContext<I, L, C, S: Clone>: DelegateDiagnostics<S> + Sized {
    type StmtContext: StmtContext<I, L, C, S, Parent = Self>;
    fn enter_stmt(self, label: Option<(I, S)>) -> Self::StmtContext;
}

pub(crate) trait StmtContext<I, L, C, S: Clone>: DelegateDiagnostics<S> + Sized {
    type CommandContext: CommandContext<S, Ident = I, Command = C, Literal = L, Parent = Self>;
    type MacroParamsContext: MacroParamsContext<
        S,
        Ident = I,
        Command = C,
        Literal = L,
        Parent = Self,
    >;
    type MacroInvocationContext: MacroInvocationContext<S, Token = Token<I, L, C>, Parent = Self>;
    type Parent;
    fn enter_command(self, name: (C, S)) -> Self::CommandContext;
    fn enter_macro_def(self, keyword: S) -> Self::MacroParamsContext;
    fn enter_macro_invocation(self, name: (I, S)) -> Self::MacroInvocationContext;
    fn exit(self) -> Self::Parent;
}

pub(crate) trait CommandContext<S: Clone>: DelegateDiagnostics<S> + Sized {
    type Ident;
    type Command;
    type Literal;
    type ArgContext: ExprContext<S, Ident = Self::Ident, Literal = Self::Literal, Parent = Self>;
    type Parent;
    fn add_argument(self) -> Self::ArgContext;
    fn exit(self) -> Self::Parent;
}

pub(crate) trait ExprContext<S: Clone>: DelegateDiagnostics<S> {
    type Ident;
    type Literal;
    type Parent;
    fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, S));
    fn apply_operator(&mut self, operator: (Operator, S));
    fn exit(self) -> Self::Parent;
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprAtom<I, L> {
    Ident(I),
    Literal(L),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operator {
    Unary(UnaryOperator),
    Binary(BinaryOperator),
}

#[derive(Clone, Debug, PartialEq)]
pub enum UnaryOperator {
    Parentheses,
}

pub(crate) trait MacroParamsContext<S: Clone>: DelegateDiagnostics<S> {
    type Ident;
    type Command;
    type Literal;
    type MacroBodyContext: TokenSeqContext<
        S,
        Token = Token<Self::Ident, Self::Literal, Self::Command>,
        Parent = Self::Parent,
    >;
    type Parent;
    fn add_parameter(&mut self, param: (Self::Ident, S));
    fn exit(self) -> Self::MacroBodyContext;
}

pub(crate) trait MacroInvocationContext<S: Clone>: DelegateDiagnostics<S> + Sized {
    type Token;
    type Parent;
    type MacroArgContext: TokenSeqContext<S, Token = Self::Token, Parent = Self>;
    fn enter_macro_arg(self) -> Self::MacroArgContext;
    fn exit(self) -> Self::Parent;
}

pub(crate) trait TokenSeqContext<S: Clone>: DelegateDiagnostics<S> {
    type Token;
    type Parent;
    fn push_token(&mut self, token: (Self::Token, S));
    fn exit(self) -> Self::Parent;
}
