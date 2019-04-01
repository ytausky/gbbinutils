use crate::diag::DelegateDiagnostics;
use crate::expr::BinaryOperator;
use std::borrow::Borrow;

#[cfg(test)]
#[macro_use]
mod ast;
pub mod keyword;
pub mod lexer;
mod parser;

pub use self::keyword::{Command, Directive, Mnemonic, Operand};

#[derive(Clone, Debug, PartialEq)]
pub enum Token<I, L, C> {
    Command(C),
    Ident(I),
    Label(I),
    Literal(L),
    Simple(SimpleToken),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimpleToken {
    ClosingParenthesis,
    Comma,
    Dot,
    Endm,
    Eof,
    Eol,
    Expr,
    Macro,
    Minus,
    OpeningParenthesis,
    Pipe,
    Plus,
    Slash,
    Star,
}

impl<I, L, C> From<SimpleToken> for Token<I, L, C> {
    fn from(simple: SimpleToken) -> Self {
        Token::Simple(simple)
    }
}

pub(super) fn tokenize<B, F, I>(src: B, mk_ident: F) -> self::lexer::Lexer<B, F>
where
    B: Borrow<str>,
    F: for<'a> Fn(&'a str) -> I,
{
    self::lexer::Lexer::new(src, mk_ident)
}

pub(crate) use self::parser::parse_src as parse_token_seq;

pub(crate) trait FileContext<I, L, C, S: Clone>: DelegateDiagnostics<S> + Sized {
    type StmtContext: StmtContext<I, L, C, S, Parent = Self>;
    fn enter_stmt(self, label: Option<(I, S)>) -> Self::StmtContext;
}

pub(crate) trait StmtContext<I, L, C, S: Clone>: DelegateDiagnostics<S> + Sized {
    type CommandContext: CommandContext<S, Ident = I, Command = C, Literal = L, Parent = Self>;
    type ExprParamsContext: ParamsContext<S, Ident = I> + ToExprBody<S, Literal = L, Parent = Self>;
    type MacroParamsContext: ParamsContext<S, Ident = I>
        + ToMacroBody<S, Command = C, Literal = L, Parent = Self>;
    type MacroInvocationContext: MacroInvocationContext<S, Token = Token<I, L, C>, Parent = Self>;
    type Parent;
    fn enter_command(self, name: (C, S)) -> Self::CommandContext;
    fn enter_expr_def(self, keyword: S) -> Self::ExprParamsContext;
    fn enter_macro_def(self, keyword: S) -> Self::MacroParamsContext;
    fn enter_macro_invocation(self, name: (I, S)) -> Self::MacroInvocationContext;
    fn exit(self) -> Self::Parent;
}

pub(crate) trait CommandContext<S: Clone>: DelegateDiagnostics<S> + Sized {
    type Ident;
    type Command;
    type Literal;
    type ArgContext: ExprContext<S, Ident = Self::Ident, Literal = Self::Literal>
        + FinalContext<ReturnTo = Self>;
    type Parent;
    fn add_argument(self) -> Self::ArgContext;
    fn exit(self) -> Self::Parent;
}

pub(crate) trait FinalContext {
    type ReturnTo;

    fn exit(self) -> Self::ReturnTo;
}

pub(crate) trait ExprContext<S: Clone>: AssocExpr + DelegateDiagnostics<S> {
    fn push_atom(&mut self, atom: (ExprAtom<Self::Ident, Self::Literal>, S));
    fn apply_operator(&mut self, operator: (Operator, S));
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprAtom<I, L> {
    Ident(I),
    Literal(L),
    LocationCounter,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operator {
    Unary(UnaryOperator),
    Binary(BinaryOperator),
    FnCall(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub enum UnaryOperator {
    Parentheses,
}

pub(crate) trait AssocIdent {
    type Ident;
}

pub(crate) trait AssocExpr: AssocIdent {
    type Literal;
}

pub(crate) trait AssocToken: AssocExpr {
    type Command;
}

pub(crate) trait ParamsContext<S: Clone>: AssocIdent + DelegateDiagnostics<S> {
    fn add_parameter(&mut self, param: (Self::Ident, S));
}

pub(crate) trait ToExprBody<S: Clone>: AssocExpr {
    type Parent;
    type Next: ExprContext<S, Ident = Self::Ident, Literal = Self::Literal>
        + FinalContext<ReturnTo = Self::Parent>;

    fn next(self) -> Self::Next;
}

pub(crate) trait ToMacroBody<S: Clone>: AssocToken {
    type Parent;
    type Next: TokenSeqContext<
        S,
        Token = Token<Self::Ident, Self::Literal, Self::Command>,
        Parent = Self::Parent,
    >;

    fn next(self) -> Self::Next;
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
