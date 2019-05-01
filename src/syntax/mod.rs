use crate::diag::DelegateDiagnostics;
use crate::model::BinOp;
use std::borrow::Borrow;

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
    Comma,
    Dot,
    Endm,
    Eof,
    Eol,
    LParen,
    Macro,
    Minus,
    Pipe,
    Plus,
    RParen,
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
    type LabelContext: ParamsContext<I, S, Next = Self::StmtContext>;
    type StmtContext: StmtContext<I, L, C, S, Parent = Self>;

    fn enter_labeled_stmt(self, label: (I, S)) -> Self::LabelContext;
    fn enter_unlabeled_stmt(self) -> Self::StmtContext;
}

pub(crate) trait StmtContext<I, L, C, S: Clone>: DelegateDiagnostics<S> + Sized {
    type CommandContext: CommandContext<S, Ident = I, Command = C, Literal = L, Parent = Self>;
    type MacroDefContext: TokenSeqContext<S, Token = Token<I, L, C>, Parent = Self>;
    type MacroCallContext: MacroCallContext<S, Token = Token<I, L, C>, Parent = Self>;
    type Parent;

    fn enter_command(self, command: (C, S)) -> Self::CommandContext;
    fn enter_macro_def(self, keyword: S) -> Self::MacroDefContext;
    fn enter_macro_call(self, name: (I, S)) -> Self::MacroCallContext;
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

pub(crate) trait ExprContext<S: Clone>: DelegateDiagnostics<S> {
    type Ident;
    type Literal;

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
    Binary(BinOp),
    FnCall(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub enum UnaryOperator {
    Parentheses,
}

pub(crate) trait ParamsContext<I, S: Clone>: DelegateDiagnostics<S> {
    type Next;

    fn add_parameter(&mut self, param: (I, S));
    fn next(self) -> Self::Next;
}

pub(crate) trait MacroCallContext<S: Clone>: DelegateDiagnostics<S> + Sized {
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
