pub(super) use self::lexer::{LexError, Lexer};

use crate::diag::Diagnostics;
use crate::model::BinOp;

mod lexer;
mod parser;

#[derive(Clone, Debug, PartialEq)]
pub enum Token<I, L> {
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
    Minus,
    Pipe,
    Plus,
    RParen,
    Slash,
    Star,
}

impl<I, L> From<SimpleToken> for Token<I, L> {
    fn from(simple: SimpleToken) -> Self {
        Token::Simple(simple)
    }
}

pub(super) trait IdentSource {
    type Ident: Clone + PartialEq;
}

pub(super) trait IdentFactory: IdentSource {
    fn mk_ident(&mut self, spelling: &str) -> Self::Ident;
}

#[cfg(test)]
impl<I: Clone + PartialEq, F: for<'a> Fn(&'a str) -> I> IdentSource for F {
    type Ident = I;
}

#[cfg(test)]
impl<I: Clone + PartialEq, F: for<'a> Fn(&'a str) -> I> IdentFactory for F {
    fn mk_ident(&mut self, spelling: &str) -> Self::Ident {
        self(spelling)
    }
}

pub(super) use self::parser::parse_src as parse_token_seq;

pub(super) trait FileContext<I, L, S: Clone>: Diagnostics<S> + Sized {
    type LabelContext: ParamsContext<I, S, Next = Self::StmtContext>;
    type StmtContext: StmtContext<I, L, S, Parent = Self>;

    fn enter_labeled_stmt(self, label: (I, S)) -> Self::LabelContext;
    fn enter_unlabeled_stmt(self) -> Self::StmtContext;
}

pub(super) trait StmtContext<I, L, S: Clone>: Diagnostics<S> + Sized {
    type Command;
    type MacroId;

    type CommandContext: CommandContext<S, Ident = I, Literal = L, Parent = Self>;
    type MacroDefContext: TokenSeqContext<S, Token = Token<I, L>, Parent = Self>;
    type MacroCallContext: MacroCallContext<S, Token = Token<I, L>, Parent = Self>;
    type Parent;

    fn key_lookup(&mut self, ident: I) -> KeyLookupResult<Self::Command, Self::MacroId>;

    fn enter_command(self, command: (Self::Command, S)) -> Self::CommandContext;
    fn enter_macro_def(self, keyword: S) -> Self::MacroDefContext;
    fn enter_macro_call(self, id: (Self::MacroId, S)) -> Self::MacroCallContext;
    fn exit(self) -> Self::Parent;
}

pub(super) type KeyLookupResult<C, M> = Result<Key<C, M>, KeyError>;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Key<C, M> {
    Command(C),
    Macro(M),
    Keyword(Keyword),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Keyword {
    Macro,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum KeyError {
    Reloc,
    Unknown,
}

#[cfg(test)]
impl<C, M> Key<C, M> {
    pub fn macro_call(self) -> Option<M> {
        match self {
            Key::Macro(id) => Some(id),
            _ => None,
        }
    }
}

pub(super) trait CommandContext<S: Clone>: Diagnostics<S> + Sized {
    type Ident;
    type Literal;
    type ArgContext: ExprContext<S, Ident = Self::Ident, Literal = Self::Literal>
        + FinalContext<ReturnTo = Self>;
    type Parent;
    fn add_argument(self) -> Self::ArgContext;
    fn exit(self) -> Self::Parent;
}

pub(super) trait FinalContext {
    type ReturnTo;

    fn exit(self) -> Self::ReturnTo;
}

pub(super) trait ExprContext<S: Clone>: Diagnostics<S> {
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

pub(super) trait ParamsContext<I, S: Clone>: Diagnostics<S> {
    type Next;

    fn add_parameter(&mut self, param: (I, S));
    fn next(self) -> Self::Next;
}

pub(super) trait MacroCallContext<S: Clone>: Diagnostics<S> + Sized {
    type Token;
    type Parent;
    type MacroArgContext: TokenSeqContext<S, Token = Self::Token, Parent = Self>;
    fn enter_macro_arg(self) -> Self::MacroArgContext;
    fn exit(self) -> Self::Parent;
}

pub(super) trait TokenSeqContext<S: Clone>: Diagnostics<S> {
    type Token;
    type Parent;
    fn push_token(&mut self, token: (Self::Token, S));
    fn exit(self) -> Self::Parent;
}
