pub use self::lexer::{LexError, Lexer};
pub(crate) use self::parser::ParseTokenStream;

use std::fmt::Debug;

pub mod actions;
mod lexer;
pub mod parser;

pub type LexItem<R, S> = (Result<SemanticToken<R>, LexError>, S);
pub type SemanticToken<R> = crate::assembler::syntax::Token<R, Literal<R>>;

#[derive(Clone, Debug, PartialEq)]
pub enum Literal<R> {
    Number(i32),
    String(R),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token<I, L> {
    Ident(I),
    Label(I),
    Literal(L),
    Sigil(Sigil),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sigil {
    Comma,
    Dot,
    Eos,
    Eol,
    EqEq,
    LParen,
    Minus,
    Pipe,
    Plus,
    RParen,
    Slash,
    Star,
}

impl<I, L> From<Sigil> for Token<I, L> {
    fn from(sigil: Sigil) -> Self {
        Token::Sigil(sigil)
    }
}
