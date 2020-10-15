pub use self::lexer::{LexError, Lexer};
pub(crate) use self::parser::ParseTokenStream;

use super::string_ref::StringRef;

use std::fmt::Debug;

pub mod actions;
mod lexer;
pub mod parser;

pub type LexItem<S> = (Result<SemanticToken, LexError>, S);
pub type SemanticToken = crate::assembler::syntax::Token<StringRef, Literal>;

#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Number(i32),
    String(StringRef),
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
