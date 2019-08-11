pub(super) use self::lexer::{LexError, Lexer};
pub(super) use self::parser::ParseTokenStream;

use self::parser::DefaultParser;

#[cfg(test)]
pub(super) use self::mock::*;

#[cfg(test)]
pub(super) use self::parser::mock::*;

pub mod actions;
mod lexer;
mod parser;

pub(super) trait ParserFactory<I, L, E, S: Clone> {
    type Parser: ParseTokenStream<I, L, E, S>;

    fn mk_parser(&mut self) -> Self::Parser;
}

pub(super) struct DefaultParserFactory;

impl<I, L, E, S: Clone> ParserFactory<I, L, E, S> for DefaultParserFactory {
    type Parser = DefaultParser;

    fn mk_parser(&mut self) -> Self::Parser {
        DefaultParser
    }
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

pub(super) trait IdentSource {
    type Ident: Clone + PartialEq + AsRef<str>;
}

pub(super) trait IdentFactory: IdentSource {
    fn mk_ident(&mut self, spelling: &str) -> Self::Ident;
}

#[cfg(test)]
impl<I: Clone + PartialEq + AsRef<str>, F: for<'a> Fn(&'a str) -> I> IdentSource for F {
    type Ident = I;
}

#[cfg(test)]
impl<I: Clone + PartialEq + AsRef<str>, F: for<'a> Fn(&'a str) -> I> IdentFactory for F {
    fn mk_ident(&mut self, spelling: &str) -> Self::Ident {
        self(spelling)
    }
}

// The following traits represent different positions within the grammar's production rules.

#[cfg(test)]
mod mock {
    use super::*;

    use super::actions::TokenStreamActions;

    use crate::log::Log;

    pub(in crate::analysis) struct MockParserFactory<T> {
        log: Log<T>,
    }

    impl<T> MockParserFactory<T> {
        pub fn new(log: Log<T>) -> Self {
            Self { log }
        }
    }

    impl<I, L, E, T, S: Clone> ParserFactory<I, L, E, S> for MockParserFactory<T>
    where
        T: From<ParserEvent<I, L, E, S>>,
    {
        type Parser = MockParser<T>;

        fn mk_parser(&mut self) -> Self::Parser {
            MockParser {
                log: self.log.clone(),
            }
        }
    }

    pub(in crate::analysis) struct MockParser<T> {
        log: Log<T>,
    }

    impl<I, L, E, T, S: Clone> ParseTokenStream<I, L, E, S> for MockParser<T>
    where
        T: From<ParserEvent<I, L, E, S>>,
    {
        fn parse_token_stream<R, A>(&mut self, tokens: R, actions: A) -> A
        where
            R: IntoIterator<Item = (Result<Token<I, L>, E>, S)>,
            A: TokenStreamActions<I, L, S>,
        {
            self.log
                .push(ParserEvent::ParseTokenStream(tokens.into_iter().collect()));
            actions
        }
    }

    #[derive(Debug, PartialEq)]
    pub(in crate::analysis) enum ParserEvent<I, L, E, S> {
        ParseTokenStream(Vec<TokenStreamItem<I, L, E, S>>),
    }

    type TokenStreamItem<I, L, E, S> = (Result<Token<I, L>, E>, S);
}
