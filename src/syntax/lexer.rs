use super::Sigil::*;
use super::{IdentFactory, Sigil, Token};

use crate::session::lex::Literal;

use std::borrow::Borrow;
use std::ops::Range;
use std::str;

#[derive(Clone, Copy, PartialEq)]
enum TokenKind {
    Ident,
    Label,
    Number(Radix),
    Sigil(Sigil),
    String,
}

impl From<Sigil> for TokenKind {
    fn from(sigil: Sigil) -> Self {
        TokenKind::Sigil(sigil)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Radix {
    Decimal,
    Hexadecimal,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LexError {
    NoDigits,
    UnterminatedString,
}

struct Scanner<B> {
    src: B,
    range: Range<usize>,
    is_at_line_start: bool,
    is_at_file_end: bool,
}

impl<B: Borrow<str>> Iterator for Scanner<B> {
    type Item = (Result<TokenKind, LexError>, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        self.skip_irrelevant_characters();
        if self.range.end < self.src.borrow().len() {
            self.range.start = self.range.end;
            Some(self.lex_token())
        } else if !self.is_at_file_end {
            self.is_at_file_end = true;
            Some((Ok(Eos.into()), self.range.end..self.range.end))
        } else {
            None
        }
    }
}

impl<B: Borrow<str>> Scanner<B> {
    pub fn new(src: B) -> Scanner<B> {
        Scanner {
            src,
            range: Range { start: 0, end: 0 },
            is_at_line_start: true,
            is_at_file_end: false,
        }
    }

    fn skip_irrelevant_characters(&mut self) {
        self.skip_characters_if(is_horizontal_whitespace);
        if self.current_char() == Some(';') {
            self.skip_characters_if(|c| c != '\n')
        }
    }

    fn skip_characters_if<P: FnMut(char) -> bool>(&mut self, mut predicate: P) {
        while self.current_char().map_or(false, &mut predicate) {
            self.advance();
        }
    }

    fn current_char(&mut self) -> Option<char> {
        self.src.borrow()[self.range.end..].chars().next()
    }

    fn advance(&mut self) -> Option<char> {
        let current = self.current_char();
        self.is_at_line_start = current == Some('\n');
        self.range.end += current.map_or(0, char::len_utf8);
        current
    }

    fn lex_token(&mut self) -> <Self as Iterator>::Item {
        let first_char = self.current_char().unwrap();
        let next_token = match first_char {
            ',' => self.take(Comma),
            '.' => self.take(Dot),
            '\n' => self.take(Eol),
            '=' => self.lex_eq_or_eq_eq(),
            '(' => self.take(LParen),
            '-' => self.take(Minus),
            '|' => self.take(Pipe),
            '+' => self.take(Plus),
            ')' => self.take(RParen),
            '/' => self.take(Slash),
            '*' => self.take(Star),
            '0'..='9' => self.lex_decimal_number(),
            '$' => self.lex_hex_number(),
            '"' => self.lex_quoted_string(),
            _ => self.lex_ident(),
        };
        (next_token, self.range.clone())
    }

    fn take(&mut self, token: impl Into<TokenKind>) -> Result<TokenKind, LexError> {
        self.advance();
        Ok(token.into())
    }

    fn lex_eq_or_eq_eq(&mut self) -> Result<TokenKind, LexError> {
        self.advance();
        match self.current_char() {
            Some('=') => self.take(EqEq),
            _ => Ok(TokenKind::Ident),
        }
    }

    fn lex_decimal_number(&mut self) -> Result<TokenKind, LexError> {
        self.advance();
        self.skip_characters_if(is_dec_digit);
        Ok(TokenKind::Number(Radix::Decimal))
    }

    fn lex_hex_number(&mut self) -> Result<TokenKind, LexError> {
        self.advance();
        self.skip_characters_if(is_hex_digit);
        Ok(TokenKind::Number(Radix::Hexadecimal))
    }

    fn lex_quoted_string(&mut self) -> Result<TokenKind, LexError> {
        self.advance();
        self.skip_characters_if(|c| c != '"' && c != '\n');
        if self.current_char() == Some('"') {
            self.advance();
            Ok(TokenKind::String)
        } else {
            Err(LexError::UnterminatedString)
        }
    }

    fn lex_ident(&mut self) -> Result<TokenKind, LexError> {
        let is_label = self.is_at_line_start;
        self.advance();
        self.find_word_end();
        Ok(if is_label {
            TokenKind::Label
        } else {
            TokenKind::Ident
        })
    }

    fn find_word_end(&mut self) {
        self.skip_characters_if(is_ident_continuation)
    }
}

fn is_horizontal_whitespace(character: char) -> bool {
    character.is_whitespace() && character != '\n'
}

fn is_ident_continuation(character: char) -> bool {
    character.is_alphanumeric() || character == '_' || character == '#'
}

fn is_dec_digit(character: char) -> bool {
    character.is_digit(10)
}

fn is_hex_digit(character: char) -> bool {
    character.is_digit(16)
}

pub struct Lexer<B, F> {
    scanner: Scanner<B>,
    factory: TokenFactory<F>,
}

struct TokenFactory<F> {
    ident_factory: F,
}

impl<F: IdentFactory> TokenFactory<F> {
    fn new(ident_factory: F) -> Self {
        Self { ident_factory }
    }

    fn mk_token(
        &mut self,
        kind: TokenKind,
        lexeme: &str,
    ) -> Result<Token<F::Ident, Literal<String>>, LexError> {
        match kind {
            TokenKind::Ident => Ok(Token::Ident(self.ident_factory.mk_ident(lexeme))),
            TokenKind::Label => Ok(Token::Label(self.ident_factory.mk_ident(lexeme))),
            TokenKind::Number(Radix::Decimal) => Ok(Token::Literal(Literal::Number(
                i32::from_str_radix(lexeme, 10).unwrap(),
            ))),
            TokenKind::Number(Radix::Hexadecimal) => match i32::from_str_radix(&lexeme[1..], 16) {
                Ok(n) => Ok(Token::Literal(Literal::Number(n))),
                Err(_) => Err(LexError::NoDigits),
            },
            TokenKind::Sigil(sigil) => Ok(Token::Sigil(sigil)),
            TokenKind::String => Ok(Token::Literal(Literal::String(
                lexeme[1..(lexeme.len() - 1)].to_string(),
            ))),
        }
    }
}

impl<B: Borrow<str>, F: IdentFactory> Lexer<B, F> {
    pub fn new(src: B, ident_factory: F) -> Lexer<B, F> {
        Lexer {
            scanner: Scanner::new(src),
            factory: TokenFactory::new(ident_factory),
        }
    }
}

type LexResult<I> = Result<Token<I, Literal<String>>, LexError>;

impl<B: Borrow<str>, F: IdentFactory> Iterator for Lexer<B, F> {
    type Item = (LexResult<F::Ident>, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        self.scanner.next().map(|(result, range)| {
            (
                result.and_then(|kind| {
                    self.factory
                        .mk_token(kind, &self.scanner.src.borrow()[range.clone()])
                }),
                range,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::Literal::Number;
    use super::Token::*;
    use std::borrow::Borrow;

    #[test]
    fn range_of_eos_in_empty_str() {
        test_byte_range_at_eos("", [(Eos.into(), 0..0)])
    }

    #[test]
    fn range_of_eos_after_ident() {
        test_byte_range_at_eos(
            "    ident",
            [(Ident("ident".into()), 4..9), (Eos.into(), 9..9)],
        )
    }

    #[test]
    fn range_of_eos_after_trailing_whitespace() {
        test_byte_range_at_eos(
            "    ident ",
            [(Ident("ident".into()), 4..9), (Eos.into(), 10..10)],
        )
    }

    fn test_byte_range_at_eos(src: &'static str, tokens: impl Borrow<[(TestToken, Range<usize>)]>) {
        let expected: Vec<_> = tokens
            .borrow()
            .iter()
            .cloned()
            .map(|(t, r)| (Ok(t), r))
            .collect();
        assert_eq!(
            Lexer::new(src, ToString::to_string).collect::<Vec<_>>(),
            expected
        )
    }

    fn assert_eq_tokens(src: &str, expected_without_eos: impl Borrow<[TestToken]>) {
        assert_eq_lex_results(src, expected_without_eos.borrow().iter().cloned().map(Ok))
    }

    fn assert_eq_lex_results<I>(src: &str, expected_without_eos: I)
    where
        I: IntoIterator<Item = Result<TestToken, LexError>>,
    {
        let mut expected: Vec<_> = expected_without_eos.into_iter().collect();
        expected.push(Ok(Eos.into()));
        assert_eq!(
            Lexer::new(src, ToString::to_string)
                .map(|(t, _)| t)
                .collect::<Vec<_>>(),
            expected
        )
    }

    type TestToken = Token<String, crate::session::lex::Literal<String>>;

    #[test]
    fn lex_empty_str() {
        assert_eq_tokens("", [])
    }

    #[test]
    fn lex_eol() {
        assert_eq_tokens("\n", [Eol.into()])
    }

    #[test]
    fn lex_ident() {
        assert_eq_tokens("    ident", [Ident("ident".into())])
    }

    #[test]
    fn lex_ident_after_eol() {
        assert_eq_tokens("    \n    ident", [Eol.into(), Ident("ident".into())])
    }

    #[test]
    fn lex_ident_with_underscore() {
        assert_eq_tokens(
            "    ident_with_underscore",
            [Ident("ident_with_underscore".into())],
        )
    }

    #[test]
    fn lex_comma() {
        assert_eq_tokens(",", [Comma.into()])
    }

    #[test]
    fn lex_quoted_string() {
        assert_eq_tokens(
            "\"file.asm\"",
            [Literal(super::Literal::String("file.asm".to_string()))],
        )
    }

    #[test]
    fn lex_decimal_number() {
        assert_eq_tokens("1234", [Literal(Number(1234))])
    }

    #[test]
    fn lex_hex_number() {
        assert_eq_tokens("$19af", [Literal(Number(0x19af))])
    }

    #[test]
    fn lex_label() {
        assert_eq_tokens(
            "label nop\n",
            [Label("label".into()), Ident("nop".into()), Eol.into()],
        )
    }

    #[test]
    fn lex_label_after_eol() {
        assert_eq_tokens("    \nlabel", [Eol.into(), Label("label".into())])
    }

    #[test]
    fn lex_brackets() {
        assert_eq_tokens("()", [LParen.into(), RParen.into()])
    }

    #[test]
    fn lex_plus() {
        assert_eq_tokens("+", [Plus.into()])
    }

    #[test]
    fn lex_minus() {
        assert_eq_tokens("-", [Minus.into()])
    }

    #[test]
    fn lex_slash() {
        assert_eq_tokens("/", [Slash.into()])
    }

    #[test]
    fn lex_star() {
        assert_eq_tokens("*", [Star.into()])
    }

    #[test]
    fn lex_pipe() {
        assert_eq_tokens("|", [Pipe.into()])
    }

    #[test]
    fn lex_dot() {
        assert_eq_tokens(".", [Dot.into()])
    }

    #[test]
    fn lex_eq() {
        assert_eq_tokens("=", [Ident("=".into())])
    }

    #[test]
    fn lex_eq_eq() {
        assert_eq_tokens("==", [EqEq.into()])
    }

    #[test]
    fn ignore_comment() {
        assert_eq_tokens("; comment", [])
    }

    #[test]
    fn ignore_comment_at_end_of_line() {
        assert_eq_tokens("+ ; comment\n", [Plus.into(), Eol.into()])
    }

    #[test]
    fn ignore_comment_at_end_of_input() {
        assert_eq_tokens("+ ; comment", [Plus.into()])
    }

    #[test]
    fn lex_unterminated_string() {
        assert_eq_lex_results("\"unterminated", vec![Err(LexError::UnterminatedString)])
    }

    #[test]
    fn lex_number_without_digits() {
        assert_eq_lex_results("$", vec![Err(LexError::NoDigits)])
    }
}
