use frontend::syntax::{self, token, Literal, Token, keyword::{self, Command, Operand}};

use std::iter;
use std::ops::{Index, Range};
use std::str;

#[derive(Clone, Copy, PartialEq)]
enum TokenKind {
    ClosingParenthesis,
    Colon,
    Comma,
    Eol,
    Ident,
    Number(Radix),
    OpeningParenthesis,
    String,
}

#[derive(Clone, Copy, PartialEq)]
enum Radix {
    Decimal,
    Hexadecimal,
}

struct Scanner<I: Iterator> {
    chars: iter::Peekable<I>,
    range: Range<usize>,
    is_at_line_start: bool,
}

impl<I: Iterator<Item = char>> Iterator for Scanner<I> {
    type Item = (TokenKind, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        self.skip_irrelevant_characters();
        if self.chars.peek().is_some() {
            self.range.start = self.range.end;
            Some(self.lex_token())
        } else {
            None
        }
    }
}

impl<I: Iterator<Item = char>> Scanner<I> {
    pub fn new(chars: I) -> Scanner<I> {
        Scanner {
            chars: chars.peekable(),
            range: Range { start: 0, end: 0 },
            is_at_line_start: true,
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
        self.chars.peek().cloned()
    }

    fn advance(&mut self) -> Option<char> {
        if self.current_char() != Some('\n') {
            self.is_at_line_start = false;
        }
        self.range.end += self.current_char().map_or(0, |ch| ch.len_utf8());
        self.chars.next()
    }

    fn lex_token(&mut self) -> <Self as Iterator>::Item {
        let first_char = self.current_char().unwrap();
        let next_token = match first_char {
            ')' => self.take(TokenKind::ClosingParenthesis),
            ':' => self.take(TokenKind::Colon),
            ',' => self.take(TokenKind::Comma),
            '\n' => self.take(TokenKind::Eol),
            '(' => self.take(TokenKind::OpeningParenthesis),
            '0'...'9' => self.lex_decimal_number(),
            '$' => self.lex_hex_number(),
            '"' => self.lex_quoted_string(),
            _ => self.lex_ident(),
        };
        (next_token, self.range.clone())
    }

    fn take(&mut self, token: TokenKind) -> TokenKind {
        self.advance();
        token
    }

    fn lex_decimal_number(&mut self) -> TokenKind {
        self.advance();
        self.skip_characters_if(is_dec_digit);
        TokenKind::Number(Radix::Decimal)
    }

    fn lex_hex_number(&mut self) -> TokenKind {
        self.advance();
        self.skip_characters_if(is_hex_digit);
        TokenKind::Number(Radix::Hexadecimal)
    }

    fn lex_quoted_string(&mut self) -> TokenKind {
        self.advance();
        self.skip_characters_if(|c| c != '"');
        self.advance();
        TokenKind::String
    }

    fn lex_ident(&mut self) -> TokenKind {
        self.advance();
        self.find_word_end();
        TokenKind::Ident
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

pub struct Lexer<'a> {
    src: &'a str,
    scanner: Scanner<str::Chars<'a>>,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Lexer<'a> {
        Lexer {
            src,
            scanner: Scanner::new(src.chars()),
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = (Token, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        self.scanner
            .next()
            .map(|(kind, range)| (mk_token(kind, self.src.index(range.clone())), range))
    }
}

fn mk_token(kind: TokenKind, lexeme: &str) -> Token {
    match kind {
        TokenKind::ClosingParenthesis => token::ClosingParenthesis,
        TokenKind::Colon => token::Colon,
        TokenKind::Comma => token::Comma,
        TokenKind::Eol => token::Eol,
        TokenKind::Ident => mk_keyword_or(token::Ident, lexeme),
        TokenKind::Number(Radix::Decimal) => {
            token::Literal(Literal::Number(i32::from_str_radix(lexeme, 10).unwrap()))
        }
        TokenKind::Number(Radix::Hexadecimal) => token::Literal(Literal::Number(
            i32::from_str_radix(&lexeme[1..], 16).unwrap(),
        )),
        TokenKind::OpeningParenthesis => token::OpeningParenthesis,
        TokenKind::String => {
            token::Literal(Literal::String(lexeme[1..(lexeme.len() - 1)].to_string()))
        }
    }
}

fn mk_keyword_or<F: FnOnce(String) -> Token>(f: F, lexeme: &str) -> Token {
    identify_keyword(lexeme).map_or_else(
        || f(lexeme.to_string()),
        |keyword| match keyword {
            Keyword::Command(command) => token::Command(command),
            Keyword::Endm => token::Endm,
            Keyword::Macro => token::Macro,
            Keyword::Operand(operand) => token::Literal(syntax::Literal::Operand(operand)),
        },
    )
}

fn identify_keyword(word: &str) -> Option<Keyword> {
    KEYWORDS
        .iter()
        .find(|&&(spelling, _)| spelling.eq_ignore_ascii_case(word))
        .map(|&(_, keyword)| keyword)
}

#[derive(Clone, Copy)]
enum Keyword {
    Command(keyword::Command),
    Endm,
    Macro,
    Operand(keyword::Operand),
}

const KEYWORDS: &[(&str, Keyword)] = &[
    ("a", Keyword::Operand(Operand::A)),
    ("af", Keyword::Operand(Operand::Af)),
    ("add", Keyword::Command(Command::Add)),
    ("and", Keyword::Command(Command::And)),
    ("b", Keyword::Operand(Operand::B)),
    ("bc", Keyword::Operand(Operand::Bc)),
    ("c", Keyword::Operand(Operand::C)),
    ("charmap", Keyword::Command(Command::Charmap)),
    ("cp", Keyword::Command(Command::Cp)),
    ("d", Keyword::Operand(Operand::D)),
    ("db", Keyword::Command(Command::Db)),
    ("de", Keyword::Operand(Operand::De)),
    ("dec", Keyword::Command(Command::Dec)),
    ("dw", Keyword::Command(Command::Dw)),
    ("e", Keyword::Operand(Operand::E)),
    ("endm", Keyword::Endm),
    ("h", Keyword::Operand(Operand::H)),
    ("halt", Keyword::Command(Command::Halt)),
    ("hl", Keyword::Operand(Operand::Hl)),
    ("include", Keyword::Command(Command::Include)),
    ("jp", Keyword::Command(Command::Jp)),
    ("jr", Keyword::Command(Command::Jr)),
    ("l", Keyword::Operand(Operand::L)),
    ("ld", Keyword::Command(Command::Ld)),
    ("macro", Keyword::Macro),
    ("nc", Keyword::Operand(Operand::Nc)),
    ("nop", Keyword::Command(Command::Nop)),
    ("nz", Keyword::Operand(Operand::Nz)),
    ("push", Keyword::Command(Command::Push)),
    ("sp", Keyword::Operand(Operand::Sp)),
    ("stop", Keyword::Command(Command::Stop)),
    ("xor", Keyword::Command(Command::Xor)),
    ("z", Keyword::Operand(Operand::Z)),
];

#[cfg(test)]
mod tests {
    use super::*;

    use super::Operand::*;
    use super::keyword::Command::*;
    use super::syntax::Literal::{Number, Operand};
    use super::syntax::token::*;

    fn assert_eq_tokens<'a>(src: &'a str, expected_tokens: &[Token]) {
        assert_eq!(
            Lexer::new(src).map(|(t, _)| t).collect::<Vec<_>>(),
            expected_tokens
        )
    }

    #[test]
    fn lex_empty_str() {
        assert_eq_tokens("", &[])
    }

    #[test]
    fn lex_eol() {
        assert_eq_tokens("\n", &[Eol])
    }

    #[test]
    fn lex_ident() {
        assert_eq_tokens("ident", &[Ident("ident".to_string())])
    }

    #[test]
    fn lex_ident_after_eol() {
        assert_eq_tokens("    \nident", &[Eol, Ident("ident".to_string())])
    }

    #[test]
    fn lex_ident_after_whitespace() {
        assert_eq_tokens("    ident", &[Ident("ident".to_string())])
    }

    #[test]
    fn lex_ident_with_underscore() {
        assert_eq_tokens(
            "ident_with_underscore",
            &[Ident("ident_with_underscore".to_string())],
        )
    }

    #[test]
    fn lex_two_keywords() {
        assert_eq_tokens("push bc", &[Command(Push), Literal(Operand(Bc))])
    }

    #[test]
    fn lex_comma() {
        assert_eq_tokens(",", &[Comma])
    }

    #[test]
    fn lex_quoted_string() {
        assert_eq_tokens(
            "\"file.asm\"",
            &[Literal(syntax::Literal::String("file.asm".to_string()))],
        )
    }

    #[test]
    fn lex_decimal_number() {
        assert_eq_tokens("1234", &[Literal(Number(1234))])
    }

    #[test]
    fn lex_hex_number() {
        assert_eq_tokens("$19af", &[Literal(Number(0x19af))])
    }

    #[test]
    fn lex_label() {
        assert_eq_tokens(
            "label: nop\n",
            &[
                Ident("label".to_string()),
                Colon,
                Command(keyword::Command::Nop),
                Eol,
            ],
        )
    }

    #[test]
    fn lex_keywords_lowercase() {
        lex_transformed_keywords(|k| k.to_lowercase())
    }

    #[test]
    fn lex_keywords_uppercase() {
        lex_transformed_keywords(|k| k.to_uppercase())
    }

    fn lex_transformed_keywords<F: Fn(&str) -> String>(f: F) {
        for &(spelling, keyword) in KEYWORDS.iter() {
            let token = match keyword {
                Keyword::Command(command) => Command(command),
                Keyword::Endm => Endm,
                Keyword::Macro => Macro,
                Keyword::Operand(operand) => Literal(Operand(operand)),
            };
            assert_eq_tokens(&f(spelling), &[token])
        }
    }

    #[test]
    fn lex_brackets() {
        assert_eq_tokens("()", &[OpeningParenthesis, ClosingParenthesis])
    }

    #[test]
    fn ignore_comment() {
        assert_eq_tokens("; comment", &[])
    }

    #[test]
    fn ignore_comment_at_end_of_line() {
        assert_eq_tokens("nop ; comment\n", &[Command(Nop), Eol])
    }

    #[test]
    fn ignore_comment_at_end_of_input() {
        assert_eq_tokens("nop ; comment", &[Command(Nop)])
    }
}
