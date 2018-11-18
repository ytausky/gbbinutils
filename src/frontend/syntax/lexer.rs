use crate::frontend::syntax::{
    self,
    keyword::{self as kw, *},
    Literal, Token,
};

use std::iter;
use std::ops::{Index, Range};
use std::str;

#[derive(Clone, Copy, PartialEq)]
enum TokenKind {
    ClosingParenthesis,
    Colon,
    Comma,
    Eof,
    Eol,
    Error(LexError),
    Ident,
    Label,
    Number(Radix),
    OpeningParenthesis,
    Plus,
    String,
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

struct Scanner<I: Iterator> {
    chars: iter::Peekable<I>,
    range: Range<usize>,
    is_at_line_start: bool,
    is_at_file_end: bool,
}

impl<I: Iterator<Item = char>> Iterator for Scanner<I> {
    type Item = (TokenKind, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        self.skip_irrelevant_characters();
        if self.chars.peek().is_some() {
            self.range.start = self.range.end;
            Some(self.lex_token())
        } else if !self.is_at_file_end {
            self.is_at_file_end = true;
            Some((TokenKind::Eof, self.range.end..self.range.end))
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
        self.chars.peek().cloned()
    }

    fn advance(&mut self) -> Option<char> {
        self.is_at_line_start = self.current_char() == Some('\n');
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
            '+' => self.take(TokenKind::Plus),
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
        self.skip_characters_if(|c| c != '"' && c != '\n');
        if self.current_char() == Some('"') {
            self.advance();
            TokenKind::String
        } else {
            TokenKind::Error(LexError::UnterminatedString)
        }
    }

    fn lex_ident(&mut self) -> TokenKind {
        let is_label = self.is_at_line_start;
        self.advance();
        self.find_word_end();
        if is_label {
            TokenKind::Label
        } else {
            TokenKind::Ident
        }
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
    type Item = (Token<String>, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        self.scanner
            .next()
            .map(|(kind, range)| (mk_token(kind, self.src.index(range.clone())), range))
    }
}

fn mk_token(kind: TokenKind, lexeme: &str) -> Token<String> {
    match kind {
        TokenKind::ClosingParenthesis => Token::ClosingParenthesis,
        TokenKind::Colon => Token::Colon,
        TokenKind::Comma => Token::Comma,
        TokenKind::Eof => Token::Eof,
        TokenKind::Eol => Token::Eol,
        TokenKind::Error(error) => Token::Error(error),
        TokenKind::Ident => mk_keyword_or(Token::Ident, lexeme),
        TokenKind::Label => mk_keyword_or(Token::Label, lexeme),
        TokenKind::Number(Radix::Decimal) => {
            Token::Literal(Literal::Number(i32::from_str_radix(lexeme, 10).unwrap()))
        }
        TokenKind::Number(Radix::Hexadecimal) => match i32::from_str_radix(&lexeme[1..], 16) {
            Ok(n) => Token::Literal(Literal::Number(n)),
            Err(_) => Token::Error(LexError::NoDigits),
        },
        TokenKind::OpeningParenthesis => Token::OpeningParenthesis,
        TokenKind::Plus => Token::Plus,
        TokenKind::String => {
            Token::Literal(Literal::String(lexeme[1..(lexeme.len() - 1)].to_string()))
        }
    }
}

fn mk_keyword_or<F: FnOnce(String) -> Token<String>>(f: F, lexeme: &str) -> Token<String> {
    identify_keyword(lexeme).map_or_else(
        || f(lexeme.to_string()),
        |keyword| match keyword {
            Keyword::Command(command) => Token::Command(command),
            Keyword::Endm => Token::Endm,
            Keyword::Macro => Token::Macro,
            Keyword::Operand(operand) => Token::Literal(syntax::Literal::Operand(operand)),
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
    Command(kw::Command),
    Endm,
    Macro,
    Operand(Operand),
}

impl From<kw::Command> for Keyword {
    fn from(command: kw::Command) -> Self {
        Keyword::Command(command)
    }
}

impl From<Operand> for Keyword {
    fn from(operand: Operand) -> Self {
        Keyword::Operand(operand)
    }
}

const KEYWORDS: &[(&str, Keyword)] = &[
    ("a", Keyword::Operand(Operand::A)),
    ("adc", Keyword::Command(Command::Mnemonic(Mnemonic::Adc))),
    ("add", Keyword::Command(Command::Mnemonic(Mnemonic::Add))),
    ("af", Keyword::Operand(Operand::Af)),
    ("and", Keyword::Command(Command::Mnemonic(Mnemonic::And))),
    ("b", Keyword::Operand(Operand::B)),
    ("bc", Keyword::Operand(Operand::Bc)),
    ("bit", Keyword::Command(Command::Mnemonic(Mnemonic::Bit))),
    ("c", Keyword::Operand(Operand::C)),
    ("call", Keyword::Command(Command::Mnemonic(Mnemonic::Call))),
    ("cp", Keyword::Command(Command::Mnemonic(Mnemonic::Cp))),
    ("cpl", Keyword::Command(Command::Mnemonic(Mnemonic::Cpl))),
    ("d", Keyword::Operand(Operand::D)),
    ("daa", Keyword::Command(Command::Mnemonic(Mnemonic::Daa))),
    ("db", Keyword::Command(Command::Directive(Directive::Db))),
    ("de", Keyword::Operand(Operand::De)),
    ("dec", Keyword::Command(Command::Mnemonic(Mnemonic::Dec))),
    ("di", Keyword::Command(Command::Mnemonic(Mnemonic::Di))),
    ("ds", Keyword::Command(Command::Directive(Directive::Ds))),
    ("dw", Keyword::Command(Command::Directive(Directive::Dw))),
    ("e", Keyword::Operand(Operand::E)),
    ("ei", Keyword::Command(Command::Mnemonic(Mnemonic::Ei))),
    ("endm", Keyword::Endm),
    ("h", Keyword::Operand(Operand::H)),
    ("halt", Keyword::Command(Command::Mnemonic(Mnemonic::Halt))),
    ("hl", Keyword::Operand(Operand::Hl)),
    ("hld", Keyword::Operand(Operand::Hld)),
    ("hli", Keyword::Operand(Operand::Hli)),
    ("inc", Keyword::Command(Command::Mnemonic(Mnemonic::Inc))),
    (
        "include",
        Keyword::Command(Command::Directive(Directive::Include)),
    ),
    ("jp", Keyword::Command(Command::Mnemonic(Mnemonic::Jp))),
    ("jr", Keyword::Command(Command::Mnemonic(Mnemonic::Jr))),
    ("l", Keyword::Operand(Operand::L)),
    ("ld", Keyword::Command(Command::Mnemonic(Mnemonic::Ld))),
    ("ldhl", Keyword::Command(Command::Mnemonic(Mnemonic::Ldhl))),
    ("macro", Keyword::Macro),
    ("nc", Keyword::Operand(Operand::Nc)),
    ("nop", Keyword::Command(Command::Mnemonic(Mnemonic::Nop))),
    ("nz", Keyword::Operand(Operand::Nz)),
    ("or", Keyword::Command(Command::Mnemonic(Mnemonic::Or))),
    ("org", Keyword::Command(Command::Directive(Directive::Org))),
    ("pop", Keyword::Command(Command::Mnemonic(Mnemonic::Pop))),
    ("push", Keyword::Command(Command::Mnemonic(Mnemonic::Push))),
    ("res", Keyword::Command(Command::Mnemonic(Mnemonic::Res))),
    ("ret", Keyword::Command(Command::Mnemonic(Mnemonic::Ret))),
    ("reti", Keyword::Command(Command::Mnemonic(Mnemonic::Reti))),
    ("rl", Keyword::Command(Command::Mnemonic(Mnemonic::Rl))),
    ("rla", Keyword::Command(Command::Mnemonic(Mnemonic::Rla))),
    ("rlc", Keyword::Command(Command::Mnemonic(Mnemonic::Rlc))),
    ("rlca", Keyword::Command(Command::Mnemonic(Mnemonic::Rlca))),
    ("rr", Keyword::Command(Command::Mnemonic(Mnemonic::Rr))),
    ("rra", Keyword::Command(Command::Mnemonic(Mnemonic::Rra))),
    ("rrc", Keyword::Command(Command::Mnemonic(Mnemonic::Rrc))),
    ("rrca", Keyword::Command(Command::Mnemonic(Mnemonic::Rrca))),
    ("rst", Keyword::Command(Command::Mnemonic(Mnemonic::Rst))),
    ("sbc", Keyword::Command(Command::Mnemonic(Mnemonic::Sbc))),
    ("set", Keyword::Command(Command::Mnemonic(Mnemonic::Set))),
    ("sla", Keyword::Command(Command::Mnemonic(Mnemonic::Sla))),
    ("sp", Keyword::Operand(Operand::Sp)),
    ("sra", Keyword::Command(Command::Mnemonic(Mnemonic::Sra))),
    ("srl", Keyword::Command(Command::Mnemonic(Mnemonic::Srl))),
    ("stop", Keyword::Command(Command::Mnemonic(Mnemonic::Stop))),
    ("sub", Keyword::Command(Command::Mnemonic(Mnemonic::Sub))),
    ("swap", Keyword::Command(Command::Mnemonic(Mnemonic::Swap))),
    ("xor", Keyword::Command(Command::Mnemonic(Mnemonic::Xor))),
    ("z", Keyword::Operand(Operand::Z)),
];

#[cfg(test)]
mod tests {
    use super::*;

    use super::kw::Mnemonic::*;
    use super::syntax::Literal::{Number, Operand};
    use super::syntax::Token::*;
    use super::Operand::*;
    use std::borrow::Borrow;

    #[test]
    fn range_of_eof_in_empty_str() {
        test_byte_range_at_eof("", [(Eof, 0..0)])
    }

    #[test]
    fn range_of_eof_after_ident() {
        test_byte_range_at_eof("    ident", [(Ident("ident".into()), 4..9), (Eof, 9..9)])
    }

    #[test]
    fn range_of_eof_after_trailing_whitespace() {
        test_byte_range_at_eof("    ident ", [(Ident("ident".into()), 4..9), (Eof, 10..10)])
    }

    fn test_byte_range_at_eof(src: &str, tokens: impl Borrow<[(Token<String>, Range<usize>)]>) {
        assert_eq!(Lexer::new(src).collect::<Vec<_>>(), tokens.borrow())
    }

    fn assert_eq_tokens<'a>(src: &'a str, expected_without_eof: impl Borrow<[Token<String>]>) {
        let mut expected: Vec<_> = expected_without_eof.borrow().iter().cloned().collect();
        expected.push(Eof);
        assert_eq!(
            Lexer::new(src).map(|(t, _)| t).collect::<Vec<_>>(),
            expected
        )
    }

    #[test]
    fn lex_empty_str() {
        assert_eq_tokens("", [])
    }

    #[test]
    fn lex_eol() {
        assert_eq_tokens("\n", [Eol])
    }

    #[test]
    fn lex_ident() {
        assert_eq_tokens("    ident", [Ident("ident".to_string())])
    }

    #[test]
    fn lex_ident_after_eol() {
        assert_eq_tokens("    \n    ident", [Eol, Ident("ident".to_string())])
    }

    #[test]
    fn lex_ident_with_underscore() {
        assert_eq_tokens(
            "    ident_with_underscore",
            [Ident("ident_with_underscore".to_string())],
        )
    }

    #[test]
    fn lex_two_keywords() {
        assert_eq_tokens("push bc", [Push.into(), Literal(Operand(Bc))])
    }

    #[test]
    fn lex_comma() {
        assert_eq_tokens(",", [Comma])
    }

    #[test]
    fn lex_quoted_string() {
        assert_eq_tokens(
            "\"file.asm\"",
            [Literal(syntax::Literal::String("file.asm".to_string()))],
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
        assert_eq_tokens("label nop\n", [Label("label".to_string()), Nop.into(), Eol])
    }

    #[test]
    fn lex_label_after_eol() {
        assert_eq_tokens("    \nlabel", [Eol, Label("label".to_string())])
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
            assert_eq_tokens(&f(spelling), [token])
        }
    }

    #[test]
    fn lex_brackets() {
        assert_eq_tokens("()", [OpeningParenthesis, ClosingParenthesis])
    }

    #[test]
    fn lex_plus() {
        assert_eq_tokens("+", [Plus])
    }

    #[test]
    fn ignore_comment() {
        assert_eq_tokens("; comment", [])
    }

    #[test]
    fn ignore_comment_at_end_of_line() {
        assert_eq_tokens("nop ; comment\n", [Nop.into(), Eol])
    }

    #[test]
    fn ignore_comment_at_end_of_input() {
        assert_eq_tokens("nop ; comment", [Nop.into()])
    }

    #[test]
    fn lex_unterminated_string() {
        assert_eq_tokens("\"unterminated", [Error(LexError::UnterminatedString)])
    }

    #[test]
    fn lex_number_without_digits() {
        assert_eq_tokens("$", [Error(LexError::NoDigits)])
    }

    impl<T: Into<kw::Command>> From<T> for Token<String> {
        fn from(t: T) -> Self {
            Command(t.into())
        }
    }
}
