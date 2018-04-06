use frontend::syntax::{Atom, Token, keyword::{Command, Operand}};

use std::iter;
use std::ops::{Index, Range};
use std::str;

#[derive(PartialEq)]
enum ScannerTokenKind {
    ClosingBracket,
    Colon,
    Comma,
    Eol,
    Identifier,
    Label,
    Number,
    OpeningBracket,
    String,
}

struct Scanner<I: Iterator> {
    chars: iter::Peekable<I>,
    range: Range<usize>,
    is_at_line_start: bool,
}

impl<I: Iterator<Item = char>> Iterator for Scanner<I> {
    type Item = (ScannerTokenKind, Range<usize>);

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
            ']' => self.take(ScannerTokenKind::ClosingBracket),
            ':' => self.take(ScannerTokenKind::Colon),
            ',' => self.take(ScannerTokenKind::Comma),
            '\n' => self.take(ScannerTokenKind::Eol),
            '[' => self.take(ScannerTokenKind::OpeningBracket),
            '$' => self.lex_number(),
            '"' => self.lex_quoted_string(),
            _ => self.lex_word(),
        };
        if next_token == ScannerTokenKind::Eol {
            self.is_at_line_start = true
        }
        (next_token, self.range.clone())
    }

    fn take(&mut self, token: ScannerTokenKind) -> ScannerTokenKind {
        self.advance();
        token
    }

    fn lex_quoted_string(&mut self) -> ScannerTokenKind {
        self.advance();
        self.skip_characters_if(|c| c != '"');
        self.advance();
        ScannerTokenKind::String
    }

    fn lex_number(&mut self) -> ScannerTokenKind {
        self.advance();
        self.skip_characters_if(is_hex_digit);
        ScannerTokenKind::Number
    }

    fn lex_word(&mut self) -> ScannerTokenKind {
        let starts_on_first_column = self.is_at_line_start;
        self.advance();
        self.find_word_end();
        if starts_on_first_column {
            ScannerTokenKind::Label
        } else {
            ScannerTokenKind::Identifier
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
    type Item = Token<String>;

    fn next(&mut self) -> Option<Self::Item> {
        self.scanner.next().map(|(token, range)| {
            let lexeme = self.src.index(range);
            match token {
                ScannerTokenKind::ClosingBracket => Token::ClosingBracket,
                ScannerTokenKind::Colon => Token::Colon,
                ScannerTokenKind::Comma => Token::Comma,
                ScannerTokenKind::Eol => Token::Eol,
                ScannerTokenKind::Identifier => {
                    mk_keyword_or(|x| Token::Atom(Atom::Ident(x.to_string())), lexeme)
                }
                ScannerTokenKind::Label => mk_keyword_or(|x| Token::Label(x.to_string()), lexeme),
                ScannerTokenKind::Number => Token::Atom(Atom::Number(
                    isize::from_str_radix(&lexeme[1..], 16).unwrap(),
                )),
                ScannerTokenKind::OpeningBracket => Token::OpeningBracket,
                ScannerTokenKind::String => {
                    Token::Atom(Atom::String(lexeme[1..(lexeme.len() - 1)].to_string()))
                }
            }
        })
    }
}

fn mk_keyword_or<'a, F>(f: F, lexeme: &'a str) -> Token<String>
where
    F: FnOnce(&'a str) -> Token<String>,
{
    identify_keyword(lexeme).map_or(f(lexeme), |keyword| match keyword {
        Keyword::Command(command) => Token::Command(command),
        Keyword::Endm => Token::Endm,
        Keyword::Macro => Token::Macro,
        Keyword::Operand(operand) => Token::Atom(Atom::Operand(operand)),
    })
}

fn identify_keyword(word: &str) -> Option<Keyword> {
    KEYWORDS
        .iter()
        .find(|&&(spelling, _)| spelling.eq_ignore_ascii_case(word))
        .map(|&(_, keyword)| keyword)
}

#[derive(Clone, Copy)]
enum Keyword {
    Command(Command),
    Endm,
    Macro,
    Operand(Operand),
}

const KEYWORDS: [(&str, Keyword); 28] = [
    ("a", Keyword::Operand(Operand::A)),
    ("and", Keyword::Command(Command::And)),
    ("b", Keyword::Operand(Operand::B)),
    ("bc", Keyword::Operand(Operand::Bc)),
    ("c", Keyword::Operand(Operand::C)),
    ("charmap", Keyword::Command(Command::Charmap)),
    ("cp", Keyword::Command(Command::Cp)),
    ("d", Keyword::Operand(Operand::D)),
    ("db", Keyword::Command(Command::Db)),
    ("dec", Keyword::Command(Command::Dec)),
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
    ("stop", Keyword::Command(Command::Stop)),
    ("xor", Keyword::Command(Command::Xor)),
    ("z", Keyword::Operand(Operand::Z)),
];

#[cfg(test)]
mod tests {
    use super::*;

    use super::Atom::{Ident, Number, Operand};
    use super::Command::*;
    use super::Operand::*;
    use super::Token::*;

    fn assert_eq_tokens<'a>(src: &'a str, expected_tokens: &[Token<String>]) {
        assert_eq!(
            Lexer::new(src).collect::<Vec<Token<String>>>(),
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
    fn lex_label() {
        assert_eq_tokens("label", &[Label("label".to_string())])
    }

    #[test]
    fn lex_label_after_eol() {
        assert_eq_tokens("    \nlabel", &[Eol, Label("label".to_string())])
    }

    #[test]
    fn lex_word_after_whitespace() {
        assert_eq_tokens("    word", &[Atom(Ident("word".to_string()))])
    }

    #[test]
    fn lex_label_and_word_with_underscore() {
        assert_eq_tokens(
            "first_label then_word",
            &[
                Label("first_label".to_string()),
                Atom(Ident("then_word".to_string())),
            ],
        )
    }

    #[test]
    fn lex_label_and_ident_with_hash() {
        assert_eq_tokens(
            "first#label then#word",
            &[
                Label("first#label".to_string()),
                Atom(Ident("then#word".to_string())),
            ],
        )
    }

    #[test]
    fn lex_two_keywords() {
        assert_eq_tokens("push bc", &[Command(Push), Atom(Operand(Bc))])
    }

    #[test]
    fn lex_comma() {
        assert_eq_tokens(",", &[Comma])
    }

    #[test]
    fn lex_quoted_string() {
        assert_eq_tokens(
            "\"file.asm\"",
            &[Atom(super::Atom::String("file.asm".to_string()))],
        )
    }

    #[test]
    fn lex_hex_number() {
        assert_eq_tokens("$19af", &[Atom(Number(0x19af))])
    }

    #[test]
    fn lex_macro_definition() {
        assert_eq_tokens("f: macro\n", &[Label("f".to_string()), Colon, Macro, Eol])
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
                Keyword::Operand(operand) => Atom(Operand(operand)),
            };
            assert_eq_tokens(&f(spelling), &[token])
        }
    }

    #[test]
    fn lex_brackets() {
        assert_eq_tokens("[]", &[OpeningBracket, ClosingBracket])
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
