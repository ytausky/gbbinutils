use frontend::syntax::{Keyword, TokenKind, StrToken};

use std::iter;
use std::ops::{Index, Range};
use std::str;

pub struct Scanner<'a> {
    src: &'a str,
    char_indices: iter::Peekable<str::CharIndices<'a>>,
    range: Range<usize>,
    is_at_line_start: bool,
}

impl<'a> Iterator for Scanner<'a> {
    type Item = (TokenKind, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        self.skip_irrelevant_characters();
        match self.char_indices.peek() {
            None => None,
            Some(&(index, _)) => {
                self.range = Range {
                    start: index,
                    end: index,
                };
                Some(self.lex_token())
            }
        }
    }
}

impl<'a> Scanner<'a> {
    pub fn new(src: &str) -> Scanner {
        Scanner {
            src,
            char_indices: src.char_indices().peekable(),
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
        self.char_indices.peek().map(|&(_, c)| c)
    }

    fn advance(&mut self) -> Option<(usize, char)> {
        if self.current_char() != Some('\n') {
            self.is_at_line_start = false;
        }
        self.range.end += self.current_char().map_or(0, |ch| ch.len_utf8());
        self.char_indices.next()
    }

    fn lex_token(&mut self) -> <Scanner<'a> as Iterator>::Item {
        let first_char = self.current_char().unwrap();
        let next_token = match first_char {
            ']' => self.take(TokenKind::ClosingBracket),
            ':' => self.take(TokenKind::Colon),
            ',' => self.take(TokenKind::Comma),
            '\n' => self.take(TokenKind::Eol),
            '[' => self.take(TokenKind::OpeningBracket),
            '$' => self.lex_number(),
            '"' => self.lex_quoted_string(),
            _ => self.lex_word(),
        };
        if next_token == TokenKind::Eol {
            self.is_at_line_start = true
        }
        (next_token, self.range.clone())
    }

    fn take(&mut self, token: TokenKind) -> TokenKind {
        self.advance();
        token
    }

    fn lex_quoted_string(&mut self) -> TokenKind {
        self.advance();
        let (_, first_char) = self.advance().unwrap();
        let mut c = first_char;
        while c != '"' {
            let (_, next_char) = self.advance().unwrap();
            c = next_char;
        }
        TokenKind::QuotedString
    }

    fn lex_number(&mut self) -> TokenKind {
        self.advance();
        const RADIX: u32 = 16;
        let mut has_next_digit = true;
        while has_next_digit {
            if let Some(c) = self.current_char() {
                if c.to_digit(RADIX).is_some() {
                    self.advance();
                } else {
                    has_next_digit = false;
                }
            } else {
                has_next_digit = false;
            }
        }
        TokenKind::Number
    }

    fn lex_word(&mut self) -> TokenKind {
        let starts_on_first_column = self.is_at_line_start;
        self.advance();
        self.find_word_end();
        if starts_on_first_column {
            TokenKind::Label
        } else {
            TokenKind::Identifier
        }
    }

    fn find_word_end(&mut self) -> usize {
        while let Some(&(end, c)) = self.char_indices.peek() {
            if !c.is_alphanumeric() && c != '_' {
                return end;
            }
            self.advance();
        }
        self.src.len()
    }
}

fn is_horizontal_whitespace(character: char) -> bool {
    character.is_whitespace() && character != '\n'
}

pub struct Lexer<'a> {
    src: &'a str,
    scanner: Scanner<'a>,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Lexer<'a> {
        Lexer {
            src,
            scanner: Scanner::new(src),
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = StrToken<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.scanner.next().map(|(token, range)| {
            let lexeme = self.src.index(range);
            match token {
                TokenKind::ClosingBracket => StrToken::ClosingBracket,
                TokenKind::Colon => StrToken::Colon,
                TokenKind::Comma => StrToken::Comma,
                TokenKind::Eol => StrToken::Eol,
                TokenKind::Identifier => identify_keyword(lexeme)
                    .map_or(StrToken::Identifier(lexeme), |keyword| {
                        StrToken::Keyword(keyword)
                    }),
                TokenKind::Keyword(_) => panic!(),
                TokenKind::Label => identify_keyword(lexeme)
                    .map_or(StrToken::Label(lexeme), |keyword| {
                        StrToken::Keyword(keyword)
                    }),
                TokenKind::Number => {
                    StrToken::Number(isize::from_str_radix(&lexeme[1..], 16).unwrap())
                }
                TokenKind::OpeningBracket => StrToken::OpeningBracket,
                TokenKind::QuotedString => {
                    StrToken::QuotedString(&lexeme[1..(lexeme.len() - 1)])
                }
                
            }
        })
    }
}

fn identify_keyword(word: &str) -> Option<Keyword> {
    let lowercase = word.to_lowercase();
    KEYWORDS
        .iter()
        .find(|&&(spelling, _)| spelling == lowercase)
        .map(|&(_, keyword)| keyword)
}

const KEYWORDS: [(&str, Keyword); 27] = [
    ("a", Keyword::A),
    ("and", Keyword::And),
    ("b", Keyword::B),
    ("bc", Keyword::Bc),
    ("c", Keyword::C),
    ("cp", Keyword::Cp),
    ("d", Keyword::D),
    ("db", Keyword::Db),
    ("dec", Keyword::Dec),
    ("e", Keyword::E),
    ("endm", Keyword::Endm),
    ("h", Keyword::H),
    ("halt", Keyword::Halt),
    ("hl", Keyword::Hl),
    ("include", Keyword::Include),
    ("jp", Keyword::Jp),
    ("jr", Keyword::Jr),
    ("l", Keyword::L),
    ("ld", Keyword::Ld),
    ("macro", Keyword::Macro),
    ("nc", Keyword::Nc),
    ("nop", Keyword::Nop),
    ("nz", Keyword::Nz),
    ("push", Keyword::Push),
    ("stop", Keyword::Stop),
    ("xor", Keyword::Xor),
    ("z", Keyword::Z),
];

#[cfg(test)]
mod tests {
    use super::*;

    use super::Keyword::*;
    use super::StrToken::*;

    fn assert_eq_tokens(src: &str, expected_tokens: &[StrToken]) {
        assert_eq!(Lexer::new(src).collect::<Vec<StrToken>>(), expected_tokens)
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
        assert_eq_tokens("label", &[Label("label")])
    }

    #[test]
    fn lex_label_after_eol() {
        assert_eq_tokens("    \nlabel", &[Eol, Label("label")])
    }

    #[test]
    fn lex_word_after_whitespace() {
        assert_eq_tokens("    word", &[Identifier("word")])
    }

    #[test]
    fn lex_label_and_word_with_underscore() {
        assert_eq_tokens(
            "first_label then_word",
            &[Label("first_label"), Identifier("then_word")],
        )
    }

    #[test]
    fn lex_two_keywords() {
        assert_eq_tokens("push bc", &[Keyword(Push), Keyword(Bc)])
    }

    #[test]
    fn lex_comma() {
        assert_eq_tokens(",", &[Comma])
    }

    #[test]
    fn lex_quoted_string() {
        assert_eq_tokens("\"file.asm\"", &[QuotedString("file.asm")])
    }

    #[test]
    fn lex_hex_number() {
        assert_eq_tokens("$19af", &[Number(0x19af)])
    }

    #[test]
    fn lex_macro_definition() {
        assert_eq_tokens("f: macro\n", &[Label("f"), Colon, Keyword(Macro), Eol])
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
            assert_eq_tokens(&f(spelling), &[Keyword(keyword)])
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
        assert_eq_tokens("nop ; comment\n", &[Keyword(Nop), Eol])
    }

    #[test]
    fn ignore_comment_at_end_of_input() {
        assert_eq_tokens("nop ; comment", &[Keyword(Nop)])
    }
}
