use frontend::syntax::{Keyword, Token};

use std::iter;
use std::str;

pub struct Lexer<'a> {
    src: &'a str,
    char_indices: iter::Peekable<str::CharIndices<'a>>,
    is_at_line_start: bool,
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Token<'a>> {
        self.skip_irrelevant_characters();
        match self.char_indices.peek() {
            None => None,
            Some(&(index, _)) => Some(self.lex_token(index)),
        }
    }
}

impl<'a> Lexer<'a> {
    pub fn new(src: &str) -> Lexer {
        Lexer {
            src: src,
            char_indices: src.char_indices().peekable(),
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
            self.advance()
        }
    }

    fn current_char(&mut self) -> Option<char> {
        self.char_indices.peek().map(|&(_, c)| c)
    }

    fn advance(&mut self) {
        if self.current_char() != Some('\n') {
            self.is_at_line_start = false;
        }
        self.char_indices.next();
    }

    fn lex_token(&mut self, start: usize) -> Token<'a> {
        let first_char = self.current_char().unwrap();
        let next_token = match first_char {
            ']' => self.take(Token::ClosingBracket),
            ':' => self.take(Token::Colon),
            ',' => self.take(Token::Comma),
            '\n' => self.take(Token::Eol),
            '[' => self.take(Token::OpeningBracket),
            '$' => self.lex_number(),
            '"' => self.lex_quoted_string(),
            _ => self.lex_word(start),
        };
        if next_token == Token::Eol {
            self.is_at_line_start = true
        }
        next_token
    }

    fn take(&mut self, token: Token<'a>) -> Token<'a> {
        self.advance();
        token
    }

    fn lex_quoted_string(&mut self) -> Token<'a> {
        self.advance();
        let (start, first_char) = self.char_indices.next().unwrap();
        let mut end = start;
        let mut c = first_char;
        while c != '"' {
            let (next_index, next_char) = self.char_indices.next().unwrap();
            end = next_index;
            c = next_char;
        }
        Token::QuotedString(&self.src[start..end])
    }

    fn lex_number(&mut self) -> Token<'a> {
        self.advance();
        const RADIX: u32 = 16;
        let mut value = 0isize;
        let mut has_next_digit = true;
        while has_next_digit {
            if let Some(c) = self.current_char() {
                if let Some(digit) = c.to_digit(RADIX) {
                    value *= RADIX as isize;
                    value += digit as isize;
                    self.advance();
                } else {
                    has_next_digit = false;
                }
            } else {
                has_next_digit = false;
            }
        }
        Token::Number(value)
    }

    fn lex_word(&mut self, start: usize) -> Token<'a> {
        let starts_on_first_column = self.is_at_line_start;
        self.advance();
        let end = self.find_word_end();
        let word = &self.src[start..end];
        if let Some(keyword) = identify_keyword(word) {
            Token::Keyword(keyword)
        } else if starts_on_first_column {
            Token::Label(word)
        } else {
            Token::Identifier(word)
        }
    }

    fn find_word_end(&mut self) -> usize {
        while let Some(&(end, c)) = self.char_indices.peek() {
            if !c.is_alphanumeric() {
                return end;
            }
            self.advance()
        }
        self.src.len()
    }
}

fn is_horizontal_whitespace(character: char) -> bool {
    character.is_whitespace() && character != '\n'
}

fn identify_keyword(word: &str) -> Option<Keyword> {
    KEYWORDS
        .iter()
        .find(|&&(spelling, _)| spelling == word)
        .map(|&(_, keyword)| keyword)
}

const KEYWORDS: [(&'static str, Keyword); 24] = [
    ("a", Keyword::A),
    ("and", Keyword::And),
    ("b", Keyword::B),
    ("bc", Keyword::Bc),
    ("c", Keyword::C),
    ("cp", Keyword::Cp),
    ("d", Keyword::D),
    ("dec", Keyword::Dec),
    ("e", Keyword::E),
    ("endm", Keyword::Endm),
    ("h", Keyword::H),
    ("halt", Keyword::Halt),
    ("hl", Keyword::Hl),
    ("include", Keyword::Include),
    ("jr", Keyword::Jr),
    ("l", Keyword::L),
    ("ld", Keyword::Ld),
    ("macro", Keyword::Macro),
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
    use super::Token::*;

    fn assert_eq_tokens(src: &str, expected_tokens: &[Token]) {
        assert_eq!(Lexer::new(src).collect::<Vec<Token>>(), expected_tokens)
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
    fn lex_keywords() {
        for &(spelling, keyword) in KEYWORDS.iter() {
            assert_eq_tokens(spelling, &[Keyword(keyword)])
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
