use token::Token;

use std::iter;
use std::str;

pub struct Lexer<'a> {
    src: &'a str,
    char_indices: iter::Peekable<str::CharIndices<'a>>,
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Token<'a>> {
        self.skip_horizontal_whitespace();
        match self.char_indices.peek() {
            None => None,
            Some(&(_, '\n')) => {
                self.advance();
                Some(Token::Eol)
            },
            _ => Some(self.lex_word())
        }
    }
}

impl<'a> Lexer<'a> {
    pub fn new(src: &str) -> Lexer {
        Lexer {
            src: src,
            char_indices: src.char_indices().peekable(),
        }
    }

    fn skip_horizontal_whitespace(&mut self) {
        loop {
            match self.current_char() {
                Some(c) if c.is_whitespace() && c != '\n' => self.advance(),
                _ => break,
            }
        }
    }

    fn current_char(&mut self) -> Option<char> {
        self.char_indices.peek().map(|&(_, c)| c)
    }

    fn advance(&mut self) {
        self.char_indices.next();
    }

    fn lex_word(&mut self) -> Token<'a> {
        let (start, _) = self.char_indices.next().unwrap();
        loop {
            match self.char_indices.peek() {
                None => return Token::Word(&self.src[start ..]),
                Some(&(end, c)) if c.is_whitespace() => return Token::Word(&self.src[start .. end]),
                _ => self.advance(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use token::Token::*;

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
    fn lex_word() {
        assert_eq_tokens("nop", &[Word("nop")])
    }

    #[test]
    fn lex_word_after_whitespace() {
        assert_eq_tokens("    nop", &[Word("nop")])
    }
}
