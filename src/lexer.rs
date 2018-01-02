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
        match self.char_indices.next() {
            None => None,
            Some((index, first_char)) => Some(self.lex_token(index, first_char)),
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
        while let Some(c) = self.current_char() {
            if c.is_whitespace() && c != '\n' {
                self.advance()
            } else {
                break
            }
        }
    }

    fn current_char(&mut self) -> Option<char> {
        self.char_indices.peek().map(|&(_, c)| c)
    }

    fn advance(&mut self) {
        self.char_indices.next();
    }

    fn lex_token(&mut self, start: usize, first_char: char) -> Token<'a> {
        if let Some(token) = lex_single_char_token(first_char) {
            token
        } else if first_char == '"' {
            self.lex_quoted_string()
        } else {
            self.lex_word(start)
        }
    }

    fn lex_quoted_string(&mut self) -> Token<'a> {
        let (start, first_char) = self.char_indices.next().unwrap();
        let mut end = start;
        let mut c = first_char;
        while c != '"' {
            let (next_index, next_char) = self.char_indices.next().unwrap();
            end = next_index;
            c = next_char;
        }
        Token::QuotedString(&self.src[start .. end])
    }

    fn lex_word(&mut self, start: usize) -> Token<'a> {
        loop {
            match self.char_indices.peek() {
                None => return Token::Word(&self.src[start ..]),
                Some(&(end, c)) if !c.is_alphanumeric() => return Token::Word(&self.src[start .. end]),
                _ => self.advance(),
            }
        }
    }
}

fn lex_single_char_token<'a>(c: char) -> Option<Token<'a>> {
    match c {
        ',' => Some(Token::Comma),
        '\n' => Some(Token::Eol),
        _ => None,
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

    #[test]
    fn lex_two_words() {
        assert_eq_tokens("push bc", &[Word("push"), Word("bc")])
    }

    #[test]
    fn lex_comma() {
        assert_eq_tokens(",", &[Comma])
    }

    #[test]
    fn lex_quoted_string() {
        assert_eq_tokens("\"file.asm\"", &[QuotedString("file.asm")])
    }
}
