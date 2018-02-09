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
        match first_char {
            ',' => Token::Comma,
            '\n' => Token::Eol,
            '$' => self.lex_number(),
            '"' => self.lex_quoted_string(),
            _ => self.lex_word(start),
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

    fn lex_number(&mut self) -> Token<'a> {
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
        while let Some(&(end, c)) = self.char_indices.peek() {
            if !c.is_alphanumeric() {
                return Token::Word(&self.src[start .. end])
            }
            self.advance()
        }
        Token::Word(&self.src[start ..])
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

    #[test]
    fn lex_hex_number() {
        assert_eq_tokens("$19af", &[Number(0x19af)])
    }
}
