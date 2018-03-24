use keyword::Keyword;
use token::Token;

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
            is_at_line_start: true,
        }
    }

    fn skip_horizontal_whitespace(&mut self) {
        while let Some(c) = self.current_char() {
            if c.is_whitespace() && c != '\n' {
                self.is_at_line_start = false;
                self.advance()
            } else {
                break;
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
        let next_token = match first_char {
            ':' => Token::Colon,
            ',' => Token::Comma,
            '\n' => Token::Eol,
            '$' => self.lex_number(),
            '"' => self.lex_quoted_string(),
            _ => self.lex_word(start),
        };
        if next_token == Token::Eol {
            self.is_at_line_start = true
        }
        next_token
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
        Token::QuotedString(&self.src[start..end])
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
        let end = self.find_word_end();
        let word = &self.src[start..end];
        if let Some(keyword) = identify_keyword(word) {
            Token::Keyword(keyword)
        } else if self.is_at_line_start {
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

fn identify_keyword(word: &str) -> Option<Keyword> {
    use keyword::Keyword::*;
    match word {
        "a" => Some(A),
        "b" => Some(B),
        "bc" => Some(Bc),
        "endm" => Some(Endm),
        "halt" => Some(Halt),
        "include" => Some(Include),
        "ld" => Some(Ld),
        "macro" => Some(Macro),
        "nop" => Some(Nop),
        "push" => Some(Push),
        "stop" => Some(Stop),
        "xor" => Some(Xor),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use keyword::Keyword::*;
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
    fn lex_label() {
        assert_eq_tokens("label", &[Label("label")])
    }

    #[test]
    fn lex_label_after_eol() {
        assert_eq_tokens("    \nlabel", &[Eol, Label("label")])
    }

    #[test]
    fn lex_nop() {
        assert_eq_tokens("nop", &[Keyword(Nop)])
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
    fn lex_keyword_endm() {
        assert_eq_tokens("endm", &[Keyword(Endm)])
    }

    #[test]
    fn lex_keyword_a() {
        assert_eq_tokens("a", &[Keyword(A)])
    }

    #[test]
    fn lex_keyword_b() {
        assert_eq_tokens("b", &[Keyword(B)])
    }

    #[test]
    fn lex_keyword_include() {
        assert_eq_tokens("include", &[Keyword(Include)])
    }

    #[test]
    fn lex_keyword_halt() {
        assert_eq_tokens("halt", &[Keyword(Halt)])
    }

    #[test]
    fn lex_keyword_ld() {
        assert_eq_tokens("ld", &[Keyword(Ld)])
    }

    #[test]
    fn lex_keyword_stop() {
        assert_eq_tokens("stop", &[Keyword(Stop)])
    }

    #[test]
    fn lex_keyword_xor() {
        assert_eq_tokens("xor", &[Keyword(Xor)])
    }
}
