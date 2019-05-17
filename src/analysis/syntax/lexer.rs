use super::keyword::*;
use super::{Command::*, Directive::*, Mnemonic::*, Operand::*, SimpleToken::*};
use super::{IdentFactory, SimpleToken, Token};
use crate::analysis::Literal;

use std::borrow::Borrow;
use std::ops::Range;
use std::str;

#[derive(Clone, Copy, PartialEq)]
enum TokenKind {
    Ident,
    Label,
    Number(Radix),
    Simple(SimpleToken),
    String,
}

impl From<SimpleToken> for TokenKind {
    fn from(simple: SimpleToken) -> Self {
        TokenKind::Simple(simple)
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
            Some((Ok(Eof.into()), self.range.end..self.range.end))
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

pub(in crate::analysis) struct Lexer<B, F> {
    scanner: Scanner<B>,
    mk_ident: F,
}

impl<B, F> Lexer<B, F>
where
    B: Borrow<str>,
    F: IdentFactory,
{
    pub fn new(src: B, mk_ident: F) -> Lexer<B, F> {
        Lexer {
            scanner: Scanner::new(src),
            mk_ident,
        }
    }

    fn mk_token(
        &self,
        kind: TokenKind,
        lexeme: &str,
    ) -> Result<Token<F::Ident, Literal<String>, Command>, LexError> {
        match kind {
            TokenKind::Ident => Ok(self.mk_keyword_or(Token::Ident, lexeme)),
            TokenKind::Label => Ok(self.mk_keyword_or(Token::Label, lexeme)),
            TokenKind::Number(Radix::Decimal) => Ok(Token::Literal(Literal::Number(
                i32::from_str_radix(lexeme, 10).unwrap(),
            ))),
            TokenKind::Number(Radix::Hexadecimal) => match i32::from_str_radix(&lexeme[1..], 16) {
                Ok(n) => Ok(Token::Literal(Literal::Number(n))),
                Err(_) => Err(LexError::NoDigits),
            },
            TokenKind::Simple(simple) => Ok(Token::Simple(simple)),
            TokenKind::String => Ok(Token::Literal(Literal::String(
                lexeme[1..(lexeme.len() - 1)].to_string(),
            ))),
        }
    }

    fn mk_keyword_or<G>(&self, g: G, lexeme: &str) -> Token<F::Ident, Literal<String>, Command>
    where
        G: FnOnce(F::Ident) -> Token<F::Ident, Literal<String>, Command>,
    {
        identify_keyword(lexeme).map_or_else(|| g(self.mk_ident.mk_ident(lexeme)), Into::into)
    }
}

type LexResult<I> = Result<Token<I, Literal<String>, Command>, LexError>;

impl<B: Borrow<str>, F: IdentFactory> Iterator for Lexer<B, F> {
    type Item = (LexResult<F::Ident>, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        self.scanner.next().map(|(result, range)| {
            (
                result.and_then(|kind| {
                    self.mk_token(kind, &self.scanner.src.borrow()[range.clone()])
                }),
                range,
            )
        })
    }
}

fn identify_keyword(word: &str) -> Option<Keyword> {
    KEYWORDS
        .iter()
        .find(|(spelling, _)| spelling.eq_ignore_ascii_case(word))
        .map(|(_, keyword)| *keyword)
}

#[derive(Clone, Copy)]
enum Keyword {
    Command(Command),
    Endm,
    Macro,
    Operand(Operand),
}

impl<I, R> From<Keyword> for Token<I, Literal<R>, Command> {
    fn from(keyword: Keyword) -> Self {
        match keyword {
            Keyword::Command(command) => Token::Command(command),
            Keyword::Endm => Endm.into(),
            Keyword::Macro => Macro.into(),
            Keyword::Operand(operand) => Token::Literal(Literal::Operand(operand)),
        }
    }
}

impl From<Command> for Keyword {
    fn from(command: Command) -> Self {
        Keyword::Command(command)
    }
}

impl From<Operand> for Keyword {
    fn from(operand: Operand) -> Self {
        Keyword::Operand(operand)
    }
}

const KEYWORDS: &[(&str, Keyword)] = &[
    ("a", Keyword::Operand(A)),
    ("adc", Keyword::Command(Mnemonic(Adc))),
    ("add", Keyword::Command(Mnemonic(Add))),
    ("af", Keyword::Operand(Af)),
    ("and", Keyword::Command(Mnemonic(And))),
    ("b", Keyword::Operand(B)),
    ("bc", Keyword::Operand(Bc)),
    ("bit", Keyword::Command(Mnemonic(Bit))),
    ("c", Keyword::Operand(C)),
    ("call", Keyword::Command(Mnemonic(Call))),
    ("cp", Keyword::Command(Mnemonic(Cp))),
    ("cpl", Keyword::Command(Mnemonic(Cpl))),
    ("d", Keyword::Operand(D)),
    ("daa", Keyword::Command(Mnemonic(Daa))),
    ("db", Keyword::Command(Directive(Db))),
    ("de", Keyword::Operand(De)),
    ("dec", Keyword::Command(Mnemonic(Dec))),
    ("di", Keyword::Command(Mnemonic(Di))),
    ("ds", Keyword::Command(Directive(Ds))),
    ("dw", Keyword::Command(Directive(Dw))),
    ("e", Keyword::Operand(E)),
    ("ei", Keyword::Command(Mnemonic(Ei))),
    ("endm", Keyword::Endm),
    ("equ", Keyword::Command(Directive(Equ))),
    ("h", Keyword::Operand(H)),
    ("halt", Keyword::Command(Mnemonic(Halt))),
    ("hl", Keyword::Operand(Hl)),
    ("hld", Keyword::Operand(Hld)),
    ("hli", Keyword::Operand(Hli)),
    ("inc", Keyword::Command(Mnemonic(Inc))),
    ("include", Keyword::Command(Directive(Include))),
    ("jp", Keyword::Command(Mnemonic(Jp))),
    ("jr", Keyword::Command(Mnemonic(Jr))),
    ("l", Keyword::Operand(L)),
    ("ld", Keyword::Command(Mnemonic(Ld))),
    ("ldhl", Keyword::Command(Mnemonic(Ldhl))),
    ("macro", Keyword::Macro),
    ("nc", Keyword::Operand(Nc)),
    ("nop", Keyword::Command(Mnemonic(Nop))),
    ("nz", Keyword::Operand(Nz)),
    ("or", Keyword::Command(Mnemonic(Or))),
    ("org", Keyword::Command(Directive(Org))),
    ("pop", Keyword::Command(Mnemonic(Pop))),
    ("push", Keyword::Command(Mnemonic(Push))),
    ("res", Keyword::Command(Mnemonic(Res))),
    ("ret", Keyword::Command(Mnemonic(Ret))),
    ("reti", Keyword::Command(Mnemonic(Reti))),
    ("rl", Keyword::Command(Mnemonic(Rl))),
    ("rla", Keyword::Command(Mnemonic(Rla))),
    ("rlc", Keyword::Command(Mnemonic(Rlc))),
    ("rlca", Keyword::Command(Mnemonic(Rlca))),
    ("rr", Keyword::Command(Mnemonic(Rr))),
    ("rra", Keyword::Command(Mnemonic(Rra))),
    ("rrc", Keyword::Command(Mnemonic(Rrc))),
    ("rrca", Keyword::Command(Mnemonic(Rrca))),
    ("rst", Keyword::Command(Mnemonic(Rst))),
    ("sbc", Keyword::Command(Mnemonic(Sbc))),
    ("section", Keyword::Command(Directive(Section))),
    ("set", Keyword::Command(Mnemonic(Set))),
    ("sla", Keyword::Command(Mnemonic(Sla))),
    ("sp", Keyword::Operand(Sp)),
    ("sra", Keyword::Command(Mnemonic(Sra))),
    ("srl", Keyword::Command(Mnemonic(Srl))),
    ("stop", Keyword::Command(Mnemonic(Stop))),
    ("sub", Keyword::Command(Mnemonic(Sub))),
    ("swap", Keyword::Command(Mnemonic(Swap))),
    ("xor", Keyword::Command(Mnemonic(Xor))),
    ("z", Keyword::Operand(Z)),
];

#[cfg(test)]
mod tests {
    use super::*;

    use super::Literal::{Number, Operand};
    use super::Token::*;
    use std::borrow::Borrow;

    #[test]
    fn range_of_eof_in_empty_str() {
        test_byte_range_at_eof("", [(Eof.into(), 0..0)])
    }

    #[test]
    fn range_of_eof_after_ident() {
        test_byte_range_at_eof(
            "    ident",
            [(Ident("ident".into()), 4..9), (Eof.into(), 9..9)],
        )
    }

    #[test]
    fn range_of_eof_after_trailing_whitespace() {
        test_byte_range_at_eof(
            "    ident ",
            [(Ident("ident".into()), 4..9), (Eof.into(), 10..10)],
        )
    }

    fn test_byte_range_at_eof(src: &str, tokens: impl Borrow<[(TestToken, Range<usize>)]>) {
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

    fn assert_eq_tokens(src: &str, expected_without_eof: impl Borrow<[TestToken]>) {
        assert_eq_lex_results(src, expected_without_eof.borrow().iter().cloned().map(Ok))
    }

    fn assert_eq_lex_results<I>(src: &str, expected_without_eof: I)
    where
        I: IntoIterator<Item = Result<TestToken, LexError>>,
    {
        let mut expected: Vec<_> = expected_without_eof.into_iter().collect();
        expected.push(Ok(Eof.into()));
        assert_eq!(
            Lexer::new(src, ToString::to_string)
                .map(|(t, _)| t)
                .collect::<Vec<_>>(),
            expected
        )
    }

    type TestToken = Token<String, crate::analysis::Literal<String>, super::Command>;

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
    fn lex_two_keywords() {
        assert_eq_tokens("push bc", [Push.into(), Literal(Operand(Bc))])
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
            [Label("label".into()), Nop.into(), Eol.into()],
        )
    }

    #[test]
    fn lex_label_after_eol() {
        assert_eq_tokens("    \nlabel", [Eol.into(), Label("label".into())])
    }

    #[test]
    fn lex_keywords_lowercase() {
        lex_transformed_keywords(str::to_lowercase)
    }

    #[test]
    fn lex_keywords_uppercase() {
        lex_transformed_keywords(str::to_uppercase)
    }

    fn lex_transformed_keywords<F: Fn(&str) -> String>(f: F) {
        for &(spelling, keyword) in KEYWORDS.iter() {
            let token = match keyword {
                Keyword::Command(command) => Command(command),
                Keyword::Endm => Endm.into(),
                Keyword::Macro => Macro.into(),
                Keyword::Operand(operand) => Literal(Operand(operand)),
            };
            assert_eq_tokens(&f(spelling), [token])
        }
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
    fn ignore_comment() {
        assert_eq_tokens("; comment", [])
    }

    #[test]
    fn ignore_comment_at_end_of_line() {
        assert_eq_tokens("nop ; comment\n", [Nop.into(), Eol.into()])
    }

    #[test]
    fn ignore_comment_at_end_of_input() {
        assert_eq_tokens("nop ; comment", [Nop.into()])
    }

    #[test]
    fn lex_unterminated_string() {
        assert_eq_lex_results("\"unterminated", vec![Err(LexError::UnterminatedString)])
    }

    #[test]
    fn lex_number_without_digits() {
        assert_eq_lex_results("$", vec![Err(LexError::NoDigits)])
    }

    impl<T: Into<super::Command>> From<T> for TestToken {
        fn from(t: T) -> Self {
            Command(t.into())
        }
    }
}
