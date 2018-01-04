use ast;
use keyword;

use token::Token;

use std::iter;

pub fn parse_src<'a, I: Iterator<Item = Token<'a>>>(tokens: I) -> Parser<I> {
    Parser {
        lexer: tokens.peekable(),
    }
}

pub struct Parser<L: Iterator> {
    lexer: iter::Peekable<L>,
}

impl<'a, L: Iterator<Item = Token<'a>>> Iterator for Parser<L> {
    type Item = ast::AsmItem<'a>;

    fn next(&mut self) -> Option<ast::AsmItem<'a>> {
        let mut parsed_line = None;
        while parsed_line.is_none() && self.lexer.peek().is_some() {
            parsed_line = self.parse_line()
        };
        parsed_line
    }
}

impl<'a, L: Iterator<Item = Token<'a>>> Parser<L> {
    fn next_word(&mut self) -> Option<Token<'a>> {
        self.lexer.next()
    }

    fn parse_line(&mut self) -> Option<ast::AsmItem<'a>> {
        match self.next_word()? {
            Token::Word(first_word) => Some(self.parse_nonempty_line(first_word)),
            Token::Eol => None,
            _ => panic!()
        }
    }

    fn parse_nonempty_line(&mut self, first_word: &str) -> ast::AsmItem<'a> {
        match parse_mnemonic(first_word) {
            keyword::Mnemonic::Include => self.parse_include(),
            mnemonic => inst(mnemonic, &self.parse_operands()),
        }
    }

    fn parse_include(&mut self) -> ast::AsmItem<'a> {
        match self.next_word().unwrap() {
            Token::QuotedString(include_path) => include(include_path),
            _ => unimplemented!(),
        }
    }

    fn parse_operands(&mut self) -> Vec<ast::Operand> {
        let mut operands = vec![];
        if let Some(&Token::Word(word)) = self.lexer.peek() {
            operands.push(parse_operand(word).unwrap());
            self.next_word();
            while let Some(&Token::Comma) = self.lexer.peek() {
                self.next_word();
                let next_operand = match self.next_word().unwrap() {
                    Token::Word(w) => w,
                    _ => panic!(),
                };
                operands.push(parse_operand(next_operand).unwrap())
            }
        }
        operands
    }
}

fn parse_mnemonic(spelling: &str) -> keyword::Mnemonic {
    use keyword::Mnemonic::*;
    match spelling {
        "halt" => Halt,
        "include" => Include,
        "ld" => Ld,
        "nop" => Nop,
        "push" => Push,
        "stop" => Stop,
        _ => unimplemented!(),
    }
}

fn parse_operand(src: &str) -> Option<ast::Operand> {
    match src {
        "a" => Some(ast::Operand::Register(keyword::Register::A)),
        "b" => Some(ast::Operand::Register(keyword::Register::B)),
        "bc" => Some(ast::Operand::RegisterPair(keyword::RegisterPair::Bc)),
        _ => None,
    }
}

fn inst<'a>(mnemonic: keyword::Mnemonic, operands: &[ast::Operand]) -> ast::AsmItem<'a> {
    ast::AsmItem::Instruction(ast::Instruction::new(mnemonic, operands))
}

fn include(path: &str) -> ast::AsmItem {
    ast::AsmItem::Include(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ast::*;

    use keyword::Mnemonic::*;
    use token::Token::*;

    fn assert_eq_ast(tokens: &[Token], expected_ast: &[AsmItem]) {
        let cloned_tokens = tokens.into_iter().cloned();
        let parsed_ast = parse_src(cloned_tokens).collect::<Vec<AsmItem>>();
        assert_eq!(parsed_ast, expected_ast)
    }

    #[test]
    fn parse_empty_src() {
        assert_eq_ast(&[], &[])
    }

    #[test]
    fn parse_empty_line() {
        assert_eq_ast(&[Eol], &[])
    }

    #[test]
    fn parse_nop() {
        parse_nullary_instruction("nop")
    }

    #[test]
    fn parse_halt() {
        parse_nullary_instruction("halt")
    }

    #[test]
    fn parse_stop() {
        parse_nullary_instruction("stop")
    }

    fn parse_nullary_instruction(src: &str) {
        assert_eq_ast(&[Word(src)], &[inst(parse_mnemonic(src), &[])])
    }

    #[test]
    fn parse_push_bc() {
        assert_eq_ast(&[Word("push"), Word("bc")], &[inst(Push, &[BC])])
    }

    #[test]
    fn parse_ld_a_a() {
        assert_eq_ast(
            &[Word("ld"), Word("a"), Comma, Word("a")],
            &[inst(Ld, &[A, A])]
        )
    }

    #[test]
    fn parse_ld_a_b() {
        assert_eq_ast(
            &[Word("ld"), Word("a"), Comma, Word("b")],
            &[inst(Ld, &[A, B])]
        )
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = &[
            Word("ld"), Word("a"), Comma, Word("b"), Eol,
            Word("ld"), Word("a"), Comma, Word("b"),
        ];
        assert_eq_ast(tokens, &[inst(Ld, &[A, B]), inst(Ld, &[A, B])])
    }

    #[test]
    fn parse_two_instructions_separated_by_blank_line() {
        let tokens = &[
            Word("ld"), Word("a"), Comma, Word("b"), Eol, Eol,
            Word("ld"), Word("a"), Comma, Word("b"),
        ];
        assert_eq_ast(tokens, &[inst(Ld, &[A, B]), inst(Ld, &[A, B])])
    }

    #[test]
    fn parse_include() {
        assert_eq_ast(&[Word("include"), QuotedString("file.asm")], &[include("file.asm")])
    }
}
