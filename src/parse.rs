use ast;
use keyword;
use lexer;

use std::iter;
use std::str;

pub fn parse_src(src: &str) -> Parser {
    Parser {
        lexer: lexer::Lexer::new(src).peekable(),
    }
}

pub struct Parser<'a> {
    lexer: iter::Peekable<lexer::Lexer<'a>>,
}

impl<'a> Iterator for Parser<'a> {
    type Item = ast::AsmItem<'a>;

    fn next(&mut self) -> Option<ast::AsmItem<'a>> {
        let mut parsed_line = None;
        while parsed_line.is_none() && self.lexer.peek().is_some() {
            parsed_line = self.parse_line()
        };
        parsed_line
    }
}

impl<'a> Parser<'a> {
    fn next_word(&mut self) -> Option<lexer::Token<'a>> {
        self.lexer.next()
    }

    fn parse_line(&mut self) -> Option<ast::AsmItem<'a>> {
        match self.next_word()? {
            lexer::Token::Word(first_word) => Some(self.parse_nonempty_line(first_word)),
            lexer::Token::Eol => None,
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
            lexer::Token::Word(include_path) => include(parse_include_path(include_path)),
            lexer::Token::Eol => unimplemented!(),
        }
    }

    fn parse_operands(&mut self) -> Vec<ast::Operand> {
        let mut operands = vec![];
        while let Some(lexer::Token::Word(word)) = self.next_word() {
            operands.push(parse_operand(word).unwrap())
        };
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

fn parse_include_path(path: &str) -> &str {
    &path[1 .. path.len() - 1]
}

fn parse_operand(src: &str) -> Option<ast::Operand> {
    let without_comma = if src.ends_with(',') {
        &src[0 .. src.len() - 1]
    } else {
        src
    };
    match without_comma {
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

    fn assert_ast_eq(src: &str, expected_ast: &[AsmItem]) {
        let actual = parse_src(src).collect::<Vec<AsmItem>>();
        assert_eq!(actual, expected_ast)
    }

    #[test]
    fn parse_empty_src() {
        assert_ast_eq("", &[])
    }

    #[test]
    fn parse_empty_line() {
        assert_ast_eq("\n", &[])
    }

    #[test]
    fn parse_nop() {
        parse_nullary_instruction("nop")
    }

    #[test]
    fn parse_nop_after_whitespace() {
        assert_ast_eq("    nop", &[inst(Nop, &[])])
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
        assert_ast_eq(src, &[inst(parse_mnemonic(src), &[])])
    }

    #[test]
    fn parse_push_bc() {
        assert_ast_eq("push bc", &[inst(Push, &[BC])])
    }

    #[test]
    fn parse_ld_a_a() {
        assert_ast_eq("ld a, a", &[inst(Ld, &[A, A])])
    }

    #[test]
    fn parse_ld_a_b() {
        assert_ast_eq("ld a, b", &[inst(Ld, &[A, B])])
    }

    #[test]
    fn parse_two_instructions() {
        assert_ast_eq("ld a, b\nld a, b", &[
            inst(Ld, &[A, B]),
            inst(Ld, &[A, B]),
        ])
    }

    #[test]
    fn parse_two_instructions_separated_by_blank_line() {
        assert_ast_eq("ld a, b\n\nld a, b", &[
            inst(Ld, &[A, B]),
            inst(Ld, &[A, B]),
        ])
    }

    #[test]
    fn parse_include() {
        assert_ast_eq("include \"file.asm\"", &[include("file.asm")])
    }
}
