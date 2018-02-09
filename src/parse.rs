use std::marker::PhantomData;
use ast;
use keyword;

use token::Token;

use std::iter;
use std::vec;

trait Reduce<'a> {
    fn reduce_include(&mut self, path: &'a str);
    fn reduce_mnemonic(&mut self, mnemonic: keyword::Mnemonic, operands: &[ast::Operand]);
}

struct DefaultReduce<'a> {
    items: Vec<ast::AsmItem<'a>>,
}

impl<'a> Reduce<'a> for DefaultReduce<'a> {
    fn reduce_include(&mut self, path: &'a str) {
        self.items.push(include(path))
    }

    fn reduce_mnemonic(&mut self, mnemonic: keyword::Mnemonic, operands: &[ast::Operand]) {
        self.items.push(inst(mnemonic, operands))
    }
}

pub fn parse_src<'a, I: Iterator<Item = Token<'a>>>(tokens: I) -> vec::IntoIter<ast::AsmItem<'a>> {
    let mut reduce = DefaultReduce { items: vec![] };
    {
        let parser = Parser {
            tokens: tokens.peekable(),
            reduce: &mut reduce,
            phantom: PhantomData,
        };
        parser.parse();
    }
    reduce.items.into_iter()
}

struct Parser<'a, 'b, L: Iterator, R: 'b + Reduce<'a>> {
    tokens: iter::Peekable<L>,
    reduce: &'b mut R,
    phantom: PhantomData<&'a ()>
}

impl<'a, 'b, L: Iterator<Item = Token<'a>>, R: Reduce<'a>> Parser<'a, 'b, L, R> {
    fn next_word(&mut self) -> Option<Token<'a>> {
        self.tokens.next()
    }

    fn parse(mut self) {
        while self.tokens.peek().is_some() {
            self.parse_line()
        }
    }

    fn parse_line(&mut self) {
        match self.next_word() {
            Some(Token::Word(first_word)) => self.parse_nonempty_line(first_word),
            Some(Token::Eol) | None => (),
            _ => panic!()
        }
    }

    fn parse_nonempty_line(&mut self, first_word: &str) {
        match parse_mnemonic(first_word) {
            keyword::Mnemonic::Include => self.parse_include(),
            mnemonic => {
                let operands = &self.parse_operands();
                self.reduce.reduce_mnemonic(mnemonic, operands)
            },
        }
    }

    fn parse_include(&mut self) {
        match self.next_word().unwrap() {
            Token::QuotedString(include_path) => self.reduce.reduce_include(include_path),
            _ => unimplemented!(),
        }
    }

    fn parse_operands(&mut self) -> Vec<ast::Operand> {
        let mut operands = vec![];
        if let Some(&Token::Word(word)) = self.tokens.peek() {
            operands.push(parse_operand(word).unwrap());
            self.next_word();
            while let Some(&Token::Comma) = self.tokens.peek() {
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
