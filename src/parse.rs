use std::marker::PhantomData;
use ast;
use keyword;

use syntax;
use syntax::Terminal;
use syntax::TerminalKind::*;
use token::Token;

use std::iter;
use std::vec;

struct DefaultReduce<'a>(PhantomData<&'a ()>);

impl<'a> syntax::Reduce for DefaultReduce<'a> {
    type Token = Token<'a>;
    type Item = ast::AsmItem<'a>;
    type Expr = Token<'a>;

    fn build_name_expr(&mut self, token: Token<'a>) -> Token<'a> {
        token
    }

    fn reduce_command(&mut self, name: Token<'a>, args: &[Self::Expr]) -> Self::Item {
        match name {
            Token::Word(spelling) => {
                match parse_mnemonic(spelling) {
                    keyword::Mnemonic::Include => self.reduce_include(args[0].clone()),
                    _ => self.reduce_mnemonic(name, args),
                }
            },
            _ => panic!(),
        }
    }
}

impl<'a> DefaultReduce<'a> {
    fn reduce_include(&mut self, path: Token<'a>) -> ast::AsmItem<'a> {
        match path {
            Token::QuotedString(path_str) => include(path_str),
            _ => panic!()
        }
    }

    fn reduce_mnemonic(&mut self, mnemonic: Token<'a>, operands: &[Token<'a>]) -> ast::AsmItem<'a> {
        let parsed_operands: Vec<ast::Operand> = operands.iter().map(|t| parse_operand(t).unwrap()).collect();
        match mnemonic {
            Token::Word(spelling) => inst(parse_mnemonic(spelling), &parsed_operands),
            _ => panic!()
        }
    }
}

pub fn parse_src<'a, I: Iterator<Item = Token<'a>>>(tokens: I) -> vec::IntoIter<ast::AsmItem<'a>> {
    let parser = Parser {
        tokens: tokens.peekable(),
        reduce: DefaultReduce(PhantomData),
    };
    parser.parse().into_iter()
}

struct Parser<L: Iterator, R: syntax::Reduce> {
    tokens: iter::Peekable<L>,
    reduce: R,
}

impl<L, R> Parser<L, R> where R: syntax::Reduce, L: Iterator<Item = R::Token> {
    fn next_word(&mut self) -> Option<R::Token> {
        self.tokens.next()
    }

    fn parse(mut self) -> Vec<R::Item> {
        let mut src = vec![];
        while let Some(token) = self.tokens.next() {
            if let Some(item) = self.parse_line(token) {
                src.push(item)
            }
        };
        src
    }

    fn parse_line(&mut self, first_token: R::Token) -> Option<R::Item> {
        match first_token.kind() {
            Word => Some(self.parse_nonempty_line(first_token)),
            Eol => None,
            _ => panic!()
        }
    }

    fn parse_nonempty_line(&mut self, first_token: R::Token) -> R::Item {
        let operands = self.parse_operands();
        self.reduce.reduce_command(first_token, &operands)
    }

    fn parse_operands(&mut self) -> Vec<R::Expr> {
        let mut operands = vec![];
        if let Some(_) = self.tokens.peek() {
            let first_word = self.tokens.next().unwrap();
            operands.push(self.reduce.build_name_expr(first_word));
            while let Some(Comma) = self.tokens.peek().map(|t| t.kind()) {
                self.next_word();
                let next_word = self.tokens.next().unwrap();
                operands.push(self.reduce.build_name_expr(next_word))
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

fn parse_operand<'a>(token: &Token<'a>) -> Option<ast::Operand> {
    match *token {
        Token::Word("a") => Some(ast::Operand::Register(keyword::Register::A)),
        Token::Word("b") => Some(ast::Operand::Register(keyword::Register::B)),
        Token::Word("bc") => Some(ast::Operand::RegisterPair(keyword::RegisterPair::Bc)),
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
