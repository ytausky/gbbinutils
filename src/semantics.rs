use ast;
use keyword;
use syntax;

use token::Token;

use std::marker::PhantomData;

pub struct DefaultReduce<'a>(pub PhantomData<&'a ()>);

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

pub fn parse_mnemonic(spelling: &str) -> keyword::Mnemonic {
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

pub fn inst<'a>(mnemonic: keyword::Mnemonic, operands: &[ast::Operand]) -> ast::AsmItem<'a> {
    ast::AsmItem::Instruction(ast::Instruction::new(mnemonic, operands))
}

pub fn include(path: &str) -> ast::AsmItem {
    ast::AsmItem::Include(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    use syntax::Reduce;

    #[test]
    fn build_include_item() {
        let mut builder = DefaultReduce(PhantomData);
        let item = builder.reduce_command(Token::Word("include"), &[Token::QuotedString("file.asm")]);
        assert_eq!(item, include("file.asm"))
    }
}
