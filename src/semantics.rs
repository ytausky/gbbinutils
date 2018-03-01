use ast;
use keyword;
use syntax;

use token::Token;

use std::marker::PhantomData;

pub struct DefaultReduce<'a>(pub PhantomData<&'a ()>);

impl<'a> DefaultReduce<'a> {
    pub fn new() -> DefaultReduce<'a> {
        DefaultReduce(PhantomData)
    }
}

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
                match identify_keyword(spelling).unwrap() {
                    IdentMeaning::Command(keyword::Mnemonic::Include) => reduce_include(args[0].clone()),
                    IdentMeaning::Command(other_command) => reduce_mnemonic(other_command, args),
                    _ => panic!(),
                }
            },
            _ => panic!(),
        }
    }
}

fn reduce_include<'a>(path: Token<'a>) -> ast::AsmItem<'a> {
    match path {
        Token::QuotedString(path_str) => include(path_str),
        _ => panic!()
    }
}

fn reduce_mnemonic<'a>(command: keyword::Mnemonic, operands: &[Token<'a>]) -> ast::AsmItem<'a> {
    let parsed_operands: Vec<ast::Operand> = operands.iter().map(|t| parse_operand(t).unwrap()).collect();
    inst(command, &parsed_operands)
}

enum IdentMeaning {
    Command(keyword::Mnemonic),
    Operand(ast::Operand),
}

fn identify_keyword(spelling: &str) -> Option<IdentMeaning> {
    use self::IdentMeaning::*;
    use keyword::Mnemonic;
    match spelling {
        "halt" => Some(Command(Mnemonic::Halt)),
        "include" => Some(Command(Mnemonic::Include)),
        "ld" => Some(Command(Mnemonic::Ld)),
        "nop" => Some(Command(Mnemonic::Nop)),
        "push" => Some(Command(Mnemonic::Push)),
        "stop" => Some(Command(Mnemonic::Stop)),
        "a" => Some(Operand(ast::Operand::Register(keyword::Register::A))),
        "b" => Some(Operand(ast::Operand::Register(keyword::Register::B))),
        "bc" => Some(Operand(ast::Operand::RegisterPair(keyword::RegisterPair::Bc))),
        _ => None
    }
}

fn parse_operand<'a>(token: &Token<'a>) -> Option<ast::Operand> {
    match *token {
        Token::Word(spelling) => match identify_keyword(spelling) {
            Some(IdentMeaning::Operand(operand)) => Some(operand),
            _ => panic!(),
        },
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
        let filename = "file.asm";
        let item = analyze_instruction("include", &[Token::QuotedString(filename)]);
        assert_eq!(item, include(filename))
    }

    #[test]
    fn parse_nop() {
        analyze_nullary_instruction("nop", keyword::Mnemonic::Nop)
    }

    #[test]
    fn parse_halt() {
        analyze_nullary_instruction("halt", keyword::Mnemonic::Halt)
    }

    #[test]
    fn parse_stop() {
        analyze_nullary_instruction("stop", keyword::Mnemonic::Stop)
    }

    #[test]
    fn analyze_push_bc() {
        let item = analyze_instruction("push", &[Token::Word("bc")]);
        assert_eq!(item, inst(keyword::Mnemonic::Push, &[ast::BC]))
    }

    fn analyze_nullary_instruction(name: &str, mnemonic: keyword::Mnemonic) {
        let item = analyze_instruction(name, &[]);
        assert_eq!(item, inst(mnemonic, &[]))
    }

    fn analyze_instruction<'a>(name: &'a str, operands: &[Token<'a>]) -> ast::AsmItem<'a> {
        let mut builder = DefaultReduce(PhantomData);
        builder.reduce_command(Token::Word(name), operands)
    }
}
