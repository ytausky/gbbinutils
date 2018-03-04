use ast;
use keyword;
use syntax;

use keyword::Keyword;
use token::Token;

use std::marker::PhantomData;

pub struct DefaultReduce<'a>(pub PhantomData<&'a ()>);

impl<'a> DefaultReduce<'a> {
    pub fn new() -> DefaultReduce<'a> {
        DefaultReduce(PhantomData)
    }
}

impl<T> syntax::Block for Vec<T> {
    type Item = T;

    fn new() -> Self {
        Vec::new()
    }

    fn push(&mut self, item: Self::Item) {
        self.push(item)
    }
}

impl<T: syntax::Terminal> syntax::Expr for T {
    type Terminal = T;

    fn from_terminal(terminal: Self::Terminal) -> Self {
        terminal
    }
}

impl<'a> syntax::ProductionRules for DefaultReduce<'a> {
    type Token = Token<'a>;
    type Item = ast::AsmItem<'a>;
    type Expr = Token<'a>;
    type Block = Vec<Self::Item>;

    fn define_macro(&mut self, _label: Token<'a>, _block: Self::Block) -> Self::Item {
        inst(keyword::Mnemonic::Nop, &[])
    }

    fn reduce_command(&mut self, name: Token<'a>, args: &[Self::Expr]) -> Self::Item {
        match name {
            Token::Keyword(Keyword::Include) => reduce_include(args[0].clone()),
            Token::Keyword(keyword) => reduce_mnemonic(keyword, args),
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

fn reduce_mnemonic<'a>(command: keyword::Keyword, operands: &[Token<'a>]) -> ast::AsmItem<'a> {
    let parsed_operands: Vec<ast::Operand> = operands.iter().map(|t| parse_operand(t).unwrap()).collect();
    inst(to_mnemonic(command), &parsed_operands)
}

fn identify_keyword(keyword: &Keyword) -> Option<ast::Operand> {
    match *keyword {
        Keyword::A => Some(ast::Operand::Register(keyword::Register::A)),
        Keyword::B => Some(ast::Operand::Register(keyword::Register::B)),
        Keyword::Bc => Some(ast::Operand::RegisterPair(keyword::RegisterPair::Bc)),
        _ => None
    }
}

fn parse_operand<'a>(token: &Token<'a>) -> Option<ast::Operand> {
    match *token {
        Token::Keyword(ref keyword) => match identify_keyword(keyword) {
            Some(operand) => Some(operand),
            _ => panic!(),
        },
        _ => None,
    }
}

fn to_mnemonic(keyword: Keyword) -> keyword::Mnemonic {
    use keyword::Mnemonic;
    match keyword {
        Keyword::Halt => Mnemonic::Halt,
        Keyword::Ld => Mnemonic::Ld,
        Keyword::Nop => Mnemonic::Nop,
        Keyword::Push => Mnemonic::Push,
        Keyword::Stop => Mnemonic::Stop,
        _ => panic!(),
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

    use keyword::Keyword;
    use syntax::ProductionRules;

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let item = analyze_instruction(Keyword::Include, &[Token::QuotedString(filename)]);
        assert_eq!(item, include(filename))
    }

    #[test]
    fn parse_nop() {
        analyze_nullary_instruction(Keyword::Nop, keyword::Mnemonic::Nop)
    }

    #[test]
    fn parse_halt() {
        analyze_nullary_instruction(Keyword::Halt, keyword::Mnemonic::Halt)
    }

    #[test]
    fn parse_stop() {
        analyze_nullary_instruction(Keyword::Stop, keyword::Mnemonic::Stop)
    }

    #[test]
    fn analyze_push_bc() {
        let item = analyze_instruction(Keyword::Push, &[Token::Keyword(Keyword::Bc)]);
        assert_eq!(item, inst(keyword::Mnemonic::Push, &[ast::BC]))
    }

    #[test]
    fn analyze_ld_a_a() {
        let token_a = Token::Keyword(Keyword::A);
        let item = analyze_instruction(Keyword::Ld, &[token_a.clone(), token_a]);
        assert_eq!(item, inst(keyword::Mnemonic::Ld, &[ast::A, ast::A]))
    }

    #[test]
    fn analyze_ld_a_b() {
        let token_a = Token::Keyword(Keyword::A);
        let token_b = Token::Keyword(Keyword::B);
        let item = analyze_instruction(Keyword::Ld, &[token_a, token_b]);
        assert_eq!(item, inst(keyword::Mnemonic::Ld, &[ast::A, ast::B]))
    }

    fn analyze_nullary_instruction(keyword: Keyword, mnemonic: keyword::Mnemonic) {
        let item = analyze_instruction(keyword, &[]);
        assert_eq!(item, inst(mnemonic, &[]))
    }

    fn analyze_instruction<'a>(keyword: Keyword, operands: &[Token<'a>]) -> ast::AsmItem<'a> {
        let mut builder = DefaultReduce(PhantomData);
        builder.reduce_command(Token::Keyword(keyword), operands)
    }
}
