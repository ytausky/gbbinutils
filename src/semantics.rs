use ast;
use keyword;
use syntax;

use keyword::Keyword;
use token::Token;

pub struct AstBuilder<'a> {
    ast: Vec<ast::AsmItem<'a>>,
    contexts: Vec<Context<'a>>,
}

enum Context<'a> {
    Block,
    Expression(Vec<Token<'a>>),
    Instruction(Token<'a>, Vec<Token<'a>>),
}

impl<'a> AstBuilder<'a> {
    pub fn new<S: ast::Section>(_section: S) -> AstBuilder<'a> {
        AstBuilder {
            ast: Vec::new(),
            contexts: vec![Context::Block],
        }
    }

    pub fn ast(&self) -> &Vec<ast::AsmItem<'a>> {
        &self.ast
    }
}

impl<'a> syntax::BlockContext for AstBuilder<'a> {
    type Terminal = Token<'a>;
    type CommandContext = Self;
    type TerminalSequenceContext = Self;

    fn enter_command(&mut self, name: Self::Terminal) -> &mut Self::CommandContext {
        self.contexts.push(Context::Instruction(name, vec![]));
        self
    }

    fn enter_macro_definition(&mut self, _label: Self::Terminal) -> &mut Self::TerminalSequenceContext {
        unimplemented!()
    }
}

impl<'a> syntax::CommandContext for AstBuilder<'a> {
    type Terminal = Token<'a>;
    type ExpressionContext = Self;

    fn enter_argument(&mut self) -> &mut Self::ExpressionContext {
        self.contexts.push(Context::Expression(Vec::new()));
        self
    }

    fn exit_command(&mut self) {
        if let Some(Context::Instruction(name, args)) = self.contexts.pop() {
            let item = match name {
                Token::Keyword(Keyword::Include) => reduce_include(args[0].clone()),
                Token::Keyword(keyword) => reduce_mnemonic(keyword, &args),
                _ => panic!(),
            };
            self.ast.push(item)
        } else {
            panic!()
        }
    }
}

impl<'a> syntax::ExpressionContext for AstBuilder<'a> {
    type Terminal = Token<'a>;

    fn push_atom(&mut self, atom: Self::Terminal) {
        if let Some(&mut Context::Expression(ref mut stack)) = self.contexts.last_mut() {
            stack.push(atom)
        } else {
            panic!()
        }
    }

    fn exit_expression(&mut self) {
        if let Some(Context::Expression(mut stack)) = self.contexts.pop() {
            assert_eq!(stack.len(), 1);
            let expression = stack.pop().unwrap();
            match self.contexts.last_mut() {
                Some(&mut Context::Instruction(_, ref mut args)) => args.push(expression),
                _ => panic!(),
            }
        } else {
            panic!()
        }
    }
}

impl<'a> syntax::TerminalSequenceContext for AstBuilder<'a> {
    type Terminal = Token<'a>;

    fn push_terminal(&mut self, _terminal: Self::Terminal) {
        unimplemented!()
    }

    fn exit_terminal_sequence(&mut self) {
        unimplemented!()
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
        Keyword::A => Some(ast::Operand::Register(ast::Register::A)),
        Keyword::B => Some(ast::Operand::Register(ast::Register::B)),
        Keyword::Bc => Some(ast::Operand::RegisterPair(ast::RegisterPair::Bc)),
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

fn to_mnemonic(keyword: Keyword) -> ast::Mnemonic {
    use ast::Mnemonic;
    match keyword {
        Keyword::Halt => Mnemonic::Halt,
        Keyword::Ld => Mnemonic::Ld,
        Keyword::Nop => Mnemonic::Nop,
        Keyword::Push => Mnemonic::Push,
        Keyword::Stop => Mnemonic::Stop,
        _ => panic!(),
    }
}

fn inst<'a>(mnemonic: ast::Mnemonic, operands: &[ast::Operand]) -> ast::AsmItem<'a> {
    ast::AsmItem::Instruction(ast::Instruction::new(mnemonic, operands))
}

fn include(path: &str) -> ast::AsmItem {
    ast::AsmItem::Include(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    use keyword::Keyword;
    use syntax::*;

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let item = analyze_instruction(Keyword::Include, &[Token::QuotedString(filename)]);
        assert_eq!(item, include(filename))
    }

    #[test]
    fn parse_nop() {
        analyze_nullary_instruction(Keyword::Nop, ast::Mnemonic::Nop)
    }

    #[test]
    fn parse_halt() {
        analyze_nullary_instruction(Keyword::Halt, ast::Mnemonic::Halt)
    }

    #[test]
    fn parse_stop() {
        analyze_nullary_instruction(Keyword::Stop, ast::Mnemonic::Stop)
    }

    #[test]
    fn analyze_push_bc() {
        let item = analyze_instruction(Keyword::Push, &[Token::Keyword(Keyword::Bc)]);
        assert_eq!(item, inst(ast::Mnemonic::Push, &[ast::BC]))
    }

    #[test]
    fn analyze_ld_a_a() {
        let token_a = Token::Keyword(Keyword::A);
        let item = analyze_instruction(Keyword::Ld, &[token_a.clone(), token_a]);
        assert_eq!(item, inst(ast::Mnemonic::Ld, &[ast::A, ast::A]))
    }

    #[test]
    fn analyze_ld_a_b() {
        let token_a = Token::Keyword(Keyword::A);
        let token_b = Token::Keyword(Keyword::B);
        let item = analyze_instruction(Keyword::Ld, &[token_a, token_b]);
        assert_eq!(item, inst(ast::Mnemonic::Ld, &[ast::A, ast::B]))
    }

    fn analyze_nullary_instruction(keyword: Keyword, mnemonic: ast::Mnemonic) {
        let item = analyze_instruction(keyword, &[]);
        assert_eq!(item, inst(mnemonic, &[]))
    }

    fn analyze_instruction<'a>(keyword: Keyword, operands: &[Token<'a>]) -> ast::AsmItem<'a> {
        let mut builder = AstBuilder::new(TestSection::new());
        builder.enter_command(Token::Keyword(keyword));
        for arg in operands {
            let expr = builder.enter_argument();
            expr.push_atom(arg.clone());
            expr.exit_expression();
        }
        builder.exit_command();
        builder.ast.pop().unwrap()
    }

    struct TestSection;

    impl TestSection {
        fn new() -> TestSection {
            TestSection {}
        }
    }

    impl ast::Section for TestSection {
        fn add_instruction(&mut self, _instruction: ast::Instruction) {
            unimplemented!()
        }
    }
}
