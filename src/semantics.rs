use ast;
use keyword;
use syntax;

use keyword::Keyword;
use token::Token;

pub struct AstBuilder<'a, S: ast::Section> {
    ast: Vec<ast::AsmItem<'a>>,
    contexts: Vec<Context<'a>>,
    section: S,
}

enum Context<'a> {
    Block,
    Expression(Vec<Expression<'a>>),
    Instruction(Token<'a>, Vec<Expression<'a>>),
}

enum Expression<'a> {
    Atom(Token<'a>),
}

impl<'a, S: ast::Section> AstBuilder<'a, S> {
    pub fn new(section: S) -> AstBuilder<'a, S> {
        AstBuilder {
            ast: Vec::new(),
            contexts: vec![Context::Block],
            section: section,
        }
    }

    pub fn ast(&self) -> &Vec<ast::AsmItem<'a>> {
        &self.ast
    }
}

impl<'a, S: ast::Section> syntax::BlockContext for AstBuilder<'a, S> {
    type Terminal = Token<'a>;
    type CommandContext = Self;
    type TerminalSequenceContext = Self;

    fn add_label(&mut self, label: Self::Terminal) {
        match label {
            Token::Label(spelling) => self.section.add_label(spelling),
            _ => panic!(),
        }
    }

    fn enter_command(&mut self, name: Self::Terminal) -> &mut Self::CommandContext {
        self.contexts.push(Context::Instruction(name, vec![]));
        self
    }

    fn enter_macro_definition(
        &mut self,
        _label: Self::Terminal,
    ) -> &mut Self::TerminalSequenceContext {
        unimplemented!()
    }
}

impl<'a, S: ast::Section> syntax::CommandContext for AstBuilder<'a, S> {
    type Terminal = Token<'a>;
    type ExpressionContext = Self;

    fn enter_argument(&mut self) -> &mut Self::ExpressionContext {
        self.contexts.push(Context::Expression(Vec::new()));
        self
    }

    fn exit_command(&mut self) {
        if let Some(Context::Instruction(name, args)) = self.contexts.pop() {
            match name {
                Token::Keyword(Keyword::Include) => self.ast.push(reduce_include(args)),
                Token::Keyword(keyword) => self.section
                    .add_instruction(reduce_mnemonic(keyword, args.into_iter())),
                _ => panic!(),
            }
        } else {
            panic!()
        }
    }
}

impl<'a, S: ast::Section> syntax::ExpressionContext for AstBuilder<'a, S> {
    type Terminal = Token<'a>;

    fn apply_deref(&mut self) {
        unimplemented!()
    }

    fn push_atom(&mut self, atom: Self::Terminal) {
        if let Some(&mut Context::Expression(ref mut stack)) = self.contexts.last_mut() {
            stack.push(Expression::Atom(atom))
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

impl<'a, S: ast::Section> syntax::TerminalSequenceContext for AstBuilder<'a, S> {
    type Terminal = Token<'a>;

    fn push_terminal(&mut self, _terminal: Self::Terminal) {
        unimplemented!()
    }

    fn exit_terminal_sequence(&mut self) {
        unimplemented!()
    }
}

fn reduce_include<'a>(mut arguments: Vec<Expression<'a>>) -> ast::AsmItem<'a> {
    assert_eq!(arguments.len(), 1);
    let path = arguments.pop().unwrap();
    match path {
        Expression::Atom(Token::QuotedString(path_str)) => include(path_str),
        _ => panic!(),
    }
}

fn reduce_mnemonic<'a, I>(command: keyword::Keyword, operands: I) -> ast::Instruction
where I: Iterator<Item = Expression<'a>>
{
    let parsed_operands: Vec<ast::Operand> = operands.map(parse_operand).collect();
    instruction(to_mnemonic(command), &parsed_operands)
}

fn parse_operand<'a>(expression: Expression<'a>) -> ast::Operand {
    match expression {
        Expression::Atom(Token::Keyword(keyword)) => parse_keyword_operand(keyword),
        _ => panic!(),
    }
}

fn parse_keyword_operand(keyword: Keyword) -> ast::Operand {
    match keyword {
        Keyword::A => ast::Operand::Register(ast::Register::A),
        Keyword::B => ast::Operand::Register(ast::Register::B),
        Keyword::Bc => ast::Operand::RegisterPair(ast::RegisterPair::Bc),
        _ => panic!(),
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
        Keyword::Xor => Mnemonic::Xor,
        _ => panic!(),
    }
}

fn instruction<'a>(mnemonic: ast::Mnemonic, operands: &[ast::Operand]) -> ast::Instruction {
    ast::Instruction::new(mnemonic, operands)
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
        let (_, mut items) = analyze_command(Keyword::Include, &[Token::QuotedString(filename)]);
        let item = items.pop().unwrap();
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

    #[test]
    fn analyze_xor_a() {
        let actions = analyze_instruction(Keyword::Xor, &[Token::Keyword(Keyword::A)]);
        assert_eq!(actions, inst(ast::Mnemonic::Xor, &[ast::A]))
    }

    fn analyze_nullary_instruction(keyword: Keyword, mnemonic: ast::Mnemonic) {
        let item = analyze_instruction(keyword, &[]);
        assert_eq!(item, inst(mnemonic, &[]))
    }

    fn inst(mnemonic: ast::Mnemonic, operands: &[ast::Operand]) -> TestActions {
        vec![Action::AddInstruction(instruction(mnemonic, operands))]
    }

    fn analyze_instruction<'a>(keyword: Keyword, operands: &[Token<'a>]) -> TestActions {
        analyze_command(keyword, operands).0
    }

    fn analyze_command<'a>(
        keyword: Keyword,
        operands: &[Token<'a>],
    ) -> (TestActions, Vec<ast::AsmItem<'a>>) {
        let mut instructions = Vec::new();
        let ast;
        {
            let mut builder = AstBuilder::new(TestSection::new(&mut instructions));
            builder.enter_command(Token::Keyword(keyword));
            for arg in operands {
                let expr = builder.enter_argument();
                expr.push_atom(arg.clone());
                expr.exit_expression();
            }
            builder.exit_command();
            ast = builder.ast().to_vec();
        }
        (instructions, ast)
    }

    type TestActions = Vec<Action>;

    #[derive(Debug, PartialEq)]
    enum Action {
        AddLabel(String),
        AddInstruction(ast::Instruction),
    }

    struct TestSection<'a> {
        actions: &'a mut TestActions,
    }

    impl<'a> TestSection<'a> {
        fn new(actions: &'a mut TestActions) -> TestSection<'a> {
            TestSection { actions: actions }
        }
    }

    impl<'a> ast::Section for TestSection<'a> {
        fn add_instruction(&mut self, instruction: ast::Instruction) {
            self.actions.push(Action::AddInstruction(instruction))
        }

        fn add_label(&mut self, label: &str) {
            self.actions.push(Action::AddLabel(label.to_string()))
        }
    }

    #[test]
    fn analyze_label() {
        let mut actions = Vec::new();
        {
            let mut builder = AstBuilder::new(TestSection::new(&mut actions));
            builder.add_label(Token::Label("label"));
        }
        assert_eq!(actions, vec![Action::AddLabel("label".to_string())])
    }
}
