use ir;

use std;

mod ast;
mod semantics;
mod syntax;

use ir::*;
use self::ast::Expression;
use self::syntax::*;

pub fn analyze_file(name: &str) {
    use std::io::prelude::*;
    let mut file = std::fs::File::open(name).unwrap();
    let mut src = String::new();
    file.read_to_string(&mut src).unwrap();
    let mut ast_builder = AstBuilder::new(DumpSection::new());
    syntax::parse_src(
        syntax::lexer::Lexer::new(&src),
        &mut ast_builder,
        ast::ExprBuilder::new(),
    )
}

struct DumpSection;

impl DumpSection {
    fn new() -> DumpSection {
        DumpSection {}
    }
}

impl ir::Section for DumpSection {
    fn add_instruction(&mut self, instruction: ir::Instruction) {
        println!("{:?}", instruction)
    }

    fn add_label(&mut self, label: &str) {
        println!("Define symbol: {}", label)
    }
}

pub struct AstBuilder<'a, S: ir::Section> {
    ast: Vec<ast::AsmItem<'a>>,
    contexts: Vec<Context<'a>>,
    section: S,
}

enum Context<'a> {
    Block,
    Instruction(syntax::Token<'a>, Vec<ast::Expression<syntax::Token<'a>>>),
}

impl<'a, S: ir::Section> AstBuilder<'a, S> {
    pub fn new(section: S) -> AstBuilder<'a, S> {
        AstBuilder {
            ast: Vec::new(),
            contexts: vec![Context::Block],
            section: section,
        }
    }

    #[cfg(test)]
    fn ast(&self) -> &Vec<ast::AsmItem<'a>> {
        &self.ast
    }
}

impl<'a, S: Section> syntax::BlockContext for AstBuilder<'a, S> {
    type Terminal = Token<'a>;
    type Expr = Expression<Self::Terminal>;
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

impl<'a, S: Section> syntax::CommandContext for AstBuilder<'a, S> {
    type Terminal = Token<'a>;
    type Expr = Expression<Self::Terminal>;

    fn add_argument(&mut self, expr: Self::Expr) {
        match self.contexts.last_mut() {
            Some(&mut Context::Instruction(_, ref mut args)) => args.push(expr),
            _ => panic!(),
        }
    }

    fn exit_command(&mut self) {
        if let Some(Context::Instruction(name, args)) = self.contexts.pop() {
            match name {
                Token::Keyword(Keyword::Include) => self.ast.push(semantics::reduce_include(args)),
                Token::Keyword(keyword) => self.section
                    .add_instruction(semantics::interpret_instruction(keyword, args.into_iter())),
                _ => panic!(),
            }
        } else {
            panic!()
        }
    }
}

impl<'a, S: Section> syntax::TerminalSequenceContext for AstBuilder<'a, S> {
    type Terminal = Token<'a>;

    fn push_terminal(&mut self, _terminal: Self::Terminal) {
        unimplemented!()
    }

    fn exit_terminal_sequence(&mut self) {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use self::ast::ExprFactory;
    use frontend::semantics::*;

    #[test]
    fn build_include_item() {
        let filename = "file.asm";
        let (_, mut items) = analyze_command(Keyword::Include, &[Token::QuotedString(filename)]);
        let item = items.pop().unwrap();
        assert_eq!(item, include(filename))
    }

    #[test]
    fn parse_nop() {
        analyze_nullary_instruction(Keyword::Nop, Mnemonic::Nop)
    }

    #[test]
    fn parse_halt() {
        analyze_nullary_instruction(Keyword::Halt, Mnemonic::Halt)
    }

    #[test]
    fn parse_stop() {
        analyze_nullary_instruction(Keyword::Stop, Mnemonic::Stop)
    }

    #[test]
    fn analyze_ld_a_a() {
        let token_a = Token::Keyword(Keyword::A);
        let item = analyze_instruction(Keyword::Ld, &[token_a.clone(), token_a]);
        assert_eq!(item, inst(Mnemonic::Ld, &[A, A]))
    }

    const A: Operand = Operand::Alu(AluOperand::A);

    #[test]
    fn analyze_ld_a_b() {
        let token_a = Token::Keyword(Keyword::A);
        let token_b = Token::Keyword(Keyword::B);
        let item = analyze_instruction(Keyword::Ld, &[token_a, token_b]);
        assert_eq!(item, inst(Mnemonic::Ld, &[A, B]))
    }

    const B: Operand = Operand::Alu(AluOperand::B);

    #[test]
    fn analyze_xor_a() {
        let actions = analyze_instruction(Keyword::Xor, &[Token::Keyword(Keyword::A)]);
        assert_eq!(actions, inst(Mnemonic::Xor, &[A]))
    }

    #[test]
    fn analyze_xor_deref_hl() {
        let mut actions = Vec::new();
        {
            let mut builder = AstBuilder::new(TestSection::new(&mut actions));
            let command = builder.enter_command(Token::Keyword(Keyword::Xor));
            let mut expr_builder = ast::ExprBuilder::new();
            let atom = expr_builder.from_atom(Token::Keyword(Keyword::Hl));
            let expr = expr_builder.apply_deref(atom);
            command.add_argument(expr);
            command.exit_command()
        }
        assert_eq!(
            actions,
            inst(Mnemonic::Xor, &[Operand::Alu(AluOperand::DerefHl)])
        )
    }

    fn analyze_nullary_instruction(keyword: Keyword, mnemonic: Mnemonic) {
        let item = analyze_instruction(keyword, &[]);
        assert_eq!(item, inst(mnemonic, &[]))
    }

    fn inst(mnemonic: Mnemonic, operands: &[Operand]) -> TestActions {
        vec![
            Action::AddInstruction(instruction(mnemonic, operands.iter().cloned())),
        ]
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
                let mut expr_builder = ast::ExprBuilder::new();
                let expr = expr_builder.from_atom(arg.clone());
                builder.add_argument(expr);
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
        AddInstruction(Instruction),
    }

    struct TestSection<'a> {
        actions: &'a mut TestActions,
    }

    impl<'a> TestSection<'a> {
        fn new(actions: &'a mut TestActions) -> TestSection<'a> {
            TestSection { actions: actions }
        }
    }

    impl<'a> Section for TestSection<'a> {
        fn add_instruction(&mut self, instruction: Instruction) {
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
