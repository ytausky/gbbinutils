use super::*;
use self::TerminalKind::*;

use std::iter;
use std::marker::PhantomData;

type Lookahead = Option<TerminalKind>;

fn follows_line(lookahead: &Lookahead) -> bool {
    match *lookahead {
        None | Some(Eol) => true,
        _ => false,
    }
}

pub fn parse_src<I, B>(tokens: I, block_context: &mut B)
where
    I: Iterator<Item = B::Terminal>,
    B: BlockContext,
{
    let mut parser = Parser {
        tokens: tokens.peekable(),
        _phantom: PhantomData,
    };
    parser.parse_block(block_context)
}

struct Parser<I: Iterator, B: BlockContext> {
    tokens: iter::Peekable<I>,
    _phantom: PhantomData<B>,
}

impl<I, B> Parser<I, B>
where
    B: BlockContext,
    I: Iterator<Item = B::Terminal>,
{
    fn lookahead(&mut self) -> Lookahead {
        self.tokens.peek().map(|t| t.kind())
    }

    fn bump(&mut self) -> I::Item {
        self.tokens.next().unwrap()
    }

    fn expect(&mut self, expected: &Lookahead) -> I::Item {
        assert_eq!(self.lookahead(), *expected);
        self.bump()
    }

    fn parse_block(&mut self, block_context: &mut B) {
        self.parse_list(&Some(Eol), Option::is_none, |p| p.parse_line(block_context))
    }

    fn parse_line(&mut self, block_context: &mut B) {
        if self.lookahead() == Some(Label) {
            self.parse_labeled_line(block_context)
        } else {
            self.parse_unlabeled_line(block_context)
        }
    }

    fn parse_labeled_line(&mut self, block_context: &mut B) {
        let label = self.expect(&Some(Label));
        if self.lookahead() == Some(Colon) {
            self.bump();
        }
        if self.lookahead() == Some(Macro) {
            self.parse_macro_definition(label, block_context)
        } else {
            block_context.add_label(label);
            self.parse_unlabeled_line(block_context)
        }
    }

    fn parse_unlabeled_line(&mut self, block_context: &mut B) {
        match self.lookahead() {
            ref t if follows_line(t) => (),
            Some(Word) => self.parse_command(block_context),
            _ => panic!(),
        }
    }

    fn parse_macro_definition(&mut self, label: I::Item, block_context: &mut B) {
        self.expect(&Some(Macro));
        let macro_block_context = block_context.enter_macro_definition(label);
        self.expect(&Some(Eol));
        while self.lookahead() != Some(Endm) {
            macro_block_context.push_terminal(self.bump())
        }
        self.expect(&Some(Endm));
        macro_block_context.exit_terminal_sequence()
    }

    fn parse_command(&mut self, block_context: &mut B) {
        let first_token = self.expect(&Some(Word));
        let instruction_context = block_context.enter_command(first_token);
        self.parse_argument_list(instruction_context);
        instruction_context.exit_command()
    }

    fn parse_argument_list(&mut self, instruction_context: &mut B::CommandContext) {
        self.parse_list(&Some(Comma), follows_line, |p| {
            p.parse_argument(instruction_context)
        })
    }

    fn parse_list<F, P>(&mut self, delimiter: &Lookahead, mut follow: F, mut parser: P)
    where
        F: FnMut(&Lookahead) -> bool,
        P: FnMut(&mut Self),
    {
        let first_terminal = self.lookahead();
        if !follow(&first_terminal) {
            parser(self);
            while self.lookahead() == *delimiter {
                self.bump();
                parser(self)
            }
            assert!(follow(&self.lookahead()))
        }
    }

    fn parse_argument(&mut self, instruction_context: &mut B::CommandContext) {
        let expr = self.parse_expression();
        instruction_context.add_argument(expr)
    }

    fn parse_expression(&mut self) -> SynExpr<I::Item> {
        if self.lookahead() == Some(OpeningBracket) {
            self.parse_deref_expression()
        } else {
            let token = self.bump();
            SynExpr::from(token)
        }
    }

    fn parse_deref_expression(&mut self) -> SynExpr<I::Item> {
        self.expect(&Some(OpeningBracket));
        let expr = self.parse_expression();
        self.expect(&Some(ClosingBracket));
        expr.deref()
    }
}

#[cfg(test)]
mod tests {
    use super::parse_src;

    use frontend::syntax;
    use self::syntax::TerminalKind::*;

    #[test]
    fn parse_empty_src() {
        assert_eq_actions(&[], &[])
    }

    struct TestContext {
        actions: Vec<Action>,
    }

    impl TestContext {
        fn new() -> TestContext {
            TestContext {
                actions: Vec::new(),
            }
        }
    }

    #[derive(Debug, PartialEq)]
    enum Action {
        AddArgument(TestExpr),
        AddLabel(TestToken),
        EnterInstruction(TestToken),
        EnterMacroDef(TestToken),
        ExitInstruction,
        ExitMacroDef,
        PushTerminal(TestToken),
    }

    type TestToken = (syntax::TerminalKind, usize);
    type TestExpr = syntax::SynExpr<TestToken>;

    impl syntax::Terminal for TestToken {
        fn kind(&self) -> syntax::TerminalKind {
            let (ref terminal_kind, _) = *self;
            terminal_kind.clone()
        }
    }

    impl syntax::BlockContext for TestContext {
        type Terminal = TestToken;
        type CommandContext = Self;
        type TerminalSequenceContext = Self;

        fn add_label(&mut self, label: Self::Terminal) {
            self.actions.push(Action::AddLabel(label))
        }

        fn enter_command(&mut self, name: Self::Terminal) -> &mut Self::CommandContext {
            self.actions.push(Action::EnterInstruction(name));
            self
        }

        fn enter_macro_definition(
            &mut self,
            label: Self::Terminal,
        ) -> &mut Self::TerminalSequenceContext {
            self.actions.push(Action::EnterMacroDef(label));
            self
        }
    }

    impl syntax::CommandContext for TestContext {
        type Terminal = TestToken;

        fn add_argument(&mut self, expr: syntax::SynExpr<Self::Terminal>) {
            self.actions.push(Action::AddArgument(expr))
        }

        fn exit_command(&mut self) {
            self.actions.push(Action::ExitInstruction)
        }
    }

    impl syntax::TerminalSequenceContext for TestContext {
        type Terminal = TestToken;

        fn push_terminal(&mut self, terminal: Self::Terminal) {
            self.actions.push(Action::PushTerminal(terminal))
        }

        fn exit_terminal_sequence(&mut self) {
            self.actions.push(Action::ExitMacroDef)
        }
    }

    #[test]
    fn parse_empty_line() {
        assert_eq_actions(&[(Eol, 0)], &[])
    }

    fn assert_eq_actions(tokens: &[TestToken], expected_actions: &[Action]) {
        let mut parsing_constext = TestContext::new();
        parse_src(tokens.iter().cloned(), &mut parsing_constext);
        assert_eq!(parsing_constext.actions, expected_actions)
    }

    #[test]
    fn parse_nullary_instruction() {
        assert_eq_actions(&[(Word, 0)], &inst((Word, 0), vec![]))
    }

    #[test]
    fn parse_nullary_instruction_after_eol() {
        assert_eq_actions(&[(Eol, 0), (Word, 1)], &inst((Word, 1), vec![]))
    }

    fn inst(name: TestToken, args: Vec<Vec<Action>>) -> Vec<Action> {
        let mut result = vec![Action::EnterInstruction(name)];
        for mut arg in args {
            result.append(&mut arg);
        }
        result.push(Action::ExitInstruction);
        result
    }

    #[test]
    fn parse_nullary_instruction_followed_by_eol() {
        assert_eq_actions(&[(Word, 0), (Eol, 1)], &inst((Word, 0), vec![]))
    }

    #[test]
    fn parse_unary_instruction() {
        assert_eq_actions(
            &[(Word, 0), (Word, 1)],
            &inst((Word, 0), vec![expr(ident((Word, 1)))]),
        )
    }

    fn expr(expression: TestExpr) -> Vec<Action> {
        vec![Action::AddArgument(expression)]
    }

    fn ident(identifier: TestToken) -> TestExpr {
        syntax::SynExpr::Atom(identifier)
    }

    #[test]
    fn parse_binary_instruction() {
        assert_eq_actions(
            &[(Word, 0), (Word, 1), (Comma, 2), (Word, 3)],
            &inst(
                (Word, 0),
                vec![expr(ident((Word, 1))), expr(ident((Word, 3)))],
            ),
        );
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = &[
            (Word, 0),
            (Word, 1),
            (Comma, 2),
            (Word, 3),
            (Eol, 4),
            (Word, 5),
            (Word, 6),
            (Comma, 7),
            (Word, 8),
        ];
        let expected_actions = &concat(vec![
            inst(
                (Word, 0),
                vec![expr(ident((Word, 1))), expr(ident((Word, 3)))],
            ),
            inst(
                (Word, 5),
                vec![expr(ident((Word, 6))), expr(ident((Word, 8)))],
            ),
        ]);
        assert_eq_actions(tokens, expected_actions)
    }

    fn concat(actions: Vec<Vec<Action>>) -> Vec<Action> {
        let mut result = Vec::new();
        for mut vector in actions {
            result.append(&mut vector)
        }
        result
    }

    #[test]
    fn parse_two_instructions_separated_by_blank_line() {
        let tokens = &[
            (Word, 0),
            (Word, 1),
            (Comma, 2),
            (Word, 3),
            (Eol, 4),
            (Eol, 5),
            (Word, 6),
            (Word, 7),
            (Comma, 8),
            (Word, 9),
        ];
        let expected_actions = &concat(vec![
            inst(
                (Word, 0),
                vec![expr(ident((Word, 1))), expr(ident((Word, 3)))],
            ),
            inst(
                (Word, 6),
                vec![expr(ident((Word, 7))), expr(ident((Word, 9)))],
            ),
        ]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_include() {
        let tokens = &[(Word, 0), (QuotedString, 1)];
        let expected_actions = &inst((Word, 0), vec![expr(ident((QuotedString, 1)))]);
        assert_eq_actions(tokens, expected_actions);
    }

    #[test]
    fn parse_empty_macro_definition() {
        let tokens = &[(Label, 0), (Colon, 1), (Macro, 2), (Eol, 3), (Endm, 4)];
        let expected_actions = &macro_def((Label, 0), vec![]);
        assert_eq_actions(tokens, expected_actions);
    }

    fn macro_def(label: TestToken, tokens: Vec<TestToken>) -> Vec<Action> {
        let mut result = vec![Action::EnterMacroDef(label)];
        result.extend(tokens.into_iter().map(|t| Action::PushTerminal(t)));
        result.push(Action::ExitMacroDef);
        result
    }

    #[test]
    fn parse_macro_definition_with_instruction() {
        let tokens = &[
            (Label, 0),
            (Colon, 1),
            (Macro, 2),
            (Eol, 3),
            (Word, 4),
            (Eol, 5),
            (Endm, 6),
        ];
        let expected_actions = &macro_def((Label, 0), vec![(Word, 4), (Eol, 5)]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_label() {
        let tokens = &[(Label, 0), (Eol, 1)];
        let expected_actions = &add_label((Label, 0), vec![]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_label_colon() {
        let tokens = &[(Label, 0), (Colon, 1), (Eol, 2)];
        let expected_actions = &add_label((Label, 0), vec![]);
        assert_eq_actions(tokens, expected_actions)
    }

    #[test]
    fn parse_labeled_instruction() {
        let tokens = &[(Label, 0), (Colon, 1), (Word, 2), (Eol, 3)];
        let expected_actions = &add_label((Label, 0), inst((Word, 2), vec![]));
        assert_eq_actions(tokens, expected_actions)
    }

    fn add_label(label: TestToken, mut following_actions: Vec<Action>) -> Vec<Action> {
        let mut result = vec![Action::AddLabel(label)];
        result.append(&mut following_actions);
        result
    }

    #[test]
    fn parse_deref_operand() {
        let tokens = &[
            (Word, 0),
            (OpeningBracket, 1),
            (Word, 2),
            (ClosingBracket, 3),
        ];
        let expected_actions = &inst((Word, 0), vec![expr(deref(ident((Word, 2))))]);
        assert_eq_actions(tokens, expected_actions)
    }

    fn deref(expr: TestExpr) -> TestExpr {
        syntax::SynExpr::Deref(Box::new(expr))
    }
}
