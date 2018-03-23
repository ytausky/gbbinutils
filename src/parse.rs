use syntax::*;
use syntax::TerminalKind::*;

use std::iter;
use std::marker::PhantomData;

pub fn parse_src<'a, I, B>(tokens: I, block_context: &mut B)
    where I: Iterator<Item = B::Terminal>, B: BlockContext
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

impl<I, B> Parser<I, B> where B: BlockContext, I: Iterator<Item = B::Terminal> {
    fn terminal_kind(&mut self) -> Option<TerminalKind> {
        self.tokens.peek().map(|t| t.kind())
    }

    fn bump(&mut self) -> I::Item {
        self.tokens.next().unwrap()
    }

    fn parse_block(&mut self, block_context: &mut B) {
        while let Some(token) = self.next_token_if_not_block_delimiter() {
            self.parse_line(token, block_context)
        }
    }

    fn next_token_if_not_block_delimiter(&mut self) -> Option<I::Item> {
        let take_next = match self.tokens.peek() {
            Some(token) if token.kind() != Endm => true,
            _ => false,
        };
        if take_next {
            self.tokens.next()
        } else {
            None
        }
    }

    fn parse_line(&mut self, first_token: I::Item, block_context: &mut B) {
        if first_token.kind() == Label {
            self.parse_labeled_line(first_token, block_context)
        } else {
            self.parse_non_labeled_line(first_token, block_context)
        }
    }

    fn parse_labeled_line(&mut self, label: I::Item, block_context: &mut B) {
        if self.terminal_kind() == Some(Colon) {
            self.bump();
        }
        if self.terminal_kind() != None {
            let next_token = self.bump();
            if next_token.kind() == Macro {
                self.parse_macro_definition(label, block_context)
            } else {
                block_context.add_label(label);
                self.parse_non_labeled_line(next_token, block_context)
            }
        }
    }

    fn parse_non_labeled_line(&mut self, first_token: I::Item, block_context: &mut B) {
        match first_token.kind() {
            Eol => (),
            Word => self.parse_command(first_token, block_context),
            _ => panic!(),
        }
    }

    fn parse_macro_definition(&mut self, label: I::Item, block_context: &mut B) {
        let macro_block_context = block_context.enter_macro_definition(label);
        assert_eq!(self.tokens.next().unwrap().kind(), Eol);
        while let Some(token) = self.next_token_if_not_block_delimiter() {
            macro_block_context.push_terminal(token)
        }
        assert_eq!(self.tokens.next().unwrap().kind(), Endm);
        macro_block_context.exit_terminal_sequence()
    }

    fn parse_command(&mut self, first_token: I::Item, block_context: &mut B) {
        let instruction_context = block_context.enter_command(first_token);
        self.parse_argument_list(instruction_context);
        instruction_context.exit_command()
    }

    fn parse_argument_list(&mut self, instruction_context: &mut B::CommandContext) {
        if !self.outside_line() {
            self.parse_argument(instruction_context);
            while self.terminal_kind() == Some(Comma) {
                self.bump();
                self.parse_argument(instruction_context)
            }
        }
    }

    fn outside_line(&mut self) -> bool {
        match self.terminal_kind() {
            None | Some(Eol) => true,
            _ => false,
        }
    }

    fn parse_argument(&mut self, instruction_context: &mut B::CommandContext) {
        let expression_context = instruction_context.enter_argument();
        self.parse_expression(expression_context)
    }

    fn parse_expression<E>(&mut self, expression_context: &mut E)
        where E: ExpressionContext<Terminal = I::Item>
    {
        let token = self.bump();
        expression_context.push_atom(token);
        expression_context.exit_expression()
    }
}

#[cfg(test)]
mod tests {
    use super::parse_src;

    use syntax;
    use syntax::TerminalKind::*;

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
        AddLabel(TestToken),
        EnterExpression,
        EnterInstruction(TestToken),
        EnterMacroDef(TestToken),
        ExitExpression,
        ExitInstruction,
        ExitMacroDef,
        PushAtom(TestToken),
        PushTerminal(TestToken),
    }

    type TestToken = (syntax::TerminalKind, usize);

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

        fn enter_macro_definition(&mut self, label: Self::Terminal) -> &mut Self::TerminalSequenceContext {
            self.actions.push(Action::EnterMacroDef(label));
            self
        }
    }

    impl syntax::CommandContext for TestContext {
        type Terminal = TestToken;
        type ExpressionContext = Self;

        fn enter_argument(&mut self) -> &mut Self::ExpressionContext {
            self.actions.push(Action::EnterExpression);
            self
        }

        fn exit_command(&mut self) {
            self.actions.push(Action::ExitInstruction)
        }
    }

    impl syntax::ExpressionContext for TestContext {
        type Terminal = TestToken;

        fn push_atom(&mut self, atom: Self::Terminal) {
            self.actions.push(Action::PushAtom(atom))
        }

        fn exit_expression(&mut self) {
            self.actions.push(Action::ExitExpression)
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
        assert_eq_actions(&[(Word, 0), (Word, 1)], &inst((Word, 0), vec![expr(ident((Word, 1)))]))
    }

    fn expr(mut actions: Vec<Action>) -> Vec<Action> {
        let mut result = vec![Action::EnterExpression];
        result.append(&mut actions);
        result.push(Action::ExitExpression);
        result
    }

    fn ident(identifier: TestToken) -> Vec<Action> {
        vec![Action::PushAtom(identifier)]
    }

    #[test]
    fn parse_binary_instruction() {
        assert_eq_actions(&[(Word, 0), (Word, 1), (Comma, 2), (Word, 3)],
                          &inst((Word, 0), vec![expr(ident((Word, 1))), expr(ident((Word, 3)))]));
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = &[
            (Word, 0), (Word, 1), (Comma, 2), (Word, 3), (Eol, 4),
            (Word, 5), (Word, 6), (Comma, 7), (Word, 8),
        ];
        let expected_actions = &concat(vec![
            inst((Word, 0), vec![
                expr(ident((Word, 1))),
                expr(ident((Word, 3))),
            ]),
            inst((Word, 5), vec![
                expr(ident((Word, 6))),
                expr(ident((Word, 8))),
            ]),
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
            (Word, 0), (Word, 1), (Comma, 2), (Word, 3), (Eol, 4),
            (Eol, 5),
            (Word, 6), (Word, 7), (Comma, 8), (Word, 9),
        ];
        let expected_actions = &concat(vec![
            inst((Word, 0), vec![
                expr(ident((Word, 1))),
                expr(ident((Word, 3))),
            ]),
            inst((Word, 6), vec![
                expr(ident((Word, 7))),
                expr(ident((Word, 9))),
            ]),
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
        let tokens = &[
            (Label, 0), (Colon, 1), (Macro, 2), (Eol, 3),
            (Endm, 4),
        ];
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
            (Label, 0), (Colon, 1), (Macro, 2), (Eol, 3),
            (Word, 4), (Eol, 5),
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
}
