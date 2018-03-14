use syntax::*;
use syntax::TerminalKind::*;

use std::iter;
use std::marker::PhantomData;

pub fn parse_src<'a, I, R>(tokens: I, reduce: &mut R)
    where I: Iterator<Item = R::Token>, R: ParsingContext, R::Token: Clone
{
    let mut parser = Parser {
        tokens: tokens.peekable(),
        _phantom: PhantomData,
    };
    parser.parse_block(reduce)
}

struct Parser<L: Iterator, R: ParsingContext> {
    tokens: iter::Peekable<L>,
    _phantom: PhantomData<R>,
}

impl<L, R> Parser<L, R> where R: ParsingContext, L: Iterator<Item = R::Token>, R::Token: Clone {
    fn parse_block(&mut self, reduce: &mut R) {
        while let Some(token) = self.next_token_if_not_block_delimiter() {
            self.parse_line(token, reduce)
        }
    }

    fn next_token_if_not_block_delimiter(&mut self) -> Option<R::Token> {
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

    fn parse_line(&mut self, first_token: R::Token, reduce: &mut R) {
        if first_token.kind() != Eol {
            self.parse_nonempty_line(first_token, reduce)
        }
    }

    fn parse_nonempty_line(&mut self, first_token: R::Token, reduce: &mut R) {
        if first_token.kind() == Label {
            self.parse_macro_definition(first_token, reduce)
        } else {
            reduce.enter_instruction(first_token.clone());
            self.parse_operands(reduce);
            reduce.exit_instruction()
        }
    }

    fn parse_macro_definition(&mut self, label: R::Token, reduce: &mut R) {
        reduce.enter_macro_definition(label.clone());
        assert_eq!(self.tokens.next().unwrap().kind(), Colon);
        assert_eq!(self.tokens.next().unwrap().kind(), Macro);
        assert_eq!(self.tokens.next().unwrap().kind(), Eol);
        self.parse_block(reduce);
        assert_eq!(self.tokens.next().unwrap().kind(), Endm);
        reduce.exit_macro_definition()
    }

    fn parse_operands(&mut self, reduce: &mut R) {
        if let Some(_) = self.peek_not_eol() {
            self.parse_expression(reduce);
            while let Some(Comma) = self.tokens.peek().map(|t| t.kind()) {
                self.tokens.next();
                self.parse_expression(reduce)
            }
        }
    }

    fn peek_not_eol(&mut self) -> Option<&L::Item> {
        match self.tokens.peek() {
            Some(token) if token.kind() == Eol => None,
            option_token => option_token,
        }
    }

    fn parse_expression(&mut self, reduce: &mut R) {
        reduce.enter_expression();
        let token = self.tokens.next().unwrap();
        match token.kind() {
            Word => reduce.push_identifier(token.clone()),
            QuotedString => reduce.push_literal(token.clone()),
            _ => panic!(),
        }
        reduce.exit_expression()
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

    struct TestReduce {
        actions: Vec<Action>,
    }

    impl TestReduce {
        fn new() -> TestReduce {
            TestReduce {
                actions: Vec::new(),
            }
        }
    }

    #[derive(Debug, PartialEq)]
    enum Action {
        EnterExpression,
        EnterInstruction(TestToken),
        EnterMacroDef(TestToken),
        ExitExpression,
        ExitInstruction,
        ExitMacroDef,
        PushIdentifier(TestToken),
        PushLiteral(TestToken),
    }

    type TestToken = (syntax::TerminalKind, usize);

    impl syntax::Terminal for TestToken {
        fn kind(&self) -> syntax::TerminalKind {
            let (ref terminal_kind, _) = *self;
            terminal_kind.clone()
        }
    }

    impl syntax::ParsingContext for TestReduce {
        type Token = TestToken;

        fn enter_instruction(&mut self, name: Self::Token) {
            self.actions.push(Action::EnterInstruction(name))
        }

        fn exit_instruction(&mut self) {
            self.actions.push(Action::ExitInstruction)
        }

        fn enter_expression(&mut self) {
            self.actions.push(Action::EnterExpression)
        }

        fn push_identifier(&mut self, identifier: Self::Token) {
            self.actions.push(Action::PushIdentifier(identifier))
        }

        fn push_literal(&mut self, literal: Self::Token) {
            self.actions.push(Action::PushLiteral(literal))
        }

        fn exit_expression(&mut self) {
            self.actions.push(Action::ExitExpression)
        }

        fn enter_macro_definition(&mut self, label: Self::Token) {
            self.actions.push(Action::EnterMacroDef(label))
        }

        fn exit_macro_definition(&mut self) {
            self.actions.push(Action::ExitMacroDef)
        }
    }

    #[test]
    fn parse_empty_line() {
        assert_eq_actions(&[(Eol, 0)], &[])
    }

    fn assert_eq_actions(tokens: &[TestToken], expected_actions: &[Action]) {
        let mut parsing_constext = TestReduce::new();
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
        vec![Action::PushIdentifier(identifier)]
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
        let expected_actions = &inst((Word, 0), vec![expr(literal((QuotedString, 1)))]);
        assert_eq_actions(tokens, expected_actions);
    }

    fn literal(literal: TestToken) -> Vec<Action> {
        vec![Action::PushLiteral(literal)]
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

    fn macro_def(label: TestToken, mut instructions: Vec<Action>) -> Vec<Action> {
        let mut result = vec![Action::EnterMacroDef(label)];
        result.append(&mut instructions);
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
        let expected_actions = &macro_def((Label, 0), inst((Word, 4), vec![]));
        assert_eq_actions(tokens, expected_actions);
    }
}
