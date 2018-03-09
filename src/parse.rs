use syntax::*;
use syntax::TerminalKind::*;

use std::iter;
use std::marker::PhantomData;

pub fn parse_src<'a, I, R>(tokens: I, mut reduce: R) -> R::Block
    where I: Iterator<Item = R::Token>, R: ProductionRules
{
    let mut parser = Parser {
        tokens: tokens.peekable(),
        _phantom: PhantomData,
    };
    parser.parse_block(&mut reduce)
}

struct Parser<L: Iterator, R: ProductionRules> {
    tokens: iter::Peekable<L>,
    _phantom: PhantomData<R>,
}

impl<L, R> Parser<L, R> where R: ProductionRules, L: Iterator<Item = R::Token> {
    fn parse_block(&mut self, reduce: &mut R) -> R::Block {
        let mut block = R::Block::new();
        while let Some(token) = self.next_token_if_not_block_delimiter() {
            if let Some(item) = self.parse_line(token, reduce) {
                block.push(item)
            }
        };
        block
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

    fn parse_line(&mut self, first_token: R::Token, reduce: &mut R) -> Option<R::Item> {
        match first_token.kind() {
            Eol => None,
            _ => Some(self.parse_nonempty_line(first_token, reduce)),
        }
    }

    fn parse_nonempty_line(&mut self, first_token: R::Token, reduce: &mut R) -> R::Item {
        if first_token.kind() == Label {
            self.parse_macro_definition(first_token, reduce)
        } else {
            let operands = self.parse_operands();
            reduce.reduce_command(first_token, &operands)
        }
    }

    fn parse_macro_definition(&mut self, label: R::Token, reduce: &mut R) -> R::Item {
        assert_eq!(self.tokens.next().unwrap().kind(), Colon);
        assert_eq!(self.tokens.next().unwrap().kind(), Macro);
        assert_eq!(self.tokens.next().unwrap().kind(), Eol);
        let block = self.parse_block(reduce);
        assert_eq!(self.tokens.next().unwrap().kind(), Endm);
        reduce.define_macro(label, block)
    }

    fn parse_operands(&mut self) -> Vec<R::Expr> {
        let mut operands = vec![];
        if let Some(_) = self.peek_not_eol() {
            operands.push(self.parse_expression());
            while let Some(Comma) = self.tokens.peek().map(|t| t.kind()) {
                self.tokens.next();
                operands.push(self.parse_expression())
            }
        }
        operands
    }

    fn peek_not_eol(&mut self) -> Option<&L::Item> {
        match self.tokens.peek() {
            Some(token) if token.kind() == Eol => None,
            option_token => option_token,
        }
    }

    fn parse_expression(&mut self) -> R::Expr {
        R::Expr::from_terminal(self.tokens.next().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::parse_src;

    use syntax;
    use syntax::TerminalKind::*;

    #[test]
    fn parse_empty_src() {
        assert_eq_items(&[], &[])
    }

    struct TestReduce;

    type TestToken = (syntax::TerminalKind, usize);

    impl syntax::Terminal for TestToken {
        fn kind(&self) -> syntax::TerminalKind {
            let (ref terminal_kind, _) = *self;
            terminal_kind.clone()
        }
    }

    type TestBlock = Vec<TestItem>;

    #[derive(Debug, PartialEq)]
    enum TestItem {
        Command(TestToken, Vec<TestToken>),
        Macro(TestToken, TestBlock),
    }

    impl syntax::ProductionRules for TestReduce {
        type Token = TestToken;
        type Item = TestItem;
        type Expr = Self::Token;
        type Block = TestBlock;

        fn define_macro(&mut self, label: Self::Token, block: Self::Block) -> Self::Item {
            TestItem::Macro(label, block)
        }

        fn reduce_command(&mut self, name: Self::Token, args: &[Self::Expr]) -> Self::Item {
            TestItem::Command(name, args.iter().cloned().collect())
        }
    }

    #[test]
    fn parse_empty_line() {
        assert_eq_items(&[(Eol, 0)], &[])
    }

    fn assert_eq_items(tokens: &[TestToken], expected_items: &[TestItem]) {
        let parsed_items = parse_src(tokens.iter().cloned(), TestReduce {});
        assert_eq!(parsed_items, expected_items)
    }

    #[test]
    fn parse_nullary_instruction() {
        assert_eq_items(&[(Word, 0)], &[TestItem::Command((Word, 0), vec![])])
    }

    #[test]
    fn parse_nullary_instruction_followed_by_eol() {
        assert_eq_items(&[(Word, 0), (Eol, 1)], &[TestItem::Command((Word, 0), vec![])])
    }

    #[test]
    fn parse_unary_instruction() {
        assert_eq_items(&[(Word, 0), (Word, 1)], &[TestItem::Command((Word, 0), vec![(Word, 1)])])
    }

    #[test]
    fn parse_binary_instruction() {
        assert_eq_items(&[(Word, 0), (Word, 1), (Comma, 2), (Word, 3)],
                        &[TestItem::Command((Word, 0), vec![(Word, 1), (Word, 3)])])
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = &[
            (Word, 0), (Word, 1), (Comma, 2), (Word, 3), (Eol, 4),
            (Word, 5), (Word, 6), (Comma, 7), (Word, 8),
        ];
        let expected_items = &[
            TestItem::Command((Word, 0), vec![(Word, 1), (Word, 3)]),
            TestItem::Command((Word, 5), vec![(Word, 6), (Word, 8)]),
        ];
        assert_eq_items(tokens, expected_items)
    }

    #[test]
    fn parse_two_instructions_separated_by_blank_line() {
        let tokens = &[
            (Word, 0), (Word, 1), (Comma, 2), (Word, 3), (Eol, 4),
            (Eol, 5),
            (Word, 6), (Word, 7), (Comma, 8), (Word, 9),
        ];
        let expected_items = &[
            TestItem::Command((Word, 0), vec![(Word, 1), (Word, 3)]),
            TestItem::Command((Word, 6), vec![(Word, 7), (Word, 9)]),
        ];
        assert_eq_items(tokens, expected_items)
    }

    #[test]
    fn parse_include() {
        assert_eq_items(&[(Word, 0), (QuotedString, 1)],
                        &[TestItem::Command((Word, 0), vec![(QuotedString, 1)])])
    }

    #[test]
    fn parse_empty_macro_definition() {
        let tokens = &[
            (Label, 0), (Colon, 1), (Macro, 2), (Eol, 3),
            (Endm, 4),
        ];
        let ast = &[
            TestItem::Macro((Label, 0), vec![]),
        ];
        assert_eq_items(tokens, ast)
    }

    #[test]
    fn parse_macro_definition_with_instruction() {
        let tokens = &[
            (Label, 0), (Colon, 1), (Macro, 2), (Eol, 3),
            (Word, 4), (Eol, 5),
            (Endm, 6),
        ];
        let ast = &[
            TestItem::Macro((Label, 0), vec![TestItem::Command((Word, 4), vec![])]),
        ];
        assert_eq_items(tokens, ast)
    }
}
