use syntax;
use syntax::Block;
use syntax::Terminal;
use syntax::TerminalKind::*;

use std::iter;

pub fn parse_src<'a, I, R>(tokens: I, reduce: R) -> R::Block
    where I: Iterator<Item = R::Token>, R: syntax::ProductionRules
{
    let mut parser = Parser {
        tokens: tokens.peekable(),
        reduce: reduce,
    };
    parser.parse_block()
}

struct Parser<L: Iterator, R: syntax::ProductionRules> {
    tokens: iter::Peekable<L>,
    reduce: R,
}

impl<L, R> Parser<L, R> where R: syntax::ProductionRules, L: Iterator<Item = R::Token> {
    fn next_word(&mut self) -> Option<R::Token> {
        self.tokens.next()
    }

    fn parse_block(&mut self) -> R::Block {
        let mut block = R::Block::new();
        while let Some(token) = self.tokens.next() {
            if let Some(item) = self.parse_line(token) {
                block.push(item)
            }
        };
        block
    }

    fn parse_line(&mut self, first_token: R::Token) -> Option<R::Item> {
        match first_token.kind() {
            Word => Some(self.parse_nonempty_line(first_token)),
            Eol => None,
            _ => panic!()
        }
    }

    fn parse_nonempty_line(&mut self, first_token: R::Token) -> R::Item {
        let operands = self.parse_operands();
        self.reduce.reduce_command(first_token, &operands)
    }

    fn parse_operands(&mut self) -> Vec<R::Expr> {
        let mut operands = vec![];
        if let Some(_) = self.tokens.peek() {
            let first_word = self.tokens.next().unwrap();
            operands.push(self.reduce.build_name_expr(first_word));
            while let Some(Comma) = self.tokens.peek().map(|t| t.kind()) {
                self.next_word();
                let next_word = self.tokens.next().unwrap();
                operands.push(self.reduce.build_name_expr(next_word))
            }
        }
        operands
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

    type TestItem = (TestToken, Vec<TestToken>);

    impl syntax::ProductionRules for TestReduce {
        type Token = TestToken;
        type Item = TestItem;
        type Expr = Self::Token;
        type Block = Vec<Self::Item>;

        fn build_name_expr(&mut self, token: Self::Token) -> Self::Expr {
            token
        }

        fn reduce_command(&mut self, name: Self::Token, args: &[Self::Expr]) -> Self::Item {
            (name, args.iter().cloned().collect())
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
        assert_eq_items(&[(Word, 0)], &[((Word, 0), vec![])])
    }

    #[test]
    fn parse_unary_instruction() {
        assert_eq_items(&[(Word, 0), (Word, 1)], &[((Word, 0), vec![(Word, 1)])])
    }

    #[test]
    fn parse_binary_instruction() {
        assert_eq_items(&[(Word, 0), (Word, 1), (Comma, 2), (Word, 3)],
                        &[((Word, 0), vec![(Word, 1), (Word, 3)])])
    }

    #[test]
    fn parse_two_instructions() {
        let tokens = &[
            (Word, 0), (Word, 1), (Comma, 2), (Word, 3), (Eol, 4),
            (Word, 5), (Word, 6), (Comma, 7), (Word, 8),
        ];
        let expected_items = &[
            ((Word, 0), vec![(Word, 1), (Word, 3)]),
            ((Word, 5), vec![(Word, 6), (Word, 8)]),
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
            ((Word, 0), vec![(Word, 1), (Word, 3)]),
            ((Word, 6), vec![(Word, 7), (Word, 9)]),
        ];
        assert_eq_items(tokens, expected_items)
    }

    #[test]
    fn parse_include() {
        assert_eq_items(&[(Word, 0), (QuotedString, 1)], &[((Word, 0), vec![(QuotedString, 1)])])
    }
}
